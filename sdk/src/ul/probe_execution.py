from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import secrets
import threading
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Literal, cast, get_args

from ul_core.contracts import ObservationSource, ProbeInvoker, StateEnvironment
from ul_core.evaluation import (
    EnvironmentCapabilities,
    EnvironmentLifecycleEvidence,
    EnvironmentLifecycleFailureCode,
    EnvironmentResetEvidence,
    EnvironmentStateEvidence,
    EnvironmentTurnEvidence,
    EvaluationCase,
    ExecutionEvidence,
    ObservationRequest,
    ProbeCapabilities,
    ProbeExecutionEvent,
    ProbeObservation,
    ProbeRequest,
    ProbeResult,
    ProbeTurn,
    StateFixtureRequest,
    StateObservationAuthority,
    StateOperationResult,
    StateSnapshot,
    evidence_profile_from_capabilities,
)

_ENVIRONMENT_LIFECYCLE_FAILURE_CODES = frozenset(get_args(EnvironmentLifecycleFailureCode))


class CapabilityExecutionError(RuntimeError):
    def __init__(
        self,
        code: EnvironmentLifecycleFailureCode,
        reason: str,
        *,
        delivery_uncertain: bool = False,
        not_delivered: bool = False,
        state_operation_result: StateOperationResult | None = None,
        _reason_is_safe: bool = False,
    ) -> None:
        super().__init__(reason)
        self.code: EnvironmentLifecycleFailureCode = code
        self.delivery_uncertain = delivery_uncertain
        self.not_delivered = not_delivered
        self.state_operation_result = state_operation_result
        self.safe_reason = reason if _reason_is_safe else None


async def _resolve[T](value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value


class _SyncAdapterRunner:
    def __init__(self, thread_name_prefix: str) -> None:
        self._thread_name = thread_name_prefix
        self._lock = threading.Lock()
        self._running = False
        self._unavailable = False

    async def call[RequestT, ResultT](
        self,
        operation: Callable[[RequestT], ResultT | Awaitable[ResultT]],
        request: RequestT,
    ) -> ResultT:
        loop = asyncio.get_running_loop()
        result_future: asyncio.Future[ResultT | Awaitable[ResultT]] = loop.create_future()
        with self._lock:
            if self._running or self._unavailable:
                raise RuntimeError("synchronous adapter operation is unavailable")
            self._running = True

        def deliver_result(value: ResultT | Awaitable[ResultT]) -> None:
            if not result_future.done():
                result_future.set_result(value)

        def deliver_error(error: BaseException) -> None:
            if not result_future.done():
                result_future.set_exception(error)

        def run() -> None:
            try:
                value = operation(request)
            except BaseException as error:
                with self._lock:
                    self._running = False
                with suppress(RuntimeError):
                    loop.call_soon_threadsafe(deliver_error, error)
            else:
                with self._lock:
                    self._running = False
                with suppress(RuntimeError):
                    loop.call_soon_threadsafe(deliver_result, value)

        threading.Thread(
            target=run,
            name=self._thread_name,
            daemon=True,
        ).start()
        try:
            value = await result_future
            return await _resolve(value)
        except asyncio.CancelledError:
            with self._lock:
                self._unavailable = True
            raise


async def _call[RequestT, ResultT](
    operation: Callable[[RequestT], ResultT | Awaitable[ResultT]],
    request: RequestT,
    sync_runner: _SyncAdapterRunner,
) -> ResultT:
    if inspect.iscoroutinefunction(operation):
        return await cast(Awaitable[ResultT], operation(request))
    return await sync_runner.call(operation, request)


class ComposedEnvironmentExecutor:
    def __init__(
        self,
        invoker: ProbeInvoker,
        *,
        config_sha256: str,
        observation_source: ObservationSource | None = None,
        state_environment: StateEnvironment | None = None,
        fixture_id: str | None = None,
        observation_timeout_seconds: float = 1.0,
        cleanup_grace_seconds: float = 1.0,
    ) -> None:
        if len(config_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in config_sha256
        ):
            raise ValueError("config_sha256 must be a lowercase SHA-256 digest")
        if not math.isfinite(observation_timeout_seconds) or observation_timeout_seconds <= 0:
            raise ValueError("observation_timeout_seconds must be finite and positive")
        if not math.isfinite(cleanup_grace_seconds) or cleanup_grace_seconds <= 0:
            raise ValueError("cleanup_grace_seconds must be finite and positive")
        state_capabilities = (
            state_environment.capabilities if state_environment is not None else None
        )
        if state_capabilities is not None and not all(
            (
                state_capabilities.supports_reset,
                state_capabilities.supports_snapshot,
                state_capabilities.supports_cleanup,
            )
        ):
            raise ValueError(
                "composed state execution requires reset, snapshot, and cleanup capabilities"
            )
        self._invoker = invoker
        self._observation_source = observation_source
        self._state_environment = state_environment
        self._fixture_id = fixture_id or (
            state_capabilities.environment_id
            if state_capabilities is not None
            else invoker.capabilities.invoker_id
        )
        self._config_sha256 = config_sha256
        self._observation_timeout_seconds = observation_timeout_seconds
        self._cleanup_grace_seconds = cleanup_grace_seconds
        self._lock = asyncio.Lock()
        self._state_uncertain = False
        self._invoker_sync_runner = _SyncAdapterRunner("ul-probe-invoker")
        self._observer_sync_runner = _SyncAdapterRunner("ul-probe-observer")
        self._state_sync_runner = _SyncAdapterRunner("ul-probe-state")
        self._cleanup_sync_runner = _SyncAdapterRunner("ul-probe-cleanup")
        self.probe_capabilities = ProbeCapabilities(
            invoker=invoker.capabilities,
            observation_source=(
                observation_source.capabilities if observation_source is not None else None
            ),
            state_environment=state_capabilities,
        )
        self.evidence_profile = evidence_profile_from_capabilities(self.probe_capabilities)
        self.capabilities = EnvironmentCapabilities(
            request_isolation=invoker.capabilities.request_isolation,
            supports_conversations=invoker.capabilities.supports_conversations,
            supports_state_observation=(
                state_capabilities.supports_snapshot if state_capabilities is not None else False
            ),
            state_observation_authority=(
                state_capabilities.state_observation_authority
                if state_capabilities is not None
                else None
            ),
            state_observer_id=(
                state_capabilities.state_observer_id if state_capabilities is not None else None
            ),
            cancellation_guarantee=invoker.capabilities.cancellation_guarantee,
        )

    @property
    def environment_id(self) -> str:
        state_capabilities = self.probe_capabilities.state_environment
        return (
            state_capabilities.environment_id
            if state_capabilities is not None
            else self.probe_capabilities.invoker.invoker_id
        )

    @property
    def config_sha256(self) -> str:
        return self._config_sha256

    @property
    def state_uncertain(self) -> bool:
        return self._state_uncertain

    def api_calls_for_case(self, case: EvaluationCase) -> int:
        if len(case.turns) > 1 and not self.capabilities.supports_conversations:
            raise ValueError("probe invoker does not support conversations")
        if case.required_state_observation_authority is not None:
            if not self.capabilities.supports_state_observation:
                raise ValueError("probe composition does not support state observation")
            if (
                self.capabilities.state_observation_authority
                != case.required_state_observation_authority
            ):
                raise ValueError("probe composition state authority does not match the case")
            if self.capabilities.state_observer_id != case.required_state_observer_id:
                raise ValueError("probe composition state observer does not match the case")
        if case.timeout_after_commit_event is not None:
            raise ValueError("composed probe execution does not support stress events")
        state_capabilities = self.probe_capabilities.state_environment
        observation_calls = len(case.turns) if self._observation_source is not None else 0
        if state_capabilities is None:
            return len(case.turns) + observation_calls
        return len(case.turns) * 2 + 3 + int(state_capabilities.supports_setup) + observation_calls

    async def execute(self, case: EvaluationCase) -> ExecutionEvidence:
        required_calls = self.api_calls_for_case(case)
        if required_calls > case.max_environment_api_calls:
            raise ValueError("evaluation case exceeds its environment API call budget")
        async with self._lock:
            execution_id = f"ul-probe-{secrets.token_hex(16)}"
            async with asyncio.timeout(case.timeout_seconds):
                if self._state_environment is None:
                    evidence = await self._execute_response_only(case, execution_id)
                else:
                    evidence = await self._execute_with_state(case, execution_id)
            return await self._attach_observations(case, evidence, execution_id)

    async def _execute_response_only(
        self,
        case: EvaluationCase,
        execution_id: str,
    ) -> ExecutionEvidence:
        turns: list[EnvironmentTurnEvidence] = []
        observations: list[ProbeObservation] = []
        execution_events: list[ProbeExecutionEvent] = []
        for turn in case.turns:
            correlation_id = _correlation_id(execution_id, turn.id)
            try:
                result = await self._invoke(
                    case,
                    turn.id,
                    turn.content,
                    correlation_id,
                    execution_id,
                )
            except CapabilityExecutionError as error:
                return self._response_only_evidence(
                    case,
                    tuple(turns),
                    tuple(observations),
                    tuple(execution_events),
                    error=error,
                )
            turns.append(
                EnvironmentTurnEvidence(
                    turn_id=turn.id,
                    response=result.response,
                    response_source_id=self._invoker.capabilities.invoker_id,
                    correlation_id=correlation_id,
                )
            )
            execution_events.extend(result.execution_events)
        return self._response_only_evidence(
            case,
            tuple(turns),
            tuple(observations),
            tuple(execution_events),
            error=None,
        )

    async def _execute_with_state(
        self,
        case: EvaluationCase,
        execution_id: str,
    ) -> ExecutionEvidence:
        if self._state_uncertain:
            return self._state_evidence(
                case,
                (),
                (),
                (),
                initial_state=None,
                initial_reset=None,
                cleanup_reset=None,
                completed_phases=(),
                error=CapabilityExecutionError(
                    "environment_state_uncertain",
                    "environment state is uncertain after an earlier lifecycle failure",
                    delivery_uncertain=True,
                ),
                failed_phase="blocked_state_uncertain",
                cleanup="not_attempted",
                cleanup_error=None,
            )
        state_environment = cast(StateEnvironment, self._state_environment)
        self._state_uncertain = True
        completed_phases: list[str] = []
        turns: list[EnvironmentTurnEvidence] = []
        observations: list[ProbeObservation] = []
        execution_events: list[ProbeExecutionEvent] = []
        initial_state: StateSnapshot | None = None
        initial_reset: StateOperationResult | None = None
        cleanup_result: StateOperationResult | None = None
        error: CapabilityExecutionError | None = None
        cleanup_error: CapabilityExecutionError | None = None
        failed_phase: str | None = None
        lifecycle_started = False
        current_phase = "reset"
        try:
            initial_reset = await self._state_operation(
                await _call(
                    state_environment.reset,
                    self._state_request(
                        case,
                        correlation_id=execution_id,
                        session_id=execution_id,
                    ),
                    self._state_sync_runner,
                ),
                "reset",
                self._state_request(
                    case,
                    correlation_id=execution_id,
                    session_id=execution_id,
                ),
            )
            lifecycle_started = True
            completed_phases.append("reset")
            if state_environment.capabilities.supports_setup:
                current_phase = "setup"
                await self._state_operation(
                    await _call(
                        state_environment.setup,
                        self._state_request(
                            case,
                            correlation_id=execution_id,
                            session_id=execution_id,
                        ),
                        self._state_sync_runner,
                    ),
                    "setup",
                    self._state_request(
                        case,
                        correlation_id=execution_id,
                        session_id=execution_id,
                    ),
                )
                completed_phases.append("setup")
            current_phase = "initial_snapshot"
            initial_state = await _call(
                state_environment.snapshot,
                self._state_request(
                    case,
                    correlation_id=execution_id,
                    session_id=execution_id,
                    turn_id="__ul_initial_state__",
                ),
                self._state_sync_runner,
            )
            self._validate_snapshot(initial_state, execution_id)
            completed_phases.append("initial_snapshot")
            for index, turn in enumerate(case.turns, start=1):
                correlation_id = _correlation_id(execution_id, turn.id)
                current_phase = _phase("execute_turn", index, len(case.turns))
                result = await self._invoke(
                    case,
                    turn.id,
                    turn.content,
                    correlation_id,
                    execution_id,
                )
                completed_phases.append(current_phase)
                turns.append(
                    EnvironmentTurnEvidence(
                        turn_id=turn.id,
                        response=result.response,
                        response_source_id=self._invoker.capabilities.invoker_id,
                        correlation_id=correlation_id,
                    )
                )
                execution_events.extend(result.execution_events)
                current_phase = _phase("snapshot", index, len(case.turns))
                snapshot = await _call(
                    state_environment.snapshot,
                    self._state_request(
                        case,
                        correlation_id=correlation_id,
                        session_id=execution_id,
                        turn_id=turn.id,
                    ),
                    self._state_sync_runner,
                )
                self._validate_snapshot(snapshot, correlation_id)
                completed_phases.append(current_phase)
                turns[-1] = turns[-1].model_copy(
                    update={
                        "state_snapshot": snapshot.value,
                        "state_observation_authority": snapshot.authority,
                        "state_observer_id": snapshot.observer_id,
                    }
                )
        except CapabilityExecutionError as caught_error:
            error = caught_error
            failed_phase = current_phase
            if current_phase == "reset":
                initial_reset = caught_error.state_operation_result
        except asyncio.CancelledError:
            raise
        except Exception:
            error = CapabilityExecutionError(
                "environment_lifecycle_error",
                "environment lifecycle failed",
                delivery_uncertain=True,
            )
            failed_phase = current_phase
        finally:
            if lifecycle_started:
                try:
                    async with asyncio.timeout(self._cleanup_grace_seconds):
                        cleanup_result = await self._state_operation(
                            await _call(
                                state_environment.cleanup,
                                self._state_request(
                                    case,
                                    correlation_id=execution_id,
                                    session_id=execution_id,
                                ),
                                self._cleanup_sync_runner,
                            ),
                            "cleanup",
                            self._state_request(
                                case,
                                correlation_id=execution_id,
                                session_id=execution_id,
                            ),
                        )
                    completed_phases.append("cleanup_reset")
                except TimeoutError:
                    cleanup_error = CapabilityExecutionError(
                        "environment_cleanup_error",
                        "environment cleanup timed out",
                        delivery_uncertain=True,
                    )
                except CapabilityExecutionError as caught_cleanup_error:
                    cleanup_error = caught_cleanup_error
                    cleanup_result = caught_cleanup_error.state_operation_result
                except asyncio.CancelledError:
                    raise
                except Exception:
                    cleanup_error = CapabilityExecutionError(
                        "environment_cleanup_error",
                        "environment cleanup failed",
                        delivery_uncertain=True,
                    )
        if cleanup_error is None and (
            (lifecycle_started and (error is None or not error.delivery_uncertain))
            or (not lifecycle_started and error is not None and error.not_delivered)
        ):
            self._state_uncertain = False
        return self._state_evidence(
            case,
            tuple(turns),
            tuple(observations),
            tuple(execution_events),
            initial_state=initial_state,
            initial_reset=initial_reset,
            cleanup_reset=cleanup_result,
            completed_phases=tuple(completed_phases),
            error=error or cleanup_error,
            failed_phase=failed_phase or ("cleanup_reset" if cleanup_error else None),
            cleanup=(
                "failed"
                if cleanup_error is not None
                else "succeeded"
                if lifecycle_started
                else "not_attempted"
            ),
            cleanup_error=cleanup_error,
        )

    async def _invoke(
        self,
        case: EvaluationCase,
        turn_id: str,
        content: str,
        correlation_id: str,
        session_id: str,
    ) -> ProbeResult:
        try:
            result = await _call(
                self._invoker.invoke,
                ProbeRequest(
                    case_id=case.id,
                    session_id=session_id,
                    correlation_id=correlation_id,
                    turn=ProbeTurn(id=turn_id, input=content),
                    context=case.probe_context,
                ),
                self._invoker_sync_runner,
            )
        except CapabilityExecutionError:
            raise
        except Exception:
            raise CapabilityExecutionError(
                "environment_lifecycle_error",
                "probe invocation failed",
                delivery_uncertain=True,
            ) from None
        if result.correlation_id != correlation_id:
            raise CapabilityExecutionError(
                "case_identity",
                "probe result did not match its request",
                delivery_uncertain=True,
            )
        response_size_bytes = len(
            json.dumps(
                result.response,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if result.response_truncated or response_size_bytes > (
            self._invoker.capabilities.response_size_limit_bytes
        ):
            raise CapabilityExecutionError(
                "response_too_large",
                "probe response exceeds the size limit",
                delivery_uncertain=True,
            )
        if (
            result.execution_events
            and not self._invoker.capabilities.supports_structured_execution_events
        ):
            raise CapabilityExecutionError(
                "response_mapping",
                "probe invoker returned unsupported structured execution events",
                delivery_uncertain=True,
            )
        execution_events_size_bytes = len(
            json.dumps(
                [event.model_dump(mode="json") for event in result.execution_events],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if execution_events_size_bytes > (
            self._invoker.capabilities.execution_events_size_limit_bytes
        ):
            raise CapabilityExecutionError(
                "response_too_large",
                "probe execution events exceed the size limit",
                delivery_uncertain=True,
            )
        return result

    async def _attach_observations(
        self,
        case: EvaluationCase,
        evidence: ExecutionEvidence,
        execution_id: str,
    ) -> ExecutionEvidence:
        if self._observation_source is None or not evidence.turns:
            return evidence
        observations = tuple(
            [
                await self._observe(
                    case,
                    cast(str, turn.correlation_id),
                    execution_id,
                )
                for turn in evidence.turns
            ]
        )
        return evidence.model_copy(update={"observations": observations})

    async def _observe(
        self,
        case: EvaluationCase,
        correlation_id: str,
        session_id: str,
    ) -> ProbeObservation:
        observation_source = cast(ObservationSource, self._observation_source)
        try:
            async with asyncio.timeout(self._observation_timeout_seconds):
                observation = await _call(
                    observation_source.observe,
                    ObservationRequest(
                        case_id=case.id,
                        session_id=session_id,
                        correlation_id=correlation_id,
                    ),
                    self._observer_sync_runner,
                )
            capabilities = observation_source.capabilities
            if observation.correlation_id != correlation_id or (
                observation.source_id != capabilities.source_id
                or observation.authority != capabilities.authority
                or (observation.traces and not capabilities.supports_traces)
                or (observation.tool_calls and not capabilities.supports_tool_calls)
                or (observation.handoffs and not capabilities.supports_handoffs)
                or (observation.errors and not capabilities.supports_errors)
                or (observation.usage is not None and not capabilities.supports_usage)
                or (observation.metadata and not capabilities.supports_metadata)
                or len(observation.model_dump_json().encode("utf-8"))
                > capabilities.observation_size_limit_bytes
            ):
                raise ValueError
            return observation
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception:
            return ProbeObservation(
                id=f"{correlation_id}:missing-observation",
                source_id=observation_source.capabilities.source_id,
                correlation_id=correlation_id,
                authority=observation_source.capabilities.authority,
                status="missing",
                limitation="observation source did not provide correlated evidence",
            )

    async def _state_operation(
        self,
        value: StateOperationResult | Awaitable[StateOperationResult],
        operation: Literal["reset", "setup", "cleanup"],
        request: StateFixtureRequest,
    ) -> StateOperationResult:
        try:
            result = await _resolve(value)
        except CapabilityExecutionError:
            raise
        except Exception:
            raise CapabilityExecutionError(
                "environment_lifecycle_error",
                f"environment {operation} failed",
                delivery_uncertain=True,
            ) from None
        if (
            result.operation != operation
            or result.fixture_id != request.fixture_id
            or result.correlation_id != request.correlation_id
        ):
            raise CapabilityExecutionError(
                "response_mapping",
                "state operation result did not match its request",
                delivery_uncertain=True,
            )
        if not result.succeeded:
            code = (
                result.failure_code
                if result.failure_code in _ENVIRONMENT_LIFECYCLE_FAILURE_CODES
                else "environment_lifecycle_error"
            )
            raise CapabilityExecutionError(
                cast(EnvironmentLifecycleFailureCode, code),
                f"environment {operation} failed",
                delivery_uncertain=result.state_uncertain,
                state_operation_result=result,
                _reason_is_safe=True,
            )
        return result

    def _validate_snapshot(self, snapshot: StateSnapshot, correlation_id: str) -> None:
        capabilities = cast(StateEnvironment, self._state_environment).capabilities
        if (
            snapshot.fixture_id != self._fixture_id
            or snapshot.correlation_id != correlation_id
            or snapshot.source_id != capabilities.environment_id
            or snapshot.authority != capabilities.state_observation_authority
            or snapshot.observer_id != capabilities.state_observer_id
        ):
            raise CapabilityExecutionError(
                "environment_identity",
                "state snapshot did not match its request",
                delivery_uncertain=True,
            )
        snapshot_size_bytes = len(
            json.dumps(
                snapshot.value,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if snapshot_size_bytes > capabilities.snapshot_size_limit_bytes:
            raise CapabilityExecutionError(
                "response_too_large",
                "state snapshot exceeds the size limit",
            )

    def _state_request(
        self,
        case: EvaluationCase,
        *,
        correlation_id: str,
        session_id: str,
        turn_id: str | None = None,
    ) -> StateFixtureRequest:
        return StateFixtureRequest(
            fixture_id=self._fixture_id,
            case_id=case.id,
            session_id=session_id,
            correlation_id=correlation_id,
            turn_id=turn_id,
        )

    def _response_only_evidence(
        self,
        case: EvaluationCase,
        turns: tuple[EnvironmentTurnEvidence, ...],
        observations: tuple[ProbeObservation, ...],
        execution_events: tuple[ProbeExecutionEvent, ...],
        *,
        error: CapabilityExecutionError | None,
    ) -> ExecutionEvidence:
        return ExecutionEvidence(
            evidence_scope="response_only",
            case_id=case.id,
            environment_id=self.environment_id,
            environment_config_sha256=self.config_sha256,
            turns=turns,
            final_response=turns[-1].response if turns and error is None else None,
            observations=observations,
            execution_events=execution_events,
            lifecycle=EnvironmentLifecycleEvidence(
                terminal_status="succeeded" if error is None else "failed",
                completed_phases=tuple("execute_turn" for _ in turns),
                failed_phase=None if error is None else "execute_turn",
                failure_code=None if error is None else error.code,
                failure_reason=(
                    None if error is None else error.safe_reason or "probe invocation failed"
                ),
                delivery=(
                    "uncertain" if error is not None and error.delivery_uncertain else "certain"
                ),
                cleanup="not_attempted",
                environment_state_uncertain=(
                    error.delivery_uncertain if error is not None else False
                ),
            ),
        )

    def _state_evidence(
        self,
        case: EvaluationCase,
        turns: tuple[EnvironmentTurnEvidence, ...],
        observations: tuple[ProbeObservation, ...],
        execution_events: tuple[ProbeExecutionEvent, ...],
        *,
        initial_state: StateSnapshot | None,
        initial_reset: StateOperationResult | None,
        cleanup_reset: StateOperationResult | None,
        completed_phases: tuple[str, ...],
        error: CapabilityExecutionError | None,
        failed_phase: str | None,
        cleanup: Literal["succeeded", "failed", "not_attempted"],
        cleanup_error: CapabilityExecutionError | None,
    ) -> ExecutionEvidence:
        final_turn = turns[-1] if turns else None
        return ExecutionEvidence(
            evidence_scope="response_and_state",
            case_id=case.id,
            environment_id=self.environment_id,
            environment_config_sha256=self.config_sha256,
            initial_state=(
                EnvironmentStateEvidence(
                    value=initial_state.value,
                    authority=initial_state.authority,
                    observer_id=initial_state.observer_id,
                )
                if initial_state is not None
                else None
            ),
            turns=turns,
            final_response=final_turn.response if final_turn is not None else None,
            final_state=(
                EnvironmentStateEvidence(
                    value=final_turn.state_snapshot,
                    authority=cast(
                        StateObservationAuthority,
                        final_turn.state_observation_authority,
                    ),
                    observer_id=final_turn.state_observer_id,
                )
                if final_turn is not None and error is None
                else None
            ),
            observations=observations,
            execution_events=execution_events,
            lifecycle=EnvironmentLifecycleEvidence(
                terminal_status="succeeded" if error is None else "failed",
                completed_phases=completed_phases,
                failed_phase=failed_phase,
                failure_code=None if error is None else error.code,
                failure_reason=(
                    None if error is None else error.safe_reason or "environment lifecycle failed"
                ),
                delivery=(
                    "uncertain"
                    if (
                        (error is not None and error.delivery_uncertain)
                        or (cleanup_error is not None and cleanup_error.delivery_uncertain)
                    )
                    else "certain"
                ),
                cleanup=cleanup,
                cleanup_failure_code=(cleanup_error.code if cleanup_error is not None else None),
                cleanup_failure_reason=(
                    cleanup_error.safe_reason or "environment cleanup failed"
                    if cleanup_error is not None
                    else None
                ),
                environment_state_uncertain=self._state_uncertain,
                initial_reset=_reset_evidence(initial_reset),
                cleanup_reset=(
                    _reset_evidence(cleanup_reset) if cleanup != "not_attempted" else None
                ),
            ),
        )


def _correlation_id(execution_id: str, turn_id: str) -> str:
    turn_digest = hashlib.sha256(turn_id.encode("utf-8")).hexdigest()
    return f"{execution_id}:{turn_digest}"


def _phase(name: str, index: int, turn_count: int) -> str:
    return name if turn_count == 1 else f"{name}:{index}"


def _reset_evidence(result: StateOperationResult | None) -> EnvironmentResetEvidence:
    return EnvironmentResetEvidence(
        reset_session_requested=(result.reset_session_requested if result is not None else False),
        reset_session_acknowledged=(
            result.reset_session_acknowledged if result is not None else False
        ),
        reset_env_requested=(result.reset_environment_requested if result is not None else False),
        reset_env_acknowledged=(
            result.reset_environment_acknowledged if result is not None else False
        ),
    )
