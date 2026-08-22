from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import pytest
from ul.environment import evaluation_case_from_inputs
from ul.probe_execution import CapabilityExecutionError, ComposedEnvironmentExecutor
from ul_core.evaluation import (
    ObservationRequest,
    ObservationSourceCapabilities,
    ProbeExecutionEvent,
    ProbeInvokerCapabilities,
    ProbeObservation,
    ProbeRequest,
    ProbeResult,
    StateEnvironmentCapabilities,
    StateFixtureRequest,
    StateOperationResult,
    StateSnapshot,
)

_CONFIG_SHA256 = "a" * 64


@dataclass
class _Invoker:
    capabilities: ProbeInvokerCapabilities = field(
        default_factory=lambda: ProbeInvokerCapabilities(
            invoker_id="test-invoker",
            response_size_limit_bytes=1_000,
            supports_conversations=True,
            cancellation_guarantee="best_effort",
        )
    )
    requests: list[ProbeRequest] = field(default_factory=list)

    def invoke(self, request: ProbeRequest) -> ProbeResult:
        self.requests.append(request)
        response = {"echo": request.turn.input}
        return ProbeResult(
            id="result-1",
            correlation_id=request.correlation_id,
            response=response,
            response_size_bytes=len(json.dumps(response, separators=(",", ":")).encode("utf-8")),
        )


@dataclass
class _Observer:
    capabilities: ObservationSourceCapabilities = field(
        default_factory=lambda: ObservationSourceCapabilities(
            source_id="test-observer",
            authority="independent_observer",
            supports_traces=True,
        )
    )
    fail: bool = False

    async def observe(self, request: ObservationRequest) -> ProbeObservation:
        if self.fail:
            raise RuntimeError("private observer failure")
        return ProbeObservation(
            id=f"observation:{request.correlation_id}",
            source_id=self.capabilities.source_id,
            correlation_id=request.correlation_id,
            authority=self.capabilities.authority,
            traces=({"span": "agent"},),
        )


@dataclass
class _StateEnvironment:
    capabilities: StateEnvironmentCapabilities = field(
        default_factory=lambda: StateEnvironmentCapabilities(
            environment_id="test-state",
            supports_reset=True,
            supports_setup=True,
            supports_snapshot=True,
            supports_cleanup=True,
            state_observation_authority="environment_self_reported",
            supports_deterministic_replay=True,
        )
    )
    operations: list[tuple[str, str | None]] = field(default_factory=list)

    def reset(self, request: StateFixtureRequest) -> StateOperationResult:
        self.operations.append(("reset", request.turn_id))
        return _successful_state_operation(request, "reset")

    async def setup(self, request: StateFixtureRequest) -> StateOperationResult:
        self.operations.append(("setup", request.turn_id))
        return _successful_state_operation(request, "setup")

    def snapshot(self, request: StateFixtureRequest) -> StateSnapshot:
        self.operations.append(("snapshot", request.turn_id))
        return StateSnapshot(
            id="snapshot-1",
            fixture_id=request.fixture_id,
            correlation_id=request.correlation_id,
            source_id=self.capabilities.environment_id,
            value={"turn_id": request.turn_id},
            authority="environment_self_reported",
        )

    def cleanup(self, request: StateFixtureRequest) -> StateOperationResult:
        self.operations.append(("cleanup", request.turn_id))
        return _successful_state_operation(request, "cleanup")


def _successful_state_operation(
    request: StateFixtureRequest,
    operation: str,
) -> StateOperationResult:
    reset_operation = operation in {"reset", "cleanup"}
    return StateOperationResult.model_validate(
        {
            "id": f"{operation}:{request.case_id}",
            "fixture_id": request.fixture_id,
            "correlation_id": request.correlation_id,
            "operation": operation,
            "succeeded": True,
            "reset_session_requested": reset_operation,
            "reset_session_acknowledged": reset_operation,
            "reset_environment_requested": reset_operation,
            "reset_environment_acknowledged": reset_operation,
        }
    )


def _case(*inputs: str):
    return evaluation_case_from_inputs(
        case_id="case-1",
        raw_inputs=inputs,
        max_environment_api_calls=20,
        timeout_seconds=1,
    )


@pytest.mark.asyncio
async def test_invoker_only_produces_response_evidence_with_provenance() -> None:
    invoker = _Invoker()
    executor = ComposedEnvironmentExecutor(invoker, config_sha256=_CONFIG_SHA256)

    evidence = await executor.execute(_case("hello", "again"))

    assert [request.turn.input for request in invoker.requests] == ["hello", "again"]
    assert evidence.schema_version == "1.3.0"
    assert evidence.evidence_scope == "response_only"
    assert evidence.final_response == {"echo": "again"}
    assert evidence.turns[0].response_source_id == "test-invoker"
    assert evidence.turns[0].correlation_id is not None
    assert evidence.turns[0].correlation_id.startswith("ul-probe-")
    assert executor.evidence_profile.supported_facts == frozenset({"response_observed"})


@pytest.mark.asyncio
async def test_missing_observation_does_not_block_probe_execution() -> None:
    executor = ComposedEnvironmentExecutor(
        _Invoker(),
        config_sha256=_CONFIG_SHA256,
        observation_source=_Observer(fail=True),
    )

    evidence = await executor.execute(_case("hello"))

    assert evidence.lifecycle.terminal_status == "succeeded"
    assert evidence.observations[0].status == "missing"
    assert "private observer failure" not in evidence.observations[0].limitation
    assert executor.evidence_profile.supported_facts == frozenset(
        {"response_observed", "trajectory_observed"}
    )


@pytest.mark.asyncio
async def test_all_capabilities_preserve_response_trace_and_state_provenance() -> None:
    state_environment = _StateEnvironment()
    executor = ComposedEnvironmentExecutor(
        _Invoker(),
        config_sha256=_CONFIG_SHA256,
        observation_source=_Observer(),
        state_environment=state_environment,
        fixture_id="fixture-1",
    )

    evidence = await executor.execute(_case("hello"))

    assert state_environment.operations == [
        ("reset", None),
        ("setup", None),
        ("snapshot", "__ul_initial_state__"),
        ("snapshot", "case-1:turn-1"),
        ("cleanup", None),
    ]
    assert evidence.evidence_scope == "response_and_state"
    assert evidence.turns[0].state_observation_authority == "environment_self_reported"
    assert evidence.observations[0].authority == "independent_observer"
    assert executor.evidence_profile.supported_facts == frozenset(
        {
            "response_observed",
            "trajectory_observed",
            "committed_state_verified",
            "deterministic_replay_verified",
        }
    )


def test_composition_rejects_partial_state_lifecycle() -> None:
    state_environment = _StateEnvironment()
    state_environment.capabilities = state_environment.capabilities.model_copy(
        update={"supports_cleanup": False, "supports_deterministic_replay": False}
    )

    with pytest.raises(ValueError, match="requires reset, snapshot, and cleanup"):
        ComposedEnvironmentExecutor(
            _Invoker(),
            config_sha256=_CONFIG_SHA256,
            state_environment=state_environment,
        )


@pytest.mark.asyncio
async def test_invocation_failure_is_safe_and_quarantines_stateful_composition() -> None:
    class _FailingInvoker(_Invoker):
        def invoke(self, request: ProbeRequest) -> ProbeResult:
            raise CapabilityExecutionError(
                "response_timeout",
                "safe timeout",
                delivery_uncertain=True,
            )

    executor = ComposedEnvironmentExecutor(
        _FailingInvoker(),
        config_sha256=_CONFIG_SHA256,
        state_environment=_StateEnvironment(),
    )

    first = await executor.execute(_case("hello"))
    second = await executor.execute(_case("again"))

    assert first.lifecycle.failure_code == "response_timeout"
    assert first.lifecycle.failure_reason == "environment lifecycle failed"
    assert "safe timeout" not in first.model_dump_json()
    assert first.lifecycle.cleanup == "succeeded"
    assert first.lifecycle.environment_state_uncertain is True
    assert second.lifecycle.failed_phase == "blocked_state_uncertain"


@pytest.mark.asyncio
async def test_repeated_cases_receive_fresh_correlation_identifiers() -> None:
    invoker = _Invoker()
    executor = ComposedEnvironmentExecutor(invoker, config_sha256=_CONFIG_SHA256)

    await executor.execute(_case("hello"))
    await executor.execute(_case("hello"))

    assert invoker.requests[0].correlation_id != invoker.requests[1].correlation_id
    assert invoker.requests[0].session_id != invoker.requests[1].session_id


@pytest.mark.asyncio
async def test_observation_authority_mismatch_becomes_missing_evidence() -> None:
    class _MismatchedObserver(_Observer):
        async def observe(self, request: ObservationRequest) -> ProbeObservation:
            return ProbeObservation(
                id="spoofed-observation",
                source_id=self.capabilities.source_id,
                correlation_id=request.correlation_id,
                authority="source_self_reported",
                traces=({"span": "agent"},),
            )

    executor = ComposedEnvironmentExecutor(
        _Invoker(),
        config_sha256=_CONFIG_SHA256,
        observation_source=_MismatchedObserver(),
    )

    evidence = await executor.execute(_case("hello"))

    assert evidence.observations[0].status == "missing"
    assert evidence.observations[0].authority == "independent_observer"


@pytest.mark.asyncio
async def test_structured_execution_events_are_preserved_when_declared() -> None:
    class _EventInvoker(_Invoker):
        def invoke(self, request: ProbeRequest) -> ProbeResult:
            response = {"status": "done"}
            return ProbeResult(
                id="result-1",
                correlation_id=request.correlation_id,
                response=response,
                response_size_bytes=len(
                    json.dumps(response, separators=(",", ":")).encode("utf-8")
                ),
                execution_events=(
                    ProbeExecutionEvent(
                        id="event-1",
                        correlation_id=request.correlation_id,
                        kind="tool_call",
                        payload={"tool": "lookup"},
                    ),
                ),
            )

    evidence = await ComposedEnvironmentExecutor(
        _EventInvoker(
            capabilities=ProbeInvokerCapabilities(
                invoker_id="event-invoker",
                response_size_limit_bytes=1_000,
                supports_structured_execution_events=True,
            )
        ),
        config_sha256=_CONFIG_SHA256,
    ).execute(_case("hello"))

    assert evidence.execution_events[0].payload == {"tool": "lookup"}


@pytest.mark.asyncio
async def test_stale_cleanup_receipt_does_not_clear_quarantine() -> None:
    class _StaleCleanupState(_StateEnvironment):
        def cleanup(self, request: StateFixtureRequest) -> StateOperationResult:
            result = _successful_state_operation(request, "cleanup")
            return result.model_copy(update={"correlation_id": "stale-attempt"})

    executor = ComposedEnvironmentExecutor(
        _Invoker(),
        config_sha256=_CONFIG_SHA256,
        state_environment=_StaleCleanupState(),
    )

    first = await executor.execute(_case("hello"))
    second = await executor.execute(_case("again"))

    assert first.lifecycle.failed_phase == "cleanup_reset"
    assert first.lifecycle.environment_state_uncertain is True
    assert second.lifecycle.failed_phase == "blocked_state_uncertain"


@pytest.mark.asyncio
async def test_delivered_initial_reset_failure_quarantines_without_cleanup() -> None:
    class _FailedResetState(_StateEnvironment):
        def reset(self, request: StateFixtureRequest) -> StateOperationResult:
            return StateOperationResult(
                id="failed-reset",
                fixture_id=request.fixture_id,
                correlation_id=request.correlation_id,
                operation="reset",
                succeeded=False,
                failure_code="environment_lifecycle_error",
                failure_reason="private reset failure containing a secret",
            )

    executor = ComposedEnvironmentExecutor(
        _Invoker(),
        config_sha256=_CONFIG_SHA256,
        state_environment=_FailedResetState(),
    )

    first = await executor.execute(_case("hello"))
    second = await executor.execute(_case("again"))

    assert first.lifecycle.cleanup == "not_attempted"
    assert first.lifecycle.environment_state_uncertain is True
    assert first.lifecycle.failure_reason == "environment reset failed"
    assert "private reset failure" not in first.model_dump_json()
    assert second.lifecycle.failed_phase == "blocked_state_uncertain"


@pytest.mark.asyncio
async def test_cleanup_uncertainty_is_reflected_in_top_level_delivery() -> None:
    class _CleanupTimeoutState(_StateEnvironment):
        def snapshot(self, request: StateFixtureRequest) -> StateSnapshot:
            if request.turn_id != "__ul_initial_state__":
                raise CapabilityExecutionError(
                    "response_mapping",
                    "safe mapping failure",
                )
            return super().snapshot(request)

        def cleanup(self, request: StateFixtureRequest) -> StateOperationResult:
            raise CapabilityExecutionError(
                "environment_cleanup_error",
                "safe cleanup timeout",
                delivery_uncertain=True,
            )

    evidence = await ComposedEnvironmentExecutor(
        _Invoker(),
        config_sha256=_CONFIG_SHA256,
        state_environment=_CleanupTimeoutState(),
    ).execute(_case("hello"))

    assert evidence.lifecycle.delivery == "uncertain"
    assert evidence.lifecycle.environment_state_uncertain is True


@pytest.mark.asyncio
async def test_long_existing_turn_identifier_uses_bounded_correlation_identifier() -> None:
    evidence = await ComposedEnvironmentExecutor(
        _Invoker(),
        config_sha256=_CONFIG_SHA256,
        state_environment=_StateEnvironment(),
    ).execute(
        _case("hello").model_copy(
            update={"turns": (_case("hello").turns[0].model_copy(update={"id": "t" * 1_000}),)}
        )
    )

    assert evidence.lifecycle.terminal_status == "succeeded"
    assert len(evidence.turns[0].correlation_id or "") < 500


@pytest.mark.asyncio
async def test_blocking_sync_invoker_remains_bounded_by_case_timeout() -> None:
    class _BlockingInvoker(_Invoker):
        def invoke(self, request: ProbeRequest) -> ProbeResult:
            time.sleep(0.1)
            return super().invoke(request)

    executor = ComposedEnvironmentExecutor(
        _BlockingInvoker(),
        config_sha256=_CONFIG_SHA256,
    )
    case = _case("hello").model_copy(update={"timeout_seconds": 0.01})

    with pytest.raises(TimeoutError):
        await executor.execute(case)
