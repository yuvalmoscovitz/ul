from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import threading
from collections.abc import Awaitable, Callable
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator
from ul_core.evaluation import (
    StateEnvironmentCapabilities,
    StateFixtureRequest,
    StateObservationAuthority,
    StateOperationResult,
    StateSnapshot,
)

_MAXIMUM_JSON_DEPTH = 100
_VOLATILE_VALUE: dict[str, JsonValue] = {"__ul_volatile__": True}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class JsonStateNormalization(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    volatile_json_pointers: tuple[str, ...] = Field(default=(), max_length=1_000)
    unordered_json_pointers: tuple[str, ...] = Field(default=(), max_length=1_000)

    @field_validator("volatile_json_pointers", "unordered_json_pointers")
    @classmethod
    def validate_json_pointers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("state normalization JSON pointers must be unique")
        for pointer in value:
            if len(pointer) > 2_000:
                raise ValueError("state normalization JSON pointers cannot exceed 2000 characters")
            if pointer and not pointer.startswith("/"):
                raise ValueError("state normalization JSON pointers must be RFC 6901 pointers")
            index = 0
            while index < len(pointer):
                if pointer[index] == "~":
                    if index + 1 == len(pointer) or pointer[index + 1] not in {"0", "1"}:
                        raise ValueError(
                            "state normalization JSON pointers must use valid escape sequences"
                        )
                    index += 1
                index += 1
        return value


class JsonStateDifference(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    path: str
    kind: Literal["added", "removed", "changed"]
    before: JsonValue = None
    after: JsonValue = None


class StateCallbackContext(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    phase: Literal["reset", "setup", "snapshot", "cleanup"]
    fixture_id: str = Field(min_length=1, max_length=500)
    case_id: str = Field(min_length=1, max_length=500)
    session_id: str = Field(min_length=1, max_length=500)
    correlation_id: str = Field(min_length=1, max_length=500)
    turn_id: str | None = Field(default=None, min_length=1, max_length=500)
    generation: int = Field(ge=0)
    case_context: dict[str, JsonValue] = Field(default_factory=dict)


type StateLifecycleCallback = Callable[[StateCallbackContext], Awaitable[None] | None]
type StateSnapshotCallback = Callable[[StateCallbackContext], JsonValue | Awaitable[JsonValue]]


class StateResetConformanceReport(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    fixture_id: str = Field(min_length=1, max_length=500)
    first_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    second_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    deterministic: bool
    differences: tuple[JsonStateDifference, ...] = ()


class _SyncCallbackRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._unavailable = False

    async def call[ResultT](
        self,
        callback: Callable[[StateCallbackContext], ResultT],
        context: StateCallbackContext,
    ) -> ResultT:
        loop = asyncio.get_running_loop()
        result_future: asyncio.Future[ResultT] = loop.create_future()
        with self._lock:
            if self._running or self._unavailable:
                raise RuntimeError("state callback runner is unavailable")
            self._running = True

        def deliver_result(value: ResultT) -> None:
            if not result_future.done():
                result_future.set_result(value)

        def deliver_error(error: BaseException) -> None:
            if not result_future.done():
                result_future.set_exception(error)

        def run() -> None:
            try:
                value = callback(context)
            except BaseException as error:
                loop.call_soon_threadsafe(deliver_error, error)
            else:
                loop.call_soon_threadsafe(deliver_result, value)
            finally:
                with self._lock:
                    self._running = False

        threading.Thread(target=run, name="ul-state-callback", daemon=True).start()
        try:
            return await result_future
        except asyncio.CancelledError:
            with self._lock:
                self._unavailable = True
            raise


class CallbackStateEnvironment:
    def __init__(
        self,
        *,
        environment_id: str,
        reset: StateLifecycleCallback,
        snapshot: StateSnapshotCallback,
        setup: StateLifecycleCallback | None = None,
        cleanup: StateLifecycleCallback | None = None,
        authority: StateObservationAuthority = "environment_self_reported",
        observer_id: str | None = None,
        normalization: JsonStateNormalization | None = None,
        snapshot_size_limit_bytes: int = 1_000_000,
    ) -> None:
        if authority == "environment_self_reported" and observer_id is not None:
            raise ValueError("self-reported callback state cannot name an independent observer")
        resolved_observer_id = (
            observer_id or environment_id if authority == "independent_observer" else None
        )
        self.capabilities = StateEnvironmentCapabilities(
            environment_id=environment_id,
            snapshot_size_limit_bytes=snapshot_size_limit_bytes,
            supports_reset=True,
            supports_setup=setup is not None,
            supports_snapshot=True,
            supports_cleanup=True,
            state_observation_authority=authority,
            state_observer_id=resolved_observer_id,
            supports_deterministic_replay=False,
        )
        self._reset_callback = reset
        self._setup_callback = setup
        self._snapshot_callback = snapshot
        self._cleanup_callback = cleanup or reset
        self._normalization = normalization or JsonStateNormalization()
        self._generation = 0
        self._generation_lock = threading.Lock()
        self._sync_runner = _SyncCallbackRunner()

    @property
    def normalization(self) -> JsonStateNormalization:
        return self._normalization

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(
                {
                    "capabilities": self.capabilities.model_dump(mode="json", exclude_none=True),
                    "normalization": self._normalization.model_dump(mode="json"),
                }
            )
        ).hexdigest()

    async def reset(self, request: StateFixtureRequest) -> StateOperationResult:
        generation = self._next_generation()
        await self._call_lifecycle(self._reset_callback, request, "reset", generation)
        return _successful_operation(request, "reset")

    async def setup(self, request: StateFixtureRequest) -> StateOperationResult:
        callback = self._setup_callback
        if callback is None:
            raise RuntimeError("state setup callback is unavailable")
        await self._call_lifecycle(callback, request, "setup", self._current_generation())
        return _successful_operation(request, "setup")

    async def snapshot(self, request: StateFixtureRequest) -> StateSnapshot:
        context = self._context(request, "snapshot", self._current_generation())
        value = await self._call(self._snapshot_callback, context)
        normalized = normalize_json_state(value, self._normalization)
        if len(_canonical_json(normalized)) > self.capabilities.snapshot_size_limit_bytes:
            raise ValueError("state callback snapshot exceeds its configured size limit")
        return StateSnapshot(
            id=f"{request.correlation_id}:snapshot",
            fixture_id=request.fixture_id,
            correlation_id=request.correlation_id,
            source_id=self.capabilities.environment_id,
            value=normalized,
            authority=cast(
                StateObservationAuthority,
                self.capabilities.state_observation_authority,
            ),
            observer_id=self.capabilities.state_observer_id,
        )

    async def cleanup(self, request: StateFixtureRequest) -> StateOperationResult:
        generation = self._next_generation()
        await self._call_lifecycle(self._cleanup_callback, request, "cleanup", generation)
        return _successful_operation(request, "cleanup")

    async def _call_lifecycle(
        self,
        callback: StateLifecycleCallback,
        request: StateFixtureRequest,
        phase: Literal["reset", "setup", "cleanup"],
        generation: int,
    ) -> None:
        result = await self._call(callback, self._context(request, phase, generation))
        if result is not None:
            raise TypeError("state lifecycle callbacks must return None")

    async def _call[ResultT](
        self,
        callback: Callable[[StateCallbackContext], ResultT | Awaitable[ResultT]],
        context: StateCallbackContext,
    ) -> ResultT:
        if inspect.iscoroutinefunction(callback):
            return await cast(Awaitable[ResultT], callback(context))
        result = await self._sync_runner.call(callback, context)
        if inspect.isawaitable(result):
            return await cast(Awaitable[ResultT], result)
        return result

    def _context(
        self,
        request: StateFixtureRequest,
        phase: Literal["reset", "setup", "snapshot", "cleanup"],
        generation: int,
    ) -> StateCallbackContext:
        return StateCallbackContext(
            phase=phase,
            fixture_id=request.fixture_id,
            case_id=request.case_id,
            session_id=request.session_id,
            correlation_id=request.correlation_id,
            turn_id=request.turn_id,
            generation=generation,
            case_context=request.configuration,
        )

    def _next_generation(self) -> int:
        with self._generation_lock:
            self._generation += 1
            return self._generation

    def _current_generation(self) -> int:
        with self._generation_lock:
            return self._generation


def normalize_json_state(
    value: JsonValue,
    normalization: JsonStateNormalization | None = None,
) -> JsonValue:
    configuration = normalization or JsonStateNormalization()
    return _normalize_json_value(
        value,
        path="",
        depth=0,
        volatile_paths=frozenset(configuration.volatile_json_pointers),
        unordered_paths=frozenset(configuration.unordered_json_pointers),
    )


def json_state_digest(
    value: JsonValue,
    normalization: JsonStateNormalization | None = None,
) -> str:
    normalized = normalize_json_state(value, normalization)
    return hashlib.sha256(_canonical_json(normalized)).hexdigest()


def diff_json_states(
    before: JsonValue,
    after: JsonValue,
    normalization: JsonStateNormalization | None = None,
    *,
    max_differences: int = 10_000,
) -> tuple[JsonStateDifference, ...]:
    if type(max_differences) is not int or max_differences < 1:
        raise ValueError("max_differences must be a positive integer")
    normalized_before = normalize_json_state(before, normalization)
    normalized_after = normalize_json_state(after, normalization)
    differences: list[JsonStateDifference] = []
    _append_differences(
        normalized_before,
        normalized_after,
        path="",
        differences=differences,
        max_differences=max_differences,
    )
    return tuple(differences)


async def check_deterministic_reset(
    environment: CallbackStateEnvironment,
    request: StateFixtureRequest,
    *,
    timeout_seconds: float = 5.0,
) -> StateResetConformanceReport:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    async with asyncio.timeout(timeout_seconds):
        await environment.reset(request)
        first = await environment.snapshot(request)
        await environment.reset(request)
        second = await environment.snapshot(request)
    first_digest = json_state_digest(first.value)
    second_digest = json_state_digest(second.value)
    differences = diff_json_states(first.value, second.value)
    environment.capabilities = environment.capabilities.model_copy(
        update={"supports_deterministic_replay": first_digest == second_digest}
    )
    return StateResetConformanceReport(
        fixture_id=request.fixture_id,
        first_digest=first_digest,
        second_digest=second_digest,
        deterministic=first_digest == second_digest,
        differences=differences,
    )


def _successful_operation(
    request: StateFixtureRequest,
    operation: Literal["reset", "setup", "cleanup"],
) -> StateOperationResult:
    resets_state = operation in {"reset", "cleanup"}
    return StateOperationResult(
        id=f"{request.correlation_id}:{operation}",
        fixture_id=request.fixture_id,
        correlation_id=request.correlation_id,
        operation=operation,
        succeeded=True,
        reset_session_requested=resets_state,
        reset_session_acknowledged=resets_state,
        reset_environment_requested=resets_state,
        reset_environment_acknowledged=resets_state,
    )


def _normalize_json_value(
    value: JsonValue,
    *,
    path: str,
    depth: int,
    volatile_paths: frozenset[str],
    unordered_paths: frozenset[str],
) -> JsonValue:
    if depth > _MAXIMUM_JSON_DEPTH:
        raise ValueError("state snapshot exceeds the maximum JSON depth")
    if path in volatile_paths:
        return dict(_VOLATILE_VALUE)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("state snapshots cannot contain non-finite numbers")
        return value
    if isinstance(value, list):
        normalized_items = [
            _normalize_json_value(
                item,
                path=f"{path}/{index}",
                depth=depth + 1,
                volatile_paths=volatile_paths,
                unordered_paths=unordered_paths,
            )
            for index, item in enumerate(value)
        ]
        if path in unordered_paths:
            normalized_items.sort(key=_canonical_json)
        return normalized_items
    return {
        key: _normalize_json_value(
            item,
            path=f"{path}/{_escape_pointer_token(key)}",
            depth=depth + 1,
            volatile_paths=volatile_paths,
            unordered_paths=unordered_paths,
        )
        for key, item in sorted(value.items())
    }


def _append_differences(
    before: JsonValue,
    after: JsonValue,
    *,
    path: str,
    differences: list[JsonStateDifference],
    max_differences: int,
) -> None:
    if type(before) is not type(after):
        _append_difference(differences, path, "changed", before, after, max_differences)
        return
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(before.keys() | after.keys()):
            child_path = f"{path}/{_escape_pointer_token(key)}"
            if key not in before:
                _append_difference(
                    differences, child_path, "added", None, after[key], max_differences
                )
            elif key not in after:
                _append_difference(
                    differences, child_path, "removed", before[key], None, max_differences
                )
            else:
                _append_differences(
                    before[key],
                    after[key],
                    path=child_path,
                    differences=differences,
                    max_differences=max_differences,
                )
        return
    if isinstance(before, list) and isinstance(after, list):
        for index in range(max(len(before), len(after))):
            child_path = f"{path}/{index}"
            if index >= len(before):
                _append_difference(
                    differences, child_path, "added", None, after[index], max_differences
                )
            elif index >= len(after):
                _append_difference(
                    differences, child_path, "removed", before[index], None, max_differences
                )
            else:
                _append_differences(
                    before[index],
                    after[index],
                    path=child_path,
                    differences=differences,
                    max_differences=max_differences,
                )
        return
    if before != after:
        _append_difference(differences, path, "changed", before, after, max_differences)


def _append_difference(
    differences: list[JsonStateDifference],
    path: str,
    kind: Literal["added", "removed", "changed"],
    before: JsonValue,
    after: JsonValue,
    maximum: int,
) -> None:
    if len(differences) >= maximum:
        raise ValueError("state difference exceeds max_differences")
    differences.append(JsonStateDifference(path=path, kind=kind, before=before, after=after))


def _canonical_json(value: JsonValue) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _escape_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")
