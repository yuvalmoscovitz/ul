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
from ul_core.contracts import StateEnvironment
from ul_core.evaluation import (
    StateEnvironmentCapabilities,
    StateFixtureRequest,
    StateObservationAuthority,
    StateOperationResult,
    StateSnapshot,
)

_MAXIMUM_JSON_DEPTH = 100
_DEFAULT_MAXIMUM_JSON_NODES = 100_000
_STABLE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,499}$"
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


class StateAdapterIdentity(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    adapter_id: str = Field(pattern=_STABLE_ID_PATTERN)
    adapter_version: str = Field(pattern=_STABLE_ID_PATTERN)
    fixture_id: str = Field(pattern=_STABLE_ID_PATTERN)
    fixture_version: str = Field(pattern=_STABLE_ID_PATTERN)


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
                with self._lock:
                    self._running = False
                loop.call_soon_threadsafe(deliver_error, error)
            else:
                with self._lock:
                    self._running = False
                loop.call_soon_threadsafe(deliver_result, value)

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
        identity: StateAdapterIdentity,
        reset: StateLifecycleCallback,
        snapshot: StateSnapshotCallback,
        setup: StateLifecycleCallback | None = None,
        cleanup: StateLifecycleCallback | None = None,
        authority: StateObservationAuthority = "environment_self_reported",
        observer_id: str | None = None,
        normalization: JsonStateNormalization | None = None,
        snapshot_size_limit_bytes: int = 1_000_000,
        snapshot_node_limit: int = _DEFAULT_MAXIMUM_JSON_NODES,
    ) -> None:
        if type(snapshot_node_limit) is not int or not 1 <= snapshot_node_limit <= 1_000_000:
            raise ValueError("snapshot_node_limit must be an integer between 1 and 1000000")
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
        self.identity = identity
        self._reset_callback = reset
        self._setup_callback = setup
        self._snapshot_callback = snapshot
        self._cleanup_callback = cleanup or reset
        self._normalization = normalization or JsonStateNormalization()
        self._snapshot_node_limit = snapshot_node_limit
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
                    "identity": self.identity.model_dump(mode="json"),
                    "normalization": self._normalization.model_dump(mode="json"),
                    "snapshot_node_limit": self._snapshot_node_limit,
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
        normalized = normalize_json_state(
            value,
            self._normalization,
            max_bytes=self.capabilities.snapshot_size_limit_bytes,
            max_nodes=self._snapshot_node_limit,
        )
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
        if request.fixture_id != self.identity.fixture_id:
            raise ValueError(
                "state callback request does not match its configured fixture identity"
            )
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


def require_state_adapter_identity(
    environment: StateEnvironment,
) -> StateAdapterIdentity:
    identity = getattr(environment, "identity", None)
    if not isinstance(identity, StateAdapterIdentity):
        raise ValueError(
            "composed state environments must provide a validated StateAdapterIdentity"
        )
    return identity


def normalize_json_state(
    value: JsonValue,
    normalization: JsonStateNormalization | None = None,
    *,
    max_bytes: int = 1_000_000,
    max_nodes: int = _DEFAULT_MAXIMUM_JSON_NODES,
) -> JsonValue:
    _validate_json_limits(value, max_bytes=max_bytes, max_nodes=max_nodes)
    configuration = normalization or JsonStateNormalization()
    normalized = _normalize_json_value(
        value,
        path="",
        depth=0,
        volatile_paths=frozenset(configuration.volatile_json_pointers),
        unordered_paths=frozenset(configuration.unordered_json_pointers),
    )
    bounded_json_size(normalized, max_bytes=max_bytes)
    return normalized


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


def bounded_json_size(value: object, *, max_bytes: int) -> int:
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    total_bytes = 0
    active_containers: set[int] = set()

    def add_bytes(amount: int) -> None:
        nonlocal total_bytes
        total_bytes += amount
        if total_bytes > max_bytes:
            raise ValueError("JSON value exceeds its configured size limit")

    def add_string(value: str) -> None:
        add_bytes(2)
        for character in value:
            if character in {'"', "\\"} or character in {"\b", "\f", "\n", "\r", "\t"}:
                add_bytes(2)
            elif ord(character) < 0x20:
                add_bytes(6)
            else:
                add_bytes(len(character.encode("utf-8")))

    def visit(item: object, depth: int) -> None:
        if depth > _MAXIMUM_JSON_DEPTH:
            raise ValueError("JSON value exceeds the maximum depth")
        item_type = type(item)
        if item is None:
            add_bytes(4)
            return
        if item_type is bool:
            add_bytes(4 if item else 5)
            return
        if item_type is str:
            add_string(cast(str, item))
            return
        if item_type is int:
            try:
                add_bytes(len(str(item)))
            except ValueError:
                raise ValueError("value must contain bounded standard JSON") from None
            return
        if item_type is float:
            float_item = cast(float, item)
            if not math.isfinite(float_item):
                raise ValueError("value must contain bounded standard JSON")
            add_bytes(len(json.dumps(float_item)))
            return
        if item_type not in {list, dict}:
            raise ValueError("value must contain bounded standard JSON")
        container_id = id(item)
        if container_id in active_containers:
            raise ValueError("value must contain bounded standard JSON")
        active_containers.add(container_id)
        try:
            if item_type is list:
                list_item = cast(list[object], item)
                add_bytes(2 + max(0, len(list_item) - 1))
                for child in list_item:
                    visit(child, depth + 1)
                return
            dict_item = cast(dict[object, object], item)
            add_bytes(2 + max(0, len(dict_item) - 1))
            for key, child in dict_item.items():
                if type(key) is not str:
                    raise ValueError("value must contain bounded standard JSON")
                add_string(key)
                add_bytes(1)
                visit(child, depth + 1)
        finally:
            active_containers.remove(container_id)

    visit(value, 0)
    return total_bytes


def _validate_json_limits(
    value: object,
    *,
    max_bytes: int,
    max_nodes: int,
) -> None:
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer")
    node_count = [0]
    _validate_json_structure(
        value,
        depth=0,
        active_containers=set(),
        node_count=node_count,
        max_nodes=max_nodes,
    )
    bounded_json_size(value, max_bytes=max_bytes)


def _validate_json_structure(
    value: object,
    *,
    depth: int,
    active_containers: set[int],
    node_count: list[int],
    max_nodes: int,
) -> None:
    if depth > _MAXIMUM_JSON_DEPTH:
        raise ValueError("state snapshot exceeds the maximum JSON depth")
    node_count[0] += 1
    if node_count[0] > max_nodes:
        raise ValueError("state snapshot exceeds the maximum JSON node count")
    value_type = type(value)
    if value is None or value_type in {str, bool, int}:
        return
    if value_type is float:
        if not math.isfinite(cast(float, value)):
            raise ValueError("state snapshots cannot contain non-finite numbers")
        return
    if value_type not in {list, dict}:
        raise TypeError("state snapshots must contain standard JSON values")
    container_id = id(value)
    if container_id in active_containers:
        raise ValueError("state snapshots cannot contain cycles")
    active_containers.add(container_id)
    try:
        if value_type is list:
            for item in cast(list[object], value):
                _validate_json_structure(
                    item,
                    depth=depth + 1,
                    active_containers=active_containers,
                    node_count=node_count,
                    max_nodes=max_nodes,
                )
            return
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise TypeError("state snapshot object keys must be strings")
            _validate_json_structure(
                item,
                depth=depth + 1,
                active_containers=active_containers,
                node_count=node_count,
                max_nodes=max_nodes,
            )
    finally:
        active_containers.remove(container_id)


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
