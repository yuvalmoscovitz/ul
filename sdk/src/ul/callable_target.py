from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Literal, cast

from pydantic import JsonValue
from ul_core.contracts import DatasetTargetExecutor
from ul_core.dataset import ObservedAgentOutput
from ul_core.models import SafetyEnvelope

_MAXIMUM_VALUE_BYTES = 1_000_000


def callable_target_factory(
    invoke: Callable[[str], object],
    *,
    safety_envelope: SafetyEnvelope,
    fresh_state_per_execution: Literal[True],
    reset: Callable[[], object] | None = None,
    snapshot: Callable[[object], object] | None = None,
    cleanup: Callable[[], object] | None = None,
) -> Callable[[], DatasetTargetExecutor]:
    if fresh_state_per_execution is not True:
        raise ValueError("callable targets must explicitly declare fresh state per execution")

    def create_target() -> DatasetTargetExecutor:
        return _CallableDatasetTarget(
            invoke=invoke,
            safety_envelope=safety_envelope,
            reset=reset,
            snapshot=snapshot,
            cleanup=cleanup,
        )

    return create_target


class _CallableDatasetTarget:
    fresh_state_per_execution = True

    def __init__(
        self,
        *,
        invoke: Callable[[str], object],
        safety_envelope: SafetyEnvelope,
        reset: Callable[[], object] | None,
        snapshot: Callable[[object], object] | None,
        cleanup: Callable[[], object] | None,
    ) -> None:
        self._invoke = invoke
        self.safety_envelope = safety_envelope
        self._reset = reset
        self._snapshot = snapshot
        self._cleanup = cleanup

    async def execute(self, raw_input: str) -> ObservedAgentOutput:
        if self._reset is not None:
            await _call_hook(self._reset, phase="reset")
        result = await _call_hook(self._invoke, raw_input, phase="invoke")
        raw_output = _validate_json_value(result, name="result")
        metadata: dict[str, JsonValue] = {}
        if self._snapshot is not None:
            state = await _call_hook(self._snapshot, result, phase="snapshot")
            metadata["committed_state_snapshot"] = _validate_json_value(state, name="snapshot")
        return ObservedAgentOutput(raw_output=raw_output, metadata=metadata)

    async def aclose(self) -> None:
        if self._cleanup is not None:
            await _call_hook(self._cleanup, phase="cleanup")


async def _call_hook(hook: Callable[..., object], *args: object, phase: str) -> object:
    try:
        result = hook(*args)
        if inspect.isawaitable(result):
            return await cast(Awaitable[object], result)
        return result
    except Exception:
        raise RuntimeError(f"Callable target {phase} failed") from None


def _validate_json_value(value: object, *, name: str) -> JsonValue:
    try:
        observation = ObservedAgentOutput.model_validate({"raw_output": value})
        encoded = observation.model_dump_json().encode("utf-8")
    except Exception:
        raise RuntimeError(f"Callable target returned an invalid {name}") from None
    if len(encoded) > _MAXIMUM_VALUE_BYTES:
        raise RuntimeError(f"Callable target {name} exceeded 1 MB")
    return observation.raw_output
