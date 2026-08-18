from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from contextlib import suppress
from threading import Thread
from typing import Literal, cast

from pydantic import JsonValue
from ul_core.contracts import DatasetTargetExecutor, DatasetTargetLifecycleError
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
        self._execution_lock = asyncio.Lock()
        self._active_sync_thread: Thread | None = None

    async def execute(self, raw_input: str) -> ObservedAgentOutput:
        async with self._execution_lock:
            self._reject_overlapping_sync_hook()
            if self._reset is not None:
                await self._call_hook(self._reset, phase="reset")
            result = await self._call_hook(self._invoke, raw_input, phase="invoke")
            raw_output = _validate_json_value(result, name="result")
            metadata: dict[str, JsonValue] = {}
            if self._snapshot is not None:
                state = await self._call_hook(self._snapshot, result, phase="snapshot")
                metadata["committed_state_snapshot"] = _validate_json_value(state, name="snapshot")
            return ObservedAgentOutput(raw_output=raw_output, metadata=metadata)

    async def aclose(self) -> None:
        async with self._execution_lock:
            self._reject_overlapping_sync_hook()
            if self._cleanup is not None:
                await self._call_hook(self._cleanup, phase="cleanup")

    def _reject_overlapping_sync_hook(self) -> None:
        if self._active_sync_thread is None:
            return
        if not self._active_sync_thread.is_alive():
            self._active_sync_thread = None
            return
        raise DatasetTargetLifecycleError(
            failed_phase="previous_sync_hook_still_running",
            completed_phases=(),
            cleanup_reset_failed=False,
            target_state_uncertain=True,
        )

    async def _call_hook(self, hook: Callable[..., object], *args: object, phase: str) -> object:
        try:
            if _is_async_callable(hook):
                result = hook(*args)
            else:
                result = await self._call_sync_hook(hook, args)
            if inspect.isawaitable(result):
                return await cast(Awaitable[object], result)
            return result
        except DatasetTargetLifecycleError:
            raise
        except Exception:
            raise RuntimeError(f"Callable target {phase} failed") from None

    async def _call_sync_hook(
        self, hook: Callable[..., object], args: tuple[object, ...]
    ) -> object:
        event_loop = asyncio.get_running_loop()
        result_future: asyncio.Future[object] = event_loop.create_future()

        def run_hook() -> None:
            try:
                result = hook(*args)
            except BaseException:
                _notify_event_loop(event_loop, result_future, None, failed=True)
            else:
                _notify_event_loop(event_loop, result_future, result, failed=False)

        self._active_sync_thread = Thread(target=run_hook, daemon=True)
        self._active_sync_thread.start()
        result = await result_future
        self._active_sync_thread = None
        return result


def _is_async_callable(hook: Callable[..., object]) -> bool:
    return inspect.iscoroutinefunction(hook) or inspect.iscoroutinefunction(type(hook).__call__)


def _notify_event_loop(
    event_loop: asyncio.AbstractEventLoop,
    result_future: asyncio.Future[object],
    result: object,
    *,
    failed: bool,
) -> None:
    with suppress(RuntimeError):
        event_loop.call_soon_threadsafe(_finish_sync_hook, result_future, result, failed)


def _finish_sync_hook(result_future: asyncio.Future[object], result: object, failed: bool) -> None:
    if result_future.done():
        return
    if failed:
        result_future.set_exception(RuntimeError("Callable target hook failed"))
    else:
        result_future.set_result(result)


def _validate_json_value(value: object, *, name: str) -> JsonValue:
    try:
        observation = ObservedAgentOutput.model_validate({"raw_output": value})
        encoded = observation.model_dump_json().encode("utf-8")
    except Exception:
        raise RuntimeError(f"Callable target returned an invalid {name}") from None
    if len(encoded) > _MAXIMUM_VALUE_BYTES:
        raise RuntimeError(f"Callable target {name} exceeded 1 MB")
    return observation.raw_output
