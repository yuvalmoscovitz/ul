from __future__ import annotations

import asyncio
import importlib
import inspect
import re
from collections.abc import Awaitable, Callable
from contextlib import suppress
from types import TracebackType
from typing import cast

from ul_core.contracts import DatasetTargetExecutor, DatasetTargetLifecycleError
from ul_core.dataset import ObservedAgentOutput
from ul_core.models import SafetyEnvelope

_FACTORY_REFERENCE_PATTERN = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$")


def validate_python_target_factory_reference(reference: str) -> str:
    if len(reference) > 500 or not _FACTORY_REFERENCE_PATTERN.fullmatch(reference):
        raise ValueError("Python target factory must use the form package.module:create_target")
    return reference


def load_python_dataset_target(
    reference: str,
    *,
    max_target_calls: int | None = None,
) -> PythonDatasetTarget:
    validate_python_target_factory_reference(reference)
    if max_target_calls is not None and (
        type(max_target_calls) is not int or max_target_calls <= 0
    ):
        raise ValueError("max_target_calls must be positive")
    module_name, factory_name = reference.split(":", maxsplit=1)
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
    except Exception:
        raise RuntimeError("Python target factory could not be imported") from None
    if not callable(factory):
        raise ValueError("Python target factory must be callable")
    try:
        target = cast(Callable[[], object], factory)()
    except Exception:
        raise RuntimeError("Python target factory failed during initialization") from None
    if inspect.isawaitable(target):
        close = getattr(target, "close", None)
        if callable(close):
            close()
        raise ValueError("Python target factory must be synchronous")
    if not isinstance(target, DatasetTargetExecutor):
        _close_rejected_target(target)
        raise ValueError("Python target factory returned an invalid dataset target")
    try:
        safety_envelope: object = target.safety_envelope
        fresh_state_per_execution: object = target.fresh_state_per_execution
    except Exception:
        _close_rejected_target(target)
        raise ValueError("Python target factory returned an invalid dataset target") from None
    try:
        _validate_target_properties(safety_envelope, fresh_state_per_execution)
    except ValueError:
        _close_rejected_target(target)
        raise
    return PythonDatasetTarget(target, max_target_calls=max_target_calls)


class PythonDatasetTarget:
    def __init__(
        self,
        target: DatasetTargetExecutor,
        *,
        max_target_calls: int | None,
    ) -> None:
        if max_target_calls is not None and (
            type(max_target_calls) is not int or max_target_calls <= 0
        ):
            raise ValueError("max_target_calls must be positive")
        self._target = target
        self._remaining_target_calls = max_target_calls

    @property
    def safety_envelope(self) -> SafetyEnvelope:
        return self._target.safety_envelope

    @property
    def fresh_state_per_execution(self) -> bool:
        return self._target.fresh_state_per_execution

    async def __aenter__(self) -> PythonDatasetTarget:
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        close = getattr(self._target, "aclose", None)
        if close is None:
            return
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception:
            raise RuntimeError("Python dataset target failed during cleanup") from None

    async def execute(self, raw_input: str) -> ObservedAgentOutput:
        self._reserve_target_call()
        try:
            output: object = await self._target.execute(raw_input)
        except DatasetTargetLifecycleError:
            raise
        except Exception:
            raise RuntimeError("Python dataset target execution failed") from None
        return _validate_observed_output(output)

    def _reserve_target_call(self) -> None:
        if self._remaining_target_calls is None:
            return
        if self._remaining_target_calls < 1:
            raise RuntimeError("Python dataset target call budget exhausted")
        self._remaining_target_calls -= 1


def _validate_target_properties(safety_envelope: object, fresh_state: object) -> None:
    if not isinstance(safety_envelope, SafetyEnvelope) or not isinstance(fresh_state, bool):
        raise ValueError("Python target factory returned an invalid dataset target")


def _validate_observed_output(output: object) -> ObservedAgentOutput:
    if not isinstance(output, ObservedAgentOutput):
        raise RuntimeError("Python dataset target returned an invalid observation")
    return output


def _close_rejected_target(target: object) -> None:
    try:
        close = getattr(target, "aclose", None)
        if close is None:
            close = getattr(target, "close", None)
        if not callable(close):
            return
        result = close()
        if not inspect.isawaitable(result):
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            with suppress(Exception):
                asyncio.run(_await_cleanup(result))
        else:
            cleanup_task = running_loop.create_task(_await_cleanup(result))
            cleanup_task.add_done_callback(_discard_cleanup_error)
    except Exception:
        pass


def _discard_cleanup_error(cleanup_task: asyncio.Task[object]) -> None:
    with suppress(asyncio.CancelledError, Exception):
        cleanup_task.exception()


async def _await_cleanup(cleanup: Awaitable[object]) -> object:
    return await cleanup
