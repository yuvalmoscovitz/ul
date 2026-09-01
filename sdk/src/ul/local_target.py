from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Annotated, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError
from ul_core.contracts import StateEnvironment
from ul_core.evaluation import (
    EnvironmentCapabilities,
    EnvironmentLifecycleFailureCode,
    EvaluationCase,
    ExecutionEvidence,
    ProbeExecutionEvent,
    ProbeInvokerCapabilities,
    ProbeRequest,
    ProbeResult,
)

from ul.outcome_projection import OutcomeProjection
from ul.probe_execution import CapabilityExecutionError, ComposedEnvironmentExecutor
from ul.state_hooks import CallbackStateEnvironment, require_state_adapter_identity

_PROTOCOL_VERSION = "1.0.0"
_MAXIMUM_CONFIG_BYTES = 1_000_000
_MAXIMUM_JSON_DEPTH = 100
_MAXIMUM_ARGUMENTS = 128
_MAXIMUM_ARGUMENT_BYTES = 16_384
_MAXIMUM_ENVIRONMENT_VARIABLES = 128
_MAXIMUM_EXECUTABLE_BYTES = 256 * 1024 * 1024
_TARGET_REFERENCE_PATTERN = re.compile(
    r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
)
_ENVIRONMENT_VARIABLE_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class LocalTargetLimits(_StrictModel):
    startup_timeout_seconds: float = Field(default=10, gt=0, le=300)
    turn_timeout_seconds: float = Field(default=20, gt=0, le=3_600)
    total_execution_timeout_seconds: float = Field(default=300, gt=0, le=86_400)
    shutdown_timeout_seconds: float = Field(default=2, gt=0, le=60)
    max_executions: int = Field(default=100, ge=1, le=100_000)
    max_input_bytes: int = Field(default=1_000_000, ge=1, le=64_000_000)
    max_output_bytes: int = Field(default=1_000_000, ge=1, le=64_000_000)
    max_stderr_bytes: int = Field(default=64_000, ge=1, le=64_000_000)

    def model_post_init(self, context: object, /) -> None:
        del context
        for value in (
            self.startup_timeout_seconds,
            self.turn_timeout_seconds,
            self.total_execution_timeout_seconds,
            self.shutdown_timeout_seconds,
        ):
            if not math.isfinite(value):
                raise ValueError("local target timeouts must be finite")


class _LocalTargetBase(_StrictModel):
    version: Literal[1] = 1
    target_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
    working_directory: Path
    environment_allowlist: tuple[str, ...] = Field(
        default=(), max_length=_MAXIMUM_ENVIRONMENT_VARIABLES
    )
    limits: LocalTargetLimits = Field(default_factory=LocalTargetLimits)
    outcome: OutcomeProjection | None = None

    def model_post_init(self, context: object, /) -> None:
        del context
        if len(set(self.environment_allowlist)) != len(self.environment_allowlist):
            raise ValueError("environment allowlist entries must be unique")
        if any(
            _ENVIRONMENT_VARIABLE_PATTERN.fullmatch(name) is None
            for name in self.environment_allowlist
        ):
            raise ValueError("environment allowlist contains an invalid variable name")


class PythonCallableTargetConfig(_LocalTargetBase):
    kind: Literal["python_callable"] = "python_callable"
    interpreter: Path
    target: str = Field(min_length=3, max_length=500)
    input_mode: Literal["value", "request"] = "value"

    def model_post_init(self, context: object, /) -> None:
        super().model_post_init(context)
        if _TARGET_REFERENCE_PATTERN.fullmatch(self.target) is None:
            raise ValueError("Python target must use module.path:callable.path syntax")


class CommandTargetConfig(_LocalTargetBase):
    kind: Literal["command"] = "command"
    argv: tuple[str, ...] = Field(min_length=1, max_length=_MAXIMUM_ARGUMENTS)

    def model_post_init(self, context: object, /) -> None:
        super().model_post_init(context)
        if any(
            not argument
            or "\x00" in argument
            or len(argument.encode("utf-8")) > _MAXIMUM_ARGUMENT_BYTES
            for argument in self.argv
        ):
            raise ValueError("command arguments must be non-empty bounded UTF-8 strings")


LocalTargetConfig = Annotated[
    PythonCallableTargetConfig | CommandTargetConfig,
    Field(discriminator="kind"),
]
_LOCAL_TARGET_CONFIG_ADAPTER = cast(TypeAdapter[LocalTargetConfig], TypeAdapter(LocalTargetConfig))


class LocalTargetDryRunPlan(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    target_id: str
    target_kind: Literal["python_callable", "command"]
    worker_command: tuple[str, ...]
    selected_executable: str
    resolved_executable: str
    capability_level: Literal["response_only"] = "response_only"
    maximum_executions: int = Field(ge=1)
    maximum_active_wall_seconds: float = Field(gt=0)
    maximum_input_bytes: int = Field(ge=1)
    maximum_output_bytes: int = Field(ge=1)
    maximum_stderr_bytes: int = Field(ge=1)
    environment_variable_names: tuple[str, ...]
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _RuntimeReport(_StrictModel):
    name: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=200)


class _ReadyMessage(_StrictModel):
    protocol_version: Literal["1.0.0"]
    type: Literal["ready"]
    request_id: Literal["startup"]
    runtime: _RuntimeReport
    target_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class _SessionReadyMessage(_StrictModel):
    protocol_version: Literal["1.0.0"]
    type: Literal["session_ready"]
    request_id: str = Field(min_length=1, max_length=500)
    session_id: str = Field(min_length=1, max_length=500)


class _WorkerExecutionEvent(_StrictModel):
    id: str = Field(min_length=1, max_length=200)
    kind: str = Field(min_length=1, max_length=200)
    payload: JsonValue


class _ResultMessage(_StrictModel):
    protocol_version: Literal["1.0.0"]
    type: Literal["result"]
    request_id: str = Field(min_length=1, max_length=500)
    response: JsonValue
    execution_events: tuple[_WorkerExecutionEvent, ...] = Field(default=(), max_length=1_000)


class _ShutdownMessage(_StrictModel):
    protocol_version: Literal["1.0.0"]
    type: Literal["shutdown_complete"]
    request_id: Literal["shutdown"]


class _ErrorMessage(_StrictModel):
    protocol_version: Literal["1.0.0"]
    type: Literal["error"]
    request_id: str = Field(min_length=1, max_length=500)
    code: str = Field(min_length=1, max_length=100)


class _LocalTargetFailure(RuntimeError):
    def __init__(
        self,
        code: EnvironmentLifecycleFailureCode,
        *,
        delivery_uncertain: bool,
        not_delivered: bool = False,
        safe_reason: str | None = None,
    ) -> None:
        super().__init__(safe_reason or "local target execution failed")
        self.code: EnvironmentLifecycleFailureCode = code
        self.delivery_uncertain = delivery_uncertain
        self.not_delivered = not_delivered
        self.safe_reason = safe_reason


@dataclass(frozen=True)
class _FileIdentity:
    resolved_path: Path
    device: int
    inode: int
    size: int
    modified_nanoseconds: int
    sha256: str


def load_local_target_config(path: str | Path) -> LocalTargetConfig:
    config_path = Path(path)
    try:
        with _open_regular_file(
            config_path,
            maximum_bytes=_MAXIMUM_CONFIG_BYTES,
            allow_final_symlink=False,
        ) as (descriptor, _):
            encoded = _read_bounded_descriptor(descriptor, _MAXIMUM_CONFIG_BYTES)
    except _FileTooLargeError:
        raise ValueError("local target configuration exceeds the 1 MB limit") from None
    except OSError:
        raise RuntimeError("local target configuration could not be read") from None
    try:
        raw = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
        _reject_deep_json(raw)
        normalized = json.dumps(
            raw,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        return _LOCAL_TARGET_CONFIG_ADAPTER.validate_json(normalized, strict=True)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValidationError, ValueError):
        raise ValueError("local target configuration is invalid") from None


def local_target_config_sha256(config: LocalTargetConfig) -> str:
    encoded = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _composed_local_target_config_sha256(
    config: LocalTargetConfig,
    *,
    state_environment: StateEnvironment | None,
) -> str:
    if state_environment is None:
        return local_target_config_sha256(config)
    state_identity = require_state_adapter_identity(state_environment)
    encoded = json.dumps(
        {
            "local_target_sha256": local_target_config_sha256(config),
            "state_capabilities": state_environment.capabilities.model_dump(
                mode="json", exclude_none=True
            ),
            "state_identity": state_identity.model_dump(mode="json"),
            **(
                {"state_config_sha256": state_environment.config_sha256}
                if isinstance(state_environment, CallbackStateEnvironment)
                else {}
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_local_target_config(config: LocalTargetConfig) -> None:
    working_directory = config.working_directory
    if not working_directory.is_absolute() or not working_directory.is_dir():
        raise ValueError("local target working_directory must be an existing absolute directory")
    executable_path = Path(_target_command(config)[0])
    if not executable_path.is_absolute():
        raise ValueError("local target executable must be an existing absolute file")
    try:
        with _open_executable_identity(executable_path):
            pass
    except (OSError, _FileTooLargeError):
        raise ValueError(
            "local target executable must be an existing bounded regular file"
        ) from None
    if sys.platform != "win32" and not os.access(executable_path.resolve(strict=True), os.X_OK):
        raise ValueError("local target executable must be executable")
    missing_variables = tuple(
        name for name in config.environment_allowlist if name not in os.environ
    )
    if missing_variables:
        raise ValueError(
            "local target allowlisted environment variables are missing: "
            + ", ".join(missing_variables)
        )


def create_local_target_dry_run_plan(config: LocalTargetConfig) -> LocalTargetDryRunPlan:
    validate_local_target_config(config)
    selected_executable = Path(_target_command(config)[0])
    with _open_executable_identity(selected_executable) as (_, executable_identity):
        resolved_executable = executable_identity.resolved_path
    return LocalTargetDryRunPlan(
        target_id=config.target_id,
        target_kind=config.kind,
        worker_command=_target_command(config),
        selected_executable=str(selected_executable),
        resolved_executable=str(resolved_executable),
        maximum_executions=config.limits.max_executions,
        maximum_active_wall_seconds=(
            config.limits.total_execution_timeout_seconds + config.limits.shutdown_timeout_seconds
        ),
        maximum_input_bytes=config.limits.max_input_bytes,
        maximum_output_bytes=config.limits.max_output_bytes,
        maximum_stderr_bytes=config.limits.max_stderr_bytes,
        environment_variable_names=config.environment_allowlist,
        config_sha256=local_target_config_sha256(config),
    )


class _LocalTargetInvoker:
    def __init__(self, config: LocalTargetConfig) -> None:
        validate_local_target_config(config)
        self._config = config
        self._config_sha256 = local_target_config_sha256(config)
        with _open_executable_identity(Path(_target_command(config)[0])) as (_, identity):
            self._executable_identity = identity
        self._executable_sha256 = self._executable_identity.sha256
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_bytes = 0
        self._stderr_overflow = False
        self._active_session_id: str | None = None
        self._runtime_report: _RuntimeReport | None = None
        self._target_sha256: str | None = None
        self._worker_attempt = 0
        self._execution_count = 0
        self._consumed_execution_seconds = 0.0
        self._lock = asyncio.Lock()
        self.capabilities = ProbeInvokerCapabilities(
            invoker_id=config.target_id,
            response_size_limit_bytes=config.limits.max_output_bytes,
            execution_events_size_limit_bytes=config.limits.max_output_bytes,
            supports_structured_execution_events=True,
            supports_conversations=True,
            cancellation_guarantee="best_effort",
        )

    async def invoke(self, request: ProbeRequest) -> ProbeResult:
        async with self._lock:
            if self._execution_count >= self._config.limits.max_executions:
                raise CapabilityExecutionError(
                    "call_budget",
                    "local target execution limit reached",
                    not_delivered=True,
                    _reason_is_safe=True,
                )
            try:
                await self._ensure_started()
                await self._ensure_session(request)
                self._execution_count += 1
                raw_result = await self._exchange(
                    {
                        "protocol_version": _PROTOCOL_VERSION,
                        "type": "invoke",
                        "request_id": request.correlation_id,
                        "case_id": request.case_id,
                        "session_id": request.session_id,
                        "turn": request.turn.model_dump(mode="json"),
                        "context": request.context,
                    },
                    timeout_seconds=self._config.limits.turn_timeout_seconds,
                    delivery_uncertain=True,
                )
                result = _result_message(raw_result)
                if result.request_id != request.correlation_id:
                    raise _LocalTargetFailure(
                        "response_mapping",
                        delivery_uncertain=True,
                    )
                execution_events = self._execution_events(request, result)
                return ProbeResult(
                    id=f"{request.correlation_id}:result",
                    correlation_id=request.correlation_id,
                    response=result.response,
                    execution_events=execution_events,
                )
            except asyncio.CancelledError:
                await self._terminate()
                raise
            except (_LocalTargetFailure, ValidationError, ValueError) as error:
                failure = (
                    error
                    if isinstance(error, _LocalTargetFailure)
                    else _LocalTargetFailure("response_mapping", delivery_uncertain=True)
                )
                await self._terminate()
                raise CapabilityExecutionError(
                    failure.code,
                    failure.safe_reason or "local target execution failed",
                    delivery_uncertain=failure.delivery_uncertain,
                    not_delivered=failure.not_delivered,
                    _reason_is_safe=failure.safe_reason is not None,
                ) from None

    async def aclose(self) -> None:
        async with self._lock:
            process = self._process
            if process is None:
                return
            try:
                raw = await self._exchange(
                    {
                        "protocol_version": _PROTOCOL_VERSION,
                        "type": "shutdown",
                        "request_id": "shutdown",
                    },
                    timeout_seconds=self._config.limits.shutdown_timeout_seconds,
                    delivery_uncertain=False,
                    count_toward_total=False,
                )
                _ShutdownMessage.model_validate(raw)
                async with asyncio.timeout(self._config.limits.shutdown_timeout_seconds):
                    await process.wait()
            except (TimeoutError, _LocalTargetFailure, ValidationError, ValueError):
                pass
            finally:
                await self._terminate()

    async def _ensure_started(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return
        self._worker_attempt += 1
        self._stderr_bytes = 0
        self._stderr_overflow = False
        self._active_session_id = None
        environment = {name: os.environ[name] for name in self._config.environment_allowlist}
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            process = await self._spawn_process(environment)
        except _ExecutableIdentityError:
            raise _LocalTargetFailure(
                "environment_identity",
                delivery_uncertain=False,
                not_delivered=True,
            ) from None
        except OSError:
            raise _LocalTargetFailure(
                "transport_failed",
                delivery_uncertain=False,
                not_delivered=True,
            ) from None
        self._process = process
        self._stderr_task = asyncio.create_task(self._drain_stderr(process))
        raw_ready = await self._exchange(
            {
                "protocol_version": _PROTOCOL_VERSION,
                "type": "start",
                "request_id": "startup",
                "target_id": self._config.target_id,
                "config_sha256": self._config_sha256,
            },
            timeout_seconds=self._config.limits.startup_timeout_seconds,
            delivery_uncertain=False,
        )
        ready = _ReadyMessage.model_validate(raw_ready)
        self._runtime_report = ready.runtime
        self._target_sha256 = (
            ready.target_sha256
            if isinstance(self._config, PythonCallableTargetConfig)
            else self._executable_sha256
        )

    async def _spawn_process(self, environment: dict[str, str]) -> asyncio.subprocess.Process:
        executable_path = Path(_target_command(self._config)[0])
        with _open_executable_identity(executable_path) as (descriptor, identity):
            if identity != self._executable_identity:
                raise _ExecutableIdentityError
            if (
                isinstance(self._config, PythonCallableTargetConfig)
                and sys.platform != "win32"
                and executable_path != identity.resolved_path
            ):
                environment["__PYVENV_LAUNCHER__"] = str(executable_path)
            command = _supervised_target_command(
                self._config,
                identity,
                platform=sys.platform,
            )
            if sys.platform == "win32":
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self._config.working_directory,
                    env=environment,
                    limit=self._config.limits.max_output_bytes + 1,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self._config.working_directory,
                    env=environment,
                    limit=self._config.limits.max_output_bytes + 1,
                    start_new_session=True,
                )
            if (
                not _identity_still_matches_path(executable_path, identity)
                or _descriptor_sha256(descriptor) != identity.sha256
            ):
                _signal_process(process, force=True)
                with suppress(TimeoutError):
                    async with asyncio.timeout(self._config.limits.shutdown_timeout_seconds):
                        await process.wait()
                raise _ExecutableIdentityError
            return process

    async def _ensure_session(self, request: ProbeRequest) -> None:
        if self._active_session_id == request.session_id:
            return
        raw = await self._exchange(
            {
                "protocol_version": _PROTOCOL_VERSION,
                "type": "session_start",
                "request_id": f"{request.correlation_id}:session",
                "case_id": request.case_id,
                "session_id": request.session_id,
            },
            timeout_seconds=self._config.limits.startup_timeout_seconds,
            delivery_uncertain=False,
        )
        session_ready = _SessionReadyMessage.model_validate(raw)
        if session_ready.session_id != request.session_id:
            raise _LocalTargetFailure("response_mapping", delivery_uncertain=False)
        self._active_session_id = request.session_id

    async def _exchange(
        self,
        message: Mapping[str, JsonValue],
        *,
        timeout_seconds: float,
        delivery_uncertain: bool,
        count_toward_total: bool = True,
    ) -> dict[str, object]:
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise _LocalTargetFailure(
                "transport_failed",
                delivery_uncertain=delivery_uncertain,
                not_delivered=not delivery_uncertain,
            )
        encoded = json.dumps(
            message,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > self._config.limits.max_input_bytes:
            raise _LocalTargetFailure(
                "request_too_large",
                delivery_uncertain=False,
                not_delivered=True,
            )
        remaining_total = (
            self._config.limits.total_execution_timeout_seconds - self._consumed_execution_seconds
        )
        if count_toward_total and remaining_total <= 0:
            raise _LocalTargetFailure(
                "response_timeout",
                delivery_uncertain=False,
                not_delivered=True,
            )
        operation_timeout = (
            min(timeout_seconds, remaining_total) if count_toward_total else timeout_seconds
        )
        started_at = time.monotonic()
        try:
            async with asyncio.timeout(operation_timeout):
                process.stdin.write(encoded + b"\n")
                await process.stdin.drain()
                try:
                    line = await process.stdout.readline()
                except ValueError:
                    raise _LocalTargetFailure(
                        "response_too_large",
                        delivery_uncertain=delivery_uncertain,
                    ) from None
        except TimeoutError:
            raise _LocalTargetFailure(
                "response_timeout",
                delivery_uncertain=delivery_uncertain,
            ) from None
        except (BrokenPipeError, ConnectionResetError):
            raise _LocalTargetFailure(
                "transport_failed",
                delivery_uncertain=delivery_uncertain,
            ) from None
        finally:
            if count_toward_total:
                self._consumed_execution_seconds += time.monotonic() - started_at
        if self._stderr_overflow:
            raise _LocalTargetFailure(
                "response_too_large",
                delivery_uncertain=delivery_uncertain,
            )
        if not line:
            raise _LocalTargetFailure(
                "transport_failed",
                delivery_uncertain=delivery_uncertain,
            )
        if len(line) > self._config.limits.max_output_bytes:
            raise _LocalTargetFailure(
                "response_too_large",
                delivery_uncertain=delivery_uncertain,
            )
        try:
            raw = cast(
                object,
                json.loads(
                    line.decode("utf-8"),
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=_reject_nonstandard_constant,
                ),
            )
            _reject_deep_json(raw)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
            raise _LocalTargetFailure(
                "invalid_json",
                delivery_uncertain=delivery_uncertain,
            ) from None
        if not isinstance(raw, dict):
            raise _LocalTargetFailure(
                "response_mapping",
                delivery_uncertain=delivery_uncertain,
            )
        typed_raw = cast(dict[str, object], raw)
        error_message = _parse_error(typed_raw)
        if error_message is not None:
            if error_message.request_id != message["request_id"]:
                raise _LocalTargetFailure(
                    "response_mapping",
                    delivery_uncertain=delivery_uncertain,
                )
            if (
                error_message.code == "target_load_failed"
                and isinstance(self._config, PythonCallableTargetConfig)
                and message.get("type") == "start"
            ):
                raise _LocalTargetFailure(
                    "target_load_failed",
                    delivery_uncertain=False,
                    not_delivered=True,
                    safe_reason="Python target could not be loaded by the selected interpreter",
                )
            raise _LocalTargetFailure(
                "environment_lifecycle_error",
                delivery_uncertain=delivery_uncertain,
            )
        if typed_raw.get("protocol_version") != _PROTOCOL_VERSION or typed_raw.get(
            "request_id"
        ) != message.get("request_id"):
            raise _LocalTargetFailure(
                "response_mapping",
                delivery_uncertain=delivery_uncertain,
            )
        return typed_raw

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        if process.stderr is None:
            return
        while True:
            chunk = await process.stderr.read(4096)
            if not chunk:
                return
            self._stderr_bytes += len(chunk)
            if self._stderr_bytes > self._config.limits.max_stderr_bytes:
                self._stderr_overflow = True
                _signal_process(process, force=True)
                return

    def _execution_events(
        self,
        request: ProbeRequest,
        result: _ResultMessage,
    ) -> tuple[ProbeExecutionEvent, ...]:
        runtime = self._runtime_report
        if runtime is None or self._target_sha256 is None:
            raise _LocalTargetFailure("response_mapping", delivery_uncertain=True)
        runtime_event = ProbeExecutionEvent(
            id=f"{request.correlation_id}:local-runtime",
            correlation_id=request.correlation_id,
            kind="ul.local_target.runtime",
            payload={
                "protocol_version": _PROTOCOL_VERSION,
                "target_id": self._config.target_id,
                "target_kind": self._config.kind,
                "target_sha256": self._target_sha256,
                "config_sha256": self._config_sha256,
                "worker_attempt": self._worker_attempt,
                "execution_attempt": self._execution_count,
                "runtime_name": runtime.name,
                "runtime_version": runtime.version,
                "selected_executable": _target_command(self._config)[0],
                "resolved_executable": str(self._executable_identity.resolved_path),
                "executable_sha256": self._executable_sha256,
            },
        )
        worker_events = tuple(
            ProbeExecutionEvent(
                id=(
                    f"{request.correlation_id}:event:"
                    f"{hashlib.sha256(event.id.encode('utf-8')).hexdigest()[:16]}"
                ),
                correlation_id=request.correlation_id,
                kind=event.kind,
                payload=event.payload,
            )
            for event in result.execution_events
        )
        event_ids = tuple(event.id for event in worker_events)
        if len(event_ids) != len(set(event_ids)):
            raise _LocalTargetFailure("response_mapping", delivery_uncertain=True)
        return (runtime_event, *worker_events)

    async def _terminate(self) -> None:
        process = self._process
        stderr_task = self._stderr_task
        self._process = None
        self._stderr_task = None
        self._active_session_id = None
        self._runtime_report = None
        self._target_sha256 = None
        if process is not None and process.returncode is None:
            _signal_process(process, force=False)
            try:
                async with asyncio.timeout(self._config.limits.shutdown_timeout_seconds):
                    await process.wait()
            except TimeoutError:
                _signal_process(process, force=True)
                with suppress(TimeoutError):
                    async with asyncio.timeout(self._config.limits.shutdown_timeout_seconds):
                        await process.wait()
        if process is not None and process.stdin is not None:
            process.stdin.close()
        if stderr_task is not None:
            with suppress(asyncio.CancelledError, TimeoutError):
                async with asyncio.timeout(self._config.limits.shutdown_timeout_seconds):
                    await stderr_task


class LocalTargetConnection:
    def __init__(
        self,
        config: LocalTargetConfig,
        *,
        customer_code_execution_confirmed: bool,
        state_environment: StateEnvironment | None = None,
    ) -> None:
        if customer_code_execution_confirmed is not True:
            raise ValueError(
                "local target execution requires explicit customer-code trust confirmation"
            )
        validate_local_target_config(config)
        self._config = config
        self._invoker = _LocalTargetInvoker(config)
        state_identity = (
            require_state_adapter_identity(state_environment)
            if state_environment is not None
            else None
        )
        self._executor = ComposedEnvironmentExecutor(
            self._invoker,
            config_sha256=_composed_local_target_config_sha256(
                config,
                state_environment=state_environment,
            ),
            state_environment=state_environment,
            fixture_id=state_identity.fixture_id if state_identity is not None else None,
            outcome_projection=config.outcome,
        )
        self.capabilities: EnvironmentCapabilities = self._executor.capabilities

    @classmethod
    def from_config(
        cls,
        config: LocalTargetConfig,
        *,
        customer_code_execution_confirmed: bool,
        state_environment: StateEnvironment | None = None,
    ) -> Self:
        return cls(
            config,
            customer_code_execution_confirmed=customer_code_execution_confirmed,
            state_environment=state_environment,
        )

    @property
    def environment_id(self) -> str:
        return self._executor.environment_id

    @property
    def config_sha256(self) -> str:
        return self._executor.config_sha256

    @property
    def outcome_projection(self) -> OutcomeProjection | None:
        return self._executor.outcome_projection

    def api_calls_for_case(self, case: EvaluationCase) -> int:
        return self._executor.api_calls_for_case(case)

    async def execute(self, case: EvaluationCase) -> ExecutionEvidence:
        return await self._executor.execute(case)

    async def aclose(self) -> None:
        await self._invoker.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


def _target_command(config: LocalTargetConfig) -> tuple[str, ...]:
    if isinstance(config, CommandTargetConfig):
        return config.argv
    worker = Path(__file__).with_name("_local_worker.py").resolve()
    return (
        str(config.interpreter),
        "-u",
        str(worker),
        "--target",
        config.target,
        "--input-mode",
        config.input_mode,
    )


def _supervised_target_command(
    config: LocalTargetConfig,
    identity: _FileIdentity,
    *,
    platform: str,
) -> tuple[str, ...]:
    target_command = _target_command(config)
    if platform != "win32":
        return (str(identity.resolved_path), *target_command[1:])
    supervisor = Path(__file__).with_name("_windows_job_worker.py").resolve()
    return (
        sys.executable,
        "-u",
        str(supervisor),
        "--expected-executable-sha256",
        identity.sha256,
        "--",
        *target_command,
    )


class _FileTooLargeError(OSError):
    pass


class _ExecutableIdentityError(OSError):
    pass


@contextmanager
def _open_regular_file(
    path: Path,
    *,
    maximum_bytes: int,
    allow_final_symlink: bool,
) -> Generator[tuple[int, os.stat_result], None, None]:
    opened_path = path.resolve(strict=True) if allow_final_symlink else path
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    if not allow_final_symlink:
        flags |= getattr(os, "O_NOFOLLOW", 0)
    before = None if hasattr(os, "O_NOFOLLOW") else os.lstat(opened_path)
    if before is not None and stat.S_ISLNK(before.st_mode):
        raise OSError("symbolic links are not allowed")
    descriptor = os.open(opened_path, flags)
    try:
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise OSError("path is not a regular file")
        if descriptor_stat.st_size > maximum_bytes:
            raise _FileTooLargeError
        if before is not None:
            after = os.lstat(opened_path)
            if _stat_identity(after) != _stat_identity(descriptor_stat):
                raise OSError("file identity changed while opening")
        yield descriptor, descriptor_stat
    finally:
        os.close(descriptor)


def _read_bounded_descriptor(descriptor: int, maximum_bytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = maximum_bytes + 1
    while remaining > 0:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    encoded = b"".join(chunks)
    if len(encoded) > maximum_bytes:
        raise _FileTooLargeError
    return encoded


@contextmanager
def _open_executable_identity(
    path: Path,
) -> Generator[tuple[int, _FileIdentity], None, None]:
    resolved_path = path.resolve(strict=True)
    with _open_regular_file(
        resolved_path,
        maximum_bytes=_MAXIMUM_EXECUTABLE_BYTES,
        allow_final_symlink=False,
    ) as (descriptor, descriptor_stat):
        digest = _descriptor_sha256(descriptor)
        identity = _FileIdentity(
            resolved_path=resolved_path,
            device=descriptor_stat.st_dev,
            inode=descriptor_stat.st_ino,
            size=descriptor_stat.st_size,
            modified_nanoseconds=descriptor_stat.st_mtime_ns,
            sha256=digest,
        )
        if not _identity_still_matches_path(path, identity):
            raise _ExecutableIdentityError
        yield descriptor, identity


def _descriptor_sha256(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _identity_still_matches_path(path: Path, identity: _FileIdentity) -> bool:
    try:
        if path.resolve(strict=True) != identity.resolved_path:
            return False
        current = os.stat(identity.resolved_path, follow_symlinks=False)
    except OSError:
        return False
    return _stat_identity(current) == (
        identity.device,
        identity.inode,
        identity.size,
        identity.modified_nanoseconds,
    ) and stat.S_ISREG(current.st_mode)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _signal_process(process: asyncio.subprocess.Process, *, force: bool) -> None:
    if process.returncode is not None:
        return
    try:
        if sys.platform == "win32":
            process.kill() if force else process.terminate()
        else:
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def _parse_error(raw: object) -> _ErrorMessage | None:
    if not isinstance(raw, dict):
        return None
    typed_raw = cast(dict[str, object], raw)
    if typed_raw.get("type") != "error":
        return None
    try:
        return _ErrorMessage.model_validate(typed_raw)
    except ValidationError:
        raise _LocalTargetFailure("response_mapping", delivery_uncertain=True) from None


def _result_message(raw: dict[str, object]) -> _ResultMessage:
    execution_events = raw.get("execution_events")
    if not isinstance(execution_events, list):
        raise _LocalTargetFailure("response_mapping", delivery_uncertain=True)
    normalized: dict[str, object] = {
        **raw,
        "execution_events": tuple(cast(list[object], execution_events)),
    }
    return _ResultMessage.model_validate(normalized)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError
        value[key] = item
    return value


def _reject_nonstandard_constant(value: str) -> None:
    del value
    raise ValueError


def _reject_deep_json(value: object, *, depth: int = 0) -> None:
    if depth > _MAXIMUM_JSON_DEPTH:
        raise ValueError
    if isinstance(value, dict):
        for item in cast(dict[object, object], value).values():
            _reject_deep_json(item, depth=depth + 1)
    elif isinstance(value, list):
        for item in cast(list[object], value):
            _reject_deep_json(item, depth=depth + 1)
