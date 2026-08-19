from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import socket
import ssl
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Never, Self, cast
from urllib.parse import urlsplit

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)
from ul_core.evaluation import (
    EvaluationCase,
    ExecutionEvidence,
    SandboxCapabilities,
    SandboxLifecycleEvidence,
    SandboxLifecycleFailureCode,
    SandboxStateEvidence,
    SandboxTurnEvidence,
    TimeoutAfterCommitEventEvidence,
    TimeoutAfterCommitEventRequest,
    TimeoutAfterCommitTriggerStatus,
)

_HEADER_NAME_PATTERN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
_SANDBOX_ENVIRONMENT_VARIABLE_PATTERN = re.compile(r"UL_SANDBOX_[A-Z][A-Z0-9_]*")
_INPUT_PLACEHOLDER = "{{input}}"
_CASE_ID_PLACEHOLDER = "{{case_id}}"
_TURN_ID_PLACEHOLDER = "{{turn_id}}"
_SANDBOX_SETUP_PLACEHOLDER = "{{sandbox_setup}}"
_INITIAL_STATE_TURN_ID = "__ul_initial_state__"
_MAXIMUM_CONFIG_BYTES = 1_000_000
_MAXIMUM_HEADER_COUNT = 32
_MAXIMUM_HEADER_VALUE_BYTES = 8_192
_MAXIMUM_TOTAL_HEADER_BYTES = 32_768
_MAXIMUM_JSON_DEPTH = 100
_UNSAFE_HEADER_NAMES = {
    "accept-encoding",
    "connection",
    "content-length",
    "content-type",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class _SandboxProtocolError(RuntimeError):
    def __init__(
        self,
        code: SandboxLifecycleFailureCode,
        reason: str,
        *,
        delivery_uncertain: bool = False,
    ) -> None:
        super().__init__(reason)
        self.code: SandboxLifecycleFailureCode = code
        self.delivery_uncertain: bool = delivery_uncertain


class _TargetDeliveryUncertainError(_SandboxProtocolError):
    def __init__(self, code: SandboxLifecycleFailureCode, reason: str) -> None:
        super().__init__(code, reason, delivery_uncertain=True)


class _TargetNotDeliveredError(_SandboxProtocolError):
    pass


class _SandboxIdentityMismatchError(_SandboxProtocolError):
    def __init__(self, code: SandboxLifecycleFailureCode, reason: str) -> None:
        super().__init__(code, reason, delivery_uncertain=True)


class JsonHttpLifecycleCallConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    url: str
    request_json_template: JsonValue = Field(
        default_factory=lambda: {"case_id": _CASE_ID_PLACEHOLDER}
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, url: str) -> str:
        _validate_endpoint(url, allow_insecure_http=True)
        return url

    @field_validator("request_json_template", mode="before")
    @classmethod
    def validate_request_json_template(cls, request_json: object) -> JsonValue:
        return _validated_lifecycle_request_json_template(request_json)


class JsonHttpLifecycleObservationConfig(JsonHttpLifecycleCallConfig):
    request_json_template: JsonValue = Field(
        default_factory=lambda: {
            "case_id": _CASE_ID_PLACEHOLDER,
            "turn_id": _TURN_ID_PLACEHOLDER,
        }
    )
    response_json_pointer: str = ""
    sandbox_id_json_pointer: str = "/sandbox_id"
    case_id_json_pointer: str = "/case_id"
    turn_id_json_pointer: str = "/turn_id"

    @field_validator("request_json_template", mode="before")
    @classmethod
    def validate_request_json_template(cls, request_json: object) -> JsonValue:
        return _validated_turn_observation_request_json_template(request_json)

    @field_validator(
        "response_json_pointer",
        "sandbox_id_json_pointer",
        "case_id_json_pointer",
        "turn_id_json_pointer",
    )
    @classmethod
    def validate_response_json_pointer(cls, pointer: str) -> str:
        _parse_json_pointer(pointer)
        return pointer


class JsonHttpLifecycleMutationConfig(JsonHttpLifecycleCallConfig):
    sandbox_id_json_pointer: str = "/sandbox_id"
    case_id_json_pointer: str = "/case_id"

    @field_validator("request_json_template", mode="before")
    @classmethod
    def validate_request_json_template(cls, request_json: object) -> JsonValue:
        return _validated_setup_request_json_template(request_json)

    @field_validator("sandbox_id_json_pointer", "case_id_json_pointer")
    @classmethod
    def validate_sandbox_id_json_pointer(cls, pointer: str) -> str:
        _parse_json_pointer(pointer)
        return pointer


class JsonHttpLifecycleResetConfig(JsonHttpLifecycleCallConfig):
    sandbox_id_json_pointer: str = "/sandbox_id"
    case_id_json_pointer: str = "/case_id"
    generation_json_pointer: str
    clean_state_json_pointer: str
    clean_state_value: JsonValue

    @field_validator(
        "sandbox_id_json_pointer",
        "case_id_json_pointer",
        "generation_json_pointer",
        "clean_state_json_pointer",
    )
    @classmethod
    def validate_json_pointer(cls, pointer: str) -> str:
        _parse_json_pointer(pointer)
        return pointer

    @field_validator("clean_state_value")
    @classmethod
    def validate_clean_state_value(cls, value: JsonValue) -> JsonValue:
        if isinstance(value, dict | list | float):
            raise ValueError("clean_state_value must be a JSON string, integer, boolean, or null")
        return value

    @model_validator(mode="after")
    def validate_distinct_pointers(self) -> Self:
        pointers = {
            self.sandbox_id_json_pointer,
            self.case_id_json_pointer,
            self.generation_json_pointer,
            self.clean_state_json_pointer,
        }
        if len(pointers) != 4:
            raise ValueError("reset identity, generation, and clean-state pointers must differ")
        return self


class JsonHttpLifecycleExecuteTurnConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    url: str
    request_json_template: JsonValue
    response_json_pointer: str = ""
    sandbox_id_json_pointer: str = "/sandbox_id"
    case_id_json_pointer: str = "/case_id"
    turn_id_json_pointer: str = "/turn_id"

    @field_validator("url")
    @classmethod
    def validate_url(cls, url: str) -> str:
        _validate_endpoint(url, allow_insecure_http=True)
        return url

    @field_validator("request_json_template", mode="before")
    @classmethod
    def validate_request_json_template(cls, template: object) -> JsonValue:
        return _validated_execute_request_json_template(template)

    @field_validator(
        "response_json_pointer",
        "sandbox_id_json_pointer",
        "case_id_json_pointer",
        "turn_id_json_pointer",
    )
    @classmethod
    def validate_response_json_pointer(cls, pointer: str) -> str:
        _parse_json_pointer(pointer)
        return pointer


class JsonHttpTimeoutAfterCommitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    operator_id: Literal["environment.tool.timeout_after_commit"] = (
        "environment.tool.timeout_after_commit"
    )
    version: Literal["1.0.0"] = "1.0.0"
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, url: str) -> str:
        _validate_endpoint(url, allow_insecure_http=True)
        return url


class _TimeoutAfterCommitControlResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sandbox_id: str = Field(min_length=1, max_length=500)
    case_id: str = Field(min_length=1, max_length=500)
    operator_id: Literal["environment.tool.timeout_after_commit"]
    operator_version: Literal["1.0.0"]
    event_id: str = Field(min_length=1, max_length=500)
    turn_id: str = Field(min_length=1, max_length=500)
    action_id: str = Field(min_length=1, max_length=500)
    operation: Literal["arm", "observe", "clean"]
    status: Literal["armed", "fired", "not_fired", "cleaned"]

    @model_validator(mode="after")
    def validate_operation_status(self) -> Self:
        valid_statuses = {
            "arm": {"armed"},
            "observe": {"fired", "not_fired"},
            "clean": {"cleaned"},
        }
        if self.status not in valid_statuses[self.operation]:
            raise ValueError("timeout-after-commit status does not match its operation")
        return self


class JsonHttpSandboxConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[3]
    sandbox_id: str = Field(min_length=1, max_length=500)
    headers_from_env: dict[str, str] = Field(default_factory=dict)
    reset: JsonHttpLifecycleResetConfig
    setup: JsonHttpLifecycleMutationConfig | None = None
    execute_turn: JsonHttpLifecycleExecuteTurnConfig
    snapshot: JsonHttpLifecycleObservationConfig
    timeout_after_commit: JsonHttpTimeoutAfterCommitConfig | None = None

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, version: object) -> object:
        if type(version) is not int or version != 3:
            raise ValueError("version must be 3")
        return version

    @field_validator("headers_from_env")
    @classmethod
    def validate_headers_from_env(cls, headers: dict[str, str]) -> dict[str, str]:
        return _validate_header_environment_variables(headers)

    @model_validator(mode="after")
    def validate_same_origin(self) -> Self:
        origins = {_endpoint_origin(url) for url in json_http_sandbox_config_urls(self)}
        if len(origins) != 1:
            raise ValueError("all lifecycle endpoints must use the same origin")
        return self


def load_json_http_sandbox_config(path: str | Path) -> JsonHttpSandboxConfig:
    try:
        encoded_config = _read_bounded_regular_file(Path(path), maximum_bytes=_MAXIMUM_CONFIG_BYTES)
    except OSError:
        raise RuntimeError("sandbox API config could not be read") from None
    if len(encoded_config) > _MAXIMUM_CONFIG_BYTES:
        raise ValueError("sandbox API config exceeds the size limit")
    try:
        decoded_config = encoded_config.decode("utf-8")
        raw_config = json.loads(
            decoded_config,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
        _reject_deep_json(raw_config)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise ValueError("sandbox API config contains invalid JSON") from None
    try:
        return JsonHttpSandboxConfig.model_validate(raw_config)
    except RecursionError:
        raise ValueError("sandbox API config is invalid") from None
    except ValidationError as error:
        validation_reasons: list[str] = []
        for issue in error.errors(include_url=False, include_context=False, include_input=False):
            field_path = ".".join(str(part) for part in issue["loc"])
            message = str(issue["msg"]).removeprefix("Value error, ")
            validation_reasons.append(f"{field_path}: {message}")
        raise ValueError(
            f"sandbox API config is invalid: {'; '.join(validation_reasons)}"
        ) from None


class JsonHttpSandboxConnection:
    def __init__(
        self,
        config: JsonHttpSandboxConfig,
        *,
        sandbox_confirmed: bool,
        allow_insecure_http: bool = False,
        timeout_seconds: float = 30,
        max_request_bytes: int = 1_000_000,
        max_response_bytes: int = 1_000_000,
        max_sandbox_api_calls: int | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if max_sandbox_api_calls is not None and (
            isinstance(max_sandbox_api_calls, bool) or max_sandbox_api_calls <= 0
        ):
            raise ValueError("max_sandbox_api_calls must be positive")
        self._headers = validate_json_http_sandbox_configuration(
            config,
            sandbox_confirmed=sandbox_confirmed,
            allow_insecure_http=allow_insecure_http,
            timeout_seconds=timeout_seconds,
            max_request_bytes=max_request_bytes,
            max_response_bytes=max_response_bytes,
        )
        self._timeout_seconds = timeout_seconds
        self._max_request_bytes = max_request_bytes
        self._max_response_bytes = max_response_bytes
        self._remaining_sandbox_api_calls = max_sandbox_api_calls
        self._config = config
        self._config_sha256 = json_http_sandbox_config_sha256(config)
        self._lifecycle_lock = asyncio.Lock()
        self._last_reset_generation: str | int | None = None
        self._lifecycle_state_uncertain = False
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(follow_redirects=False, trust_env=False)
        self.capabilities = SandboxCapabilities(
            supports_conversations=True,
            supports_state_observation=True,
            state_observation_authority="sandbox_self_reported",
            cancellation_guarantee="best_effort",
            timeout_after_commit_version=(
                config.timeout_after_commit.version
                if config.timeout_after_commit is not None
                else None
            ),
        )

    @property
    def sandbox_id(self) -> str:
        return self._config.sandbox_id

    @property
    def config_sha256(self) -> str:
        return self._config_sha256

    @classmethod
    def from_config(
        cls,
        config: JsonHttpSandboxConfig,
        *,
        sandbox_confirmed: bool,
        allow_insecure_http: bool = False,
        timeout_seconds: float = 30,
        max_request_bytes: int = 1_000_000,
        max_response_bytes: int = 1_000_000,
        max_sandbox_api_calls: int | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> JsonHttpSandboxConnection:
        return cls(
            config,
            sandbox_confirmed=sandbox_confirmed,
            allow_insecure_http=allow_insecure_http,
            timeout_seconds=timeout_seconds,
            max_request_bytes=max_request_bytes,
            max_response_bytes=max_response_bytes,
            max_sandbox_api_calls=max_sandbox_api_calls,
            client=client,
        )

    async def __aenter__(self) -> JsonHttpSandboxConnection:
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def api_calls_for_case(self, case: EvaluationCase) -> int:
        _validate_case_sandbox_setup(self._config, case)
        calls = json_http_sandbox_calls_per_conversation(self._config, len(case.turns))
        if case.timeout_after_commit_event is not None:
            if self._config.timeout_after_commit is None:
                raise ValueError(
                    "sandbox does not support environment.tool.timeout_after_commit@1.0.0"
                )
            calls += 3
        return calls

    async def execute(self, case: EvaluationCase) -> ExecutionEvidence:
        required_calls = self.api_calls_for_case(case)
        if required_calls > case.max_sandbox_api_calls:
            raise ValueError("evaluation case exceeds its sandbox API call budget")
        async with self._lifecycle_lock:
            async with asyncio.timeout(case.timeout_seconds):
                return await self._execute_stateful(case)

    async def _execute_stateful(self, case: EvaluationCase) -> ExecutionEvidence:
        config = self._config
        event_request = case.timeout_after_commit_event
        event_config = config.timeout_after_commit
        if event_request is not None and event_config is None:
            raise ValueError("sandbox does not support environment.tool.timeout_after_commit@1.0.0")
        if self._lifecycle_state_uncertain:
            return self._execution_evidence(
                case,
                (),
                initial_state=None,
                terminal_status="failed",
                completed_phases=(),
                failed_phase="blocked_state_uncertain",
                failure_code="sandbox_state_uncertain",
                failure_reason="sandbox state is uncertain after an earlier lifecycle failure",
                delivery="uncertain",
                cleanup="not_attempted",
                cleanup_failure_code=None,
                cleanup_failure_reason=None,
                sandbox_state_uncertain=True,
            )
        self._reserve_sandbox_api_calls(self.api_calls_for_case(case))
        self._lifecycle_state_uncertain = True
        completed_phase_names: list[str] = []
        lifecycle_started = False
        failed_phase: str | None = None
        failure_code: SandboxLifecycleFailureCode | None = None
        failure_reason: str | None = None
        cleanup_attempted = False
        cleanup_reset_failed = False
        event_cleanup_failed = False
        cleanup_failure_code: SandboxLifecycleFailureCode | None = None
        cleanup_failure_reason: str | None = None
        delivery_uncertain = False
        reset_not_delivered = False
        current_phase = "reset"
        turn_observations: list[SandboxTurnEvidence] = []
        initial_state: SandboxStateEvidence | None = None
        event_arm_attempted = False
        event_armed = False
        event_trigger_status: TimeoutAfterCommitTriggerStatus = "unknown"
        event_cleaned = False
        cleanup_cancellation: asyncio.CancelledError | None = None
        try:
            lifecycle_started = True
            await self._reset(config.reset, case.id)
            completed_phase_names.append("reset")
            if config.setup is not None:
                current_phase = "setup"
                setup_response = await self._post_for_json(
                    config.setup.url,
                    _replace_request_placeholders(
                        config.setup.request_json_template,
                        case_id=case.id,
                        sandbox_setup=(
                            case.sandbox_setup.payload if case.sandbox_setup is not None else None
                        ),
                    ),
                    "",
                    consume_budget=False,
                )
                self._validate_response_identity(
                    setup_response,
                    sandbox_id_pointer=config.setup.sandbox_id_json_pointer,
                    case_id_pointer=config.setup.case_id_json_pointer,
                    case_id=case.id,
                )
                completed_phase_names.append("setup")
            current_phase = "initial_snapshot"
            initial_state = SandboxStateEvidence(
                value=await self._snapshot(
                    config.snapshot,
                    case.id,
                    _INITIAL_STATE_TURN_ID,
                    allow_null=True,
                ),
                authority="sandbox_self_reported",
            )
            completed_phase_names.append(current_phase)
            for turn_index, turn in enumerate(case.turns, start=1):
                if event_request is not None and event_request.turn_id == turn.id:
                    if event_config is None:
                        raise AssertionError("event capability was validated before execution")
                    current_phase = "arm_timeout_after_commit"
                    event_arm_attempted = True
                    await self._timeout_after_commit_control(
                        event_config,
                        case.id,
                        event_request,
                        operation="arm",
                    )
                    event_armed = True
                    completed_phase_names.append(current_phase)
                current_phase = _conversation_phase_name(
                    "execute_turn", turn_index, len(case.turns)
                )
                execute_payload = await self._post_for_json(
                    config.execute_turn.url,
                    _replace_request_placeholders(
                        config.execute_turn.request_json_template,
                        case_id=case.id,
                        turn_id=turn.id,
                        raw_input=turn.content,
                    ),
                    "",
                    consume_budget=False,
                )
                self._validate_response_identity(
                    execute_payload,
                    sandbox_id_pointer=config.execute_turn.sandbox_id_json_pointer,
                    case_id_pointer=config.execute_turn.case_id_json_pointer,
                    case_id=case.id,
                    turn_id_pointer=config.execute_turn.turn_id_json_pointer,
                    turn_id=turn.id,
                )
                execute_response = _resolve_json_pointer(
                    execute_payload,
                    config.execute_turn.response_json_pointer,
                )
                completed_phase_names.append(current_phase)
                current_phase = _conversation_phase_name("snapshot", turn_index, len(case.turns))
                state_snapshot = await self._snapshot(config.snapshot, case.id, turn.id)
                completed_phase_names.append(current_phase)
                turn_observations.append(
                    SandboxTurnEvidence(
                        turn_id=turn.id,
                        response=execute_response,
                        state_snapshot=state_snapshot,
                        state_observation_authority="sandbox_self_reported",
                    )
                )
                if event_request is not None and event_request.turn_id == turn.id:
                    if event_config is None:
                        raise AssertionError("event capability was validated before execution")
                    current_phase = "observe_timeout_after_commit"
                    observed_status = await self._timeout_after_commit_control(
                        event_config,
                        case.id,
                        event_request,
                        operation="observe",
                    )
                    if observed_status not in {"fired", "not_fired"}:
                        raise AssertionError("observe operation returned an invalid status")
                    event_trigger_status = cast(TimeoutAfterCommitTriggerStatus, observed_status)
                    completed_phase_names.append(current_phase)
        except _SandboxProtocolError as error:
            failed_phase = current_phase
            failure_code = error.code
            failure_reason = str(error)
            delivery_uncertain = error.delivery_uncertain
            reset_not_delivered = current_phase == "reset" and isinstance(
                error, _TargetNotDeliveredError
            )
        except asyncio.CancelledError:
            failed_phase = current_phase
            delivery_uncertain = True
            raise
        except Exception:
            failed_phase = current_phase
            failure_code = "sandbox_lifecycle_error"
            failure_reason = "sandbox lifecycle failed"
            delivery_uncertain = True
        finally:
            if lifecycle_started and failed_phase != "reset":
                cleanup_attempted = True
                try:
                    if event_arm_attempted:
                        if event_config is None or event_request is None:
                            raise AssertionError("event cleanup requires an event configuration")
                        try:
                            await self._timeout_after_commit_control(
                                event_config,
                                case.id,
                                event_request,
                                operation="clean",
                            )
                            event_cleaned = True
                            completed_phase_names.append("cleanup_timeout_after_commit")
                        except asyncio.CancelledError as error:
                            event_cleanup_failed = True
                            self._lifecycle_state_uncertain = True
                            cleanup_cancellation = error
                        except _SandboxProtocolError as error:
                            event_cleanup_failed = True
                            cleanup_failure_code = error.code
                            cleanup_failure_reason = str(error)
                            delivery_uncertain = delivery_uncertain or error.delivery_uncertain
                        except Exception:
                            event_cleanup_failed = True
                            cleanup_failure_code = "sandbox_cleanup_error"
                            cleanup_failure_reason = "sandbox event cleanup failed"
                            delivery_uncertain = True
                finally:
                    try:
                        await self._reset(config.reset, case.id)
                        completed_phase_names.append("cleanup_reset")
                        if not delivery_uncertain and not event_cleanup_failed:
                            self._lifecycle_state_uncertain = False
                    except asyncio.CancelledError as error:
                        cleanup_reset_failed = True
                        self._lifecycle_state_uncertain = True
                        cleanup_cancellation = cleanup_cancellation or error
                    except _SandboxProtocolError as error:
                        cleanup_reset_failed = True
                        cleanup_failure_code = cleanup_failure_code or error.code
                        cleanup_failure_reason = cleanup_failure_reason or str(error)
                        delivery_uncertain = delivery_uncertain or error.delivery_uncertain
                    except Exception:
                        cleanup_reset_failed = True
                        cleanup_failure_code = cleanup_failure_code or "sandbox_cleanup_error"
                        cleanup_failure_reason = cleanup_failure_reason or "sandbox cleanup failed"
                        delivery_uncertain = True
            if cleanup_cancellation is not None:
                raise cleanup_cancellation
        if reset_not_delivered:
            self._lifecycle_state_uncertain = False
        if failed_phase is not None or cleanup_reset_failed or event_cleanup_failed:
            return self._execution_evidence(
                case,
                tuple(turn_observations),
                initial_state=initial_state,
                terminal_status="failed",
                completed_phases=tuple(completed_phase_names),
                failed_phase=(
                    failed_phase
                    or ("cleanup_timeout_after_commit" if event_cleanup_failed else "cleanup_reset")
                ),
                failure_code=failure_code or cleanup_failure_code,
                failure_reason=failure_reason or cleanup_failure_reason,
                delivery="uncertain" if delivery_uncertain else "certain",
                cleanup=(
                    "failed"
                    if cleanup_reset_failed or event_cleanup_failed
                    else "succeeded"
                    if cleanup_attempted
                    else "not_attempted"
                ),
                cleanup_failure_code=cleanup_failure_code,
                cleanup_failure_reason=cleanup_failure_reason,
                sandbox_state_uncertain=self._lifecycle_state_uncertain,
                event_armed=event_armed,
                event_trigger_status=event_trigger_status,
                event_cleaned=event_cleaned,
            )
        if len(turn_observations) != len(case.turns):
            raise AssertionError(
                "successful lifecycle requires every turn and snapshot observation"
            )
        return self._execution_evidence(
            case,
            tuple(turn_observations),
            initial_state=initial_state,
            terminal_status="succeeded",
            completed_phases=tuple(completed_phase_names),
            failed_phase=None,
            failure_code=None,
            failure_reason=None,
            delivery="certain",
            cleanup="succeeded",
            cleanup_failure_code=None,
            cleanup_failure_reason=None,
            sandbox_state_uncertain=False,
            event_armed=event_armed,
            event_trigger_status=event_trigger_status,
            event_cleaned=event_cleaned,
        )

    async def _timeout_after_commit_control(
        self,
        config: JsonHttpTimeoutAfterCommitConfig,
        case_id: str,
        event: TimeoutAfterCommitEventRequest,
        *,
        operation: Literal["arm", "observe", "clean"],
    ) -> Literal["armed", "fired", "not_fired", "cleaned"]:
        payload = await self._post_for_json(
            config.url,
            {
                "sandbox_id": self._config.sandbox_id,
                "case_id": case_id,
                "operator_id": event.operator_id,
                "operator_version": event.operator_version,
                "event_id": event.event_id,
                "turn_id": event.turn_id,
                "action_id": event.action_id,
                "operation": operation,
            },
            "",
            consume_budget=False,
        )
        try:
            response = _TimeoutAfterCommitControlResponse.model_validate(payload)
        except ValidationError:
            self._lifecycle_state_uncertain = True
            raise _SandboxIdentityMismatchError(
                "response_mapping",
                "timeout-after-commit control response is invalid",
            ) from None
        expected_identity = (
            self._config.sandbox_id,
            case_id,
            event.operator_id,
            event.operator_version,
            event.event_id,
            event.turn_id,
            event.action_id,
            operation,
        )
        response_identity = (
            response.sandbox_id,
            response.case_id,
            response.operator_id,
            response.operator_version,
            response.event_id,
            response.turn_id,
            response.action_id,
            response.operation,
        )
        if response_identity != expected_identity:
            self._lifecycle_state_uncertain = True
            raise _SandboxIdentityMismatchError(
                "sandbox_identity",
                "timeout-after-commit control response did not match its request",
            )
        return response.status

    async def _reset(self, config: JsonHttpLifecycleResetConfig, case_id: str) -> JsonValue:
        reset_response = await self._post_for_json(
            config.url,
            _replace_request_placeholders(config.request_json_template, case_id=case_id),
            "",
            consume_budget=False,
        )
        self._validate_response_identity(
            reset_response,
            sandbox_id_pointer=config.sandbox_id_json_pointer,
            case_id_pointer=config.case_id_json_pointer,
            case_id=case_id,
        )
        generation = _resolve_json_pointer(
            reset_response,
            config.generation_json_pointer,
        )
        if isinstance(generation, bool) or not isinstance(generation, str | int):
            raise _SandboxProtocolError(
                "reset_generation", "sandbox API reset generation is invalid"
            )
        if isinstance(generation, str) and not generation:
            raise _SandboxProtocolError(
                "reset_generation", "sandbox API reset generation is invalid"
            )
        if generation == self._last_reset_generation:
            raise _SandboxProtocolError(
                "reset_generation_reused",
                "sandbox API reset generation did not change",
                delivery_uncertain=True,
            )
        self._last_reset_generation = generation
        clean_state = _resolve_json_pointer(
            reset_response,
            config.clean_state_json_pointer,
            allow_null=True,
        )
        if (
            type(clean_state) is not type(config.clean_state_value)
            or clean_state != config.clean_state_value
        ):
            raise _SandboxProtocolError(
                "reset_not_clean",
                "sandbox API reset did not report clean state",
                delivery_uncertain=True,
            )
        return clean_state

    async def _snapshot(
        self,
        config: JsonHttpLifecycleObservationConfig,
        case_id: str,
        turn_id: str,
        *,
        allow_null: bool = False,
    ) -> JsonValue:
        response = await self._post_for_json(
            config.url,
            _replace_request_placeholders(
                config.request_json_template, case_id=case_id, turn_id=turn_id
            ),
            "",
            consume_budget=False,
        )
        self._validate_response_identity(
            response,
            sandbox_id_pointer=config.sandbox_id_json_pointer,
            case_id_pointer=config.case_id_json_pointer,
            case_id=case_id,
            turn_id_pointer=config.turn_id_json_pointer,
            turn_id=turn_id,
        )
        return _resolve_json_pointer(
            response,
            config.response_json_pointer,
            allow_null=allow_null,
        )

    def _validate_response_identity(
        self,
        response: JsonValue,
        *,
        sandbox_id_pointer: str,
        case_id_pointer: str,
        case_id: str,
        turn_id_pointer: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        sandbox_id = _resolve_json_pointer(response, sandbox_id_pointer)
        if sandbox_id != self._config.sandbox_id:
            self._lifecycle_state_uncertain = True
            raise _SandboxIdentityMismatchError(
                "sandbox_identity", "HTTP sandbox identity did not match its configuration"
            )
        if _resolve_json_pointer(response, case_id_pointer) != case_id:
            self._lifecycle_state_uncertain = True
            raise _SandboxIdentityMismatchError(
                "case_identity", "HTTP sandbox response did not match its case"
            )
        if turn_id_pointer is not None and (
            turn_id is None or _resolve_json_pointer(response, turn_id_pointer) != turn_id
        ):
            self._lifecycle_state_uncertain = True
            raise _SandboxIdentityMismatchError(
                "turn_identity", "HTTP sandbox response did not match its turn"
            )

    def _execution_evidence(
        self,
        case: EvaluationCase,
        turns: tuple[SandboxTurnEvidence, ...],
        *,
        initial_state: SandboxStateEvidence | None,
        terminal_status: Literal["succeeded", "failed", "timed_out", "cancelled"],
        completed_phases: tuple[str, ...],
        failed_phase: str | None,
        failure_code: SandboxLifecycleFailureCode | None,
        failure_reason: str | None,
        delivery: Literal["certain", "uncertain"],
        cleanup: Literal["succeeded", "failed", "not_attempted"],
        cleanup_failure_code: SandboxLifecycleFailureCode | None,
        cleanup_failure_reason: str | None,
        sandbox_state_uncertain: bool,
        event_armed: bool = False,
        event_trigger_status: TimeoutAfterCommitTriggerStatus = "unknown",
        event_cleaned: bool = False,
    ) -> ExecutionEvidence:
        event_request = case.timeout_after_commit_event
        return ExecutionEvidence(
            case_id=case.id,
            sandbox_id=self._config.sandbox_id,
            sandbox_config_sha256=self._config_sha256,
            sandbox_setup_sha256=(
                case.sandbox_setup.sha256 if case.sandbox_setup is not None else None
            ),
            initial_state=initial_state,
            turns=turns,
            final_response=turns[-1].response if turns else None,
            final_state=(
                SandboxStateEvidence(
                    value=turns[-1].state_snapshot,
                    authority=turns[-1].state_observation_authority,
                    observer_id=turns[-1].state_observer_id,
                )
                if turns
                and turns[-1].state_snapshot is not None
                and turns[-1].state_observation_authority is not None
                else None
            ),
            timeout_after_commit_event=(
                TimeoutAfterCommitEventEvidence(
                    **event_request.model_dump(),
                    armed=event_armed,
                    trigger_status=event_trigger_status,
                    cleaned=event_cleaned,
                )
                if event_request is not None
                else None
            ),
            lifecycle=SandboxLifecycleEvidence(
                terminal_status=terminal_status,
                completed_phases=completed_phases,
                failed_phase=failed_phase,
                failure_code=failure_code,
                failure_reason=failure_reason,
                delivery=delivery,
                cleanup=cleanup,
                cleanup_failure_code=cleanup_failure_code,
                cleanup_failure_reason=cleanup_failure_reason,
                sandbox_state_uncertain=sandbox_state_uncertain,
            ),
        )

    async def _post_without_observation(
        self,
        endpoint: str,
        request_json: JsonValue,
        *,
        consume_budget: bool = True,
    ) -> None:
        await self._post(
            endpoint,
            request_json,
            response_json_pointer=None,
            consume_budget=consume_budget,
        )

    async def _post_for_json(
        self,
        endpoint: str,
        request_json: JsonValue,
        response_json_pointer: str,
        *,
        consume_budget: bool = True,
    ) -> JsonValue:
        result = await self._post(
            endpoint,
            request_json,
            response_json_pointer=response_json_pointer,
            consume_budget=consume_budget,
        )
        if result is None:
            raise AssertionError("JSON observation request requires a result")
        return result

    async def _post(
        self,
        endpoint: str,
        request_json: JsonValue,
        *,
        response_json_pointer: str | None,
        consume_budget: bool = True,
    ) -> JsonValue | None:
        request_body = json.dumps(
            request_json,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(request_body) > self._max_request_bytes:
            raise _TargetNotDeliveredError(
                "request_too_large", "sandbox API request exceeds the size limit"
            )
        if consume_budget:
            self._reserve_sandbox_api_calls(1)
        try:
            async with asyncio.timeout(self._timeout_seconds):
                async with self._client.stream(
                    "POST",
                    endpoint,
                    headers={
                        "Accept-Encoding": "identity",
                        "Content-Type": "application/json",
                        **self._headers,
                    },
                    content=request_body,
                    follow_redirects=False,
                    timeout=self._timeout_seconds,
                ) as response:
                    if not 200 <= response.status_code < 300:
                        raise _SandboxProtocolError(
                            _http_status_failure_code(response.status_code),
                            f"sandbox API returned HTTP {response.status_code}",
                        )
                    if response_json_pointer is not None and not _is_json_content_type(
                        response.headers.get("content-type")
                    ):
                        raise _SandboxProtocolError(
                            "response_content_type",
                            "sandbox API response must be JSON",
                        )
                    content_encoding = response.headers.get("content-encoding")
                    if content_encoding is not None and content_encoding.casefold() != "identity":
                        raise _SandboxProtocolError(
                            "response_content_encoding",
                            "sandbox API response must not use content encoding",
                        )
                    response_body = await _read_bounded_response(response, self._max_response_bytes)
        except TimeoutError:
            raise _TargetDeliveryUncertainError(
                "request_timeout", "sandbox API request timed out"
            ) from None
        except httpx.ConnectTimeout:
            raise _TargetNotDeliveredError(
                "connect_timeout", "sandbox API connection timed out"
            ) from None
        except httpx.ConnectError as error:
            if _exception_chain_contains(error, ssl.SSLError):
                reason = "sandbox API TLS connection failed"
            elif _exception_chain_contains(error, socket.gaierror):
                reason = "sandbox API DNS resolution failed"
            else:
                reason = "sandbox API connection failed"
            code = (
                "tls_connection"
                if "TLS" in reason
                else "dns_resolution"
                if "DNS" in reason
                else "connect_failed"
            )
            raise _TargetNotDeliveredError(code, reason) from None
        except httpx.ReadTimeout:
            raise _TargetDeliveryUncertainError(
                "response_timeout", "sandbox API response timed out"
            ) from None
        except httpx.WriteTimeout:
            raise _TargetDeliveryUncertainError(
                "write_timeout", "sandbox API request write timed out"
            ) from None
        except httpx.PoolTimeout:
            raise _TargetNotDeliveredError(
                "pool_timeout", "sandbox API connection pool timed out"
            ) from None
        except httpx.RemoteProtocolError:
            raise _TargetDeliveryUncertainError(
                "transport_protocol", "sandbox API transport protocol failed"
            ) from None
        except RuntimeError:
            raise
        except httpx.HTTPError:
            raise _TargetDeliveryUncertainError(
                "transport_failed", "sandbox API transport failed"
            ) from None

        if response_json_pointer is None:
            return None
        try:
            raw_output = json.loads(
                response_body,
                object_pairs_hook=_reject_duplicate_object_keys,
                parse_constant=_reject_nonstandard_json_constant,
            )
            _reject_deep_json(raw_output)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
            raise _SandboxProtocolError(
                "invalid_json",
                "sandbox API returned invalid JSON",
            ) from None
        if raw_output is None:
            raise _SandboxProtocolError(
                "null_json",
                "sandbox API returned null JSON",
            )
        return _resolve_json_pointer(raw_output, response_json_pointer)

    def _reserve_sandbox_api_calls(self, sandbox_api_calls: int) -> None:
        if self._remaining_sandbox_api_calls is None:
            return
        if sandbox_api_calls > self._remaining_sandbox_api_calls:
            raise _SandboxProtocolError("call_budget", "HTTP sandbox API call budget exhausted")
        self._remaining_sandbox_api_calls -= sandbox_api_calls


def validate_json_http_sandbox_configuration(
    config: JsonHttpSandboxConfig,
    *,
    sandbox_confirmed: bool,
    allow_insecure_http: bool = False,
    timeout_seconds: float = 30,
    max_request_bytes: int = 1_000_000,
    max_response_bytes: int = 1_000_000,
) -> dict[str, str]:
    if sandbox_confirmed is not True:
        raise ValueError("sandbox API access requires explicit isolation attestation")
    for endpoint in json_http_sandbox_config_urls(config):
        _validate_endpoint(endpoint, allow_insecure_http)
    if (
        isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be positive and finite")
    if isinstance(max_request_bytes, bool) or max_request_bytes <= 0:
        raise ValueError("max_request_bytes must be positive")
    if isinstance(max_response_bytes, bool) or max_response_bytes <= 0:
        raise ValueError("max_response_bytes must be positive")
    return _headers_from_environment(config.headers_from_env)


def json_http_sandbox_calls_per_execution(
    config: JsonHttpSandboxConfig,
) -> int:
    return json_http_sandbox_calls_per_conversation(config, 1)


def json_http_sandbox_uses_per_record_setup(config: JsonHttpSandboxConfig) -> bool:
    return (
        config.setup is not None
        and _count_placeholder(config.setup.request_json_template, _SANDBOX_SETUP_PLACEHOLDER) == 1
    )


def json_http_sandbox_config_sha256(config: JsonHttpSandboxConfig) -> str:
    canonical_config = json.dumps(
        config.model_dump(mode="json", exclude_none=True),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical_config).hexdigest()


def json_http_sandbox_calls_per_conversation(
    config: JsonHttpSandboxConfig,
    turn_count: int,
) -> int:
    if type(turn_count) is not int or turn_count < 1:
        raise ValueError("turn_count must be a positive integer")
    return 3 + (1 if config.setup is not None else 0) + (2 * turn_count)


def _conversation_phase_name(phase: str, turn_index: int, turn_count: int) -> str:
    return phase if turn_count == 1 else f"{phase}:{turn_index}"


def json_http_sandbox_config_urls(
    config: JsonHttpSandboxConfig,
) -> tuple[str, ...]:
    return (
        config.reset.url,
        *((config.setup.url,) if config.setup is not None else ()),
        config.execute_turn.url,
        config.snapshot.url,
        *((config.timeout_after_commit.url,) if config.timeout_after_commit is not None else ()),
    )


def _endpoint_origin(endpoint: str) -> tuple[str, str, int]:
    parsed = urlsplit(endpoint)
    hostname = parsed.hostname
    if hostname is None:
        raise ValueError("endpoint must be a valid HTTP(S) URL")
    default_port = 443 if parsed.scheme == "https" else 80
    return parsed.scheme, hostname.casefold(), parsed.port or default_port


def _validate_endpoint(endpoint: str, allow_insecure_http: bool) -> str:
    if (
        not endpoint
        or endpoint != endpoint.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in endpoint)
    ):
        raise ValueError("endpoint must be a non-empty HTTP(S) URL")
    try:
        parsed_endpoint = urlsplit(endpoint)
        _ = parsed_endpoint.port
    except ValueError:
        raise ValueError("endpoint must be a valid HTTP(S) URL") from None
    if parsed_endpoint.scheme not in {"http", "https"} or parsed_endpoint.hostname is None:
        raise ValueError("endpoint must be a valid HTTP(S) URL")
    if parsed_endpoint.username is not None or parsed_endpoint.password is not None:
        raise ValueError("endpoint must not contain credentials")
    try:
        httpx.URL(endpoint)
    except (httpx.InvalidURL, UnicodeError):
        raise ValueError("endpoint must be a valid HTTP(S) URL") from None
    if "?" in endpoint or "#" in endpoint:
        raise ValueError("endpoint must not contain a query or fragment")
    if parsed_endpoint.scheme == "http" and not allow_insecure_http:
        raise ValueError("HTTP endpoints require explicit insecure transport opt-in")
    return endpoint


def _validate_header_environment_variables(
    header_environment_variables: Mapping[str, str],
) -> dict[str, str]:
    if len(header_environment_variables) > _MAXIMUM_HEADER_COUNT:
        raise ValueError("header_environment_variables contains too many headers")
    validated: dict[str, str] = {}
    normalized_names: set[str] = set()
    for header_name, environment_variable in header_environment_variables.items():
        normalized_name = header_name.casefold()
        if (
            _HEADER_NAME_PATTERN.fullmatch(header_name) is None
            or normalized_name in _UNSAFE_HEADER_NAMES
            or normalized_name in normalized_names
        ):
            raise ValueError("header_environment_variables contains an unsafe header name")
        if _SANDBOX_ENVIRONMENT_VARIABLE_PATTERN.fullmatch(environment_variable) is None:
            raise ValueError("header environment variable names must use the UL_SANDBOX_ namespace")
        normalized_names.add(normalized_name)
        validated[header_name] = environment_variable
    return validated


def _headers_from_environment(header_environment_variables: Mapping[str, str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    total_header_bytes = 0
    for header_name, environment_variable in header_environment_variables.items():
        value = os.environ.get(environment_variable)
        if value is None or not value.strip():
            raise RuntimeError("sandbox API header environment variable is not set")
        try:
            encoded_value = value.encode("ascii")
        except UnicodeEncodeError:
            raise RuntimeError("sandbox API header environment variable is invalid") from None
        if any(byte < 32 or byte == 127 for byte in encoded_value):
            raise RuntimeError("sandbox API header environment variable is invalid")
        encoded_value_bytes = len(encoded_value)
        if encoded_value_bytes > _MAXIMUM_HEADER_VALUE_BYTES:
            raise RuntimeError("sandbox API header environment variable is too large")
        total_header_bytes += len(header_name.encode("ascii")) + encoded_value_bytes
        if total_header_bytes > _MAXIMUM_TOTAL_HEADER_BYTES:
            raise RuntimeError("sandbox API headers exceed the size limit")
        headers[header_name] = value
    return headers


def _is_json_content_type(content_type: str | None) -> bool:
    if content_type is None:
        return False
    media_type = content_type.partition(";")[0].strip().casefold()
    return media_type == "application/json" or media_type.endswith("+json")


def _http_status_failure_code(status_code: int) -> SandboxLifecycleFailureCode:
    if status_code in {401, 403}:
        return "authentication_rejected"
    if status_code == 429:
        return "rate_limited"
    return "http_status"


def _reject_nonstandard_json_constant(value: str) -> Never:
    raise ValueError("nonstandard JSON constant")


def _exception_chain_contains(error: BaseException, exception_type: type[BaseException]) -> bool:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        if isinstance(current, exception_type):
            return True
        visited.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    object_value: dict[str, Any] = {}
    for key, value in pairs:
        if key in object_value:
            raise ValueError("duplicate JSON object key")
        object_value[key] = value
    return object_value


def _reject_deep_json(value: object) -> None:
    values_to_visit: list[tuple[object, int]] = [(value, 0)]
    while values_to_visit:
        current, depth = values_to_visit.pop()
        if depth > _MAXIMUM_JSON_DEPTH:
            raise ValueError("JSON nesting limit exceeded")
        if isinstance(current, list):
            values_to_visit.extend((item, depth + 1) for item in cast(list[object], current))
        elif isinstance(current, dict):
            values_to_visit.extend(
                (item, depth + 1) for item in cast(dict[object, object], current).values()
            )


def _read_bounded_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    requires_identity_check = no_follow_flag == 0
    if requires_identity_check and stat.S_ISLNK(os.lstat(path).st_mode):
        raise OSError("path is a symbolic link")
    binary_flag = os.O_BINARY if sys.platform == "win32" else 0
    nonblocking_flag = getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, os.O_RDONLY | no_follow_flag | nonblocking_flag | binary_flag)
    try:
        descriptor_status = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_status.st_mode):
            raise OSError("path is not a regular file")
        if requires_identity_check:
            path_status = os.lstat(path)
            if stat.S_ISLNK(path_status.st_mode) or not os.path.samestat(
                descriptor_status, path_status
            ):
                raise OSError("path changed while it was opened")
        chunks: list[bytes] = []
        remaining_bytes = maximum_bytes + 1
        while remaining_bytes:
            chunk = os.read(descriptor, min(remaining_bytes, 65_536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining_bytes -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _validated_template(template: object, *, name: str) -> tuple[JsonValue, dict[str, int]]:
    def validate_and_copy(value: object, depth: int) -> tuple[JsonValue, int]:
        if depth > _MAXIMUM_JSON_DEPTH:
            raise ValueError(f"{name} exceeds the nesting limit")
        if value is None or isinstance(value, bool | int | str):
            placeholder_count = int(
                value in {_INPUT_PLACEHOLDER, _CASE_ID_PLACEHOLDER}
                if isinstance(value, str)
                else False
            )
            return cast(JsonValue, value), placeholder_count
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"{name} must contain standard JSON values")
            return value, 0
        if isinstance(value, list):
            validated_items: list[JsonValue] = []
            placeholder_count = 0
            for item in cast(list[object], value):
                validated_item, item_placeholder_count = validate_and_copy(item, depth + 1)
                validated_items.append(validated_item)
                placeholder_count += item_placeholder_count
            return validated_items, placeholder_count
        if isinstance(value, dict):
            object_value = cast(dict[object, object], value)
            if not all(isinstance(key, str) for key in object_value):
                raise ValueError("request_json_template object keys must be strings")
            validated_object: dict[str, JsonValue] = {}
            placeholder_count = 0
            for key, item in object_value.items():
                validated_item, item_placeholder_count = validate_and_copy(item, depth + 1)
                validated_object[cast(str, key)] = validated_item
                placeholder_count += item_placeholder_count
            return validated_object, placeholder_count
        raise ValueError("request_json_template must contain JSON values")

    if not isinstance(template, dict | list):
        raise ValueError(f"{name} must be an object or array")
    validated_template, _ = validate_and_copy(cast(object, template), 0)
    return validated_template, {
        _INPUT_PLACEHOLDER: _count_placeholder(validated_template, _INPUT_PLACEHOLDER),
        _CASE_ID_PLACEHOLDER: _count_placeholder(validated_template, _CASE_ID_PLACEHOLDER),
        _TURN_ID_PLACEHOLDER: _count_placeholder(validated_template, _TURN_ID_PLACEHOLDER),
        _SANDBOX_SETUP_PLACEHOLDER: _count_placeholder(
            validated_template, _SANDBOX_SETUP_PLACEHOLDER
        ),
    }


def _validated_execute_request_json_template(template: object) -> JsonValue:
    validated, counts = _validated_template(template, name="request_json_template")
    if (
        counts[_INPUT_PLACEHOLDER] != 1
        or counts[_CASE_ID_PLACEHOLDER] != 1
        or counts[_TURN_ID_PLACEHOLDER] != 1
    ):
        raise ValueError(
            "execute request_json_template must contain exactly one {{input}} and "
            "one {{case_id}} and {{turn_id}} leaf"
        )
    return validated


def _validated_lifecycle_request_json_template(template: object) -> JsonValue:
    validated, counts = _validated_template(template, name="request_json_template")
    if (
        counts[_INPUT_PLACEHOLDER]
        or counts[_CASE_ID_PLACEHOLDER] != 1
        or counts[_TURN_ID_PLACEHOLDER]
        or counts[_SANDBOX_SETUP_PLACEHOLDER]
    ):
        raise ValueError(
            "lifecycle request_json_template must contain exactly one {{case_id}} leaf "
            "and no {{input}} leaf"
        )
    return validated


def _validated_setup_request_json_template(template: object) -> JsonValue:
    validated, counts = _validated_template(template, name="request_json_template")
    if (
        counts[_INPUT_PLACEHOLDER]
        or counts[_CASE_ID_PLACEHOLDER] != 1
        or counts[_TURN_ID_PLACEHOLDER]
        or counts[_SANDBOX_SETUP_PLACEHOLDER] > 1
    ):
        raise ValueError(
            "setup request_json_template must contain exactly one {{case_id}} leaf, at most "
            "one {{sandbox_setup}} leaf, and no {{input}} or {{turn_id}} leaf"
        )
    return validated


def _validated_turn_observation_request_json_template(template: object) -> JsonValue:
    validated, counts = _validated_template(template, name="request_json_template")
    if (
        counts[_INPUT_PLACEHOLDER]
        or counts[_CASE_ID_PLACEHOLDER] != 1
        or counts[_TURN_ID_PLACEHOLDER] != 1
        or counts[_SANDBOX_SETUP_PLACEHOLDER]
    ):
        raise ValueError(
            "snapshot request_json_template must contain exactly one {{case_id}} and "
            "{{turn_id}} leaf and no {{input}} leaf"
        )
    return validated


def _validated_request_json_value(value: object, depth: int = 0) -> JsonValue:
    if depth > _MAXIMUM_JSON_DEPTH:
        raise ValueError("request_json exceeds the nesting limit")
    if value is None or isinstance(value, bool | int | str):
        return cast(JsonValue, value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("request_json must contain standard JSON values")
        return value
    if isinstance(value, list):
        return [
            _validated_request_json_value(item, depth + 1) for item in cast(list[object], value)
        ]
    if isinstance(value, dict):
        object_value = cast(dict[object, object], value)
        if not all(isinstance(key, str) for key in object_value):
            raise ValueError("request_json object keys must be strings")
        return {
            cast(str, key): _validated_request_json_value(item, depth + 1)
            for key, item in object_value.items()
        }
    raise ValueError("request_json must contain JSON values")


def _count_placeholder(value: JsonValue, placeholder: str) -> int:
    if value == placeholder:
        return 1
    if isinstance(value, list):
        return sum(_count_placeholder(item, placeholder) for item in value)
    if isinstance(value, dict):
        return sum(_count_placeholder(item, placeholder) for item in value.values())
    return 0


def _replace_request_placeholders(
    template: JsonValue,
    *,
    case_id: str,
    turn_id: str | None = None,
    raw_input: str | None = None,
    sandbox_setup: dict[str, JsonValue] | None = None,
) -> JsonValue:
    if template == _INPUT_PLACEHOLDER:
        if raw_input is None:
            raise AssertionError("input placeholder requires a raw input")
        return raw_input
    if template == _CASE_ID_PLACEHOLDER:
        return case_id
    if template == _TURN_ID_PLACEHOLDER:
        if turn_id is None:
            raise AssertionError("turn ID placeholder requires a turn ID")
        return turn_id
    if template == _SANDBOX_SETUP_PLACEHOLDER:
        if sandbox_setup is None:
            raise AssertionError("sandbox setup placeholder requires a fixture")
        return sandbox_setup
    if isinstance(template, list):
        return [
            _replace_request_placeholders(
                item,
                case_id=case_id,
                turn_id=turn_id,
                raw_input=raw_input,
                sandbox_setup=sandbox_setup,
            )
            for item in template
        ]
    if isinstance(template, dict):
        return {
            key: _replace_request_placeholders(
                value,
                case_id=case_id,
                turn_id=turn_id,
                raw_input=raw_input,
                sandbox_setup=sandbox_setup,
            )
            for key, value in template.items()
        }
    return template


def _validate_case_sandbox_setup(config: JsonHttpSandboxConfig, case: EvaluationCase) -> None:
    if case.sandbox_setup is not None:
        case.sandbox_setup.verify_digest()
    uses_per_record_setup = json_http_sandbox_uses_per_record_setup(config)
    if case.sandbox_setup is not None and not uses_per_record_setup:
        raise ValueError(
            "evaluation case has a sandbox setup fixture but setup request_json_template does "
            "not contain {{sandbox_setup}}"
        )
    if case.sandbox_setup is None and uses_per_record_setup:
        raise ValueError(
            "setup request_json_template requires a sandbox setup fixture for every case"
        )


def _parse_json_pointer(pointer: str) -> tuple[str, ...]:
    if pointer == "":
        return ()
    if not pointer.startswith("/"):
        raise ValueError("response_json_pointer must be an RFC 6901 JSON pointer")
    tokens: list[str] = []
    for encoded_token in pointer[1:].split("/"):
        token_characters: list[str] = []
        index = 0
        while index < len(encoded_token):
            character = encoded_token[index]
            if character != "~":
                token_characters.append(character)
                index += 1
                continue
            if index + 1 >= len(encoded_token) or encoded_token[index + 1] not in {"0", "1"}:
                raise ValueError("response_json_pointer must be an RFC 6901 JSON pointer")
            token_characters.append("~" if encoded_token[index + 1] == "0" else "/")
            index += 2
        tokens.append("".join(token_characters))
    return tuple(tokens)


def _resolve_json_pointer(
    document: JsonValue,
    pointer: str,
    *,
    allow_null: bool = False,
) -> JsonValue:
    selected: JsonValue = document
    for token in _parse_json_pointer(pointer):
        if isinstance(selected, dict):
            if token not in selected:
                raise _SandboxProtocolError(
                    "response_mapping", "sandbox API response JSON pointer was not found"
                )
            selected = selected[token]
        elif isinstance(selected, list):
            if token == "-" or not token.isascii() or not token.isdecimal():
                raise _SandboxProtocolError(
                    "response_mapping", "sandbox API response JSON pointer was not found"
                )
            if len(token) > 1 and token.startswith("0"):
                raise _SandboxProtocolError(
                    "response_mapping", "sandbox API response JSON pointer was not found"
                )
            index = int(token)
            if index >= len(selected):
                raise _SandboxProtocolError(
                    "response_mapping", "sandbox API response JSON pointer was not found"
                )
            selected = selected[index]
        else:
            raise _SandboxProtocolError(
                "response_mapping", "sandbox API response JSON pointer was not found"
            )
    if selected is None and not allow_null:
        raise _SandboxProtocolError(
            "response_mapping", "sandbox API response JSON pointer selected null"
        )
    return selected


async def _read_bounded_response(response: httpx.Response, maximum_bytes: int) -> bytes:
    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > maximum_bytes:
                raise _SandboxProtocolError(
                    "response_too_large", "sandbox API response exceeds the size limit"
                )
        except ValueError:
            pass
    response_body = bytearray()
    async for chunk in response.aiter_raw():
        response_body.extend(chunk)
        if len(response_body) > maximum_bytes:
            raise _SandboxProtocolError(
                "response_too_large", "sandbox API response exceeds the size limit"
            )
    return bytes(response_body)
