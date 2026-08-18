from __future__ import annotations

import asyncio
import json
import math
import os
import re
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
from ul_core.contracts import DatasetTargetLifecycleError
from ul_core.dataset import ObservedAgentOutput
from ul_core.models import SafetyEnvelope

_HEADER_NAME_PATTERN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
_ENVIRONMENT_VARIABLE_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_INPUT_PLACEHOLDER = "{{input}}"
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


class _TargetDeliveryUncertainError(RuntimeError):
    pass


class JsonHttpLifecycleCallConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    url: str
    request_json: JsonValue = Field(default_factory=dict)

    @field_validator("url")
    @classmethod
    def validate_url(cls, url: str) -> str:
        _validate_endpoint(url, allow_insecure_http=True)
        return url

    @field_validator("request_json", mode="before")
    @classmethod
    def validate_request_json(cls, request_json: object) -> JsonValue:
        return _validated_static_request_json(request_json)


class JsonHttpLifecycleObservationConfig(JsonHttpLifecycleCallConfig):
    response_json_pointer: str = ""

    @field_validator("response_json_pointer")
    @classmethod
    def validate_response_json_pointer(cls, pointer: str) -> str:
        _parse_json_pointer(pointer)
        return pointer


class JsonHttpLifecycleResetConfig(JsonHttpLifecycleCallConfig):
    generation_json_pointer: str
    clean_state_json_pointer: str
    clean_state_value: JsonValue

    @field_validator("generation_json_pointer", "clean_state_json_pointer")
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
        if self.generation_json_pointer == self.clean_state_json_pointer:
            raise ValueError("reset generation and clean-state pointers must be different")
        return self


class JsonHttpLifecycleExecuteTurnConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    url: str
    request_json_template: JsonValue
    response_json_pointer: str = ""

    @field_validator("url")
    @classmethod
    def validate_url(cls, url: str) -> str:
        _validate_endpoint(url, allow_insecure_http=True)
        return url

    @field_validator("request_json_template", mode="before")
    @classmethod
    def validate_request_json_template(cls, template: object) -> JsonValue:
        return _validated_request_json_template(template)

    @field_validator("response_json_pointer")
    @classmethod
    def validate_response_json_pointer(cls, pointer: str) -> str:
        _parse_json_pointer(pointer)
        return pointer


class JsonHttpDatasetTargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[2]
    headers_from_env: dict[str, str] = Field(default_factory=dict)
    reset: JsonHttpLifecycleResetConfig
    setup: JsonHttpLifecycleCallConfig | None = None
    execute_turn: JsonHttpLifecycleExecuteTurnConfig
    snapshot: JsonHttpLifecycleObservationConfig

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, version: object) -> object:
        if type(version) is not int or version != 2:
            raise ValueError("version must be 2")
        return version

    @field_validator("headers_from_env")
    @classmethod
    def validate_headers_from_env(cls, headers: dict[str, str]) -> dict[str, str]:
        return _validate_header_environment_variables(headers)

    @model_validator(mode="after")
    def validate_same_origin(self) -> Self:
        origins = {_endpoint_origin(url) for url in json_http_target_config_urls(self)}
        if len(origins) != 1:
            raise ValueError("all lifecycle endpoints must use the same origin")
        return self


def load_json_http_dataset_target_config(path: str | Path) -> JsonHttpDatasetTargetConfig:
    try:
        encoded_config = _read_bounded_regular_file(Path(path), maximum_bytes=_MAXIMUM_CONFIG_BYTES)
    except OSError:
        raise RuntimeError("HTTP target config could not be read") from None
    if len(encoded_config) > _MAXIMUM_CONFIG_BYTES:
        raise ValueError("HTTP target config exceeds the size limit")
    try:
        decoded_config = encoded_config.decode("utf-8")
        raw_config = json.loads(
            decoded_config,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
        _reject_deep_json(raw_config)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise ValueError("HTTP target config contains invalid JSON") from None
    try:
        return JsonHttpDatasetTargetConfig.model_validate(raw_config)
    except RecursionError:
        raise ValueError("HTTP target config is invalid") from None
    except ValidationError as error:
        validation_reasons: list[str] = []
        for issue in error.errors(include_url=False, include_context=False, include_input=False):
            field_path = ".".join(str(part) for part in issue["loc"])
            message = str(issue["msg"]).removeprefix("Value error, ")
            validation_reasons.append(f"{field_path}: {message}")
        raise ValueError(
            f"HTTP target config is invalid: {'; '.join(validation_reasons)}"
        ) from None


class JsonHttpDatasetTarget:
    def __init__(
        self,
        config: JsonHttpDatasetTargetConfig,
        *,
        sandbox_confirmed: bool,
        allow_insecure_http: bool = False,
        timeout_seconds: float = 30,
        max_request_bytes: int = 1_000_000,
        max_response_bytes: int = 1_000_000,
        max_target_calls: int | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if max_target_calls is not None and (
            isinstance(max_target_calls, bool) or max_target_calls <= 0
        ):
            raise ValueError("max_target_calls must be positive")
        self._headers = validate_json_http_dataset_target_configuration(
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
        self._remaining_target_calls = max_target_calls
        self._config = config
        self._lifecycle_lock = asyncio.Lock()
        self._last_reset_generation: str | int | None = None
        self._lifecycle_state_uncertain = False
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(follow_redirects=False, trust_env=False)
        self.safety_envelope = SafetyEnvelope(
            description="Customer-confirmed isolated JSON HTTP sandbox with an explicit lifecycle.",
            isolated=True,
            allows_network_egress=True,
            allows_business_side_effects=False,
        )

    @classmethod
    def from_config(
        cls,
        config: JsonHttpDatasetTargetConfig,
        *,
        sandbox_confirmed: bool,
        allow_insecure_http: bool = False,
        timeout_seconds: float = 30,
        max_request_bytes: int = 1_000_000,
        max_response_bytes: int = 1_000_000,
        max_target_calls: int | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> JsonHttpDatasetTarget:
        return cls(
            config,
            sandbox_confirmed=sandbox_confirmed,
            allow_insecure_http=allow_insecure_http,
            timeout_seconds=timeout_seconds,
            max_request_bytes=max_request_bytes,
            max_response_bytes=max_response_bytes,
            max_target_calls=max_target_calls,
            client=client,
        )

    @property
    def fresh_state_per_execution(self) -> bool:
        return True

    async def __aenter__(self) -> JsonHttpDatasetTarget:
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

    async def execute(self, raw_input: str) -> ObservedAgentOutput:
        async with self._lifecycle_lock:
            return await self._execute_stateful(raw_input)

    async def _execute_stateful(self, raw_input: str) -> ObservedAgentOutput:
        config = self._config
        if self._lifecycle_state_uncertain:
            raise DatasetTargetLifecycleError(
                failed_phase="blocked_state_uncertain",
                completed_phases=(),
                cleanup_reset_failed=True,
                target_state_uncertain=True,
            )
        self._reserve_target_calls(json_http_target_calls_per_execution(config))
        self._lifecycle_state_uncertain = True
        completed_phase_names: list[str] = []
        completed_phases: list[JsonValue] = []
        lifecycle_started = False
        failed_phase: str | None = None
        cleanup_reset_failed = False
        delivery_uncertain = False
        current_phase = "reset"
        execute_response: JsonValue | None = None
        committed_state_snapshot: JsonValue | None = None
        try:
            lifecycle_started = True
            await self._reset(config.reset)
            completed_phase_names.append("reset")
            completed_phases.append({"phase": "reset", "status": "succeeded"})
            if config.setup is not None:
                current_phase = "setup"
                await self._post_without_observation(
                    config.setup.url,
                    config.setup.request_json,
                    consume_budget=False,
                )
                completed_phase_names.append("setup")
                completed_phases.append({"phase": "setup", "status": "succeeded"})
            current_phase = "execute_turn"
            execute_response = await self._post_for_json(
                config.execute_turn.url,
                _replace_input_placeholder(config.execute_turn.request_json_template, raw_input),
                config.execute_turn.response_json_pointer,
                consume_budget=False,
            )
            completed_phase_names.append("execute_turn")
            completed_phases.append({"phase": "execute_turn", "status": "succeeded"})
            current_phase = "snapshot"
            committed_state_snapshot = await self._post_for_json(
                config.snapshot.url,
                config.snapshot.request_json,
                config.snapshot.response_json_pointer,
                consume_budget=False,
            )
            completed_phase_names.append("snapshot")
            completed_phases.append({"phase": "snapshot", "status": "succeeded"})
        except _TargetDeliveryUncertainError:
            failed_phase = current_phase
            delivery_uncertain = current_phase in {"setup", "execute_turn", "snapshot"}
        except asyncio.CancelledError:
            if current_phase in {"setup", "execute_turn", "snapshot"}:
                delivery_uncertain = True
            raise
        except RuntimeError:
            failed_phase = _next_lifecycle_phase(
                tuple(completed_phase_names), config.setup is not None
            )
        finally:
            if lifecycle_started:
                try:
                    await self._reset(config.reset)
                    completed_phase_names.append("cleanup_reset")
                    completed_phases.append({"phase": "cleanup_reset", "status": "succeeded"})
                    if not delivery_uncertain:
                        self._lifecycle_state_uncertain = False
                except RuntimeError:
                    cleanup_reset_failed = True
        if failed_phase is not None or cleanup_reset_failed:
            raise DatasetTargetLifecycleError(
                failed_phase=failed_phase or "cleanup_reset",
                completed_phases=tuple(completed_phase_names),
                cleanup_reset_failed=cleanup_reset_failed,
                target_state_uncertain=self._lifecycle_state_uncertain,
            ) from None
        if execute_response is None or committed_state_snapshot is None:
            raise AssertionError(
                "successful lifecycle requires execution and snapshot observations"
            )
        try:
            return ObservedAgentOutput(
                raw_output=execute_response,
                metadata={
                    "target_protocol_version": 2,
                    "lifecycle_calls": completed_phases,
                    "committed_state_snapshot": committed_state_snapshot,
                },
            )
        except (RecursionError, ValidationError):
            raise RuntimeError("HTTP dataset target returned invalid JSON") from None

    async def _reset(self, config: JsonHttpLifecycleResetConfig) -> None:
        reset_response = await self._post_for_json(
            config.url,
            config.request_json,
            "",
            consume_budget=False,
        )
        generation = _resolve_json_pointer(reset_response, config.generation_json_pointer)
        if isinstance(generation, bool) or not isinstance(generation, str | int):
            raise RuntimeError("HTTP dataset target reset generation is invalid")
        if isinstance(generation, str) and not generation:
            raise RuntimeError("HTTP dataset target reset generation is invalid")
        if generation == self._last_reset_generation:
            raise RuntimeError("HTTP dataset target reset generation did not change")
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
            raise RuntimeError("HTTP dataset target reset did not report clean state")

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
            raise RuntimeError("HTTP dataset target request exceeds the size limit")
        if consume_budget:
            self._reserve_target_calls(1)
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
                        raise RuntimeError("HTTP dataset target returned a non-success status")
                    if response_json_pointer is not None and not _is_json_content_type(
                        response.headers.get("content-type")
                    ):
                        raise RuntimeError("HTTP dataset target response must be JSON")
                    content_encoding = response.headers.get("content-encoding")
                    if content_encoding is not None and content_encoding.casefold() != "identity":
                        raise RuntimeError(
                            "HTTP dataset target response must not use content encoding"
                        )
                    response_body = await _read_bounded_response(response, self._max_response_bytes)
        except TimeoutError:
            raise _TargetDeliveryUncertainError(
                "HTTP dataset target request delivery is uncertain"
            ) from None
        except RuntimeError:
            raise
        except httpx.HTTPError:
            raise _TargetDeliveryUncertainError(
                "HTTP dataset target request delivery is uncertain"
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
            raise RuntimeError("HTTP dataset target returned invalid JSON") from None
        if raw_output is None:
            raise RuntimeError("HTTP dataset target returned null JSON")
        return _resolve_json_pointer(raw_output, response_json_pointer)

    def _reserve_target_calls(self, target_calls: int) -> None:
        if self._remaining_target_calls is None:
            return
        if target_calls > self._remaining_target_calls:
            raise RuntimeError("HTTP dataset target call budget exhausted")
        self._remaining_target_calls -= target_calls


def validate_json_http_dataset_target_configuration(
    config: JsonHttpDatasetTargetConfig,
    *,
    sandbox_confirmed: bool,
    allow_insecure_http: bool = False,
    timeout_seconds: float = 30,
    max_request_bytes: int = 1_000_000,
    max_response_bytes: int = 1_000_000,
) -> dict[str, str]:
    if sandbox_confirmed is not True:
        raise ValueError("HTTP dataset targets require explicit sandbox confirmation")
    for endpoint in json_http_target_config_urls(config):
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


def json_http_target_calls_per_execution(
    config: JsonHttpDatasetTargetConfig,
) -> int:
    return 5 if config.setup is not None else 4


def json_http_target_config_urls(
    config: JsonHttpDatasetTargetConfig,
) -> tuple[str, ...]:
    return (
        config.reset.url,
        *((config.setup.url,) if config.setup is not None else ()),
        config.execute_turn.url,
        config.snapshot.url,
    )


def _next_lifecycle_phase(
    completed_phases: tuple[str, ...],
    setup_configured: bool,
) -> str:
    phases = ("reset", *(("setup",) if setup_configured else ()), "execute_turn", "snapshot")
    return phases[len(completed_phases)]


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
        if _ENVIRONMENT_VARIABLE_PATTERN.fullmatch(environment_variable) is None:
            raise ValueError("header environment variable names must be valid identifiers")
        normalized_names.add(normalized_name)
        validated[header_name] = environment_variable
    return validated


def _headers_from_environment(header_environment_variables: Mapping[str, str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    total_header_bytes = 0
    for header_name, environment_variable in header_environment_variables.items():
        value = os.environ.get(environment_variable)
        if value is None or not value.strip():
            raise RuntimeError("HTTP dataset target header environment variable is not set")
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise RuntimeError("HTTP dataset target header environment variable is invalid")
        encoded_value_bytes = len(value.encode("utf-8"))
        if encoded_value_bytes > _MAXIMUM_HEADER_VALUE_BYTES:
            raise RuntimeError("HTTP dataset target header environment variable is too large")
        total_header_bytes += len(header_name.encode("ascii")) + encoded_value_bytes
        if total_header_bytes > _MAXIMUM_TOTAL_HEADER_BYTES:
            raise RuntimeError("HTTP dataset target headers exceed the size limit")
        headers[header_name] = value
    return headers


def _is_json_content_type(content_type: str | None) -> bool:
    if content_type is None:
        return False
    media_type = content_type.partition(";")[0].strip().casefold()
    return media_type == "application/json" or media_type.endswith("+json")


def _reject_nonstandard_json_constant(value: str) -> Never:
    raise ValueError("nonstandard JSON constant")


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


def _validated_request_json_template(template: object) -> JsonValue:
    def validate_and_copy(value: object, depth: int) -> tuple[JsonValue, int]:
        if depth > _MAXIMUM_JSON_DEPTH:
            raise ValueError("request_json_template exceeds the nesting limit")
        if value is None or isinstance(value, bool | int | str):
            return cast(JsonValue, value), int(value == _INPUT_PLACEHOLDER)
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("request_json_template must contain standard JSON values")
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
        raise ValueError("request_json_template must be an object or array")
    validated_template, placeholder_count = validate_and_copy(cast(object, template), 0)
    if placeholder_count != 1:
        raise ValueError("request_json_template must contain exactly one {{input}} leaf")
    return validated_template


def _validated_static_request_json(request_json: object) -> JsonValue:
    if not isinstance(request_json, dict | list):
        raise ValueError("request_json must be an object or array")
    validated = _validated_request_json_value(cast(object, request_json))
    if _contains_input_placeholder(validated):
        raise ValueError("request_json must not contain {{input}}")
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


def _contains_input_placeholder(value: JsonValue) -> bool:
    if value == _INPUT_PLACEHOLDER:
        return True
    if isinstance(value, list):
        return any(_contains_input_placeholder(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_input_placeholder(item) for item in value.values())
    return False


def _replace_input_placeholder(template: JsonValue, raw_input: str) -> JsonValue:
    if template == _INPUT_PLACEHOLDER:
        return raw_input
    if isinstance(template, list):
        return [_replace_input_placeholder(item, raw_input) for item in template]
    if isinstance(template, dict):
        return {
            key: _replace_input_placeholder(value, raw_input) for key, value in template.items()
        }
    return template


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
                raise RuntimeError("HTTP dataset target response JSON pointer was not found")
            selected = selected[token]
        elif isinstance(selected, list):
            if token == "-" or not token.isascii() or not token.isdecimal():
                raise RuntimeError("HTTP dataset target response JSON pointer was not found")
            if len(token) > 1 and token.startswith("0"):
                raise RuntimeError("HTTP dataset target response JSON pointer was not found")
            index = int(token)
            if index >= len(selected):
                raise RuntimeError("HTTP dataset target response JSON pointer was not found")
            selected = selected[index]
        else:
            raise RuntimeError("HTTP dataset target response JSON pointer was not found")
    if selected is None and not allow_null:
        raise RuntimeError("HTTP dataset target response JSON pointer selected null")
    return selected


async def _read_bounded_response(response: httpx.Response, maximum_bytes: int) -> bytes:
    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > maximum_bytes:
                raise RuntimeError("HTTP dataset target response exceeds the size limit")
        except ValueError:
            pass
    response_body = bytearray()
    async for chunk in response.aiter_raw():
        response_body.extend(chunk)
        if len(response_body) > maximum_bytes:
            raise RuntimeError("HTTP dataset target response exceeds the size limit")
    return bytes(response_body)
