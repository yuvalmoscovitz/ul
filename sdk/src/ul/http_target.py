from __future__ import annotations

import asyncio
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Never, cast
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator
from ul_core.dataset import ObservedAgentOutput
from ul_core.models import SafetyEnvelope

_HEADER_NAME_PATTERN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
_ENVIRONMENT_VARIABLE_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_REQUEST_FIELD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,127}")
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


class JsonHttpDatasetTargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[1]
    url: str
    headers_from_env: dict[str, str] = Field(default_factory=dict)
    request_json_template: JsonValue
    response_json_pointer: str = ""

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, version: object) -> object:
        if type(version) is not int or version != 1:
            raise ValueError("version must be 1")
        return version

    @field_validator("url")
    @classmethod
    def validate_url(cls, url: str) -> str:
        _validate_endpoint(url, allow_insecure_http=True)
        return url

    @field_validator("headers_from_env")
    @classmethod
    def validate_headers_from_env(cls, headers: dict[str, str]) -> dict[str, str]:
        return _validate_header_environment_variables(headers)

    @field_validator("request_json_template", mode="before")
    @classmethod
    def validate_request_json_template(cls, template: object) -> JsonValue:
        return _validated_request_json_template(template)

    @field_validator("response_json_pointer")
    @classmethod
    def validate_response_json_pointer(cls, pointer: str) -> str:
        _parse_json_pointer(pointer)
        return pointer


def load_json_http_dataset_target_config(path: str | Path) -> JsonHttpDatasetTargetConfig:
    try:
        with Path(path).open("rb") as config_file:
            encoded_config = config_file.read(_MAXIMUM_CONFIG_BYTES + 1)
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
        endpoint: str,
        *,
        sandbox_confirmed: bool,
        fresh_state_confirmed: bool,
        request_field: str = "input",
        header_environment_variables: Mapping[str, str] | None = None,
        request_json_template: JsonValue | None = None,
        response_json_pointer: str = "",
        allow_insecure_http: bool = False,
        timeout_seconds: float = 30,
        max_request_bytes: int = 1_000_000,
        max_response_bytes: int = 1_000_000,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._headers = validate_json_http_dataset_target_configuration(
            endpoint,
            sandbox_confirmed=sandbox_confirmed,
            fresh_state_confirmed=fresh_state_confirmed,
            request_field=request_field,
            header_environment_variables=header_environment_variables,
            request_json_template=request_json_template,
            response_json_pointer=response_json_pointer,
            allow_insecure_http=allow_insecure_http,
            timeout_seconds=timeout_seconds,
            max_request_bytes=max_request_bytes,
            max_response_bytes=max_response_bytes,
        )
        self._endpoint = endpoint
        self._request_field = request_field
        self._request_json_template = (
            None
            if request_json_template is None
            else _validated_request_json_template(request_json_template)
        )
        self._response_json_pointer = response_json_pointer
        self._timeout_seconds = timeout_seconds
        self._max_request_bytes = max_request_bytes
        self._max_response_bytes = max_response_bytes
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(follow_redirects=False, trust_env=False)
        self.safety_envelope = SafetyEnvelope(
            description="Customer-confirmed isolated, fresh-state JSON HTTP sandbox.",
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
        fresh_state_confirmed: bool,
        allow_insecure_http: bool = False,
        timeout_seconds: float = 30,
        max_request_bytes: int = 1_000_000,
        max_response_bytes: int = 1_000_000,
        client: httpx.AsyncClient | None = None,
    ) -> JsonHttpDatasetTarget:
        return cls(
            config.url,
            sandbox_confirmed=sandbox_confirmed,
            fresh_state_confirmed=fresh_state_confirmed,
            header_environment_variables=config.headers_from_env,
            request_json_template=config.request_json_template,
            response_json_pointer=config.response_json_pointer,
            allow_insecure_http=allow_insecure_http,
            timeout_seconds=timeout_seconds,
            max_request_bytes=max_request_bytes,
            max_response_bytes=max_response_bytes,
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
        request_body = json.dumps(
            self._request_body(raw_input),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(request_body) > self._max_request_bytes:
            raise RuntimeError("HTTP dataset target request exceeds the size limit")
        try:
            async with asyncio.timeout(self._timeout_seconds):
                async with self._client.stream(
                    "POST",
                    self._endpoint,
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
                    if not _is_json_content_type(response.headers.get("content-type")):
                        raise RuntimeError("HTTP dataset target response must be JSON")
                    content_encoding = response.headers.get("content-encoding")
                    if content_encoding is not None and content_encoding.casefold() != "identity":
                        raise RuntimeError(
                            "HTTP dataset target response must not use content encoding"
                        )
                    response_body = await _read_bounded_response(response, self._max_response_bytes)
        except TimeoutError:
            raise RuntimeError("HTTP dataset target request timed out") from None
        except RuntimeError:
            raise
        except httpx.HTTPError:
            raise RuntimeError("HTTP dataset target request failed") from None

        try:
            raw_output = json.loads(
                response_body,
                object_pairs_hook=_reject_duplicate_object_keys,
                parse_constant=_reject_nonstandard_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
            raise RuntimeError("HTTP dataset target returned invalid JSON") from None
        if raw_output is None:
            raise RuntimeError("HTTP dataset target returned null JSON")
        raw_output = _resolve_json_pointer(raw_output, self._response_json_pointer)
        try:
            return ObservedAgentOutput(raw_output=raw_output)
        except (RecursionError, ValidationError):
            raise RuntimeError("HTTP dataset target returned invalid JSON") from None

    def _request_body(self, raw_input: str) -> JsonValue:
        if self._request_json_template is None:
            return {self._request_field: raw_input}
        return _replace_input_placeholder(self._request_json_template, raw_input)


def validate_json_http_dataset_target_configuration(
    endpoint: str,
    *,
    sandbox_confirmed: bool,
    fresh_state_confirmed: bool,
    request_field: str = "input",
    header_environment_variables: Mapping[str, str] | None = None,
    request_json_template: JsonValue | None = None,
    response_json_pointer: str = "",
    allow_insecure_http: bool = False,
    timeout_seconds: float = 30,
    max_request_bytes: int = 1_000_000,
    max_response_bytes: int = 1_000_000,
) -> dict[str, str]:
    if sandbox_confirmed is not True:
        raise ValueError("HTTP dataset targets require explicit sandbox confirmation")
    if fresh_state_confirmed is not True:
        raise ValueError("HTTP dataset targets require explicit fresh-state confirmation")
    _validate_endpoint(endpoint, allow_insecure_http)
    if _REQUEST_FIELD_PATTERN.fullmatch(request_field) is None:
        raise ValueError("request_field must be a simple JSON field name")
    if request_json_template is not None:
        if request_field != "input":
            raise ValueError("request_field cannot be combined with request_json_template")
        _validated_request_json_template(request_json_template)
    _parse_json_pointer(response_json_pointer)
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
    validated_header_environment_variables = _validate_header_environment_variables(
        header_environment_variables or {}
    )
    return _headers_from_environment(validated_header_environment_variables)


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


def _resolve_json_pointer(document: JsonValue, pointer: str) -> JsonValue:
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
    if selected is None:
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
