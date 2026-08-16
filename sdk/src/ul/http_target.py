from __future__ import annotations

import asyncio
import json
import math
import os
import re
from collections.abc import Mapping
from types import TracebackType
from typing import Any, Never
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError
from ul_core.dataset import ObservedAgentOutput
from ul_core.models import SafetyEnvelope

_HEADER_NAME_PATTERN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
_ENVIRONMENT_VARIABLE_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_REQUEST_FIELD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,127}")
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


class JsonHttpDatasetTarget:
    def __init__(
        self,
        endpoint: str,
        *,
        sandbox_confirmed: bool,
        fresh_state_confirmed: bool,
        request_field: str = "input",
        header_environment_variables: Mapping[str, str] | None = None,
        allow_insecure_http: bool = False,
        timeout_seconds: float = 30,
        max_response_bytes: int = 1_000_000,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._headers = validate_json_http_dataset_target_configuration(
            endpoint,
            sandbox_confirmed=sandbox_confirmed,
            fresh_state_confirmed=fresh_state_confirmed,
            request_field=request_field,
            header_environment_variables=header_environment_variables,
            allow_insecure_http=allow_insecure_http,
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
        )
        self._endpoint = endpoint
        self._request_field = request_field
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(follow_redirects=False, trust_env=False)
        self.safety_envelope = SafetyEnvelope(
            description="Customer-confirmed isolated, fresh-state JSON HTTP sandbox.",
            isolated=True,
            allows_network_egress=True,
            allows_business_side_effects=False,
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
        try:
            async with asyncio.timeout(self._timeout_seconds):
                async with self._client.stream(
                    "POST",
                    self._endpoint,
                    headers={"Accept-Encoding": "identity", **self._headers},
                    json={self._request_field: raw_input},
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
        try:
            return ObservedAgentOutput(raw_output=raw_output)
        except (RecursionError, ValidationError):
            raise RuntimeError("HTTP dataset target returned invalid JSON") from None


def validate_json_http_dataset_target_configuration(
    endpoint: str,
    *,
    sandbox_confirmed: bool,
    fresh_state_confirmed: bool,
    request_field: str = "input",
    header_environment_variables: Mapping[str, str] | None = None,
    allow_insecure_http: bool = False,
    timeout_seconds: float = 30,
    max_response_bytes: int = 1_000_000,
) -> dict[str, str]:
    if sandbox_confirmed is not True:
        raise ValueError("HTTP dataset targets require explicit sandbox confirmation")
    if fresh_state_confirmed is not True:
        raise ValueError("HTTP dataset targets require explicit fresh-state confirmation")
    _validate_endpoint(endpoint, allow_insecure_http)
    if _REQUEST_FIELD_PATTERN.fullmatch(request_field) is None:
        raise ValueError("request_field must be a simple JSON field name")
    if (
        isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be positive and finite")
    if isinstance(max_response_bytes, bool) or max_response_bytes <= 0:
        raise ValueError("max_response_bytes must be positive")
    validated_header_environment_variables = _validate_header_environment_variables(
        header_environment_variables or {}
    )
    return _headers_from_environment(validated_header_environment_variables)


def _validate_endpoint(endpoint: str, allow_insecure_http: bool) -> str:
    if not endpoint or endpoint != endpoint.strip():
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
    for header_name, environment_variable in header_environment_variables.items():
        value = os.environ.get(environment_variable)
        if value is None or not value.strip():
            raise RuntimeError("HTTP dataset target header environment variable is not set")
        if "\r" in value or "\n" in value:
            raise RuntimeError("HTTP dataset target header environment variable is invalid")
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
