from __future__ import annotations

import asyncio
import hashlib
import ipaddress
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, Literal, Protocol, Self, cast
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, SecretStr, model_validator
from ul_core.models import ULModel

type LLMRole = Literal["deconstruct", "render", "equivalence", "materiality"]
type LLMProviderType = Literal["openrouter", "openai-compatible"]
type LLMReasoningMode = Literal["required", "omitted"]
type LLMReasoningEffort = Literal["minimal", "none", "low"]


class DatasetLLMSettings(Protocol):
    @property
    def live_calls(self) -> bool: ...

    @property
    def allow_external_data_processing(self) -> bool: ...

    @property
    def api_key(self) -> SecretStr | None: ...

    @property
    def semantic_provider_id(self) -> str: ...

    @property
    def semantic_provider_type(self) -> LLMProviderType: ...

    @property
    def semantic_base_url(self) -> str: ...

    @property
    def api_key_environment_variable(self) -> str: ...

    @property
    def api_key_required(self) -> bool: ...

    @property
    def upstream_provider(self) -> str | None: ...

    @property
    def model(self) -> str: ...

    @property
    def render_model(self) -> str: ...

    @property
    def equivalence_model(self) -> str: ...

    @property
    def materiality_model(self) -> str: ...

    @property
    def deconstruct_reasoning(self) -> LLMReasoningMode: ...

    @property
    def render_reasoning(self) -> LLMReasoningMode: ...

    @property
    def equivalence_reasoning(self) -> LLMReasoningMode: ...

    @property
    def max_output_tokens(self) -> int: ...

    @property
    def max_render_tokens(self) -> int: ...

    @property
    def max_response_bytes(self) -> int: ...

    @property
    def timeout_seconds(self) -> float: ...


class LLMRoleConfig(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    role: LLMRole
    model: str = Field(min_length=1, max_length=200)
    max_output_tokens: int = Field(ge=1, le=32_768)
    token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    reasoning_mode: LLMReasoningMode
    reasoning_effort: LLMReasoningEffort | None = None

    @model_validator(mode="after")
    def validate_reasoning(self) -> Self:
        if (self.reasoning_mode == "required") != (self.reasoning_effort is not None):
            raise ValueError("required reasoning needs an effort and omitted reasoning forbids it")
        return self

    def reasoning_option(self) -> dict[str, JsonValue] | None:
        if self.reasoning_effort is None:
            return None
        return {"effort": self.reasoning_effort}

    def reasoning_metadata(self) -> dict[str, JsonValue]:
        return {"mode": self.reasoning_mode, "effort": self.reasoning_effort}


class LLMClientIdentity(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    provider_id: str = Field(min_length=1, max_length=100)
    provider_type: LLMProviderType
    upstream_provider: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]*$",
    )
    endpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    roles: tuple[LLMRoleConfig, ...] = Field(min_length=4, max_length=4)
    temperature: Literal[0] = 0
    timeout_seconds: float = Field(gt=0, le=300)
    max_response_bytes: int = Field(ge=1_024, le=5_000_000)
    data_policy: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        configured_roles = tuple(role.role for role in self.roles)
        required_roles: tuple[LLMRole, ...] = (
            "deconstruct",
            "render",
            "equivalence",
            "materiality",
        )
        if len(set(configured_roles)) != len(configured_roles) or set(configured_roles) != set(
            required_roles
        ):
            raise ValueError("LLM client identity requires each semantic role exactly once")
        if (self.provider_type == "openrouter") != (self.upstream_provider is not None):
            raise ValueError("only an OpenRouter LLM identity requires an upstream provider")
        return self

    def role_config(self, role: LLMRole) -> LLMRoleConfig:
        return next(configuration for configuration in self.roles if configuration.role == role)


class LLMClientConfig(ULModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        hide_input_in_errors=True,
    )

    provider_id: str = Field(min_length=1, max_length=100)
    provider_type: LLMProviderType
    base_url: str = Field(min_length=1, max_length=2_000)
    api_key: SecretStr | None = Field(default=None, repr=False)
    api_key_environment_variable: str = Field(min_length=1, max_length=100)
    api_key_required: bool
    live_calls: bool
    allow_external_data_processing: bool
    upstream_provider: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]*$",
    )
    roles: tuple[LLMRoleConfig, ...] = Field(min_length=4, max_length=4)
    temperature: Literal[0] = 0
    timeout_seconds: float = Field(gt=0, le=300)
    max_response_bytes: int = Field(ge=1_024, le=5_000_000)

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        object.__setattr__(self, "base_url", _validated_base_url(self.base_url))
        configured_roles = tuple(role.role for role in self.roles)
        required_roles: tuple[LLMRole, ...] = (
            "deconstruct",
            "render",
            "equivalence",
            "materiality",
        )
        if len(set(configured_roles)) != len(configured_roles) or set(configured_roles) != set(
            required_roles
        ):
            raise ValueError("LLM client configuration requires each semantic role exactly once")
        if self.provider_type == "openrouter":
            if self.upstream_provider is None:
                raise ValueError("OpenRouter LLM configuration requires upstream_provider")
        elif self.upstream_provider is not None:
            raise ValueError("OpenAI-compatible LLM configuration cannot pin an upstream provider")
        return self

    @property
    def endpoint_sha256(self) -> str:
        return hashlib.sha256(self.base_url.encode()).hexdigest()

    @property
    def enforces_parameter_support(self) -> bool:
        return self.provider_type == "openrouter"

    @property
    def trust_environment_transport(self) -> bool:
        return self.provider_type == "openrouter"

    def role_config(self, role: LLMRole) -> LLMRoleConfig:
        return next(configuration for configuration in self.roles if configuration.role == role)

    def data_policy(self) -> dict[str, JsonValue]:
        if self.provider_type == "openrouter":
            return {
                "external_processing": True,
                "provider_policy_declared": True,
                "data_collection": "deny",
                "zero_data_retention_required": True,
                "upstream_provider": self.upstream_provider,
                "implication": (
                    "The configured route requires data collection to be denied and zero data "
                    "retention; the evaluator request is still processed externally."
                ),
            }
        return {
            "external_processing": True,
            "provider_policy_declared": False,
            "implication": (
                "The configured endpoint receives evaluator prompts and sample data; UL cannot "
                "verify its retention or training policy."
            ),
        }

    def evidence_identity(self) -> LLMClientIdentity:
        return LLMClientIdentity(
            provider_id=self.provider_id,
            provider_type=self.provider_type,
            upstream_provider=self.upstream_provider,
            endpoint_sha256=self.endpoint_sha256,
            roles=self.roles,
            temperature=self.temperature,
            timeout_seconds=self.timeout_seconds,
            max_response_bytes=self.max_response_bytes,
            data_policy=self.data_policy(),
        )

    def evaluator_judge_configuration(self, role: LLMRole) -> dict[str, JsonValue]:
        role_config = self.role_config(role)
        return {
            "base_url": self.base_url,
            "allow_external_data_processing": True,
            "data_policy": (
                "openrouter_zdr" if self.provider_type == "openrouter" else "provider_default"
            ),
            "upstream_provider": self.upstream_provider,
            "timeout_seconds": self.timeout_seconds,
            "max_output_tokens": role_config.max_output_tokens,
            "token_parameter": role_config.token_parameter,
            "max_response_bytes": self.max_response_bytes,
        }

    def request_options(
        self,
        *,
        role: LLMRole,
        seed: int,
        top_p: float | None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        role_config = self.role_config(role)
        requested_tokens = max_output_tokens or role_config.max_output_tokens
        if requested_tokens > role_config.max_output_tokens:
            raise ValueError("LLM request exceeds the configured role token limit")
        options: dict[str, Any] = {
            "model": role_config.model,
            "temperature": self.temperature,
            "seed": seed,
            role_config.token_parameter: requested_tokens,
        }
        if self.provider_type == "openrouter":
            reasoning = role_config.reasoning_option()
            if reasoning is not None:
                options["reasoning"] = reasoning
            options["provider"] = {
                "require_parameters": True,
                "data_collection": "deny",
                "zdr": True,
                "only": [self.upstream_provider],
                "allow_fallbacks": False,
            }
        if top_p is not None:
            options["top_p"] = top_p
        return options


def llm_client_config_from_dataset_settings(settings: DatasetLLMSettings) -> LLMClientConfig:
    upstream_provider = (
        settings.upstream_provider if settings.semantic_provider_type == "openrouter" else None
    )
    supports_reasoning_options = settings.semantic_provider_type == "openrouter"
    return LLMClientConfig(
        provider_id=settings.semantic_provider_id,
        provider_type=settings.semantic_provider_type,
        base_url=settings.semantic_base_url,
        api_key=settings.api_key,
        api_key_environment_variable=settings.api_key_environment_variable,
        api_key_required=settings.api_key_required,
        live_calls=settings.live_calls,
        allow_external_data_processing=settings.allow_external_data_processing,
        upstream_provider=upstream_provider,
        roles=(
            LLMRoleConfig(
                role="deconstruct",
                model=settings.model,
                max_output_tokens=settings.max_output_tokens,
                reasoning_mode=(
                    settings.deconstruct_reasoning if supports_reasoning_options else "omitted"
                ),
                reasoning_effort=(
                    "minimal"
                    if supports_reasoning_options and settings.deconstruct_reasoning == "required"
                    else None
                ),
            ),
            LLMRoleConfig(
                role="render",
                model=settings.render_model,
                max_output_tokens=settings.max_render_tokens,
                reasoning_mode=(
                    settings.render_reasoning if supports_reasoning_options else "omitted"
                ),
                reasoning_effort=(
                    "none"
                    if supports_reasoning_options and settings.render_reasoning == "required"
                    else None
                ),
            ),
            LLMRoleConfig(
                role="equivalence",
                model=settings.equivalence_model,
                max_output_tokens=min(settings.max_output_tokens, 1_024),
                reasoning_mode=(
                    settings.equivalence_reasoning if supports_reasoning_options else "omitted"
                ),
                reasoning_effort=(
                    "low"
                    if supports_reasoning_options and settings.equivalence_reasoning == "required"
                    else None
                ),
            ),
            LLMRoleConfig(
                role="materiality",
                model=settings.materiality_model,
                max_output_tokens=512,
                reasoning_mode="omitted",
            ),
        ),
        timeout_seconds=settings.timeout_seconds,
        max_response_bytes=settings.max_response_bytes,
    )


class _LLMResponseMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    content: str


class _LLMResponseChoice(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    message: _LLMResponseMessage


class LLMUsage(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, hide_input_in_errors=True)

    prompt_tokens: int | None = Field(default=None, ge=0, le=1_000_000_000_000)
    completion_tokens: int | None = Field(default=None, ge=0, le=1_000_000_000_000)
    total_tokens: int | None = Field(default=None, ge=0, le=1_000_000_000_000)
    cost: float | None = Field(default=None, ge=0, le=1_000_000_000, allow_inf_nan=False)

    def evidence_value(self) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], self.model_dump(mode="json", exclude_none=True))


class LLMCompletion(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    id: str | None = Field(default=None, min_length=1, max_length=500)
    model: str | None = Field(default=None, min_length=1, max_length=200)
    provider: str | None = Field(default=None, max_length=200)
    choices: tuple[_LLMResponseChoice, ...] = Field(min_length=1)
    usage: LLMUsage | None = None


class LLMPreflightHTTPError(ValueError):
    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(str(status_code))


class LLMProviderMismatchError(ValueError):
    pass


class LLMClient:
    def __init__(
        self,
        config: LLMClientConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=config.timeout_seconds,
            follow_redirects=False,
            trust_env=config.trust_environment_transport,
        )

    async def __aenter__(self) -> Self:
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

    def request_options(
        self,
        *,
        role: LLMRole,
        seed: int,
        top_p: float | None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        return self.config.request_options(
            role=role,
            seed=seed,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )

    async def complete(
        self,
        *,
        role: LLMRole,
        seed: int,
        top_p: float | None,
        schema_name: str,
        schema: dict[str, Any],
        strict_schema: bool,
        system_prompt: str,
        user_payload: str,
        max_output_tokens: int | None = None,
        preflight_error_body_limit: int = 0,
    ) -> LLMCompletion:
        api_key = self._require_live_access()
        request_body = {
            **self.request_options(
                role=role,
                seed=seed,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
            ),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": strict_schema,
                    "schema": schema,
                },
            },
            "stream": False,
        }
        endpoint = f"{self.config.base_url}/chat/completions"
        request_headers = {"Accept-Encoding": "identity", "Content-Type": "application/json"}
        if api_key is not None:
            request_headers["Authorization"] = f"Bearer {api_key}"
        async with asyncio.timeout(self.config.timeout_seconds):
            async with self._client.stream(
                "POST",
                endpoint,
                headers=request_headers,
                json=request_body,
                timeout=self.config.timeout_seconds,
                follow_redirects=False,
            ) as response:
                if not _same_origin(response.url, httpx.URL(endpoint)):
                    raise ValueError("LLM response changed request origin")
                if 300 <= response.status_code < 400:
                    raise ValueError("LLM redirects are not allowed")
                if 400 <= response.status_code < 500 and preflight_error_body_limit:
                    raise LLMPreflightHTTPError(
                        response.status_code,
                        await _read_response_prefix(
                            response,
                            maximum_bytes=preflight_error_body_limit,
                        ),
                    )
                response.raise_for_status()
                if response.headers.get("content-encoding", "identity").strip().lower() != (
                    "identity"
                ):
                    raise ValueError("LLM response Content-Encoding is not allowed")
                response_body = await _read_response(
                    response,
                    maximum_bytes=self.config.max_response_bytes,
                )
        completion = LLMCompletion.model_validate_json(response_body)
        parsed_value = completion.model_dump(mode="json")
        if _contains_secret(parsed_value, self.config.base_url):
            raise ValueError("LLM response contains the configured endpoint URL")
        if api_key is not None and _contains_secret(parsed_value, api_key):
            raise ValueError("LLM response contains the configured credential")
        self._validate_upstream_provider(completion)
        return completion

    def generation_metadata(self, response: LLMCompletion) -> dict[str, JsonValue]:
        if response.id is None or response.model is None:
            raise ValueError("LLM response is missing generation identity")
        usage = response.usage.evidence_value() if response.usage is not None else {}
        return {
            "semantic_provider": self.config.provider_id,
            "semantic_protocol": "openai-chat-completions",
            "semantic_endpoint_sha256": self.config.endpoint_sha256,
            "semantic_generation_id": response.id,
            "semantic_model": response.model,
            "semantic_upstream_provider": response.provider,
            "semantic_usage": usage,
        }

    def _require_live_access(self) -> str | None:
        if not self.config.live_calls:
            raise RuntimeError(
                "semantic model calls require UL_LIVE=true (or UL_DATASET_LIVE_CALLS=true)"
            )
        if not self.config.allow_external_data_processing:
            raise RuntimeError(
                "semantic model calls process raw inputs and outputs at the configured endpoint; "
                "set UL_LIVE=true (or UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING=true) to allow "
                "this"
            )
        api_key = (
            self.config.api_key.get_secret_value().strip()
            if self.config.api_key is not None
            else ""
        )
        if not api_key and self.config.api_key_required:
            raise RuntimeError(
                f"semantic model calls require {self.config.api_key_environment_variable}"
            )
        return api_key or None

    def _validate_upstream_provider(self, completion: LLMCompletion) -> None:
        configured_provider = self.config.upstream_provider
        if configured_provider is None:
            return
        if completion.provider is None or _normalized_provider_name(
            completion.provider
        ) != _normalized_provider_name(configured_provider):
            raise LLMProviderMismatchError(
                "LLM response did not honor the configured OpenRouter provider"
            )


def _validated_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("LLM base_url must use https or loopback http")
    if not parsed.hostname:
        raise ValueError("LLM base_url must include a host")
    try:
        _ = parsed.port
    except ValueError:
        raise ValueError("LLM base_url has an invalid port") from None
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("LLM base_url must not include credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("LLM base_url must not include a query or fragment")
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise ValueError("LLM base_url only permits plaintext HTTP on loopback")
    normalized_path = parsed.path.rstrip("/")
    if normalized_path.endswith("/chat/completions"):
        raise ValueError("LLM base_url must be an API root")
    return urlunsplit((parsed.scheme, parsed.netloc, normalized_path, "", ""))


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _normalized_provider_name(value: str) -> str:
    return value.strip().casefold().replace(" ", "-")


def _same_origin(left: httpx.URL, right: httpx.URL) -> bool:
    return (left.scheme, left.host, left.port) == (right.scheme, right.host, right.port)


async def _single_chunk(content: bytes) -> AsyncIterator[bytes]:
    yield content


async def _read_response(response: httpx.Response, *, maximum_bytes: int) -> bytes:
    chunks: list[bytes] = []
    response_size = 0
    response_chunks = (
        _single_chunk(response.content) if response.is_stream_consumed else response.aiter_raw()
    )
    async for chunk in response_chunks:
        response_size += len(chunk)
        if response_size > maximum_bytes:
            raise ValueError("LLM response exceeds max_response_bytes")
        chunks.append(chunk)
    return b"".join(chunks)


async def _read_response_prefix(response: httpx.Response, *, maximum_bytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = maximum_bytes
    response_chunks = (
        _single_chunk(response.content) if response.is_stream_consumed else response.aiter_raw()
    )
    async for chunk in response_chunks:
        if remaining <= 0:
            break
        retained = chunk[:remaining]
        chunks.append(retained)
        remaining -= len(retained)
        if len(chunk) > len(retained):
            break
    return b"".join(chunks)


def _contains_secret(value: object, secret: str) -> bool:
    if isinstance(value, str):
        return secret in value
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        return any(_contains_secret(item, secret) for item in mapping.values())
    if isinstance(value, list):
        sequence = cast(list[object], value)
        return any(_contains_secret(item, secret) for item in sequence)
    return False
