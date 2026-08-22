from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from types import TracebackType
from typing import Any, Literal, Protocol, Self, cast
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SecretStr,
    ValidationError,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from ul_core.dataset import (
    EvidenceReference,
    InteractionRecord,
    RenderedUserInput,
    SemanticEquivalenceAssessment,
    SemanticFactor,
    SemanticFrame,
    UserInputRecord,
)
from ul_core.prompts import PromptManager, prompt_provenance

_PROMPTS = PromptManager.instance()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class ProviderDiagnostic(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    operation: Literal["deconstruct", "render", "verify", "preflight"]
    category: Literal[
        "authentication",
        "bad_request",
        "connection",
        "invalid_response",
        "provider_unavailable",
        "rate_limit",
        "timeout",
    ]
    retryable: bool
    retry_status: Literal["not_retried"] = "not_retried"
    suggested_action: str
    endpoint_sha256: str
    http_status: int | None = None


class ProviderDiagnosticError(RuntimeError):
    def __init__(self, diagnostic: ProviderDiagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__(
            f"Semantic provider {diagnostic.provider} failed during {diagnostic.operation} "
            f"({diagnostic.category}; retryable: {'yes' if diagnostic.retryable else 'no'}). "
            f"Next: {diagnostic.suggested_action}"
        )


def _provider_diagnostic(
    error: BaseException,
    *,
    provider: str,
    operation: Literal["deconstruct", "render", "verify", "preflight"],
    endpoint_sha256: str,
) -> ProviderDiagnostic:
    http_status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
    if http_status in {401, 403}:
        category = "authentication"
        retryable = False
        suggested_action = "check the provider credential and account access, then retry."
    elif http_status == 429:
        category = "rate_limit"
        retryable = True
        suggested_action = "wait for the provider rate limit to reset, then resume the run."
    elif http_status == 408 or isinstance(error, (TimeoutError, httpx.TimeoutException)):
        category = "timeout"
        retryable = True
        suggested_action = "check provider availability, then resume the run."
    elif isinstance(error, httpx.RequestError):
        category = "connection"
        retryable = True
        suggested_action = "check provider connectivity, then resume the run."
    elif http_status is not None and http_status >= 500:
        category = "provider_unavailable"
        retryable = True
        suggested_action = "check provider status, then resume the run."
    elif http_status is not None:
        category = "bad_request"
        retryable = False
        suggested_action = "check the configured provider and model settings before retrying."
    else:
        category = "invalid_response"
        retryable = False
        suggested_action = "check that the provider supports the configured response format."
    return ProviderDiagnostic(
        provider=provider,
        operation=operation,
        category=category,
        retryable=retryable,
        suggested_action=suggested_action,
        endpoint_sha256=endpoint_sha256,
        http_status=http_status,
    )


class OpenRouterDatasetSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    live_calls: bool = Field(default=False, validation_alias="UL_DATASET_LIVE_CALLS")
    allow_external_data_processing: bool = Field(
        default=False,
        validation_alias="UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING",
    )
    api_key: SecretStr | None = Field(default=None, validation_alias="OPEN_ROUTER_API_KEY")
    ul_live: bool = Field(default=False, validation_alias="UL_LIVE", exclude=True, repr=False)

    @model_validator(mode="after")
    def apply_ul_live_shorthand(self) -> Self:
        if self.ul_live:
            if "live_calls" not in self.model_fields_set:
                self.live_calls = True
            if "allow_external_data_processing" not in self.model_fields_set:
                self.allow_external_data_processing = True
        return self

    model: str = Field(
        default="google/gemini-2.5-flash",
        min_length=1,
        max_length=200,
        validation_alias="UL_DATASET_MODEL",
    )
    render_model: str = Field(
        default="x-ai/grok-4.3",
        min_length=1,
        max_length=200,
        validation_alias="UL_DATASET_RENDER_MODEL",
    )
    equivalence_model: str = Field(
        default="google/gemini-3.5-flash",
        min_length=1,
        max_length=200,
        validation_alias="UL_DATASET_EQUIVALENCE_MODEL",
    )
    max_input_chars: int = Field(
        default=50_000,
        ge=1,
        le=1_000_000,
        validation_alias="UL_DATASET_MAX_INPUT_CHARS",
    )
    max_output_tokens: int = Field(
        default=4_096,
        ge=1,
        le=32_768,
        validation_alias="UL_DATASET_MAX_OUTPUT_TOKENS",
    )
    max_render_tokens: int = Field(
        default=512,
        ge=1,
        le=4_096,
        validation_alias="UL_DATASET_MAX_RENDER_TOKENS",
    )
    max_response_bytes: int = Field(
        default=1_000_000,
        ge=1_024,
        le=5_000_000,
        validation_alias="UL_DATASET_MAX_RESPONSE_BYTES",
    )
    timeout_seconds: float = Field(
        default=60,
        gt=0,
        le=300,
        validation_alias="UL_DATASET_TIMEOUT_SECONDS",
    )

    @property
    def semantic_provider_id(self) -> str:
        return "openrouter"

    @property
    def semantic_provider_type(self) -> Literal["openrouter", "openai-compatible"]:
        return "openrouter"

    @property
    def semantic_base_url(self) -> str:
        return "https://openrouter.ai/api/v1"

    @property
    def semantic_endpoint_sha256(self) -> str:
        return _sha256_text(self.semantic_base_url)

    @property
    def api_key_environment_variable(self) -> str:
        return "OPEN_ROUTER_API_KEY"

    @property
    def api_key_required(self) -> bool:
        return True


class OpenAICompatibleDatasetSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
        populate_by_name=True,
    )

    live_calls: bool = Field(default=False, validation_alias="UL_DATASET_LIVE_CALLS")
    allow_external_data_processing: bool = Field(
        default=False,
        validation_alias="UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING",
    )
    api_key: SecretStr | None = Field(
        default=None,
        validation_alias="UL_DATASET_OPENAI_API_KEY",
    )
    ul_live: bool = Field(default=False, validation_alias="UL_LIVE", exclude=True, repr=False)
    provider_id: str = Field(
        default="openai-compatible",
        min_length=1,
        max_length=100,
        pattern=r"[a-z0-9][a-z0-9._-]*",
        validation_alias="UL_DATASET_OPENAI_PROVIDER_ID",
    )
    base_url: str = Field(
        default="",
        min_length=1,
        max_length=2_000,
        validation_alias="UL_DATASET_OPENAI_BASE_URL",
    )
    model: str = Field(
        default="",
        min_length=1,
        max_length=200,
        validation_alias="UL_DATASET_MODEL",
    )
    render_model: str = Field(
        default="",
        max_length=200,
        validation_alias="UL_DATASET_RENDER_MODEL",
    )
    equivalence_model: str = Field(
        default="",
        max_length=200,
        validation_alias="UL_DATASET_EQUIVALENCE_MODEL",
    )
    max_input_chars: int = Field(
        default=50_000,
        ge=1,
        le=1_000_000,
        validation_alias="UL_DATASET_MAX_INPUT_CHARS",
    )
    max_output_tokens: int = Field(
        default=4_096,
        ge=1,
        le=32_768,
        validation_alias="UL_DATASET_MAX_OUTPUT_TOKENS",
    )
    max_render_tokens: int = Field(
        default=512,
        ge=1,
        le=4_096,
        validation_alias="UL_DATASET_MAX_RENDER_TOKENS",
    )
    max_response_bytes: int = Field(
        default=1_000_000,
        ge=1_024,
        le=5_000_000,
        validation_alias="UL_DATASET_MAX_RESPONSE_BYTES",
    )
    timeout_seconds: float = Field(
        default=60,
        gt=0,
        le=300,
        validation_alias="UL_DATASET_TIMEOUT_SECONDS",
    )

    @model_validator(mode="after")
    def validate_and_normalize(self) -> Self:
        if self.ul_live:
            if "live_calls" not in self.model_fields_set:
                self.live_calls = True
            if "allow_external_data_processing" not in self.model_fields_set:
                self.allow_external_data_processing = True
        self.base_url = _validated_openai_compatible_base_url(self.base_url)
        if self.provider_id == "openrouter":
            raise ValueError("UL_DATASET_OPENAI_PROVIDER_ID cannot use the reserved openrouter ID")
        if not self.model.strip():
            raise ValueError("UL_DATASET_MODEL must contain non-whitespace text")
        if not self.render_model:
            self.render_model = self.model
        elif not self.render_model.strip():
            raise ValueError("UL_DATASET_RENDER_MODEL must contain non-whitespace text")
        if not self.equivalence_model:
            self.equivalence_model = self.model
        elif not self.equivalence_model.strip():
            raise ValueError("UL_DATASET_EQUIVALENCE_MODEL must contain non-whitespace text")
        return self

    @property
    def semantic_provider_id(self) -> str:
        return self.provider_id

    @property
    def semantic_provider_type(self) -> Literal["openrouter", "openai-compatible"]:
        return "openai-compatible"

    @property
    def semantic_base_url(self) -> str:
        return self.base_url

    @property
    def semantic_endpoint_sha256(self) -> str:
        return _sha256_text(self.semantic_base_url)

    @property
    def api_key_environment_variable(self) -> str:
        return "UL_DATASET_OPENAI_API_KEY"

    @property
    def api_key_required(self) -> bool:
        return False


class _DatasetSemanticProviderSelection(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    provider: Literal["openrouter", "openai-compatible"] = Field(
        default="openrouter",
        validation_alias="UL_DATASET_SEMANTIC_PROVIDER",
    )


type DatasetSemanticSettings = OpenRouterDatasetSettings | OpenAICompatibleDatasetSettings


def load_dataset_semantic_settings() -> DatasetSemanticSettings:
    try:
        selection = _DatasetSemanticProviderSelection()
    except ValidationError as error:
        raise _semantic_configuration_error(error, provider="selection") from None
    if selection.provider == "openai-compatible":
        try:
            return OpenAICompatibleDatasetSettings()
        except ValidationError as error:
            raise _semantic_configuration_error(error, provider="openai-compatible") from None
    try:
        return OpenRouterDatasetSettings()
    except ValidationError as error:
        raise _semantic_configuration_error(error, provider="openrouter") from None


def _semantic_configuration_error(
    error: ValidationError,
    *,
    provider: Literal["selection", "openrouter", "openai-compatible"],
) -> ValueError:
    first_error = error.errors(include_input=False, include_url=False)[0]
    location = first_error["loc"]
    field_name = str(location[0]) if location else ""
    field_name = {
        "UL_DATASET_SEMANTIC_PROVIDER": "provider",
        "UL_DATASET_OPENAI_BASE_URL": "base_url",
        "UL_DATASET_OPENAI_PROVIDER_ID": "provider_id",
        "UL_DATASET_MODEL": "model",
        "UL_DATASET_RENDER_MODEL": "render_model",
        "UL_DATASET_EQUIVALENCE_MODEL": "equivalence_model",
        "UL_DATASET_MAX_INPUT_CHARS": "max_input_chars",
        "UL_DATASET_MAX_OUTPUT_TOKENS": "max_output_tokens",
        "UL_DATASET_MAX_RENDER_TOKENS": "max_render_tokens",
        "UL_DATASET_MAX_RESPONSE_BYTES": "max_response_bytes",
        "UL_DATASET_TIMEOUT_SECONDS": "timeout_seconds",
        "UL_DATASET_LIVE_CALLS": "live_calls",
        "UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING": "allow_external_data_processing",
        "UL_LIVE": "ul_live",
    }.get(field_name, field_name)
    if not field_name:
        safe_detail = str(first_error.get("ctx", {}).get("error", ""))
        field_name = next(
            (
                candidate
                for environment_name, candidate in {
                    "UL_DATASET_OPENAI_BASE_URL": "base_url",
                    "UL_DATASET_OPENAI_PROVIDER_ID": "provider_id",
                    "UL_DATASET_RENDER_MODEL": "render_model",
                    "UL_DATASET_EQUIVALENCE_MODEL": "equivalence_model",
                    "UL_DATASET_MODEL": "model",
                }.items()
                if environment_name in safe_detail
            ),
            "",
        )
    messages = {
        "provider": "UL_DATASET_SEMANTIC_PROVIDER must be openrouter or openai-compatible",
        "base_url": (
            "UL_DATASET_OPENAI_BASE_URL must be an HTTPS API root without credentials, query, "
            "fragment, or /chat/completions; loopback HTTP is allowed"
        ),
        "provider_id": (
            "UL_DATASET_OPENAI_PROVIDER_ID must be 1-100 lowercase letters, digits, dots, "
            "underscores, or hyphens and must not be openrouter"
        ),
        "model": "UL_DATASET_MODEL must be 1-200 non-whitespace characters",
        "render_model": (
            "UL_DATASET_RENDER_MODEL must be 1-200 non-whitespace characters when set"
        ),
        "equivalence_model": (
            "UL_DATASET_EQUIVALENCE_MODEL must be 1-200 non-whitespace characters when set"
        ),
        "max_input_chars": "UL_DATASET_MAX_INPUT_CHARS must be between 1 and 1000000",
        "max_output_tokens": "UL_DATASET_MAX_OUTPUT_TOKENS must be between 1 and 32768",
        "max_render_tokens": "UL_DATASET_MAX_RENDER_TOKENS must be between 1 and 4096",
        "max_response_bytes": ("UL_DATASET_MAX_RESPONSE_BYTES must be between 1024 and 5000000"),
        "timeout_seconds": "UL_DATASET_TIMEOUT_SECONDS must be greater than 0 and at most 300",
        "live_calls": "UL_DATASET_LIVE_CALLS must be true or false",
        "allow_external_data_processing": (
            "UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING must be true or false"
        ),
        "ul_live": "UL_LIVE must be true or false",
    }
    fallback = (
        "OpenRouter semantic provider configuration is invalid"
        if provider == "openrouter"
        else "OpenAI-compatible semantic provider configuration is invalid"
    )
    return ValueError(messages.get(field_name, fallback))


def _validated_openai_compatible_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("UL_DATASET_OPENAI_BASE_URL must use https or loopback http")
    if not parsed.hostname:
        raise ValueError("UL_DATASET_OPENAI_BASE_URL must include a host")
    try:
        _ = parsed.port
    except ValueError:
        raise ValueError("UL_DATASET_OPENAI_BASE_URL has an invalid port") from None
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("UL_DATASET_OPENAI_BASE_URL must not include credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("UL_DATASET_OPENAI_BASE_URL must not include a query or fragment")
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise ValueError("UL_DATASET_OPENAI_BASE_URL only permits plaintext HTTP on loopback")
    normalized_path = parsed.path.rstrip("/")
    if normalized_path.endswith("/chat/completions"):
        raise ValueError(
            "UL_DATASET_OPENAI_BASE_URL must be an API root, not a chat-completions endpoint"
        )
    return urlunsplit((parsed.scheme, parsed.netloc, normalized_path, "", ""))


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class _ResponseMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    content: str


class _ResponseChoice(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    message: _ResponseMessage


class _UsageMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, hide_input_in_errors=True)

    prompt_tokens: int | None = Field(default=None, ge=0, le=1_000_000_000_000)
    completion_tokens: int | None = Field(default=None, ge=0, le=1_000_000_000_000)
    total_tokens: int | None = Field(default=None, ge=0, le=1_000_000_000_000)
    cost: float | None = Field(
        default=None,
        ge=0,
        le=1_000_000_000,
        allow_inf_nan=False,
    )

    def evidence_value(self) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], self.model_dump(mode="json", exclude_none=True))


class _ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    id: str = Field(min_length=1, max_length=500)
    model: str = Field(min_length=1, max_length=200)
    provider: str | None = Field(default=None, max_length=200)
    choices: tuple[_ResponseChoice, ...] = Field(min_length=1)
    usage: _UsageMetadata | None = None


class _RenderedInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rendered_input: str = Field(min_length=1)


class _EvaluatorPreflightSample(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    compatible: Literal[True]


class EvaluatorModelProfilePreflight(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    roles: tuple[Literal["deconstruct", "render", "equivalence"], ...] = Field(min_length=1)
    requested_model: str = Field(min_length=1, max_length=200)
    routed_model: str = Field(min_length=1, max_length=200)
    upstream_provider: str | None = Field(default=None, max_length=200)
    required_parameters: tuple[str, ...]
    parameter_support: Literal["routing_enforced", "endpoint_accepted_unverified"]
    unverified_options: tuple[str, ...] = ()


class EvaluatorModelPreflight(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    provider: str = Field(min_length=1, max_length=100)
    profiles: tuple[EvaluatorModelProfilePreflight, ...] = Field(min_length=1)
    verified_capabilities: tuple[
        Literal["routing", "structured_output", "required_parameters"], ...
    ]
    ignored_or_unsupported_options: tuple[str, ...] = ()
    unverified_options: tuple[str, ...] = ()
    data_policy: dict[str, JsonValue]


class EvaluatorModelCompatibilityError(ValueError):
    pass


class _EvaluatorPreflightHTTPError(ValueError):
    def __init__(self, capability: str) -> None:
        self.capability = capability
        super().__init__(capability)


@dataclass(frozen=True)
class _EvaluatorPreflightProfile:
    roles: tuple[Literal["deconstruct", "render", "equivalence"], ...]
    model: str
    reasoning: dict[str, JsonValue]
    max_tokens: int
    temperature: float
    seed: int
    top_p: float | None
    required_parameters: tuple[str, ...]


class SemanticCompletionProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    @property
    def base_url(self) -> str: ...

    @property
    def endpoint_sha256(self) -> str: ...

    @property
    def extractor_version(self) -> str: ...

    @property
    def equivalence_verifier_version(self) -> str: ...

    @property
    def requires_api_key(self) -> bool: ...

    @property
    def trust_environment_transport(self) -> bool: ...

    def add_request_options(
        self,
        request_body: dict[str, Any],
        reasoning: dict[str, JsonValue],
    ) -> None: ...

    def generation_metadata(
        self,
        response: _ChatCompletionResponse,
    ) -> dict[str, JsonValue]: ...

    def preflight_data_policy(self) -> dict[str, JsonValue]: ...

    @property
    def enforces_parameter_support(self) -> bool: ...


@dataclass(frozen=True)
class OpenAICompatibleSemanticProvider:
    provider_id: str
    base_url: str
    endpoint_sha256: str
    extractor_version: str = "semantic-deconstructor/2.0.0"
    equivalence_verifier_version: str = "semantic-equivalence-verifier/2.0.0"
    requires_api_key: bool = False
    trust_environment_transport: bool = False

    @property
    def enforces_parameter_support(self) -> bool:
        return False

    def add_request_options(
        self,
        request_body: dict[str, Any],
        reasoning: dict[str, JsonValue],
    ) -> None:
        return None

    def generation_metadata(
        self,
        response: _ChatCompletionResponse,
    ) -> dict[str, JsonValue]:
        usage = response.usage.evidence_value() if response.usage is not None else {}
        return {
            "semantic_provider": self.provider_id,
            "semantic_protocol": "openai-chat-completions",
            "semantic_endpoint_sha256": self.endpoint_sha256,
            "semantic_generation_id": response.id,
            "semantic_model": response.model,
            "semantic_upstream_provider": response.provider,
            "semantic_usage": usage,
        }

    def preflight_data_policy(self) -> dict[str, JsonValue]:
        return {
            "external_processing": True,
            "provider_policy_declared": False,
            "implication": (
                "The configured endpoint receives evaluator prompts and sample data; UL cannot "
                "verify its retention or training policy."
            ),
        }


@dataclass(frozen=True)
class OpenRouterSemanticProvider(OpenAICompatibleSemanticProvider):
    provider_id: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"
    endpoint_sha256: str = _sha256_text("https://openrouter.ai/api/v1")
    requires_api_key: bool = True
    trust_environment_transport: bool = True

    @property
    def enforces_parameter_support(self) -> bool:
        return True

    def add_request_options(
        self,
        request_body: dict[str, Any],
        reasoning: dict[str, JsonValue],
    ) -> None:
        request_body["reasoning"] = reasoning
        request_body["provider"] = {
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
        }

    def preflight_data_policy(self) -> dict[str, JsonValue]:
        return {
            "external_processing": True,
            "provider_policy_declared": True,
            "data_collection": "deny",
            "zero_data_retention_required": True,
            "implication": (
                "The configured route requires data collection to be denied and zero data "
                "retention; the evaluator request is still processed externally."
            ),
        }


class SemanticModelDeconstructor:
    def __init__(
        self,
        settings: DatasetSemanticSettings,
        provider: SemanticCompletionProvider,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=self.settings.timeout_seconds,
            follow_redirects=False,
            trust_env=self.provider.trust_environment_transport,
        )
        self._preflight_result: EvaluatorModelPreflight | None = None

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

    async def preflight(self) -> EvaluatorModelPreflight:
        profile_results: list[EvaluatorModelProfilePreflight] = []
        for profile in self._preflight_profiles():
            try:
                response = await self._request(
                    operation="preflight",
                    model=profile.model,
                    reasoning=profile.reasoning,
                    max_tokens=profile.max_tokens,
                    temperature=profile.temperature,
                    seed=profile.seed,
                    top_p=profile.top_p,
                    schema_name="evaluator_preflight",
                    schema=_EvaluatorPreflightSample.model_json_schema(mode="validation"),
                    strict_schema=True,
                    system_prompt=_PROMPTS.get_prompt("semantic.preflight"),
                    untrusted_payload="{}",
                    preflight_error_body_limit=4_096,
                )
            except TimeoutError:
                raise self._compatibility_error(profile, "timeout") from None
            except _EvaluatorPreflightHTTPError as error:
                raise self._compatibility_error(profile, error.capability) from None
            except ProviderDiagnosticError as error:
                capability = {
                    "timeout": "timeout",
                    "invalid_response": "structured output",
                }.get(error.diagnostic.category, "provider routing")
                raise self._compatibility_error(profile, capability) from None
            except httpx.HTTPError:
                raise self._compatibility_error(profile, "routing") from None
            try:
                _EvaluatorPreflightSample.model_validate_json(response.choices[0].message.content)
            except (ValidationError, ValueError):
                raise self._compatibility_error(profile, "structured output") from None
            routing_enforced = self.provider.enforces_parameter_support
            profile_results.append(
                EvaluatorModelProfilePreflight(
                    roles=profile.roles,
                    requested_model=profile.model,
                    routed_model=response.model,
                    upstream_provider=response.provider,
                    required_parameters=profile.required_parameters,
                    parameter_support=(
                        "routing_enforced" if routing_enforced else "endpoint_accepted_unverified"
                    ),
                    unverified_options=() if routing_enforced else profile.required_parameters,
                )
            )
        unverified_options = tuple(
            dict.fromkeys(
                option
                for profile_result in profile_results
                for option in profile_result.unverified_options
            )
        )
        self._preflight_result = EvaluatorModelPreflight(
            provider=self.provider.provider_id,
            profiles=tuple(profile_results),
            verified_capabilities=(
                ("routing", "structured_output", "required_parameters")
                if self.provider.enforces_parameter_support
                else ("routing", "structured_output")
            ),
            unverified_options=unverified_options,
            data_policy=self.provider.preflight_data_policy(),
        )
        return self._preflight_result

    def reuse_preflight(self, result: EvaluatorModelPreflight) -> None:
        expected_profiles = self._preflight_profiles()
        expected_roles_and_models = tuple(
            (profile.roles, profile.model, profile.required_parameters)
            for profile in expected_profiles
        )
        actual_roles_and_models = tuple(
            (profile.roles, profile.requested_model, profile.required_parameters)
            for profile in result.profiles
        )
        if (
            result.provider != self.provider.provider_id
            or actual_roles_and_models != expected_roles_and_models
        ):
            raise ValueError("evaluator preflight does not match the configured semantic profiles")
        self._preflight_result = result

    def _preflight_profiles(self) -> tuple[_EvaluatorPreflightProfile, ...]:
        render_seed = self._render_seed("UL evaluator preflight", "Check renderer compatibility.")
        candidates = (
            self._preflight_profile(
                role="deconstruct",
                model=self.settings.model,
                reasoning={"effort": "minimal"},
                max_tokens=min(self.settings.max_output_tokens, 16),
                temperature=0,
                seed=0,
                top_p=None,
            ),
            self._preflight_profile(
                role="render",
                model=self.settings.render_model,
                reasoning={"effort": "none"},
                max_tokens=min(self.settings.max_render_tokens, 16),
                temperature=0.7,
                seed=render_seed,
                top_p=0.95,
            ),
            self._preflight_profile(
                role="equivalence",
                model=self.settings.equivalence_model,
                reasoning={"effort": "low"},
                max_tokens=min(self.settings.max_output_tokens, 1_024, 16),
                temperature=0,
                seed=0,
                top_p=None,
            ),
        )
        profiles_by_signature: dict[str, _EvaluatorPreflightProfile] = {}
        for candidate in candidates:
            signature_body: dict[str, Any] = {
                "model": candidate.model,
                "max_tokens": candidate.max_tokens,
                "temperature": candidate.temperature,
                "seed": candidate.seed,
            }
            self.provider.add_request_options(signature_body, candidate.reasoning)
            if candidate.top_p is not None:
                signature_body["top_p"] = candidate.top_p
            signature = json.dumps(signature_body, sort_keys=True, separators=(",", ":"))
            existing = profiles_by_signature.get(signature)
            if existing is None:
                profiles_by_signature[signature] = candidate
            else:
                profiles_by_signature[signature] = dataclass_replace(
                    existing,
                    roles=(*existing.roles, *candidate.roles),
                )
        return tuple(profiles_by_signature.values())

    def _preflight_profile(
        self,
        *,
        role: Literal["deconstruct", "render", "equivalence"],
        model: str,
        reasoning: dict[str, JsonValue],
        max_tokens: int,
        temperature: float,
        seed: int,
        top_p: float | None,
    ) -> _EvaluatorPreflightProfile:
        request_options: dict[str, Any] = {}
        self.provider.add_request_options(request_options, reasoning)
        required_parameters = ["response_format", "seed", "temperature", "max_tokens"]
        if "reasoning" in request_options:
            required_parameters.append("reasoning")
        if top_p is not None:
            required_parameters.append("top_p")
        return _EvaluatorPreflightProfile(
            roles=(role,),
            model=model,
            reasoning=reasoning,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            required_parameters=tuple(required_parameters),
        )

    def _compatibility_error(
        self,
        profile: _EvaluatorPreflightProfile,
        capability: str,
    ) -> EvaluatorModelCompatibilityError:
        roles = ", ".join(profile.roles)
        return EvaluatorModelCompatibilityError(
            f"semantic model profile for {roles} is incompatible with required {capability} "
            "capability; "
            "choose another configured evaluator model or verify the configured route and retry"
        )

    async def deconstruct(
        self,
        record: InteractionRecord | UserInputRecord,
        reference_frame: SemanticFrame | None = None,
    ) -> SemanticFrame:
        observed_output = (
            record.raw_observed_output if isinstance(record, InteractionRecord) else None
        )
        request_payload: dict[str, JsonValue] = {
            "raw_input": record.raw_input,
            "raw_observed_output": observed_output,
        }
        if reference_frame is not None:
            request_payload["reference_vocabulary"] = self._reference_vocabulary(reference_frame)
        untrusted_record = self._bounded_json(request_payload)
        response = await self._request(
            operation="deconstruct",
            model=self.settings.model,
            reasoning={"effort": "minimal"},
            max_tokens=self.settings.max_output_tokens,
            temperature=0,
            seed=0,
            top_p=None,
            schema_name="semantic_frame",
            schema=SemanticFrame.model_json_schema(mode="validation"),
            strict_schema=True,
            system_prompt=_PROMPTS.get_prompt("semantic.deconstruct"),
            untrusted_payload=untrusted_record,
        )
        try:
            raw_frame = self._decode_object(response.choices[0].message.content)
            raw_frame.update(
                {
                    "schema_version": "1.0.0",
                    "interaction_id": record.id,
                    "extractor_version": self.provider.extractor_version,
                    "metadata": {
                        **self._generation_metadata(response),
                        "prompts": prompt_provenance("semantic.deconstruct"),
                    },
                }
            )
            frame = SemanticFrame.model_validate_json(json.dumps(raw_frame))
        except (ValidationError, ValueError) as error:
            raise self._invalid_response(error, operation="deconstruct") from None
        frame = self._expand_unambiguous_evidence_quotes(record, frame)
        frame = self._ground_self_correction_evidence(record, frame)
        self._validate_evidence(record, frame)
        return frame

    async def render(
        self,
        raw_input: str,
        instruction: str,
        *,
        allow_temporary_value: bool = False,
    ) -> RenderedUserInput:
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        untrusted_payload = self._bounded_json(
            {"raw_input": raw_input, "transformation_instruction": instruction}
        )
        render_seed = self._render_seed(raw_input, instruction)
        temporary_value_prompt = (
            "semantic.render.temporary_value_allowed"
            if allow_temporary_value
            else "semantic.render.temporary_value_forbidden"
        )
        temporary_value_rule = _PROMPTS.get_prompt(temporary_value_prompt)
        response = await self._request(
            operation="render",
            model=self.settings.render_model,
            reasoning={"effort": "none"},
            max_tokens=self.settings.max_render_tokens,
            temperature=0.7,
            seed=render_seed,
            top_p=0.95,
            schema_name="rendered_input",
            schema=_RenderedInput.model_json_schema(mode="validation"),
            strict_schema=True,
            system_prompt=_PROMPTS.get_prompt(
                "semantic.render",
                temporary_value_rule=temporary_value_rule,
            ),
            untrusted_payload=untrusted_payload,
        )
        try:
            rendered = _RenderedInput.model_validate_json(
                response.choices[0].message.content
            ).rendered_input
        except (ValidationError, ValueError) as error:
            raise self._invalid_response(error, operation="render") from None
        if len(rendered) > self.settings.max_input_chars:
            raise ValueError("rendered input exceeds max_input_chars")
        return RenderedUserInput(
            text=rendered,
            metadata={
                **self._generation_metadata(response),
                "requested_model": self.settings.render_model,
                "prompts": prompt_provenance("semantic.render", temporary_value_prompt),
                "sampling": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "seed": render_seed,
                    "max_tokens": self.settings.max_render_tokens,
                },
            },
        )

    async def verify(
        self,
        source_input: str,
        candidate_input: str,
    ) -> SemanticEquivalenceAssessment:
        untrusted_payload = self._bounded_json(
            {"source_input": source_input, "candidate_input": candidate_input}
        )
        response = await self._request(
            operation="verify",
            model=self.settings.equivalence_model,
            reasoning={"effort": "low"},
            max_tokens=min(self.settings.max_output_tokens, 1_024),
            temperature=0,
            seed=0,
            top_p=None,
            schema_name="semantic_equivalence_assessment",
            schema=SemanticEquivalenceAssessment.model_json_schema(mode="validation"),
            strict_schema=True,
            system_prompt=_PROMPTS.get_prompt("semantic.verify"),
            untrusted_payload=untrusted_payload,
        )
        try:
            raw_assessment = self._decode_object(response.choices[0].message.content)
            raw_assessment.update(
                {
                    "schema_version": "1.0.0",
                    "verifier_version": self.provider.equivalence_verifier_version,
                    "metadata": {
                        **self._generation_metadata(response),
                        "requested_model": self.settings.equivalence_model,
                        "prompts": prompt_provenance("semantic.verify"),
                    },
                }
            )
            assessment = SemanticEquivalenceAssessment.model_validate_json(
                json.dumps(raw_assessment)
            )
        except (ValidationError, ValueError) as error:
            raise self._invalid_response(error, operation="verify") from None
        assessment = assessment.model_copy(
            update={
                "deltas": tuple(
                    delta.model_copy(
                        update={
                            "source_quote": (
                                delta.source_quote.strip()
                                if delta.source_quote is not None
                                else None
                            ),
                            "candidate_quote": (
                                delta.candidate_quote.strip()
                                if delta.candidate_quote is not None
                                else None
                            ),
                        }
                    )
                    for delta in assessment.deltas
                )
            }
        )
        for delta in assessment.deltas:
            if delta.source_quote is not None and (
                not delta.source_quote or delta.source_quote not in source_input
            ):
                raise ValueError("semantic equivalence source evidence is invalid")
            if delta.candidate_quote is not None and (
                not delta.candidate_quote or delta.candidate_quote not in candidate_input
            ):
                raise ValueError("semantic equivalence candidate evidence is invalid")
        return assessment

    def _invalid_response(
        self,
        error: BaseException,
        *,
        operation: Literal["deconstruct", "render", "verify"],
    ) -> ProviderDiagnosticError:
        return ProviderDiagnosticError(
            _provider_diagnostic(
                error,
                provider=self.provider.provider_id,
                operation=operation,
                endpoint_sha256=self.provider.endpoint_sha256,
            )
        )

    async def _request(
        self,
        *,
        operation: Literal["deconstruct", "render", "verify", "preflight"],
        model: str,
        reasoning: dict[str, JsonValue],
        max_tokens: int,
        temperature: float,
        seed: int,
        top_p: float | None,
        schema_name: str,
        schema: dict[str, Any],
        strict_schema: bool,
        system_prompt: str,
        untrusted_payload: str,
        preflight_error_body_limit: int = 0,
    ) -> _ChatCompletionResponse:
        api_key = self._require_live_access()
        try:
            return await self._request_completion(
                api_key=api_key,
                model=model,
                reasoning=reasoning,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                top_p=top_p,
                schema_name=schema_name,
                schema=schema,
                strict_schema=strict_schema,
                system_prompt=system_prompt,
                untrusted_payload=untrusted_payload,
                preflight_error_body_limit=preflight_error_body_limit,
            )
        except (
            TimeoutError,
            ValidationError,
            httpx.RequestError,
            httpx.HTTPStatusError,
        ) as error:
            raise ProviderDiagnosticError(
                _provider_diagnostic(
                    error,
                    provider=self.provider.provider_id,
                    operation=operation,
                    endpoint_sha256=self.provider.endpoint_sha256,
                )
            ) from None

    async def _request_completion(
        self,
        *,
        api_key: str | None,
        model: str,
        reasoning: dict[str, JsonValue],
        max_tokens: int,
        temperature: float,
        seed: int,
        top_p: float | None,
        schema_name: str,
        schema: dict[str, Any],
        strict_schema: bool,
        system_prompt: str,
        untrusted_payload: str,
        preflight_error_body_limit: int = 0,
    ) -> _ChatCompletionResponse:
        async with asyncio.timeout(self.settings.timeout_seconds):
            request_body: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": untrusted_payload},
                ],
                "temperature": temperature,
                "seed": seed,
                "max_tokens": max_tokens,
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
            self.provider.add_request_options(request_body, reasoning)
            if top_p is not None:
                request_body["top_p"] = top_p
            endpoint = f"{self.provider.base_url}/chat/completions"
            request_headers = {"Accept-Encoding": "identity"}
            if api_key is not None:
                request_headers["Authorization"] = f"Bearer {api_key}"
            async with self._client.stream(
                "POST",
                endpoint,
                headers=request_headers,
                json=request_body,
                timeout=self.settings.timeout_seconds,
                follow_redirects=False,
            ) as response:
                if not _same_origin(response.url, httpx.URL(endpoint)):
                    raise ValueError("semantic provider response changed request origin")
                if 300 <= response.status_code < 400:
                    raise ValueError("semantic provider redirects are not allowed")
                if 400 <= response.status_code < 500 and preflight_error_body_limit:
                    error_body = await _read_bounded_response(
                        response,
                        maximum_bytes=preflight_error_body_limit,
                    )
                    raise _EvaluatorPreflightHTTPError(
                        _preflight_http_capability(response.status_code, error_body)
                    )
                response.raise_for_status()
                content_encoding = response.headers.get("content-encoding", "identity")
                if content_encoding.strip().lower() != "identity":
                    raise ValueError("semantic provider response Content-Encoding is not allowed")
                chunks: list[bytes] = []
                response_size = 0
                response_chunks = (
                    _single_chunk(response.content)
                    if response.is_stream_consumed
                    else response.aiter_raw()
                )
                async for chunk in response_chunks:
                    response_size += len(chunk)
                    if response_size > self.settings.max_response_bytes:
                        raise ValueError("semantic provider response exceeds max_response_bytes")
                    chunks.append(chunk)
        completion_response = _ChatCompletionResponse.model_validate_json(b"".join(chunks))
        if _contains_secret(
            completion_response.model_dump(mode="json"), self.settings.semantic_base_url
        ):
            raise ValueError("semantic provider response contains the configured endpoint URL")
        if api_key is not None and _contains_secret(
            completion_response.model_dump(mode="json"), api_key
        ):
            raise ValueError("semantic provider response contains the configured credential")
        return completion_response

    @staticmethod
    def _render_seed(raw_input: str, instruction: str) -> int:
        digest = hashlib.sha256(f"{raw_input}\0{instruction}".encode()).digest()
        return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF

    def _require_live_access(self) -> str | None:
        if not self.settings.live_calls:
            raise RuntimeError(
                "semantic model calls require UL_LIVE=true (or UL_DATASET_LIVE_CALLS=true)"
            )
        if not self.settings.allow_external_data_processing:
            raise RuntimeError(
                "semantic model calls process raw inputs and outputs at the configured endpoint; "
                "set "
                "UL_LIVE=true (or UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING=true) to allow this"
            )
        api_key = (
            self.settings.api_key.get_secret_value().strip()
            if self.settings.api_key is not None
            else ""
        )
        if not api_key and self.provider.requires_api_key:
            raise RuntimeError(
                f"semantic model calls require {self.settings.api_key_environment_variable}"
            )
        return api_key or None

    def _bounded_json(self, payload: dict[str, JsonValue]) -> str:
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(serialized) > self.settings.max_input_chars:
            raise ValueError("request content exceeds max_input_chars")
        return serialized

    @staticmethod
    def _reference_vocabulary(frame: SemanticFrame) -> dict[str, JsonValue]:
        return cast(
            dict[str, JsonValue],
            {
                "request_modes": sorted({request.mode for request in frame.request_units}),
                "request_predicates": sorted(
                    {request.predicate for request in frame.request_units}
                ),
                "factor_types": [
                    {"kind": kind, "role": role}
                    for kind, role in sorted(
                        {(factor.kind, factor.role) for factor in frame.factors}
                    )
                ],
                "relation_kinds": sorted({relation.kind for relation in frame.relations}),
                "communication_kinds": sorted(
                    {communication_act.kind for communication_act in frame.communication_acts}
                ),
                "outcome_kinds": sorted({outcome.kind for outcome in frame.outcomes}),
                "outcome_predicates": sorted({outcome.predicate for outcome in frame.outcomes}),
                "outcome_field_names": sorted(
                    {field_name for outcome in frame.outcomes for field_name in outcome.fields}
                ),
            },
        )

    @classmethod
    def _expand_unambiguous_evidence_quotes(
        cls,
        interaction: InteractionRecord | UserInputRecord,
        frame: SemanticFrame,
    ) -> SemanticFrame:
        observed_output = (
            interaction.raw_observed_output if isinstance(interaction, InteractionRecord) else None
        )
        evidence_payload: JsonValue = {
            "raw_input": interaction.raw_input,
            "raw_observed_output": observed_output,
        }

        def expand_element(element: Any) -> Any:
            expanded_evidence: list[EvidenceReference] = []
            for evidence in element.evidence:
                resolved_value = cls._resolve_json_pointer(evidence_payload, evidence.json_pointer)
                quote = evidence.text_quote
                if (
                    isinstance(resolved_value, str)
                    and quote is not None
                    and quote not in resolved_value
                ):
                    parts = re.split(r"\s*(?:\.\.\.|…)\s*", quote)
                    if len(parts) == 2 and all(parts):
                        prefix_start = resolved_value.find(parts[0])
                        suffix_start = resolved_value.find(parts[1], prefix_start + len(parts[0]))
                        if (
                            prefix_start >= 0
                            and suffix_start >= 0
                            and prefix_start == resolved_value.rfind(parts[0])
                            and suffix_start == resolved_value.rfind(parts[1])
                        ):
                            quote = resolved_value[prefix_start : suffix_start + len(parts[1])]
                            evidence = evidence.model_copy(update={"text_quote": quote})
                expanded_evidence.append(evidence)
            return element.model_copy(update={"evidence": tuple(expanded_evidence)})

        return frame.model_copy(
            update={
                "request_units": tuple(expand_element(element) for element in frame.request_units),
                "factors": tuple(expand_element(element) for element in frame.factors),
                "relations": tuple(expand_element(element) for element in frame.relations),
                "communication_acts": tuple(
                    expand_element(element) for element in frame.communication_acts
                ),
                "outcomes": tuple(expand_element(element) for element in frame.outcomes),
            }
        )

    @staticmethod
    def _ground_self_correction_evidence(
        interaction: InteractionRecord | UserInputRecord,
        frame: SemanticFrame,
    ) -> SemanticFrame:
        correction_acts = tuple(
            act for act in frame.communication_acts if act.kind == "self_correction"
        )
        correction_relations = tuple(
            relation for relation in frame.relations if relation.kind == "superseded_by"
        )
        if len(correction_acts) != 1 or len(correction_relations) != 1:
            return frame
        correction_act = correction_acts[0]
        correction_relation = correction_relations[0]
        if (
            correction_act.attributes
            or len(correction_relation.source_ids) != 1
            or len(correction_relation.target_ids) != 1
            or correction_act.factor_ids
            != (*correction_relation.source_ids, *correction_relation.target_ids)
        ):
            return frame
        factors_by_id = {factor.id: factor for factor in frame.factors}
        provisional_factor = factors_by_id.get(correction_relation.source_ids[0])
        final_factor = factors_by_id.get(correction_relation.target_ids[0])
        if (
            provisional_factor is None
            or final_factor is None
            or provisional_factor.status != "superseded"
            or (provisional_factor.kind, provisional_factor.role)
            != (final_factor.kind, final_factor.role)
            or any(provisional_factor.id in request.factor_ids for request in frame.request_units)
        ):
            return frame

        def unique_input_quote(factor: SemanticFactor) -> str | None:
            quotes = tuple(
                dict.fromkeys(
                    evidence.text_quote
                    for evidence in factor.evidence
                    if evidence.source == "input" and evidence.text_quote is not None
                )
            )
            return quotes[0] if len(quotes) == 1 else None

        provisional_quote = unique_input_quote(provisional_factor)
        final_quote = unique_input_quote(final_factor)
        if (
            provisional_quote is None
            or final_quote is None
            or interaction.raw_input.count(provisional_quote) != 1
            or interaction.raw_input.count(final_quote) != 1
        ):
            return frame
        provisional_start = interaction.raw_input.index(provisional_quote)
        final_start = interaction.raw_input.index(final_quote)
        if provisional_start >= final_start:
            return frame
        between_values = interaction.raw_input[
            provisional_start + len(provisional_quote) : final_start
        ]
        if not any(character.isalpha() for character in between_values):
            return frame
        repair_evidence = (
            EvidenceReference(
                source="input",
                json_pointer="/raw_input",
                text_quote=interaction.raw_input[
                    provisional_start : final_start + len(final_quote)
                ],
            ),
        )
        return frame.model_copy(
            update={
                "relations": tuple(
                    relation.model_copy(update={"evidence": repair_evidence})
                    if relation.id == correction_relation.id
                    else relation
                    for relation in frame.relations
                ),
                "communication_acts": tuple(
                    act.model_copy(update={"evidence": repair_evidence})
                    if act.id == correction_act.id
                    else act
                    for act in frame.communication_acts
                ),
            }
        )

    @classmethod
    def _validate_evidence(
        cls,
        interaction: InteractionRecord | UserInputRecord,
        frame: SemanticFrame,
    ) -> None:
        observed_output = (
            interaction.raw_observed_output if isinstance(interaction, InteractionRecord) else None
        )
        if observed_output is None and frame.outcomes:
            raise ValueError("input-only frames must not contain outcomes")
        if observed_output is not None and not frame.outcomes:
            raise ValueError("observed outputs must produce at least one grounded outcome")
        if any(
            not any(evidence.source == "output" for evidence in outcome.evidence)
            for outcome in frame.outcomes
        ):
            raise ValueError("every observed outcome must include output evidence")
        elements = (
            *frame.request_units,
            *frame.factors,
            *frame.relations,
            *frame.communication_acts,
            *frame.outcomes,
        )
        evidence_payload: JsonValue = {
            "raw_input": interaction.raw_input,
            "raw_observed_output": observed_output,
        }
        for element in elements:
            if not element.evidence:
                raise ValueError(f"semantic element {element.id} requires source evidence")
            for evidence in element.evidence:
                if evidence.source == "output" and observed_output is None:
                    raise ValueError("input-only frames must not contain output evidence")
                expected_prefix = (
                    "/raw_input" if evidence.source == "input" else "/raw_observed_output"
                )
                if (
                    evidence.json_pointer != expected_prefix
                    and not evidence.json_pointer.startswith(f"{expected_prefix}/")
                ):
                    raise ValueError("evidence json_pointer does not match its source")
                resolved_value = cls._resolve_json_pointer(evidence_payload, evidence.json_pointer)
                cls._validate_text_quote(resolved_value, evidence)
                if isinstance(resolved_value, str) and evidence.text_quote is None:
                    raise ValueError("evidence on text must include an exact quote")

    @staticmethod
    def _resolve_json_pointer(value: JsonValue, pointer: str) -> JsonValue:
        if not pointer:
            return value
        current: object = value
        for encoded_token in pointer[1:].split("/"):
            token = encoded_token.replace("~1", "/").replace("~0", "~")
            if isinstance(current, dict):
                current_mapping = cast(dict[str, object], current)
                if token in current_mapping:
                    current = current_mapping[token]
                    continue
            valid_array_index = token == "0" or (token.isdecimal() and not token.startswith("0"))
            if isinstance(current, list) and valid_array_index:
                current_sequence = cast(list[object], current)
                index = int(token)
                if index < len(current_sequence):
                    current = current_sequence[index]
                    continue
            raise ValueError("evidence json_pointer does not resolve")
        return cast(JsonValue, current)

    @staticmethod
    def _validate_text_quote(
        resolved_value: JsonValue,
        evidence: EvidenceReference,
    ) -> None:
        if evidence.text_quote is None:
            return
        if not isinstance(resolved_value, str) or evidence.text_quote not in resolved_value:
            raise ValueError("evidence text_quote does not occur inside the selected string")

    @staticmethod
    def _decode_object(content: str) -> dict[str, Any]:
        decoded = json.loads(content)
        if not isinstance(decoded, dict):
            raise ValueError("structured response must be a JSON object")
        return cast(dict[str, Any], decoded)

    def _generation_metadata(
        self,
        response: _ChatCompletionResponse,
    ) -> dict[str, JsonValue]:
        metadata = self.provider.generation_metadata(response)
        if self._preflight_result is not None:
            metadata["evaluator_preflight"] = cast(
                JsonValue, self._preflight_result.model_dump(mode="json")
            )
        return metadata


def create_semantic_model_deconstructor(
    settings: DatasetSemanticSettings,
    *,
    client: httpx.AsyncClient | None = None,
) -> SemanticModelDeconstructor:
    if isinstance(settings, OpenAICompatibleDatasetSettings):
        provider: SemanticCompletionProvider = OpenAICompatibleSemanticProvider(
            provider_id=settings.semantic_provider_id,
            base_url=settings.semantic_base_url,
            endpoint_sha256=settings.semantic_endpoint_sha256,
        )
    else:
        provider = OpenRouterSemanticProvider()
    return SemanticModelDeconstructor(settings, provider, client=client)


def _same_origin(left: httpx.URL, right: httpx.URL) -> bool:
    return (left.scheme, left.host, left.port) == (right.scheme, right.host, right.port)


async def _single_chunk(content: bytes) -> AsyncIterator[bytes]:
    yield content


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


async def _read_bounded_response(response: httpx.Response, *, maximum_bytes: int) -> bytes:
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


def _preflight_http_capability(status_code: int, error_body: bytes) -> str:
    if status_code == 404:
        return "model availability or routing"
    detail = error_body.decode("utf-8", errors="ignore").lower()
    if "response_format" in detail or "json_schema" in detail or "structured" in detail:
        return "structured output"
    if "seed" in detail:
        return "seed"
    return "provider routing"
