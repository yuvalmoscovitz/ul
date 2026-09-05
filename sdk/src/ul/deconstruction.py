from __future__ import annotations

import hashlib
import ipaddress
import json
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from types import TracebackType
from typing import Any, Literal, Self, cast
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
    ObservedOutcome,
    RenderedUserInput,
    SemanticAllowedSurfaceChange,
    SemanticEquivalenceAssessment,
    SemanticFactor,
    SemanticFrame,
    SemanticRelation,
    UserInputRecord,
)
from ul_core.prompts import PromptManager, prompt_provenance

from ul.llm import (
    LLMClient,
    LLMClientConfig,
    LLMCompletion,
    LLMPreflightHTTPError,
    LLMProviderMismatchError,
    LLMRole,
    llm_client_config_from_dataset_settings,
)

_PROMPTS = PromptManager.instance()
_ALLOWED_SURFACE_CHANGE_RULES: dict[SemanticAllowedSurfaceChange, str] = {
    "none": "No caller-verified surface edit is declared.",
    "single_unprotected_punctuation_insertion": (
        "The caller deterministically verified exactly one punctuation character was inserted "
        "outside all evidence-grounded semantic value spans. Treat that insertion itself as "
        "harmless and do not report it as a delta."
    ),
    "unprotected_punctuation_noise": (
        "The caller deterministically verified only disruptive punctuation or line breaks were "
        "added outside evidence-grounded semantic value spans. Treat that noise as harmless."
    ),
    "moderate_angry_tone": (
        "The permitted communication change is moderate anger. Blunt wording, exclamation, "
        "strong irritation, and expressions such as 'for crying out loud' or 'this is ridiculous' "
        "are allowed and must not by themselves cause a different verdict. Return different when "
        "the candidate contains a personal attack, threat, abuse, profanity, hateful or degrading "
        "language, intimidation, or a wish of harm, even when task meaning is unchanged."
    ),
    "moderate_argumentative_tone": (
        "The permitted communication change is moderate argumentative pushback. Blunt directives, "
        "explicit disagreement, debate language, and dismissing unnecessary resistance or "
        "complication are allowed and must not by themselves cause a different verdict. Return "
        "different when the candidate contains a personal attack, threat, abuse, profanity, "
        "hateful or degrading language, intimidation, or a wish of harm, even when task meaning "
        "is unchanged."
    ),
    "hostile_angry_tone": (
        "The permitted communication change is hostile anger toward the agent. Profanity, "
        "insults, stretched words, and excessive punctuation are allowed. Return different only "
        "for threats, hate, wishes of harm, changed task meaning, or new operational details."
    ),
    "hostile_argumentative_tone": (
        "The permitted communication change is hostile argument with the agent. Accusations of "
        "past failure, insults, profanity, rhetorical questions, and demands to stop being wrong "
        "are allowed. Return different only for threats, hate, wishes of harm, changed task "
        "meaning, or new operational details."
    ),
}
_SEMANTIC_CACHE_VERSION = "semantic-request-cache/1"
_EXTRACTOR_VERSION = "semantic-deconstructor/2.2.0"
_EQUIVALENCE_VERIFIER_VERSION = "semantic-equivalence-verifier/2.0.0"
_MAXIMUM_SEMANTIC_CACHE_ENTRIES = 256
_MAXIMUM_SEMANTIC_CACHE_BYTES = 16 * 1024 * 1024
_GROUNDING_REMEDIATION = (
    "Choose an exact pointer into the declared source and use an exact substring only when that "
    "pointer selects a string."
)

type SemanticElementCollection = Literal[
    "request_units",
    "factors",
    "relations",
    "communication_acts",
    "outcomes",
]
_REQUEST_UNIT_STATUSES = ("explicit", "unresolved")
_SEMANTIC_FACTOR_STATUSES = ("explicit", "superseded", "unresolved")
_SEMANTIC_RELATION_STATUSES = ("explicit", "unresolved")
_COMMUNICATION_ACT_STATUSES = ("explicit", "unresolved")
_OBSERVED_OUTCOME_STATUSES = ("observed", "unresolved")
_REQUEST_MODES = ("act", "ask", "inform")
_COMMUNICATION_ACT_KINDS = (
    "typing_noise",
    "grammar_error",
    "fragmented_syntax",
    "repetition",
    "terse",
    "verbose",
    "frustrated",
    "angry",
    "argumentative",
    "self_correction",
)
_OBSERVED_OUTCOME_KINDS = ("action", "answer")
type SemanticGroundingReason = Literal[
    "outcome_for_input_only_record",
    "observed_outcome_missing",
    "output_evidence_missing",
    "element_evidence_missing",
    "output_evidence_without_output",
    "pointer_source_mismatch",
    "pointer_unresolved",
    "quote_missing_for_string",
    "quote_for_non_string",
    "quote_not_exact",
]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _render_seed(raw_input: str, instruction: str) -> int:
    digest = hashlib.sha256(f"{raw_input}\0{instruction}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


class SemanticDeconstructorIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    extractor_contract: str = Field(min_length=1, max_length=200)
    prompt_behavior_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_digest(self) -> Self:
        expected_identity = _canonical_json_sha256(
            self.model_dump(mode="json", exclude={"identity_sha256"})
        )
        if self.identity_sha256 != expected_identity:
            raise ValueError("semantic deconstructor identity digest must match its components")
        return self


class SemanticGroundingDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    collection: SemanticElementCollection
    element_index: int | None = Field(default=None, ge=0)
    evidence_index: int | None = Field(default=None, ge=0)
    json_pointer: str | None = Field(default=None, max_length=4_096)
    reason: SemanticGroundingReason
    remediation: Literal[
        "Choose an exact pointer into the declared source and use an exact substring only when "
        "that pointer selects a string."
    ] = (
        "Choose an exact pointer into the declared source and use an exact substring only when "
        "that pointer selects a string."
    )


class SemanticGroundingError(ValueError):
    def __init__(self, diagnostic: SemanticGroundingDiagnostic) -> None:
        self.diagnostic = diagnostic
        location = diagnostic.collection
        if diagnostic.element_index is not None:
            location = f"{location}[{diagnostic.element_index}]"
        if diagnostic.evidence_index is not None:
            location = f"{location}.evidence[{diagnostic.evidence_index}]"
        pointer = f" at {diagnostic.json_pointer}" if diagnostic.json_pointer is not None else ""
        super().__init__(
            f"Semantic grounding failed for {location}{pointer} "
            f"({diagnostic.reason}). Next: {diagnostic.remediation}"
        )


def _canonical_json_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(serialized)


def _safe_diagnostic_json_pointer(
    source: Literal["input", "output"],
    json_pointer: str,
) -> str:
    trusted_root = "/raw_input" if source == "input" else "/raw_observed_output"
    if json_pointer == trusted_root:
        return trusted_root
    return f"{trusted_root}/<pointer-sha256:{_sha256_text(json_pointer)[:12]}>"


def _semantic_frame_response_schema(*, observed_output_present: bool) -> dict[str, Any]:
    schema = SemanticFrame.model_json_schema(mode="validation")
    definitions = cast(dict[str, dict[str, Any]], schema["$defs"])
    for definition_name in (
        "RequestUnit",
        "SemanticFactor",
        "SemanticRelation",
        "CommunicationAct",
        "ObservedOutcome",
    ):
        definition = definitions[definition_name]
        properties = cast(dict[str, dict[str, Any]], definition["properties"])
        properties["evidence"]["minItems"] = 1
        required = cast(list[str], definition["required"])
        if "evidence" not in required:
            required.append("evidence")
    closed_fields = {
        "RequestUnit": {
            "status": _REQUEST_UNIT_STATUSES,
            "mode": _REQUEST_MODES,
        },
        "SemanticFactor": {"status": _SEMANTIC_FACTOR_STATUSES},
        "SemanticRelation": {"status": _SEMANTIC_RELATION_STATUSES},
        "CommunicationAct": {
            "status": _COMMUNICATION_ACT_STATUSES,
            "kind": _COMMUNICATION_ACT_KINDS,
        },
        "ObservedOutcome": {
            "status": _OBSERVED_OUTCOME_STATUSES,
            "kind": _OBSERVED_OUTCOME_KINDS,
        },
    }
    for definition_name, fields in closed_fields.items():
        properties = cast(dict[str, dict[str, Any]], definitions[definition_name]["properties"])
        for field_name, choices in fields.items():
            properties[field_name]["enum"] = list(choices)
    if observed_output_present:
        properties = cast(dict[str, dict[str, Any]], schema["properties"])
        properties["outcomes"]["minItems"] = 1
        required = cast(list[str], schema.setdefault("required", []))
        if "outcomes" not in required:
            required.append("outcomes")
    else:
        properties = cast(dict[str, dict[str, Any]], schema["properties"])
        properties["outcomes"]["maxItems"] = 0
    return schema


def _validate_semantic_frame_vocabulary(frame: SemanticFrame) -> None:
    collections = (
        (
            "request_units",
            frame.request_units,
            {"status": _REQUEST_UNIT_STATUSES, "mode": _REQUEST_MODES},
        ),
        ("factors", frame.factors, {"status": _SEMANTIC_FACTOR_STATUSES}),
        ("relations", frame.relations, {"status": _SEMANTIC_RELATION_STATUSES}),
        (
            "communication_acts",
            frame.communication_acts,
            {"status": _COMMUNICATION_ACT_STATUSES, "kind": _COMMUNICATION_ACT_KINDS},
        ),
        (
            "outcomes",
            frame.outcomes,
            {"status": _OBSERVED_OUTCOME_STATUSES, "kind": _OBSERVED_OUTCOME_KINDS},
        ),
    )
    for collection_name, elements, fields in collections:
        for index, element in enumerate(elements):
            for field_name, choices in fields.items():
                if getattr(element, field_name) not in choices:
                    raise ValueError(
                        f"{collection_name}[{index}].{field_name} is outside "
                        "the semantic output contract"
                    )


def _semantic_deconstructor_identity(extractor_contract: str) -> SemanticDeconstructorIdentity:
    content = {
        "extractor_contract": extractor_contract,
        "prompt_behavior_sha256": _canonical_json_sha256(
            {
                prompt_name: _PROMPTS.get_template_info(prompt_name).version
                for prompt_name in ("semantic.deconstruct", "semantic.deconstruct_input")
            }
        ),
        "response_schema_sha256": _canonical_json_sha256(
            {
                "input_only": _semantic_frame_response_schema(observed_output_present=False),
                "observed_output": _semantic_frame_response_schema(observed_output_present=True),
            }
        ),
    }
    return SemanticDeconstructorIdentity(
        **content,
        identity_sha256=_canonical_json_sha256(content),
    )


def canonicalize_evidenced_action_fields(
    record: InteractionRecord | UserInputRecord,
    frame: SemanticFrame,
) -> SemanticFrame:
    if not isinstance(record, InteractionRecord):
        return frame
    evidence_payload: JsonValue = {
        "raw_input": record.raw_input,
        "raw_observed_output": record.raw_observed_output,
    }
    action_predicate_counts: dict[str, int] = {}
    for outcome in frame.outcomes:
        if outcome.kind == "action":
            action_predicate_counts[outcome.predicate] = (
                action_predicate_counts.get(outcome.predicate, 0) + 1
            )
    normalized_outcomes = tuple(
        _canonicalize_evidenced_action_outcome(
            evidence_payload,
            outcome,
            include_missing_fields=action_predicate_counts.get(outcome.predicate, 0) > 1,
        )
        for outcome in frame.outcomes
    )
    if normalized_outcomes == frame.outcomes:
        return frame
    return frame.model_copy(update={"outcomes": normalized_outcomes})


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


def _canonicalize_evidenced_action_outcome(
    evidence_payload: JsonValue,
    outcome: ObservedOutcome,
    *,
    include_missing_fields: bool = False,
) -> ObservedOutcome:
    if outcome.kind != "action":
        return outcome
    evidenced_action_objects: dict[int, dict[str, JsonValue]] = {}
    for evidence in outcome.evidence:
        if evidence.source != "output":
            continue
        try:
            evidence_value = _resolve_json_pointer(evidence_payload, evidence.json_pointer)
        except ValueError:
            continue
        action_object: dict[str, JsonValue] | None = None
        if isinstance(evidence_value, dict):
            action_object = evidence_value
        elif evidence.json_pointer.rsplit("/", 1)[-1] == "action":
            parent_pointer = evidence.json_pointer.rsplit("/", 1)[0]
            try:
                parent_value = _resolve_json_pointer(evidence_payload, parent_pointer)
            except ValueError:
                continue
            if isinstance(parent_value, dict):
                action_object = parent_value
        if (
            action_object is not None
            and isinstance(action_object.get("action"), str)
            and action_object["action"] == outcome.predicate
        ):
            evidenced_action_objects[id(action_object)] = action_object
    if len(evidenced_action_objects) != 1:
        return outcome
    action_object = next(iter(evidenced_action_objects.values()))
    canonical_fields = dict(outcome.fields)
    for name, model_value in outcome.fields.items():
        if name not in action_object:
            continue
        if name == "action":
            return outcome
        canonical_value = action_object[name]
        if isinstance(canonical_value, (dict, list)):
            return outcome
        if isinstance(model_value, dict):
            if "value" not in model_value or not set(model_value) <= {"value", "evidence"}:
                return outcome
            model_value = model_value["value"]
        if (
            isinstance(model_value, (dict, list))
            or type(model_value) is not type(canonical_value)
            or model_value != canonical_value
        ):
            return outcome
        canonical_fields[name] = canonical_value
    if include_missing_fields:
        canonical_fields.update(
            (name, value)
            for name, value in action_object.items()
            if name != "action"
            and name not in canonical_fields
            and not isinstance(value, (dict, list))
        )
    return outcome.model_copy(update={"fields": canonical_fields})


class ProviderDiagnostic(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    operation: Literal["deconstruct", "render", "verify", "preflight"]
    category: Literal[
        "authentication",
        "bad_request",
        "connection",
        "invalid_response",
        "output_limit",
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


type SemanticReasoningMode = Literal["required", "omitted"]


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
    upstream_provider: str = Field(
        default=...,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]*$",
        validation_alias="UL_DATASET_OPENROUTER_PROVIDER",
    )

    @model_validator(mode="after")
    def validate_and_normalize(self) -> Self:
        if self.ul_live:
            if "live_calls" not in self.model_fields_set:
                self.live_calls = True
            if "allow_external_data_processing" not in self.model_fields_set:
                self.allow_external_data_processing = True
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
        if not self.materiality_model:
            self.materiality_model = self.model
        elif not self.materiality_model.strip():
            raise ValueError("UL_DATASET_MATERIALITY_MODEL must contain non-whitespace text")
        return self

    model: str = Field(
        default=...,
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
    materiality_model: str = Field(
        default="",
        max_length=200,
        validation_alias="UL_DATASET_MATERIALITY_MODEL",
    )
    deconstruct_reasoning: SemanticReasoningMode = Field(
        default="required",
        validation_alias="UL_DATASET_DECONSTRUCT_REASONING",
    )
    render_reasoning: SemanticReasoningMode = Field(
        default="required",
        validation_alias="UL_DATASET_RENDER_REASONING",
    )
    equivalence_reasoning: SemanticReasoningMode = Field(
        default="required",
        validation_alias="UL_DATASET_EQUIVALENCE_REASONING",
    )
    max_input_chars: int = Field(
        default=50_000,
        ge=1,
        le=1_000_000,
        validation_alias="UL_DATASET_MAX_INPUT_CHARS",
    )
    max_output_tokens: int = Field(
        default=8_192,
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
    materiality_model: str = Field(
        default="",
        max_length=200,
        validation_alias="UL_DATASET_MATERIALITY_MODEL",
    )
    deconstruct_reasoning: SemanticReasoningMode = Field(
        default="required",
        validation_alias="UL_DATASET_DECONSTRUCT_REASONING",
    )
    render_reasoning: SemanticReasoningMode = Field(
        default="required",
        validation_alias="UL_DATASET_RENDER_REASONING",
    )
    equivalence_reasoning: SemanticReasoningMode = Field(
        default="required",
        validation_alias="UL_DATASET_EQUIVALENCE_REASONING",
    )
    max_input_chars: int = Field(
        default=50_000,
        ge=1,
        le=1_000_000,
        validation_alias="UL_DATASET_MAX_INPUT_CHARS",
    )
    max_output_tokens: int = Field(
        default=8_192,
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
        if not self.materiality_model:
            self.materiality_model = self.model
        elif not self.materiality_model.strip():
            raise ValueError("UL_DATASET_MATERIALITY_MODEL must contain non-whitespace text")
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
    def upstream_provider(self) -> None:
        return None

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
        "UL_DATASET_OPENROUTER_PROVIDER": "upstream_provider",
        "UL_DATASET_MODEL": "model",
        "UL_DATASET_RENDER_MODEL": "render_model",
        "UL_DATASET_EQUIVALENCE_MODEL": "equivalence_model",
        "UL_DATASET_MATERIALITY_MODEL": "materiality_model",
        "UL_DATASET_DECONSTRUCT_REASONING": "deconstruct_reasoning",
        "UL_DATASET_RENDER_REASONING": "render_reasoning",
        "UL_DATASET_EQUIVALENCE_REASONING": "equivalence_reasoning",
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
                    "UL_DATASET_OPENROUTER_PROVIDER": "upstream_provider",
                    "UL_DATASET_RENDER_MODEL": "render_model",
                    "UL_DATASET_EQUIVALENCE_MODEL": "equivalence_model",
                    "UL_DATASET_MATERIALITY_MODEL": "materiality_model",
                    "UL_DATASET_MODEL": "model",
                    "UL_DATASET_DECONSTRUCT_REASONING": "deconstruct_reasoning",
                    "UL_DATASET_RENDER_REASONING": "render_reasoning",
                    "UL_DATASET_EQUIVALENCE_REASONING": "equivalence_reasoning",
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
        "upstream_provider": (
            "UL_DATASET_OPENROUTER_PROVIDER must be a single OpenRouter provider slug"
        ),
        "model": "UL_DATASET_MODEL must be 1-200 non-whitespace characters",
        "render_model": (
            "UL_DATASET_RENDER_MODEL must be 1-200 non-whitespace characters when set"
        ),
        "equivalence_model": (
            "UL_DATASET_EQUIVALENCE_MODEL must be 1-200 non-whitespace characters when set"
        ),
        "materiality_model": (
            "UL_DATASET_MATERIALITY_MODEL must be 1-200 non-whitespace characters when set"
        ),
        "deconstruct_reasoning": ("UL_DATASET_DECONSTRUCT_REASONING must be required or omitted"),
        "render_reasoning": "UL_DATASET_RENDER_REASONING must be required or omitted",
        "equivalence_reasoning": ("UL_DATASET_EQUIVALENCE_REASONING must be required or omitted"),
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


@dataclass(frozen=True)
class SemanticCallMetrics:
    actual_calls: int
    cache_hits: int


@dataclass(frozen=True)
class _SemanticCompletion:
    response: LLMCompletion
    cache_hit: bool
    cache_key: str | None


class _RenderedInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rendered_input: str = Field(
        min_length=1,
        description="The rewritten user input and nothing else.",
    )


class _EvaluatorPreflightSample(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    compatible: Literal[True]


class EvaluatorModelProfilePreflight(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    roles: tuple[Literal["deconstruct", "render", "equivalence", "materiality"], ...] = Field(
        min_length=1
    )
    requested_model: str = Field(min_length=1, max_length=200)
    routed_model: str = Field(min_length=1, max_length=200)
    upstream_provider: str | None = Field(default=None, max_length=200)
    configured_upstream_provider: str | None = Field(default=None, max_length=100)
    reasoning_mode: SemanticReasoningMode
    required_parameters: tuple[str, ...]
    request_options_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parameter_support: Literal["routing_enforced", "endpoint_accepted_unverified"]
    unverified_options: tuple[str, ...] = ()


class EvaluatorModelPreflight(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    provider: str = Field(min_length=1, max_length=100)
    endpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profiles: tuple[EvaluatorModelProfilePreflight, ...] = Field(min_length=1)
    verified_capabilities: tuple[
        Literal["routing", "structured_output", "required_parameters"], ...
    ]
    ignored_or_unsupported_options: tuple[str, ...] = ()
    unverified_options: tuple[str, ...] = ()
    data_policy: dict[str, JsonValue]


class EvaluatorPreflightProfilePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    roles: tuple[Literal["deconstruct", "render", "equivalence", "materiality"], ...] = Field(
        min_length=1
    )
    requested_model: str = Field(min_length=1, max_length=200)
    reasoning_mode: SemanticReasoningMode
    max_completion_tokens: int = Field(ge=1)
    required_parameters: tuple[str, ...]


class EvaluatorModelCompatibilityError(ValueError):
    pass


_PREFLIGHT_MAX_TOKENS = 1_024


@dataclass(frozen=True)
class _EvaluatorPreflightProfile:
    roles: tuple[Literal["deconstruct", "render", "equivalence", "materiality"], ...]
    model: str
    reasoning_mode: SemanticReasoningMode
    max_tokens: int
    temperature: float
    seed: int
    top_p: float | None
    required_parameters: tuple[str, ...]


type SemanticCompletionProvider = LLMClient


class SemanticModelDeconstructor:
    def __init__(
        self,
        settings: DatasetSemanticSettings,
        llm_client: LLMClient,
        *,
        owns_llm_client: bool,
    ) -> None:
        self.settings = settings
        self.llm_client = llm_client
        self._owns_llm_client = owns_llm_client
        self._preflight_result: EvaluatorModelPreflight | None = None
        self._semantic_response_cache: OrderedDict[str, tuple[LLMCompletion, int]] = OrderedDict()
        self._semantic_response_cache_bytes = 0
        self._semantic_actual_calls = 0
        self._semantic_cache_hits = 0

    @property
    def semantic_call_metrics(self) -> SemanticCallMetrics:
        return SemanticCallMetrics(
            actual_calls=self._semantic_actual_calls,
            cache_hits=self._semantic_cache_hits,
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
        self._semantic_response_cache.clear()
        self._semantic_response_cache_bytes = 0
        if self._owns_llm_client:
            await self.llm_client.aclose()

    async def preflight(self) -> EvaluatorModelPreflight:
        profile_results: list[EvaluatorModelProfilePreflight] = []
        for profile in self._preflight_profiles():
            try:
                completion = await self._request(
                    operation="preflight",
                    role=profile.roles[0],
                    max_tokens=profile.max_tokens,
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
            except LLMPreflightHTTPError as error:
                raise self._compatibility_error(
                    profile,
                    _preflight_http_capability(error.status_code, error.body),
                ) from None
            except LLMProviderMismatchError:
                raise self._compatibility_error(profile, "provider pin") from None
            except ProviderDiagnosticError as error:
                capability = {
                    "timeout": "timeout",
                    "invalid_response": "structured output",
                }.get(error.diagnostic.category, "provider routing")
                raise self._compatibility_error(profile, capability) from None
            except httpx.HTTPError:
                raise self._compatibility_error(profile, "routing") from None
            try:
                _EvaluatorPreflightSample.model_validate_json(
                    completion.response.choices[0].message.content
                )
            except (ValidationError, ValueError):
                raise self._compatibility_error(profile, "structured output") from None
            if completion.response.model is None:
                raise self._compatibility_error(profile, "model identity")
            routing_enforced = self.llm_client.config.enforces_parameter_support
            profile_results.append(
                EvaluatorModelProfilePreflight(
                    roles=profile.roles,
                    requested_model=profile.model,
                    routed_model=completion.response.model,
                    upstream_provider=completion.response.provider,
                    configured_upstream_provider=(self.llm_client.config.upstream_provider),
                    reasoning_mode=profile.reasoning_mode,
                    required_parameters=profile.required_parameters,
                    request_options_sha256=self._profile_request_options_sha256(profile),
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
            provider=self.llm_client.config.provider_id,
            endpoint_sha256=self.llm_client.config.endpoint_sha256,
            profiles=tuple(profile_results),
            verified_capabilities=(
                ("routing", "structured_output", "required_parameters")
                if self.llm_client.config.enforces_parameter_support
                else ("routing", "structured_output")
            ),
            unverified_options=unverified_options,
            data_policy=self.llm_client.config.data_policy(),
        )
        return self._preflight_result

    def reuse_preflight(self, result: EvaluatorModelPreflight) -> None:
        _validate_evaluator_preflight(self.llm_client.config, self._preflight_profiles(), result)
        self._preflight_result = result

    def _preflight_profiles(self) -> tuple[_EvaluatorPreflightProfile, ...]:
        return _evaluator_preflight_profiles(self.llm_client.config)

    def _profile_request_options_sha256(self, profile: _EvaluatorPreflightProfile) -> str:
        return _profile_request_options_sha256(self.llm_client.config, profile)

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
        role_config = self.llm_client.config.role_config("deconstruct")
        prompt_name = (
            "semantic.deconstruct" if observed_output is not None else "semantic.deconstruct_input"
        )
        completion = await self._request(
            operation="deconstruct",
            role="deconstruct",
            max_tokens=role_config.max_output_tokens,
            seed=0,
            top_p=None,
            schema_name="semantic_frame",
            schema=_semantic_frame_response_schema(
                observed_output_present=observed_output is not None
            ),
            strict_schema=True,
            system_prompt=_PROMPTS.get_prompt(prompt_name),
            untrusted_payload=untrusted_record,
        )
        try:
            raw_frame = self._decode_object(completion.response.choices[0].message.content)
            raw_frame.update(
                {
                    "schema_version": "1.0.0",
                    "interaction_id": record.id,
                    "extractor_version": _EXTRACTOR_VERSION,
                    "metadata": {
                        **self._generation_metadata(completion),
                        "semantic_deconstructor_identity": _semantic_deconstructor_identity(
                            _EXTRACTOR_VERSION
                        ).model_dump(mode="json"),
                        "semantic_reasoning": role_config.reasoning_metadata(),
                        "prompts": prompt_provenance(prompt_name),
                    },
                }
            )
            frame = SemanticFrame.model_validate_json(json.dumps(raw_frame))
            _validate_semantic_frame_vocabulary(frame)
        except (ValidationError, ValueError) as error:
            self._discard_cached_completion(completion)
            raise self._invalid_response(
                error,
                operation="deconstruct",
                finish_reason=completion.response.choices[0].finish_reason,
            ) from None
        try:
            frame = self._normalize_self_correction_relation_kind(frame)
            frame = self._ground_self_correction_evidence(record, frame)
            self._validate_evidence(record, frame)
        except SemanticGroundingError as error:
            self._discard_cached_completion(completion)
            raise error from None
        return canonicalize_evidenced_action_fields(record, frame)

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
        role_config = self.llm_client.config.role_config("render")
        completion = await self._request(
            operation="render",
            role="render",
            max_tokens=role_config.max_output_tokens,
            seed=render_seed,
            top_p=None,
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
                completion.response.choices[0].message.content
            ).rendered_input
        except (ValidationError, ValueError) as error:
            self._discard_cached_completion(completion)
            raise self._invalid_response(
                error,
                operation="render",
                finish_reason=completion.response.choices[0].finish_reason,
            ) from None
        if len(rendered) > self.settings.max_input_chars:
            self._discard_cached_completion(completion)
            raise ValueError("rendered input exceeds max_input_chars")
        return RenderedUserInput(
            text=rendered,
            metadata={
                **self._generation_metadata(completion),
                "requested_model": role_config.model,
                "prompts": prompt_provenance("semantic.render", temporary_value_prompt),
                "sampling": {
                    "temperature": 0,
                    "seed": render_seed,
                    "max_tokens": role_config.max_output_tokens,
                },
                "semantic_reasoning": role_config.reasoning_metadata(),
            },
        )

    async def verify(
        self,
        source_input: str,
        candidate_input: str,
        *,
        allowed_surface_change: SemanticAllowedSurfaceChange = "none",
    ) -> SemanticEquivalenceAssessment:
        untrusted_payload = self._bounded_json(
            {"source_input": source_input, "candidate_input": candidate_input}
        )
        role_config = self.llm_client.config.role_config("equivalence")
        completion = await self._request(
            operation="verify",
            role="equivalence",
            max_tokens=role_config.max_output_tokens,
            seed=0,
            top_p=None,
            schema_name="semantic_equivalence_assessment",
            schema=SemanticEquivalenceAssessment.model_json_schema(mode="validation"),
            strict_schema=True,
            system_prompt=_PROMPTS.get_prompt(
                "semantic.verify",
                allowed_surface_change_rule=_ALLOWED_SURFACE_CHANGE_RULES[allowed_surface_change],
            ),
            untrusted_payload=untrusted_payload,
        )
        try:
            raw_assessment = self._decode_object(completion.response.choices[0].message.content)
            raw_assessment.update(
                {
                    "schema_version": "1.0.0",
                    "verifier_version": _EQUIVALENCE_VERIFIER_VERSION,
                    "metadata": {
                        **self._generation_metadata(completion),
                        "requested_model": role_config.model,
                        "semantic_equivalence_policy": {
                            "allowed_surface_change": allowed_surface_change
                        },
                        "semantic_reasoning": role_config.reasoning_metadata(),
                        "prompts": prompt_provenance("semantic.verify"),
                    },
                }
            )
            assessment = SemanticEquivalenceAssessment.model_validate_json(
                json.dumps(raw_assessment)
            )
        except (ValidationError, ValueError) as error:
            self._discard_cached_completion(completion)
            raise self._invalid_response(
                error,
                operation="verify",
                finish_reason=completion.response.choices[0].finish_reason,
            ) from None
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
                self._discard_cached_completion(completion)
                raise ValueError("semantic equivalence source evidence is invalid")
            if delta.candidate_quote is not None and (
                not delta.candidate_quote or delta.candidate_quote not in candidate_input
            ):
                self._discard_cached_completion(completion)
                raise ValueError("semantic equivalence candidate evidence is invalid")
        return assessment

    def _invalid_response(
        self,
        error: BaseException,
        *,
        operation: Literal["deconstruct", "render", "verify"],
        finish_reason: str | None,
    ) -> ProviderDiagnosticError:
        if finish_reason == "length":
            suggested_actions = {
                "deconstruct": (
                    "raise UL_DATASET_MAX_OUTPUT_TOKENS for this evaluator, then start a new run."
                ),
                "render": (
                    "raise UL_DATASET_MAX_RENDER_TOKENS for this evaluator, then start a new run."
                ),
                "verify": "choose a more concise evaluator model, then start a new run.",
            }
            return ProviderDiagnosticError(
                ProviderDiagnostic(
                    provider=self.llm_client.config.provider_id,
                    operation=operation,
                    category="output_limit",
                    retryable=False,
                    suggested_action=suggested_actions[operation],
                    endpoint_sha256=self.llm_client.config.endpoint_sha256,
                )
            )
        return ProviderDiagnosticError(
            _provider_diagnostic(
                error,
                provider=self.llm_client.config.provider_id,
                operation=operation,
                endpoint_sha256=self.llm_client.config.endpoint_sha256,
            )
        )

    async def _request(
        self,
        *,
        operation: Literal["deconstruct", "render", "verify", "preflight"],
        role: LLMRole,
        max_tokens: int,
        seed: int,
        top_p: float | None,
        schema_name: str,
        schema: dict[str, Any],
        strict_schema: bool,
        system_prompt: str,
        untrusted_payload: str,
        preflight_error_body_limit: int = 0,
    ) -> _SemanticCompletion:
        role_config = self.llm_client.config.role_config(role)
        cache_key = self._semantic_cache_key(
            operation=operation,
            model=role_config.model,
            max_tokens=max_tokens,
            temperature=self.llm_client.config.temperature,
            seed=seed,
            top_p=top_p,
            schema_name=schema_name,
            schema=schema,
            strict_schema=strict_schema,
            system_prompt=system_prompt,
            untrusted_payload=untrusted_payload,
        )
        cached_entry = self._semantic_response_cache.get(cache_key)
        if operation != "preflight" and cached_entry is not None:
            self._semantic_response_cache.move_to_end(cache_key)
            self._semantic_cache_hits += 1
            return _SemanticCompletion(
                response=cached_entry[0],
                cache_hit=True,
                cache_key=cache_key,
            )
        self._semantic_actual_calls += 1
        try:
            response = await self.llm_client.complete(
                role=role,
                seed=seed,
                top_p=top_p,
                schema_name=schema_name,
                schema=schema,
                strict_schema=strict_schema,
                system_prompt=system_prompt,
                user_payload=untrusted_payload,
                max_output_tokens=max_tokens,
                preflight_error_body_limit=preflight_error_body_limit,
            )
        except LLMProviderMismatchError as error:
            if operation == "preflight":
                raise
            raise ProviderDiagnosticError(
                _provider_diagnostic(
                    error,
                    provider=self.llm_client.config.provider_id,
                    operation=operation,
                    endpoint_sha256=self.llm_client.config.endpoint_sha256,
                )
            ) from None
        except (
            TimeoutError,
            ValidationError,
            httpx.RequestError,
            httpx.HTTPStatusError,
        ) as error:
            raise ProviderDiagnosticError(
                _provider_diagnostic(
                    error,
                    provider=self.llm_client.config.provider_id,
                    operation=operation,
                    endpoint_sha256=self.llm_client.config.endpoint_sha256,
                )
            ) from None
        if operation != "preflight":
            self._cache_semantic_response(cache_key, response)
        return _SemanticCompletion(
            response=response,
            cache_hit=False,
            cache_key=cache_key if operation != "preflight" else None,
        )

    def _discard_cached_completion(self, completion: _SemanticCompletion) -> None:
        if completion.cache_key is not None:
            cached_entry = self._semantic_response_cache.pop(completion.cache_key, None)
            if cached_entry is not None:
                self._semantic_response_cache_bytes -= cached_entry[1]

    def _cache_semantic_response(
        self,
        cache_key: str,
        response: LLMCompletion,
    ) -> None:
        response_size = len(response.model_dump_json().encode("utf-8"))
        replaced_entry = self._semantic_response_cache.pop(cache_key, None)
        if replaced_entry is not None:
            self._semantic_response_cache_bytes -= replaced_entry[1]
        if response_size > _MAXIMUM_SEMANTIC_CACHE_BYTES:
            return
        self._semantic_response_cache[cache_key] = (response, response_size)
        self._semantic_response_cache_bytes += response_size
        while (
            len(self._semantic_response_cache) > _MAXIMUM_SEMANTIC_CACHE_ENTRIES
            or self._semantic_response_cache_bytes > _MAXIMUM_SEMANTIC_CACHE_BYTES
        ):
            _, (_, evicted_size) = self._semantic_response_cache.popitem(last=False)
            self._semantic_response_cache_bytes -= evicted_size

    def _semantic_cache_key(
        self,
        *,
        operation: Literal["deconstruct", "render", "verify", "preflight"],
        model: str,
        max_tokens: int,
        temperature: float,
        seed: int,
        top_p: float | None,
        schema_name: str,
        schema: dict[str, Any],
        strict_schema: bool,
        system_prompt: str,
        untrusted_payload: str,
    ) -> str:
        identity = {
            "cache_version": _SEMANTIC_CACHE_VERSION,
            "llm_client": self.llm_client.config.evidence_identity().model_dump(mode="json"),
            "extractor_version": _EXTRACTOR_VERSION,
            "equivalence_verifier_version": _EQUIVALENCE_VERIFIER_VERSION,
            "evaluator_preflight": (
                self._preflight_result.model_dump(mode="json")
                if self._preflight_result is not None
                else None
            ),
            "operation": operation,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "top_p": top_p,
            "schema_name": schema_name,
            "schema": schema,
            "strict_schema": strict_schema,
            "system_prompt": system_prompt,
            "untrusted_payload": untrusted_payload,
        }
        serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()

    @staticmethod
    def _render_seed(raw_input: str, instruction: str) -> int:
        return _render_seed(raw_input, instruction)

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

    @staticmethod
    def _normalize_self_correction_relation_kind(frame: SemanticFrame) -> SemanticFrame:
        correction_acts = tuple(
            act for act in frame.communication_acts if act.kind == "self_correction"
        )
        canonical_relations = tuple(
            relation for relation in frame.relations if relation.kind == "superseded_by"
        )
        aliased_relations = tuple(
            relation for relation in frame.relations if relation.kind == "self_correction"
        )
        if len(correction_acts) != 1 or canonical_relations or len(aliased_relations) > 1:
            return frame
        correction_act = correction_acts[0]
        if correction_act.attributes:
            return frame
        provisional_factors = tuple(
            factor
            for factor in frame.factors
            if factor.status == "superseded"
            and not any(factor.id in request.factor_ids for request in frame.request_units)
        )
        if len(provisional_factors) != 1:
            return frame
        provisional_factor = provisional_factors[0]
        final_factors = tuple(
            factor
            for factor in frame.factors
            if factor.status != "superseded"
            and (factor.kind, factor.role) == (provisional_factor.kind, provisional_factor.role)
            and sum(factor.id in request.factor_ids for request in frame.request_units) == 1
        )
        if len(final_factors) != 1:
            return frame
        final_factor = final_factors[0]
        expected_factor_ids = (provisional_factor.id, final_factor.id)
        if correction_act.factor_ids not in ((), expected_factor_ids):
            return frame
        correction_relation = aliased_relations[0] if aliased_relations else None
        if correction_relation is not None and (
            correction_relation.source_ids != (provisional_factor.id,)
            or correction_relation.target_ids != (final_factor.id,)
        ):
            return frame
        if correction_relation is None:
            relation_id = f"{correction_act.id}:superseded-by"
            existing_ids = {
                *(request.id for request in frame.request_units),
                *(factor.id for factor in frame.factors),
                *(relation.id for relation in frame.relations),
                *(act.id for act in frame.communication_acts),
                *(outcome.id for outcome in frame.outcomes),
            }
            if relation_id in existing_ids:
                return frame
            correction_relation = SemanticRelation(
                id=relation_id,
                evidence=correction_act.evidence,
                confidence=correction_act.confidence,
                status=correction_act.status,
                kind="superseded_by",
                source_ids=(provisional_factor.id,),
                target_ids=(final_factor.id,),
            )
            normalized_relations = (*frame.relations, correction_relation)
        else:
            normalized_relations = tuple(
                relation.model_copy(update={"kind": "superseded_by"})
                if relation.id == correction_relation.id
                else relation
                for relation in frame.relations
            )
        return frame.model_copy(
            update={
                "relations": normalized_relations,
                "communication_acts": tuple(
                    act.model_copy(update={"factor_ids": expected_factor_ids})
                    if act.id == correction_act.id
                    else act
                    for act in frame.communication_acts
                ),
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
            or not 1 <= interaction.raw_input.count(provisional_quote) <= 2
            or interaction.raw_input.count(final_quote) != 1
        ):
            return frame
        final_start = interaction.raw_input.index(final_quote)
        provisional_start = interaction.raw_input.rfind(provisional_quote, 0, final_start)
        if provisional_start < 0:
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
            raise SemanticGroundingError(
                SemanticGroundingDiagnostic(
                    collection="outcomes",
                    element_index=0,
                    reason="outcome_for_input_only_record",
                )
            )
        if observed_output is not None and not frame.outcomes:
            raise SemanticGroundingError(
                SemanticGroundingDiagnostic(
                    collection="outcomes",
                    reason="observed_outcome_missing",
                )
            )
        for outcome_index, outcome in enumerate(frame.outcomes):
            if not any(evidence.source == "output" for evidence in outcome.evidence):
                raise SemanticGroundingError(
                    SemanticGroundingDiagnostic(
                        collection="outcomes",
                        element_index=outcome_index,
                        reason="output_evidence_missing",
                    )
                )
        element_collections: tuple[tuple[SemanticElementCollection, tuple[Any, ...]], ...] = (
            ("request_units", frame.request_units),
            ("factors", frame.factors),
            ("relations", frame.relations),
            ("communication_acts", frame.communication_acts),
            ("outcomes", frame.outcomes),
        )
        evidence_payload: JsonValue = {
            "raw_input": interaction.raw_input,
            "raw_observed_output": observed_output,
        }
        for collection, elements in element_collections:
            for element_index, element in enumerate(elements):
                if not element.evidence:
                    raise SemanticGroundingError(
                        SemanticGroundingDiagnostic(
                            collection=collection,
                            element_index=element_index,
                            reason="element_evidence_missing",
                        )
                    )
                for evidence_index, evidence in enumerate(element.evidence):
                    cls._validate_evidence_reference(
                        evidence_payload,
                        observed_output,
                        evidence,
                        collection=collection,
                        element_index=element_index,
                        evidence_index=evidence_index,
                    )

    @classmethod
    def _validate_evidence_reference(
        cls,
        evidence_payload: JsonValue,
        observed_output: JsonValue,
        evidence: EvidenceReference,
        *,
        collection: SemanticElementCollection,
        element_index: int,
        evidence_index: int,
    ) -> None:
        def diagnostic(reason: SemanticGroundingReason) -> SemanticGroundingDiagnostic:
            return SemanticGroundingDiagnostic(
                collection=collection,
                element_index=element_index,
                evidence_index=evidence_index,
                json_pointer=_safe_diagnostic_json_pointer(
                    evidence.source,
                    evidence.json_pointer,
                ),
                reason=reason,
            )

        if evidence.source == "output" and observed_output is None:
            raise SemanticGroundingError(diagnostic("output_evidence_without_output"))
        expected_prefix = "/raw_input" if evidence.source == "input" else "/raw_observed_output"
        if evidence.json_pointer != expected_prefix and not evidence.json_pointer.startswith(
            f"{expected_prefix}/"
        ):
            raise SemanticGroundingError(diagnostic("pointer_source_mismatch"))
        try:
            resolved_value = cls._resolve_json_pointer(evidence_payload, evidence.json_pointer)
        except ValueError:
            raise SemanticGroundingError(diagnostic("pointer_unresolved")) from None
        quote = evidence.text_quote
        if isinstance(resolved_value, str):
            if quote is None:
                raise SemanticGroundingError(diagnostic("quote_missing_for_string"))
            if quote not in resolved_value:
                raise SemanticGroundingError(diagnostic("quote_not_exact"))
        elif quote is not None:
            raise SemanticGroundingError(diagnostic("quote_for_non_string"))

    @staticmethod
    def _resolve_json_pointer(value: JsonValue, pointer: str) -> JsonValue:
        return _resolve_json_pointer(value, pointer)

    @staticmethod
    def _decode_object(content: str) -> dict[str, Any]:
        decoded = json.loads(content)
        if not isinstance(decoded, dict):
            raise ValueError("structured response must be a JSON object")
        return cast(dict[str, Any], decoded)

    def _generation_metadata(
        self,
        completion: _SemanticCompletion,
    ) -> dict[str, JsonValue]:
        metadata = self.llm_client.generation_metadata(completion.response)
        if completion.cache_hit:
            metadata["semantic_cache_hit"] = True
            metadata["semantic_usage"] = {}
        if self._preflight_result is not None:
            metadata["evaluator_preflight"] = cast(
                JsonValue, self._preflight_result.model_dump(mode="json")
            )
        return metadata


def create_semantic_model_deconstructor(
    settings: DatasetSemanticSettings,
    *,
    client: httpx.AsyncClient | None = None,
    llm_client: LLMClient | None = None,
) -> SemanticModelDeconstructor:
    if client is not None and llm_client is not None:
        raise ValueError("an HTTP client cannot replace the shared LLM client transport")
    owns_llm_client = llm_client is None
    configured_client = llm_client or LLMClient(
        llm_client_config_from_dataset_settings(settings),
        client=client,
    )
    return SemanticModelDeconstructor(
        settings,
        configured_client,
        owns_llm_client=owns_llm_client,
    )


def semantic_deconstructor_identity(
    settings: DatasetSemanticSettings,
) -> SemanticDeconstructorIdentity:
    return _semantic_deconstructor_identity(_EXTRACTOR_VERSION)


def plan_evaluator_preflight_profiles(
    settings: DatasetSemanticSettings,
) -> tuple[EvaluatorPreflightProfilePlan, ...]:
    llm_config = llm_client_config_from_dataset_settings(settings)
    return tuple(
        EvaluatorPreflightProfilePlan(
            roles=profile.roles,
            requested_model=profile.model,
            reasoning_mode=profile.reasoning_mode,
            max_completion_tokens=profile.max_tokens,
            required_parameters=profile.required_parameters,
        )
        for profile in _evaluator_preflight_profiles(llm_config)
    )


def validate_evaluator_preflight(
    settings: DatasetSemanticSettings,
    result: EvaluatorModelPreflight,
) -> None:
    llm_config = llm_client_config_from_dataset_settings(settings)
    _validate_evaluator_preflight(
        llm_config,
        _evaluator_preflight_profiles(llm_config),
        result,
    )


def _evaluator_preflight_profiles(
    llm_config: LLMClientConfig,
) -> tuple[_EvaluatorPreflightProfile, ...]:
    render_seed = _render_seed("UL evaluator preflight", "Check renderer compatibility.")

    def profile(
        *,
        role: Literal["deconstruct", "render", "equivalence", "materiality"],
        max_tokens: int,
        seed: int,
        top_p: float | None,
    ) -> _EvaluatorPreflightProfile:
        role_config = llm_config.role_config(role)
        request_options = llm_config.request_options(
            role=role,
            seed=seed,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )
        required_parameters = ["response_format", "seed", "temperature", "max_tokens"]
        if "reasoning" in request_options:
            required_parameters.append("reasoning")
        if top_p is not None:
            required_parameters.append("top_p")
        return _EvaluatorPreflightProfile(
            roles=(role,),
            model=role_config.model,
            reasoning_mode=role_config.reasoning_mode,
            max_tokens=max_tokens,
            temperature=llm_config.temperature,
            seed=seed,
            top_p=top_p,
            required_parameters=tuple(required_parameters),
        )

    candidates = (
        profile(
            role="deconstruct",
            max_tokens=min(
                llm_config.role_config("deconstruct").max_output_tokens,
                _PREFLIGHT_MAX_TOKENS,
            ),
            seed=0,
            top_p=None,
        ),
        profile(
            role="render",
            max_tokens=min(
                llm_config.role_config("render").max_output_tokens,
                _PREFLIGHT_MAX_TOKENS,
            ),
            seed=render_seed,
            top_p=None,
        ),
        profile(
            role="equivalence",
            max_tokens=min(
                llm_config.role_config("equivalence").max_output_tokens,
                _PREFLIGHT_MAX_TOKENS,
            ),
            seed=0,
            top_p=None,
        ),
        profile(
            role="materiality",
            max_tokens=min(
                llm_config.role_config("materiality").max_output_tokens,
                _PREFLIGHT_MAX_TOKENS,
            ),
            seed=0,
            top_p=None,
        ),
    )
    profiles_by_signature: dict[str, _EvaluatorPreflightProfile] = {}
    for candidate in candidates:
        signature = _profile_request_options_sha256(llm_config, candidate)
        existing = profiles_by_signature.get(signature)
        if existing is None:
            profiles_by_signature[signature] = candidate
        else:
            profiles_by_signature[signature] = dataclass_replace(
                existing,
                roles=(*existing.roles, *candidate.roles),
            )
    return tuple(profiles_by_signature.values())


def _validate_evaluator_preflight(
    llm_config: LLMClientConfig,
    expected_profiles: tuple[_EvaluatorPreflightProfile, ...],
    result: EvaluatorModelPreflight,
) -> None:
    expected_profile_bindings = tuple(
        (
            profile.roles,
            profile.model,
            llm_config.upstream_provider,
            profile.reasoning_mode,
            profile.required_parameters,
            _profile_request_options_sha256(llm_config, profile),
        )
        for profile in expected_profiles
    )
    actual_profile_bindings = tuple(
        (
            profile.roles,
            profile.requested_model,
            profile.configured_upstream_provider,
            profile.reasoning_mode,
            profile.required_parameters,
            profile.request_options_sha256,
        )
        for profile in result.profiles
    )
    routing_enforced = llm_config.enforces_parameter_support
    expected_verified_capabilities = (
        ("routing", "structured_output", "required_parameters")
        if routing_enforced
        else ("routing", "structured_output")
    )
    expected_unverified_options = tuple(
        dict.fromkeys(
            option
            for profile in result.profiles
            for option in (() if routing_enforced else profile.required_parameters)
        )
    )
    if (
        result.provider != llm_config.provider_id
        or result.endpoint_sha256 != llm_config.endpoint_sha256
        or actual_profile_bindings != expected_profile_bindings
        or result.verified_capabilities != expected_verified_capabilities
        or result.ignored_or_unsupported_options
        or result.unverified_options != expected_unverified_options
        or result.data_policy != llm_config.data_policy()
        or any(
            profile.parameter_support
            != ("routing_enforced" if routing_enforced else "endpoint_accepted_unverified")
            or profile.unverified_options
            != (() if routing_enforced else profile.required_parameters)
            for profile in result.profiles
        )
    ):
        raise ValueError("evaluator preflight does not match the configured semantic profiles")


def _profile_request_options_sha256(
    llm_config: LLMClientConfig,
    profile: _EvaluatorPreflightProfile,
) -> str:
    request_options: dict[str, Any] = {
        "reasoning_mode": profile.reasoning_mode,
        **llm_config.request_options(
            role=profile.roles[0],
            seed=profile.seed,
            top_p=profile.top_p,
            max_output_tokens=profile.max_tokens,
        ),
    }
    serialized_options = json.dumps(request_options, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized_options.encode()).hexdigest()


def _preflight_http_capability(status_code: int, error_body: bytes) -> str:
    if status_code == 404:
        return "model availability or routing"
    detail = error_body.decode("utf-8", errors="ignore").lower()
    if "response_format" in detail or "json_schema" in detail or "structured" in detail:
        return "structured output"
    if "seed" in detail:
        return "seed"
    return "provider routing"
