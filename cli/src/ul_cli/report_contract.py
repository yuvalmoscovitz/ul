from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationInfo, model_validator

ReportEvidenceType = Literal[
    "dataset_evaluation",
    "correction_after_first_response",
    "retry_after_successful_commit",
    "timeout_after_commit",
]
ReportReviewStatus = Literal["resolved", "action_required", "inconclusive"]
ReportEvaluationMode = Literal["variance"]
ReportEvidenceScope = Literal["response_only", "response_and_state"]
ReportCapabilityLimitation = Literal[
    "cleanup_verification",
    "conversation_replay",
    "state_observation",
]
FindingKind = Literal["behavior_difference", "customer_invariant_violation"]
FindingCategory = Literal[
    "duplicate_effect",
    "unexpected_effect",
    "missing_effect",
    "changed_grounded_effect_argument",
    "unstable_behavior",
    "customer_invariant_violation",
]
FindingSeverity = Literal["unrated", "low", "medium", "high", "critical"]
FindingReviewStatus = Literal[
    "needs_review",
    "confirmed",
    "expected",
    "unsupported",
    "inconclusive",
]
FindingNextAction = Literal[
    "review_dataset_finding",
    "inspect_dataset_evidence",
    "inspect_stateful_evidence",
]
FindingSummaryText = Literal[
    "The changed input made the agent repeat an action.",
    "The changed input made the agent take a new action.",
    "The changed input made the agent skip a baseline action.",
    "The changed input altered an important action detail.",
    "The changed input produced inconsistent behavior across repetitions.",
    "The agent violated a customer-defined rule.",
]

_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$"
_VERSION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,49}$"
_FINDING_ID_PATTERN = r"^ulf_v1_[0-9a-f]{64}$"
_PATTERN_ID_PATTERN = r"^ulp_v1_[0-9a-f]{64}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_JSON_POINTER_PATTERN = re.compile(r"(?:/(?:[^~/]|~[01])*)*")
_MAXIMUM_RUN_RECEIPT_BYTES = 1_000_000
_FindingReferenceCode = Annotated[str, Field(pattern=_IDENTIFIER_PATTERN)]
_SEVERITY_RANK: dict[FindingSeverity, int] = {
    "unrated": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}
_BEHAVIOR_SUMMARIES: dict[str, str] = {
    "duplicate_effect": "The changed input made the agent repeat an action.",
    "unexpected_effect": "The changed input made the agent take a new action.",
    "missing_effect": "The changed input made the agent skip a baseline action.",
    "changed_grounded_effect_argument": "The changed input altered an important action detail.",
    "unstable_behavior": "The changed input produced inconsistent behavior across repetitions.",
}


class ReportInputError(ValueError):
    pass


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


_PUBLIC_REFERENCE_PATTERN = r"^ulref_v1_[0-9a-f]{64}$"
_EVIDENCE_POINTER_ID_PATTERN = r"^ulep_v1_[0-9a-f]{64}$"
_RUN_RECEIPT_ID_PATTERN = r"^ulrr_v1_[0-9a-f]{64}$"
_REVIEW_REFERENCE_PATTERN = r"^ulreview_v1_[0-9a-f]{64}$"
_CAPTURED_JSON_BYTES = 250_000
OccurrenceCapability = Literal[
    "cleanup_verification",
    "conversation_replay",
    "response_observation",
    "state_observation",
    "tool_observation",
    "trace_observation",
]
OccurrenceLimitation = Literal[
    "correctness_not_verified",
    "evaluator_provenance_unavailable",
    "model_provenance_unavailable",
    "one_or_more_repetitions_inconclusive",
    "production_prevalence_not_measured",
    "source_execution_unavailable",
]


class VersionedReference(_StrictModel):
    id: str = Field(pattern=_PUBLIC_REFERENCE_PATTERN)
    version: str = Field(pattern=_PUBLIC_REFERENCE_PATTERN)


class EvidencePointer(_StrictModel):
    pointer_id: str = Field(pattern=_EVIDENCE_POINTER_ID_PATTERN)
    kind: Literal[
        "input",
        "response",
        "action",
        "rule",
        "state",
        "tool_call",
        "tool_result",
        "lifecycle",
        "trace",
    ]
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    record_id: str | None = Field(default=None, min_length=1, max_length=500)
    json_pointer: str = Field(max_length=2_000)
    arm: Literal["source", "probe", "shared"]
    authority: Literal[
        "customer_declared",
        "deterministic_evaluator",
        "invoker_self_reported",
        "source_self_reported",
        "environment_self_reported",
        "independent_observer",
    ]
    source_id: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_json_pointer(self) -> Self:
        if _JSON_POINTER_PATTERN.fullmatch(self.json_pointer) is None:
            raise ValueError("evidence pointer must be an RFC 6901 JSON pointer")
        return self


class ProbeChange(_StrictModel):
    kind: Literal["input", "context", "turn_sequence", "state_setup", "event_behavior"]
    source_descriptor: Literal[
        "baseline_context",
        "baseline_turn_sequence",
        "declared_state_setup",
        "recorded_input",
        "unmodified_event_behavior",
    ]
    probe_descriptor: Literal[
        "augmented_context",
        "augmented_input",
        "augmented_turn_sequence",
        "injected_event_behavior",
        "modified_state_setup",
    ]
    source_evidence_pointer_ids: tuple[str, ...] = Field(min_length=1, max_length=100)
    probe_evidence_pointer_ids: tuple[str, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_pointer_ids(self) -> Self:
        for pointer_ids in (
            self.source_evidence_pointer_ids,
            self.probe_evidence_pointer_ids,
        ):
            _validate_sorted_unique(pointer_ids, "probe change evidence pointer IDs")
        if set(self.source_evidence_pointer_ids) & set(self.probe_evidence_pointer_ids):
            raise ValueError("source and probe evidence pointers must be disjoint")
        return self


class ObservedDelta(_StrictModel):
    kind: Literal["response", "action", "rule", "state"]
    change: Literal["added", "removed", "changed", "violated", "unstable"]
    subject_ref: str = Field(pattern=_PUBLIC_REFERENCE_PATTERN)
    rule: VersionedReference | None = None
    source_state: Literal[
        "observed",
        "not_observed",
        "satisfied",
        "violated",
        "stable",
        "unstable",
        "unknown",
    ]
    probe_state: Literal[
        "observed",
        "not_observed",
        "satisfied",
        "violated",
        "stable",
        "unstable",
        "unknown",
    ]
    evidence_pointer_ids: tuple[str, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_delta(self) -> Self:
        _validate_sorted_unique(self.evidence_pointer_ids, "delta evidence pointer IDs")
        if (self.kind == "rule") != (self.rule is not None):
            raise ValueError("only rule deltas carry a versioned rule reference")
        if self.change == "added" and (
            self.source_state != "not_observed" or self.probe_state != "observed"
        ):
            raise ValueError("added deltas require absent source and observed probe facts")
        if self.change == "removed" and (
            self.source_state != "observed" or self.probe_state != "not_observed"
        ):
            raise ValueError("removed deltas require observed source and absent probe facts")
        if self.change == "changed" and (
            self.source_state != "observed" or self.probe_state != "observed"
        ):
            raise ValueError("changed deltas require observed source and probe facts")
        if self.change == "violated" and self.probe_state != "violated":
            raise ValueError("violated deltas require a violated probe fact")
        if self.change == "unstable" and self.probe_state != "unstable":
            raise ValueError("unstable deltas require an unstable probe fact")
        return self


class FindingRepetition(_StrictModel):
    repetition: int = Field(ge=1)
    outcome: Literal["finding_observed", "finding_not_observed", "inconclusive"]
    source_receipt_id: str | None = Field(default=None, pattern=_RUN_RECEIPT_ID_PATTERN)
    probe_receipt_id: str | None = Field(default=None, pattern=_RUN_RECEIPT_ID_PATTERN)
    evidence_pointer_ids: tuple[str, ...] = Field(default=(), max_length=500)
    inconclusive_reason: str | None = Field(default=None, pattern=_IDENTIFIER_PATTERN)

    @model_validator(mode="after")
    def validate_repetition(self) -> Self:
        _validate_sorted_unique(self.evidence_pointer_ids, "repetition evidence pointer IDs")
        if self.outcome == "inconclusive":
            if self.inconclusive_reason is None:
                raise ValueError("inconclusive repetitions require a reason")
        elif (
            self.probe_receipt_id is None
            or not self.evidence_pointer_ids
            or self.inconclusive_reason is not None
        ):
            raise ValueError("conclusive repetitions require probe evidence and no reason")
        return self


class RepetitionSummary(_StrictModel):
    requested: int = Field(ge=1)
    conclusive: int = Field(ge=0)
    observed: int = Field(ge=0)
    violated: int | None = Field(default=None, ge=0)
    inconclusive: int = Field(ge=0)
    stability: Literal["stable", "unstable", "inconclusive"]
    reproducibility: Literal[
        "reproduced",
        "intermittent",
        "not_reproduced",
        "not_established",
    ]


class FindingOccurrence(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    occurrence_id: str = Field(pattern=_FINDING_ID_PATTERN)
    kind: FindingKind
    category: FindingCategory
    campaign_ref: str = Field(pattern=_PUBLIC_REFERENCE_PATTERN)
    source_interaction_ref: str | None = Field(default=None, pattern=_PUBLIC_REFERENCE_PATTERN)
    fixture: VersionedReference | None = None
    case_ref: str = Field(pattern=_PUBLIC_REFERENCE_PATTERN)
    operator: VersionedReference
    bundle: VersionedReference | None = None
    probe_change: ProbeChange
    observed_deltas: tuple[ObservedDelta, ...] = Field(min_length=1, max_length=100)
    violated_rule: VersionedReference | None = None
    rule_definition_evidence_pointer_ids: tuple[str, ...] = Field(default=(), max_length=100)
    evidence_pointer_ids: tuple[str, ...] = Field(min_length=1, max_length=1_000)
    repetitions: tuple[FindingRepetition, ...] = Field(min_length=1, max_length=1_000)
    repetition_summary: RepetitionSummary
    required_capabilities: tuple[OccurrenceCapability, ...] = Field(default=(), max_length=100)
    limitations: tuple[OccurrenceLimitation, ...] = Field(default=(), max_length=100)
    review_history_ids: tuple[str, ...] = Field(
        default=(),
        max_length=1_000,
    )
    next_action: FindingNextAction

    @model_validator(mode="after")
    def validate_occurrence(self, info: ValidationInfo) -> Self:
        if self.kind == "behavior_difference":
            if self.category == "customer_invariant_violation" or self.violated_rule is not None:
                raise ValueError("behavior occurrences cannot claim a customer rule violation")
            if self.rule_definition_evidence_pointer_ids:
                raise ValueError("behavior occurrences cannot cite a customer rule definition")
        elif (
            self.category != "customer_invariant_violation"
            or self.violated_rule is None
            or not self.rule_definition_evidence_pointer_ids
        ):
            raise ValueError("invariant occurrences require a customer rule and its definition")
        _validate_sorted_unique(self.evidence_pointer_ids, "occurrence evidence pointer IDs")
        _validate_sorted_unique(
            self.rule_definition_evidence_pointer_ids,
            "rule definition evidence pointer IDs",
        )
        for values, label in (
            (self.required_capabilities, "required capabilities"),
            (self.limitations, "limitations"),
            (self.review_history_ids, "review history IDs"),
        ):
            _validate_sorted_unique(values, label)
        if any(
            re.fullmatch(_REVIEW_REFERENCE_PATTERN, value) is None
            for value in self.review_history_ids
        ):
            raise ValueError("review history IDs must be privacy-safe references")
        repetition_numbers = tuple(item.repetition for item in self.repetitions)
        if repetition_numbers != tuple(range(1, len(self.repetitions) + 1)):
            raise ValueError("finding repetitions must be contiguous and ordered")
        referenced_pointer_ids = {
            *self.probe_change.source_evidence_pointer_ids,
            *self.probe_change.probe_evidence_pointer_ids,
            *self.rule_definition_evidence_pointer_ids,
            *(
                pointer_id
                for delta in self.observed_deltas
                for pointer_id in delta.evidence_pointer_ids
            ),
            *(pointer_id for item in self.repetitions for pointer_id in item.evidence_pointer_ids),
        }
        if referenced_pointer_ids != set(self.evidence_pointer_ids):
            raise ValueError("occurrence evidence pointers must exactly match its claims")
        observed = sum(item.outcome == "finding_observed" for item in self.repetitions)
        inconclusive = sum(item.outcome == "inconclusive" for item in self.repetitions)
        conclusive = len(self.repetitions) - inconclusive
        expected_reproducibility = (
            "not_established"
            if conclusive == 0
            else "not_reproduced"
            if observed == 0
            else "reproduced"
            if observed == conclusive
            else "intermittent"
        )
        expected_violated = observed if self.kind == "customer_invariant_violation" else None
        expected_summary = self.repetition_summary.model_copy(
            update={
                "requested": len(self.repetitions),
                "conclusive": conclusive,
                "observed": observed,
                "violated": expected_violated,
                "inconclusive": inconclusive,
                "reproducibility": expected_reproducibility,
            }
        )
        if self.repetition_summary != expected_summary:
            raise ValueError("repetition summary must match exact repetition evidence")
        if conclusive == 0 and self.repetition_summary.stability != "inconclusive":
            raise ValueError("evidence without conclusive repetitions is inconclusive")
        if (
            expected_reproducibility == "intermittent"
            and self.repetition_summary.stability != "unstable"
        ):
            raise ValueError("intermittent reproduction is unstable")
        _validate_category_delta(self)
        if info.context != {"building_occurrence": True} and (
            self.occurrence_id != _finding_occurrence_id(self)
        ):
            raise ValueError("finding occurrence ID must match its canonical claims")
        return self


class CapturedJson(_StrictModel):
    canonical_json: str = Field(min_length=1, max_length=_CAPTURED_JSON_BYTES)
    sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_capture(self) -> Self:
        encoded = self.canonical_json.encode("utf-8")
        if len(encoded) > _CAPTURED_JSON_BYTES:
            raise ValueError("captured JSON exceeds its byte limit")
        try:
            value = json.loads(self.canonical_json, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError):
            raise ValueError("captured value must be finite JSON") from None
        if self.canonical_json != _canonical_json(value):
            raise ValueError("captured JSON must use the canonical representation")
        if hashlib.sha256(encoded).hexdigest() != self.sha256:
            raise ValueError("captured JSON digest does not match its content")
        return self


class ReceiptEvidenceValue(_StrictModel):
    evidence_pointer_id: str = Field(pattern=_EVIDENCE_POINTER_ID_PATTERN)
    value: CapturedJson


class ToolExchangeReceipt(_StrictModel):
    sequence: int = Field(ge=1)
    call: ReceiptEvidenceValue
    result: ReceiptEvidenceValue | None = None


class StateReceipt(_StrictModel):
    evidence: ReceiptEvidenceValue


class LifecycleReceipt(_StrictModel):
    phase: Literal["initial_reset", "setup", "execution", "cleanup_reset"]
    status: Literal["succeeded", "failed", "not_attempted", "unknown"]
    evidence_pointer_ids: tuple[str, ...] = Field(default=(), max_length=100)
    limitation: str | None = Field(default=None, pattern=_IDENTIFIER_PATTERN)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> Self:
        _validate_sorted_unique(self.evidence_pointer_ids, "lifecycle evidence pointer IDs")
        if (self.status != "succeeded") != (self.limitation is not None):
            raise ValueError("unverified lifecycle phases require one limitation")
        if self.status == "succeeded" and not self.evidence_pointer_ids:
            raise ValueError("successful lifecycle phases require exact evidence")
        return self


class ProvenanceReceipt(_StrictModel):
    role: Literal["customer", "invoker", "target", "model", "evaluator", "environment", "observer"]
    id: str = Field(min_length=1, max_length=500)
    version: str | None = Field(default=None, min_length=1, max_length=100)
    config_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)


class UsageReceipt(_StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0)
    duration_ms: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_tokens(self) -> Self:
        if (
            self.total_tokens is not None
            and self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens != self.input_tokens + self.output_tokens
        ):
            raise ValueError("total tokens must equal input plus output tokens")
        return self


class RedactionReceipt(_StrictModel):
    policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    matched_value_count: int = Field(ge=0)
    redacted_value_count: int = Field(ge=0)
    retained_private_value_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if (
            self.redacted_value_count + self.retained_private_value_count
            != self.matched_value_count
        ):
            raise ValueError("redaction accounting must cover every matched value")
        return self


class RunReceiptContent(_StrictModel):
    disclosure: Literal["private"] = "private"
    repetition: int = Field(ge=1)
    arm: Literal["source", "probe"]
    evidence_scope: ReportEvidenceScope
    input: ReceiptEvidenceValue
    response: ReceiptEvidenceValue | None = None
    tool_exchanges: tuple[ToolExchangeReceipt, ...] = Field(default=(), max_length=1_000)
    state_before: StateReceipt | None = None
    state_after: StateReceipt | None = None
    lifecycle: tuple[LifecycleReceipt, ...] = Field(min_length=1, max_length=10)
    provenance: tuple[ProvenanceReceipt, ...] = Field(min_length=1, max_length=100)
    trace_evidence_pointer_ids: tuple[str, ...] = Field(default=(), max_length=1_000)
    usage: UsageReceipt | None = None
    redaction: RedactionReceipt
    evidence_pointers: tuple[EvidencePointer, ...] = Field(min_length=1, max_length=1_000)
    limitations: tuple[_FindingReferenceCode, ...] = Field(default=(), max_length=100)
    recorded_at: datetime

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        pointer_ids = tuple(pointer.pointer_id for pointer in self.evidence_pointers)
        _validate_sorted_unique(pointer_ids, "receipt evidence pointers")
        pointers = {pointer.pointer_id: pointer for pointer in self.evidence_pointers}
        referenced = {self.input.evidence_pointer_id, *self.trace_evidence_pointer_ids}
        _validate_receipt_value_pointer(self.input, pointers, "input", self.arm)
        if self.response is None:
            if "response_missing" not in self.limitations:
                raise ValueError("missing responses require an explicit limitation")
        else:
            referenced.add(self.response.evidence_pointer_id)
            _validate_receipt_value_pointer(self.response, pointers, "response", self.arm)
        sequences = tuple(exchange.sequence for exchange in self.tool_exchanges)
        if sequences != tuple(range(1, len(sequences) + 1)):
            raise ValueError("tool exchanges must be contiguous and ordered")
        for exchange in self.tool_exchanges:
            referenced.add(exchange.call.evidence_pointer_id)
            _validate_receipt_value_pointer(exchange.call, pointers, "tool_call", self.arm)
            if exchange.result is not None:
                referenced.add(exchange.result.evidence_pointer_id)
                _validate_receipt_value_pointer(exchange.result, pointers, "tool_result", self.arm)
        for state in (self.state_before, self.state_after):
            if state is not None:
                referenced.add(state.evidence.evidence_pointer_id)
                _validate_receipt_value_pointer(state.evidence, pointers, "state", self.arm)
        if self.evidence_scope == "response_only":
            if self.state_before is not None or self.state_after is not None:
                raise ValueError("response-only receipts cannot contain state evidence")
        elif self.state_before is None or self.state_after is None:
            raise ValueError("response-and-state receipts require before and after state")
        lifecycle_phases = tuple(phase.phase for phase in self.lifecycle)
        expected_lifecycle = tuple(
            sorted(
                set(lifecycle_phases),
                key=("initial_reset", "setup", "execution", "cleanup_reset").index,
            )
        )
        if lifecycle_phases != expected_lifecycle or "execution" not in lifecycle_phases:
            raise ValueError("lifecycle phases must be unique, ordered, and include execution")
        for phase in self.lifecycle:
            referenced.update(phase.evidence_pointer_ids)
            for pointer_id in phase.evidence_pointer_ids:
                _validate_pointer(pointers, pointer_id, "lifecycle", self.arm)
        for pointer_id in self.trace_evidence_pointer_ids:
            _validate_pointer(pointers, pointer_id, "trace", self.arm)
        for values, label in (
            (self.trace_evidence_pointer_ids, "trace evidence pointer IDs"),
            (self.limitations, "receipt limitations"),
        ):
            _validate_sorted_unique(values, label)
        provenance_keys = tuple((item.role, item.id, item.version) for item in self.provenance)
        expected_provenance = tuple(
            sorted(set(provenance_keys), key=lambda item: (item[0], item[1], item[2] or ""))
        )
        if provenance_keys != expected_provenance:
            raise ValueError("provenance must be sorted and unique")
        provenance_roles = {item.role for item in self.provenance}
        if (
            "model" not in provenance_roles
            and "model_provenance_unavailable" not in self.limitations
        ):
            raise ValueError("missing model provenance requires an explicit limitation")
        if (
            "evaluator" not in provenance_roles
            and "evaluator_provenance_unavailable" not in self.limitations
        ):
            raise ValueError("missing evaluator provenance requires an explicit limitation")
        _validate_pointer_provenance(self.evidence_pointers, self.provenance)
        if self.recorded_at.tzinfo is None or self.recorded_at.utcoffset() is None:
            raise ValueError("receipt timestamp must include a UTC offset")
        if not referenced.issubset(pointers):
            raise ValueError("receipt references an unknown evidence pointer")
        return self


class RunReceipt(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    receipt_id: str = Field(pattern=_RUN_RECEIPT_ID_PATTERN)
    content: RunReceiptContent

    @model_validator(mode="after")
    def validate_receipt_id(self) -> Self:
        if self.receipt_id != _run_receipt_id(self.content):
            raise ValueError("run receipt ID must match its canonical content")
        if len(serialize_run_receipt(self).encode("utf-8")) > _MAXIMUM_RUN_RECEIPT_BYTES:
            raise ValueError("run receipt exceeds the 1 MB JSON limit")
        return self


class FindingEvidencePackage(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    disclosure: Literal["private"] = "private"
    occurrence: FindingOccurrence
    receipts: tuple[RunReceipt, ...] = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_package(self) -> Self:
        receipt_ids = tuple(receipt.receipt_id for receipt in self.receipts)
        if receipt_ids != tuple(sorted(set(receipt_ids))):
            raise ValueError("package receipts must be sorted and unique")
        referenced_receipt_ids = {
            receipt_id
            for repetition in self.occurrence.repetitions
            for receipt_id in (repetition.source_receipt_id, repetition.probe_receipt_id)
            if receipt_id is not None
        }
        if referenced_receipt_ids != set(receipt_ids):
            raise ValueError("package receipts must exactly match repetition references")
        receipts_by_id = {receipt.receipt_id: receipt for receipt in self.receipts}
        pointers: dict[str, EvidencePointer] = {}
        pointers_by_receipt: dict[str, set[str]] = {}
        for receipt in self.receipts:
            receipt_pointer_ids = {
                pointer.pointer_id for pointer in receipt.content.evidence_pointers
            }
            pointers_by_receipt[receipt.receipt_id] = receipt_pointer_ids
            for pointer in receipt.content.evidence_pointers:
                if pointer.pointer_id in pointers:
                    raise ValueError("evidence pointer IDs must be unique across package receipts")
                pointers[pointer.pointer_id] = pointer
        if not set(self.occurrence.evidence_pointer_ids).issubset(pointers):
            raise ValueError("package does not contain every public evidence reference")
        for repetition in self.occurrence.repetitions:
            allowed_pointer_ids: set[str] = set()
            for expected_arm, receipt_id in (
                ("source", repetition.source_receipt_id),
                ("probe", repetition.probe_receipt_id),
            ):
                if receipt_id is not None:
                    receipt_content = receipts_by_id[receipt_id].content
                    if (
                        receipt_content.arm != expected_arm
                        or receipt_content.repetition != repetition.repetition
                    ):
                        raise ValueError(
                            "repetition receipt must match its declared arm and repetition"
                        )
                    allowed_pointer_ids.update(pointers_by_receipt[receipt_id])
            if not set(repetition.evidence_pointer_ids).issubset(allowed_pointer_ids):
                raise ValueError("repetition evidence must come from its referenced receipts")
            if repetition.outcome != "inconclusive" and not any(
                _repetition_supports_category(
                    self.occurrence,
                    delta,
                    repetition,
                    pointers,
                )
                for delta in _category_supporting_deltas(self.occurrence)
            ):
                raise ValueError(
                    "conclusive repetition requires category evidence from that execution"
                )
        observed_repetition_pointer_ids = {
            pointer_id
            for repetition in self.occurrence.repetitions
            if repetition.outcome == "finding_observed"
            for pointer_id in repetition.evidence_pointer_ids
        }
        supporting_delta_pointer_ids = {
            pointer_id
            for delta in _category_supporting_deltas(self.occurrence)
            for pointer_id in delta.evidence_pointer_ids
        }
        if not supporting_delta_pointer_ids.issubset(observed_repetition_pointer_ids):
            raise ValueError("aggregate observed deltas must come only from observed repetitions")
        _validate_occurrence_evidence(self.occurrence, pointers)
        return self


def capture_json(value: JsonValue) -> CapturedJson:
    canonical_json = _canonical_json(value)
    return CapturedJson(
        canonical_json=canonical_json,
        sha256=hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
    )


def build_run_receipt(content: RunReceiptContent) -> RunReceipt:
    return RunReceipt(receipt_id=_run_receipt_id(content), content=content)


def build_finding_occurrence(**values: object) -> FindingOccurrence:
    values["occurrence_id"] = f"ulf_v1_{'0' * 64}"
    occurrence = FindingOccurrence.model_validate(
        values,
        context={"building_occurrence": True},
    )
    values["occurrence_id"] = _finding_occurrence_id(occurrence)
    return FindingOccurrence.model_validate(values)


def parse_run_receipt(serialized: str | bytes) -> RunReceipt:
    return RunReceipt.model_validate_json(serialized)


def serialize_run_receipt(receipt: RunReceipt) -> str:
    return _canonical_json(receipt.model_dump(mode="json"))


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _run_receipt_id(content: RunReceiptContent) -> str:
    digest = hashlib.sha256(_canonical_json(content.model_dump(mode="json")).encode("utf-8"))
    return f"ulrr_v1_{digest.hexdigest()}"


def _finding_occurrence_id(occurrence: FindingOccurrence) -> str:
    claims = occurrence.model_dump(
        mode="json",
        exclude={"occurrence_id", "review_history_ids"},
    )
    digest = hashlib.sha256(_canonical_json(claims).encode("utf-8"))
    return f"ulf_v1_{digest.hexdigest()}"


def _validate_sorted_unique(values: tuple[str, ...], label: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{label} must be sorted and unique")


def _validate_receipt_value_pointer(
    evidence: ReceiptEvidenceValue,
    pointers: dict[str, EvidencePointer],
    kind: str,
    arm: str,
) -> None:
    _validate_pointer(pointers, evidence.evidence_pointer_id, kind, arm)


def _validate_pointer(
    pointers: dict[str, EvidencePointer],
    pointer_id: str,
    kind: str,
    arm: str,
) -> None:
    pointer = pointers.get(pointer_id)
    if pointer is None:
        raise ValueError("receipt references an unknown evidence pointer")
    if pointer.kind != kind or pointer.arm not in {arm, "shared"}:
        raise ValueError("receipt value references incompatible evidence")


def _validate_pointer_provenance(
    pointers: tuple[EvidencePointer, ...],
    provenance: tuple[ProvenanceReceipt, ...],
) -> None:
    roles_by_source: dict[str, set[str]] = {}
    for item in provenance:
        roles_by_source.setdefault(item.id, set()).add(item.role)
    authority_roles = {
        "customer_declared": {"customer"},
        "deterministic_evaluator": {"evaluator"},
        "invoker_self_reported": {"invoker"},
        "source_self_reported": {"target"},
        "environment_self_reported": {"environment"},
        "independent_observer": {"observer"},
    }
    if any(
        not (roles_by_source.get(pointer.source_id, set()) & authority_roles[pointer.authority])
        for pointer in pointers
    ):
        raise ValueError("evidence authority requires compatible source provenance")


def _validate_category_delta(occurrence: FindingOccurrence) -> None:
    if not _category_supporting_deltas(occurrence):
        raise ValueError("finding category must match its observed delta")
    if (
        occurrence.category == "unstable_behavior"
        and occurrence.repetition_summary.stability != "unstable"
    ):
        raise ValueError("unstable findings require unstable repetition evidence")
    rule_deltas = tuple(
        delta
        for delta in occurrence.observed_deltas
        if delta.kind == "rule" and delta.change == "violated"
    )
    if occurrence.violated_rule is None and rule_deltas:
        raise ValueError("behavior findings cannot contain rule-violation deltas")
    if occurrence.violated_rule is not None and (
        len(rule_deltas) != 1 or rule_deltas[0].rule != occurrence.violated_rule
    ):
        raise ValueError("violated rule identity must match one exact rule delta")


def _category_supporting_deltas(
    occurrence: FindingOccurrence,
) -> tuple[ObservedDelta, ...]:
    required_category_delta = {
        "duplicate_effect": ("action", "added"),
        "unexpected_effect": ("action", "added"),
        "missing_effect": ("action", "removed"),
        "changed_grounded_effect_argument": ("action", "changed"),
        "unstable_behavior": (None, "unstable"),
        "customer_invariant_violation": ("rule", "violated"),
    }[occurrence.category]
    return tuple(
        delta
        for delta in occurrence.observed_deltas
        if (
            (required_category_delta[0] is None or delta.kind == required_category_delta[0])
            and delta.change == required_category_delta[1]
        )
    )


def _repetition_supports_category(
    occurrence: FindingOccurrence,
    delta: ObservedDelta,
    repetition: FindingRepetition,
    pointers: dict[str, EvidencePointer],
) -> bool:
    repetition_pointer_ids = set(repetition.evidence_pointer_ids)
    if repetition.outcome == "finding_observed":
        supporting_pointer_ids = set(delta.evidence_pointer_ids) & repetition_pointer_ids
    else:
        all_observed_pointer_ids = {
            pointer_id
            for supporting_delta in _category_supporting_deltas(occurrence)
            for pointer_id in supporting_delta.evidence_pointer_ids
        }
        expected_kinds = {
            "response": {"response"},
            "action": {"action", "tool_call", "tool_result"},
            "rule": {"rule"},
            "state": {"state"},
        }[delta.kind]
        supporting_pointer_ids = {
            pointer_id
            for pointer_id in repetition_pointer_ids - all_observed_pointer_ids
            if pointers[pointer_id].kind in expected_kinds
            and pointers[pointer_id].authority
            in {"deterministic_evaluator", "independent_observer"}
        }
    arms = {pointers[pointer_id].arm for pointer_id in supporting_pointer_ids}
    if occurrence.category == "duplicate_effect":
        return {"source", "probe"}.issubset(arms)
    if delta.change == "changed":
        return {"source", "probe"}.issubset(arms)
    if delta.change in {"added", "violated", "unstable"}:
        return "probe" in arms
    if delta.change == "removed":
        return "source" in arms
    return False


def _validate_occurrence_evidence(
    occurrence: FindingOccurrence,
    pointers: dict[str, EvidencePointer],
) -> None:
    change_kinds = {
        "input": {"input"},
        "context": {"input"},
        "turn_sequence": {"input"},
        "state_setup": {"state"},
        "event_behavior": {"action", "tool_call", "tool_result", "lifecycle"},
    }[occurrence.probe_change.kind]
    for pointer_id in occurrence.probe_change.source_evidence_pointer_ids:
        pointer = pointers[pointer_id]
        if pointer.arm != "source" or pointer.kind not in change_kinds:
            raise ValueError("source change evidence is incompatible")
    for pointer_id in occurrence.probe_change.probe_evidence_pointer_ids:
        pointer = pointers[pointer_id]
        if pointer.arm != "probe" or pointer.kind not in change_kinds:
            raise ValueError("probe change evidence is incompatible")
    delta_kinds = {
        "response": {"response"},
        "action": {"action", "tool_call", "tool_result"},
        "rule": {"rule"},
        "state": {"state"},
    }
    for delta in occurrence.observed_deltas:
        delta_pointers = tuple(pointers[pointer_id] for pointer_id in delta.evidence_pointer_ids)
        if any(pointer.kind not in delta_kinds[delta.kind] for pointer in delta_pointers):
            raise ValueError("observed delta references incompatible evidence")
        arms = {pointer.arm for pointer in delta_pointers}
        if delta.change == "changed" and not {"source", "probe"}.issubset(arms):
            raise ValueError("changed deltas require source and probe evidence")
        if delta.change in {"added", "violated", "unstable"} and "probe" not in arms:
            raise ValueError(f"{delta.change} deltas require probe evidence")
        if delta.change == "removed" and "source" not in arms:
            raise ValueError("removed deltas require source evidence")
    if occurrence.category == "duplicate_effect":
        duplicate_delta = next(
            delta
            for delta in occurrence.observed_deltas
            if delta.kind == "action" and delta.change == "added"
        )
        if {pointers[pointer_id].arm for pointer_id in duplicate_delta.evidence_pointer_ids} != {
            "source",
            "probe",
        }:
            raise ValueError("duplicate effects require matching source and probe action evidence")
    if occurrence.violated_rule is not None:
        if any(
            pointers[pointer_id].authority != "customer_declared"
            for pointer_id in occurrence.rule_definition_evidence_pointer_ids
        ):
            raise ValueError("customer rule definitions require customer-declared evidence")
        violation_delta = next(
            delta
            for delta in occurrence.observed_deltas
            if delta.kind == "rule" and delta.change == "violated"
        )
        if any(
            pointers[pointer_id].authority
            not in {"deterministic_evaluator", "independent_observer"}
            for pointer_id in violation_delta.evidence_pointer_ids
        ):
            raise ValueError("rule violations require evaluator or independent evidence")


class FindingSummary(_StrictModel):
    finding_id: str | None = Field(default=None, pattern=_FINDING_ID_PATTERN)
    kind: FindingKind
    category: FindingCategory
    operator_id: str | None = Field(default=None, pattern=_IDENTIFIER_PATTERN)
    operator_version: str | None = Field(default=None, pattern=_VERSION_PATTERN)
    rule_id: str | None = Field(default=None, pattern=_IDENTIFIER_PATTERN)
    rule_version: str | None = Field(default=None, pattern=_VERSION_PATTERN)
    declared_severity: FindingSeverity | None = None
    review_status: FindingReviewStatus | None = None
    review_severity: FindingSeverity | None = None
    requested_repetitions: int = Field(ge=1)
    conclusive_repetitions: int = Field(ge=0)
    inconclusive_repetitions: int = Field(ge=0)
    stability: Literal["stable", "unstable", "inconclusive"] | None = None
    violated_repetitions: int | None = Field(default=None, ge=0)
    next_action: FindingNextAction
    summary: FindingSummaryText

    @model_validator(mode="after")
    def validate_finding(self) -> Self:
        if (self.operator_id is None) != (self.operator_version is None):
            raise ValueError("operator ID and version must be present together")
        if (self.rule_id is None) != (self.rule_version is None):
            raise ValueError("rule ID and version must be present together")
        if (self.review_status is None) != (self.review_severity is None):
            raise ValueError("review status and severity must be present together")
        if self.review_status not in {None, "confirmed"} and self.review_severity != "unrated":
            raise ValueError("only confirmed findings can have a rated review severity")
        if (
            self.conclusive_repetitions + self.inconclusive_repetitions
            != self.requested_repetitions
        ):
            raise ValueError("finding repetition counts must match requested repetitions")
        if (
            self.violated_repetitions is not None
            and self.violated_repetitions > self.conclusive_repetitions
        ):
            raise ValueError("violated repetitions cannot exceed conclusive repetitions")
        if self.kind == "behavior_difference":
            expected_summary = _BEHAVIOR_SUMMARIES.get(self.category)
            if (
                expected_summary is None
                or self.rule_id is not None
                or self.declared_severity is not None
                or self.summary != expected_summary
            ):
                raise ValueError("behavior finding fields are inconsistent")
        elif (
            self.category != "customer_invariant_violation"
            or self.rule_id is None
            or self.declared_severity is None
            or self.summary != "The agent violated a customer-defined rule."
        ):
            raise ValueError("invariant finding fields are inconsistent")
        return self


class PatternOperator(_StrictModel):
    operator_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    operator_version: str = Field(pattern=_VERSION_PATTERN)
    summary: str | None = Field(default=None, min_length=1, max_length=500)


def _effective_finding_severity(finding: FindingSummary) -> FindingSeverity:
    if finding.review_status == "confirmed":
        if finding.review_severity is None:
            raise AssertionError("validated confirmed finding requires review severity")
        return finding.review_severity
    return finding.declared_severity or "unrated"


class FailurePattern(_StrictModel):
    pattern_id: str = Field(pattern=_PATTERN_ID_PATTERN)
    kind: FindingKind
    category: FindingCategory
    rule_id: str | None = Field(default=None, pattern=_IDENTIFIER_PATTERN)
    rule_version: str | None = Field(default=None, pattern=_VERSION_PATTERN)
    summary: FindingSummaryText
    severity: FindingSeverity
    finding_count: int = Field(ge=1)
    source_case_count: int = Field(ge=1)
    operators: tuple[PatternOperator, ...] = Field(min_length=1)
    needs_review_count: int = Field(ge=0)
    confirmed_count: int = Field(ge=0)
    finding_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_pattern(self) -> Self:
        if (self.rule_id is None) != (self.rule_version is None):
            raise ValueError("pattern rule ID and version must be present together")
        if self.kind == "behavior_difference" and self.rule_id is not None:
            raise ValueError("behavior patterns cannot reference a customer rule")
        if self.kind == "customer_invariant_violation" and self.rule_id is None:
            raise ValueError("invariant patterns require a customer rule")
        if self.finding_count != len(self.finding_ids):
            raise ValueError("pattern finding count must match finding IDs")
        if self.source_case_count > self.finding_count:
            raise ValueError("pattern source case count cannot exceed finding count")
        if self.finding_ids != tuple(sorted(set(self.finding_ids))):
            raise ValueError("pattern finding IDs must be sorted and unique")
        if self.needs_review_count + self.confirmed_count != self.finding_count:
            raise ValueError("pattern review counts must match finding count")
        operator_keys = tuple(
            (operator.operator_id, operator.operator_version) for operator in self.operators
        )
        if operator_keys != tuple(sorted(set(operator_keys))):
            raise ValueError("pattern operators must be sorted and unique")
        return self


class ReviewStatusCounts(_StrictModel):
    needs_review: int = Field(ge=0)
    confirmed: int = Field(ge=0)
    expected: int = Field(ge=0)
    unsupported: int = Field(ge=0)
    inconclusive: int = Field(ge=0)


class ReportSummary(_StrictModel):
    finding_count: int = Field(ge=0)
    actionable_finding_count: int = Field(ge=0)
    review_status_counts: ReviewStatusCounts


def build_report_summary(findings: tuple[FindingSummary, ...]) -> ReportSummary:
    return ReportSummary(
        finding_count=len(findings),
        actionable_finding_count=sum(
            finding.review_status in {None, "needs_review", "confirmed"} for finding in findings
        ),
        review_status_counts=ReviewStatusCounts(
            needs_review=sum(finding.review_status == "needs_review" for finding in findings),
            confirmed=sum(finding.review_status == "confirmed" for finding in findings),
            expected=sum(finding.review_status == "expected" for finding in findings),
            unsupported=sum(finding.review_status == "unsupported" for finding in findings),
            inconclusive=sum(finding.review_status == "inconclusive" for finding in findings),
        ),
    )


class UnifiedReport(_StrictModel):
    schema_version: Literal["1.5.0"] = "1.5.0"
    evidence_type: ReportEvidenceType
    evidence_schema_versions: tuple[str, ...] = Field(min_length=1)
    evidence_scope: ReportEvidenceScope
    evaluation_mode: ReportEvaluationMode | None = None
    capability_limitations: tuple[ReportCapabilityLimitation, ...] = ()
    review_status: ReportReviewStatus
    exit_code: Literal[0, 1, 2]
    summary: ReportSummary
    patterns: tuple[FailurePattern, ...] = ()
    findings: tuple[FindingSummary, ...] = ()

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.evidence_type != "dataset_evaluation" and self.evaluation_mode is not None:
            raise ValueError("evaluation mode is available only for dataset reports")
        if self.capability_limitations != tuple(sorted(set(self.capability_limitations))):
            raise ValueError("capability limitations must be sorted and unique")
        if self.evidence_scope == "response_only" and self.capability_limitations != (
            "cleanup_verification",
            "conversation_replay",
            "state_observation",
        ):
            raise ValueError("response-only reports must name every unverified capability")
        if self.evidence_scope == "response_and_state" and self.capability_limitations:
            raise ValueError("response-and-state reports cannot name adapter limitations")
        if self.evidence_schema_versions != tuple(sorted(set(self.evidence_schema_versions))):
            raise ValueError("evidence schema versions must be sorted and unique")
        expected_exit_code = {"resolved": 0, "action_required": 1, "inconclusive": 2}[
            self.review_status
        ]
        if self.exit_code != expected_exit_code:
            raise ValueError("exit code must match report review status")
        if self.summary.finding_count != len(self.findings):
            raise ValueError("finding count must match report findings")
        pattern_ids = tuple(pattern.pattern_id for pattern in self.patterns)
        expected_pattern_order = tuple(
            pattern.pattern_id
            for pattern in sorted(
                self.patterns,
                key=lambda pattern: (-_SEVERITY_RANK[pattern.severity], pattern.pattern_id),
            )
        )
        if len(pattern_ids) != len(set(pattern_ids)) or pattern_ids != expected_pattern_order:
            raise ValueError("patterns must be sorted and unique")
        known_findings = {
            finding.finding_id: finding
            for finding in self.findings
            if finding.finding_id is not None
        }
        patterned_findings: set[str] = set()
        for pattern in self.patterns:
            pattern_findings: list[FindingSummary] = []
            for finding_id in pattern.finding_ids:
                finding = known_findings.get(finding_id)
                if finding is None:
                    raise ValueError("pattern references an unknown finding")
                if finding_id in patterned_findings:
                    raise ValueError("finding cannot belong to multiple patterns")
                if (
                    finding.kind != pattern.kind
                    or finding.category != pattern.category
                    or finding.rule_id != pattern.rule_id
                    or finding.rule_version != pattern.rule_version
                    or finding.summary != pattern.summary
                    or finding.review_status not in {"needs_review", "confirmed"}
                ):
                    raise ValueError("pattern fields must match its actionable findings")
                patterned_findings.add(finding_id)
                pattern_findings.append(finding)
            if pattern.needs_review_count != sum(
                finding.review_status == "needs_review" for finding in pattern_findings
            ) or pattern.confirmed_count != sum(
                finding.review_status == "confirmed" for finding in pattern_findings
            ):
                raise ValueError("pattern review counts must match member findings")
            member_severities: list[FindingSeverity] = [
                _effective_finding_severity(finding) for finding in pattern_findings
            ]
            expected_severity = sorted(
                member_severities,
                key=_SEVERITY_RANK.__getitem__,
            )[-1]
            if pattern.severity != expected_severity:
                raise ValueError("pattern severity must match member findings")
        expected_review_counts = {
            status: sum(finding.review_status == status for finding in self.findings)
            for status in (
                "needs_review",
                "confirmed",
                "expected",
                "unsupported",
                "inconclusive",
            )
        }
        if self.summary.review_status_counts.model_dump() != expected_review_counts:
            raise ValueError("review status counts must match report findings")
        expected_actionable_count = sum(
            finding.review_status in {None, "needs_review", "confirmed"}
            for finding in self.findings
        )
        if self.summary.actionable_finding_count != expected_actionable_count:
            raise ValueError("actionable finding count must match report findings")
        if self.review_status == "resolved" and (
            self.summary.actionable_finding_count or self.summary.review_status_counts.inconclusive
        ):
            raise ValueError("resolved reports cannot contain unresolved findings")
        if self.review_status == "action_required" and not self.summary.actionable_finding_count:
            raise ValueError("action-required reports require an actionable finding")
        return self
