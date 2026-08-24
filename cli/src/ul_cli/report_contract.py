from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime
from typing import Annotated, Literal, Self, cast

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
DecisionEvidenceScope = Literal["response_only", "response_and_state", "mixed"]
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
PatternEvidenceAuthority = Literal[
    "customer_declared",
    "deterministic_evaluator",
    "independent_observer",
    "model_derived_unverified",
]
PatternEvidenceLimitation = Literal["semantic_model_output_not_independently_verified"]

_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$"
_VERSION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,49}$"
_FINDING_ID_PATTERN = r"^ulf_v1_[0-9a-f]{64}$"
_PATTERN_FINGERPRINT_PATTERN = r"^ulpf_v1_[0-9a-f]{64}$"
_PATTERN_SNAPSHOT_ID_PATTERN = r"^ulps_v1_[0-9a-f]{64}$"
_PATTERN_MECHANISM_PSEUDONYM_PATTERN = r"^ulpm_v1_[0-9a-f]{64}$"
_PATTERN_EVIDENCE_REFERENCE_PATTERN = r"^ulpe_v1_[0-9a-f]{64}$"
_PATTERN_REVIEW_ID_PATTERN = (
    r"^ulpr_[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_JSON_POINTER_PATTERN = re.compile(r"(?:/(?:[^~/]|~[01])*)*")
_MAXIMUM_RUN_RECEIPT_BYTES = 1_000_000
_MAXIMUM_PATTERN_FINGERPRINT_BYTES = 16_000
_MAXIMUM_PATTERN_SNAPSHOT_BYTES = 1_000_000
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
_MAXIMUM_FINDING_PACKAGE_BYTES = 16_000_000
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
EvidenceAuthority = Literal[
    "customer_declared",
    "deterministic_evaluator",
    "invoker_self_reported",
    "source_self_reported",
    "environment_self_reported",
    "independent_observer",
]
DecisionFindingClassification = Literal[
    "observed_variance",
    "customer_rule_violation",
    "inconclusive_evidence",
]
DecisionClaimKind = Literal[
    "tested_change",
    "agent_behavior",
    "observed_consequence",
    "flag_reason",
]
CrossExaminationSignal = Literal["observed", "not_observed", "inconclusive"]
CrossExaminationEvidenceLevel = Literal[
    "response_observed",
    "trajectory_observed",
    "committed_state_verified",
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
    authority: EvidenceAuthority
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


class CrossExaminationArm(_StrictModel):
    role: Literal["historical_reference", "current_baseline", "variation"]
    response_evidence_pointer_ids: tuple[str, ...] = Field(min_length=1, max_length=1_000)
    requested_repetitions: int = Field(ge=0)
    observed_repetitions: int = Field(ge=0)
    inconclusive_repetitions: int = Field(ge=0)
    stability: Literal["stable", "unstable", "inconclusive", "not_applicable"]

    @model_validator(mode="after")
    def validate_arm(self) -> Self:
        _validate_sorted_unique(
            self.response_evidence_pointer_ids,
            "cross-examination response evidence pointer IDs",
        )
        if self.role == "historical_reference":
            if (
                self.requested_repetitions,
                self.observed_repetitions,
                self.inconclusive_repetitions,
                self.stability,
            ) != (0, 0, 0, "not_applicable"):
                raise ValueError("historical references are observations, not executions")
        elif (
            self.observed_repetitions + self.inconclusive_repetitions != self.requested_repetitions
            or self.stability == "not_applicable"
        ):
            raise ValueError("executed cross-examination arms require exact repetition counts")
        return self


class FindingCrossExamination(_StrictModel):
    historical_reference: CrossExaminationArm
    current_baseline: CrossExaminationArm
    variation: CrossExaminationArm
    augmentation_relation: VersionedReference
    baseline_drift: CrossExaminationSignal
    augmentation_sensitivity: CrossExaminationSignal
    intrinsic_instability: CrossExaminationSignal
    material_delta_evidence_pointer_ids: tuple[str, ...] = Field(min_length=1, max_length=1_000)
    evidence_level: CrossExaminationEvidenceLevel
    limitations: tuple[
        Literal[
            "causality_not_established",
            "correctness_not_verified",
            "historical_reference_not_an_oracle",
        ],
        ...,
    ]

    @model_validator(mode="after")
    def validate_cross_examination(self) -> Self:
        if (
            self.historical_reference.role != "historical_reference"
            or self.current_baseline.role != "current_baseline"
            or self.variation.role != "variation"
        ):
            raise ValueError("cross-examination arms must match their declared roles")
        if len(self.historical_reference.response_evidence_pointer_ids) != 1:
            raise ValueError("cross-examination requires one historical response reference")
        _validate_sorted_unique(
            self.material_delta_evidence_pointer_ids,
            "cross-examination material delta evidence pointer IDs",
        )
        _validate_sorted_unique(self.limitations, "cross-examination limitations")
        required_limitations = {
            "causality_not_established",
            "correctness_not_verified",
            "historical_reference_not_an_oracle",
        }
        if set(self.limitations) != required_limitations:
            raise ValueError("variance cross-examination requires every interpretation limit")
        unstable = "unstable" in {
            self.current_baseline.stability,
            self.variation.stability,
        }
        baseline_incomplete = self.current_baseline.stability == "inconclusive"
        comparison_incomplete = "inconclusive" in {
            self.current_baseline.stability,
            self.variation.stability,
        }
        expected_instability: CrossExaminationSignal = (
            "inconclusive" if comparison_incomplete else "observed" if unstable else "not_observed"
        )
        if self.intrinsic_instability != expected_instability:
            raise ValueError("intrinsic instability must match exact arm stability")
        if baseline_incomplete and self.baseline_drift != "inconclusive":
            raise ValueError("incomplete baseline executions make baseline drift inconclusive")
        if comparison_incomplete and self.augmentation_sensitivity != "inconclusive":
            raise ValueError("incomplete executions make augmentation sensitivity inconclusive")
        if unstable and self.augmentation_sensitivity != "inconclusive":
            raise ValueError("unstable executions cannot support augmentation sensitivity")
        return self


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
    cross_examination: FindingCrossExamination | None = None
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
        if self.kind == "behavior_difference" and self.source_interaction_ref is not None:
            if self.cross_examination is None:
                raise ValueError("dataset behavior occurrences require cross-examination evidence")
            if self.cross_examination.augmentation_relation != self.operator:
                raise ValueError("cross-examination relation must match the finding operator")
        elif self.cross_examination is not None:
            raise ValueError("cross-examination is available only for dataset behavior findings")
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
            *(
                (
                    pointer_id
                    for arm in (
                        self.cross_examination.historical_reference,
                        self.cross_examination.current_baseline,
                        self.cross_examination.variation,
                    )
                    for pointer_id in arm.response_evidence_pointer_ids
                )
                if self.cross_examination is not None
                else ()
            ),
            *(
                self.cross_examination.material_delta_evidence_pointer_ids
                if self.cross_examination is not None
                else ()
            ),
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


class EvidenceArtifact(_StrictModel):
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    value: CapturedJson

    @model_validator(mode="after")
    def validate_digest(self) -> Self:
        if self.artifact_sha256 != self.value.sha256:
            raise ValueError("evidence artifact digest must match its retained value")
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
    historical_reference_response: ReceiptEvidenceValue | None = None
    response: ReceiptEvidenceValue | None = None
    tool_exchanges: tuple[ToolExchangeReceipt, ...] = Field(default=(), max_length=1_000)
    state_before: StateReceipt | None = None
    state_after: StateReceipt | None = None
    lifecycle: tuple[LifecycleReceipt, ...] = Field(min_length=1, max_length=10)
    provenance: tuple[ProvenanceReceipt, ...] = Field(min_length=1, max_length=100)
    trace_evidence_pointer_ids: tuple[str, ...] = Field(default=(), max_length=1_000)
    usage: UsageReceipt | None = None
    redaction: RedactionReceipt | None = None
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
        if self.historical_reference_response is not None:
            referenced.add(self.historical_reference_response.evidence_pointer_id)
            _validate_receipt_value_pointer(
                self.historical_reference_response,
                pointers,
                "response",
                "shared",
            )
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
        if (self.redaction is None) != ("redaction_accounting_unavailable" in self.limitations):
            raise ValueError(
                "receipts require producer redaction accounting or an explicit limitation"
            )
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


class FindingPrivateReferences(_StrictModel):
    disclosure: Literal["private"] = "private"
    campaign_id: str = Field(min_length=1, max_length=500)
    case_id: str = Field(min_length=1, max_length=500)
    source_interaction_id: str | None = Field(default=None, min_length=1, max_length=500)
    operator_id: str = Field(min_length=1, max_length=500)
    operator_version: str = Field(min_length=1, max_length=100)
    rule_id: str | None = Field(default=None, min_length=1, max_length=500)
    rule_version: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_rule_reference(self) -> Self:
        if (self.rule_id is None) != (self.rule_version is None):
            raise ValueError("private rule ID and version must be provided together")
        return self


class FindingEvidencePackage(_StrictModel):
    schema_version: Literal["1.1.0"] = "1.1.0"
    disclosure: Literal["private"] = "private"
    occurrence: FindingOccurrence
    private_references: FindingPrivateReferences
    receipts: tuple[RunReceipt, ...] = Field(min_length=1, max_length=2_000)
    artifact_retention: Literal["external", "embedded"] = "external"
    artifacts: tuple[EvidenceArtifact, ...] = Field(default=(), max_length=4_000)

    @model_validator(mode="after")
    def validate_package(self) -> Self:
        if (self.occurrence.source_interaction_ref is None) != (
            self.private_references.source_interaction_id is None
        ):
            raise ValueError("private source reference must match the public occurrence shape")
        if (self.occurrence.kind == "customer_invariant_violation") != (
            self.private_references.rule_id is not None
        ):
            raise ValueError("private rule reference must match the finding kind")
        artifact_digests = tuple(artifact.artifact_sha256 for artifact in self.artifacts)
        if artifact_digests != tuple(sorted(set(artifact_digests))):
            raise ValueError("retained evidence artifacts must be sorted and unique")
        if self.artifact_retention == "external":
            if self.artifacts:
                raise ValueError("externally retained packages cannot embed artifacts")
        elif not self.artifacts:
            raise ValueError("embedded packages require retained evidence artifacts")
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
        historical_reference_pointer_ids = tuple(
            receipt.content.historical_reference_response.evidence_pointer_id
            for receipt in self.receipts
            if receipt.content.historical_reference_response is not None
        )
        expected_historical_reference_pointer_ids = (
            self.occurrence.cross_examination.historical_reference.response_evidence_pointer_ids
            if self.occurrence.cross_examination is not None
            else ()
        )
        if historical_reference_pointer_ids != expected_historical_reference_pointer_ids:
            raise ValueError("package must retain exactly the cited historical response reference")
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
        if self.artifact_retention == "embedded":
            artifacts = {artifact.artifact_sha256: artifact.value for artifact in self.artifacts}
            for pointer_id in self.occurrence.evidence_pointer_ids:
                pointer = pointers[pointer_id]
                artifact = artifacts.get(pointer.artifact_sha256)
                if artifact is None:
                    raise ValueError("package does not retain every cited evidence artifact")
                _resolve_json_pointer(
                    json.loads(artifact.canonical_json),
                    pointer.json_pointer,
                )
            receipt_values = tuple(
                value
                for receipt in self.receipts
                for value in _receipt_evidence_values(receipt.content)
            )
            for receipt_value in receipt_values:
                pointer = pointers[receipt_value.evidence_pointer_id]
                artifact = artifacts.get(pointer.artifact_sha256)
                if artifact is not None and not _receipt_value_matches_artifact(
                    _resolve_json_pointer(
                        json.loads(artifact.canonical_json),
                        pointer.json_pointer,
                    ),
                    receipt_value,
                ):
                    raise ValueError("receipt value must match its retained evidence artifact")
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
        if (
            len(_canonical_json(self.model_dump(mode="json")).encode("utf-8"))
            > _MAXIMUM_FINDING_PACKAGE_BYTES
        ):
            raise ValueError("finding evidence package exceeds the 16 MB JSON limit")
        return self


class FindingDecisionClaim(_StrictModel):
    kind: DecisionClaimKind
    summary: str = Field(min_length=1, max_length=500)
    evidence_pointer_ids: tuple[str, ...] = Field(min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        _validate_sorted_unique(self.evidence_pointer_ids, "decision claim evidence pointers")
        return self


class FindingDecisionLimitation(_StrictModel):
    code: OccurrenceLimitation
    summary: str = Field(min_length=1, max_length=500)


class DecisionReadyFinding(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    occurrence_id: str = Field(pattern=_FINDING_ID_PATTERN)
    campaign_ref: str = Field(pattern=_PUBLIC_REFERENCE_PATTERN)
    case_ref: str = Field(pattern=_PUBLIC_REFERENCE_PATTERN)
    operator: VersionedReference
    probe_change_kind: Literal[
        "input",
        "context",
        "turn_sequence",
        "state_setup",
        "event_behavior",
    ]
    violated_rule: VersionedReference | None = None
    classification: DecisionFindingClassification
    headline: FindingSummaryText
    claims: tuple[FindingDecisionClaim, ...] = Field(min_length=4, max_length=4)
    repetition_summary: RepetitionSummary
    evidence_scope: DecisionEvidenceScope
    evidence_level: Literal[
        "response_observed",
        "response_and_state_observed",
        "customer_rule_evaluated",
    ]
    evidence_authorities: tuple[EvidenceAuthority, ...] = Field(min_length=1, max_length=10)
    limitations: tuple[FindingDecisionLimitation, ...] = ()
    review_workflow: Literal["dataset_review", "external_review_required"]
    review_status: Literal["needs_review"] = "needs_review"
    human_confirmed_severity: Literal["unrated"] = "unrated"
    next_action: FindingNextAction
    next_action_summary: str = Field(min_length=1, max_length=500)
    receipt_ids: tuple[str, ...] = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        claim_kinds = tuple(claim.kind for claim in self.claims)
        if claim_kinds != (
            "tested_change",
            "agent_behavior",
            "observed_consequence",
            "flag_reason",
        ):
            raise ValueError("decision claims must be complete and ordered")
        if self.evidence_authorities != tuple(sorted(set(self.evidence_authorities))):
            raise ValueError("decision evidence authorities must be sorted and unique")
        limitation_codes = tuple(limitation.code for limitation in self.limitations)
        if limitation_codes != tuple(sorted(set(limitation_codes))):
            raise ValueError("decision limitations must be sorted and unique")
        _validate_sorted_unique(self.receipt_ids, "decision receipt IDs")
        if (self.review_workflow == "dataset_review") != (
            self.next_action == "review_dataset_finding"
        ):
            raise ValueError("decision review workflow must match its next action")
        return self


class FindingDecisionReport(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    disclosure: Literal["safe"] = "safe"
    evidence_schema_versions: tuple[str, ...] = Field(min_length=1, max_length=100)
    campaign_ref: str = Field(pattern=_PUBLIC_REFERENCE_PATTERN)
    evidence_scope: DecisionEvidenceScope
    review_status: Literal["action_required", "inconclusive"]
    exit_code: Literal[1, 2]
    findings: tuple[DecisionReadyFinding, ...] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.evidence_schema_versions != tuple(sorted(set(self.evidence_schema_versions))):
            raise ValueError("decision report evidence versions must be sorted and unique")
        occurrence_ids = tuple(finding.occurrence_id for finding in self.findings)
        if occurrence_ids != tuple(sorted(set(occurrence_ids))):
            raise ValueError("decision-ready findings must be sorted and unique")
        if any(finding.campaign_ref != self.campaign_ref for finding in self.findings):
            raise ValueError("decision report campaign must match every finding")
        expected_scope: DecisionEvidenceScope = (
            self.findings[0].evidence_scope
            if len({finding.evidence_scope for finding in self.findings}) == 1
            else "mixed"
        )
        if self.evidence_scope != expected_scope:
            raise ValueError("decision report evidence scope must summarize its findings")
        expected_status = (
            "inconclusive"
            if all(finding.classification == "inconclusive_evidence" for finding in self.findings)
            else "action_required"
        )
        if self.review_status != expected_status:
            raise ValueError("decision report status must match its findings")
        if self.exit_code != (2 if self.review_status == "inconclusive" else 1):
            raise ValueError("decision report exit code must match its status")
        return self


def capture_json(value: JsonValue) -> CapturedJson:
    canonical_json = _canonical_json(value)
    return CapturedJson(
        canonical_json=canonical_json,
        sha256=hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
    )


def build_run_receipt(content: RunReceiptContent) -> RunReceipt:
    return RunReceipt(receipt_id=_run_receipt_id(content), content=content)


_AGENT_BEHAVIOR_SUMMARIES: dict[FindingCategory, str] = {
    "duplicate_effect": "The probe made the agent repeat an observed action.",
    "unexpected_effect": "The probe made the agent produce a new observed action.",
    "missing_effect": "The probe made the agent omit an observed source action.",
    "changed_grounded_effect_argument": (
        "The probe made the agent change an observed action detail."
    ),
    "unstable_behavior": "The agent produced inconsistent observed behavior.",
}
_CONSEQUENCE_SUMMARIES: dict[FindingCategory, FindingSummaryText] = {
    "duplicate_effect": "The changed input made the agent repeat an action.",
    "unexpected_effect": "The changed input made the agent take a new action.",
    "missing_effect": "The changed input made the agent skip a baseline action.",
    "changed_grounded_effect_argument": "The changed input altered an important action detail.",
    "unstable_behavior": "The changed input produced inconsistent behavior across repetitions.",
    "customer_invariant_violation": "The agent violated a customer-defined rule.",
}
_LIMITATION_SUMMARIES: dict[OccurrenceLimitation, str] = {
    "correctness_not_verified": (
        "UL observed a difference; it did not determine whether the behavior was correct."
    ),
    "evaluator_provenance_unavailable": "Evaluator provenance was not available.",
    "model_provenance_unavailable": "Model provenance was not available.",
    "one_or_more_repetitions_inconclusive": (
        "At least one requested repetition did not produce conclusive evidence."
    ),
    "production_prevalence_not_measured": (
        "Generated probe repetitions do not measure production prevalence."
    ),
    "source_execution_unavailable": (
        "At least one source execution was unavailable for direct comparison."
    ),
}


def build_finding_decision(package: FindingEvidencePackage) -> DecisionReadyFinding:
    if package.artifact_retention != "embedded":
        raise ValueError("decision-ready findings require embedded evidence artifacts")
    occurrence = package.occurrence
    pointers = {
        pointer.pointer_id: pointer
        for receipt in package.receipts
        for pointer in receipt.content.evidence_pointers
    }
    tested_change_pointer_ids = tuple(
        sorted(
            {
                *occurrence.probe_change.source_evidence_pointer_ids,
                *occurrence.probe_change.probe_evidence_pointer_ids,
            }
        )
    )
    category_pointer_ids = tuple(
        sorted(
            {
                pointer_id
                for delta in _category_supporting_deltas(occurrence)
                for pointer_id in delta.evidence_pointer_ids
            }
        )
    )
    flag_pointer_ids = tuple(
        sorted({*category_pointer_ids, *occurrence.rule_definition_evidence_pointer_ids})
    )
    probe_change_summary = {
        "input": "UL compared the recorded input with an augmented input.",
        "context": "UL compared the baseline context with augmented context.",
        "turn_sequence": "UL compared the baseline turns with augmented turns.",
        "state_setup": "UL compared declared state setup with modified state setup.",
        "event_behavior": "UL compared normal event behavior with an injected event behavior.",
    }[occurrence.probe_change.kind]
    classification: DecisionFindingClassification = (
        "inconclusive_evidence"
        if occurrence.repetition_summary.observed == 0
        else "customer_rule_violation"
        if occurrence.kind == "customer_invariant_violation"
        else "observed_variance"
    )
    flag_summary = {
        "observed_variance": (
            "UL flagged an observed source-versus-probe difference for human review."
        ),
        "customer_rule_violation": ("UL flagged evidence that violated a customer-defined rule."),
        "inconclusive_evidence": (
            "UL could not establish a conclusive source-versus-probe result."
        ),
    }[classification]
    receipt_scopes = {receipt.content.evidence_scope for receipt in package.receipts}
    evidence_scope: DecisionEvidenceScope = (
        cast(ReportEvidenceScope, receipt_scopes.pop()) if len(receipt_scopes) == 1 else "mixed"
    )
    cited_pointer_ids = {
        *tested_change_pointer_ids,
        *category_pointer_ids,
        *flag_pointer_ids,
    }
    evidence_authorities = cast(
        tuple[EvidenceAuthority, ...],
        tuple(sorted({pointers[pointer_id].authority for pointer_id in cited_pointer_ids})),
    )
    next_action_summary = (
        "Record a human decision in the dataset review workflow."
        if occurrence.next_action == "review_dataset_finding"
        else "Inspect the private normalized receipt before making a decision."
    )
    agent_behavior_summary = (
        "The probe arm violated the customer rule."
        if occurrence.kind == "customer_invariant_violation"
        else _AGENT_BEHAVIOR_SUMMARIES[occurrence.category]
    )
    evidence_level = (
        "customer_rule_evaluated"
        if occurrence.kind == "customer_invariant_violation"
        else "response_and_state_observed"
        if evidence_scope in {"response_and_state", "mixed"}
        else "response_observed"
    )
    return DecisionReadyFinding(
        occurrence_id=occurrence.occurrence_id,
        campaign_ref=occurrence.campaign_ref,
        case_ref=occurrence.case_ref,
        operator=occurrence.operator,
        probe_change_kind=occurrence.probe_change.kind,
        violated_rule=occurrence.violated_rule,
        classification=classification,
        headline=_CONSEQUENCE_SUMMARIES[occurrence.category],
        claims=(
            FindingDecisionClaim(
                kind="tested_change",
                summary=probe_change_summary,
                evidence_pointer_ids=tested_change_pointer_ids,
            ),
            FindingDecisionClaim(
                kind="agent_behavior",
                summary=agent_behavior_summary,
                evidence_pointer_ids=category_pointer_ids,
            ),
            FindingDecisionClaim(
                kind="observed_consequence",
                summary=_CONSEQUENCE_SUMMARIES[occurrence.category],
                evidence_pointer_ids=category_pointer_ids,
            ),
            FindingDecisionClaim(
                kind="flag_reason",
                summary=flag_summary,
                evidence_pointer_ids=flag_pointer_ids,
            ),
        ),
        repetition_summary=occurrence.repetition_summary,
        evidence_scope=evidence_scope,
        evidence_level=evidence_level,
        evidence_authorities=evidence_authorities,
        limitations=tuple(
            FindingDecisionLimitation(code=code, summary=_LIMITATION_SUMMARIES[code])
            for code in occurrence.limitations
        ),
        review_workflow=(
            "dataset_review"
            if occurrence.next_action == "review_dataset_finding"
            else "external_review_required"
        ),
        next_action=occurrence.next_action,
        next_action_summary=next_action_summary,
        receipt_ids=tuple(receipt.receipt_id for receipt in package.receipts),
    )


def build_finding_decision_report(
    packages: tuple[FindingEvidencePackage, ...],
) -> FindingDecisionReport:
    if not packages:
        raise ValueError("finding decision report requires at least one package")
    findings = tuple(
        sorted(
            (build_finding_decision(package) for package in packages),
            key=lambda finding: finding.occurrence_id,
        )
    )
    campaign_refs = {finding.campaign_ref for finding in findings}
    if len(campaign_refs) != 1:
        raise ValueError("finding decision report requires one campaign")
    scopes = {finding.evidence_scope for finding in findings}
    evidence_scope = cast(DecisionEvidenceScope, scopes.pop() if len(scopes) == 1 else "mixed")
    review_status = (
        "inconclusive"
        if all(finding.classification == "inconclusive_evidence" for finding in findings)
        else "action_required"
    )
    return FindingDecisionReport(
        evidence_schema_versions=tuple(
            sorted(
                {package.schema_version for package in packages}
                | {package.occurrence.schema_version for package in packages}
                | {receipt.schema_version for package in packages for receipt in package.receipts}
            )
        ),
        campaign_ref=campaign_refs.pop(),
        evidence_scope=evidence_scope,
        review_status=review_status,
        exit_code=2 if review_status == "inconclusive" else 1,
        findings=findings,
    )


def build_finding_occurrence(**values: object) -> FindingOccurrence:
    values["occurrence_id"] = f"ulf_v1_{'0' * 64}"
    occurrence = FindingOccurrence.model_validate(
        values,
        context={"building_occurrence": True},
    )
    values["occurrence_id"] = _finding_occurrence_id(occurrence)
    return FindingOccurrence.model_validate(values)


def build_failure_pattern(**values: object) -> FailurePattern:
    values["pattern_fingerprint"] = f"ulpf_v1_{'0' * 64}"
    values["pattern_snapshot_id"] = f"ulps_v1_{'0' * 64}"
    pattern = FailurePattern.model_validate(
        values,
        context={"building_pattern": True},
    )
    values["pattern_fingerprint"] = _pattern_fingerprint(pattern)
    pattern = FailurePattern.model_validate(
        values,
        context={"building_pattern": True},
    )
    values["pattern_snapshot_id"] = _pattern_snapshot_id(pattern)
    return FailurePattern.model_validate(values)


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


def _resolve_json_pointer(value: JsonValue, pointer: str) -> JsonValue:
    current = value
    if pointer == "":
        return current
    for encoded_token in pointer[1:].split("/"):
        token = encoded_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif (
            isinstance(current, list)
            and token.isascii()
            and token.isdecimal()
            and (token == "0" or not token.startswith("0"))
            and int(token) < len(current)
        ):
            current = current[int(token)]
        else:
            raise ValueError("evidence pointer does not resolve in its retained artifact")
    return current


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


def _pattern_fingerprint(pattern: FailurePattern) -> str:
    grouping_facets = pattern.model_dump(
        mode="json",
        include={
            "kind",
            "category",
            "rule_id",
            "rule_version",
            "stability",
            "evidence_authorities",
            "evidence_limitations",
            "horizontal_facets",
            "vertical_facets",
        },
    )
    if pattern.vertical_facets is None:
        grouping_facets.pop("vertical_facets")
    canonical_grouping_facets = _canonical_json(grouping_facets).encode("utf-8")
    if len(canonical_grouping_facets) > _MAXIMUM_PATTERN_FINGERPRINT_BYTES:
        raise ValueError("pattern fingerprint input exceeds the size limit")
    digest = hashlib.sha256(canonical_grouping_facets)
    return f"ulpf_v1_{digest.hexdigest()}"


def _pattern_snapshot_id(pattern: FailurePattern) -> str:
    snapshot = _canonical_json(
        pattern.model_dump(mode="json", exclude={"pattern_snapshot_id"})
    ).encode("utf-8")
    if len(snapshot) > _MAXIMUM_PATTERN_SNAPSHOT_BYTES:
        raise ValueError("pattern snapshot input exceeds the size limit")
    digest = hashlib.sha256(snapshot)
    return f"ulps_v1_{digest.hexdigest()}"


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


def _receipt_evidence_values(content: RunReceiptContent) -> tuple[ReceiptEvidenceValue, ...]:
    return (
        content.input,
        *((content.response,) if content.response is not None else ()),
        *(exchange.call for exchange in content.tool_exchanges),
        *(exchange.result for exchange in content.tool_exchanges if exchange.result is not None),
        *((content.state_before.evidence,) if content.state_before is not None else ()),
        *((content.state_after.evidence,) if content.state_after is not None else ()),
    )


def _receipt_value_matches_artifact(
    artifact_value: JsonValue,
    receipt_value: ReceiptEvidenceValue,
) -> bool:
    return _canonical_json(artifact_value) == receipt_value.value.canonical_json


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
    if occurrence.cross_examination is not None:
        cross_examination = occurrence.cross_examination
        for pointer_id in cross_examination.historical_reference.response_evidence_pointer_ids:
            pointer = pointers[pointer_id]
            if (
                pointer.kind != "response"
                or pointer.arm != "shared"
                or pointer.authority != "customer_declared"
            ):
                raise ValueError(
                    "historical response references require customer-declared shared evidence"
                )
        for arm, expected_pointer_arm in (
            (cross_examination.current_baseline, "source"),
            (cross_examination.variation, "probe"),
        ):
            if any(
                pointers[pointer_id].kind != "response"
                or pointers[pointer_id].arm != expected_pointer_arm
                for pointer_id in arm.response_evidence_pointer_ids
            ):
                raise ValueError("current cross-examination arms require response evidence")
        if not set(cross_examination.material_delta_evidence_pointer_ids).issubset(
            {
                pointer_id
                for delta in occurrence.observed_deltas
                for pointer_id in delta.evidence_pointer_ids
            }
        ):
            raise ValueError("material cross-examination deltas must be observed finding deltas")
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


class CrossExaminationRunSummary(_StrictModel):
    requested_repetitions: int = Field(ge=0)
    observed_repetitions: int = Field(ge=0)
    inconclusive_repetitions: int = Field(ge=0)
    stability: Literal["stable", "unstable", "inconclusive", "not_applicable"]

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.stability == "not_applicable":
            if any(
                (
                    self.requested_repetitions,
                    self.observed_repetitions,
                    self.inconclusive_repetitions,
                )
            ):
                raise ValueError("non-executed arms cannot contain repetition counts")
        elif (
            self.observed_repetitions + self.inconclusive_repetitions != self.requested_repetitions
        ):
            raise ValueError("cross-examination summary counts must match repetitions")
        return self


class FindingCrossExaminationSummary(_StrictModel):
    historical_reference_available: bool
    current_baseline: CrossExaminationRunSummary
    variation: CrossExaminationRunSummary
    baseline_drift: CrossExaminationSignal
    augmentation_sensitivity: CrossExaminationSignal
    intrinsic_instability: CrossExaminationSignal
    material_delta_count: int = Field(ge=0)
    evidence_level: CrossExaminationEvidenceLevel
    limitations: tuple[
        Literal[
            "causality_not_established",
            "correctness_not_verified",
            "historical_reference_not_an_oracle",
        ],
        ...,
    ]

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if not self.historical_reference_available:
            raise ValueError("cross-examination requires historical reference evidence")
        _validate_sorted_unique(self.limitations, "cross-examination summary limitations")
        if set(self.limitations) != {
            "causality_not_established",
            "correctness_not_verified",
            "historical_reference_not_an_oracle",
        }:
            raise ValueError("cross-examination summary requires every interpretation limit")
        baseline_incomplete = self.current_baseline.stability == "inconclusive"
        comparison_incomplete = "inconclusive" in {
            self.current_baseline.stability,
            self.variation.stability,
        }
        unstable = "unstable" in {
            self.current_baseline.stability,
            self.variation.stability,
        }
        expected_instability: CrossExaminationSignal = (
            "inconclusive" if comparison_incomplete else "observed" if unstable else "not_observed"
        )
        if self.intrinsic_instability != expected_instability:
            raise ValueError("summary instability must match its arm stability")
        if baseline_incomplete and self.baseline_drift != "inconclusive":
            raise ValueError("incomplete summary baseline requires inconclusive drift")
        if comparison_incomplete and self.augmentation_sensitivity != "inconclusive":
            raise ValueError("incomplete summary arms require inconclusive sensitivity")
        if unstable and self.augmentation_sensitivity != "inconclusive":
            raise ValueError("unstable summary arms cannot support sensitivity")
        return self


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
    evidence_authorities: tuple[PatternEvidenceAuthority, ...] = ()
    evidence_limitations: tuple[PatternEvidenceLimitation, ...] = ()
    violated_repetitions: int | None = Field(default=None, ge=0)
    next_action: FindingNextAction
    summary: FindingSummaryText
    cross_examination: FindingCrossExaminationSummary | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_finding(self) -> Self:
        if self.evidence_authorities != tuple(sorted(set(self.evidence_authorities))):
            raise ValueError("finding evidence authorities must be sorted and unique")
        if self.evidence_limitations != tuple(sorted(set(self.evidence_limitations))):
            raise ValueError("finding evidence limitations must be sorted and unique")
        model_derived = "model_derived_unverified" in self.evidence_authorities
        if model_derived != (
            self.evidence_limitations == ("semantic_model_output_not_independently_verified",)
        ):
            raise ValueError("model-derived evidence requires its explicit limitation")
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


PatternMembershipReason = Literal[
    "same_action_shape",
    "same_customer_rule",
    "same_evidence_authority",
    "same_evidence_limitation",
    "same_finding_category",
    "same_finding_kind",
    "same_outcome_stability",
]


class PatternHorizontalFacets(_StrictModel):
    failure_type: FindingCategory
    affected_subject: Literal["action", "outcome", "rule"]
    evidence_level: Literal[
        "observed_action",
        "observed_outcome",
        "model_derived_action",
        "model_derived_outcome",
        "evaluated_rule",
    ]
    mechanism_pseudonym: str = Field(pattern=_PATTERN_MECHANISM_PSEUDONYM_PATTERN)


class PatternVerticalFacets(_StrictModel):
    domain: str | None = Field(default=None, min_length=1, max_length=200)
    workflow: str | None = Field(default=None, min_length=1, max_length=200)
    role: str | None = Field(default=None, min_length=1, max_length=200)
    use_case: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_facets(self) -> Self:
        values = (self.domain, self.workflow, self.role, self.use_case)
        if not any(value is not None for value in values):
            raise ValueError("vertical facets require at least one customer value")
        if any(value is not None and value != value.strip() for value in values):
            raise ValueError("vertical facet values cannot have surrounding whitespace")
        if any(
            value is not None
            and (
                value != unicodedata.normalize("NFC", value)
                or not all(character.isprintable() for character in value)
            )
            for value in values
        ):
            raise ValueError("vertical facet values must be printable NFC text")
        return self


class PatternMember(_StrictModel):
    finding_id: str = Field(pattern=_FINDING_ID_PATTERN)
    evidence_record_ref: str = Field(pattern=_PATTERN_EVIDENCE_REFERENCE_PATTERN)
    membership_reasons: tuple[PatternMembershipReason, ...] = Field(min_length=1)
    review_status: FindingReviewStatus
    review_severity: FindingSeverity

    @model_validator(mode="after")
    def validate_member(self) -> Self:
        if self.membership_reasons != tuple(sorted(set(self.membership_reasons))):
            raise ValueError("pattern membership reasons must be sorted and unique")
        if self.review_status != "confirmed" and self.review_severity != "unrated":
            raise ValueError("only confirmed members can have a rated review severity")
        return self


def _effective_finding_severity(finding: FindingSummary) -> FindingSeverity:
    if finding.review_status == "confirmed":
        if finding.review_severity is None:
            raise AssertionError("validated confirmed finding requires review severity")
        return finding.review_severity
    return finding.declared_severity or "unrated"


class FailurePattern(_StrictModel):
    pattern_fingerprint: str = Field(pattern=_PATTERN_FINGERPRINT_PATTERN)
    pattern_snapshot_id: str = Field(pattern=_PATTERN_SNAPSHOT_ID_PATTERN)
    kind: FindingKind
    category: FindingCategory
    rule_id: str | None = Field(default=None, pattern=_IDENTIFIER_PATTERN)
    rule_version: str | None = Field(default=None, pattern=_VERSION_PATTERN)
    summary: FindingSummaryText
    severity: FindingSeverity
    stability: Literal["stable", "unstable", "inconclusive"]
    evidence_authorities: tuple[PatternEvidenceAuthority, ...] = Field(min_length=1)
    evidence_limitations: tuple[PatternEvidenceLimitation, ...] = ()
    horizontal_facets: PatternHorizontalFacets
    vertical_facets: PatternVerticalFacets | None = None
    finding_count: int = Field(ge=1)
    source_case_count: int = Field(ge=1)
    operators: tuple[PatternOperator, ...] = Field(min_length=1, max_length=100)
    needs_review_count: int = Field(ge=0)
    confirmed_count: int = Field(ge=0)
    expected_count: int = Field(ge=0)
    unsupported_count: int = Field(ge=0)
    inconclusive_count: int = Field(ge=0)
    members: tuple[PatternMember, ...] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_pattern(self, info: ValidationInfo) -> Self:
        if (self.rule_id is None) != (self.rule_version is None):
            raise ValueError("pattern rule ID and version must be present together")
        if self.kind == "behavior_difference" and self.rule_id is not None:
            raise ValueError("behavior patterns cannot reference a customer rule")
        if self.kind == "customer_invariant_violation" and self.rule_id is None:
            raise ValueError("invariant patterns require a customer rule")
        if self.finding_count != len(self.members):
            raise ValueError("pattern finding count must match members")
        if self.source_case_count > self.finding_count:
            raise ValueError("pattern source case count cannot exceed finding count")
        member_ids = tuple(member.finding_id for member in self.members)
        if member_ids != tuple(sorted(set(member_ids))):
            raise ValueError("pattern members must be sorted and unique")
        if (
            self.needs_review_count
            + self.confirmed_count
            + self.expected_count
            + self.unsupported_count
            + self.inconclusive_count
            != self.finding_count
        ):
            raise ValueError("pattern review counts must match finding count")
        if (
            self.needs_review_count
            != sum(member.review_status == "needs_review" for member in self.members)
            or self.confirmed_count
            != sum(member.review_status == "confirmed" for member in self.members)
            or self.expected_count
            != sum(member.review_status == "expected" for member in self.members)
            or self.unsupported_count
            != sum(member.review_status == "unsupported" for member in self.members)
            or self.inconclusive_count
            != sum(member.review_status == "inconclusive" for member in self.members)
        ):
            raise ValueError("pattern review counts must match member snapshots")
        operator_keys = tuple(
            (operator.operator_id, operator.operator_version) for operator in self.operators
        )
        if operator_keys != tuple(sorted(set(operator_keys))):
            raise ValueError("pattern operators must be sorted and unique")
        if self.evidence_authorities != tuple(sorted(set(self.evidence_authorities))):
            raise ValueError("pattern evidence authorities must be sorted and unique")
        if self.evidence_limitations != tuple(sorted(set(self.evidence_limitations))):
            raise ValueError("pattern evidence limitations must be sorted and unique")
        model_derived = "model_derived_unverified" in self.evidence_authorities
        if model_derived != (
            self.evidence_limitations == ("semantic_model_output_not_independently_verified",)
        ):
            raise ValueError("model-derived patterns require their explicit limitation")
        action_evidence_level = "model_derived_action" if model_derived else "observed_action"
        outcome_evidence_level = "model_derived_outcome" if model_derived else "observed_outcome"
        expected_horizontal_scope = {
            "duplicate_effect": ("action", action_evidence_level),
            "unexpected_effect": ("action", action_evidence_level),
            "missing_effect": ("action", action_evidence_level),
            "changed_grounded_effect_argument": ("action", action_evidence_level),
            "unstable_behavior": ("outcome", outcome_evidence_level),
            "customer_invariant_violation": ("rule", "evaluated_rule"),
        }[self.category]
        if (
            self.horizontal_facets.failure_type != self.category
            or (
                self.horizontal_facets.affected_subject,
                self.horizontal_facets.evidence_level,
            )
            != expected_horizontal_scope
        ):
            raise ValueError("horizontal facets must match the pattern finding type")
        expected_reasons: tuple[PatternMembershipReason, ...] = tuple(
            sorted(
                (
                    "same_finding_kind",
                    "same_finding_category",
                    (
                        "same_customer_rule"
                        if self.kind == "customer_invariant_violation"
                        else "same_action_shape"
                    ),
                    "same_outcome_stability",
                    "same_evidence_authority",
                    "same_evidence_limitation",
                )
            )
        )
        if any(member.membership_reasons != expected_reasons for member in self.members):
            raise ValueError("pattern members require the exact deterministic grouping reasons")
        if info.context != {"building_pattern": True}:
            if self.pattern_fingerprint != _pattern_fingerprint(self):
                raise ValueError("pattern fingerprint must match its grouping facets")
            if self.pattern_snapshot_id != _pattern_snapshot_id(self):
                raise ValueError("pattern snapshot ID must match its exact snapshot")
        return self


class ReviewStatusCounts(_StrictModel):
    needs_review: int = Field(ge=0)
    confirmed: int = Field(ge=0)
    expected: int = Field(ge=0)
    unsupported: int = Field(ge=0)
    inconclusive: int = Field(ge=0)


class PatternReviewSummary(_StrictModel):
    pattern_review_id: str = Field(pattern=_PATTERN_REVIEW_ID_PATTERN)
    pattern_fingerprint: str = Field(pattern=_PATTERN_FINGERPRINT_PATTERN)
    pattern_snapshot_id: str = Field(pattern=_PATTERN_SNAPSHOT_ID_PATTERN)
    status: Literal["confirmed", "expected", "unsupported", "inconclusive"]
    severity: FindingSeverity = "unrated"
    reviewed_finding_ids: tuple[str, ...] = Field(min_length=1, max_length=10_000)
    exception_finding_ids: tuple[str, ...] = Field(default=(), max_length=10_000)
    reviewed_at: datetime

    @model_validator(mode="after")
    def validate_review(self) -> Self:
        for values, label in (
            (self.reviewed_finding_ids, "reviewed finding IDs"),
            (self.exception_finding_ids, "exception finding IDs"),
        ):
            _validate_sorted_unique(values, label)
            if any(re.fullmatch(_FINDING_ID_PATTERN, value) is None for value in values):
                raise ValueError(f"{label} must contain finding IDs")
        if set(self.reviewed_finding_ids) & set(self.exception_finding_ids):
            raise ValueError("reviewed and exception finding IDs must be disjoint")
        if self.status != "confirmed" and self.severity != "unrated":
            raise ValueError("only confirmed pattern reviews can have a rated severity")
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("pattern review timestamp must include a UTC offset")
        return self


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
    schema_version: Literal["1.7.0"] = "1.7.0"
    evidence_type: ReportEvidenceType
    evidence_schema_versions: tuple[str, ...] = Field(min_length=1)
    evidence_scope: ReportEvidenceScope
    evaluation_mode: ReportEvaluationMode | None = None
    capability_limitations: tuple[ReportCapabilityLimitation, ...] = ()
    review_status: ReportReviewStatus
    exit_code: Literal[0, 1, 2]
    summary: ReportSummary
    stable_pattern_count: int = Field(default=0, ge=0)
    patterns: tuple[FailurePattern, ...] = ()
    pattern_reviews: tuple[PatternReviewSummary, ...] = ()
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
        pattern_ids = tuple(pattern.pattern_snapshot_id for pattern in self.patterns)
        expected_pattern_order = tuple(
            pattern.pattern_snapshot_id
            for pattern in sorted(
                self.patterns,
                key=lambda pattern: (
                    -_SEVERITY_RANK[pattern.severity],
                    pattern.pattern_snapshot_id,
                ),
            )
        )
        if len(pattern_ids) != len(set(pattern_ids)) or pattern_ids != expected_pattern_order:
            raise ValueError("patterns must be sorted and unique")
        if self.stable_pattern_count != len(
            {pattern.pattern_fingerprint for pattern in self.patterns}
        ):
            raise ValueError("stable pattern count must match unique fingerprints")
        pattern_review_ids = tuple(review.pattern_review_id for review in self.pattern_reviews)
        expected_review_order = tuple(
            review.pattern_review_id
            for review in sorted(
                self.pattern_reviews,
                key=lambda review: (review.reviewed_at, review.pattern_review_id),
            )
        )
        if len(pattern_review_ids) != len(set(pattern_review_ids)) or (
            pattern_review_ids != expected_review_order
        ):
            raise ValueError("pattern reviews must be sorted and unique")
        known_findings = {
            finding.finding_id: finding
            for finding in self.findings
            if finding.finding_id is not None
        }
        patterned_findings: set[str] = set()
        for pattern in self.patterns:
            pattern_findings: list[FindingSummary] = []
            for member in pattern.members:
                finding_id = member.finding_id
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
                    or finding.stability != pattern.stability
                    or finding.evidence_authorities != pattern.evidence_authorities
                    or finding.evidence_limitations != pattern.evidence_limitations
                    or finding.review_status != member.review_status
                    or finding.review_severity != member.review_severity
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
