from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

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
_RUN_RECEIPT_ID_PATTERN = r"^ulrr_v1_[A-Za-z0-9][A-Za-z0-9._-]{0,90}$"
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
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class VersionedReference(_StrictModel):
    id: str = Field(pattern=_IDENTIFIER_PATTERN)
    version: str = Field(pattern=_VERSION_PATTERN)


class EvidencePointer(_StrictModel):
    pointer_id: str = Field(pattern=_IDENTIFIER_PATTERN)
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
    source_evidence_pointer_ids: tuple[str, ...] = Field(min_length=1, max_length=100)
    probe_evidence_pointer_ids: tuple[str, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_pointer_ids(self) -> Self:
        for pointer_ids in (
            self.source_evidence_pointer_ids,
            self.probe_evidence_pointer_ids,
        ):
            if pointer_ids != tuple(sorted(set(pointer_ids))):
                raise ValueError("probe change evidence pointer IDs must be sorted and unique")
        return self


class ObservedDelta(_StrictModel):
    kind: Literal["response", "action", "rule", "state"]
    change: Literal["added", "removed", "changed", "violated", "unstable"]
    subject: str | None = Field(default=None, pattern=_IDENTIFIER_PATTERN)
    evidence_pointer_ids: tuple[str, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_pointer_ids(self) -> Self:
        if self.evidence_pointer_ids != tuple(sorted(set(self.evidence_pointer_ids))):
            raise ValueError("delta evidence pointer IDs must be sorted and unique")
        return self


class RepetitionEvidence(_StrictModel):
    requested: int = Field(ge=1)
    conclusive: int = Field(ge=0)
    violated: int = Field(ge=0)
    inconclusive: int = Field(ge=0)
    stability: Literal["stable", "unstable", "inconclusive"]
    reproducibility: Literal[
        "reproduced",
        "intermittent",
        "not_reproduced",
        "not_established",
    ]

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.conclusive + self.inconclusive != self.requested:
            raise ValueError("repetition counts must match requested repetitions")
        if self.violated > self.conclusive:
            raise ValueError("violated repetitions cannot exceed conclusive repetitions")
        expected_reproducibility = (
            "not_established"
            if self.conclusive == 0
            else "not_reproduced"
            if self.violated == 0
            else "reproduced"
            if self.violated == self.conclusive
            else "intermittent"
        )
        if self.reproducibility != expected_reproducibility:
            raise ValueError("reproducibility must match observed repetition counts")
        if self.conclusive == 0 and self.stability != "inconclusive":
            raise ValueError("evidence without a conclusive repetition is inconclusive")
        if self.reproducibility == "intermittent" and self.stability != "unstable":
            raise ValueError("intermittent reproduction is unstable")
        return self


class FindingOccurrence(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    occurrence_id: str = Field(pattern=_FINDING_ID_PATTERN)
    kind: FindingKind
    category: FindingCategory
    campaign_id: str | None = Field(default=None, min_length=1, max_length=500)
    source_interaction_id: str | None = Field(default=None, min_length=1, max_length=500)
    fixture: VersionedReference | None = None
    case_id: str = Field(min_length=1, max_length=500)
    operator: VersionedReference
    bundle: VersionedReference | None = None
    probe_change: ProbeChange
    observed_deltas: tuple[ObservedDelta, ...] = Field(min_length=1, max_length=100)
    violated_rule: VersionedReference | None = None
    evidence_pointers: tuple[EvidencePointer, ...] = Field(min_length=1, max_length=500)
    repetitions: RepetitionEvidence
    required_capabilities: tuple[_FindingReferenceCode, ...] = Field(default=(), max_length=100)
    limitations: tuple[_FindingReferenceCode, ...] = Field(default=(), max_length=100)
    inconclusive_reasons: tuple[_FindingReferenceCode, ...] = Field(default=(), max_length=100)
    review_history_ids: tuple[_FindingReferenceCode, ...] = Field(default=(), max_length=1_000)
    run_receipt_id: str | None = Field(default=None, pattern=_RUN_RECEIPT_ID_PATTERN)
    next_action: FindingNextAction

    @model_validator(mode="after")
    def validate_occurrence(self) -> Self:
        if self.kind == "behavior_difference":
            if self.category == "customer_invariant_violation" or self.violated_rule is not None:
                raise ValueError("behavior occurrences cannot claim a customer rule violation")
        elif self.category != "customer_invariant_violation" or self.violated_rule is None:
            raise ValueError("invariant occurrences require a violated customer rule")
        pointer_ids = tuple(pointer.pointer_id for pointer in self.evidence_pointers)
        if pointer_ids != tuple(sorted(set(pointer_ids))):
            raise ValueError("evidence pointers must be sorted and unique")
        referenced_pointer_ids = {
            *self.probe_change.source_evidence_pointer_ids,
            *self.probe_change.probe_evidence_pointer_ids,
            *(
                pointer_id
                for delta in self.observed_deltas
                for pointer_id in delta.evidence_pointer_ids
            ),
        }
        if not referenced_pointer_ids.issubset(pointer_ids):
            raise ValueError("finding occurrence references an unknown evidence pointer")
        if referenced_pointer_ids != set(pointer_ids):
            raise ValueError("finding occurrence contains unused evidence pointers")
        pointers_by_id = {pointer.pointer_id: pointer for pointer in self.evidence_pointers}
        source_pointer_ids = set(self.probe_change.source_evidence_pointer_ids)
        probe_pointer_ids = set(self.probe_change.probe_evidence_pointer_ids)
        if source_pointer_ids & probe_pointer_ids:
            raise ValueError("source and probe evidence pointers must be disjoint")
        if any(pointers_by_id[pointer_id].arm != "source" for pointer_id in source_pointer_ids):
            raise ValueError("source change evidence must reference the source arm")
        if any(pointers_by_id[pointer_id].arm != "probe" for pointer_id in probe_pointer_ids):
            raise ValueError("probe change evidence must reference the probe arm")
        change_pointer_kinds = {
            "input": {"input"},
            "context": {"input"},
            "turn_sequence": {"input"},
            "state_setup": {"state"},
            "event_behavior": {"action", "tool_call", "tool_result", "lifecycle"},
        }[self.probe_change.kind]
        if any(
            pointers_by_id[pointer_id].kind not in change_pointer_kinds
            for pointer_id in source_pointer_ids | probe_pointer_ids
        ):
            raise ValueError("probe change references incompatible evidence")
        delta_pointer_kinds = {
            "response": {"response"},
            "action": {"action", "tool_call", "tool_result"},
            "rule": {"rule"},
            "state": {"state"},
        }
        if any(
            pointers_by_id[pointer_id].kind not in delta_pointer_kinds[delta.kind]
            for delta in self.observed_deltas
            for pointer_id in delta.evidence_pointer_ids
        ):
            raise ValueError("observed delta references incompatible evidence")
        for delta in self.observed_deltas:
            delta_arms = {
                pointers_by_id[pointer_id].arm for pointer_id in delta.evidence_pointer_ids
            }
            if delta.change == "changed" and not {"source", "probe"}.issubset(delta_arms):
                raise ValueError("changed deltas require source and probe evidence")
            if delta.change in {"added", "violated", "unstable"} and "probe" not in delta_arms:
                raise ValueError(f"{delta.change} deltas require probe evidence")
            if delta.change == "removed" and "source" not in delta_arms:
                raise ValueError("removed deltas require source evidence")
        required_category_delta = {
            "duplicate_effect": ("action", "added"),
            "unexpected_effect": ("action", "added"),
            "missing_effect": ("action", "removed"),
            "changed_grounded_effect_argument": ("action", "changed"),
            "unstable_behavior": (None, "unstable"),
            "customer_invariant_violation": ("rule", "violated"),
        }[self.category]
        if not any(
            (required_category_delta[0] is None or delta.kind == required_category_delta[0])
            and delta.change == required_category_delta[1]
            for delta in self.observed_deltas
        ):
            raise ValueError("finding category must match its observed delta")
        if self.category == "unstable_behavior" and self.repetitions.stability != "unstable":
            raise ValueError("unstable findings require unstable repetition evidence")
        rule_violation_deltas = tuple(
            delta
            for delta in self.observed_deltas
            if delta.kind == "rule" and delta.change == "violated"
        )
        if self.violated_rule is None and rule_violation_deltas:
            raise ValueError("violated rules require an exact rule violation delta")
        if self.violated_rule is not None and (
            len(rule_violation_deltas) != 1
            or rule_violation_deltas[0].subject != self.violated_rule.id
        ):
            raise ValueError("violated rule identity must match one exact rule violation delta")
        for values, label in (
            (self.required_capabilities, "required capabilities"),
            (self.limitations, "limitations"),
            (self.inconclusive_reasons, "inconclusive reasons"),
            (self.review_history_ids, "review history IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{label} must be sorted and unique")
        if self.repetitions.inconclusive == 0 and self.inconclusive_reasons:
            raise ValueError("conclusive occurrences cannot declare inconclusive reasons")
        if self.repetitions.inconclusive > 0 and not self.inconclusive_reasons:
            raise ValueError("inconclusive repetitions require at least one reason")
        return self


class ToolExchangeReceipt(_StrictModel):
    sequence: int = Field(ge=1)
    call: JsonValue
    result: JsonValue | None = None
    authority: Literal[
        "invoker_self_reported",
        "source_self_reported",
        "environment_self_reported",
        "independent_observer",
    ]
    source_id: str = Field(min_length=1, max_length=500)


class StateReceipt(_StrictModel):
    value: JsonValue
    authority: Literal["environment_self_reported", "independent_observer"]
    observer_id: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_observer(self) -> Self:
        if (self.authority == "independent_observer") != (self.observer_id is not None):
            raise ValueError("only independent state evidence names an observer")
        return self


class LifecycleReceipt(_StrictModel):
    phase: Literal["initial_reset", "setup", "execution", "cleanup_reset"]
    status: Literal["succeeded", "failed", "not_attempted", "unknown"]
    evidence_pointer_ids: tuple[str, ...] = Field(default=(), max_length=100)
    limitation: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> Self:
        if self.evidence_pointer_ids != tuple(sorted(set(self.evidence_pointer_ids))):
            raise ValueError("lifecycle evidence pointer IDs must be sorted and unique")
        if (self.status != "succeeded") != (self.limitation is not None):
            raise ValueError("unverified lifecycle phases require one limitation")
        return self


class ProvenanceReceipt(_StrictModel):
    role: Literal["target", "model", "evaluator", "environment", "observer"]
    id: str = Field(min_length=1, max_length=500)
    version: str | None = Field(default=None, min_length=1, max_length=100)
    config_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)


class UsageReceipt(_StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    duration_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)

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


class RunReceipt(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    receipt_id: str = Field(pattern=_RUN_RECEIPT_ID_PATTERN)
    disclosure: Literal["private"] = "private"
    evidence_scope: ReportEvidenceScope
    source_input: JsonValue
    probe_input: JsonValue
    source_response: JsonValue
    probe_response: JsonValue
    tool_exchanges: tuple[ToolExchangeReceipt, ...] = Field(default=(), max_length=1_000)
    state_before: StateReceipt | None = None
    state_after: StateReceipt | None = None
    lifecycle: tuple[LifecycleReceipt, ...] = Field(default=(), max_length=10)
    provenance: tuple[ProvenanceReceipt, ...] = Field(default=(), max_length=100)
    trace_references: tuple[str, ...] = Field(default=(), max_length=1_000)
    usage: UsageReceipt | None = None
    redaction: RedactionReceipt
    evidence_pointers: tuple[EvidencePointer, ...] = Field(min_length=1, max_length=500)
    limitations: tuple[str, ...] = Field(default=(), max_length=100)
    recorded_at: datetime

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        sequences = tuple(exchange.sequence for exchange in self.tool_exchanges)
        if sequences != tuple(range(1, len(sequences) + 1)):
            raise ValueError("tool exchanges must be contiguous and ordered")
        pointer_ids = tuple(pointer.pointer_id for pointer in self.evidence_pointers)
        if pointer_ids != tuple(sorted(set(pointer_ids))):
            raise ValueError("receipt evidence pointers must be sorted and unique")
        known_pointer_ids = set(pointer_ids)
        if any(
            not set(phase.evidence_pointer_ids).issubset(known_pointer_ids)
            for phase in self.lifecycle
        ):
            raise ValueError("lifecycle references an unknown evidence pointer")
        pointers_by_id = {pointer.pointer_id: pointer for pointer in self.evidence_pointers}
        if any(
            pointers_by_id[pointer_id].kind != "lifecycle"
            for phase in self.lifecycle
            for pointer_id in phase.evidence_pointer_ids
        ):
            raise ValueError("lifecycle phases require lifecycle evidence pointers")
        if any(
            phase.status == "succeeded" and not phase.evidence_pointer_ids
            for phase in self.lifecycle
        ):
            raise ValueError("successful lifecycle phases require exact evidence")
        lifecycle_phases = tuple(phase.phase for phase in self.lifecycle)
        if len(lifecycle_phases) != len(set(lifecycle_phases)):
            raise ValueError("lifecycle phases must be unique")
        lifecycle_order = {
            "initial_reset": 0,
            "setup": 1,
            "execution": 2,
            "cleanup_reset": 3,
        }
        if lifecycle_phases != tuple(sorted(lifecycle_phases, key=lifecycle_order.__getitem__)):
            raise ValueError("lifecycle phases must preserve execution order")
        if self.evidence_scope == "response_only":
            if self.state_before is not None or self.state_after is not None:
                raise ValueError("response-only receipts cannot contain state evidence")
            if any(pointer.kind == "state" for pointer in self.evidence_pointers):
                raise ValueError("response-only receipts cannot reference state evidence")
        elif self.state_before is None or self.state_after is None:
            raise ValueError("response-and-state receipts require before and after state")
        core_evidence_roles = {
            (pointer.kind, pointer.arm)
            for pointer in self.evidence_pointers
            if pointer.kind in {"input", "response"}
        }
        if not {
            ("input", "source"),
            ("input", "probe"),
            ("response", "source"),
            ("response", "probe"),
        }.issubset(core_evidence_roles):
            raise ValueError("run receipts require source and probe input and response evidence")
        for values, label in (
            (self.trace_references, "trace references"),
            (self.limitations, "receipt limitations"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{label} must be sorted and unique")
            if any(not value or len(value) > 500 for value in values):
                raise ValueError(f"{label} must contain bounded non-empty values")
        provenance_keys = tuple((item.role, item.id, item.version) for item in self.provenance)
        expected_provenance_keys = tuple(
            sorted(
                set(provenance_keys),
                key=lambda item: (item[0], item[1], item[2] or ""),
            )
        )
        if provenance_keys != expected_provenance_keys:
            raise ValueError("provenance must be sorted and unique")
        observer_ids = {item.id for item in self.provenance if item.role == "observer"}
        independent_source_ids = {
            pointer.source_id
            for pointer in self.evidence_pointers
            if pointer.authority == "independent_observer"
        } | {
            exchange.source_id
            for exchange in self.tool_exchanges
            if exchange.authority == "independent_observer"
        }
        if self.state_before is not None and self.state_before.authority == "independent_observer":
            assert self.state_before.observer_id is not None
            independent_source_ids.add(self.state_before.observer_id)
        if self.state_after is not None and self.state_after.authority == "independent_observer":
            assert self.state_after.observer_id is not None
            independent_source_ids.add(self.state_after.observer_id)
        if not independent_source_ids.issubset(observer_ids):
            raise ValueError("independent evidence requires matching observer provenance")
        if self.recorded_at.tzinfo is None or self.recorded_at.utcoffset() is None:
            raise ValueError("receipt timestamp must include a UTC offset")
        if len(self.model_dump_json().encode("utf-8")) > _MAXIMUM_RUN_RECEIPT_BYTES:
            raise ValueError("run receipt exceeds the 1 MB JSON limit")
        return self


def serialize_run_receipt(receipt: RunReceipt) -> str:
    serialized = receipt.model_dump_json()
    return RunReceipt.model_validate_json(serialized).model_dump_json()


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
