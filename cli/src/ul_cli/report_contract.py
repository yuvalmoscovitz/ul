from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

ReportEvidenceType = Literal[
    "dataset_evaluation",
    "correction_after_first_response",
    "retry_after_successful_commit",
    "timeout_after_commit",
]
ReportReviewStatus = Literal["resolved", "action_required", "inconclusive"]
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
    schema_version: Literal["1.3.0"] = "1.3.0"
    evidence_type: ReportEvidenceType
    evidence_schema_versions: tuple[str, ...] = Field(min_length=1)
    evidence_scope: ReportEvidenceScope
    capability_limitations: tuple[ReportCapabilityLimitation, ...] = ()
    review_status: ReportReviewStatus
    exit_code: Literal[0, 1, 2]
    summary: ReportSummary
    findings: tuple[FindingSummary, ...] = ()

    @model_validator(mode="after")
    def validate_report(self) -> Self:
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
