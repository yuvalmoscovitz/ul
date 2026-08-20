from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

ReportEvidenceType = Literal[
    "dataset_evaluation",
    "correction_after_first_response",
    "retry_after_successful_commit",
    "timeout_after_commit",
]
ReportStatus = Literal["passed", "failed", "inconclusive"]
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
FindingSummaryText = Literal[
    "The variation repeated an observed action effect.",
    "The variation produced an unexpected action effect.",
    "The variation omitted an expected action effect.",
    "The variation changed a grounded action argument.",
    "The variation produced unstable behavior across repetitions.",
    "A customer invariant was violated.",
]

_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$"
_VERSION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,49}$"
_FINDING_ID_PATTERN = r"^ulf_v1_[0-9a-f]{64}$"
_BEHAVIOR_SUMMARIES: dict[str, str] = {
    "duplicate_effect": "The variation repeated an observed action effect.",
    "unexpected_effect": "The variation produced an unexpected action effect.",
    "missing_effect": "The variation omitted an expected action effect.",
    "changed_grounded_effect_argument": "The variation changed a grounded action argument.",
    "unstable_behavior": "The variation produced unstable behavior across repetitions.",
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
            or self.summary != "A customer invariant was violated."
        ):
            raise ValueError("invariant finding fields are inconsistent")
        return self


class UnifiedReport(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    evidence_type: ReportEvidenceType
    evidence_schema_versions: tuple[str, ...] = Field(min_length=1)
    status: ReportStatus
    exit_code: Literal[0, 1, 2]
    finding_count: int = Field(ge=0)
    findings: tuple[FindingSummary, ...] = ()

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.evidence_schema_versions != tuple(sorted(set(self.evidence_schema_versions))):
            raise ValueError("evidence schema versions must be sorted and unique")
        expected_exit_code = {"passed": 0, "failed": 1, "inconclusive": 2}[self.status]
        if self.exit_code != expected_exit_code:
            raise ValueError("exit code must match report status")
        if self.finding_count != len(self.findings):
            raise ValueError("finding count must match report findings")
        if self.status == "passed" and self.findings:
            raise ValueError("passed reports cannot contain findings")
        if self.status == "failed" and not self.findings:
            raise ValueError("failed reports require at least one finding")
        return self
