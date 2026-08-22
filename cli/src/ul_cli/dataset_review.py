from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import unicodedata
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Protocol, Self, cast
from uuid import uuid4

import typer
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator
from rich.console import Console
from ul import DatasetEvaluationResult, InteractionRecord
from ul.dataset_invariants import (
    DatasetInvariantArrayUniqueRuleEvaluation,
    DatasetInvariantArrayUniqueTrialEvaluation,
    DatasetInvariantEvaluation,
    DatasetInvariantRule,
    DatasetInvariantRuleEvaluation,
    DatasetInvariantRuleResult,
    DatasetInvariantSuite,
    DatasetInvariantTransitionRuleEvaluation,
    DatasetInvariantTransitionTrialEvaluation,
    DatasetInvariantTrialEvaluation,
    DatasetInvariantValueEqualsRuleEvaluation,
    DatasetInvariantValueEqualsTrialEvaluation,
    DatasetInvariantValueInSetRuleEvaluation,
    DatasetInvariantValueInSetTrialEvaluation,
    ExactlyOneNewEffectInvariant,
    JsonArrayItemsUniqueByInvariant,
    JsonValueEqualsLiteralInvariant,
    JsonValueInAllowedSetInvariant,
    JsonValuesEqualInvariant,
    NoNewEffectInvariant,
    UnchangedBetweenCheckpointsInvariant,
    evaluate_dataset_invariants,
)
from ul.dataset_regression import dataset_regression_target_config_sha256
from ul.http_environment import JsonHttpIsolatedResponseConfig, JsonHttpTargetConfig
from ul_core.augmentation_catalog import builtin_augmentation_catalog

from ul_cli.report_contract import (
    FailurePattern,
    FindingCategory,
    FindingReviewStatus,
    FindingSeverity,
    FindingSummary,
    FindingSummaryText,
    PatternOperator,
    ReportInputError,
    ReportReviewStatus,
    UnifiedReport,
    build_report_summary,
)

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

_MAXIMUM_EVIDENCE_BYTES = 128_000_000
_MAXIMUM_EVIDENCE_RECORDS = 100
_MAXIMUM_REVIEWS_BYTES = 10_000_000
_MAXIMUM_REVIEW_RECORDS = 10_000
_MAXIMUM_SENSITIVE_DISCLOSURE_BYTES = 32_768
_MAXIMUM_SENSITIVE_DISCLOSURE_LINES = 50
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_FINDING_ID_PATTERN = r"^ulf_v1_[0-9a-f]{64}$"
_REVIEW_ID_PATTERN = r"^ulr_[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
_DATASET_EVALUATION_PIPELINE_VERSION = "1.4.0"
_MAXIMUM_PATTERN_EFFECTS = 100
_MAXIMUM_PATTERN_FIELDS = 100
_MAXIMUM_PATTERN_LABEL_CHARACTERS = 500
_PATTERN_SEVERITY_RANK: dict[FindingSeverity, int] = {
    "unrated": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

ReviewStatus = Literal["confirmed", "expected", "unsupported", "inconclusive"]
ReviewSeverity = Literal["unrated", "low", "medium", "high", "critical"]

console = Console()


class _FindingIdHash(Protocol):
    def update(self, data: bytes, /) -> None: ...

    def copy(self) -> Self: ...

    def hexdigest(self) -> str: ...


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _EvidenceReference(_StrictModel):
    source: Literal["input", "output"]
    json_pointer: str
    text_quote: str | None


class _Effect(_StrictModel):
    id: str
    evidence: list[_EvidenceReference]
    confidence: float = Field(ge=0, le=1)
    status: str
    request_unit_ids: list[str]
    position: int = Field(ge=0)
    kind: str
    predicate: str
    fields: dict[str, JsonValue]
    propositions: list[str]


class _Finding(_StrictModel):
    finding_id: str = Field(pattern=_FINDING_ID_PATTERN)
    category: str
    grounded_field_names: list[str]
    severity: Literal["unrated"]
    review_status: Literal["needs_review"]
    summary: str
    reference_effects: list[_Effect]
    observed_effects: list[_Effect]


class _OutcomeGroup(_StrictModel):
    repetitions: list[int]
    count: int = Field(ge=1)
    representative_effects: list[_Effect]


class _LifecycleFailure(_StrictModel):
    protocol_version: Literal[5]
    failed_phase: str
    completed_phases: list[str]
    cleanup_reset_failed: bool
    environment_state_may_remain: bool


class _Trial(_StrictModel):
    repetition: int = Field(ge=1)
    status: Literal["observed", "inconclusive"]
    inconclusive_reasons: list[str]
    lifecycle_failure: _LifecycleFailure | None = None


class _Observations(_StrictModel):
    requested_repetitions: int = Field(ge=1)
    stability: Literal["stable", "unstable", "inconclusive"]
    observed_repetitions: int = Field(ge=0)
    inconclusive_repetitions: int = Field(ge=0)
    outcome_group_count: int = Field(ge=0)
    outcome_groups: list[_OutcomeGroup]
    trials: list[_Trial]


class _ExecutionPlan(_StrictModel):
    repetitions: int = Field(ge=1)
    max_target_calls: int = Field(ge=1)
    dataset_planned_target_calls: int = Field(ge=1)


class DatasetEvidenceOperator(_StrictModel):
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)


class DatasetEvidenceSemanticSettings(_StrictModel):
    provider: str = Field(min_length=1, max_length=100)
    endpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    model: str
    render_model: str
    equivalence_model: str
    max_input_chars: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    max_render_tokens: int = Field(ge=1)
    max_response_bytes: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)


class DatasetEvidenceRedactionCoverage(_StrictModel):
    location: Literal["input", "output"]
    matched_values: int = Field(ge=0)
    matched_paths: tuple[str, ...] = ()
    matches_by_rule: dict[str, int] = Field(default_factory=dict)


class DatasetEvidenceTarget(_StrictModel):
    kind: Literal["environment_http"]
    config: JsonHttpTargetConfig
    sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        expected_sha256 = dataset_regression_target_config_sha256(self.config)
        if self.sha256 != expected_sha256:
            raise ValueError("run context target digest must match its snapshot")
        return self


class DatasetEvidenceFixture(_StrictModel):
    status: Literal["configured", "missing", "not_required"]
    id: str | None = Field(default=None, min_length=1, max_length=500)
    version: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.status == "configured":
            if self.id is None or self.version is None:
                raise ValueError("configured fixture identity requires an id and version")
        elif self.id is not None or self.version is not None:
            raise ValueError("unconfigured fixture identity cannot include an id or version")
        return self


class DatasetEvidenceRunContext(_StrictModel):
    schema_version: Literal["1.1.0", "1.2.0", "1.3.0"] = "1.3.0"
    pipeline_version: Literal["1.2.0", "1.3.0", "1.4.0"] = _DATASET_EVALUATION_PIPELINE_VERSION
    selected_dataset_sha256: str = Field(pattern=_SHA256_PATTERN)
    operators: tuple[DatasetEvidenceOperator, ...] = Field(min_length=1)
    evaluation_mode: Literal["variance"] | None = None
    repetitions: int = Field(ge=1)
    invariant_suite_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    target: DatasetEvidenceTarget
    fixture: DatasetEvidenceFixture | None = None
    semantic_settings: DatasetEvidenceSemanticSettings
    redaction_policy_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    redaction_coverage: tuple[DatasetEvidenceRedactionCoverage, ...] = ()
    context_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_digests(self) -> Self:
        if (self.schema_version, self.pipeline_version) not in {
            ("1.1.0", "1.2.0"),
            ("1.2.0", "1.3.0"),
            ("1.3.0", "1.4.0"),
        }:
            raise ValueError("run context schema and pipeline versions must match")
        if self.schema_version in {"1.2.0", "1.3.0"} and self.evaluation_mode is None:
            raise ValueError(
                f"run context schema {self.schema_version} requires an evaluation mode"
            )
        if self.schema_version == "1.1.0" and "evaluation_mode" in self.model_fields_set:
            raise ValueError("run context schema 1.1.0 does not include evaluation mode")
        if self.schema_version == "1.3.0" and self.fixture is None:
            raise ValueError("run context schema 1.3.0 requires fixture identity status")
        if self.schema_version != "1.3.0" and "fixture" in self.model_fields_set:
            raise ValueError(
                f"run context schema {self.schema_version} does not include fixture identity"
            )
        context_content = self.model_dump(mode="json", exclude={"context_sha256"})
        if self.evaluation_mode is None:
            context_content.pop("evaluation_mode")
        if self.fixture is None:
            context_content.pop("fixture")
        if self.redaction_policy_sha256 is None:
            context_content.pop("redaction_policy_sha256")
        if not self.redaction_coverage:
            context_content.pop("redaction_coverage")
        expected_context_sha256 = _canonical_json_sha256(context_content)
        if self.context_sha256 != expected_context_sha256:
            raise ValueError("run context digest must match its canonical content")
        return self


class _Baseline(_StrictModel):
    status: str
    observations: _Observations
    inconclusive_reasons: list[str]


class _Case(_StrictModel):
    operator_id: str
    operator_version: str
    augmented_input: str
    status: str
    variation_accepted: bool
    variation_rejection_reasons: list[str]
    observations: _Observations | None
    findings: list[_Finding]
    inconclusive_reasons: list[str]


class _EvidenceRecord(_StrictModel):
    schema_version: Literal["1.3.0", "1.4.0", "1.5.0", "1.6.0", "1.7.0", "1.8.0"]
    evaluation_mode: Literal["variance"] | None = None
    interaction_id: str
    original_input: str
    execution_plan: _ExecutionPlan
    limitations: str
    current_baseline: _Baseline
    cases: list[_Case]
    invariant_evaluation: DatasetInvariantEvaluation | None = None
    run_context: DatasetEvidenceRunContext | None = None
    technical_details: JsonValue

    @model_validator(mode="after")
    def validate_invariant_evaluation(self) -> Self:
        if self.schema_version == "1.8.0" and self.evaluation_mode is None:
            raise ValueError("evidence schema 1.8.0 requires an evaluation mode")
        if self.schema_version == "1.8.0" and (
            not isinstance(self.technical_details, dict)
            or self.technical_details.get("evaluation_mode") != self.evaluation_mode
        ):
            raise ValueError("evidence evaluation mode must match technical details")
        if self.schema_version != "1.8.0" and "evaluation_mode" in self.model_fields_set:
            raise ValueError("legacy evidence does not include evaluation mode")
        if self.schema_version == "1.3.0" and "invariant_evaluation" in self.model_fields_set:
            raise ValueError("schema 1.3.0 does not include invariant evaluation")
        if self.schema_version in {"1.5.0", "1.6.0", "1.7.0"} and self.run_context is None:
            raise ValueError(f"schema {self.schema_version} requires run context")
        if (
            self.schema_version not in {"1.5.0", "1.6.0", "1.7.0", "1.8.0"}
            and "run_context" in self.model_fields_set
        ):
            raise ValueError("legacy evidence does not include run context")
        uses_extended_invariants = self.invariant_evaluation is not None and any(
            rule.rule_type != "json_values_equal"
            for arm in (
                self.invariant_evaluation.baseline,
                *self.invariant_evaluation.variations,
            )
            for rule in arm.rules
        )
        if uses_extended_invariants and self.schema_version not in {"1.6.0", "1.7.0", "1.8.0"}:
            raise ValueError("extended invariant results require evidence schema 1.6.0")
        if (
            self.run_context is not None
            and self.evaluation_mode is not None
            and self.run_context.evaluation_mode != self.evaluation_mode
        ):
            raise ValueError("evidence evaluation mode must match its run context")
        if (
            self.invariant_evaluation is not None
            and self.invariant_evaluation.interaction_id != self.interaction_id
        ):
            raise ValueError("invariant evaluation must match the evidence interaction")
        return self


@dataclass(frozen=True)
class DatasetResumeEvidence:
    processed_ids: frozenset[str]
    has_review_findings: bool
    invariant_evaluations: tuple[DatasetInvariantEvaluation, ...]
    technical_results: tuple[DatasetEvaluationResult, ...]
    raw_evidence_sha256: str


def create_dataset_evidence_run_context(
    *,
    selected_records: tuple[InteractionRecord, ...],
    operators: tuple[tuple[str, str], ...],
    evaluation_mode: Literal["variance"] = "variance",
    repetitions: int,
    invariant_suite_sha256: str | None,
    target_config: JsonHttpTargetConfig | None = None,
    semantic_settings: DatasetEvidenceSemanticSettings,
    redaction_policy_sha256: str | None = None,
    redaction_coverage: tuple[DatasetEvidenceRedactionCoverage, ...] = (),
) -> DatasetEvidenceRunContext:
    selected_dataset_sha256 = _canonical_json_sha256(
        [record.model_dump(mode="json") for record in selected_records]
    )
    operator_snapshots = tuple(
        DatasetEvidenceOperator(id=operator_id, version=version)
        for operator_id, version in operators
    )
    if target_config is None:
        raise ValueError("run context requires a environment API connection")
    target = DatasetEvidenceTarget(
        kind="environment_http",
        config=target_config,
        sha256=dataset_regression_target_config_sha256(target_config),
    )
    fixture = _dataset_evidence_fixture(target_config)
    content = {
        "schema_version": "1.3.0",
        "pipeline_version": _DATASET_EVALUATION_PIPELINE_VERSION,
        "selected_dataset_sha256": selected_dataset_sha256,
        "operators": [operator.model_dump(mode="json") for operator in operator_snapshots],
        "evaluation_mode": evaluation_mode,
        "repetitions": repetitions,
        "invariant_suite_sha256": invariant_suite_sha256,
        "target": target.model_dump(mode="json"),
        "fixture": fixture.model_dump(mode="json"),
        "semantic_settings": semantic_settings.model_dump(mode="json"),
    }
    if redaction_policy_sha256 is not None:
        content["redaction_policy_sha256"] = redaction_policy_sha256
    if redaction_coverage:
        content["redaction_coverage"] = [
            item.model_dump(mode="json") for item in redaction_coverage
        ]
    return DatasetEvidenceRunContext(
        selected_dataset_sha256=selected_dataset_sha256,
        operators=operator_snapshots,
        evaluation_mode=evaluation_mode,
        repetitions=repetitions,
        invariant_suite_sha256=invariant_suite_sha256,
        target=target,
        fixture=fixture,
        semantic_settings=semantic_settings,
        redaction_policy_sha256=redaction_policy_sha256,
        redaction_coverage=redaction_coverage,
        context_sha256=_canonical_json_sha256(content),
    )


def _dataset_evidence_fixture(target_config: JsonHttpTargetConfig) -> DatasetEvidenceFixture:
    if isinstance(target_config, JsonHttpIsolatedResponseConfig):
        return DatasetEvidenceFixture(status="not_required")
    if target_config.fixture_id is None:
        return DatasetEvidenceFixture(status="missing")
    return DatasetEvidenceFixture(
        status="configured",
        id=target_config.fixture_id,
        version=target_config.fixture_version,
    )


def validate_dataset_resume_evidence(
    raw_evidence: bytes,
    *,
    expected_context: DatasetEvidenceRunContext,
    selected_records: tuple[InteractionRecord, ...],
    invariant_suite: DatasetInvariantSuite | None,
    evidence_projector: Callable[..., dict[str, JsonValue]],
) -> DatasetResumeEvidence:
    if invariant_suite is None:
        if expected_context.invariant_suite_sha256 is not None:
            raise ValueError("resume invariant suite is missing from the current evaluation plan")
    elif invariant_suite.sha256 != expected_context.invariant_suite_sha256:
        raise ValueError("resume invariant suite does not match the current evaluation plan")
    raw_lines = raw_evidence.splitlines()
    if not raw_lines:
        return DatasetResumeEvidence(
            processed_ids=frozenset(),
            has_review_findings=False,
            invariant_evaluations=(),
            technical_results=(),
            raw_evidence_sha256=hashlib.sha256(raw_evidence).hexdigest(),
        )
    if len(raw_lines) > _MAXIMUM_EVIDENCE_RECORDS:
        raise ValueError("resume evidence must contain at most 100 JSONL records")
    if any(not raw_line.strip() for raw_line in raw_lines):
        raise ValueError("resume evidence contains an empty JSONL record")
    selected_records_by_id = {record.id: record for record in selected_records}
    processed_ids: set[str] = set()
    has_review_findings = False
    invariant_evaluations: list[DatasetInvariantEvaluation] = []
    technical_results: list[DatasetEvaluationResult] = []
    for raw_line in raw_lines:
        try:
            evidence = _EvidenceRecord.model_validate_json(raw_line)
        except (ValidationError, ValueError):
            raise ValueError("resume evidence is not valid UL JSONL") from None
        if (
            evidence.schema_version not in {"1.5.0", "1.6.0", "1.7.0", "1.8.0"}
            or evidence.run_context is None
        ):
            raise ValueError(
                "resume requires evidence created with schema 1.5.0 or later run "
                "compatibility metadata"
            )
        if evidence.run_context != expected_context:
            raise ValueError("resume evidence is incompatible with the current evaluation plan")
        if evidence.interaction_id in processed_ids:
            raise ValueError("resume evidence contains duplicate interaction IDs")
        selected_record = selected_records_by_id.get(evidence.interaction_id)
        if selected_record is None:
            raise ValueError("resume evidence contains an interaction outside the selected dataset")
        try:
            technical_result = DatasetEvaluationResult.model_validate_json(
                json.dumps(evidence.technical_details, ensure_ascii=False)
            )
        except (ValidationError, ValueError):
            raise ValueError("resume evidence contains invalid technical details") from None
        technical_results.append(technical_result)
        if technical_result.evaluation_mode != evidence.evaluation_mode:
            raise ValueError("resume evidence evaluation mode does not match technical details")
        if technical_result.source != selected_record:
            raise ValueError("resume evidence source does not match the selected dataset")
        if evidence.original_input != technical_result.source.raw_input:
            raise ValueError("resume evidence original input does not match its technical details")
        if evidence.execution_plan.repetitions != expected_context.repetitions:
            raise ValueError("resume evidence repetitions do not match the current evaluation plan")
        technical_operators = tuple(
            (case.candidate.operator_id, case.candidate.operator_version)
            for case in technical_result.cases
        )
        public_operators = tuple(
            (case.operator_id, case.operator_version) for case in evidence.cases
        )
        expected_operators = tuple(
            (operator.id, operator.version) for operator in expected_context.operators
        )
        if technical_operators != expected_operators or public_operators != expected_operators:
            raise ValueError("resume evidence operators do not match the current evaluation plan")
        if (
            technical_result.baseline.trial_set.requested_repetitions
            != expected_context.repetitions
        ):
            raise ValueError("resume evidence technical repetitions are incompatible")
        expected_invariant_evaluation = (
            evaluate_dataset_invariants(technical_result, invariant_suite)
            if invariant_suite is not None
            else None
        )
        if evidence.invariant_evaluation != expected_invariant_evaluation:
            raise ValueError("resume evidence invariant results do not match technical details")
        projected_evidence = evidence_projector(
            technical_result,
            repetitions=evidence.execution_plan.repetitions,
            max_environment_api_calls=evidence.execution_plan.max_target_calls,
            planned_target_calls=evidence.execution_plan.dataset_planned_target_calls,
            run_context=expected_context,
            invariant_evaluation=expected_invariant_evaluation,
        )
        if evidence.model_dump(mode="json") != projected_evidence:
            raise ValueError("resume evidence public summary does not match its technical details")
        has_review_findings |= any(
            case.verdict == "divergence_needs_review" for case in technical_result.cases
        )
        if expected_invariant_evaluation is not None:
            invariant_evaluations.append(expected_invariant_evaluation)
        processed_ids.add(evidence.interaction_id)
    return DatasetResumeEvidence(
        processed_ids=frozenset(processed_ids),
        has_review_findings=has_review_findings,
        invariant_evaluations=tuple(invariant_evaluations),
        technical_results=tuple(technical_results),
        raw_evidence_sha256=hashlib.sha256(raw_evidence).hexdigest(),
    )


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ReviewRecord(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    review_id: str = Field(pattern=_REVIEW_ID_PATTERN)
    evidence_record_sha256: str = Field(pattern=_SHA256_PATTERN)
    finding_id: str = Field(pattern=_FINDING_ID_PATTERN)
    status: ReviewStatus
    severity: ReviewSeverity = "unrated"
    reviewer: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=4000)
    reviewed_at: datetime
    supersedes_review_id: str | None = Field(default=None, pattern=_REVIEW_ID_PATTERN)

    @model_validator(mode="after")
    def validate_review(self) -> Self:
        if not self.reviewer.strip() or not self.reason.strip():
            raise ValueError("reviewer and reason must contain non-whitespace text")
        if self.status != "confirmed" and self.severity != "unrated":
            raise ValueError("only confirmed findings can have a rated severity")
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() != UTC.utcoffset(None):
            raise ValueError("reviewed_at must use UTC")
        return self


class _ReviewInputError(ValueError):
    pass


class _LoadedEvidenceRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    evidence: _EvidenceRecord
    sha256: str


@dataclass(frozen=True)
class _IndexedFinding:
    finding_id: str
    kind: Literal["semantic_difference", "customer_invariant_violation"]
    evidence_record: _LoadedEvidenceRecord
    case: _Case
    semantic_finding: _Finding | None = None
    baseline_rule: DatasetInvariantRuleResult | None = None
    variation_rule: DatasetInvariantRuleResult | None = None


class ConfirmedDatasetFinding(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    evidence_record: _LoadedEvidenceRecord
    case: _Case
    review: ReviewRecord
    kind: Literal["semantic_difference", "customer_invariant_violation"]
    invariant_rule_id: str | None = None


def load_confirmed_dataset_finding(
    evidence_path: Path,
    reviews_path: Path,
    finding_id: str,
) -> ConfirmedDatasetFinding:
    evidence_records = _load_evidence(evidence_path)
    findings = _index_findings(evidence_records)
    selected = findings.get(finding_id)
    if selected is None:
        raise ValueError("finding ID was not found in the evidence")
    review_records = _load_reviews(reviews_path)
    _validate_review_history(review_records, findings)
    active_review = _active_reviews(review_records).get(finding_id)
    if active_review is None or active_review.status != "confirmed":
        raise ValueError("finding must have an active confirmed review")
    return ConfirmedDatasetFinding(
        evidence_record=selected.evidence_record,
        case=selected.case,
        review=active_review,
        kind=selected.kind,
        invariant_rule_id=(
            selected.variation_rule.rule_id if selected.variation_rule is not None else None
        ),
    )


_BEHAVIOR_FINDING_SUMMARIES: dict[str, FindingSummaryText] = {
    "duplicate_effect": "The changed input made the agent repeat an action.",
    "unexpected_effect": "The changed input made the agent take a new action.",
    "missing_effect": "The changed input made the agent skip a baseline action.",
    "changed_grounded_effect_argument": "The changed input altered an important action detail.",
    "unstable_behavior": "The changed input produced inconsistent behavior across repetitions.",
}


def summarize_dataset_evidence(
    evidence: Path,
    reviews: Path | None = None,
) -> UnifiedReport:
    try:
        return _summarize_dataset_evidence(evidence, reviews)
    except _ReviewInputError as error:
        raise ReportInputError(str(error)) from None
    except (ValidationError, ValueError):
        raise ReportInputError("dataset evidence cannot be summarized safely") from None


def _summarize_dataset_evidence(
    evidence: Path,
    reviews: Path | None,
) -> UnifiedReport:
    evidence_records = _load_evidence(evidence)
    evaluation_mode = _dataset_evaluation_mode(evidence_records)
    review_records = _load_reviews(reviews or _default_reviews_path(evidence))
    indexed_findings = _index_findings(evidence_records)
    _validate_review_history(review_records, indexed_findings)
    active_reviews = _active_reviews(review_records)

    finding_summaries: list[FindingSummary] = []
    pattern_contexts: dict[str, tuple[str, str, str]] = {}
    indexed_invariant_keys: set[tuple[str, str, str, str]] = set()
    for indexed_finding in indexed_findings.values():
        active_review = active_reviews.get(indexed_finding.finding_id)
        review_status: FindingReviewStatus = (
            active_review.status if active_review is not None else "needs_review"
        )
        review_severity: FindingSeverity = (
            active_review.severity if active_review is not None else "unrated"
        )
        observations = indexed_finding.case.observations
        if observations is None:
            raise AssertionError("indexed findings require variation observations")
        next_action = (
            "review_dataset_finding"
            if review_status == "needs_review"
            else "inspect_dataset_evidence"
        )
        if indexed_finding.semantic_finding is not None:
            category = indexed_finding.semantic_finding.category
            summary = _BEHAVIOR_FINDING_SUMMARIES.get(category)
            if summary is None:
                raise _ReviewInputError("evidence contains an unsupported finding category")
            finding_summaries.append(
                FindingSummary(
                    finding_id=indexed_finding.finding_id,
                    kind="behavior_difference",
                    category=cast(FindingCategory, category),
                    operator_id=indexed_finding.case.operator_id,
                    operator_version=indexed_finding.case.operator_version,
                    review_status=review_status,
                    review_severity=review_severity,
                    requested_repetitions=observations.requested_repetitions,
                    conclusive_repetitions=observations.observed_repetitions,
                    inconclusive_repetitions=observations.inconclusive_repetitions,
                    stability=observations.stability,
                    next_action=next_action,
                    summary=summary,
                )
            )
            if review_status in {"needs_review", "confirmed"}:
                semantic_finding = indexed_finding.semantic_finding
                pattern_signature = _behavior_pattern_signature(semantic_finding)
                if pattern_signature is not None:
                    pattern_contexts[indexed_finding.finding_id] = (
                        pattern_signature,
                        indexed_finding.evidence_record.evidence.interaction_id,
                        indexed_finding.case.operator_id,
                    )
            continue

        variation_rule = indexed_finding.variation_rule
        if variation_rule is None:
            raise AssertionError("indexed invariant finding requires a variation rule")
        indexed_invariant_keys.add(
            (
                indexed_finding.evidence_record.sha256,
                indexed_finding.case.operator_id,
                variation_rule.rule_id,
                variation_rule.rule_version,
            )
        )
        finding_summaries.append(
            FindingSummary(
                finding_id=indexed_finding.finding_id,
                kind="customer_invariant_violation",
                category="customer_invariant_violation",
                operator_id=indexed_finding.case.operator_id,
                operator_version=indexed_finding.case.operator_version,
                rule_id=variation_rule.rule_id,
                rule_version=variation_rule.rule_version,
                declared_severity=variation_rule.severity,
                review_status=review_status,
                review_severity=review_severity,
                requested_repetitions=observations.requested_repetitions,
                conclusive_repetitions=observations.observed_repetitions,
                inconclusive_repetitions=observations.inconclusive_repetitions,
                stability=observations.stability,
                violated_repetitions=sum(
                    trial.status == "violated" for trial in variation_rule.trials
                ),
                next_action=next_action,
                summary="The agent violated a customer-defined rule.",
            )
        )
        if review_status in {"needs_review", "confirmed"}:
            pattern_signature = _canonical_json_sha256(
                {
                    "kind": "customer_invariant_violation",
                    "rule_id": variation_rule.rule_id,
                    "rule_version": variation_rule.rule_version,
                }
            )
            pattern_contexts[indexed_finding.finding_id] = (
                pattern_signature,
                indexed_finding.evidence_record.evidence.interaction_id,
                indexed_finding.case.operator_id,
            )

    for loaded_record in evidence_records:
        for case in loaded_record.evidence.cases:
            if (
                not case.findings
                and case.observations is not None
                and case.observations.stability == "unstable"
            ):
                finding_summaries.append(
                    FindingSummary(
                        kind="behavior_difference",
                        category="unstable_behavior",
                        operator_id=case.operator_id,
                        operator_version=case.operator_version,
                        requested_repetitions=case.observations.requested_repetitions,
                        conclusive_repetitions=case.observations.observed_repetitions,
                        inconclusive_repetitions=case.observations.inconclusive_repetitions,
                        stability=case.observations.stability,
                        next_action="inspect_dataset_evidence",
                        summary=(
                            "The changed input produced inconsistent behavior across repetitions."
                        ),
                    )
                )

    invariant_statuses: list[str] = []
    for loaded_record in evidence_records:
        evaluation = loaded_record.evidence.invariant_evaluation
        if evaluation is None:
            continue
        cases_by_operator = {case.operator_id: case for case in loaded_record.evidence.cases}
        arms = (
            (None, evaluation.baseline.rules),
            *((variation.operator_id, variation.rules) for variation in evaluation.variations),
        )
        for operator_id, rules in arms:
            operator_version = (
                cases_by_operator[operator_id].operator_version if operator_id is not None else None
            )
            for rule in rules:
                invariant_statuses.append(rule.status)
                if rule.status != "violated":
                    continue
                if (
                    operator_id is not None
                    and (
                        loaded_record.sha256,
                        operator_id,
                        rule.rule_id,
                        rule.rule_version,
                    )
                    in indexed_invariant_keys
                ):
                    continue
                observations = (
                    loaded_record.evidence.current_baseline.observations
                    if operator_id is None
                    else cases_by_operator[operator_id].observations
                )
                if observations is None:
                    raise AssertionError("violated invariant rules require observations")
                finding_summaries.append(
                    FindingSummary(
                        kind="customer_invariant_violation",
                        category="customer_invariant_violation",
                        operator_id=operator_id,
                        operator_version=operator_version,
                        rule_id=rule.rule_id,
                        rule_version=rule.rule_version,
                        declared_severity=rule.severity,
                        requested_repetitions=observations.requested_repetitions,
                        conclusive_repetitions=observations.observed_repetitions,
                        inconclusive_repetitions=observations.inconclusive_repetitions,
                        stability=observations.stability,
                        violated_repetitions=sum(
                            trial.status == "violated" for trial in rule.trials
                        ),
                        next_action="inspect_dataset_evidence",
                        summary="The agent violated a customer-defined rule.",
                    )
                )

    findings = tuple(finding_summaries)
    patterns = _build_failure_patterns(findings, pattern_contexts)
    summary = build_report_summary(findings)
    if summary.actionable_finding_count:
        report_review_status: ReportReviewStatus = "action_required"
    elif (
        summary.review_status_counts.inconclusive
        or "not_evaluable" in invariant_statuses
        or _dataset_evidence_is_inconclusive(evidence_records)
    ):
        report_review_status = "inconclusive"
    else:
        report_review_status = "resolved"
    exit_code = cast(
        Literal[0, 1, 2],
        {"resolved": 0, "action_required": 1, "inconclusive": 2}[report_review_status],
    )
    target_configs = tuple(
        loaded_record.evidence.run_context.target.config
        for loaded_record in evidence_records
        if loaded_record.evidence.run_context is not None
    )
    if any(
        isinstance(config, JsonHttpIsolatedResponseConfig) for config in target_configs
    ) and not all(isinstance(config, JsonHttpIsolatedResponseConfig) for config in target_configs):
        raise _ReviewInputError("evidence combines incompatible target capability tiers")
    response_only = bool(target_configs) and all(
        isinstance(config, JsonHttpIsolatedResponseConfig) for config in target_configs
    )
    return UnifiedReport(
        evidence_type="dataset_evaluation",
        evidence_schema_versions=tuple(
            sorted({record.evidence.schema_version for record in evidence_records})
        ),
        evidence_scope="response_only" if response_only else "response_and_state",
        evaluation_mode=evaluation_mode,
        capability_limitations=(
            ("cleanup_verification", "conversation_replay", "state_observation")
            if response_only
            else ()
        ),
        review_status=report_review_status,
        exit_code=exit_code,
        summary=summary,
        patterns=patterns,
        findings=findings,
    )


def _dataset_evaluation_mode(
    records: list[_LoadedEvidenceRecord],
) -> Literal["variance"] | None:
    evaluation_modes: set[Literal["variance"] | None] = {
        record.evidence.evaluation_mode for record in records
    }
    if len(evaluation_modes) != 1:
        raise _ReviewInputError("evidence combines incompatible evaluation modes")
    return next(iter(evaluation_modes))


def _behavior_pattern_signature(finding: _Finding) -> str | None:
    if len(finding.grounded_field_names) > _MAXIMUM_PATTERN_FIELDS or any(
        len(field_name) > _MAXIMUM_PATTERN_LABEL_CHARACTERS
        for field_name in finding.grounded_field_names
    ):
        return None
    reference_effects = _bounded_effect_mechanisms(finding.reference_effects)
    observed_effects = _bounded_effect_mechanisms(finding.observed_effects)
    if reference_effects is None or observed_effects is None:
        return None
    return _canonical_json_sha256(
        {
            "kind": "behavior_difference",
            "category": finding.category,
            "grounded_field_names": sorted(finding.grounded_field_names),
            "reference_effects": reference_effects,
            "observed_effects": observed_effects,
        }
    )


def _bounded_effect_mechanisms(effects: list[_Effect]) -> list[dict[str, object]] | None:
    if len(effects) > _MAXIMUM_PATTERN_EFFECTS:
        return None
    if any(
        len(effect.fields) > _MAXIMUM_PATTERN_FIELDS
        or len(effect.kind) > _MAXIMUM_PATTERN_LABEL_CHARACTERS
        or len(effect.predicate) > _MAXIMUM_PATTERN_LABEL_CHARACTERS
        or any(len(field_name) > _MAXIMUM_PATTERN_LABEL_CHARACTERS for field_name in effect.fields)
        for effect in effects
    ):
        return None
    return sorted(
        (
            {
                "kind": effect.kind,
                "predicate": effect.predicate,
                "status": effect.status,
                "field_names": sorted(effect.fields),
            }
            for effect in effects
        ),
        key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")),
    )


def _build_failure_patterns(
    findings: tuple[FindingSummary, ...],
    contexts: dict[str, tuple[str, str, str]],
) -> tuple[FailurePattern, ...]:
    findings_by_id = {
        finding.finding_id: finding for finding in findings if finding.finding_id is not None
    }
    grouped: dict[str, list[tuple[FindingSummary, str, str]]] = {}
    for finding_id, (signature, source_case_id, operator_id) in contexts.items():
        grouped.setdefault(signature, []).append(
            (findings_by_id[finding_id], source_case_id, operator_id)
        )

    catalog = builtin_augmentation_catalog()
    patterns: list[FailurePattern] = []
    for members in grouped.values():
        first = members[0][0]
        operator_keys = sorted(
            {
                (operator_id, finding.operator_version)
                for finding, _, operator_id in members
                if finding.operator_version is not None
            }
        )
        operators: list[PatternOperator] = []
        for operator_id, operator_version in operator_keys:
            try:
                operator_summary = catalog.get(operator_id, operator_version).summary
            except KeyError:
                operator_summary = None
            operators.append(
                PatternOperator(
                    operator_id=operator_id,
                    operator_version=operator_version,
                    summary=operator_summary,
                )
            )
        finding_ids = tuple(sorted(cast(str, finding.finding_id) for finding, _, _ in members))
        public_pattern_digest = _canonical_json_sha256({"finding_ids": finding_ids})
        member_severities: list[FindingSeverity] = []
        for finding, _, _ in members:
            if finding.review_status == "confirmed":
                if finding.review_severity is None:
                    raise AssertionError("validated confirmed finding requires review severity")
                member_severities.append(finding.review_severity)
            else:
                member_severities.append(finding.declared_severity or "unrated")
        pattern_severity = sorted(
            member_severities,
            key=_PATTERN_SEVERITY_RANK.__getitem__,
        )[-1]
        patterns.append(
            FailurePattern(
                pattern_id=f"ulp_v1_{public_pattern_digest}",
                kind=first.kind,
                category=first.category,
                rule_id=first.rule_id,
                rule_version=first.rule_version,
                summary=first.summary,
                severity=pattern_severity,
                finding_count=len(members),
                source_case_count=len({source_case_id for _, source_case_id, _ in members}),
                operators=tuple(operators),
                needs_review_count=sum(
                    finding.review_status == "needs_review" for finding, _, _ in members
                ),
                confirmed_count=sum(
                    finding.review_status == "confirmed" for finding, _, _ in members
                ),
                finding_ids=finding_ids,
            )
        )
    return tuple(
        sorted(
            patterns,
            key=lambda pattern: (
                -_PATTERN_SEVERITY_RANK[pattern.severity],
                pattern.pattern_id,
            ),
        )
    )


def _dataset_evidence_is_inconclusive(records: list[_LoadedEvidenceRecord]) -> bool:
    for loaded_record in records:
        evidence = loaded_record.evidence
        if (
            evidence.current_baseline.inconclusive_reasons
            or evidence.current_baseline.observations.inconclusive_repetitions > 0
        ):
            return True
        for case in evidence.cases:
            if case.inconclusive_reasons or (
                case.variation_accepted
                and (case.observations is None or case.observations.inconclusive_repetitions > 0)
            ):
                return True
    return False


def report_dataset_evidence(
    evidence: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, readable=True, help="Evaluation evidence JSONL."
        ),
    ],
    reviews: Annotated[
        Path | None,
        typer.Option(help="Review JSONL; defaults to EVIDENCE with .reviews.jsonl suffix."),
    ] = None,
    show_sensitive_values: Annotated[
        bool,
        typer.Option(
            help=(
                "Show capped values for one reviewable invariant finding. Values may contain "
                "secrets or PII and may enter terminal scrollback, CI output, or logs."
            )
        ),
    ] = False,
    sensitive_finding_id: Annotated[
        str | None,
        typer.Option(
            "--finding",
            help="Finding ID whose invariant values may be disclosed.",
        ),
    ] = None,
) -> None:
    """Show findings and their human review state without model or network calls."""
    reviews_path = reviews or _default_reviews_path(evidence)
    try:
        evidence_records = _load_evidence(evidence)
        review_records = _load_reviews(reviews_path)
        findings = _index_findings(evidence_records)
        _validate_review_history(review_records, findings)
        sensitive_lines: tuple[str, ...] = ()
        if show_sensitive_values:
            if sensitive_finding_id is None:
                raise _ReviewInputError("--show-sensitive-values requires --finding FINDING_ID")
            sensitive_finding = findings.get(sensitive_finding_id)
            if sensitive_finding is None:
                raise _ReviewInputError("sensitive-value finding ID was not found in the evidence")
            if sensitive_finding.baseline_rule is None or sensitive_finding.variation_rule is None:
                raise _ReviewInputError(
                    "sensitive values are available only for a reviewable invariant finding"
                )
            sensitive_lines = _bounded_sensitive_invariant_lines(
                sensitive_finding.baseline_rule,
                sensitive_finding.variation_rule,
            )
        elif sensitive_finding_id is not None:
            raise _ReviewInputError("--finding is valid only with --show-sensitive-values")
    except _ReviewInputError as error:
        raise typer.BadParameter(str(error)) from None

    active_reviews = _active_reviews(review_records)
    status_counts = {
        status: 0
        for status in ("needs_review", "confirmed", "expected", "unsupported", "inconclusive")
    }
    for indexed_finding in findings.values():
        active_review = active_reviews.get(indexed_finding.finding_id)
        status_counts[active_review.status if active_review else "needs_review"] += 1

    _print_plain(f"Dataset finding report: {len(findings)} finding(s)")
    evaluation_mode = _dataset_evaluation_mode(evidence_records)
    if evaluation_mode is not None:
        _print_plain(
            f"Evaluation mode: {evaluation_mode} (historical output is not an expected answer; "
            "correctness not assessed)"
        )
    _print_plain(
        "Reviews: " + ", ".join(f"{status}={count}" for status, count in status_counts.items())
    )
    if show_sensitive_values:
        _print_plain(
            "WARNING: showing selected invariant values; they may contain secrets or PII and "
            "may be retained in terminal scrollback, CI output, or logs."
        )
    for indexed_finding in findings.values():
        loaded_record = indexed_finding.evidence_record
        case = indexed_finding.case
        matching_reviews = [
            review for review in review_records if review.finding_id == indexed_finding.finding_id
        ]
        latest_review = active_reviews.get(indexed_finding.finding_id)
        _print_plain("")
        _print_plain(f"Finding {indexed_finding.finding_id}")
        if indexed_finding.semantic_finding is not None:
            finding = indexed_finding.semantic_finding
            _print_plain(f"Machine status: {case.status}")
            _print_plain(f"Category: {finding.category}")
            _print_plain(f"Summary: {finding.summary}")
        else:
            baseline_rule = indexed_finding.baseline_rule
            variation_rule = indexed_finding.variation_rule
            if baseline_rule is None or variation_rule is None:
                raise AssertionError("invariant finding requires both rule results")
            _print_plain(f"Semantic comparison status: {case.status}")
            _print_plain(
                f"Invariant finding status: original={baseline_rule.status}; "
                f"variation={variation_rule.status}"
            )
            _print_plain("Category: customer_invariant_violation")
            _print_plain(
                f"Summary: Customer rule {variation_rule.rule_id} was satisfied by the original "
                "and violated by the variation."
            )
            _print_plain(
                f"Rule: {variation_rule.rule_id} ({variation_rule.rule_version}); "
                f"type={variation_rule.rule_type}; declared_severity={variation_rule.severity}"
            )
            _print_plain(f"Description: {variation_rule.description}")
        _print_plain(f"Original: {loaded_record.evidence.original_input}")
        _print_plain(f"Variation: {case.augmented_input}")
        _print_plain(f"Operator: {case.operator_id} ({case.operator_version})")
        _print_plain(
            "Original trials: "
            + _observations_summary(loaded_record.evidence.current_baseline.observations)
        )
        _print_plain("Variation trials: " + _observations_summary(case.observations))
        if indexed_finding.semantic_finding is not None:
            _print_plain(
                "Reference effects: "
                + _effects_summary(indexed_finding.semantic_finding.reference_effects)
            )
            _print_plain(
                "Observed effects: "
                + _effects_summary(indexed_finding.semantic_finding.observed_effects)
            )
        else:
            baseline_rule = indexed_finding.baseline_rule
            variation_rule = indexed_finding.variation_rule
            if baseline_rule is None or variation_rule is None:
                raise AssertionError("invariant finding requires both rule results")
            _print_plain(
                f"Rule transition: original={baseline_rule.status}; "
                f"variation={variation_rule.status}"
            )
            for trial in variation_rule.trials:
                _print_plain(
                    f"Variation rule trial {trial.repetition}: {trial.status}; "
                    f"{_invariant_trial_location(trial)}; reason={trial.reason_code}"
                )
            if show_sensitive_values and indexed_finding.finding_id == sensitive_finding_id:
                for sensitive_line in sensitive_lines:
                    _print_sensitive_plain(sensitive_line)
        if latest_review is None:
            _print_plain(f"Latest review: needs_review (history: {len(matching_reviews)})")
        else:
            _print_plain(
                f"Latest review: {latest_review.status}, severity={latest_review.severity}, "
                f"reviewer={latest_review.reviewer}, id={latest_review.review_id} "
                f"(history: {len(matching_reviews)})"
            )
            _print_plain(f"Review reason: {latest_review.reason}")
        _print_plain(f"Evidence record SHA-256: {loaded_record.sha256}")

    invariant_evaluations = [
        loaded_record.evidence.invariant_evaluation
        for loaded_record in evidence_records
        if loaded_record.evidence.invariant_evaluation is not None
    ]
    if invariant_evaluations:
        _print_plain("")
        _print_plain("Customer invariant evaluation")
        for invariant_evaluation in invariant_evaluations:
            _print_invariant_evaluation(invariant_evaluation)

    if show_sensitive_values:
        _print_plain(
            "Sensitive value disclosure: "
            f"finding={sensitive_finding_id}; shown={len(sensitive_lines)}; omitted=0; "
            f"limit={_MAXIMUM_SENSITIVE_DISCLOSURE_LINES} lines/"
            f"{_MAXIMUM_SENSITIVE_DISCLOSURE_BYTES} UTF-8 bytes."
        )

    _print_plain("")
    _print_plain(f"Complete technical evidence: {evidence}")
    _print_plain(f"Review history: {reviews_path}")
    _print_plain(
        "Review meanings: confirmed=problem in context; expected=supported acceptable difference; "
        "unsupported=machine claim not supported; inconclusive=insufficient context."
    )
    _print_plain(
        "Limitation: UL reports observed differences. A human review is a contextual judgment, "
        "not proof of correctness, causation, or production frequency."
    )


def review_dataset_finding(
    evidence: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, readable=True, help="Evaluation evidence JSONL."
        ),
    ],
    finding_id: Annotated[str, typer.Argument(help="Finding ID shown by 'ul dataset report'.")],
    status: Annotated[ReviewStatus, typer.Option(help="Human review judgment.")],
    reviewer: Annotated[str, typer.Option(help="Person or team making the judgment.")],
    reason: Annotated[str, typer.Option(help="Why this judgment applies in context.")],
    severity: Annotated[
        ReviewSeverity,
        typer.Option(help="Consequence severity; only confirmed findings may be rated."),
    ] = "unrated",
    supersedes: Annotated[
        str | None,
        typer.Option(help="Current active review ID replaced by this judgment."),
    ] = None,
    reviews: Annotated[
        Path | None,
        typer.Option(help="Review JSONL; defaults to EVIDENCE with .reviews.jsonl suffix."),
    ] = None,
) -> None:
    """Append a human judgment while preserving the complete review history."""
    reviews_path = reviews or _default_reviews_path(evidence)
    try:
        evidence_records = _load_evidence(evidence)
        findings = _index_findings(evidence_records)
        selected = findings.get(finding_id)
        if selected is None:
            raise _ReviewInputError("finding ID was not found in the evidence")
        loaded_record = selected.evidence_record
        new_review = ReviewRecord(
            review_id=f"ulr_{uuid4()}",
            evidence_record_sha256=loaded_record.sha256,
            finding_id=finding_id,
            status=status,
            severity=severity,
            reviewer=reviewer,
            reason=reason,
            reviewed_at=datetime.now(UTC),
            supersedes_review_id=supersedes,
        )
        with _locked_reviews_file(reviews_path) as descriptor:
            review_records = _read_reviews_descriptor(descriptor)
            _validate_review_history(review_records, findings)
            active_review = _active_reviews(review_records).get(finding_id)
            if active_review is None and supersedes is not None:
                raise _ReviewInputError(
                    "--supersedes is only valid when replacing an active review"
                )
            if active_review is not None and supersedes != active_review.review_id:
                raise _ReviewInputError(
                    "finding already has an active review; pass --supersedes "
                    f"{active_review.review_id}"
                )
            encoded_review = (
                json.dumps(
                    new_review.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")
                )
                + "\n"
            ).encode()
            if os.fstat(descriptor).st_size + len(encoded_review) > _MAXIMUM_REVIEWS_BYTES:
                raise _ReviewInputError("review file exceeds the 10 MB limit")
            remaining = memoryview(encoded_review)
            while remaining:
                written = os.write(descriptor, remaining)
                if written == 0:
                    raise OSError("could not append review")
                remaining = remaining[written:]
            os.fsync(descriptor)
    except (ValidationError, _ReviewInputError) as error:
        message = "review fields are invalid" if isinstance(error, ValidationError) else str(error)
        raise typer.BadParameter(message) from None
    except OSError as error:
        raise typer.BadParameter(
            f"cannot safely update review file ({error.__class__.__name__})"
        ) from None

    _print_plain(f"Recorded review {new_review.review_id}: {new_review.status}")
    _print_plain(f"Review history: {reviews_path}")


def _default_reviews_path(evidence: Path) -> Path:
    return evidence.with_suffix(".reviews.jsonl")


def _load_evidence(path: Path) -> list[_LoadedEvidenceRecord]:
    try:
        raw = _read_bounded_regular_file(path, _MAXIMUM_EVIDENCE_BYTES)
    except OSError as error:
        raise _ReviewInputError(
            f"cannot safely read evidence ({error.__class__.__name__})"
        ) from None
    raw_lines = raw.splitlines()
    if not raw_lines or len(raw_lines) > _MAXIMUM_EVIDENCE_RECORDS:
        raise _ReviewInputError("evidence must contain 1 to 100 JSONL records")
    if any(not raw_line.strip() for raw_line in raw_lines):
        raise _ReviewInputError("evidence contains an empty JSONL record")
    records: list[_LoadedEvidenceRecord] = []
    try:
        for raw_line in raw_lines:
            records.append(
                _LoadedEvidenceRecord(
                    evidence=_EvidenceRecord.model_validate_json(raw_line),
                    sha256=hashlib.sha256(raw_line).hexdigest(),
                )
            )
    except (ValidationError, ValueError):
        raise _ReviewInputError("evidence is not valid UL schema through 1.8.0 JSONL") from None
    return records


def is_reportable_dataset_evidence(path: Path) -> bool:
    try:
        _load_evidence(path)
    except _ReviewInputError:
        return False
    return True


def _load_reviews(path: Path) -> list[ReviewRecord]:
    try:
        descriptor = _open_regular_file(path, os.O_RDONLY)
        try:
            _lock_file(descriptor, exclusive=False)
            try:
                return _read_reviews_descriptor(descriptor)
            finally:
                _unlock_file(descriptor)
        finally:
            os.close(descriptor)
    except FileNotFoundError:
        return []
    except OSError as error:
        raise _ReviewInputError(
            f"cannot safely read review file ({error.__class__.__name__})"
        ) from None


def _read_reviews_descriptor(descriptor: int) -> list[ReviewRecord]:
    size = os.fstat(descriptor).st_size
    if size > _MAXIMUM_REVIEWS_BYTES:
        raise _ReviewInputError("review file exceeds the 10 MB limit")
    os.lseek(descriptor, 0, os.SEEK_SET)
    raw = b""
    while len(raw) < size:
        chunk = os.read(descriptor, min(65_536, size - len(raw)))
        if not chunk:
            break
        raw += chunk
    raw_lines = raw.splitlines()
    if len(raw_lines) > _MAXIMUM_REVIEW_RECORDS:
        raise _ReviewInputError("review file exceeds the 10,000 record limit")
    if any(not line.strip() for line in raw_lines):
        raise _ReviewInputError("review file contains an empty JSONL record")
    try:
        return [ReviewRecord.model_validate_json(line) for line in raw_lines]
    except (ValidationError, ValueError):
        raise _ReviewInputError("review file is not valid review JSONL") from None


def _read_bounded_regular_file(path: Path, maximum_bytes: int) -> bytes:
    descriptor = _open_regular_file(path, os.O_RDONLY)
    try:
        size = os.fstat(descriptor).st_size
        if size > maximum_bytes:
            raise _ReviewInputError("evidence exceeds the 128 MB limit")
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _open_regular_file(path: Path, flags: int) -> int:
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    requires_identity_check = no_follow_flag == 0
    if requires_identity_check:
        try:
            if stat.S_ISLNK(os.lstat(path).st_mode):
                raise OSError("path is a symbolic link")
        except FileNotFoundError:
            pass
    binary_flag = os.O_BINARY if sys.platform == "win32" else 0
    open_flags = flags | no_follow_flag | binary_flag
    descriptor = os.open(path, open_flags, 0o600)
    try:
        descriptor_status = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_status.st_mode):
            raise OSError("path is not a regular file")
        if requires_identity_check:
            path_status = os.lstat(path)
            if stat.S_ISLNK(path_status.st_mode) or not os.path.samestat(
                descriptor_status, path_status
            ):
                raise OSError("path changed while it was opened")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def _locked_reviews_file(path: Path) -> Generator[int]:
    descriptor = _open_regular_file(path, os.O_RDWR | os.O_APPEND | os.O_CREAT)
    locked = False
    try:
        _set_private_file_permissions(descriptor)
        _lock_file(descriptor, exclusive=True)
        locked = True
        yield descriptor
    finally:
        try:
            if locked:
                _unlock_file(descriptor)
        finally:
            os.close(descriptor)


def _lock_file(descriptor: int, *, exclusive: bool) -> None:
    if sys.platform == "win32":
        os.lseek(descriptor, 0, os.SEEK_SET)
        mode = msvcrt.LK_LOCK if exclusive else msvcrt.LK_RLCK
        msvcrt.locking(descriptor, mode, 1)
        return
    mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    fcntl.flock(descriptor, mode)


def _unlock_file(descriptor: int) -> None:
    if sys.platform == "win32":
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return
    fcntl.flock(descriptor, fcntl.LOCK_UN)


def _set_private_file_permissions(descriptor: int) -> None:
    if sys.platform != "win32":
        os.fchmod(descriptor, 0o600)


def _index_findings(
    records: list[_LoadedEvidenceRecord],
) -> dict[str, _IndexedFinding]:
    findings: dict[str, _IndexedFinding] = {}
    for loaded_record in records:
        for case in loaded_record.evidence.cases:
            for finding in case.findings:
                if finding.finding_id in findings:
                    raise _ReviewInputError("evidence contains a duplicate finding ID")
                findings[finding.finding_id] = _IndexedFinding(
                    finding_id=finding.finding_id,
                    kind="semantic_difference",
                    evidence_record=loaded_record,
                    case=case,
                    semantic_finding=finding,
                )
        evaluation = loaded_record.evidence.invariant_evaluation
        if evaluation is None:
            continue
        expected_repetitions = loaded_record.evidence.execution_plan.repetitions
        if not _observations_cover_repetitions(
            loaded_record.evidence.current_baseline.observations,
            expected_repetitions,
        ) or any(len(rule.trials) != expected_repetitions for rule in evaluation.baseline.rules):
            raise _ReviewInputError("invariant baseline evidence has inconsistent repetitions")
        run_context = loaded_record.evidence.run_context
        if run_context is not None:
            if (
                run_context.invariant_suite_sha256 != evaluation.suite_sha256
                or run_context.repetitions != expected_repetitions
            ):
                raise _ReviewInputError("invariant evidence does not match the run context")
            configured_operators = {
                (operator.id, operator.version) for operator in run_context.operators
            }
            if any(
                (case.operator_id, case.operator_version) not in configured_operators
                for case in loaded_record.evidence.cases
            ):
                raise _ReviewInputError("evidence case does not match a run-context operator")
        cases_by_operator: dict[str, list[_Case]] = {}
        for case in loaded_record.evidence.cases:
            cases_by_operator.setdefault(case.operator_id, []).append(case)
        for variation in evaluation.variations:
            if variation.operator_id is None:
                raise _ReviewInputError("invariant variation is missing its operator ID")
            matching_cases = cases_by_operator.get(variation.operator_id, [])
            if len(matching_cases) != 1:
                raise _ReviewInputError("invariant variation must map to exactly one evidence case")
            case = matching_cases[0]
            if (
                not case.variation_accepted
                or case.observations is None
                or not _observations_cover_repetitions(case.observations, expected_repetitions)
                or any(len(rule.trials) != expected_repetitions for rule in variation.rules)
            ):
                raise _ReviewInputError(
                    "invariant variation must map to accepted, executed, repetition-consistent "
                    "evidence"
                )
        suite = _invariant_suite_from_evaluation(evaluation)
        _validate_invariant_technical_details(loaded_record.evidence, evaluation, suite)
        original_input_json = _canonical_json_string(loaded_record.evidence.original_input)
        baseline_rules = {rule.rule_id: rule for rule in evaluation.baseline.rules}
        for variation in evaluation.variations:
            if variation.operator_id is None:
                raise AssertionError("validated invariant variation requires an operator ID")
            case = cases_by_operator[variation.operator_id][0]
            augmented_input_json = _canonical_json_string(case.augmented_input)
            finding_id_prefix = _invariant_finding_id_prefix(
                loaded_record.evidence,
                case,
                evaluation,
                original_input_json=original_input_json,
                augmented_input_json=augmented_input_json,
            )
            for variation_rule in variation.rules:
                baseline_rule = baseline_rules.get(variation_rule.rule_id)
                if baseline_rule is None:
                    raise _ReviewInputError("invariant variation rule is missing from the baseline")
                if baseline_rule.status != "satisfied" or variation_rule.status != "violated":
                    continue
                finding_id = _invariant_finding_id(
                    finding_id_prefix,
                    evaluation,
                    variation_rule,
                )
                if finding_id in findings:
                    raise _ReviewInputError("evidence contains a duplicate finding ID")
                findings[finding_id] = _IndexedFinding(
                    finding_id=finding_id,
                    kind="customer_invariant_violation",
                    evidence_record=loaded_record,
                    case=case,
                    baseline_rule=baseline_rule,
                    variation_rule=variation_rule,
                )
    return findings


def _observations_cover_repetitions(
    observations: _Observations,
    expected_repetitions: int,
) -> bool:
    return (
        observations.requested_repetitions == expected_repetitions
        and tuple(trial.repetition for trial in observations.trials)
        == tuple(range(1, expected_repetitions + 1))
        and observations.observed_repetitions
        == sum(trial.status == "observed" for trial in observations.trials)
        and observations.inconclusive_repetitions
        == sum(trial.status == "inconclusive" for trial in observations.trials)
    )


def _invariant_suite_from_evaluation(
    evaluation: DatasetInvariantEvaluation,
) -> DatasetInvariantSuite:
    rules = tuple(_invariant_rule_definition(rule) for rule in evaluation.baseline.rules)
    schema_versions: tuple[Literal["1.0.0", "1.1.0", "1.2.0"], ...] = (
        ("1.0.0", "1.1.0", "1.2.0")
        if all(isinstance(rule, JsonValuesEqualInvariant) for rule in rules)
        else ("1.2.0",)
        if any(
            isinstance(
                rule,
                (
                    NoNewEffectInvariant,
                    ExactlyOneNewEffectInvariant,
                    UnchangedBetweenCheckpointsInvariant,
                ),
            )
            for rule in rules
        )
        else ("1.1.0", "1.2.0")
    )
    for schema_version in schema_versions:
        suite = DatasetInvariantSuite(
            schema_version=schema_version,
            observation_source=evaluation.observation_source,
            observation_authority=evaluation.observation_authority,
            rules=rules,
        )
        if suite.sha256 == evaluation.suite_sha256:
            return suite
    raise _ReviewInputError("invariant suite digest does not match its rule definitions")


def _invariant_rule_definition(rule: DatasetInvariantRuleResult) -> DatasetInvariantRule:
    if isinstance(rule, DatasetInvariantRuleEvaluation):
        first_trial = rule.trials[0]
        return JsonValuesEqualInvariant(
            type="json_values_equal",
            id=rule.rule_id,
            version=rule.rule_version,
            description=rule.description,
            severity=rule.severity,
            left_pointer=first_trial.left_pointer,
            right_pointer=first_trial.right_pointer,
        )
    if isinstance(rule, DatasetInvariantValueEqualsRuleEvaluation):
        return JsonValueEqualsLiteralInvariant(
            type="json_value_equals_literal",
            id=rule.rule_id,
            version=rule.rule_version,
            description=rule.description,
            severity=rule.severity,
            value_pointer=rule.value_pointer,
            literal=rule.literal,
        )
    if isinstance(rule, DatasetInvariantValueInSetRuleEvaluation):
        return JsonValueInAllowedSetInvariant(
            type="json_value_in_allowed_set",
            id=rule.rule_id,
            version=rule.rule_version,
            description=rule.description,
            severity=rule.severity,
            value_pointer=rule.value_pointer,
            allowed_values=rule.allowed_values,
        )
    if isinstance(rule, DatasetInvariantTransitionRuleEvaluation):
        transition_rule_types = {
            "no_new_effect": NoNewEffectInvariant,
            "exactly_one_new_effect": ExactlyOneNewEffectInvariant,
            "unchanged_between_checkpoints": UnchangedBetweenCheckpointsInvariant,
        }
        transition_rule_type = transition_rule_types[rule.rule_type]
        return transition_rule_type.model_validate(
            {
                "type": rule.rule_type,
                "id": rule.rule_id,
                "version": rule.rule_version,
                "description": rule.description,
                "severity": rule.severity,
                "before_checkpoint": rule.before_checkpoint,
                "after_checkpoint": rule.after_checkpoint,
                "observation_pointer": rule.observation_pointer,
            }
        )
    return JsonArrayItemsUniqueByInvariant(
        type="json_array_items_unique_by",
        id=rule.rule_id,
        version=rule.rule_version,
        description=rule.description,
        severity=rule.severity,
        array_pointer=rule.array_pointer,
        key_pointers=rule.key_pointers,
    )


def _validate_invariant_technical_details(
    evidence: _EvidenceRecord,
    evaluation: DatasetInvariantEvaluation,
    suite: DatasetInvariantSuite,
) -> None:
    try:
        technical_details = DatasetEvaluationResult.model_validate_json(
            json.dumps(evidence.technical_details, ensure_ascii=False, separators=(",", ":"))
        )
    except (ValidationError, ValueError):
        raise _ReviewInputError(
            "reviewable invariant findings require valid technical evaluation details"
        ) from None
    if (
        technical_details.source.id != evidence.interaction_id
        or technical_details.source.raw_input != evidence.original_input
    ):
        raise _ReviewInputError("invariant technical details do not match the evidence source")
    technical_operator_ids = tuple(case.candidate.operator_id for case in technical_details.cases)
    if len(technical_operator_ids) != len(set(technical_operator_ids)):
        raise _ReviewInputError("invariant technical details contain duplicate operators")
    technical_cases = {case.candidate.operator_id: case for case in technical_details.cases}
    for public_case in evidence.cases:
        technical_case = technical_cases.get(public_case.operator_id)
        if technical_case is None:
            continue
        candidate = technical_case.candidate
        if (
            candidate.operator_version != public_case.operator_version
            or candidate.augmented_input != public_case.augmented_input
            or candidate.passed != public_case.variation_accepted
        ):
            raise _ReviewInputError("invariant technical details do not match the evidence case")
    try:
        recomputed_evaluation = evaluate_dataset_invariants(technical_details, suite)
    except (ValidationError, ValueError):
        raise _ReviewInputError("invariant technical details cannot be safely recomputed") from None
    if recomputed_evaluation != evaluation:
        raise _ReviewInputError(
            "invariant evaluation does not match the technical execution evidence"
        )


def _invariant_finding_id_prefix(
    evidence: _EvidenceRecord,
    case: _Case,
    evaluation: DatasetInvariantEvaluation,
    *,
    original_input_json: bytes,
    augmented_input_json: bytes,
) -> _FindingIdHash:
    prefix = b"".join(
        (
            b'{"augmented_input":',
            augmented_input_json,
            b',"finding_kind":"customer_invariant_violation","interaction_id":',
            _canonical_json_string(evidence.interaction_id),
            b',"observation_authority":',
            _canonical_json_string(evaluation.observation_authority),
            b',"operator_id":',
            _canonical_json_string(case.operator_id),
            b',"operator_version":',
            _canonical_json_string(case.operator_version),
            b',"original_input":',
            original_input_json,
            b",",
        )
    )
    finding_id_hash = hashlib.sha256()
    finding_id_hash.update(prefix)
    return finding_id_hash


def _invariant_finding_id(
    finding_id_prefix: _FindingIdHash,
    evaluation: DatasetInvariantEvaluation,
    rule: DatasetInvariantRuleResult,
) -> str:
    suffix = b"".join(
        (
            b'"rule_id":',
            _canonical_json_string(rule.rule_id),
            b',"rule_type":',
            _canonical_json_string(rule.rule_type),
            b',"rule_version":',
            _canonical_json_string(rule.rule_version),
            b',"suite_sha256":',
            _canonical_json_string(evaluation.suite_sha256),
            b"}",
        )
    )
    finding_id_hash = finding_id_prefix.copy()
    finding_id_hash.update(suffix)
    return f"ulf_v1_{finding_id_hash.hexdigest()}"


def _canonical_json_string(value: str) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _validate_review_history(
    reviews: list[ReviewRecord],
    findings: dict[str, _IndexedFinding],
) -> None:
    reviews_by_id: dict[str, ReviewRecord] = {}
    superseded_ids: set[str] = set()
    active_by_finding: dict[str, ReviewRecord] = {}
    for review in reviews:
        if review.review_id in reviews_by_id:
            raise _ReviewInputError("review file contains a duplicate review ID")
        selected = findings.get(review.finding_id)
        if selected is None:
            raise _ReviewInputError("review references a finding outside this evidence")
        if review.evidence_record_sha256 != selected.evidence_record.sha256:
            raise _ReviewInputError("review evidence digest does not match the evidence record")
        current_active = active_by_finding.get(review.finding_id)
        if current_active is None:
            if review.supersedes_review_id is not None:
                raise _ReviewInputError("review supersedes a review that is not active")
        elif review.supersedes_review_id != current_active.review_id:
            raise _ReviewInputError("review history has multiple active judgments")
        if review.supersedes_review_id is not None:
            previous = reviews_by_id.get(review.supersedes_review_id)
            if previous is None or previous.finding_id != review.finding_id:
                raise _ReviewInputError("review supersession target is invalid")
            if previous.review_id in superseded_ids:
                raise _ReviewInputError("review supersession target is already superseded")
            superseded_ids.add(previous.review_id)
        reviews_by_id[review.review_id] = review
        active_by_finding[review.finding_id] = review


def _active_reviews(reviews: list[ReviewRecord]) -> dict[str, ReviewRecord]:
    return {review.finding_id: review for review in reviews}


def _observations_summary(observations: _Observations | None) -> str:
    if observations is None:
        return "not executed"
    groups = (
        ", ".join(
            "+".join(str(repetition) for repetition in group.repetitions)
            for group in observations.outcome_groups
        )
        or "none"
    )
    return (
        f"{observations.stability}; {observations.observed_repetitions}/"
        f"{observations.requested_repetitions} observed; groups={groups}"
    )


def _effects_summary(effects: list[_Effect]) -> str:
    if not effects:
        return "none"
    return "; ".join(
        json.dumps(
            {
                "kind": effect.kind,
                "predicate": effect.predicate,
                "fields": effect.fields,
                "propositions": effect.propositions,
                "status": effect.status,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for effect in effects
    )


def _sensitive_invariant_lines(
    baseline_rule: DatasetInvariantRuleResult,
    variation_rule: DatasetInvariantRuleResult,
) -> Generator[str]:
    for arm_name, rule in (("Variation", variation_rule), ("Original", baseline_rule)):
        if isinstance(
            rule,
            (
                DatasetInvariantArrayUniqueRuleEvaluation,
                DatasetInvariantTransitionRuleEvaluation,
            ),
        ):
            continue
        for trial in rule.trials:
            selected_values: dict[str, JsonValue] = {
                key: value for key, value in trial.resolved_values.items()
            }
            yield _sensitive_json_line(
                f"{arm_name} trial {trial.repetition} selected values",
                selected_values,
            )

    if isinstance(variation_rule, DatasetInvariantValueEqualsRuleEvaluation):
        yield _sensitive_json_line(
            "Configured invariant literal", {"literal": variation_rule.literal}
        )
    elif isinstance(variation_rule, DatasetInvariantValueInSetRuleEvaluation):
        for index, allowed_value in enumerate(variation_rule.allowed_values):
            yield _sensitive_json_line(
                f"Configured allowed value {index + 1}", {"value": allowed_value}
            )
    elif isinstance(variation_rule, DatasetInvariantArrayUniqueRuleEvaluation):
        yield (
            "Selected values unavailable: array uniqueness evidence intentionally retains "
            "indices and pointers only."
        )
    elif isinstance(variation_rule, DatasetInvariantTransitionRuleEvaluation):
        yield (
            "Selected values unavailable: state-transition evidence intentionally retains "
            "counts and pointers only."
        )


def _bounded_sensitive_invariant_lines(
    baseline_rule: DatasetInvariantRuleResult,
    variation_rule: DatasetInvariantRuleResult,
) -> tuple[str, ...]:
    lines: list[str] = []
    encoded_bytes = 0
    for line in _sensitive_invariant_lines(baseline_rule, variation_rule):
        encoded_bytes += len((_sanitize_plain_text(line) + "\n").encode("utf-8"))
        lines.append(line)
        if (
            len(lines) > _MAXIMUM_SENSITIVE_DISCLOSURE_LINES
            or encoded_bytes > _MAXIMUM_SENSITIVE_DISCLOSURE_BYTES
        ):
            raise _ReviewInputError(
                "selected finding values exceed the safe disclosure cap; inspect the private "
                "evidence file under your data-governance controls"
            )
    return tuple(lines)


def _sensitive_json_line(label: str, values: dict[str, JsonValue]) -> str:
    return f"{label}: " + json.dumps(
        values,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _print_invariant_evaluation(evaluation: DatasetInvariantEvaluation) -> None:
    _print_plain(
        "This invariant summary shows pointers and reason codes only. Retained values for "
        "supported rule types require --show-sensitive-values with --finding FINDING_ID; "
        "transition rules do not retain selected state values."
    )
    _print_plain(f"Interaction: {evaluation.interaction_id}")
    _print_plain(f"Declared observation authority: {evaluation.observation_authority}")
    for arm in (evaluation.baseline, *evaluation.variations):
        arm_name = "original" if arm.arm == "baseline" else f"variation ({arm.operator_id})"
        for rule in arm.rules:
            status_counts = {
                status: sum(trial.status == status for trial in rule.trials)
                for status in ("satisfied", "violated", "not_evaluable")
            }
            _print_plain(
                f"Rule {rule.rule_id} ({rule.rule_version}); severity={rule.severity}; "
                f"arm={arm_name}; status={rule.status}; reason={rule.reason_code}; trials="
                + ", ".join(f"{status}={count}" for status, count in status_counts.items())
            )
            _print_plain(f"Description: {rule.description}")
            if rule.status == "violated":
                _print_plain(
                    f"Customer rule violated against declared {evaluation.observation_authority}."
                )
            for trial in rule.trials:
                _print_plain(
                    f"Trial {trial.repetition}: {trial.status}; "
                    f"{_invariant_trial_location(trial)}; "
                    f"reason={trial.reason_code}"
                )


def _invariant_trial_location(
    trial: DatasetInvariantTrialEvaluation
    | DatasetInvariantValueEqualsTrialEvaluation
    | DatasetInvariantValueInSetTrialEvaluation
    | DatasetInvariantArrayUniqueTrialEvaluation
    | DatasetInvariantTransitionTrialEvaluation,
) -> str:
    if isinstance(trial, DatasetInvariantTrialEvaluation):
        return f"left={trial.left_pointer}; right={trial.right_pointer}"
    if isinstance(
        trial,
        (DatasetInvariantValueEqualsTrialEvaluation, DatasetInvariantValueInSetTrialEvaluation),
    ):
        return f"value={trial.value_pointer}"
    if isinstance(trial, DatasetInvariantTransitionTrialEvaluation):
        location = (
            f"before={trial.before_checkpoint}; after={trial.after_checkpoint}; "
            f"value={trial.observation_pointer}"
        )
        if trial.new_effect_count is not None:
            location += f"; new_effects={trial.new_effect_count}"
        return location
    location = (
        f"array={trial.array_pointer}; keys={','.join(trial.key_pointers)}; "
        f"items={trial.item_count}"
    )
    if trial.duplicate_indices:
        location += f"; duplicate_indices={trial.duplicate_indices}"
    if trial.failed_item_index is not None:
        location += (
            f"; failed_item={trial.failed_item_index}; failed_key={trial.failed_key_pointer}"
        )
    return location


def _print_plain(message: str) -> None:
    console.print(_sanitize_plain_text(message), markup=False, highlight=False)


def _print_sensitive_plain(message: str) -> None:
    console.print(
        _sanitize_plain_text(message),
        markup=False,
        highlight=False,
        soft_wrap=True,
    )


def _sanitize_plain_text(message: str) -> str:
    return "".join(
        character
        if (ord(character) >= 32 and not 0x7F <= ord(character) <= 0x9F)
        and unicodedata.category(character) not in {"Cf", "Cs", "Zl", "Zp"}
        else f"\\u{ord(character):04x}"
        for character in message
    )
