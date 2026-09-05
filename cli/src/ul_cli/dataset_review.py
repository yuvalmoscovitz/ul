from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shlex
import stat
import sys
import tempfile
import unicodedata
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, Self, cast
from uuid import uuid4

import typer
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator
from rich.console import Console
from ul import (
    DatasetComparisonCompatibilityError,
    DatasetEvaluationResult,
    DatasetSemanticPreparationError,
    DatasetSourceOutcomeProjectionError,
    InteractionRecord,
    MaterialVarianceAssessment,
    SemanticDeconstructorIdentity,
)
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
from ul.environment import validate_outcome_projection_evidence
from ul.http_environment import JsonHttpIsolatedResponseConfig, JsonHttpTargetConfig
from ul.llm import LLMClientIdentity
from ul.material_variance import response_materiality_action_count
from ul.outcome_projection import OutcomeProjection
from ul_core.augmentations.definitions import builtin_augmentation_catalog

from ul_cli.dataset_run_config import DatasetRunConfig
from ul_cli.invariant_findings import is_reproduced_invariant_difference
from ul_cli.pattern_identity import (
    PatternIdentityKeyError,
    ReviewHistoryKeyError,
    load_pattern_identity_key,
    load_review_history_key,
    pattern_evidence_reference,
    pattern_mechanism_pseudonym,
    pattern_review_record_hmac,
)
from ul_cli.report_contract import (
    FailurePattern,
    FindingCategory,
    FindingCrossExaminationSummary,
    FindingReviewStatus,
    FindingSeverity,
    FindingSummary,
    FindingSummaryText,
    PatternEvidenceAuthority,
    PatternHorizontalFacets,
    PatternMember,
    PatternOperator,
    PatternReviewSummary,
    PatternVerticalFacets,
    ReportInputError,
    ReportReviewStatus,
    UnifiedReport,
    build_failure_pattern,
    build_report_summary,
)

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


def _is_none(value: object) -> bool:
    return value is None


_MAXIMUM_EVIDENCE_BYTES = 128_000_000
_MAXIMUM_EVIDENCE_RECORDS = 100
_MAXIMUM_REVIEWS_BYTES = 10_000_000
_MAXIMUM_REVIEW_RECORDS = 10_000
_MAXIMUM_SENSITIVE_DISCLOSURE_BYTES = 32_768
_MAXIMUM_SENSITIVE_DISCLOSURE_LINES = 50
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_FINDING_ID_PATTERN = r"^ulf_v1_[0-9a-f]{64}$"
_REVIEW_ID_PATTERN = r"^ulr_[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
_PATTERN_REVIEW_ID_PATTERN = (
    r"^ulpr_[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_DATASET_EVALUATION_PIPELINE_VERSION = "1.6.0"
_MAXIMUM_PATTERN_EFFECTS = 100
_MAXIMUM_PATTERN_FIELDS = 100
_MAXIMUM_PATTERN_LABEL_CHARACTERS = 500
_MAXIMUM_PATTERN_IDENTITY_BYTES = 64_000
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


@dataclass(frozen=True)
class _PatternContext:
    private_mechanism_key: str
    source_case_id: str
    operator_id: str
    evidence_authorities: tuple[PatternEvidenceAuthority, ...]
    evidence_record_sha256: str
    vertical_facets: PatternVerticalFacets | None = None

    def __post_init__(self) -> None:
        if len(self.private_mechanism_key) != 64 or any(
            character not in "0123456789abcdef" for character in self.private_mechanism_key
        ):
            raise ValueError("private pattern mechanism key must be a lowercase SHA-256 digest")
        if len(self.evidence_record_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.evidence_record_sha256
        ):
            raise ValueError("pattern evidence digest must be a lowercase SHA-256 digest")


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


def _response_materiality_action_counts(finding: _Finding) -> tuple[int, int] | None:
    if len(finding.reference_effects) != 1 or len(finding.observed_effects) != 1:
        return None
    original_item_count = response_materiality_action_count(finding.reference_effects[0].fields)
    variation_item_count = response_materiality_action_count(finding.observed_effects[0].fields)
    if original_item_count is None or variation_item_count is None:
        return None
    return original_item_count, variation_item_count


def _response_action_count_summary(finding: _Finding) -> str | None:
    counts = _response_materiality_action_counts(finding)
    if counts is None:
        return None
    original_item_count, variation_item_count = counts
    item_count_delta = variation_item_count - original_item_count
    direction = "more" if item_count_delta > 0 else "fewer"
    return (
        f"The agent made {original_item_count} committed actions for the original response and "
        f"{variation_item_count} after the test variation "
        f"({abs(item_count_delta)} {direction})."
    )


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
    llm_client: LLMClientIdentity
    max_input_chars: int = Field(ge=1)
    deconstructor_identity: SemanticDeconstructorIdentity | None = None
    materiality_evaluator_version_id: str | None = Field(
        default=None,
        pattern=r"^ulev_v1_[0-9a-f]{64}$",
    )


class DatasetEvidenceRedactionCoverage(_StrictModel):
    location: Literal["input", "output"]
    matched_values: int = Field(ge=0)
    matched_paths: tuple[str, ...] = ()
    matches_by_rule: dict[str, int] = Field(default_factory=dict)


class DatasetEvidenceTarget(_StrictModel):
    kind: Literal["environment_http", "probe_target"]
    config: JsonHttpTargetConfig | None = cast(Any, Field)(default=None, exclude_if=_is_none)
    receipt: dict[str, JsonValue] | None = cast(Any, Field)(default=None, exclude_if=_is_none)
    sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        if self.kind == "environment_http":
            if self.config is None or self.receipt is not None:
                raise ValueError("HTTP run context target requires only its config snapshot")
            expected_sha256 = dataset_regression_target_config_sha256(self.config)
        else:
            if self.receipt is None or self.config is not None:
                raise ValueError("probe run context target requires only its receipt snapshot")
            expected_sha256 = _canonical_json_sha256(self.receipt)
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
    schema_version: Literal["1.1.0", "1.2.0", "1.3.0", "1.4.0", "1.5.0"] = "1.5.0"
    pipeline_version: Literal["1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.0"] = (
        _DATASET_EVALUATION_PIPELINE_VERSION
    )
    selected_dataset_sha256: str = Field(pattern=_SHA256_PATTERN)
    operators: tuple[DatasetEvidenceOperator, ...] = Field(min_length=1)
    evaluation_mode: Literal["variance"] | None = None
    repetitions: int = Field(ge=1)
    target_timeout_seconds: float = Field(default=30.0, gt=0, le=3_600)
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
            ("1.4.0", "1.5.0"),
            ("1.5.0", "1.6.0"),
        }:
            raise ValueError("run context schema and pipeline versions must match")
        if (
            self.schema_version in {"1.2.0", "1.3.0", "1.4.0", "1.5.0"}
            and self.evaluation_mode is None
        ):
            raise ValueError(
                f"run context schema {self.schema_version} requires an evaluation mode"
            )
        if self.schema_version == "1.1.0" and "evaluation_mode" in self.model_fields_set:
            raise ValueError("run context schema 1.1.0 does not include evaluation mode")
        if self.schema_version in {"1.3.0", "1.4.0", "1.5.0"} and self.fixture is None:
            raise ValueError(
                f"run context schema {self.schema_version} requires fixture identity status"
            )
        if (
            self.schema_version not in {"1.3.0", "1.4.0", "1.5.0"}
            and "fixture" in self.model_fields_set
        ):
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
        if "target_timeout_seconds" not in self.model_fields_set:
            context_content.pop("target_timeout_seconds")
        expected_context_sha256 = _canonical_json_sha256(context_content)
        if self.context_sha256 != expected_context_sha256:
            raise ValueError("run context digest must match its canonical content")
        return self


class DatasetSourcePreparationFailureEvidence(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    record_type: Literal["source_preparation_failure"] = "source_preparation_failure"
    evaluation_mode: Literal["variance"] = "variance"
    interaction_id: str = Field(min_length=1, max_length=500)
    source_record_id: str | None = cast(Any, Field)(
        default=None,
        min_length=1,
        max_length=500,
        exclude_if=_is_none,
    )
    failure_stage: Literal["semantic_preparation"] = "semantic_preparation"
    reason_code: Literal[
        "source_semantic_preparation_failed",
        "source_outcome_projection_failed",
        "source_comparison_surface_incompatible",
    ]
    summary: str = Field(min_length=1, max_length=500)
    remediation: str = Field(min_length=1, max_length=500)
    execution_plan: _ExecutionPlan
    run_context: DatasetEvidenceRunContext

    @model_validator(mode="after")
    def validate_context(self) -> Self:
        if self.run_context.evaluation_mode != self.evaluation_mode:
            raise ValueError("source failure evaluation mode must match its run context")
        if self.execution_plan.repetitions != self.run_context.repetitions:
            raise ValueError("source failure repetitions must match its run context")
        expected_text = {
            "source_semantic_preparation_failed": (
                DatasetSemanticPreparationError.explanation,
                DatasetSemanticPreparationError.remediation,
            ),
            "source_outcome_projection_failed": (
                DatasetSourceOutcomeProjectionError.explanation,
                DatasetSourceOutcomeProjectionError.remediation,
            ),
            "source_comparison_surface_incompatible": (
                DatasetComparisonCompatibilityError.explanation,
                DatasetComparisonCompatibilityError.remediation,
            ),
        }[self.reason_code]
        if (self.summary, self.remediation) != expected_text:
            raise ValueError("source failure guidance must match its reason code")
        return self


class _Baseline(_StrictModel):
    status: str
    observations: _Observations
    inconclusive_reasons: list[str]


class _Case(_StrictModel):
    operator_id: str
    operator_version: str
    source_record_id: str | None = cast(Any, Field)(default=None, exclude_if=_is_none)
    augmentation_target: JsonValue | None = cast(Any, Field)(default=None, exclude_if=_is_none)
    original_value: str | None = cast(Any, Field)(default=None, exclude_if=_is_none)
    augmented_input: str
    status: str
    variation_accepted: bool
    variation_rejection_reasons: list[str]
    observations: _Observations | None
    findings: list[_Finding]
    material_variance: MaterialVarianceAssessment | None = None
    cross_examination: FindingCrossExaminationSummary | None = None
    inconclusive_reasons: list[str]


class _EvidenceRecord(_StrictModel):
    schema_version: Literal[
        "1.3.0",
        "1.4.0",
        "1.5.0",
        "1.6.0",
        "1.7.0",
        "1.8.0",
        "1.9.0",
        "1.10.0",
        "1.11.0",
        "1.12.0",
        "1.13.0",
        "1.14.0",
        "1.15.0",
    ]
    evaluation_mode: Literal["variance"] | None = None
    interaction_id: str
    source_record_id: str | None = cast(Any, Field)(default=None, exclude_if=_is_none)
    pattern_facets: PatternVerticalFacets | None = cast(Any, Field)(
        default=None, exclude_if=_is_none
    )
    augmentation_target: JsonValue | None = cast(Any, Field)(default=None, exclude_if=_is_none)
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
        current_schemas = {
            "1.8.0",
            "1.9.0",
            "1.10.0",
            "1.11.0",
            "1.12.0",
            "1.13.0",
            "1.14.0",
            "1.15.0",
        }
        if self.schema_version not in {
            "1.10.0",
            "1.11.0",
            "1.12.0",
            "1.13.0",
            "1.14.0",
            "1.15.0",
        } and "pattern_facets" in (self.model_fields_set):
            raise ValueError("vertical pattern facets require evidence schema 1.10.0")
        if self.schema_version in {"1.12.0", "1.13.0", "1.14.0", "1.15.0"}:
            if any(case.cross_examination is None for case in self.cases):
                raise ValueError("current evidence schemas require case cross-examination")
        elif any("cross_examination" in case.model_fields_set for case in self.cases):
            raise ValueError("legacy evidence cannot contain cross-examination")
        if self.schema_version in current_schemas and self.evaluation_mode is None:
            raise ValueError(f"evidence schema {self.schema_version} requires an evaluation mode")
        if self.schema_version in current_schemas and (
            not isinstance(self.technical_details, dict)
            or self.technical_details.get("evaluation_mode") != self.evaluation_mode
        ):
            raise ValueError("evidence evaluation mode must match technical details")
        if (
            self.schema_version not in current_schemas
            and "evaluation_mode" in self.model_fields_set
        ):
            raise ValueError("legacy evidence does not include evaluation mode")
        if self.schema_version == "1.3.0" and "invariant_evaluation" in self.model_fields_set:
            raise ValueError("schema 1.3.0 does not include invariant evaluation")
        if self.schema_version in {"1.5.0", "1.6.0", "1.7.0"} and self.run_context is None:
            raise ValueError(f"schema {self.schema_version} requires run context")
        if (
            self.schema_version
            not in {
                "1.5.0",
                "1.6.0",
                "1.7.0",
                "1.8.0",
                "1.9.0",
                "1.10.0",
                "1.11.0",
                "1.12.0",
                "1.13.0",
                "1.14.0",
                "1.15.0",
            }
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
        if uses_extended_invariants and self.schema_version not in {
            "1.6.0",
            "1.7.0",
            "1.8.0",
            "1.9.0",
            "1.10.0",
            "1.11.0",
            "1.12.0",
            "1.13.0",
            "1.14.0",
            "1.15.0",
        }:
            raise ValueError("extended invariant results require evidence schema 1.6.0")
        if (
            self.run_context is not None
            and self.evaluation_mode is not None
            and self.run_context.evaluation_mode != self.evaluation_mode
        ):
            raise ValueError("evidence evaluation mode must match its run context")
        has_response_findings = any(
            finding.category == "changed_response"
            for case in self.cases
            for finding in case.findings
        )
        technical_comparison_surface = (
            self.technical_details.get("comparison_surface")
            if isinstance(self.technical_details, dict)
            else None
        )
        if self.schema_version not in {"1.14.0", "1.15.0"} and (
            has_response_findings or technical_comparison_surface == "response"
        ):
            raise ValueError("response comparison evidence requires schema 1.14.0")
        if has_response_findings and technical_comparison_surface != "response":
            raise ValueError("response findings require response comparison technical evidence")
        if self.schema_version == "1.15.0":
            if any((case.material_variance is None) != (not case.findings) for case in self.cases):
                raise ValueError("schema 1.15.0 requires materiality for every semantic finding")
            if self.run_context is not None:
                expected_materiality_version = (
                    self.run_context.semantic_settings.materiality_evaluator_version_id
                )
                if expected_materiality_version is None or any(
                    case.material_variance is not None
                    and case.material_variance.evaluator_version_id != expected_materiality_version
                    for case in self.cases
                ):
                    raise ValueError(
                        "materiality assessment version must match the evidence run context"
                    )
        elif any("material_variance" in case.model_fields_set for case in self.cases):
            raise ValueError("legacy evidence cannot contain material variance assessments")
        if self.schema_version in {"1.13.0", "1.14.0", "1.15.0"}:
            from ul_cli.dataset.evidence.customer import build_customer_evidence_record

            technical_result = DatasetEvaluationResult.model_validate(
                self.technical_details,
                strict=False,
            )
            expected_record = build_customer_evidence_record(
                technical_result,
                repetitions=self.execution_plan.repetitions,
                max_environment_api_calls=self.execution_plan.max_target_calls,
                planned_target_calls=self.execution_plan.dataset_planned_target_calls,
            )
            expected_cases = cast(list[dict[str, JsonValue]], expected_record["cases"])
            if len(expected_cases) != len(self.cases) or any(
                case.cross_examination is None
                or case.cross_examination.model_dump(mode="json")
                != expected_case["cross_examination"]
                or (
                    self.schema_version == "1.15.0"
                    and (
                        case.material_variance.model_dump(mode="json")
                        if case.material_variance is not None
                        else None
                    )
                    != expected_case.get("material_variance")
                )
                for case, expected_case in zip(self.cases, expected_cases, strict=True)
            ):
                raise ValueError(
                    "cross-examination conclusions must match technical execution evidence"
                )
        if (
            self.invariant_evaluation is not None
            and self.invariant_evaluation.interaction_id != self.interaction_id
        ):
            raise ValueError("invariant evaluation must match the evidence interaction")
        return self


@dataclass(frozen=True)
class DatasetResumeEvidence:
    processed_ids: frozenset[str]
    source_preparation_failures: tuple[DatasetSourcePreparationFailureEvidence, ...]
    has_review_findings: bool
    has_inconclusive_materiality: bool
    invariant_evaluations: tuple[DatasetInvariantEvaluation, ...]
    technical_results: tuple[DatasetEvaluationResult, ...]
    raw_evidence_sha256: str


def create_dataset_evidence_run_context(
    *,
    selected_records: tuple[InteractionRecord, ...],
    operators: tuple[tuple[str, str], ...],
    run_config: DatasetRunConfig,
    invariant_suite_sha256: str | None,
    target_config: JsonHttpTargetConfig | None = None,
    target_receipt: dict[str, JsonValue] | None = None,
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
    if (target_config is None) == (target_receipt is None):
        raise ValueError("run context requires exactly one target snapshot or receipt")
    target = (
        DatasetEvidenceTarget(
            kind="environment_http",
            config=target_config,
            sha256=dataset_regression_target_config_sha256(target_config),
        )
        if target_config is not None
        else DatasetEvidenceTarget(
            kind="probe_target",
            receipt=target_receipt,
            sha256=_canonical_json_sha256(target_receipt),
        )
    )
    fixture = (
        _dataset_evidence_fixture(target_config)
        if target_config is not None
        else DatasetEvidenceFixture(status="not_required")
    )
    content = {
        "schema_version": "1.5.0",
        "pipeline_version": _DATASET_EVALUATION_PIPELINE_VERSION,
        "selected_dataset_sha256": selected_dataset_sha256,
        "operators": [operator.model_dump(mode="json") for operator in operator_snapshots],
        "evaluation_mode": run_config.evaluation_mode,
        "repetitions": run_config.repetitions,
        "target_timeout_seconds": run_config.target.trial_timeout_seconds,
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
        evaluation_mode=run_config.evaluation_mode,
        repetitions=run_config.repetitions,
        target_timeout_seconds=run_config.target.trial_timeout_seconds,
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
    if raw_lines and dataset_durable_run_marker_manifest_sha256(raw_lines[0]) is not None:
        raw_lines = raw_lines[1:]
    if not raw_lines:
        return DatasetResumeEvidence(
            processed_ids=frozenset(),
            source_preparation_failures=(),
            has_review_findings=False,
            has_inconclusive_materiality=False,
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
    has_inconclusive_materiality = False
    invariant_evaluations: list[DatasetInvariantEvaluation] = []
    technical_results: list[DatasetEvaluationResult] = []
    source_preparation_failures: list[DatasetSourcePreparationFailureEvidence] = []
    for raw_line in raw_lines:
        try:
            decoded_line: object = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("resume evidence is not valid UL JSONL") from None
        decoded_record = (
            cast(dict[str, object], decoded_line) if isinstance(decoded_line, dict) else None
        )
        if (
            decoded_record is not None
            and decoded_record.get("record_type") == "source_preparation_failure"
        ):
            try:
                source_failure = DatasetSourcePreparationFailureEvidence.model_validate_json(
                    raw_line
                )
            except (ValidationError, ValueError):
                raise ValueError("resume evidence is not valid UL JSONL") from None
            if source_failure.run_context != expected_context:
                raise ValueError("resume evidence is incompatible with the current evaluation plan")
            if source_failure.interaction_id in processed_ids:
                raise ValueError("resume evidence contains duplicate interaction IDs")
            selected_record = selected_records_by_id.get(source_failure.interaction_id)
            if selected_record is None:
                raise ValueError(
                    "resume evidence contains an interaction outside the selected dataset"
                )
            expected_source_record_id = (
                selected_record.source_interaction_id
                if getattr(selected_record, "augmentation_target", None) is not None
                else None
            )
            if source_failure.source_record_id != expected_source_record_id:
                raise ValueError("resume source failure does not match the selected dataset")
            processed_ids.add(source_failure.interaction_id)
            source_preparation_failures.append(source_failure)
            continue
        try:
            evidence = _EvidenceRecord.model_validate_json(raw_line)
        except (ValidationError, ValueError):
            raise ValueError("resume evidence is not valid UL JSONL") from None
        if (
            evidence.schema_version
            not in {
                "1.5.0",
                "1.6.0",
                "1.7.0",
                "1.8.0",
                "1.9.0",
                "1.10.0",
                "1.11.0",
                "1.12.0",
                "1.13.0",
                "1.14.0",
                "1.15.0",
            }
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
        _validate_resumed_outcome_projections(technical_result, expected_context.target)
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
        for case in technical_result.cases:
            if case.verdict != "divergence_needs_review":
                continue
            if (
                case.material_variance is None
                or case.material_variance.decision == "material_variance"
            ):
                has_review_findings = True
            elif case.material_variance.decision == "insufficient_evidence":
                has_inconclusive_materiality = True
        if expected_invariant_evaluation is not None:
            invariant_evaluations.append(expected_invariant_evaluation)
        processed_ids.add(evidence.interaction_id)
    return DatasetResumeEvidence(
        processed_ids=frozenset(processed_ids),
        source_preparation_failures=tuple(source_preparation_failures),
        has_review_findings=has_review_findings,
        has_inconclusive_materiality=has_inconclusive_materiality,
        invariant_evaluations=tuple(invariant_evaluations),
        technical_results=tuple(technical_results),
        raw_evidence_sha256=hashlib.sha256(raw_evidence).hexdigest(),
    )


def _validate_resumed_outcome_projections(
    result: DatasetEvaluationResult,
    target: DatasetEvidenceTarget,
) -> None:
    if target.kind == "environment_http":
        if target.config is None:
            raise AssertionError("validated HTTP target requires its config")
        projection = target.config.outcome
    else:
        if target.receipt is None:
            raise AssertionError("validated probe target requires its receipt")
        raw_projection = target.receipt.get("outcome_projection")
        projection = (
            OutcomeProjection.model_validate_json(json.dumps(raw_projection))
            if raw_projection is not None
            else None
        )
    trial_sets = [result.baseline.trial_set]
    trial_sets.extend(case.trial_set for case in result.cases if case.trial_set is not None)
    for trial_set in trial_sets:
        for trial in trial_set.trials:
            if trial.execution_evidence is not None:
                validate_outcome_projection_evidence(
                    trial.execution_evidence,
                    projection,
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
    record_type: Literal["occurrence_review"] = Field(default="occurrence_review", exclude=True)
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
    origin_pattern_review_id: str | None = cast(Any, Field)(
        default=None,
        pattern=_PATTERN_REVIEW_ID_PATTERN,
        exclude_if=_is_none,
    )

    @model_validator(mode="after")
    def validate_review(self) -> Self:
        if not self.reviewer.strip() or not self.reason.strip():
            raise ValueError("reviewer and reason must contain non-whitespace text")
        if self.status != "confirmed" and self.severity != "unrated":
            raise ValueError("only confirmed findings can have a rated severity")
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() != UTC.utcoffset(None):
            raise ValueError("reviewed_at must use UTC")
        return self


class PatternOccurrenceDecision(_StrictModel):
    review_id: str = Field(pattern=_REVIEW_ID_PATTERN)
    finding_id: str = Field(pattern=_FINDING_ID_PATTERN)
    evidence_record_sha256: str = Field(pattern=_SHA256_PATTERN)


class PatternEvidenceBinding(_StrictModel):
    finding_id: str = Field(pattern=_FINDING_ID_PATTERN)
    evidence_record_sha256: str = Field(pattern=_SHA256_PATTERN)


class PatternReviewRecord(_StrictModel):
    record_type: Literal["pattern_review"] = "pattern_review"
    schema_version: Literal["1.1.0"] = "1.1.0"
    pattern_review_id: str = Field(pattern=_PATTERN_REVIEW_ID_PATTERN)
    record_hmac_sha256: str = Field(pattern=_SHA256_PATTERN)
    grouping_evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    pattern_snapshot: FailurePattern
    status: ReviewStatus
    severity: ReviewSeverity = "unrated"
    reviewed_finding_ids: tuple[str, ...] = Field(min_length=1, max_length=10_000)
    exception_finding_ids: tuple[str, ...] = Field(default=(), max_length=10_000)
    evidence_record_sha256s: tuple[str, ...] = Field(min_length=1, max_length=100)
    evidence_bindings: tuple[PatternEvidenceBinding, ...] = Field(min_length=1, max_length=10_000)
    occurrence_decisions: tuple[PatternOccurrenceDecision, ...] = Field(
        min_length=1, max_length=10_000
    )
    reviewer: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=4000)
    reviewed_at: datetime

    @model_validator(mode="after")
    def validate_review(self) -> Self:
        if not self.reviewer.strip() or not self.reason.strip():
            raise ValueError("reviewer and reason must contain non-whitespace text")
        if self.status != "confirmed" and self.severity != "unrated":
            raise ValueError("only confirmed patterns can have a rated severity")
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() != UTC.utcoffset(None):
            raise ValueError("reviewed_at must use UTC")
        for values in (self.reviewed_finding_ids, self.exception_finding_ids):
            if values != tuple(sorted(set(values))):
                raise ValueError("pattern review finding IDs must be sorted and unique")
        reviewed = set(self.reviewed_finding_ids)
        exceptions = set(self.exception_finding_ids)
        if reviewed & exceptions:
            raise ValueError("reviewed and exception findings must be disjoint")
        members = {member.finding_id: member for member in self.pattern_snapshot.members}
        if reviewed != set(members) or any(
            member.review_status != "needs_review" for member in members.values()
        ):
            raise ValueError("pattern review can decide only an exact needs-review cohort")
        decision_ids = tuple(decision.finding_id for decision in self.occurrence_decisions)
        review_ids = tuple(decision.review_id for decision in self.occurrence_decisions)
        if decision_ids != self.reviewed_finding_ids:
            raise ValueError("pattern occurrence decisions must match reviewed findings")
        if len(review_ids) != len(set(review_ids)):
            raise ValueError("pattern occurrence decision IDs must be unique")
        binding_ids = tuple(binding.finding_id for binding in self.evidence_bindings)
        if any(
            re.fullmatch(_SHA256_PATTERN, value) is None for value in self.evidence_record_sha256s
        ):
            raise ValueError("pattern evidence manifest must contain SHA-256 digests")
        if binding_ids != tuple(sorted(set(binding_ids))):
            raise ValueError("pattern evidence bindings must be sorted and unique")
        if not (reviewed | exceptions <= set(binding_ids)):
            raise ValueError("pattern members and exceptions require evidence bindings")
        return self

    def derived_reviews(self) -> tuple[ReviewRecord, ...]:
        return tuple(
            ReviewRecord(
                review_id=decision.review_id,
                evidence_record_sha256=decision.evidence_record_sha256,
                finding_id=decision.finding_id,
                status=self.status,
                severity=self.severity,
                reviewer=self.reviewer,
                reason=self.reason,
                reviewed_at=self.reviewed_at,
                origin_pattern_review_id=self.pattern_review_id,
            )
            for decision in self.occurrence_decisions
        )

    def hmac_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"record_hmac_sha256"})

    def summary(self) -> PatternReviewSummary:
        return PatternReviewSummary(
            pattern_review_id=self.pattern_review_id,
            pattern_fingerprint=self.pattern_snapshot.pattern_fingerprint,
            pattern_snapshot_id=self.pattern_snapshot.pattern_snapshot_id,
            status=self.status,
            severity=self.severity,
            reviewed_finding_ids=self.reviewed_finding_ids,
            exception_finding_ids=self.exception_finding_ids,
            reviewed_at=self.reviewed_at,
        )


ReviewHistoryRecord = ReviewRecord | PatternReviewRecord


class _ReviewInputError(ValueError):
    pass


class _ReviewCommitUncertainError(OSError):
    pass


class _LoadedEvidenceRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    evidence: _EvidenceRecord
    sha256: str


@dataclass(frozen=True)
class _LoadedEvidenceDocument:
    records: list[_LoadedEvidenceRecord]
    source_failures: tuple[DatasetSourcePreparationFailureEvidence, ...]


@dataclass
class _LockedReviewsFile:
    path: Path
    descriptor: int


@dataclass(frozen=True)
class _IndexedFinding:
    finding_id: str
    kind: Literal["semantic_difference", "customer_invariant_violation"]
    evidence_record: _LoadedEvidenceRecord
    case: _Case
    semantic_finding: _Finding | None = None
    baseline_rule: DatasetInvariantRuleResult | None = None
    variation_rule: DatasetInvariantRuleResult | None = None


def _effective_finding_status(
    indexed_finding: _IndexedFinding,
    active_review: ReviewRecord | None,
) -> FindingReviewStatus:
    if active_review is not None:
        return active_review.status
    if indexed_finding.kind == "customer_invariant_violation":
        return "needs_review"
    automatic_materiality = indexed_finding.case.material_variance
    if automatic_materiality is None or automatic_materiality.decision == "material_variance":
        return "needs_review"
    if automatic_materiality.decision == "operationally_equivalent":
        return "expected"
    return "inconclusive"


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
    _validate_review_history(
        review_records,
        findings,
        evidence_records,
        review_history_key=_review_history_key_for_records(evidence_path, review_records),
    )
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
    "changed_response": "The changed input changed the agent's observed response.",
    "unstable_behavior": "The changed input produced inconsistent behavior across repetitions.",
}


def summarize_dataset_evidence(
    evidence: Path,
    reviews: Path | None = None,
    *,
    pattern_identity_key: bytes,
) -> UnifiedReport:
    try:
        return _summarize_dataset_evidence(
            evidence,
            reviews,
            pattern_identity_key=pattern_identity_key,
        )
    except (PatternIdentityKeyError, ReviewHistoryKeyError, _ReviewInputError) as error:
        raise ReportInputError(str(error)) from None
    except (ValidationError, ValueError):
        raise ReportInputError("dataset evidence cannot be summarized safely") from None


def _summarize_dataset_evidence(
    evidence: Path,
    reviews: Path | None,
    *,
    pattern_identity_key: bytes,
) -> UnifiedReport:
    evidence_document = _load_evidence_document(evidence)
    evidence_records = evidence_document.records
    source_failures = evidence_document.source_failures
    evaluation_mode = _dataset_evaluation_mode(evidence_records, source_failures)
    review_records = _load_reviews(reviews or _default_reviews_path(evidence))
    indexed_findings = _index_findings(evidence_records)
    _validate_review_history(
        review_records,
        indexed_findings,
        evidence_records,
        review_history_key=_review_history_key_for_records(evidence, review_records),
    )
    active_reviews = _active_reviews(review_records)

    finding_summaries: list[FindingSummary] = []
    pattern_contexts: dict[str, _PatternContext] = {}
    indexed_invariant_keys: set[tuple[str, str, str, str]] = set()
    for indexed_finding in indexed_findings.values():
        active_review = active_reviews.get(indexed_finding.finding_id)
        review_status = _effective_finding_status(indexed_finding, active_review)
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
                    evidence_authorities=("model_derived_unverified",),
                    evidence_limitations=("semantic_model_output_not_independently_verified",),
                    next_action=next_action,
                    summary=summary,
                    cross_examination=indexed_finding.case.cross_examination,
                )
            )
            semantic_finding = indexed_finding.semantic_finding
            pattern_signature = _behavior_pattern_signature(semantic_finding)
            if pattern_signature is not None:
                pattern_contexts[indexed_finding.finding_id] = _PatternContext(
                    private_mechanism_key=pattern_signature,
                    source_case_id=indexed_finding.evidence_record.evidence.interaction_id,
                    operator_id=indexed_finding.case.operator_id,
                    evidence_authorities=("model_derived_unverified",),
                    evidence_record_sha256=indexed_finding.evidence_record.sha256,
                    vertical_facets=indexed_finding.evidence_record.evidence.pattern_facets,
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
                requested_repetitions=len(variation_rule.trials),
                conclusive_repetitions=len(variation_rule.trials),
                inconclusive_repetitions=0,
                stability="stable",
                evidence_authorities=(
                    "customer_declared",
                    "deterministic_evaluator",
                ),
                violated_repetitions=sum(
                    trial.status == "violated" for trial in variation_rule.trials
                ),
                next_action=next_action,
                summary="The agent violated a customer-defined rule.",
            )
        )
        pattern_signature = _canonical_json_sha256(
            {
                "kind": "customer_invariant_violation",
                "rule_id": variation_rule.rule_id,
                "rule_version": variation_rule.rule_version,
            }
        )
        pattern_contexts[indexed_finding.finding_id] = _PatternContext(
            private_mechanism_key=pattern_signature,
            source_case_id=indexed_finding.evidence_record.evidence.interaction_id,
            operator_id=indexed_finding.case.operator_id,
            evidence_authorities=("customer_declared", "deterministic_evaluator"),
            evidence_record_sha256=indexed_finding.evidence_record.sha256,
            vertical_facets=indexed_finding.evidence_record.evidence.pattern_facets,
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
                        cross_examination=case.cross_examination,
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
                conclusive_repetitions = sum(
                    trial.status != "not_evaluable" for trial in rule.trials
                )
                inconclusive_repetitions = len(rule.trials) - conclusive_repetitions
                conclusive_statuses = {
                    trial.status for trial in rule.trials if trial.status != "not_evaluable"
                }
                rule_stability = (
                    "stable"
                    if not inconclusive_repetitions and len(conclusive_statuses) == 1
                    else "unstable"
                    if len(conclusive_statuses) > 1
                    else "inconclusive"
                )
                non_promoted_inconclusive = operator_id is not None or rule_stability != "stable"
                finding_summaries.append(
                    FindingSummary(
                        kind="customer_invariant_violation",
                        category="customer_invariant_violation",
                        operator_id=operator_id,
                        operator_version=operator_version,
                        rule_id=rule.rule_id,
                        rule_version=rule.rule_version,
                        declared_severity=rule.severity,
                        review_status=("inconclusive" if non_promoted_inconclusive else None),
                        review_severity=("unrated" if non_promoted_inconclusive else None),
                        requested_repetitions=len(rule.trials),
                        conclusive_repetitions=conclusive_repetitions,
                        inconclusive_repetitions=inconclusive_repetitions,
                        stability=rule_stability,
                        violated_repetitions=sum(
                            trial.status == "violated" for trial in rule.trials
                        ),
                        next_action="inspect_dataset_evidence",
                        summary="The agent violated a customer-defined rule.",
                    )
                )

    findings = tuple(finding_summaries)
    patterns = _build_failure_patterns(
        findings,
        pattern_contexts,
        pattern_identity_key=pattern_identity_key,
    )
    summary = build_report_summary(findings)
    if summary.actionable_finding_count:
        report_review_status: ReportReviewStatus = "action_required"
    elif (
        source_failures
        or summary.review_status_counts.inconclusive
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
    targets = tuple(
        loaded_record.evidence.run_context.target
        for loaded_record in evidence_records
        if loaded_record.evidence.run_context is not None
    ) + tuple(source_failure.run_context.target for source_failure in source_failures)
    response_only_targets = tuple(
        (
            target.kind == "probe_target"
            and target.receipt is not None
            and target.receipt.get("supports_state_observation") is False
        )
        or isinstance(target.config, JsonHttpIsolatedResponseConfig)
        for target in targets
    )
    if any(response_only_targets) and not all(response_only_targets):
        raise _ReviewInputError("evidence combines incompatible target capability tiers")
    response_only = bool(targets) and all(response_only_targets)
    return UnifiedReport(
        evidence_type="dataset_evaluation",
        evidence_schema_versions=tuple(
            sorted(
                {record.evidence.schema_version for record in evidence_records}
                | ({"source-preparation-failure/1.0.0"} if source_failures else set())
            )
        ),
        response_state_evidence_scope=("response_only" if response_only else "response_and_state"),
        evaluation_mode=evaluation_mode,
        capability_limitations=(
            ("cleanup_verification", "conversation_replay", "state_observation")
            if response_only
            else ()
        ),
        review_status=report_review_status,
        exit_code=exit_code,
        summary=summary,
        stable_pattern_count=len({pattern.pattern_fingerprint for pattern in patterns}),
        patterns=patterns,
        pattern_reviews=tuple(
            sorted(
                (
                    review.summary()
                    for review in review_records
                    if isinstance(review, PatternReviewRecord)
                ),
                key=lambda review: (review.reviewed_at, review.pattern_review_id),
            )
        ),
        findings=findings,
    )


def _dataset_evaluation_mode(
    records: list[_LoadedEvidenceRecord],
    source_failures: tuple[DatasetSourcePreparationFailureEvidence, ...] = (),
) -> Literal["variance"] | None:
    evaluation_modes: set[Literal["variance"] | None] = {
        record.evidence.evaluation_mode for record in records
    }
    evaluation_modes.update(failure.evaluation_mode for failure in source_failures)
    if not evaluation_modes:
        return None
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
    identity = {
        "kind": "behavior_difference",
        "category": finding.category,
        "grounded_field_names": sorted(
            unicodedata.normalize("NFC", field_name) for field_name in finding.grounded_field_names
        ),
        "reference_effects": reference_effects,
        "observed_effects": observed_effects,
    }
    canonical_identity = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(canonical_identity) > _MAXIMUM_PATTERN_IDENTITY_BYTES:
        return None
    return hashlib.sha256(canonical_identity).hexdigest()


def _indexed_pattern_grouping_descriptor(finding: _IndexedFinding) -> dict[str, object]:
    observations = finding.case.observations
    if observations is None:
        raise _ReviewInputError("pattern finding is missing variation observations")
    if finding.semantic_finding is not None:
        mechanism_key = _behavior_pattern_signature(finding.semantic_finding)
        if mechanism_key is None:
            raise _ReviewInputError("pattern finding mechanism exceeds grouping limits")
        category = finding.semantic_finding.category
        return {
            "private_mechanism_key": mechanism_key,
            "kind": "behavior_difference",
            "category": category,
            "rule_id": None,
            "rule_version": None,
            "summary": _BEHAVIOR_FINDING_SUMMARIES[cast(FindingCategory, category)],
            "declared_severity": None,
            "stability": observations.stability,
            "evidence_authorities": ("model_derived_unverified",),
            "evidence_limitations": ("semantic_model_output_not_independently_verified",),
            "vertical_facets": (
                finding.evidence_record.evidence.pattern_facets.model_dump(
                    mode="json", exclude_none=True
                )
                if finding.evidence_record.evidence.pattern_facets is not None
                else None
            ),
        }
    variation_rule = finding.variation_rule
    if variation_rule is None:
        raise _ReviewInputError("pattern invariant finding is missing its rule result")
    mechanism_key = _canonical_json_sha256(
        {
            "kind": "customer_invariant_violation",
            "rule_id": variation_rule.rule_id,
            "rule_version": variation_rule.rule_version,
        }
    )
    return {
        "private_mechanism_key": mechanism_key,
        "kind": "customer_invariant_violation",
        "category": "customer_invariant_violation",
        "rule_id": variation_rule.rule_id,
        "rule_version": variation_rule.rule_version,
        "summary": "The agent violated a customer-defined rule.",
        "declared_severity": variation_rule.severity,
        "stability": observations.stability,
        "evidence_authorities": ("customer_declared", "deterministic_evaluator"),
        "evidence_limitations": (),
        "vertical_facets": (
            finding.evidence_record.evidence.pattern_facets.model_dump(
                mode="json", exclude_none=True
            )
            if finding.evidence_record.evidence.pattern_facets is not None
            else None
        ),
    }


def _indexed_pattern_grouping_sha256(finding: _IndexedFinding) -> str:
    return _canonical_json_sha256(_indexed_pattern_grouping_descriptor(finding))


def _optional_indexed_pattern_grouping_sha256(finding: _IndexedFinding) -> str | None:
    try:
        return _indexed_pattern_grouping_sha256(finding)
    except _ReviewInputError:
        return None


def _bounded_effect_mechanisms(effects: list[_Effect]) -> list[dict[str, object]] | None:
    if len(effects) > _MAXIMUM_PATTERN_EFFECTS:
        return None
    if any(
        len(effect.fields) > _MAXIMUM_PATTERN_FIELDS
        or len(effect.kind) > _MAXIMUM_PATTERN_LABEL_CHARACTERS
        or len(effect.predicate) > _MAXIMUM_PATTERN_LABEL_CHARACTERS
        or len(effect.status) > _MAXIMUM_PATTERN_LABEL_CHARACTERS
        or any(len(field_name) > _MAXIMUM_PATTERN_LABEL_CHARACTERS for field_name in effect.fields)
        for effect in effects
    ):
        return None
    return sorted(
        (
            {
                "kind": unicodedata.normalize("NFC", effect.kind),
                "predicate": unicodedata.normalize("NFC", effect.predicate),
                "status": unicodedata.normalize("NFC", effect.status),
                "field_names": sorted(
                    unicodedata.normalize("NFC", field_name) for field_name in effect.fields
                ),
            }
            for effect in effects
        ),
        key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")),
    )


def _build_failure_patterns(
    findings: tuple[FindingSummary, ...],
    contexts: dict[str, _PatternContext],
    *,
    pattern_identity_key: bytes,
) -> tuple[FailurePattern, ...]:
    findings_by_id = {
        finding.finding_id: finding for finding in findings if finding.finding_id is not None
    }
    grouped: dict[str, list[tuple[FindingSummary, _PatternContext]]] = {}
    for finding_id, context in contexts.items():
        finding = findings_by_id[finding_id]
        grouping_key = _canonical_json_sha256(
            {
                "private_mechanism_key": context.private_mechanism_key,
                "kind": finding.kind,
                "category": finding.category,
                "rule_id": finding.rule_id,
                "rule_version": finding.rule_version,
                "stability": finding.stability,
                "evidence_authorities": context.evidence_authorities,
                "evidence_limitations": finding.evidence_limitations,
                "review_status": finding.review_status,
                "review_severity": finding.review_severity,
                "vertical_facets": (
                    context.vertical_facets.model_dump(mode="json", exclude_none=True)
                    if context.vertical_facets is not None
                    else None
                ),
            }
        )
        grouped.setdefault(grouping_key, []).append((finding, context))

    catalog = builtin_augmentation_catalog()
    patterns: list[FailurePattern] = []
    for members in grouped.values():
        first = members[0][0]
        first_context = members[0][1]
        if first.stability is None or first.review_status is None or first.review_severity is None:
            raise AssertionError("pattern members require occurrence review state")
        operator_keys = sorted(
            {
                (context.operator_id, finding.operator_version)
                for finding, context in members
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
        member_severities: list[FindingSeverity] = []
        for finding, _ in members:
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
        membership_reasons = tuple(
            sorted(
                (
                    "same_finding_kind",
                    "same_finding_category",
                    (
                        "same_customer_rule"
                        if first.kind == "customer_invariant_violation"
                        else "same_response_shape"
                        if first.category == "changed_response"
                        else "same_action_shape"
                    ),
                    "same_outcome_stability",
                    "same_evidence_authority",
                    "same_evidence_limitation",
                )
            )
        )
        patterns.append(
            build_failure_pattern(
                kind=first.kind,
                category=first.category,
                rule_id=first.rule_id,
                rule_version=first.rule_version,
                summary=first.summary,
                severity=pattern_severity,
                stability=first.stability,
                evidence_authorities=first_context.evidence_authorities,
                evidence_limitations=first.evidence_limitations,
                horizontal_facets=PatternHorizontalFacets(
                    failure_type=first.category,
                    affected_subject=(
                        "rule"
                        if first.kind == "customer_invariant_violation"
                        else "outcome"
                        if first.category in {"unstable_behavior", "changed_response"}
                        else "action"
                    ),
                    evidence_level=(
                        "evaluated_rule"
                        if first.kind == "customer_invariant_violation"
                        else "model_derived_outcome"
                        if first.category in {"unstable_behavior", "changed_response"}
                        and "model_derived_unverified" in first.evidence_authorities
                        else "model_derived_action"
                        if "model_derived_unverified" in first.evidence_authorities
                        else "observed_outcome"
                        if first.category in {"unstable_behavior", "changed_response"}
                        else "observed_action"
                    ),
                    mechanism_pseudonym=pattern_mechanism_pseudonym(
                        pattern_identity_key,
                        first_context.private_mechanism_key,
                    ),
                ),
                vertical_facets=first_context.vertical_facets,
                finding_count=len(members),
                source_case_count=len({context.source_case_id for _, context in members}),
                operators=tuple(operators),
                needs_review_count=sum(
                    finding.review_status == "needs_review" for finding, _ in members
                ),
                confirmed_count=sum(finding.review_status == "confirmed" for finding, _ in members),
                expected_count=sum(finding.review_status == "expected" for finding, _ in members),
                unsupported_count=sum(
                    finding.review_status == "unsupported" for finding, _ in members
                ),
                inconclusive_count=sum(
                    finding.review_status == "inconclusive" for finding, _ in members
                ),
                members=tuple(
                    PatternMember(
                        finding_id=finding_id,
                        evidence_record_ref=pattern_evidence_reference(
                            pattern_identity_key, context.evidence_record_sha256
                        ),
                        membership_reasons=membership_reasons,
                        review_status=cast(FindingReviewStatus, finding.review_status),
                        review_severity=cast(FindingSeverity, finding.review_severity),
                    )
                    for finding_id, finding, context in sorted(
                        (
                            (cast(str, finding.finding_id), finding, context)
                            for finding, context in members
                        )
                    )
                ),
            )
        )
    return tuple(
        sorted(
            patterns,
            key=lambda pattern: (
                -_PATTERN_SEVERITY_RANK[pattern.severity],
                pattern.pattern_snapshot_id,
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


def _dataset_case_report_bucket(
    case: _Case,
) -> Literal[
    "consequential",
    "equivalent",
    "inconclusive",
    "unstable",
    "unclassified",
    "no_difference",
    "not_evaluated",
]:
    if not case.variation_accepted:
        return "not_evaluated"
    if case.material_variance is not None:
        if case.material_variance.decision == "material_variance":
            return "consequential"
        if case.material_variance.decision == "operationally_equivalent":
            return "equivalent"
        return "inconclusive"
    if case.observations is not None and case.observations.stability == "unstable":
        return "unstable"
    if case.inconclusive_reasons or case.observations is None:
        return "inconclusive"
    if case.findings:
        return "unclassified"
    return "no_difference"


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
                "Show capped private values for one finding. Values may contain "
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
    all_findings: Annotated[
        bool,
        typer.Option(
            "--all-findings",
            help="Also list automatically equivalent or human-resolved findings.",
        ),
    ] = False,
) -> None:
    """Show findings and their human review state without model or network calls."""
    reviews_path = reviews or _default_reviews_path(evidence)
    try:
        evidence_document = _load_evidence_document(evidence)
        evidence_records = evidence_document.records
        source_failures = evidence_document.source_failures
        review_records = _load_reviews(reviews_path)
        findings = _index_findings(evidence_records)
        _validate_review_history(
            review_records,
            findings,
            evidence_records,
            review_history_key=_review_history_key_for_records(evidence, review_records),
        )
        sensitive_lines: tuple[str, ...] = ()
        if show_sensitive_values:
            if sensitive_finding_id is None:
                raise _ReviewInputError("--show-sensitive-values requires --finding FINDING_ID")
            sensitive_finding = findings.get(sensitive_finding_id)
            if sensitive_finding is None:
                raise _ReviewInputError("sensitive-value finding ID was not found in the evidence")
            sensitive_lines = (
                _bounded_sensitive_semantic_lines(sensitive_finding)
                if sensitive_finding.semantic_finding is not None
                else _bounded_sensitive_invariant_lines(
                    cast(DatasetInvariantRuleResult, sensitive_finding.baseline_rule),
                    cast(DatasetInvariantRuleResult, sensitive_finding.variation_rule),
                )
            )
        elif sensitive_finding_id is not None:
            raise _ReviewInputError("--finding is valid only with --show-sensitive-values")
    except (PatternIdentityKeyError, ReviewHistoryKeyError, _ReviewInputError) as error:
        raise typer.BadParameter(str(error)) from None

    active_reviews = _active_reviews(review_records)
    status_counts = {
        status: sum(review.status == status for review in active_reviews.values())
        for status in ("needs_review", "confirmed", "expected", "unsupported", "inconclusive")
    }

    cases = [case for loaded_record in evidence_records for case in loaded_record.evidence.cases]
    comparison_counts = {
        bucket: sum(_dataset_case_report_bucket(case) == bucket for case in cases)
        for bucket in (
            "consequential",
            "equivalent",
            "inconclusive",
            "unstable",
            "unclassified",
            "no_difference",
            "not_evaluated",
        )
    }
    findings_by_case: dict[int, list[_IndexedFinding]] = {}
    for indexed_finding in findings.values():
        findings_by_case.setdefault(id(indexed_finding.case), []).append(indexed_finding)
    consequential_count = 0
    unresolved_count = len(source_failures)
    for case in cases:
        case_bucket = _dataset_case_report_bucket(case)
        case_findings = findings_by_case.get(id(case), [])
        if not case_findings:
            consequential_count += case_bucket in {"consequential", "unstable"}
            unresolved_count += case_bucket in {"inconclusive", "unclassified"}
            continue
        effective_statuses = {
            _effective_finding_status(
                indexed_finding,
                active_reviews.get(indexed_finding.finding_id),
            )
            for indexed_finding in case_findings
        }
        if effective_statuses & {"confirmed", "needs_review"}:
            consequential_count += 1
        elif "inconclusive" in effective_statuses:
            unresolved_count += 1
    if consequential_count:
        result_summary = (
            f"ACTION REQUIRED — {consequential_count} consequential behavior "
            f"change{'s' if consequential_count != 1 else ''} found"
        )
    elif unresolved_count:
        result_summary = f"INCONCLUSIVE — {unresolved_count} item(s) need attention"
    else:
        result_summary = "CLEAR — no consequential behavior changes found"

    _print_plain("UL dataset report")
    _print_plain(f"Result: {result_summary}")
    _print_plain(
        f"Semantic comparisons: total={len(cases)}, completed="
        f"{len(cases) - comparison_counts['not_evaluated']}, "
        f"no_observed_difference={comparison_counts['no_difference']}"
    )
    _print_plain(
        "Automatic decisions: "
        f"consequential={comparison_counts['consequential']}, "
        f"equivalent={comparison_counts['equivalent']}, "
        f"inconclusive={comparison_counts['inconclusive']}"
    )
    if (
        comparison_counts["unstable"]
        or comparison_counts["unclassified"]
        or comparison_counts["not_evaluated"]
    ):
        _print_plain(
            "Other: "
            f"unstable={comparison_counts['unstable']}, "
            f"unclassified={comparison_counts['unclassified']}, "
            f"not_evaluated={comparison_counts['not_evaluated']}"
        )
    invariant_findings = [
        indexed_finding
        for indexed_finding in findings.values()
        if indexed_finding.kind == "customer_invariant_violation"
    ]
    if invariant_findings:
        invariant_attention_count = sum(
            _effective_finding_status(
                indexed_finding,
                active_reviews.get(indexed_finding.finding_id),
            )
            in {"needs_review", "confirmed"}
            for indexed_finding in invariant_findings
        )
        _print_plain(
            "Customer invariant violations: "
            f"total={len(invariant_findings)}, require_attention={invariant_attention_count}"
        )
    evaluation_mode = _dataset_evaluation_mode(evidence_records, source_failures)
    if evaluation_mode is not None:
        _print_plain(f"Scope: {evaluation_mode}; correctness and severity were not assessed")
    if active_reviews:
        _print_plain(
            "Human review overrides: "
            + ", ".join(f"{status}={count}" for status, count in status_counts.items() if count)
        )
    _print_plain(f"Source preparation failures: {len(source_failures)}")
    for source_failure in source_failures:
        _print_plain("")
        _print_plain(f"Source preparation failure {source_failure.interaction_id}")
        _print_plain(f"Stage: {source_failure.failure_stage}")
        _print_plain(f"Reason: {source_failure.reason_code}")
        _print_plain(f"Summary: {source_failure.summary}")
        _print_plain(f"Next: {source_failure.remediation}")
    if show_sensitive_values:
        _print_plain(
            "WARNING: showing selected private values; they may contain secrets or PII and "
            "may be retained in terminal scrollback, CI output, or logs."
        )
    visible_findings = [
        indexed_finding
        for indexed_finding in findings.values()
        if all_findings
        or indexed_finding.finding_id == sensitive_finding_id
        or _effective_finding_status(
            indexed_finding,
            active_reviews.get(indexed_finding.finding_id),
        )
        in {"needs_review", "confirmed", "inconclusive"}
    ]

    def finding_section(indexed_finding: _IndexedFinding) -> str:
        status = _effective_finding_status(
            indexed_finding,
            active_reviews.get(indexed_finding.finding_id),
        )
        if status == "inconclusive":
            return "Inconclusive comparisons"
        if status in {"expected", "unsupported"}:
            return "Resolved or equivalent differences"
        if status == "confirmed":
            return "Consequential behavior changes"
        if (
            indexed_finding.case.material_variance is not None
            and indexed_finding.case.material_variance.decision == "material_variance"
        ):
            return "Consequential behavior changes"
        return "Findings needing review"

    section_order = {
        "Consequential behavior changes": 0,
        "Findings needing review": 1,
        "Inconclusive comparisons": 2,
        "Resolved or equivalent differences": 3,
    }
    visible_findings.sort(
        key=lambda indexed_finding: (
            section_order[finding_section(indexed_finding)],
            indexed_finding.finding_id,
        )
    )
    current_section: str | None = None
    section_item_number = 0
    semantic_finding_case_ids = {
        id(indexed_finding.case)
        for indexed_finding in findings.values()
        if indexed_finding.kind == "semantic_difference"
    }
    unrepresented_unstable_cases = [
        (loaded_record, case)
        for loaded_record in evidence_records
        for case in loaded_record.evidence.cases
        if _dataset_case_report_bucket(case) == "unstable"
        and id(case) not in semantic_finding_case_ids
    ]
    if unrepresented_unstable_cases:
        _print_plain("")
        _print_plain("Consequential behavior changes")
        current_section = "Consequential behavior changes"
        for loaded_record, case in unrepresented_unstable_cases:
            section_item_number += 1
            _print_plain("")
            _print_plain(
                f"{section_item_number}. Behavior changed inconsistently across repeated trials."
            )
            _print_plain(
                "Case: " + (case.source_record_id or loaded_record.evidence.interaction_id)
            )
            _print_plain(
                "Test variation: " + case.operator_id.replace(".", " / ").replace("_", " ")
            )
            _print_plain("Evidence stability: " + _observations_summary(case.observations))
    for indexed_finding in visible_findings:
        section = finding_section(indexed_finding)
        if section != current_section:
            _print_plain("")
            _print_plain(section)
            current_section = section
            section_item_number = 0
        section_item_number += 1
        loaded_record = indexed_finding.evidence_record
        case = indexed_finding.case
        matching_reviews = [
            review
            for review in _occurrence_review_records(review_records)
            if review.finding_id == indexed_finding.finding_id
        ]
        latest_review = active_reviews.get(indexed_finding.finding_id)
        _print_plain("")
        if indexed_finding.semantic_finding is not None:
            finding = indexed_finding.semantic_finding
            safe_summary = (
                _BEHAVIOR_FINDING_SUMMARIES["changed_response"]
                if finding.category == "changed_response"
                else finding.summary
            )
            _print_plain(f"{section_item_number}. {safe_summary}")
            _print_plain(f"Category: {finding.category}")
            if case.material_variance is not None:
                _print_plain("Reason: " + case.material_variance.reason_code.replace("_", " "))
                if (
                    finding.category == "changed_response"
                    and case.material_variance.reason_code == "action_count_changed"
                ):
                    difference_summary = _response_action_count_summary(finding)
                    if difference_summary is not None:
                        _print_plain("What changed: " + difference_summary)
        else:
            baseline_rule = indexed_finding.baseline_rule
            variation_rule = indexed_finding.variation_rule
            if baseline_rule is None or variation_rule is None:
                raise AssertionError("invariant finding requires both rule results")
            _print_plain(f"{section_item_number}. Customer-defined rule changed state.")
            _print_plain("Category: customer_invariant_violation")
            _print_plain(f"Semantic comparison status: {case.status}")
            _print_plain(
                f"Invariant finding status: original={baseline_rule.status}; "
                f"variation={variation_rule.status}"
            )
            _print_plain(
                f"Rule: {variation_rule.rule_id} ({variation_rule.rule_version}); "
                f"type={variation_rule.rule_type}; declared_severity={variation_rule.severity}"
            )
            _print_plain(f"Description: {variation_rule.description}")
        _print_plain("Test variation: " + case.operator_id.replace(".", " / ").replace("_", " "))
        stability_label = (
            "Evidence stability"
            if indexed_finding.semantic_finding is not None
            else "Full-response stability"
        )
        _print_plain(
            f"{stability_label}: original="
            + _observations_summary(loaded_record.evidence.current_baseline.observations)
            + "; variation="
            + _observations_summary(case.observations)
        )
        if indexed_finding.semantic_finding is None:
            baseline_rule = indexed_finding.baseline_rule
            variation_rule = indexed_finding.variation_rule
            if baseline_rule is None or variation_rule is None:
                raise AssertionError("invariant finding requires both rule results")
            _print_plain(
                f"Invariant repetitions: original satisfied={len(baseline_rule.trials)}/"
                f"{len(baseline_rule.trials)}; variation violated="
                f"{len(variation_rule.trials)}/{len(variation_rule.trials)}"
            )
            _print_plain(
                f"Rule transition: original={baseline_rule.status}; "
                f"variation={variation_rule.status}"
            )
            for trial in variation_rule.trials:
                _print_plain(
                    f"Variation rule trial {trial.repetition}: {trial.status}; "
                    f"{_invariant_trial_location(trial)}; reason={trial.reason_code}"
                )
            _print_plain(
                "Finding limitations: causality not established; production prevalence not "
                "measured; whole-task correctness not established."
            )
        if show_sensitive_values and indexed_finding.finding_id == sensitive_finding_id:
            for sensitive_line in sensitive_lines:
                _print_sensitive_plain(sensitive_line)
        if latest_review is None:
            if case.material_variance is None:
                _print_plain(f"Decision: needs review (history: {len(matching_reviews)})")
        else:
            _print_plain(
                f"Latest review: {latest_review.status}, severity={latest_review.severity}, "
                f"reviewer={latest_review.reviewer}, id={latest_review.review_id} "
                f"(history: {len(matching_reviews)})"
            )
            _print_plain(f"Review reason: {latest_review.reason}")
        _print_plain(f"Finding {indexed_finding.finding_id}")
        if not show_sensitive_values:
            _print_plain(
                "Inspect private values: ul dataset report "
                f"{shlex.quote(str(evidence))} --finding "
                f"{shlex.quote(indexed_finding.finding_id)} --show-sensitive-values"
            )

    unrepresented_inconclusive_cases = [
        (loaded_record, case)
        for loaded_record in evidence_records
        for case in loaded_record.evidence.cases
        if _dataset_case_report_bucket(case) == "inconclusive"
        and id(case) not in semantic_finding_case_ids
    ]
    if unrepresented_inconclusive_cases:
        if current_section != "Inconclusive comparisons":
            _print_plain("")
            _print_plain("Inconclusive comparisons")
            section_item_number = 0
        for loaded_record, case in unrepresented_inconclusive_cases:
            section_item_number += 1
            reasons = tuple(case.inconclusive_reasons)
            if not reasons:
                reason = (
                    "variation observations unavailable"
                    if case.observations is None
                    else "variation behavior was not stable"
                )
                reasons = (reason,)
            _print_plain("")
            _print_plain(f"{section_item_number}. Comparison could not be classified.")
            _print_plain(
                "Case: " + (case.source_record_id or loaded_record.evidence.interaction_id)
            )
            _print_plain(
                "Test variation: " + case.operator_id.replace(".", " / ").replace("_", " ")
            )
            _print_plain("Reasons: " + ", ".join(reasons))

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
    if review_records:
        _print_plain(f"Review history: {reviews_path}")
        _print_plain(
            "Review meanings: confirmed=problem in context; expected=supported acceptable "
            "difference; unsupported=machine claim not supported; "
            "inconclusive=insufficient context."
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
        with _locked_reviews_file(reviews_path) as locked_reviews:
            review_records = _read_reviews_descriptor(locked_reviews.descriptor)
            _validate_review_history(
                review_records,
                findings,
                evidence_records,
                review_history_key=_review_history_key_for_records(evidence, review_records),
            )
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
            if (
                os.fstat(locked_reviews.descriptor).st_size + len(encoded_review)
                > _MAXIMUM_REVIEWS_BYTES
            ):
                raise _ReviewInputError("review file exceeds the 10 MB limit")
            _append_review_atomically(locked_reviews, encoded_review)
    except (
        ValidationError,
        PatternIdentityKeyError,
        ReviewHistoryKeyError,
        _ReviewInputError,
    ) as error:
        message = "review fields are invalid" if isinstance(error, ValidationError) else str(error)
        raise typer.BadParameter(message) from None
    except _ReviewCommitUncertainError:
        raise typer.BadParameter(
            "review was replaced but directory durability could not be confirmed; "
            "inspect the review history before retrying"
        ) from None
    except OSError as error:
        raise typer.BadParameter(
            f"cannot safely update review file ({error.__class__.__name__})"
        ) from None

    _print_plain(f"Recorded review {new_review.review_id}: {new_review.status}")
    if new_review.status == "confirmed":
        save_command = f"ul regression save {shlex.quote(str(evidence))} {shlex.quote(finding_id)}"
        if reviews is not None:
            save_command += f" --reviews {shlex.quote(str(reviews_path))}"
        if selected.kind == "semantic_difference":
            save_command += " --invariants INVARIANTS.json --rule RULE_ID"
        _print_plain(
            f"Promote this finding with '{save_command} --output CASE.json "
            "--confirm-versioned-input'. Replace uppercase placeholders first."
        )
    _print_plain(f"Review history: {reviews_path}")


def review_dataset_pattern(
    evidence: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, readable=True, help="Evaluation evidence JSONL."
        ),
    ],
    pattern_snapshot_id: Annotated[
        str,
        typer.Argument(help="Exact pattern snapshot ID shown by 'ul report'."),
    ],
    status: Annotated[
        ReviewStatus | None,
        typer.Option(help="Human judgment to apply after previewing the exact snapshot."),
    ] = None,
    reviewer: Annotated[
        str | None,
        typer.Option(help="Person or team making the judgment."),
    ] = None,
    reason: Annotated[
        str | None,
        typer.Option(help="Why this judgment applies to the reviewed occurrences."),
    ] = None,
    severity: Annotated[
        ReviewSeverity,
        typer.Option(help="Consequence severity; only confirmed patterns may be rated."),
    ] = "unrated",
    reviews: Annotated[
        Path | None,
        typer.Option(help="Review JSONL; defaults to EVIDENCE with .reviews.jsonl suffix."),
    ] = None,
) -> None:
    """Preview or append a decision bound to one exact pattern snapshot."""
    reviews_path = reviews or _default_reviews_path(evidence)
    try:
        evidence_records = _load_evidence(evidence)
        findings = _index_findings(evidence_records)
        review_records = _load_reviews(reviews_path)
        pattern_identity_key = load_pattern_identity_key(evidence)
        _validate_review_history(
            review_records,
            findings,
            evidence_records,
            review_history_key=_review_history_key_for_records(evidence, review_records),
        )
        report = _summarize_dataset_evidence(
            evidence,
            reviews_path,
            pattern_identity_key=pattern_identity_key,
        )
        matching_patterns = [
            pattern
            for pattern in report.patterns
            if pattern.pattern_snapshot_id == pattern_snapshot_id
        ]
        if len(matching_patterns) != 1:
            raise _ReviewInputError(
                "pattern snapshot was not found; rerun 'ul report' and use its exact snapshot ID"
            )
        pattern = matching_patterns[0]
        reviewed_finding_ids = tuple(
            member.finding_id
            for member in pattern.members
            if member.review_status == "needs_review"
        )
        exception_members = tuple(
            sorted(
                (
                    member
                    for sibling in report.patterns
                    if sibling.pattern_fingerprint == pattern.pattern_fingerprint
                    and sibling.pattern_snapshot_id != pattern.pattern_snapshot_id
                    for member in sibling.members
                ),
                key=lambda member: member.finding_id,
            )
        )
        exception_finding_ids = tuple(member.finding_id for member in exception_members)
        if not reviewed_finding_ids:
            raise _ReviewInputError("pattern snapshot has no unreviewed occurrences to decide")
        _print_pattern_review_preview(
            pattern,
            reviewed_finding_ids=reviewed_finding_ids,
            exception_members=exception_members,
            prior_reviews=tuple(
                review
                for review in report.pattern_reviews
                if review.pattern_fingerprint == pattern.pattern_fingerprint
            ),
        )
        if status is None:
            if reviewer is not None or reason is not None or severity != "unrated":
                raise _ReviewInputError("--reviewer, --reason, and --severity require --status")
            _print_plain("No review was written.")
            return
        if reviewer is None or reason is None:
            raise _ReviewInputError("--status requires --reviewer and --reason")
        review_history_key = load_review_history_key(evidence)
        reviewed_at = datetime.now(UTC)
        occurrence_decisions: list[PatternOccurrenceDecision] = []
        for finding_id in reviewed_finding_ids:
            selected = findings.get(finding_id)
            if selected is None:
                raise _ReviewInputError("evidence changed; rerun the pattern preview")
            occurrence_decisions.append(
                PatternOccurrenceDecision(
                    review_id=f"ulr_{uuid4()}",
                    evidence_record_sha256=selected.evidence_record.sha256,
                    finding_id=finding_id,
                )
            )
        pattern_review = PatternReviewRecord(
            pattern_review_id=f"ulpr_{uuid4()}",
            record_hmac_sha256="0" * 64,
            grouping_evidence_sha256=_indexed_pattern_grouping_sha256(
                findings[reviewed_finding_ids[0]]
            ),
            pattern_snapshot=pattern,
            status=status,
            severity=severity,
            reviewed_finding_ids=reviewed_finding_ids,
            exception_finding_ids=exception_finding_ids,
            evidence_record_sha256s=tuple(record.sha256 for record in evidence_records),
            evidence_bindings=tuple(
                PatternEvidenceBinding(
                    finding_id=finding_id,
                    evidence_record_sha256=indexed_finding.evidence_record.sha256,
                )
                for finding_id, indexed_finding in sorted(findings.items())
            ),
            occurrence_decisions=tuple(occurrence_decisions),
            reviewer=reviewer,
            reason=reason,
            reviewed_at=reviewed_at,
        )
        pattern_review = pattern_review.model_copy(
            update={
                "record_hmac_sha256": pattern_review_record_hmac(
                    review_history_key,
                    pattern_review.hmac_payload(),
                )
            }
        )
        encoded_record = (
            json.dumps(
                pattern_review.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        with _locked_reviews_file(reviews_path) as locked_reviews:
            current_records = _read_reviews_descriptor(locked_reviews.descriptor)
            if current_records != review_records:
                raise _ReviewInputError("review history changed; rerun the pattern preview")
            _validate_review_history(
                [*current_records, pattern_review],
                findings,
                evidence_records,
                review_history_key=review_history_key,
            )
            if len(current_records) + 1 > _MAXIMUM_REVIEW_RECORDS:
                raise _ReviewInputError("review file exceeds the 10,000 record limit")
            if (
                os.fstat(locked_reviews.descriptor).st_size + len(encoded_record)
                > _MAXIMUM_REVIEWS_BYTES
            ):
                raise _ReviewInputError("review file exceeds the 10 MB limit")
            _append_review_atomically(locked_reviews, encoded_record)
    except (
        ValidationError,
        PatternIdentityKeyError,
        ReviewHistoryKeyError,
        _ReviewInputError,
    ) as error:
        message = "review fields are invalid" if isinstance(error, ValidationError) else str(error)
        raise typer.BadParameter(message) from None
    except _ReviewCommitUncertainError:
        raise typer.BadParameter(
            "review was replaced but directory durability could not be confirmed; "
            "inspect the review history before retrying"
        ) from None
    except OSError as error:
        raise typer.BadParameter(
            f"cannot safely update review file ({error.__class__.__name__})"
        ) from None

    _print_plain(f"Recorded pattern review {pattern_review.pattern_review_id}: {status}")
    _print_plain(f"Bound snapshot: {pattern_snapshot_id}")
    _print_plain(f"Reviewed occurrences: {len(reviewed_finding_ids)}")
    _print_plain(f"Exceptions unchanged: {len(exception_finding_ids)}")
    _print_plain("Future matching occurrences remain needs_review; this decision is context only.")
    if status == "confirmed":
        _print_plain(
            "Promote selected occurrences one at a time with 'ul regression save EVIDENCE "
            "FINDING_ID ...'."
        )
    _print_plain(f"Review history: {reviews_path}")


def _print_pattern_review_preview(
    pattern: FailurePattern,
    *,
    reviewed_finding_ids: tuple[str, ...],
    exception_members: tuple[PatternMember, ...],
    prior_reviews: tuple[PatternReviewSummary, ...],
) -> None:
    _print_plain("Pattern review preview")
    _print_plain(f"Pattern fingerprint: {pattern.pattern_fingerprint}")
    _print_plain(f"Exact snapshot: {pattern.pattern_snapshot_id}")
    _print_plain(f"Summary: {pattern.summary}")
    _print_plain(f"Decision candidates ({len(reviewed_finding_ids)}):")
    for finding_id in reviewed_finding_ids:
        _print_plain(f"  {finding_id}")
    _print_plain(f"Exceptions unchanged ({len(exception_members)}):")
    for member in exception_members:
        _print_plain(f"  {member.finding_id}: {member.review_status}/{member.review_severity}")
    _print_plain(f"Prior decisions for this fingerprint: {len(prior_reviews)}")
    for review in prior_reviews:
        _print_plain(
            f"  {review.pattern_review_id}: {review.status}/{review.severity}; "
            f"reviewed_at={review.reviewed_at.isoformat()}; "
            f"snapshot={review.pattern_snapshot_id}"
        )
    _print_plain(
        "Limitation: grouping supports navigation and review; it does not establish causation, "
        "production prevalence, correctness, or a shared root cause."
    )


def _default_reviews_path(evidence: Path) -> Path:
    return evidence.with_suffix(".reviews.jsonl")


def _load_evidence_document(path: Path) -> _LoadedEvidenceDocument:
    try:
        raw = _read_bounded_regular_file(path, _MAXIMUM_EVIDENCE_BYTES)
    except OSError as error:
        raise _ReviewInputError(
            f"cannot safely read evidence ({error.__class__.__name__})"
        ) from None
    raw_lines = raw.splitlines()
    if raw_lines and dataset_durable_run_marker_manifest_sha256(raw_lines[0]) is not None:
        raw_lines = raw_lines[1:]
    if not raw_lines or len(raw_lines) > _MAXIMUM_EVIDENCE_RECORDS:
        raise _ReviewInputError("evidence must contain 1 to 100 JSONL records")
    if any(not raw_line.strip() for raw_line in raw_lines):
        raise _ReviewInputError("evidence contains an empty JSONL record")
    records: list[_LoadedEvidenceRecord] = []
    source_failures: list[DatasetSourcePreparationFailureEvidence] = []
    successful_interaction_ids: set[str] = set()
    failed_interaction_ids: set[str] = set()
    try:
        for raw_line in raw_lines:
            decoded_line: object = json.loads(raw_line)
            decoded_record = (
                cast(dict[str, object], decoded_line) if isinstance(decoded_line, dict) else None
            )
            if (
                decoded_record is not None
                and decoded_record.get("record_type") == "source_preparation_failure"
            ):
                source_failure = DatasetSourcePreparationFailureEvidence.model_validate_json(
                    raw_line
                )
                interaction_id = source_failure.interaction_id
                if (
                    interaction_id in successful_interaction_ids
                    or interaction_id in failed_interaction_ids
                ):
                    raise ValueError("duplicate interaction ID")
                failed_interaction_ids.add(interaction_id)
                source_failures.append(source_failure)
            else:
                loaded_record = _LoadedEvidenceRecord(
                    evidence=_EvidenceRecord.model_validate_json(raw_line),
                    sha256=hashlib.sha256(raw_line).hexdigest(),
                )
                interaction_id = loaded_record.evidence.interaction_id
                if interaction_id in failed_interaction_ids:
                    raise ValueError("duplicate interaction ID")
                successful_interaction_ids.add(interaction_id)
                records.append(loaded_record)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError):
        raise _ReviewInputError(
            "evidence is not valid UL dataset evidence JSONL; expected success schema through "
            "1.15.0 or source-preparation-failure/1.0.0"
        ) from None
    return _LoadedEvidenceDocument(
        records=records,
        source_failures=tuple(source_failures),
    )


def _load_evidence(path: Path) -> list[_LoadedEvidenceRecord]:
    return _load_evidence_document(path).records


def dataset_durable_run_marker_manifest_sha256(raw_line: bytes) -> str | None:
    try:
        decoded: object = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    value = cast(dict[str, object], decoded)
    if set(value) != {
        "schema_version",
        "record_type",
        "manifest_sha256",
    }:
        return None
    manifest_sha256 = value.get("manifest_sha256")
    if (
        value.get("schema_version") != "1.0.0"
        or value.get("record_type") != "dataset_durable_run"
        or not isinstance(manifest_sha256, str)
        or re.fullmatch(_SHA256_PATTERN, manifest_sha256) is None
    ):
        return None
    return manifest_sha256


def is_reportable_dataset_evidence(path: Path) -> bool:
    try:
        _load_evidence_document(path)
    except _ReviewInputError:
        return False
    return True


def _load_reviews(path: Path) -> list[ReviewHistoryRecord]:
    try:
        descriptor = _open_regular_file(path, os.O_RDONLY)
        try:
            return _read_reviews_descriptor(descriptor)
        finally:
            os.close(descriptor)
    except FileNotFoundError:
        return []
    except OSError as error:
        raise _ReviewInputError(
            f"cannot safely read review file ({error.__class__.__name__})"
        ) from None


def _read_reviews_descriptor(descriptor: int) -> list[ReviewHistoryRecord]:
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
        records: list[ReviewHistoryRecord] = []
        for line in raw_lines:
            raw_value: object = json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
            if not isinstance(raw_value, dict):
                raise ValueError("review rows must be objects")
            raw_record = cast(dict[str, object], raw_value)
            if raw_record.get("record_type", "occurrence_review") == "occurrence_review":
                records.append(ReviewRecord.model_validate_json(line))
            elif raw_record.get("record_type") == "pattern_review":
                records.append(PatternReviewRecord.model_validate_json(line))
            else:
                raise ValueError("unknown review record type")
        return records
    except (json.JSONDecodeError, ValidationError, ValueError):
        raise _ReviewInputError("review file is not valid review JSONL") from None


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


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
def _locked_reviews_file(path: Path) -> Generator[_LockedReviewsFile]:
    lock_descriptor = _open_regular_file(_reviews_lock_path(path), os.O_RDWR | os.O_CREAT)
    locked = False
    reviews_file: _LockedReviewsFile | None = None
    try:
        _set_private_file_permissions(lock_descriptor)
        _lock_file(lock_descriptor, exclusive=True)
        locked = True
        descriptor = _open_regular_file(path, os.O_RDWR | os.O_CREAT)
        _set_private_file_permissions(descriptor)
        reviews_file = _LockedReviewsFile(path=path, descriptor=descriptor)
        yield reviews_file
    finally:
        try:
            if reviews_file is not None and reviews_file.descriptor >= 0:
                os.close(reviews_file.descriptor)
            if locked:
                _unlock_file(lock_descriptor)
        finally:
            os.close(lock_descriptor)


def _reviews_lock_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.lock")


def _append_review_atomically(
    locked_reviews: _LockedReviewsFile,
    encoded_record: bytes,
) -> None:
    descriptor = locked_reviews.descriptor
    existing_status = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    existing = b""
    while len(existing) < existing_status.st_size:
        chunk = os.read(descriptor, min(65_536, existing_status.st_size - len(existing)))
        if not chunk:
            raise OSError("review file changed while preparing the update")
        existing += chunk
    temporary_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{locked_reviews.path.name}.",
        suffix=".tmp",
        dir=locked_reviews.path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        _set_private_file_permissions(temporary_descriptor)
        remaining = memoryview(existing + encoded_record)
        while remaining:
            written = os.write(temporary_descriptor, remaining)
            if written == 0:
                raise OSError("could not write review transaction")
            remaining = remaining[written:]
        os.fsync(temporary_descriptor)
        os.close(temporary_descriptor)
        temporary_descriptor = -1
        current_status = os.stat(locked_reviews.path, follow_symlinks=False)
        if not os.path.samestat(existing_status, current_status):
            raise OSError("review file changed while preparing the update")
        os.close(descriptor)
        locked_reviews.descriptor = -1
        os.replace(temporary_path, locked_reviews.path)
        try:
            _fsync_directory(locked_reviews.path.parent)
        except OSError as error:
            raise _ReviewCommitUncertainError from error
    finally:
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        with suppress(FileNotFoundError):
            temporary_path.unlink()


def _fsync_directory(path: Path) -> None:
    if sys.platform == "win32":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
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
                if not is_reproduced_invariant_difference(baseline_rule, variation_rule):
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


def _validate_pattern_review_snapshot(
    review: PatternReviewRecord,
    findings: dict[str, _IndexedFinding],
    evidence_records: list[_LoadedEvidenceRecord],
    active_by_finding: dict[str, ReviewRecord],
) -> None:
    manifest = review.evidence_record_sha256s
    current_manifest = tuple(record.sha256 for record in evidence_records)
    if current_manifest[: len(manifest)] != manifest:
        raise _ReviewInputError("pattern review evidence manifest is not a current prefix")
    manifest_digests = set(manifest)
    expected_binding_ids = tuple(
        sorted(
            finding_id
            for finding_id, finding in findings.items()
            if finding.evidence_record.sha256 in manifest_digests
        )
    )
    if tuple(binding.finding_id for binding in review.evidence_bindings) != expected_binding_ids:
        raise _ReviewInputError("pattern review bindings do not cover its evidence manifest")
    bound_finding_ids: list[str] = []
    for binding in review.evidence_bindings:
        finding = findings.get(binding.finding_id)
        if finding is None:
            raise _ReviewInputError("pattern review evidence binding is no longer available")
        if binding.evidence_record_sha256 != finding.evidence_record.sha256:
            raise _ReviewInputError("pattern review evidence binding does not match its record")
        bound_finding_ids.append(binding.finding_id)
    matching_finding_ids = tuple(
        sorted(
            finding_id
            for finding_id in bound_finding_ids
            if _optional_indexed_pattern_grouping_sha256(findings[finding_id])
            == review.grouping_evidence_sha256
        )
    )
    reviewed_finding_ids = tuple(
        finding_id for finding_id in matching_finding_ids if finding_id not in active_by_finding
    )
    exception_finding_ids = tuple(
        finding_id for finding_id in matching_finding_ids if finding_id in active_by_finding
    )
    if reviewed_finding_ids != review.reviewed_finding_ids:
        raise _ReviewInputError("pattern review does not bind the exact unreviewed cohort")
    if exception_finding_ids != review.exception_finding_ids:
        raise _ReviewInputError("pattern review exceptions do not match reviewed sibling cohorts")
    snapshot = review.pattern_snapshot
    selected = [findings[finding_id] for finding_id in reviewed_finding_ids]
    descriptor = _indexed_pattern_grouping_descriptor(selected[0])
    if (
        snapshot.kind != descriptor["kind"]
        or snapshot.category != descriptor["category"]
        or snapshot.rule_id != descriptor["rule_id"]
        or snapshot.rule_version != descriptor["rule_version"]
        or snapshot.summary != descriptor["summary"]
        or snapshot.stability != descriptor["stability"]
        or snapshot.evidence_authorities != descriptor["evidence_authorities"]
        or snapshot.evidence_limitations != descriptor["evidence_limitations"]
        or (
            snapshot.vertical_facets.model_dump(mode="json", exclude_none=True)
            if snapshot.vertical_facets is not None
            else None
        )
        != descriptor["vertical_facets"]
    ):
        raise _ReviewInputError("pattern review snapshot does not match its evidence grouping")
    expected_operator_keys = tuple(
        sorted({(finding.case.operator_id, finding.case.operator_version) for finding in selected})
    )
    if (
        tuple((operator.operator_id, operator.operator_version) for operator in snapshot.operators)
        != expected_operator_keys
    ):
        raise _ReviewInputError("pattern review operators do not match its evidence cohort")
    if snapshot.source_case_count != len(
        {finding.evidence_record.evidence.interaction_id for finding in selected}
    ):
        raise _ReviewInputError("pattern review source count does not match its evidence cohort")
    expected_severity = cast(FindingSeverity, descriptor["declared_severity"] or "unrated")
    if snapshot.severity != expected_severity:
        raise _ReviewInputError("pattern review severity does not match its evidence cohort")


def _validate_review_history(
    reviews: list[ReviewHistoryRecord],
    findings: dict[str, _IndexedFinding],
    evidence_records: list[_LoadedEvidenceRecord],
    *,
    review_history_key: bytes | None,
) -> None:
    reviews_by_id: dict[str, ReviewRecord] = {}
    pattern_reviews_by_id: dict[str, PatternReviewRecord] = {}
    pattern_snapshot_ids: set[str] = set()
    superseded_ids: set[str] = set()
    active_by_finding: dict[str, ReviewRecord] = {}
    historical_active_by_finding: dict[str, ReviewRecord] = {}
    for review in reviews:
        if isinstance(review, PatternReviewRecord):
            if review_history_key is None:
                raise _ReviewInputError(
                    "pattern review history requires the project review history key"
                )
            expected_hmac = pattern_review_record_hmac(
                review_history_key,
                review.hmac_payload(),
            )
            if not hmac.compare_digest(review.record_hmac_sha256, expected_hmac):
                raise _ReviewInputError("pattern review authentication does not match its record")
            if review.pattern_review_id in pattern_reviews_by_id:
                raise _ReviewInputError("review file contains a duplicate pattern review ID")
            snapshot_id = review.pattern_snapshot.pattern_snapshot_id
            if snapshot_id in pattern_snapshot_ids:
                raise _ReviewInputError("pattern snapshot already has a decision")
            for member in review.pattern_snapshot.members:
                selected = findings.get(member.finding_id)
                if selected is None:
                    raise _ReviewInputError(
                        "pattern review references a finding outside this evidence"
                    )
            if any(finding_id not in findings for finding_id in review.exception_finding_ids):
                raise _ReviewInputError(
                    "pattern review exception references a finding outside this evidence"
                )
            _validate_pattern_review_snapshot(
                review,
                findings,
                evidence_records,
                historical_active_by_finding,
            )
            pattern_reviews_by_id[review.pattern_review_id] = review
            pattern_snapshot_ids.add(snapshot_id)
            for derived_review in review.derived_reviews():
                historical_active_by_finding[derived_review.finding_id] = derived_review
        else:
            historical_active_by_finding[review.finding_id] = review
    occurrence_reviews = tuple(
        occurrence_review
        for review in reviews
        for occurrence_review in (
            review.derived_reviews() if isinstance(review, PatternReviewRecord) else (review,)
        )
    )
    for review in occurrence_reviews:
        if review.review_id in reviews_by_id:
            raise _ReviewInputError("review file contains a duplicate review ID")
        selected = findings.get(review.finding_id)
        if selected is None:
            raise _ReviewInputError("review references a finding outside this evidence")
        if review.evidence_record_sha256 != selected.evidence_record.sha256:
            raise _ReviewInputError("review evidence digest does not match the evidence record")
        if review.origin_pattern_review_id is not None:
            pattern_review = pattern_reviews_by_id.get(review.origin_pattern_review_id)
            if pattern_review is None:
                raise _ReviewInputError("occurrence review references an unknown pattern review")
            if review.finding_id not in pattern_review.reviewed_finding_ids:
                raise _ReviewInputError("pattern review did not select this occurrence")
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


def _review_history_key_for_records(
    evidence: Path,
    reviews: list[ReviewHistoryRecord],
) -> bytes | None:
    if any(isinstance(review, PatternReviewRecord) for review in reviews):
        return load_review_history_key(evidence)
    return None


def _active_reviews(reviews: list[ReviewHistoryRecord]) -> dict[str, ReviewRecord]:
    active: dict[str, ReviewRecord] = {}
    for occurrence_review in _occurrence_review_records(reviews):
        active[occurrence_review.finding_id] = occurrence_review
    return active


def _occurrence_review_records(
    reviews: list[ReviewHistoryRecord],
) -> tuple[ReviewRecord, ...]:
    return tuple(
        occurrence_review
        for review in reviews
        for occurrence_review in (
            review.derived_reviews() if isinstance(review, PatternReviewRecord) else (review,)
        )
    )


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


def _bounded_sensitive_semantic_lines(
    indexed_finding: _IndexedFinding,
) -> tuple[str, ...]:
    finding = indexed_finding.semantic_finding
    if finding is None:
        raise AssertionError("semantic disclosure requires a semantic finding")
    return _bounded_sensitive_lines(
        (
            _sensitive_json_line(
                "Original input",
                {"value": indexed_finding.evidence_record.evidence.original_input},
            ),
            _sensitive_json_line(
                "Variation input", {"value": indexed_finding.case.augmented_input}
            ),
            _sensitive_json_line(
                "Reference effects",
                {"value": [effect.model_dump(mode="json") for effect in finding.reference_effects]},
            ),
            _sensitive_json_line(
                "Observed effects",
                {"value": [effect.model_dump(mode="json") for effect in finding.observed_effects]},
            ),
        )
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
    return _bounded_sensitive_lines(_sensitive_invariant_lines(baseline_rule, variation_rule))


def _bounded_sensitive_lines(lines_to_check: Iterable[str]) -> tuple[str, ...]:
    lines: list[str] = []
    encoded_bytes = 0
    for line in lines_to_check:
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
