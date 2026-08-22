from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import defaultdict
from pathlib import Path
from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator
from ul_core.dataset import InteractionRecord
from ul_core.models import ULModel

from ul.dataset_augmentation import DatasetAugmentationOperator

QualificationCorpusSegment = Literal[
    "short",
    "long",
    "factual",
    "conversational",
    "tool_oriented",
]
QualificationApplicabilityProfile = Literal["broad", "conditional"]
QualificationDimension = Literal[
    "applicability",
    "generation_success",
    "transformation_strength",
    "meaning_preservation",
    "realism",
    "rejection",
    "repeatability",
]
QualificationGateComparison = Literal["at_least", "at_most"]
QualificationStatus = Literal["thresholds_met", "blocked"]

_CORPUS_SEGMENTS: tuple[QualificationCorpusSegment, ...] = (
    "short",
    "long",
    "factual",
    "conversational",
    "tool_oriented",
)
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_REPORT_ID_PATTERN = r"^ulaq_v1_[0-9a-f]{64}$"
_MAXIMUM_QUALIFICATION_FILE_BYTES = 10_000_000


class _StrictQualificationModel(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class AugmentationQualificationCorpusCase(_StrictQualificationModel):
    segment: QualificationCorpusSegment
    record: InteractionRecord


class AugmentationQualificationCorpus(_StrictQualificationModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    name: str = Field(min_length=1, max_length=200)
    version: str = Field(pattern=r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
    cases: tuple[AugmentationQualificationCorpusCase, ...] = Field(min_length=5, max_length=100)

    @model_validator(mode="after")
    def validate_diversity(self) -> Self:
        case_ids = tuple(case.record.id for case in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("qualification corpus case identifiers must be unique")
        missing_segments = set(_CORPUS_SEGMENTS) - {case.segment for case in self.cases}
        if missing_segments:
            raise ValueError(
                f"qualification corpus is missing segments: {sorted(missing_segments)}"
            )
        return self

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.model_dump(mode="json"))


class AugmentationQualificationHumanReview(_StrictQualificationModel):
    criteria_version: str = Field(
        pattern=r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
    )
    meaning_preserved: bool
    realistic: bool


class AugmentationQualificationAttempt(_StrictQualificationModel):
    case_id: str = Field(min_length=1, max_length=500)
    segment: QualificationCorpusSegment
    repetition: int = Field(ge=1, le=100)
    applicable: bool
    generation_succeeded: bool
    transformation_strong: bool | None = None
    meaning_preserved: bool | None = None
    rejected: bool | None = None
    output_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    failure_reasons: tuple[str, ...] = Field(default=(), max_length=100)
    human_review: AugmentationQualificationHumanReview | None = None

    @model_validator(mode="after")
    def validate_attempt(self) -> Self:
        if not self.applicable and self.generation_succeeded:
            raise ValueError("inapplicable attempts cannot generate a candidate")
        generated_fields = (
            self.transformation_strong,
            self.meaning_preserved,
            self.rejected,
            self.output_sha256,
        )
        if self.generation_succeeded:
            if not self.applicable:
                raise ValueError("generated attempts must be applicable")
            if any(value is None for value in generated_fields):
                raise ValueError("generated attempts require every automated assessment")
            if (self.transformation_strong is False or self.meaning_preserved is False) and not (
                self.rejected
            ):
                raise ValueError("failed automated assessments must reject the candidate")
            if self.rejected and not self.failure_reasons:
                raise ValueError("rejected candidates require a failure reason")
            if not self.rejected and self.failure_reasons:
                raise ValueError("accepted candidates cannot contain failure reasons")
        elif any(value is not None for value in generated_fields):
            raise ValueError("attempts without generation cannot contain candidate assessments")
        if not self.generation_succeeded and not self.failure_reasons:
            raise ValueError("attempts without generation require a reason")
        if self.human_review is not None and not self.generation_succeeded:
            raise ValueError("only generated candidates can receive human review")
        return self


class AugmentationQualificationThresholds(_StrictQualificationModel):
    minimum_applicability_rate: float = Field(ge=0, le=1)
    minimum_generation_success_rate: float = Field(ge=0, le=1)
    minimum_transformation_strength_rate: float = Field(ge=0, le=1)
    minimum_meaning_preservation_rate: float = Field(ge=0, le=1)
    minimum_realism_rate: float = Field(ge=0, le=1)
    maximum_rejection_rate: float = Field(ge=0, le=1)
    minimum_repeatability_rate: float = Field(ge=0, le=1)
    minimum_human_reviewed_cases: int = Field(ge=1, le=100)


class AugmentationQualificationRate(_StrictQualificationModel):
    passed_count: int = Field(ge=0)
    assessed_count: int = Field(ge=0)
    rate: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_rate(self) -> Self:
        if self.passed_count > self.assessed_count:
            raise ValueError("rate passed count cannot exceed its assessed count")
        expected_rate = self.passed_count / self.assessed_count if self.assessed_count else None
        if self.rate != expected_rate:
            raise ValueError("rate must match its counts")
        return self


class AugmentationQualificationSegmentRate(_StrictQualificationModel):
    segment: QualificationCorpusSegment
    applicability: AugmentationQualificationRate


class AugmentationQualificationMetrics(_StrictQualificationModel):
    applicability: AugmentationQualificationRate
    segment_applicability: tuple[AugmentationQualificationSegmentRate, ...]
    generation_success: AugmentationQualificationRate
    transformation_strength: AugmentationQualificationRate
    meaning_preservation: AugmentationQualificationRate
    realism: AugmentationQualificationRate
    rejection: AugmentationQualificationRate
    repeatability: AugmentationQualificationRate
    human_reviewed_case_count: int = Field(ge=0)


class AugmentationQualificationGate(_StrictQualificationModel):
    dimension: QualificationDimension
    scope: str = Field(min_length=1, max_length=200)
    comparison: QualificationGateComparison
    observed: float | int | None
    threshold: float | int
    passed: bool

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.observed is None:
            expected_passed = False
        elif self.comparison == "at_least":
            expected_passed = self.observed >= self.threshold
        else:
            expected_passed = self.observed <= self.threshold
        if self.passed != expected_passed:
            raise ValueError("qualification gate status must match its values")
        return self


class AugmentationQualificationOperatorReference(_StrictQualificationModel):
    id: str = Field(min_length=3, max_length=200)
    version: str = Field(pattern=r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")


class AugmentationQualificationReport(_StrictQualificationModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    report_id: str = Field(pattern=_REPORT_ID_PATTERN)
    operator: AugmentationQualificationOperatorReference
    operator_definition_sha256: str = Field(pattern=_SHA256_PATTERN)
    corpus_name: str = Field(min_length=1, max_length=200)
    corpus_version: str = Field(
        pattern=r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
    )
    corpus_sha256: str = Field(pattern=_SHA256_PATTERN)
    applicability_profile: QualificationApplicabilityProfile
    thresholds: AugmentationQualificationThresholds
    qualification_input_sha256: str = Field(pattern=_SHA256_PATTERN)
    evidence_status: Literal["caller_supplied_unverified"] = "caller_supplied_unverified"
    status: QualificationStatus
    metrics: AugmentationQualificationMetrics
    gates: tuple[AugmentationQualificationGate, ...] = Field(min_length=8)
    attempts: tuple[AugmentationQualificationAttempt, ...] = Field(min_length=10, max_length=10_000)
    failed_case_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        expected_metrics = _qualification_metrics(self.attempts)
        if self.metrics != expected_metrics:
            raise ValueError("qualification metrics must match their attempts")
        has_meaning_change = any(
            attempt.meaning_preserved is False
            or (attempt.human_review is not None and not attempt.human_review.meaning_preserved)
            for attempt in self.attempts
        )
        expected_gates = _qualification_gates(
            applicability_profile=self.applicability_profile,
            thresholds=self.thresholds,
            metrics=expected_metrics,
            has_meaning_change=has_meaning_change,
        )
        if self.gates != expected_gates:
            raise ValueError("qualification gates must match their metrics and thresholds")
        expected_failed_case_ids = tuple(
            sorted(
                {
                    attempt.case_id
                    for attempt in self.attempts
                    if _attempt_failed_release_qualification(attempt, self.applicability_profile)
                }
            )
        )
        if self.failed_case_ids != expected_failed_case_ids:
            raise ValueError("failed qualification cases must match their attempts")
        expected_input_sha256 = _qualification_input_sha256(
            operator=self.operator,
            operator_definition_sha256=self.operator_definition_sha256,
            corpus_sha256=self.corpus_sha256,
            applicability_profile=self.applicability_profile,
            thresholds=self.thresholds,
            attempts=self.attempts,
        )
        if self.qualification_input_sha256 != expected_input_sha256:
            raise ValueError("qualification input digest must match the report evidence")
        expected_status: QualificationStatus = (
            "thresholds_met" if all(gate.passed for gate in self.gates) else "blocked"
        )
        if self.status != expected_status:
            raise ValueError("qualification status must match its gates")
        if self.failed_case_ids != tuple(sorted(set(self.failed_case_ids))):
            raise ValueError("failed qualification case identifiers must be sorted and unique")
        expected_report_id = _report_id(self.model_dump(mode="json", exclude={"report_id"}))
        if self.report_id != expected_report_id:
            raise ValueError("qualification report ID must match its canonical content")
        return self


class AugmentationQualificationReplay(_StrictQualificationModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    baseline_report_id: str = Field(pattern=_REPORT_ID_PATTERN)
    current_report: AugmentationQualificationReport
    recurring_failure_case_ids: tuple[str, ...]
    resolved_failure_case_ids: tuple[str, ...]
    new_failure_case_ids: tuple[str, ...]


def augmentation_qualification_thresholds(
    profile: QualificationApplicabilityProfile,
) -> AugmentationQualificationThresholds:
    return AugmentationQualificationThresholds(
        minimum_applicability_rate=0.8 if profile == "broad" else 0.1,
        minimum_generation_success_rate=0.9,
        minimum_transformation_strength_rate=0.8,
        minimum_meaning_preservation_rate=1.0,
        minimum_realism_rate=0.8,
        maximum_rejection_rate=0.2,
        minimum_repeatability_rate=0.8,
        minimum_human_reviewed_cases=1,
    )


def augmentation_operator_definition_sha256(operator: DatasetAugmentationOperator) -> str:
    return _canonical_sha256(operator.model_dump(mode="json"))


def create_augmentation_qualification_report(
    *,
    operator: DatasetAugmentationOperator,
    corpus: AugmentationQualificationCorpus,
    applicability_profile: QualificationApplicabilityProfile,
    attempts: tuple[AugmentationQualificationAttempt, ...],
    thresholds: AugmentationQualificationThresholds | None = None,
) -> AugmentationQualificationReport:
    if applicability_profile != operator.applicability_profile:
        raise ValueError("qualification profile must match the operator applicability contract")
    selected_thresholds = thresholds or augmentation_qualification_thresholds(applicability_profile)
    attempts = tuple(sorted(attempts, key=lambda item: (item.case_id, item.repetition)))
    _validate_attempt_coverage(corpus, attempts)
    metrics = _qualification_metrics(attempts)
    gates = _qualification_gates(
        applicability_profile=applicability_profile,
        thresholds=selected_thresholds,
        metrics=metrics,
        has_meaning_change=any(
            attempt.meaning_preserved is False
            or (attempt.human_review is not None and not attempt.human_review.meaning_preserved)
            for attempt in attempts
        ),
    )
    failed_case_ids = tuple(
        sorted(
            {
                attempt.case_id
                for attempt in attempts
                if _attempt_failed_release_qualification(attempt, applicability_profile)
            }
        )
    )
    operator_reference = AugmentationQualificationOperatorReference(
        id=operator.id,
        version=operator.version,
    )
    operator_definition_sha256 = augmentation_operator_definition_sha256(operator)
    qualification_input_sha256 = _qualification_input_sha256(
        operator=operator_reference,
        operator_definition_sha256=operator_definition_sha256,
        corpus_sha256=corpus.sha256,
        applicability_profile=applicability_profile,
        thresholds=selected_thresholds,
        attempts=attempts,
    )
    status: QualificationStatus = (
        "thresholds_met" if all(gate.passed for gate in gates) else "blocked"
    )
    report_content = {
        "schema_version": "1.0.0",
        "operator": operator_reference.model_dump(mode="json"),
        "operator_definition_sha256": operator_definition_sha256,
        "corpus_name": corpus.name,
        "corpus_version": corpus.version,
        "corpus_sha256": corpus.sha256,
        "applicability_profile": applicability_profile,
        "thresholds": selected_thresholds.model_dump(mode="json"),
        "qualification_input_sha256": qualification_input_sha256,
        "evidence_status": "caller_supplied_unverified",
        "status": status,
        "metrics": metrics.model_dump(mode="json"),
        "gates": [gate.model_dump(mode="json") for gate in gates],
        "attempts": [attempt.model_dump(mode="json") for attempt in attempts],
        "failed_case_ids": list(failed_case_ids),
    }
    return AugmentationQualificationReport(
        report_id=_report_id(report_content),
        operator=operator_reference,
        operator_definition_sha256=operator_definition_sha256,
        corpus_name=corpus.name,
        corpus_version=corpus.version,
        corpus_sha256=corpus.sha256,
        applicability_profile=applicability_profile,
        thresholds=selected_thresholds,
        qualification_input_sha256=qualification_input_sha256,
        evidence_status="caller_supplied_unverified",
        status=status,
        metrics=metrics,
        gates=gates,
        attempts=attempts,
        failed_case_ids=failed_case_ids,
    )


def replay_augmentation_qualification(
    baseline: AugmentationQualificationReport,
    *,
    operator: DatasetAugmentationOperator,
    corpus: AugmentationQualificationCorpus,
    attempts: tuple[AugmentationQualificationAttempt, ...],
) -> AugmentationQualificationReplay:
    expected_reference = (operator.id, operator.version)
    baseline_reference = (baseline.operator.id, baseline.operator.version)
    if baseline_reference != expected_reference:
        raise ValueError("operator version changed; the qualification baseline is stale")
    if baseline.operator_definition_sha256 != augmentation_operator_definition_sha256(operator):
        raise ValueError("operator definition changed without a new qualification report")
    if baseline.corpus_sha256 != corpus.sha256:
        raise ValueError("qualification corpus changed; the qualification baseline is stale")
    current = create_augmentation_qualification_report(
        operator=operator,
        corpus=corpus,
        applicability_profile=baseline.applicability_profile,
        thresholds=baseline.thresholds,
        attempts=attempts,
    )
    baseline_failures = set(baseline.failed_case_ids)
    current_failures = set(current.failed_case_ids)
    return AugmentationQualificationReplay(
        baseline_report_id=baseline.report_id,
        current_report=current,
        recurring_failure_case_ids=tuple(sorted(baseline_failures & current_failures)),
        resolved_failure_case_ids=tuple(sorted(baseline_failures - current_failures)),
        new_failure_case_ids=tuple(sorted(current_failures - baseline_failures)),
    )


def load_augmentation_qualification_corpus(
    path: str | Path,
) -> AugmentationQualificationCorpus:
    encoded = _read_bounded_file(Path(path))
    _validate_json_object(encoded)
    return AugmentationQualificationCorpus.model_validate_json(encoded)


def load_augmentation_qualification_report(
    path: str | Path,
) -> AugmentationQualificationReport:
    encoded = _read_bounded_file(Path(path))
    _validate_json_object(encoded)
    return AugmentationQualificationReport.model_validate_json(encoded)


def write_augmentation_qualification_report(
    path: str | Path,
    report: AugmentationQualificationReport,
) -> None:
    destination = Path(path)
    encoded = f"{report.model_dump_json(indent=2)}\n".encode()
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _validate_attempt_coverage(
    corpus: AugmentationQualificationCorpus,
    attempts: tuple[AugmentationQualificationAttempt, ...],
) -> None:
    corpus_cases = {case.record.id: case for case in corpus.cases}
    corpus_case_ids = set(corpus_cases)
    attempted_case_ids = {attempt.case_id for attempt in attempts}
    if attempted_case_ids != corpus_case_ids:
        raise ValueError("qualification attempts must cover every corpus case exactly")
    attempts_by_case: dict[str, list[int]] = defaultdict(list)
    for attempt in attempts:
        if corpus_cases[attempt.case_id].segment != attempt.segment:
            raise ValueError("qualification attempt segment must match its corpus case")
        attempts_by_case[attempt.case_id].append(attempt.repetition)
    expected_repetitions: tuple[int, ...] | None = None
    for case_id in sorted(attempts_by_case):
        repetitions = tuple(sorted(attempts_by_case[case_id]))
        if len(repetitions) != len(set(repetitions)):
            raise ValueError("qualification case repetitions must be unique")
        if len(repetitions) < 2 or repetitions != tuple(range(1, len(repetitions) + 1)):
            raise ValueError("qualification cases require ordered repeat observations")
        if expected_repetitions is None:
            expected_repetitions = repetitions
        elif repetitions != expected_repetitions:
            raise ValueError("qualification cases must use the same repetitions")


def _qualification_metrics(
    attempts: tuple[AugmentationQualificationAttempt, ...],
) -> AugmentationQualificationMetrics:
    applicable_attempts = tuple(attempt for attempt in attempts if attempt.applicable)
    generated_attempts = tuple(
        attempt for attempt in applicable_attempts if attempt.generation_succeeded
    )
    reviewed_attempts = tuple(
        attempt for attempt in generated_attempts if attempt.human_review is not None
    )
    attempts_by_case: dict[str, list[AugmentationQualificationAttempt]] = defaultdict(list)
    for attempt in attempts:
        attempts_by_case[attempt.case_id].append(attempt)
    repeatable_cases = sum(
        len({_repeatability_signature(attempt) for attempt in case_attempts}) == 1
        for case_attempts in attempts_by_case.values()
    )
    reviewed_case_count = len({attempt.case_id for attempt in reviewed_attempts})
    return AugmentationQualificationMetrics(
        applicability=_rate(sum(attempt.applicable for attempt in attempts), len(attempts)),
        segment_applicability=tuple(
            AugmentationQualificationSegmentRate(
                segment=segment,
                applicability=_rate(
                    sum(attempt.applicable for attempt in attempts if attempt.segment == segment),
                    sum(attempt.segment == segment for attempt in attempts),
                ),
            )
            for segment in _CORPUS_SEGMENTS
        ),
        generation_success=_rate(len(generated_attempts), len(applicable_attempts)),
        transformation_strength=_rate(
            sum(attempt.transformation_strong is True for attempt in generated_attempts),
            len(generated_attempts),
        ),
        meaning_preservation=_rate(
            sum(
                attempt.meaning_preserved is True
                and (attempt.human_review is None or attempt.human_review.meaning_preserved)
                for attempt in generated_attempts
            ),
            len(generated_attempts),
        ),
        realism=_rate(
            sum(
                attempt.human_review is not None and attempt.human_review.realistic
                for attempt in reviewed_attempts
            ),
            len(reviewed_attempts),
        ),
        rejection=_rate(
            sum(attempt.rejected is True for attempt in generated_attempts),
            len(generated_attempts),
        ),
        repeatability=_rate(repeatable_cases, len(attempts_by_case)),
        human_reviewed_case_count=reviewed_case_count,
    )


def _qualification_gates(
    *,
    applicability_profile: QualificationApplicabilityProfile,
    thresholds: AugmentationQualificationThresholds,
    metrics: AugmentationQualificationMetrics,
    has_meaning_change: bool,
) -> tuple[AugmentationQualificationGate, ...]:
    applicability_rates = (
        tuple(
            (f"segment:{segment_rate.segment}", segment_rate.applicability.rate)
            for segment_rate in metrics.segment_applicability
        )
        if applicability_profile == "broad"
        else (("corpus", metrics.applicability.rate),)
    )
    gates = [
        _gate(
            "applicability",
            scope,
            observed,
            thresholds.minimum_applicability_rate,
            "at_least",
        )
        for scope, observed in applicability_rates
    ]
    gates.extend(
        (
            _gate(
                "generation_success",
                "applicable_attempts",
                metrics.generation_success.rate,
                thresholds.minimum_generation_success_rate,
                "at_least",
            ),
            _gate(
                "transformation_strength",
                "generated_attempts",
                metrics.transformation_strength.rate,
                thresholds.minimum_transformation_strength_rate,
                "at_least",
            ),
            _gate(
                "meaning_preservation",
                "generated_attempts",
                metrics.meaning_preservation.rate,
                thresholds.minimum_meaning_preservation_rate,
                "at_least",
            ),
            _gate(
                "meaning_preservation",
                "meaning_changes",
                int(has_meaning_change),
                0,
                "at_most",
            ),
            _gate(
                "realism",
                "human_reviewed_attempts",
                metrics.realism.rate,
                thresholds.minimum_realism_rate,
                "at_least",
            ),
            _gate(
                "realism",
                "human_reviewed_cases",
                metrics.human_reviewed_case_count,
                thresholds.minimum_human_reviewed_cases,
                "at_least",
            ),
            _gate(
                "rejection",
                "generated_attempts",
                metrics.rejection.rate,
                thresholds.maximum_rejection_rate,
                "at_most",
            ),
            _gate(
                "repeatability",
                "corpus_cases",
                metrics.repeatability.rate,
                thresholds.minimum_repeatability_rate,
                "at_least",
            ),
        )
    )
    return tuple(gates)


def _attempt_failed_release_qualification(
    attempt: AugmentationQualificationAttempt,
    profile: QualificationApplicabilityProfile,
) -> bool:
    if not attempt.applicable:
        return profile == "broad"
    if not attempt.generation_succeeded:
        return True
    return bool(
        not attempt.transformation_strong
        or not attempt.meaning_preserved
        or attempt.rejected
        or (
            attempt.human_review is not None
            and (not attempt.human_review.meaning_preserved or not attempt.human_review.realistic)
        )
    )


def _repeatability_signature(attempt: AugmentationQualificationAttempt) -> tuple[object, ...]:
    return (
        attempt.applicable,
        attempt.generation_succeeded,
        attempt.transformation_strong,
        attempt.meaning_preserved,
        attempt.rejected,
    )


def _rate(passed_count: int, assessed_count: int) -> AugmentationQualificationRate:
    return AugmentationQualificationRate(
        passed_count=passed_count,
        assessed_count=assessed_count,
        rate=passed_count / assessed_count if assessed_count else None,
    )


def _gate(
    dimension: QualificationDimension,
    scope: str,
    observed: float | int | None,
    threshold: float | int,
    comparison: QualificationGateComparison,
) -> AugmentationQualificationGate:
    if observed is None:
        passed = False
    elif comparison == "at_least":
        passed = observed >= threshold
    else:
        passed = observed <= threshold
    return AugmentationQualificationGate(
        dimension=dimension,
        scope=scope,
        comparison=comparison,
        observed=observed,
        threshold=threshold,
        passed=passed,
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _report_id(content: object) -> str:
    return f"ulaq_v1_{_canonical_sha256(content)}"


def _qualification_input_sha256(
    *,
    operator: AugmentationQualificationOperatorReference,
    operator_definition_sha256: str,
    corpus_sha256: str,
    applicability_profile: QualificationApplicabilityProfile,
    thresholds: AugmentationQualificationThresholds,
    attempts: tuple[AugmentationQualificationAttempt, ...],
) -> str:
    return _canonical_sha256(
        {
            "operator": operator.model_dump(mode="json"),
            "operator_definition_sha256": operator_definition_sha256,
            "corpus_sha256": corpus_sha256,
            "applicability_profile": applicability_profile,
            "thresholds": thresholds.model_dump(mode="json"),
            "attempts": [attempt.model_dump(mode="json") for attempt in attempts],
        }
    )


def _read_bounded_file(path: Path) -> bytes:
    try:
        path_metadata = path.lstat()
        if not stat.S_ISREG(path_metadata.st_mode):
            raise ValueError("qualification file is missing or exceeds the size limit")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        with os.fdopen(descriptor, "rb") as input_file:
            opened_metadata = os.fstat(input_file.fileno())
            if (
                not stat.S_ISREG(opened_metadata.st_mode)
                or (path_metadata.st_dev, path_metadata.st_ino)
                != (opened_metadata.st_dev, opened_metadata.st_ino)
                or opened_metadata.st_size > _MAXIMUM_QUALIFICATION_FILE_BYTES
            ):
                raise ValueError("qualification file is missing or exceeds the size limit")
            encoded = input_file.read(_MAXIMUM_QUALIFICATION_FILE_BYTES + 1)
    except OSError:
        raise ValueError("qualification file could not be read") from None
    if len(encoded) > _MAXIMUM_QUALIFICATION_FILE_BYTES:
        raise ValueError("qualification file exceeds the size limit")
    return encoded


def _validate_json_object(encoded: bytes) -> None:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("qualification file contains duplicate object keys")
            result[key] = value
        return result

    try:
        value = json.loads(encoded, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("qualification file contains invalid JSON") from None
    if not isinstance(value, dict):
        raise ValueError("qualification file must contain a JSON object")
