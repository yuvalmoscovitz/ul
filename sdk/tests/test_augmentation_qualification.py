from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError
from ul.augmentation_qualification import (
    AugmentationQualificationAttempt,
    AugmentationQualificationCorpus,
    AugmentationQualificationHumanReview,
    AugmentationQualificationThresholds,
    create_augmentation_qualification_report,
    load_augmentation_qualification_corpus,
    load_augmentation_qualification_report,
    replay_augmentation_qualification,
    write_augmentation_qualification_report,
)
from ul.dataset_augmentation import (
    DatasetAugmentationOperator,
    builtin_dataset_augmentation_operators,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "augmentation_qualification_corpus.json"


def _attempts(
    corpus: AugmentationQualificationCorpus,
    operator: DatasetAugmentationOperator,
    *,
    profile: str,
) -> tuple[AugmentationQualificationAttempt, ...]:
    attempts: list[AugmentationQualificationAttempt] = []
    for case in corpus.cases:
        applicable = profile == "broad" or case.segment == "tool_oriented"
        for repetition in (1, 2):
            if not applicable:
                attempts.append(
                    AugmentationQualificationAttempt(
                        case_id=case.record.id,
                        segment=case.segment,
                        repetition=repetition,
                        applicable=False,
                        generation_succeeded=False,
                        failure_reasons=("operator is not applicable to this source",),
                    )
                )
                continue
            output_sha256 = hashlib.sha256(
                f"{operator.id}\0{operator.version}\0{case.record.id}\0{repetition}".encode()
            ).hexdigest()
            attempts.append(
                AugmentationQualificationAttempt(
                    case_id=case.record.id,
                    segment=case.segment,
                    repetition=repetition,
                    applicable=True,
                    generation_succeeded=True,
                    transformation_strong=True,
                    meaning_preserved=True,
                    rejected=False,
                    output_sha256=output_sha256,
                    human_review=(
                        AugmentationQualificationHumanReview(
                            criteria_version="1.0.0",
                            meaning_preserved=True,
                            realistic=True,
                        )
                        if repetition == 1
                        and (
                            (profile == "broad" and case.segment == "short")
                            or (profile == "conditional" and case.segment == "tool_oriented")
                        )
                        else None
                    ),
                )
            )
    return tuple(attempts)


def _replace_attempt(
    attempts: tuple[AugmentationQualificationAttempt, ...],
    *,
    case_id: str,
    repetition: int,
    replacement: AugmentationQualificationAttempt,
) -> tuple[AugmentationQualificationAttempt, ...]:
    return tuple(
        replacement if attempt.case_id == case_id and attempt.repetition == repetition else attempt
        for attempt in attempts
    )


def test_every_dataset_operator_version_produces_a_reproducible_report() -> None:
    corpus = load_augmentation_qualification_corpus(_FIXTURE)

    reports = []
    for operator in builtin_dataset_augmentation_operators():
        profile = operator.applicability_profile
        attempts = _attempts(corpus, operator, profile=profile)
        report = create_augmentation_qualification_report(
            operator=operator,
            corpus=corpus,
            applicability_profile=profile,
            attempts=attempts,
        )
        repeated_report = create_augmentation_qualification_report(
            operator=operator,
            corpus=corpus,
            applicability_profile=profile,
            attempts=tuple(reversed(attempts)),
        )

        assert report == repeated_report
        assert report.status == "thresholds_met"
        assert report.evidence_status == "caller_supplied_unverified"
        assert report.operator.id == operator.id
        assert report.operator.version == operator.version
        assert {gate.dimension for gate in report.gates} == {
            "applicability",
            "generation_success",
            "transformation_strength",
            "meaning_preservation",
            "realism",
            "rejection",
            "repeatability",
        }
        reports.append(report)

    assert {(report.operator.id, report.operator.version) for report in reports} == {
        (operator.id, operator.version) for operator in builtin_dataset_augmentation_operators()
    }


def test_broad_operator_must_meet_applicability_threshold_in_every_segment() -> None:
    corpus = load_augmentation_qualification_corpus(_FIXTURE)
    operator = builtin_dataset_augmentation_operators()[0]
    attempts = _attempts(corpus, operator, profile="broad")
    for repetition in (1, 2):
        attempts = _replace_attempt(
            attempts,
            case_id="short-cancel",
            repetition=repetition,
            replacement=AugmentationQualificationAttempt(
                case_id="short-cancel",
                segment="short",
                repetition=repetition,
                applicable=False,
                generation_succeeded=False,
                failure_reasons=("operator is not applicable to this source",),
            ),
        )

    report = create_augmentation_qualification_report(
        operator=operator,
        corpus=corpus,
        applicability_profile="broad",
        attempts=attempts,
    )

    short_gate = next(gate for gate in report.gates if gate.scope == "segment:short")
    assert not short_gate.passed
    assert report.status == "blocked"
    assert report.failed_case_ids == ("short-cancel",)


def test_any_automated_or_human_meaning_change_blocks_release() -> None:
    corpus = load_augmentation_qualification_corpus(_FIXTURE)
    operator = builtin_dataset_augmentation_operators()[0]
    attempts = _attempts(corpus, operator, profile="broad")
    original = next(
        attempt
        for attempt in attempts
        if attempt.case_id == "factual-shipment" and attempt.repetition == 1
    )
    attempts = _replace_attempt(
        attempts,
        case_id=original.case_id,
        repetition=original.repetition,
        replacement=original.model_copy(
            update={
                "meaning_preserved": False,
                "rejected": True,
                "failure_reasons": ("candidate changed the shipment count",),
            }
        ),
    )
    permissive_thresholds = AugmentationQualificationThresholds(
        minimum_applicability_rate=0,
        minimum_generation_success_rate=0,
        minimum_transformation_strength_rate=0,
        minimum_meaning_preservation_rate=0,
        minimum_realism_rate=0,
        maximum_rejection_rate=1,
        minimum_repeatability_rate=0,
        minimum_human_reviewed_cases=1,
    )

    report = create_augmentation_qualification_report(
        operator=operator,
        corpus=corpus,
        applicability_profile="broad",
        thresholds=permissive_thresholds,
        attempts=attempts,
    )

    meaning_change_gate = next(gate for gate in report.gates if gate.scope == "meaning_changes")
    assert not meaning_change_gate.passed
    assert report.status == "blocked"
    assert report.failed_case_ids == ("factual-shipment",)


def test_regression_replay_reports_recurring_resolved_and_new_failures() -> None:
    corpus = load_augmentation_qualification_corpus(_FIXTURE)
    operator = next(
        operator
        for operator in builtin_dataset_augmentation_operators()
        if operator.id == "input.tone.frustrated"
    )
    passing_attempts = _attempts(corpus, operator, profile="broad")
    weak_attempts = passing_attempts
    for repetition in (1, 2):
        original = next(
            attempt
            for attempt in weak_attempts
            if attempt.case_id == "short-cancel" and attempt.repetition == repetition
        )
        weak_attempts = _replace_attempt(
            weak_attempts,
            case_id=original.case_id,
            repetition=original.repetition,
            replacement=original.model_copy(
                update={
                    "transformation_strong": False,
                    "rejected": True,
                    "failure_reasons": ("candidate did not express frustration",),
                }
            ),
        )
    baseline = create_augmentation_qualification_report(
        operator=operator,
        corpus=corpus,
        applicability_profile="broad",
        attempts=weak_attempts,
    )

    recurring = replay_augmentation_qualification(
        baseline,
        operator=operator,
        corpus=corpus,
        attempts=weak_attempts,
    )
    resolved = replay_augmentation_qualification(
        baseline,
        operator=operator,
        corpus=corpus,
        attempts=passing_attempts,
    )

    assert recurring.recurring_failure_case_ids == ("short-cancel",)
    assert recurring.resolved_failure_case_ids == ()
    assert resolved.recurring_failure_case_ids == ()
    assert resolved.resolved_failure_case_ids == ("short-cancel",)
    assert resolved.new_failure_case_ids == ()


def test_operator_version_or_definition_change_invalidates_a_stale_report() -> None:
    corpus = load_augmentation_qualification_corpus(_FIXTURE)
    operator = builtin_dataset_augmentation_operators()[0]
    attempts = _attempts(corpus, operator, profile="broad")
    report = create_augmentation_qualification_report(
        operator=operator,
        corpus=corpus,
        applicability_profile="broad",
        attempts=attempts,
    )

    with pytest.raises(ValueError, match="version changed"):
        replay_augmentation_qualification(
            report,
            operator=operator.model_copy(update={"version": "1.0.1"}),
            corpus=corpus,
            attempts=attempts,
        )
    with pytest.raises(ValueError, match="definition changed"):
        replay_augmentation_qualification(
            report,
            operator=operator.model_copy(update={"instruction": "A changed instruction."}),
            corpus=corpus,
            attempts=attempts,
        )


def test_reports_round_trip_without_overwriting_prior_evidence(tmp_path: Path) -> None:
    corpus = load_augmentation_qualification_corpus(_FIXTURE)
    operator = builtin_dataset_augmentation_operators()[0]
    report = create_augmentation_qualification_report(
        operator=operator,
        corpus=corpus,
        applicability_profile="broad",
        attempts=_attempts(corpus, operator, profile="broad"),
    )
    destination = tmp_path / "input.surface.rephrase@1.0.0.json"

    write_augmentation_qualification_report(destination, report)

    assert load_augmentation_qualification_report(destination) == report
    with pytest.raises(FileExistsError):
        write_augmentation_qualification_report(destination, report)


def test_loaded_report_recomputes_derived_metrics_from_attempt_evidence(tmp_path: Path) -> None:
    corpus = load_augmentation_qualification_corpus(_FIXTURE)
    operator = builtin_dataset_augmentation_operators()[0]
    report = create_augmentation_qualification_report(
        operator=operator,
        corpus=corpus,
        applicability_profile="broad",
        attempts=_attempts(corpus, operator, profile="broad"),
    )
    tampered = report.model_dump(mode="json")
    tampered["metrics"]["applicability"]["passed_count"] = 0
    tampered["metrics"]["applicability"]["rate"] = 0.0
    destination = tmp_path / "tampered.json"
    destination.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ValidationError, match="metrics must match"):
        load_augmentation_qualification_report(destination)


def test_qualification_loader_rejects_symlinks(tmp_path: Path) -> None:
    destination = tmp_path / "corpus.json"
    destination.symlink_to(_FIXTURE)

    with pytest.raises(ValueError, match="missing or exceeds"):
        load_augmentation_qualification_corpus(destination)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes are not available")
def test_qualification_loader_rejects_named_pipes_without_blocking(tmp_path: Path) -> None:
    destination = tmp_path / "corpus.pipe"
    os.mkfifo(destination)

    with pytest.raises(ValueError, match="missing or exceeds"):
        load_augmentation_qualification_corpus(destination)


def test_corpus_requires_every_qualification_segment() -> None:
    corpus = load_augmentation_qualification_corpus(_FIXTURE)

    with pytest.raises(ValidationError, match="missing segments"):
        AugmentationQualificationCorpus(
            name=corpus.name,
            version=corpus.version,
            cases=tuple(case for case in corpus.cases if case.segment != "short"),
        )
