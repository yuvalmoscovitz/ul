from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import NoReturn

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
    DatasetAugmentationEngine,
    DatasetAugmentationOperator,
    builtin_dataset_augmentation_operators,
)
from ul_core.dataset import (
    CommunicationAct,
    EvidenceReference,
    InteractionRecord,
    ObservedOutcome,
    RequestUnit,
    SemanticFactor,
    SemanticFrame,
    UserInputRecord,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "augmentation_qualification_corpus.json"
_FRUSTRATED_REVIEW_FIXTURE = (
    Path(__file__).parent / "fixtures" / "frustrated_tone_human_review.json"
)


class QualificationSemanticFixture:
    def __init__(self, records: tuple[InteractionRecord, ...]) -> None:
        self.records = {record.id: record for record in records}

    async def deconstruct(
        self,
        record: InteractionRecord | UserInputRecord,
        reference_frame: SemanticFrame | None = None,
    ) -> SemanticFrame:
        if isinstance(record, InteractionRecord):
            return self._frame(record.id, record.raw_input, with_outcome=True)
        assert reference_frame is not None
        source_id = record.id.removesuffix(":input.tone.frustrated")
        source = self.records[source_id]
        prefix = "Ugh, "
        has_frustration_marker = record.raw_input.startswith(prefix)
        candidate_request = (
            record.raw_input.removeprefix(prefix) if has_frustration_marker else record.raw_input
        )
        frame = self._frame(record.id, candidate_request, with_outcome=False)
        if not has_frustration_marker:
            return frame
        frustrated = CommunicationAct(
            id=f"{record.id}:frustrated",
            evidence=(
                EvidenceReference(source="input", json_pointer="/raw_input", text_quote="Ugh"),
            ),
            confidence=1,
            status="explicit",
            kind="frustrated",
        )
        assert source.raw_input == reference_frame.factors[0].value
        return frame.model_copy(update={"communication_acts": (frustrated,)})

    @staticmethod
    def _frame(interaction_id: str, request_text: str, *, with_outcome: bool) -> SemanticFrame:
        request_factor = SemanticFactor(
            id=f"{interaction_id}:request-text",
            evidence=(
                EvidenceReference(
                    source="input", json_pointer="/raw_input", text_quote=request_text
                ),
            ),
            confidence=1,
            status="explicit",
            kind="text",
            role="complete_request",
            value=request_text,
        )
        request = RequestUnit(
            id=f"{interaction_id}:request",
            evidence=(
                EvidenceReference(
                    source="input", json_pointer="/raw_input", text_quote=request_text
                ),
            ),
            confidence=1,
            status="explicit",
            mode="act",
            predicate="execute_request",
            factor_ids=(request_factor.id,),
        )
        outcomes = (
            (
                ObservedOutcome(
                    id=f"{interaction_id}:outcome",
                    evidence=(
                        EvidenceReference(
                            source="output",
                            json_pointer="/raw_observed_output",
                            text_quote=None,
                        ),
                    ),
                    confidence=1,
                    status="observed",
                    request_unit_ids=(request.id,),
                    position=0,
                    kind="action",
                    predicate="execute_request",
                ),
            )
            if with_outcome
            else ()
        )
        return SemanticFrame(
            interaction_id=interaction_id,
            request_units=(request,),
            factors=(request_factor,),
            outcomes=outcomes,
            extractor_version="qualification-semantic-fixture",
        )

    async def render(
        self,
        raw_input: str,
        instruction: str,
        *,
        allow_temporary_value: bool = False,
    ) -> NoReturn:
        del raw_input, instruction, allow_temporary_value
        raise AssertionError("frustrated tone must use its deterministic renderer")


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


@pytest.mark.asyncio
async def test_frustrated_renderer_meets_release_thresholds_on_the_varied_corpus() -> None:
    corpus = load_augmentation_qualification_corpus(_FIXTURE)
    operator = next(
        operator
        for operator in builtin_dataset_augmentation_operators()
        if operator.id == "input.tone.frustrated"
    )
    semantic_fixture = QualificationSemanticFixture(tuple(case.record for case in corpus.cases))
    review_payload = json.loads(_FRUSTRATED_REVIEW_FIXTURE.read_text(encoding="utf-8"))
    reviews = {review["case_id"]: review for review in review_payload["reviews"]}
    assert set(reviews) == {case.record.id for case in corpus.cases}
    attempts: list[AugmentationQualificationAttempt] = []

    for repetition in (1, 2):
        result = await DatasetAugmentationEngine(
            semantic_fixture,
            semantic_fixture,
        ).augment(
            (case.record for case in corpus.cases),
            max_records=len(corpus.cases),
            operator_ids=(operator.id,),
        )
        assert result.skips == ()
        candidates = {candidate.source_interaction_id: candidate for candidate in result.candidates}
        for case in corpus.cases:
            candidate = candidates[case.record.id]
            review = reviews[case.record.id]
            exact_frustrated_render = candidate.augmented_input == f"Ugh, {case.record.raw_input}"
            assert exact_frustrated_render
            assert candidate.passed, candidate.failure_reasons
            assert candidate.augmented_input == review["reviewed_output"]
            attempts.append(
                AugmentationQualificationAttempt(
                    case_id=case.record.id,
                    segment=case.segment,
                    repetition=repetition,
                    applicable=True,
                    generation_succeeded=True,
                    transformation_strong=exact_frustrated_render,
                    meaning_preserved=candidate.passed,
                    rejected=not candidate.passed,
                    output_sha256=hashlib.sha256(candidate.augmented_input.encode()).hexdigest(),
                    human_review=AugmentationQualificationHumanReview(
                        criteria_version=review_payload["criteria_version"],
                        meaning_preserved=review["meaning_preserved"],
                        realistic=review["realistic"],
                    ),
                )
            )

    report = create_augmentation_qualification_report(
        operator=operator,
        corpus=corpus,
        applicability_profile=operator.applicability_profile,
        attempts=tuple(attempts),
    )

    assert len(report.attempts) == len(corpus.cases) * 2
    assert report.status == "thresholds_met"
    assert report.failed_case_ids == ()
    assert all(gate.passed for gate in report.gates)


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
