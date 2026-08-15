from __future__ import annotations

from typing import Literal

import pytest
from pydantic import ValidationError
from ul.dataset_augmentation import DatasetAugmentationEngine
from ul.dataset_evaluation import (
    DatasetEvaluationCase,
    DatasetEvaluationResult,
    DatasetEvaluationRunner,
)
from ul_core.dataset import (
    CommunicationAct,
    EvidenceReference,
    InteractionRecord,
    ObservedAgentOutput,
    ObservedOutcome,
    RenderedUserInput,
    RequestUnit,
    SemanticFactor,
    SemanticFrame,
    UserInputRecord,
)

pytestmark = pytest.mark.asyncio


def _evidence(source: Literal["input", "output"]) -> tuple[EvidenceReference, ...]:
    return (
        EvidenceReference(
            source=source,
            json_pointer=f"/raw_{'input' if source == 'input' else 'observed_output'}",
            text_quote=None,
        ),
    )


def _outcome(
    identifier: str,
    position: int,
    *,
    predicate: str = "transfer",
    kind: str = "action",
    fields: dict[str, object] | None = None,
) -> ObservedOutcome:
    return ObservedOutcome(
        id=identifier,
        evidence=_evidence("output"),
        confidence=1,
        status="observed",
        request_unit_ids=("request",),
        position=position,
        kind=kind,
        predicate=predicate,
        fields=fields or {},
    )


def _frame(
    interaction_id: str,
    outcomes: tuple[ObservedOutcome, ...],
) -> SemanticFrame:
    amount = SemanticFactor(
        id="amount",
        evidence=_evidence("input"),
        confidence=1,
        status="explicit",
        kind="money",
        role="amount",
        value=100,
    )
    recipient = SemanticFactor(
        id="recipient",
        evidence=_evidence("input"),
        confidence=1,
        status="explicit",
        kind="entity",
        role="recipient",
        value="Alice",
    )
    return SemanticFrame(
        interaction_id=interaction_id,
        request_units=(
            RequestUnit(
                id="request",
                evidence=_evidence("input"),
                confidence=1,
                status="explicit",
                mode="act",
                predicate="transfer",
                factor_ids=(amount.id, recipient.id),
            ),
        ),
        factors=(amount, recipient),
        communication_acts=(
            CommunicationAct(
                id="request_style",
                evidence=_evidence("input"),
                confidence=1,
                status="explicit",
                kind="direct_request",
            ),
        ),
        outcomes=outcomes,
        extractor_version="test",
    )


def _source() -> InteractionRecord:
    return InteractionRecord(
        id="source",
        raw_input="Transfer 100 to Alice.",
        raw_observed_output={"payment": {"amount": 100, "recipient": "Alice"}},
    )


def _source_outcomes() -> tuple[ObservedOutcome, ...]:
    return (
        _outcome(
            "source_transfer",
            0,
            fields={"amount": 100, "recipient": "Alice", "receipt_id": "receipt-1"},
        ),
        _outcome(
            "source_answer",
            1,
            kind="answer",
            predicate="confirmation",
            fields={"text": "Done"},
        ),
    )


class DeterministicSemanticPipeline:
    def __init__(self, observed_outcomes: tuple[ObservedOutcome, ...]) -> None:
        self.source_frame = _frame("source", _source_outcomes())
        self.observed_outcomes = observed_outcomes
        self.references: list[SemanticFrame | None] = []
        self.observed_records: list[InteractionRecord] = []

    async def deconstruct(
        self,
        record: InteractionRecord | UserInputRecord,
        reference_frame: SemanticFrame | None = None,
    ) -> SemanticFrame:
        self.references.append(reference_frame)
        if record.id == "source":
            return self.source_frame
        if isinstance(record, InteractionRecord):
            self.observed_records.append(record)
            return _frame(record.id, self.observed_outcomes)
        return _frame(record.id, ())

    async def render(self, raw_input: str, instruction: str) -> RenderedUserInput:
        if "frustration" in instruction:
            return RenderedUserInput(text=raw_input)
        return RenderedUserInput(
            text="Please transfer 100 to Alice.",
            metadata={"model": "deterministic", "seed": 7},
        )


class DeterministicTarget:
    def __init__(self) -> None:
        self.raw_inputs: list[str] = []

    async def execute(self, raw_input: str) -> ObservedAgentOutput:
        self.raw_inputs.append(raw_input)
        return ObservedAgentOutput(
            raw_output={"payment": {"amount": 100, "recipient": "Alice"}},
            metadata={"run_id": "run-1"},
        )


def _runner(
    observed_outcomes: tuple[ObservedOutcome, ...],
) -> tuple[DatasetEvaluationRunner, DeterministicSemanticPipeline, DeterministicTarget]:
    semantic_pipeline = DeterministicSemanticPipeline(observed_outcomes)
    target = DeterministicTarget()
    return (
        DatasetEvaluationRunner(
            DatasetAugmentationEngine(semantic_pipeline, semantic_pipeline),
            semantic_pipeline,
            target,
        ),
        semantic_pipeline,
        target,
    )


async def test_runner_executes_only_accepted_candidates_and_keeps_rejected_candidates() -> None:
    observed_outcomes = (
        _outcome(
            "observed_transfer",
            0,
            fields={"amount": 100, "recipient": "Alice", "receipt_id": "receipt-2"},
        ),
        _outcome(
            "observed_answer",
            1,
            kind="answer",
            predicate="confirmation",
            fields={"text": "Payment completed successfully"},
        ),
    )
    runner, semantic_pipeline, target = _runner(observed_outcomes)

    result = await runner.run(_source(), operator_ids=("surface.rephrase", "tone.frustrated"))

    assert len(result.cases) == 2
    accepted, rejected = result.cases
    assert accepted.candidate.passed
    assert accepted.verdict == "no_divergence"
    assert accepted.target_output is not None
    assert accepted.target_output.metadata == {"run_id": "run-1"}
    assert accepted.findings == ()
    assert not rejected.candidate.passed
    assert rejected.verdict == "augmentation_rejected"
    assert rejected.target_output is None
    assert rejected.observed_frame is None
    assert rejected.findings == ()
    assert target.raw_inputs == ["Please transfer 100 to Alice."]
    assert semantic_pipeline.references[-1] == semantic_pipeline.source_frame
    assert semantic_pipeline.references[-1].outcomes == _source_outcomes()
    assert semantic_pipeline.observed_records[0].raw_input == target.raw_inputs[0]
    assert (
        semantic_pipeline.observed_records[0].raw_observed_output
        == accepted.target_output.raw_output
    )
    assert DatasetEvaluationResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    ("observed_outcomes", "category", "severity"),
    [
        (
            (
                _outcome(
                    "first_transfer",
                    0,
                    fields={"amount": 100, "recipient": "Alice"},
                ),
                _outcome(
                    "second_transfer",
                    1,
                    fields={"amount": 100, "recipient": "Alice"},
                ),
            ),
            "duplicate_effect",
            "critical",
        ),
        (
            (
                _outcome(
                    "expected_transfer",
                    0,
                    fields={"amount": 100, "recipient": "Alice"},
                ),
                _outcome("email", 1, predicate="send_email", fields={"recipient": "Alice"}),
            ),
            "unexpected_effect",
            "critical",
        ),
        ((), "missing_effect", "high"),
        (
            (
                _outcome(
                    "changed_transfer",
                    0,
                    fields={"amount": 120, "recipient": "Alice", "receipt_id": "receipt-9"},
                ),
            ),
            "changed_grounded_effect_argument",
            "critical",
        ),
    ],
)
async def test_runner_explains_each_observable_action_divergence(
    observed_outcomes: tuple[ObservedOutcome, ...],
    category: str,
    severity: str,
) -> None:
    runner, _, _ = _runner(observed_outcomes)

    result = await runner.run(_source())

    assert len(result.cases[0].findings) == 1
    finding = result.cases[0].findings[0]
    assert finding.category == category
    assert finding.severity == severity
    assert finding.review_status == "needs_review"
    assert finding.message.startswith("Needs review:")
    assert result.cases[0].verdict == "divergence_needs_review"
    if category == "changed_grounded_effect_argument":
        assert finding.grounded_field_names == ("amount",)


async def test_case_model_rejects_inconsistent_execution_and_verdicts() -> None:
    runner, _, _ = _runner(_source_outcomes())
    result = await runner.run(_source(), operator_ids=("surface.rephrase", "tone.frustrated"))
    accepted_case = result.cases[0]
    rejected_candidate = result.cases[1].candidate

    with pytest.raises(ValidationError, match="only accepted candidates"):
        DatasetEvaluationCase(
            candidate=rejected_candidate,
            verdict="no_divergence",
            target_output=ObservedAgentOutput(raw_output={"payment": "completed"}),
            observed_frame=result.augmentation.source_frames[0],
        )
    with pytest.raises(ValidationError, match="case verdict"):
        DatasetEvaluationCase(
            candidate=rejected_candidate,
            verdict="no_divergence",
        )
    with pytest.raises(ValidationError, match="case verdict"):
        DatasetEvaluationCase(
            candidate=accepted_case.candidate,
            verdict="divergence_needs_review",
            target_output=accepted_case.target_output,
            observed_frame=accepted_case.observed_frame,
        )


async def test_runner_does_not_execute_without_an_observable_action_baseline() -> None:
    runner, semantic_pipeline, target = _runner(())
    semantic_pipeline.source_frame = _frame(
        "source",
        (
            _outcome(
                "answer",
                0,
                kind="answer",
                predicate="confirmation",
                fields={"text": "Done"},
            ),
        ),
    )

    with pytest.raises(ValueError, match="observable action outcome"):
        await runner.run(_source())

    assert target.raw_inputs == []
