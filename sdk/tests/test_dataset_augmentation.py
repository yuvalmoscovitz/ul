from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import pytest
from ul.dataset_augmentation import DatasetAugmentationEngine
from ul_core.dataset import (
    CommunicationAct,
    EvidenceReference,
    InteractionRecord,
    ObservedOutcome,
    RequestUnit,
    SemanticFactor,
    SemanticFrame,
    SemanticRelation,
    UserInputRecord,
)

pytestmark = pytest.mark.asyncio


def evidence(source: Literal["input", "output"]) -> tuple[EvidenceReference, ...]:
    return (
        EvidenceReference(
            source=source,
            json_pointer=f"/raw_{'input' if source == 'input' else 'observed_output'}",
            text_quote=None,
        ),
    )


def source_record(identifier: str = "source") -> InteractionRecord:
    return InteractionRecord(
        id=identifier,
        raw_input="Transfer 100 to Alice, then tell me the balance.",
        raw_observed_output={"transfer": "completed", "balance": 900},
    )


def source_frame(record: InteractionRecord, *, identifier_prefix: str = "source") -> SemanticFrame:
    amount = SemanticFactor(
        id=f"{identifier_prefix}:amount",
        evidence=evidence("input"),
        confidence=1,
        status="explicit",
        kind="money",
        role="amount",
        value=100,
    )
    recipient = SemanticFactor(
        id=f"{identifier_prefix}:recipient",
        evidence=evidence("input"),
        confidence=1,
        status="explicit",
        kind="entity",
        role="recipient",
        value="Alice",
    )
    transfer = RequestUnit(
        id=f"{identifier_prefix}:transfer",
        evidence=evidence("input"),
        confidence=1,
        status="explicit",
        mode="act",
        predicate="transfer",
        factor_ids=(amount.id, recipient.id),
    )
    balance = RequestUnit(
        id=f"{identifier_prefix}:balance",
        evidence=evidence("input"),
        confidence=1,
        status="explicit",
        mode="ask",
        predicate="report_balance",
    )
    return SemanticFrame(
        interaction_id=record.id,
        request_units=(transfer, balance),
        factors=(amount, recipient),
        relations=(
            SemanticRelation(
                id=f"{identifier_prefix}:sequence",
                evidence=evidence("input"),
                confidence=1,
                status="explicit",
                kind="sequence",
                source_ids=(transfer.id,),
                target_ids=(balance.id,),
            ),
        ),
        communication_acts=(
            CommunicationAct(
                id=f"{identifier_prefix}:compound",
                evidence=evidence("input"),
                confidence=1,
                status="explicit",
                kind="compound_request",
                factor_ids=(amount.id, recipient.id),
                attributes={"connector": "then"},
            ),
        ),
        outcomes=(
            ObservedOutcome(
                id=f"{identifier_prefix}:transfer_outcome",
                evidence=evidence("output"),
                confidence=1,
                status="observed",
                request_unit_ids=(transfer.id,),
                position=0,
                kind="action",
                predicate="transfer",
                fields={"amount": 100, "recipient": "Alice"},
            ),
            ObservedOutcome(
                id=f"{identifier_prefix}:balance_outcome",
                evidence=evidence("output"),
                confidence=1,
                status="observed",
                request_unit_ids=(balance.id,),
                position=1,
                kind="answer",
                predicate="report_balance",
                fields={"balance": 900},
            ),
        ),
        extractor_version="test",
    )


class DeterministicSemanticModel:
    def __init__(
        self,
        source_frames: dict[str, SemanticFrame],
        candidate_frame: SemanticFrame | None = None,
    ) -> None:
        self.source_frames = source_frames
        self.candidate_frame = candidate_frame
        self.deconstructed_records: list[InteractionRecord | UserInputRecord] = []
        self.rendered_inputs: list[str] = []

    async def deconstruct(
        self,
        record: InteractionRecord | UserInputRecord,
        reference_frame: SemanticFrame | None = None,
    ) -> SemanticFrame:
        self.deconstructed_records.append(record)
        if record.id in self.source_frames:
            assert reference_frame is None
            return self.source_frames[record.id]
        assert self.candidate_frame is not None
        return self.candidate_frame.model_copy(update={"interaction_id": record.id})

    async def render(
        self,
        raw_input: str,
        instruction: str,
    ) -> str:
        self.rendered_inputs.append(raw_input)
        assert "without changing" in instruction
        return "Please transfer 100 to Alice and then report my balance."


async def test_engine_rephrases_and_independently_validates_full_semantics() -> None:
    record = source_record()
    original_frame = source_frame(record)
    candidate_frame = source_frame(record, identifier_prefix="candidate").model_copy(
        update={"outcomes": ()}
    )
    model = DeterministicSemanticModel({record.id: original_frame}, candidate_frame)

    result = await DatasetAugmentationEngine(model, model).augment((record,))

    assert result.source_frames == (original_frame,)
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.source_interaction_id == record.id
    assert candidate.operator_id == "surface.rephrase"
    assert candidate.passed
    assert candidate.expected_input_frame.outcomes == ()
    assert candidate.expected_input_frame.metadata == {}
    assert model.rendered_inputs == [record.raw_input]
    assert isinstance(model.deconstructed_records[1], UserInputRecord)
    assert not isinstance(model.deconstructed_records[1], InteractionRecord)


@pytest.mark.parametrize("drift", ["request_order", "relation", "communication", "factor"])
async def test_engine_rejects_each_kind_of_semantic_drift(drift: str) -> None:
    record = source_record()
    original_frame = source_frame(record)
    candidate_frame = source_frame(record, identifier_prefix="candidate").model_copy(
        update={"outcomes": ()}
    )
    if drift == "request_order":
        candidate_frame = candidate_frame.model_copy(
            update={"request_units": tuple(reversed(candidate_frame.request_units))}
        )
    elif drift == "relation":
        candidate_frame = candidate_frame.model_copy(update={"relations": ()})
    elif drift == "communication":
        changed_act = candidate_frame.communication_acts[0].model_copy(
            update={"attributes": {"connector": "before"}}
        )
        candidate_frame = candidate_frame.model_copy(update={"communication_acts": (changed_act,)})
    else:
        changed_factor = candidate_frame.factors[0].model_copy(update={"value": 500})
        candidate_frame = candidate_frame.model_copy(
            update={"factors": (changed_factor, candidate_frame.factors[1])}
        )
    model = DeterministicSemanticModel({record.id: original_frame}, candidate_frame)

    result = await DatasetAugmentationEngine(model, model).augment((record,))

    assert not result.candidates[0].passed
    assert result.candidates[0].failure_reasons


async def test_engine_filters_output_only_semantics_before_rendering() -> None:
    record = source_record()
    original_frame = source_frame(record)
    output_factor = SemanticFactor(
        id="receipt",
        evidence=evidence("output"),
        confidence=1,
        status="observed",
        kind="identifier",
        role="receipt",
        value="R-123",
    )
    mixed_evidence_factor = original_frame.factors[0].model_copy(
        update={
            "evidence": (
                *original_frame.factors[0].evidence,
                *evidence("output"),
            )
        }
    )
    original_frame = SemanticFrame.model_validate(
        {
            **original_frame.model_dump(mode="python"),
            "factors": (mixed_evidence_factor, original_frame.factors[1], output_factor),
            "metadata": {"provider_id": "private"},
        }
    )
    candidate_frame = source_frame(record, identifier_prefix="candidate").model_copy(
        update={"outcomes": ()}
    )
    model = DeterministicSemanticModel({record.id: original_frame}, candidate_frame)

    result = await DatasetAugmentationEngine(model, model).augment((record,))

    rendered_frame = result.candidates[0].expected_input_frame
    assert {factor.role for factor in rendered_frame.factors} == {"amount", "recipient"}
    assert rendered_frame.metadata == {}
    assert all(
        evidence_reference.source == "input"
        for element in (
            *rendered_frame.request_units,
            *rendered_frame.factors,
            *rendered_frame.relations,
            *rendered_frame.communication_acts,
        )
        for evidence_reference in element.evidence
    )


async def test_engine_bounds_iterables_before_any_model_call() -> None:
    model = DeterministicSemanticModel({})
    engine = DatasetAugmentationEngine(model, model)
    consumed = 0

    def records() -> Iterator[InteractionRecord]:
        nonlocal consumed
        while True:
            consumed += 1
            yield source_record(str(consumed))

    with pytest.raises(ValueError, match="record count exceeds"):
        await engine.augment(records(), max_records=3)

    assert consumed == 4
    assert model.deconstructed_records == []


async def test_engine_rejects_invalid_sources_before_any_model_call() -> None:
    record = source_record()
    model = DeterministicSemanticModel({})
    engine = DatasetAugmentationEngine(model, model)

    with pytest.raises(ValueError, match="between 1 and 100"):
        await engine.augment((), max_records=101)
    with pytest.raises(ValueError, match="identifiers must be unique"):
        await engine.augment((record, record))
    assert model.deconstructed_records == []


async def test_engine_skips_semantically_ambiguous_nodes() -> None:
    record = source_record()
    frame = source_frame(record)
    duplicate_amount = frame.factors[0].model_copy(update={"id": "duplicate-amount"})
    frame = SemanticFrame.model_validate(
        {
            **frame.model_dump(mode="python"),
            "factors": (*frame.factors, duplicate_amount),
        }
    )
    model = DeterministicSemanticModel({record.id: frame})

    result = await DatasetAugmentationEngine(model, model).augment((record,))

    assert result.candidates == ()
    assert model.rendered_inputs == []


async def test_engine_skips_unresolved_source_and_rejects_unresolved_candidate() -> None:
    record = source_record()
    frame = source_frame(record)
    unresolved_source_factor = frame.factors[0].model_copy(update={"status": "unresolved"})
    unresolved_source = frame.model_copy(
        update={"factors": (unresolved_source_factor, frame.factors[1])}
    )
    source_model = DeterministicSemanticModel({record.id: unresolved_source})

    skipped = await DatasetAugmentationEngine(source_model, source_model).augment((record,))

    assert skipped.candidates == ()
    candidate_frame = source_frame(record, identifier_prefix="candidate").model_copy(
        update={"outcomes": ()}
    )
    unresolved_candidate_factor = candidate_frame.factors[0].model_copy(
        update={"status": "unresolved.low_confidence"}
    )
    unresolved_candidate = candidate_frame.model_copy(
        update={
            "factors": (unresolved_candidate_factor, candidate_frame.factors[1]),
        }
    )
    candidate_model = DeterministicSemanticModel({record.id: frame}, unresolved_candidate)

    result = await DatasetAugmentationEngine(candidate_model, candidate_model).augment((record,))

    assert not result.candidates[0].passed
    assert "reparsed frame contains unresolved semantic elements" in (
        result.candidates[0].failure_reasons
    )
