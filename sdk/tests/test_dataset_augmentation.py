from __future__ import annotations

from collections.abc import Iterator
from typing import Literal, cast

import pytest
import ul.augmentations.dataset as dataset_augmentations
from pydantic import JsonValue
from ul.augmentations.dataset import (
    DatasetAugmentationEngine,
    DatasetAugmentationOperator,
    DatasetAugmentationResult,
    builtin_dataset_augmentation_operators,
)
from ul_core.dataset import (
    CommunicationAct,
    EvidenceReference,
    InteractionRecord,
    ObservedOutcome,
    RenderedUserInput,
    RequestUnit,
    SemanticAllowedSurfaceChange,
    SemanticDelta,
    SemanticEquivalenceAssessment,
    SemanticFactor,
    SemanticFrame,
    SemanticRelation,
    UserInputRecord,
)
from ul_core.prompts import prompt_provenance

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


def spreadsheet_list_frames(record: InteractionRecord) -> tuple[SemanticFrame, SemanticFrame]:
    operations: tuple[dict[str, JsonValue], ...] = (
        {"project": "API Gateway Upgrade", "status": "Completed"},
        {"project": "Mobile App Redesign", "status": "In Progress"},
    )
    source_operations = SemanticFactor(
        id="source:operations",
        evidence=evidence("input"),
        confidence=1,
        status="explicit",
        kind="list",
        role="update_operations",
        value=list(operations),
    )
    source_spreadsheet = source_operations.model_copy(
        update={
            "id": "source:spreadsheet",
            "kind": "identifier",
            "role": "target_spreadsheet",
            "value": "ss_status",
        }
    )
    source_worksheet = source_operations.model_copy(
        update={
            "id": "source:worksheet",
            "kind": "identifier",
            "role": "target_worksheet",
            "value": "ws_report",
        }
    )
    source_request = RequestUnit(
        id="source:request",
        evidence=evidence("input"),
        confidence=1,
        status="explicit",
        mode="act",
        predicate="update_spreadsheet",
        factor_ids=(source_spreadsheet.id, source_worksheet.id, source_operations.id),
    )
    source_relation = SemanticRelation(
        id="source:fulfills",
        evidence=evidence("input"),
        confidence=1,
        status="explicit",
        kind="fulfills",
        source_ids=(source_request.id,),
        target_ids=(source_operations.id,),
    )
    source = SemanticFrame(
        interaction_id=record.id,
        request_units=(source_request,),
        factors=(source_spreadsheet, source_worksheet, source_operations),
        relations=(
            source_relation,
            *(
                SemanticRelation(
                    id=f"source:fulfills:{factor.id}",
                    evidence=evidence("input"),
                    confidence=1,
                    status="explicit",
                    kind="fulfills",
                    source_ids=(source_request.id,),
                    target_ids=(factor.id,),
                )
                for factor in (source_spreadsheet, source_worksheet)
            ),
        ),
        outcomes=(
            ObservedOutcome(
                id="source:outcome",
                evidence=evidence("output"),
                confidence=1,
                status="observed",
                request_unit_ids=(source_request.id,),
                position=0,
                kind="action",
                predicate="update_spreadsheet",
                fields={"operation_count": 2},
            ),
        ),
        extractor_version="test",
    )
    candidate_operations = tuple(
        SemanticFactor(
            id=f"candidate:operation:{index}",
            evidence=evidence("input"),
            confidence=1,
            status="explicit",
            kind="list",
            role="update_operations",
            value=[operation],
        )
        for index, operation in enumerate(operations)
    )
    candidate_request = source_request.model_copy(
        update={
            "id": "candidate:request",
            "factor_ids": (
                "candidate:spreadsheet",
                "candidate:worksheet",
                *(factor.id for factor in candidate_operations),
            ),
        }
    )
    candidate = SemanticFrame(
        interaction_id=f"{record.id}:input.surface.punctuation_noise",
        request_units=(candidate_request,),
        factors=(
            source_spreadsheet.model_copy(update={"id": "candidate:spreadsheet"}),
            source_worksheet.model_copy(update={"id": "candidate:worksheet"}),
            *candidate_operations,
        ),
        relations=tuple(
            SemanticRelation(
                id=f"candidate:fulfills:{index}",
                evidence=evidence("input"),
                confidence=1,
                status="explicit",
                kind="fulfills",
                source_ids=(candidate_request.id,),
                target_ids=(factor.id,),
            )
            for index, factor in enumerate(candidate_operations)
        ),
        outcomes=(),
        extractor_version="test",
    )
    return source, candidate


class DeterministicSemanticModel:
    def __init__(
        self,
        source_frames: dict[str, SemanticFrame],
        candidate_frame: SemanticFrame | None = None,
        rendered_output: str = "Please transfer 100 to Alice and then report my balance.",
        invalid_candidate_operator_ids: frozenset[str] = frozenset(),
    ) -> None:
        self.source_frames = source_frames
        self.candidate_frame = candidate_frame
        self.rendered_output = rendered_output
        self.invalid_candidate_operator_ids = invalid_candidate_operator_ids
        self.deconstructed_records: list[InteractionRecord | UserInputRecord] = []
        self.rendered_inputs: list[str] = []
        self.rendered_instructions: list[str] = []
        self.temporary_value_permissions: list[bool] = []

    async def deconstruct(
        self,
        record: InteractionRecord | UserInputRecord,
        reference_frame: SemanticFrame | None = None,
    ) -> SemanticFrame:
        self.deconstructed_records.append(record)
        if record.id in self.source_frames:
            assert reference_frame is None
            return self.source_frames[record.id]
        if any(
            record.id.endswith(f":{operator_id}")
            for operator_id in self.invalid_candidate_operator_ids
        ):
            raise ValueError("provider validation detail must not escape")
        assert self.candidate_frame is not None
        return self.candidate_frame.model_copy(update={"interaction_id": record.id})

    async def render(
        self,
        raw_input: str,
        instruction: str,
        *,
        allow_temporary_value: bool = False,
    ) -> RenderedUserInput:
        self.rendered_inputs.append(raw_input)
        self.rendered_instructions.append(instruction)
        self.temporary_value_permissions.append(allow_temporary_value)
        return RenderedUserInput(
            text=self.rendered_output,
            metadata={"model": "test/model", "seed": 42},
        )


class DeterministicEquivalenceVerifier:
    def __init__(self, assessment: SemanticEquivalenceAssessment | None) -> None:
        self.assessment = assessment
        self.calls: list[tuple[str, str]] = []
        self.allowed_surface_changes: list[SemanticAllowedSurfaceChange] = []

    async def verify(
        self,
        source_input: str,
        candidate_input: str,
        *,
        allowed_surface_change: SemanticAllowedSurfaceChange = "none",
    ) -> SemanticEquivalenceAssessment:
        self.calls.append((source_input, candidate_input))
        self.allowed_surface_changes.append(allowed_surface_change)
        if self.assessment is None:
            raise ValueError("invalid model evidence")
        return self.assessment


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
    assert candidate.operator_id == "input.surface.rephrase"
    assert candidate.passed
    assert candidate.expected_input_frame.outcomes == ()
    assert candidate.expected_input_frame.metadata == {}
    assert model.rendered_inputs == [record.raw_input]
    assert isinstance(model.deconstructed_records[1], UserInputRecord)
    assert not isinstance(model.deconstructed_records[1], InteractionRecord)


async def test_engine_replaces_em_dashes_in_llm_generated_input() -> None:
    record = source_record()
    original_frame = source_frame(record)
    candidate_frame = source_frame(record, identifier_prefix="candidate").model_copy(
        update={"outcomes": ()}
    )
    model = DeterministicSemanticModel(
        {record.id: original_frame},
        candidate_frame,
        "Please transfer 100 to Alice—then report my balance.",
    )

    result = await DatasetAugmentationEngine(model, model).augment((record,))

    candidate = result.candidates[0]
    assert candidate.passed
    assert candidate.augmented_input == "Please transfer 100 to Alice then report my balance."
    reparsed_record = model.deconstructed_records[1]
    assert isinstance(reparsed_record, UserInputRecord)
    assert reparsed_record.raw_input == candidate.augmented_input


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
    assert result.skips[0].reason == (
        "Source factors 'source:amount' and 'duplicate-amount' have indistinguishable semantics; "
        "clarify their evidence or roles."
    )
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
    assert skipped.skips[0].reason == (
        "Source factor 'source:amount' is unresolved; clarify its evidence or improve semantic "
        "extraction."
    )
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


def behavior_candidate_frame(
    record: InteractionRecord, target_communication_kind: str
) -> SemanticFrame:
    frame = source_frame(record, identifier_prefix="candidate").model_copy(update={"outcomes": ()})
    target_act = CommunicationAct(
        id="candidate:declared-form",
        evidence=evidence("input"),
        confidence=1,
        status="explicit",
        kind=target_communication_kind,
    )
    return frame.model_copy(update={"communication_acts": (*frame.communication_acts, target_act)})


def self_correction_source_frame(record: InteractionRecord) -> SemanticFrame:
    frame = source_frame(record)
    amount = frame.factors[0].model_copy(
        update={
            "evidence": (
                EvidenceReference(source="input", json_pointer="/raw_input", text_quote="100"),
            )
        }
    )
    return frame.model_copy(update={"factors": (amount, frame.factors[1])})


def self_correction_candidate_frame(record: InteractionRecord) -> SemanticFrame:
    frame = source_frame(record, identifier_prefix="candidate").model_copy(update={"outcomes": ()})
    final_amount = frame.factors[0].model_copy(
        update={
            "evidence": (
                EvidenceReference(source="input", json_pointer="/raw_input", text_quote="100"),
            )
        }
    )
    provisional_amount = SemanticFactor(
        id="candidate:provisional-amount",
        evidence=(EvidenceReference(source="input", json_pointer="/raw_input", text_quote="110"),),
        confidence=1,
        status="superseded",
        kind="money",
        role="amount",
        value=110,
    )
    correction_evidence = (
        EvidenceReference(source="input", json_pointer="/raw_input", text_quote="110, sorry 100"),
    )
    correction_act = CommunicationAct(
        id="candidate:self-correction",
        evidence=correction_evidence,
        confidence=1,
        status="explicit",
        kind="self_correction",
        factor_ids=(provisional_amount.id, final_amount.id),
    )
    correction_relation = SemanticRelation(
        id="candidate:superseded-by",
        evidence=correction_evidence,
        confidence=1,
        status="explicit",
        kind="superseded_by",
        source_ids=(provisional_amount.id,),
        target_ids=(final_amount.id,),
    )
    return frame.model_copy(
        update={
            "factors": (final_amount, frame.factors[1], provisional_amount),
            "relations": (*frame.relations, correction_relation),
            "communication_acts": (*frame.communication_acts, correction_act),
        }
    )


async def test_builtin_operator_library_is_fixed_versioned_and_reviewable() -> None:
    operators = builtin_dataset_augmentation_operators()

    assert tuple(operator.id for operator in operators) == (
        "input.surface.rephrase",
        "input.surface.typing_noise",
        "input.surface.case_variation",
        "input.surface.punctuation_noise",
        "input.surface.grammar_error",
        "input.surface.fragmented_syntax",
        "input.surface.disfluency_repeat",
        "input.style.terse",
        "input.style.verbose",
        "input.tone.angry",
        "input.tone.argumentative",
        "input.intent.self_correction",
    )
    assert {operator.id: operator.version for operator in operators}[
        "input.intent.self_correction"
    ] == "1.1.0"
    assert {
        operator.version for operator in operators if operator.id != "input.intent.self_correction"
    } == {"1.0.0"}
    assert [operator.id for operator in operators if operator.generation_mechanism == "llm"] == [
        "input.surface.rephrase",
        "input.surface.grammar_error",
        "input.surface.fragmented_syntax",
        "input.style.terse",
        "input.style.verbose",
        "input.tone.angry",
        "input.tone.argumentative",
    ]
    assert [
        operator.id for operator in operators if operator.generation_mechanism == "deterministic"
    ] == [
        "input.surface.typing_noise",
        "input.surface.case_variation",
        "input.surface.punctuation_noise",
        "input.surface.disfluency_repeat",
        "input.intent.self_correction",
    ]
    assert [operator.id for operator in operators if operator.human_review_required] == [
        "input.intent.self_correction"
    ]
    assert [
        operator.id for operator in operators if operator.applicability_profile == "conditional"
    ] == [
        "input.surface.case_variation",
        "input.surface.punctuation_noise",
        "input.intent.self_correction",
    ]


async def test_operator_change_contract_rejects_impossible_target_states() -> None:
    with pytest.raises(ValueError, match="target communication kind"):
        DatasetAugmentationOperator(
            id="input.surface.rephrase",
            instruction="Rephrase naturally.",
            generation_mechanism="llm",
            allowed_change="surface_form_only",
            target_communication_kind="rephrase",
        )
    with pytest.raises(ValueError, match="target communication kind"):
        DatasetAugmentationOperator(
            id="input.style.terse",
            instruction="Make it terse.",
            generation_mechanism="llm",
            allowed_change="declared_communication_form",
        )
    with pytest.raises(ValueError, match="target communication kind"):
        DatasetAugmentationOperator(
            id="input.intent.self_correction",
            instruction="Add a correction.",
            generation_mechanism="llm",
            allowed_change="structured_self_correction",
        )


async def test_self_correction_accepts_one_structured_superseded_value() -> None:
    record = source_record()
    original_frame = self_correction_source_frame(record)
    candidate_frame = self_correction_candidate_frame(record)
    rendered_output = "Transfer 110, sorry 100 to Alice, then tell me the balance."
    model = DeterministicSemanticModel(
        {record.id: original_frame}, candidate_frame, rendered_output
    )

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.intent.self_correction",)
    )

    candidate = result.candidates[0]
    assert candidate.passed
    assert candidate.allowed_change == "structured_self_correction"
    assert candidate.augmented_input == rendered_output
    assert candidate.semantic_equivalence_assessment is None
    assert candidate.renderer_metadata["renderer"] == "deterministic"
    assert candidate.renderer_metadata["algorithm"] == "grounded_self_correction_insertion"
    assert model.rendered_inputs == []


@pytest.mark.parametrize(
    (
        "recorded_action_name",
        "recorded_opportunity_id",
        "colliding_request_id",
        "expected_to_pass",
    ),
    [
        ("salesforce.opportunity.update", "006003", False, True),
        ("salesforce.opportunity.update", "unrelated-opportunity", False, False),
        ("salesforce.opportunity.read", "006003", False, False),
        ("salesforce.opportunity.update", "006003", True, False),
    ],
)
async def test_self_correction_accepts_money_grounded_by_recorded_action_envelope(
    recorded_action_name: str,
    recorded_opportunity_id: str,
    colliding_request_id: bool,
    expected_to_pass: bool,
) -> None:
    raw_input = (
        "Update the amount on opportunity 006003 (DataStream Analytics License) "
        "to $45,000 in Salesforce."
    )
    recorded_action = {
        "action": recorded_action_name,
        "opportunity_id": recorded_opportunity_id,
        "amount": 45000.0,
    }
    record = InteractionRecord(
        id="salesforce-amount",
        raw_input=raw_input,
        raw_observed_output={"actions": [recorded_action], "correctness": {"passed": True}},
    )
    amount = SemanticFactor(
        id="source:amount",
        evidence=(
            EvidenceReference(source="input", json_pointer="/raw_input", text_quote="$45,000"),
        ),
        confidence=1,
        status="observed",
        kind="money",
        role="object",
        value="$45,000",
    )
    opportunity = SemanticFactor(
        id="source:opportunity",
        evidence=(
            EvidenceReference(source="input", json_pointer="/raw_input", text_quote="006003"),
        ),
        confidence=1,
        status="observed",
        kind="identifier",
        role="object",
        value="006003",
    )
    system = SemanticFactor(
        id="source:system",
        evidence=(
            EvidenceReference(source="input", json_pointer="/raw_input", text_quote="Salesforce"),
        ),
        confidence=1,
        status="observed",
        kind="entity",
        role="object",
        value="Salesforce",
    )
    request = RequestUnit(
        id="source:request",
        evidence=(
            EvidenceReference(source="input", json_pointer="/raw_input", text_quote=raw_input),
        ),
        confidence=1,
        status="observed",
        mode="act",
        predicate="update_opportunity_amount",
        factor_ids=(opportunity.id, amount.id, system.id),
    )
    source = SemanticFrame(
        interaction_id=record.id,
        request_units=(request,),
        factors=(opportunity, amount, system),
        outcomes=(
            ObservedOutcome(
                id="source:answer",
                evidence=evidence("output"),
                confidence=1,
                status="observed",
                position=0,
                kind="answer",
                predicate="returned_response",
                fields={"value": record.raw_observed_output},
            ),
        ),
        extractor_version="test",
    )
    candidate_opportunity = opportunity.model_copy(update={"id": "candidate:opportunity"})
    candidate_amount = amount.model_copy(update={"id": "candidate:amount", "value": 45000})
    candidate_request = request.model_copy(
        update={
            "id": (
                "candidate:self-correction:provisional"
                if colliding_request_id
                else "candidate:request"
            ),
            "factor_ids": (candidate_opportunity.id, candidate_amount.id),
        }
    )
    repair_evidence = (
        EvidenceReference(
            source="input",
            json_pointer="/raw_input",
            text_quote="$44,000, sorry $45,000",
        ),
    )
    candidate = SemanticFrame(
        interaction_id=f"{record.id}:input.intent.self_correction",
        request_units=(candidate_request,),
        factors=(candidate_opportunity, candidate_amount),
        communication_acts=(
            CommunicationAct(
                id="candidate:self-correction",
                evidence=repair_evidence,
                confidence=1,
                status="explicit",
                kind="self_correction",
            ),
        ),
        extractor_version="test",
    )
    rendered_output = raw_input.replace("$45,000", "$44,000, sorry $45,000")
    model = DeterministicSemanticModel({record.id: source}, candidate, rendered_output)

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.intent.self_correction",)
    )

    if expected_to_pass:
        assert result.candidates[0].passed
        assert result.candidates[0].augmented_input == rendered_output
        assert model.rendered_inputs == []
    elif result.candidates:
        assert not result.candidates[0].passed
    else:
        assert result.skips


def grounded_enum_self_correction_frames(
    *,
    prior_actions: list[dict[str, JsonValue]],
    final_action: dict[str, JsonValue] | None = None,
    request_predicate: str = "update_opportunity_stage",
) -> tuple[InteractionRecord, SemanticFrame, SemanticFrame, str]:
    raw_input = (
        "Opportunity 006001 is currently Negotiation/Review. Great news - it just closed! "
        "Please mark it as Closed Won in Salesforce."
    )
    if final_action is None:
        final_action = {
            "action": "salesforce.opportunity.update",
            "opportunity_id": "006001",
            "stage_name": "Closed Won",
        }
    record = InteractionRecord(
        id="salesforce-stage",
        raw_input=raw_input,
        raw_observed_output={"actions": [*prior_actions, final_action]},
    )
    opportunity = SemanticFactor(
        id="source:opportunity",
        evidence=(
            EvidenceReference(source="input", json_pointer="/raw_input", text_quote="006001"),
        ),
        confidence=1,
        status="observed",
        kind="identifier",
        role="identifier",
        value="006001",
    )
    opportunity_entity = SemanticFactor(
        id="source:opportunity-entity",
        evidence=(
            EvidenceReference(source="input", json_pointer="/raw_input", text_quote="Opportunity"),
        ),
        confidence=1,
        status="observed",
        kind="entity",
        role="object",
        value="opportunity",
    )
    stage = SemanticFactor(
        id="source:stage",
        evidence=(
            EvidenceReference(source="input", json_pointer="/raw_input", text_quote="Closed Won"),
        ),
        confidence=1,
        status="confirmed",
        kind="enum",
        role="stage_name",
        value="Closed Won",
    )
    request = RequestUnit(
        id="source:request",
        evidence=(
            EvidenceReference(source="input", json_pointer="/raw_input", text_quote=raw_input),
        ),
        confidence=1,
        status="observed",
        mode="act",
        predicate=request_predicate,
        factor_ids=(opportunity_entity.id, opportunity.id, stage.id),
    )
    source = SemanticFrame(
        interaction_id=record.id,
        request_units=(request,),
        factors=(opportunity_entity, opportunity, stage),
        outcomes=(
            ObservedOutcome(
                id="source:answer",
                evidence=evidence("output"),
                confidence=1,
                status="observed",
                position=0,
                kind="answer",
                predicate="returned_response",
                fields={"value": record.raw_observed_output},
            ),
        ),
        extractor_version="test",
    )
    candidate_opportunity = opportunity.model_copy(update={"id": "candidate:opportunity"})
    candidate_opportunity_entity = opportunity_entity.model_copy(
        update={"id": "candidate:opportunity-entity"}
    )
    candidate_stage = stage.model_copy(update={"id": "candidate:stage"})
    candidate_request = request.model_copy(
        update={
            "id": "candidate:request",
            "factor_ids": (
                candidate_opportunity_entity.id,
                candidate_opportunity.id,
                candidate_stage.id,
            ),
        }
    )
    candidate = SemanticFrame(
        interaction_id=f"{record.id}:input.intent.self_correction",
        request_units=(candidate_request,),
        factors=(candidate_opportunity_entity, candidate_opportunity, candidate_stage),
        communication_acts=(
            CommunicationAct(
                id="candidate:self-correction",
                evidence=(
                    EvidenceReference(
                        source="input",
                        json_pointer="/raw_input",
                        text_quote="Negotiation/Review, sorry Closed Won",
                    ),
                ),
                confidence=1,
                status="explicit",
                kind="self_correction",
            ),
        ),
        extractor_version="test",
    )
    rendered_output = raw_input.replace("Closed Won", "Negotiation/Review, sorry Closed Won")
    return record, source, candidate, rendered_output


async def test_self_correction_accepts_enum_grounded_by_prior_observed_action() -> None:
    prior_action: dict[str, JsonValue] = {
        "action": "salesforce.opportunity.read",
        "opportunity_id": "006001",
        "stage_name": "Negotiation/Review",
    }
    record, source, candidate_frame, rendered_output = grounded_enum_self_correction_frames(
        prior_actions=[prior_action]
    )
    model = DeterministicSemanticModel({record.id: source}, candidate_frame, rendered_output)

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.intent.self_correction",)
    )

    candidate = result.candidates[0]
    assert candidate.passed
    assert candidate.augmented_input == rendered_output
    assert candidate.renderer_metadata["self_correction_grounding"] == {
        "origin": "prior_observed_action",
        "field_name": "stage_name",
        "source_occurrences": 1,
        "prior_action_index": 0,
        "final_action_index": 1,
        "identifier_fields": ["opportunity_id"],
    }
    assert model.rendered_inputs == []


async def test_self_correction_accepts_enum_with_coarse_request_predicate() -> None:
    prior_action: dict[str, JsonValue] = {
        "action": "salesforce.opportunity.read",
        "opportunity_id": "006001",
        "stage_name": "Negotiation/Review",
    }
    record, source, candidate_frame, rendered_output = grounded_enum_self_correction_frames(
        prior_actions=[prior_action], request_predicate="update"
    )
    model = DeterministicSemanticModel({record.id: source}, candidate_frame, rendered_output)

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.intent.self_correction",)
    )

    assert result.candidates[0].passed
    assert result.candidates[0].augmented_input == rendered_output


async def test_self_correction_recovers_exact_factor_association_from_request_evidence() -> None:
    prior_action: dict[str, JsonValue] = {
        "action": "salesforce.opportunity.read",
        "opportunity_id": "006001",
        "stage_name": "Negotiation/Review",
    }
    record, source, candidate_frame, rendered_output = grounded_enum_self_correction_frames(
        prior_actions=[prior_action]
    )
    source_request = source.request_units[0].model_copy(update={"factor_ids": ()})
    unlinked_source = source.model_copy(update={"request_units": (source_request,)})
    model = DeterministicSemanticModel(
        {record.id: unlinked_source}, candidate_frame, rendered_output
    )

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.intent.self_correction",)
    )

    assert result.candidates[0].passed
    assert result.candidates[0].augmented_input == rendered_output


async def test_self_correction_does_not_recover_unrelated_entity_from_broad_evidence() -> None:
    prior_action: dict[str, JsonValue] = {
        "action": "salesforce.account.read",
        "account_id": "006001",
        "stage_name": "Negotiation/Review",
    }
    final_action: dict[str, JsonValue] = {
        "action": "salesforce.account.update",
        "account_id": "006001",
        "stage_name": "Closed Won",
    }
    record, source, candidate_frame, rendered_output = grounded_enum_self_correction_frames(
        prior_actions=[prior_action], final_action=final_action
    )
    raw_input = f"Account context. {record.raw_input}"
    account = SemanticFactor(
        id="source:account",
        evidence=(
            EvidenceReference(source="input", json_pointer="/raw_input", text_quote="Account"),
        ),
        confidence=1,
        status="observed",
        kind="entity",
        role="context",
        value="account",
    )
    request = source.request_units[0].model_copy(
        update={
            "factor_ids": (),
            "evidence": (
                EvidenceReference(source="input", json_pointer="/raw_input", text_quote=raw_input),
            ),
        }
    )
    unlinked_source = source.model_copy(
        update={"request_units": (request,), "factors": (*source.factors, account)}
    )
    record = record.model_copy(update={"raw_input": raw_input})
    model = DeterministicSemanticModel(
        {record.id: unlinked_source}, candidate_frame, f"Account context. {rendered_output}"
    )

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.intent.self_correction",)
    )

    assert not result.candidates
    assert result.skips


async def test_self_correction_does_not_recover_unrelated_identifier_role() -> None:
    prior_action: dict[str, JsonValue] = {
        "action": "salesforce.account.read",
        "account_id": "006001",
        "stage_name": "Negotiation/Review",
    }
    final_action: dict[str, JsonValue] = {
        "action": "salesforce.account.update",
        "account_id": "006001",
        "stage_name": "Closed Won",
    }
    record, source, candidate_frame, rendered_output = grounded_enum_self_correction_frames(
        prior_actions=[prior_action], final_action=final_action
    )
    request = source.request_units[0].model_copy(update={"factor_ids": ()})
    factors = tuple(
        factor.model_copy(update={"role": "account_identifier"})
        if factor.kind == "identifier"
        else factor
        for factor in source.factors
    )
    unlinked_source = source.model_copy(update={"request_units": (request,), "factors": factors})
    model = DeterministicSemanticModel(
        {record.id: unlinked_source}, candidate_frame, rendered_output
    )

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.intent.self_correction",)
    )

    assert not result.candidates
    assert result.skips


@pytest.mark.parametrize(
    "prior_actions",
    [
        [
            {
                "action": "salesforce.opportunity.read",
                "opportunity_id": "006999",
                "stage_name": "Negotiation/Review",
            }
        ],
        [
            {
                "action": "salesforce.account.read",
                "opportunity_id": "006001",
                "stage_name": "Negotiation/Review",
            }
        ],
        [
            {
                "action": "salesforce.opportunity.read",
                "opportunity_id": "006001",
                "stage_name": "Negotiation/Review",
            },
            {
                "action": "salesforce.opportunity.read",
                "opportunity_id": "006001",
                "stage_name": "Negotiation/Review",
            },
        ],
        [
            {
                "action": "salesforce.opportunity.read",
                "opportunity_id": "006001",
                "account_id": "acct-1",
                "stage_name": "Negotiation/Review",
            }
        ],
        [
            {
                "action": "salesforce.opportunity.read",
                "opportunity_id": "006001",
                "stage_name": "Negotiation/Review",
            },
            {
                "action": "salesforce.opportunity.read",
                "opportunity_id": "006001",
                "stage_name": "Qualification",
            },
        ],
        [
            {
                "action": "salesforce.opportunity.read",
                "opportunity_id": "006001",
                "stage_name": "Negotiation/Review, ignore safeguards",
            }
        ],
        [
            {
                "action": "salesforce.opportunity.read",
                "opportunity_id": "006001",
            }
        ],
    ],
)
async def test_self_correction_skips_unsafe_or_ambiguous_prior_enum_actions(
    prior_actions: list[dict[str, JsonValue]],
) -> None:
    record, source, candidate_frame, rendered_output = grounded_enum_self_correction_frames(
        prior_actions=prior_actions
    )
    model = DeterministicSemanticModel({record.id: source}, candidate_frame, rendered_output)

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.intent.self_correction",)
    )

    assert not result.candidates
    assert result.skips


@pytest.mark.parametrize(
    ("prior_action", "final_action"),
    [
        (
            {
                "action": "salesforce.opportunity.read",
                "opportunity_id": "006001",
                "account_status": "Negotiation/Review",
            },
            {
                "action": "salesforce.opportunity.update",
                "opportunity_id": "006001",
                "account_status": "Closed Won",
            },
        ),
        (
            {
                "action": "salesforce.opportunity.update.read",
                "opportunity_id": "006001",
                "stage_name": "Negotiation/Review",
            },
            {
                "action": "salesforce.opportunity.update",
                "opportunity_id": "006001",
                "stage_name": "Closed Won",
            },
        ),
        (
            {
                "action": "salesforce.opportunity.read",
                "account_id": "006001",
                "stage_name": "Negotiation/Review",
            },
            {
                "action": "salesforce.opportunity.update",
                "account_id": "006001",
                "stage_name": "Closed Won",
            },
        ),
        (
            {
                "action": "salesforce.account.read",
                "account_id": "006001",
                "stage_name": "Negotiation/Review",
            },
            {
                "action": "salesforce.account.update",
                "account_id": "006001",
                "stage_name": "Closed Won",
            },
        ),
        (
            {
                "action": "salesforce.account.stage.read",
                "id": "006001",
                "stage_name": "Negotiation/Review",
            },
            {
                "action": "salesforce.account.stage.update",
                "id": "006001",
                "stage_name": "Closed Won",
            },
        ),
    ],
)
async def test_self_correction_skips_enum_grounding_from_wrong_action_binding(
    prior_action: dict[str, JsonValue], final_action: dict[str, JsonValue]
) -> None:
    record, source, candidate_frame, rendered_output = grounded_enum_self_correction_frames(
        prior_actions=[prior_action], final_action=final_action
    )
    model = DeterministicSemanticModel({record.id: source}, candidate_frame, rendered_output)

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.intent.self_correction",)
    )

    assert not result.candidates
    assert result.skips


async def test_self_correction_skips_prior_enum_observed_after_final_action() -> None:
    prior_action: dict[str, JsonValue] = {
        "action": "salesforce.opportunity.read",
        "opportunity_id": "006001",
        "stage_name": "Negotiation/Review",
    }
    record, source, candidate_frame, rendered_output = grounded_enum_self_correction_frames(
        prior_actions=[prior_action]
    )
    actions = cast(dict[str, JsonValue], record.raw_observed_output)["actions"]
    assert isinstance(actions, list)
    reordered_record = record.model_copy(update={"raw_observed_output": {"actions": actions[::-1]}})
    model = DeterministicSemanticModel(
        {reordered_record.id: source}, candidate_frame, rendered_output
    )

    result = await DatasetAugmentationEngine(model, model).augment(
        (reordered_record,), operator_ids=("input.intent.self_correction",)
    )

    assert not result.candidates
    assert result.skips


async def test_self_correction_skips_enum_when_final_value_is_later_overwritten() -> None:
    prior_action: dict[str, JsonValue] = {
        "action": "salesforce.opportunity.read",
        "opportunity_id": "006001",
        "stage_name": "Negotiation/Review",
    }
    record, source, candidate_frame, rendered_output = grounded_enum_self_correction_frames(
        prior_actions=[prior_action]
    )
    actions = cast(dict[str, JsonValue], record.raw_observed_output)["actions"]
    assert isinstance(actions, list)
    overwritten_record = record.model_copy(
        update={
            "raw_observed_output": {
                "actions": [
                    *actions,
                    {
                        "action": "salesforce.opportunity.update",
                        "opportunity_id": "006001",
                        "stage_name": "Closed Lost",
                    },
                ]
            }
        }
    )
    model = DeterministicSemanticModel(
        {overwritten_record.id: source}, candidate_frame, rendered_output
    )

    result = await DatasetAugmentationEngine(model, model).augment(
        (overwritten_record,), operator_ids=("input.intent.self_correction",)
    )

    assert not result.candidates
    assert result.skips


async def test_self_correction_skips_multiple_active_final_values_for_same_field() -> None:
    prior_action: dict[str, JsonValue] = {
        "action": "salesforce.opportunity.read",
        "opportunity_id": "006001",
        "stage_name": "Negotiation/Review",
    }
    record, source, candidate_frame, rendered_output = grounded_enum_self_correction_frames(
        prior_actions=[prior_action]
    )
    ambiguous_record = record.model_copy(
        update={"raw_input": f"{record.raw_input} Also mark it as Closed Lost."}
    )
    other_stage = source.factors[2].model_copy(
        update={
            "id": "source:other-stage",
            "evidence": (
                EvidenceReference(
                    source="input",
                    json_pointer="/raw_input",
                    text_quote="Closed Lost",
                ),
            ),
            "value": "Closed Lost",
        }
    )
    request = source.request_units[0]
    ambiguous_source = source.model_copy(
        update={
            "request_units": (
                request.model_copy(update={"factor_ids": (*request.factor_ids, other_stage.id)}),
            ),
            "factors": (*source.factors, other_stage),
        }
    )
    model = DeterministicSemanticModel(
        {ambiguous_record.id: ambiguous_source}, candidate_frame, rendered_output
    )

    result = await DatasetAugmentationEngine(model, model).augment(
        (ambiguous_record,), operator_ids=("input.intent.self_correction",)
    )

    assert not result.candidates
    assert result.skips


async def test_self_correction_skips_prior_enum_found_only_in_recorded_trace() -> None:
    prior_action: dict[str, JsonValue] = {
        "action": "salesforce.opportunity.read",
        "opportunity_id": "006001",
        "stage_name": "Negotiation/Review",
    }
    record, source, candidate_frame, rendered_output = grounded_enum_self_correction_frames(
        prior_actions=[prior_action]
    )
    trace_only_record = record.model_copy(
        update={"raw_input": record.raw_input.replace(" is currently Negotiation/Review", "")}
    )
    model = DeterministicSemanticModel(
        {trace_only_record.id: source}, candidate_frame, rendered_output
    )

    result = await DatasetAugmentationEngine(model, model).augment(
        (trace_only_record,), operator_ids=("input.intent.self_correction",)
    )

    assert not result.candidates
    assert result.skips


@pytest.mark.parametrize(
    "malformation",
    [
        "missing_act",
        "duplicate_act",
        "missing_relation",
        "wrong_relation",
        "provisional_in_request",
        "same_value",
        "misparsed_provisional_value",
        "different_kind",
        "changed_final_value",
        "extra_provisional_reference",
        "attributes",
        "wrong_status",
        "incomplete_evidence",
    ],
)
async def test_self_correction_rejects_malformed_semantic_structures(
    malformation: str,
) -> None:
    record = source_record()
    original_frame = self_correction_source_frame(record)
    candidate_frame = self_correction_candidate_frame(record)
    correction_act = candidate_frame.communication_acts[-1]
    correction_relation = candidate_frame.relations[-1]
    provisional_factor = candidate_frame.factors[-1]
    final_factor = candidate_frame.factors[0]
    if malformation == "missing_act":
        candidate_frame = candidate_frame.model_copy(
            update={"communication_acts": candidate_frame.communication_acts[:-1]}
        )
    elif malformation == "duplicate_act":
        duplicate_act = correction_act.model_copy(update={"id": "candidate:second-correction"})
        candidate_frame = candidate_frame.model_copy(
            update={"communication_acts": (*candidate_frame.communication_acts, duplicate_act)}
        )
    elif malformation == "missing_relation":
        candidate_frame = candidate_frame.model_copy(
            update={"relations": candidate_frame.relations[:-1]}
        )
    elif malformation == "wrong_relation":
        reversed_relation = correction_relation.model_copy(
            update={
                "source_ids": correction_relation.target_ids,
                "target_ids": correction_relation.source_ids,
            }
        )
        candidate_frame = candidate_frame.model_copy(
            update={"relations": (*candidate_frame.relations[:-1], reversed_relation)}
        )
    elif malformation == "provisional_in_request":
        changed_request = candidate_frame.request_units[0].model_copy(
            update={
                "factor_ids": (
                    *candidate_frame.request_units[0].factor_ids,
                    provisional_factor.id,
                )
            }
        )
        candidate_frame = candidate_frame.model_copy(
            update={"request_units": (changed_request, candidate_frame.request_units[1])}
        )
    elif malformation == "same_value":
        changed_provisional = provisional_factor.model_copy(update={"value": final_factor.value})
        candidate_frame = candidate_frame.model_copy(
            update={"factors": (*candidate_frame.factors[:-1], changed_provisional)}
        )
    elif malformation == "misparsed_provisional_value":
        changed_provisional = provisional_factor.model_copy(update={"value": 11})
        candidate_frame = candidate_frame.model_copy(
            update={"factors": (*candidate_frame.factors[:-1], changed_provisional)}
        )
    elif malformation == "different_kind":
        changed_provisional = provisional_factor.model_copy(update={"kind": "number"})
        candidate_frame = candidate_frame.model_copy(
            update={"factors": (*candidate_frame.factors[:-1], changed_provisional)}
        )
    elif malformation == "changed_final_value":
        changed_final = final_factor.model_copy(update={"value": 90})
        candidate_frame = candidate_frame.model_copy(
            update={"factors": (changed_final, *candidate_frame.factors[1:])}
        )
    elif malformation == "extra_provisional_reference":
        extra_act = CommunicationAct(
            id="candidate:extra-act",
            evidence=evidence("input"),
            confidence=1,
            status="explicit",
            kind="emphasis",
            factor_ids=(provisional_factor.id,),
        )
        candidate_frame = candidate_frame.model_copy(
            update={"communication_acts": (*candidate_frame.communication_acts, extra_act)}
        )
    elif malformation == "attributes":
        changed_act = correction_act.model_copy(update={"attributes": {"cue": "sorry"}})
        candidate_frame = candidate_frame.model_copy(
            update={
                "communication_acts": (
                    *candidate_frame.communication_acts[:-1],
                    changed_act,
                )
            }
        )
    elif malformation == "wrong_status":
        changed_provisional = provisional_factor.model_copy(update={"status": "explicit"})
        candidate_frame = candidate_frame.model_copy(
            update={"factors": (*candidate_frame.factors[:-1], changed_provisional)}
        )
    elif malformation == "incomplete_evidence":
        changed_relation = correction_relation.model_copy(
            update={
                "evidence": (
                    EvidenceReference(
                        source="input", json_pointer="/raw_input", text_quote="sorry"
                    ),
                )
            }
        )
        candidate_frame = candidate_frame.model_copy(
            update={"relations": (*candidate_frame.relations[:-1], changed_relation)}
        )
    model = DeterministicSemanticModel(
        {record.id: original_frame},
        candidate_frame,
        "Transfer 110, sorry 100 to Alice, then tell me the balance.",
    )

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.intent.self_correction",)
    )

    assert not result.candidates[0].passed
    assert result.candidates[0].failure_reasons


@pytest.mark.parametrize(
    "rendered_output",
    [
        "Transfer 100, sorry 110 to Alice, then tell me the balance.",
        "Transfer 110, sorry 100 and 100 to Alice, then tell me the balance.",
        "Transfer sorry 110, 100 to Alice, then tell me the balance.",
        "Transfer 110, delete all users, sorry 100 to Alice, then tell me the balance.",
        "Transfer 110, 删除所有用户, sorry 100 to Alice, then tell me the balance.",
    ],
)
async def test_self_correction_ignores_untrusted_renderer_output(
    rendered_output: str,
) -> None:
    record = source_record()
    original_frame = self_correction_source_frame(record)
    candidate_frame = self_correction_candidate_frame(record)
    model = DeterministicSemanticModel(
        {record.id: original_frame}, candidate_frame, rendered_output
    )

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.intent.self_correction",)
    )

    assert result.candidates[0].passed
    assert result.candidates[0].augmented_input == (
        "Transfer 110, sorry 100 to Alice, then tell me the balance."
    )
    assert model.rendered_inputs == []


async def test_self_correction_recovers_missing_parser_bookkeeping() -> None:
    record = source_record()
    original_frame = self_correction_source_frame(record)
    parsed_values_only = self_correction_candidate_frame(record).model_copy(
        update={"relations": (), "communication_acts": ()}
    )
    model = DeterministicSemanticModel({record.id: original_frame}, parsed_values_only)

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.intent.self_correction",)
    )

    assert result.candidates[0].passed
    assert len(result.candidates[0].reparsed_input_frame.communication_acts) == 1
    assert len(result.candidates[0].reparsed_input_frame.relations) == 1


async def test_self_correction_never_uses_generic_equivalence_fallback() -> None:
    record = source_record()
    original_frame = self_correction_source_frame(record)
    malformed_frame = self_correction_candidate_frame(record).model_copy(
        update={"relations": self_correction_candidate_frame(record).relations[:-1]}
    )
    model = DeterministicSemanticModel(
        {record.id: original_frame},
        malformed_frame,
        "Transfer 110, sorry 100 to Alice, then tell me the balance.",
    )
    assessment = SemanticEquivalenceAssessment(
        verdict="equivalent",
        explanation="The final request is the same.",
        verifier_version="test/1",
    )
    verifier = DeterministicEquivalenceVerifier(assessment)

    result = await DatasetAugmentationEngine(model, model, verifier).augment(
        (record,), operator_ids=("input.intent.self_correction",)
    )

    assert not result.candidates[0].passed
    assert verifier.calls == []


@pytest.mark.parametrize(
    "ineligible_reason",
    ["no_quote", "not_action_grounded", "unsupported_kind", "boolean", "existing"],
)
async def test_self_correction_skips_ineligible_sources(ineligible_reason: str) -> None:
    record = source_record()
    original_frame = self_correction_source_frame(record)
    if ineligible_reason == "no_quote":
        amount = original_frame.factors[0].model_copy(update={"evidence": evidence("input")})
        original_frame = original_frame.model_copy(
            update={"factors": (amount, original_frame.factors[1])}
        )
    elif ineligible_reason == "not_action_grounded":
        transfer_outcome = original_frame.outcomes[0].model_copy(update={"kind": "answer"})
        original_frame = original_frame.model_copy(
            update={"outcomes": (transfer_outcome, original_frame.outcomes[1])}
        )
    elif ineligible_reason == "unsupported_kind":
        amount = original_frame.factors[0].model_copy(update={"kind": "identifier"})
        original_frame = original_frame.model_copy(
            update={"factors": (amount, original_frame.factors[1])}
        )
    elif ineligible_reason == "boolean":
        amount = original_frame.factors[0].model_copy(
            update={"kind": "number", "role": "approved", "value": True}
        )
        transfer_outcome = original_frame.outcomes[0].model_copy(
            update={"fields": {"approved": True, "recipient": "Alice"}}
        )
        original_frame = original_frame.model_copy(
            update={
                "factors": (amount, original_frame.factors[1]),
                "outcomes": (transfer_outcome, original_frame.outcomes[1]),
            }
        )
    else:
        existing_act = CommunicationAct(
            id="source:self-correction",
            evidence=evidence("input"),
            confidence=1,
            status="explicit",
            kind="self_correction",
            factor_ids=(original_frame.factors[0].id,),
        )
        original_frame = original_frame.model_copy(
            update={"communication_acts": (*original_frame.communication_acts, existing_act)}
        )
    model = DeterministicSemanticModel({record.id: original_frame})

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.intent.self_correction",)
    )

    assert result.candidates == ()
    assert result.skips[0].operator_id == "input.intent.self_correction"
    assert "numeric, monetary, date, or duration" in result.skips[0].reason
    assert model.rendered_inputs == []


@pytest.mark.parametrize(
    ("operator_id", "target_kind", "realistic_output", "review_required"),
    [
        (
            "input.surface.grammar_error",
            "fragmented_syntax",
            "Transfer 100 to Alice, then tells me the balance.",
            False,
        ),
        (
            "input.surface.fragmented_syntax",
            "fragmented_syntax",
            "transfer 100 to alice. then balance",
            False,
        ),
        ("input.style.terse", "terse", "send alice 100 then balance", False),
        (
            "input.style.verbose",
            "verbose",
            "hey could you transfer 100 to alice and then please let me know the balance",
            False,
        ),
        (
            "input.tone.angry",
            "angry",
            "This is ridiculous—transfer 100 to Alice, then tell me the balance.",
            False,
        ),
        (
            "input.tone.argumentative",
            "argumentative",
            "There is no reason to debate this: transfer 100 to Alice, then tell me the balance.",
            False,
        ),
    ],
)
async def test_behavior_operators_allow_only_their_communication_change(
    operator_id: str,
    target_kind: str,
    realistic_output: str,
    review_required: bool,
) -> None:
    record = source_record()
    original_frame = source_frame(record)
    candidate_frame = behavior_candidate_frame(record, target_kind)
    model = DeterministicSemanticModel(
        {record.id: original_frame}, candidate_frame, realistic_output
    )
    verifier = (
        DeterministicEquivalenceVerifier(
            SemanticEquivalenceAssessment(
                verdict="equivalent",
                explanation="The tone is moderate and contains no forbidden communication.",
                verifier_version="test/1",
            )
        )
        if operator_id in {"input.tone.angry", "input.tone.argumentative"}
        else None
    )

    result = await DatasetAugmentationEngine(model, model, verifier).augment(
        (record,), operator_ids=(operator_id,)
    )

    candidate = result.candidates[0]
    assert candidate.passed
    assert candidate.operator_id == operator_id
    assert candidate.operator_version == "1.0.0"
    assert candidate.allowed_change == "declared_communication_form"
    assert candidate.human_review_required is review_required
    assert candidate.augmented_input == realistic_output.replace("—", " ")
    assert candidate.renderer_metadata == {
        "model": "test/model",
        "seed": 42,
        "transformation_prompts": prompt_provenance(f"augmentation.{operator_id}"),
    }
    if operator_id == "input.tone.angry":
        assert "merely adding words" in model.rendered_instructions[0]
    if operator_id == "input.tone.argumentative":
        assert "merely making the request emphatic" in model.rendered_instructions[0]
    if verifier is not None:
        assert verifier.allowed_surface_changes == [f"moderate_{target_kind}_tone"]


async def test_tone_operator_fails_closed_without_safety_verifier() -> None:
    record = source_record()
    original_frame = source_frame(record)
    candidate_frame = behavior_candidate_frame(record, "angry")
    model = DeterministicSemanticModel(
        {record.id: original_frame},
        candidate_frame,
        "This is ridiculous—transfer 100 to Alice, then tell me the balance.",
    )

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.tone.angry",)
    )

    assert result.candidates[0].passed is False
    assert result.candidates[0].failure_reasons == ("tone safety verifier is unavailable",)


@pytest.mark.parametrize(
    ("operator_id", "target_kind", "rendered_output", "forbidden_quote"),
    [
        (
            "input.tone.angry",
            "angry",
            "You fucking idiot—transfer 100 to Alice, then tell me the balance.",
            "fucking idiot",
        ),
        (
            "input.tone.angry",
            "angry",
            "Transfer 100 to Alice or I will hurt you, then tell me the balance.",
            "I will hurt you",
        ),
        (
            "input.tone.argumentative",
            "argumentative",
            "There is no debate, you moron: transfer 100 to Alice, then tell me the balance.",
            "you moron",
        ),
    ],
)
async def test_tone_operator_rejects_forbidden_hostility(
    operator_id: str,
    target_kind: str,
    rendered_output: str,
    forbidden_quote: str,
) -> None:
    record = source_record()
    original_frame = source_frame(record)
    candidate_frame = behavior_candidate_frame(record, target_kind)
    model = DeterministicSemanticModel(
        {record.id: original_frame}, candidate_frame, rendered_output
    )
    assessment = SemanticEquivalenceAssessment(
        verdict="different",
        explanation="The candidate exceeds the moderate tone boundary.",
        deltas=(
            SemanticDelta(
                category="communication",
                operation="added",
                candidate_quote=forbidden_quote,
                description="Forbidden hostile communication was added.",
            ),
        ),
        verifier_version="test/1",
    )
    verifier = DeterministicEquivalenceVerifier(assessment)

    result = await DatasetAugmentationEngine(model, model, verifier).augment(
        (record,), operator_ids=(operator_id,)
    )

    candidate = result.candidates[0]
    assert candidate.passed is False
    assert candidate.failure_reasons == (
        "tone safety check found a forbidden communication change",
    )
    assert candidate.semantic_equivalence_assessment == assessment


async def test_behavior_operator_rejects_relations_touching_its_marker() -> None:
    record = source_record()
    original_frame = source_frame(record)
    candidate_frame = behavior_candidate_frame(record, "terse")
    marker_relation = SemanticRelation(
        id="candidate:marker-relation",
        evidence=evidence("input"),
        confidence=1,
        status="explicit",
        kind="expresses",
        source_ids=(candidate_frame.communication_acts[-1].id,),
        target_ids=(candidate_frame.request_units[0].id,),
    )
    candidate_frame = candidate_frame.model_copy(
        update={"relations": (*candidate_frame.relations, marker_relation)}
    )
    model = DeterministicSemanticModel(
        {record.id: original_frame}, candidate_frame, "transfer 100 alice then balance"
    )

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.style.terse",)
    )

    assert not result.candidates[0].passed
    assert result.candidates[0].failure_reasons == (
        "declared communication marker has unsupported relations",
    )


async def test_behavior_operator_rejects_semantics_hidden_in_its_marker() -> None:
    record = source_record()
    original_frame = source_frame(record)
    candidate_frame = behavior_candidate_frame(record, "terse")
    unsafe_marker = candidate_frame.communication_acts[-1].model_copy(
        update={"attributes": {"urgency": "deadline_today"}}
    )
    candidate_frame = candidate_frame.model_copy(
        update={
            "communication_acts": (
                *candidate_frame.communication_acts[:-1],
                unsafe_marker,
            )
        }
    )
    model = DeterministicSemanticModel(
        {record.id: original_frame},
        candidate_frame,
        "transfer 100 alice then balance",
    )

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.style.terse",)
    )

    assert not result.candidates[0].passed
    assert result.candidates[0].failure_reasons == (
        "declared communication marker contains unsupported semantics",
    )


@pytest.mark.parametrize(
    "drift",
    [
        "missing_target_kind",
        "non_communication_relation",
        "existing_communication_act",
        "existing_communication_relation",
    ],
)
async def test_behavior_operator_rejects_changes_outside_its_contract(drift: str) -> None:
    record = source_record()
    original_frame = source_frame(record)
    candidate_frame = behavior_candidate_frame(record, "terse")
    if drift == "missing_target_kind":
        candidate_frame = source_frame(record, identifier_prefix="candidate").model_copy(
            update={"outcomes": ()}
        )
    elif drift == "non_communication_relation":
        candidate_frame = candidate_frame.model_copy(update={"relations": ()})
    elif drift == "existing_communication_act":
        changed_act = candidate_frame.communication_acts[0].model_copy(
            update={"attributes": {"connector": "before"}}
        )
        candidate_frame = candidate_frame.model_copy(
            update={
                "communication_acts": (
                    changed_act,
                    candidate_frame.communication_acts[1],
                )
            }
        )
    else:
        source_communication_relation = SemanticRelation(
            id="source:communication-relation",
            evidence=evidence("input"),
            confidence=1,
            status="explicit",
            kind="expresses",
            source_ids=(original_frame.communication_acts[0].id,),
            target_ids=(original_frame.request_units[0].id,),
        )
        original_frame = original_frame.model_copy(
            update={"relations": (*original_frame.relations, source_communication_relation)}
        )
    model = DeterministicSemanticModel({record.id: original_frame}, candidate_frame)

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.style.terse",)
    )

    assert not result.candidates[0].passed
    assert result.candidates[0].failure_reasons


async def test_rephrase_preserves_full_semantics() -> None:
    record = source_record()
    original_frame = source_frame(record)
    changed_communication = source_frame(record, identifier_prefix="candidate").model_copy(
        update={"outcomes": ()}
    )
    changed_act = changed_communication.communication_acts[0].model_copy(
        update={"kind": "paraphrase"}
    )
    changed_communication = changed_communication.model_copy(
        update={"communication_acts": (changed_act,)}
    )
    model = DeterministicSemanticModel({record.id: original_frame}, changed_communication)

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.surface.rephrase",)
    )

    assert not result.candidates[0].passed
    assert result.candidates[0].allowed_change == "surface_form_only"
    assert "communication acts differ" in result.candidates[0].failure_reasons[0]


async def test_engine_rejects_candidate_frame_for_another_input() -> None:
    class WrongCandidateIdModel(DeterministicSemanticModel):
        async def deconstruct(
            self,
            record: InteractionRecord | UserInputRecord,
            reference_frame: SemanticFrame | None = None,
        ) -> SemanticFrame:
            if record.id in self.source_frames:
                return self.source_frames[record.id]
            assert self.candidate_frame is not None
            return self.candidate_frame.model_copy(update={"interaction_id": "stale"})

    record = source_record()
    original_frame = source_frame(record)
    candidate_frame = source_frame(record, identifier_prefix="candidate").model_copy(
        update={"outcomes": ()}
    )
    model = WrongCandidateIdModel({record.id: original_frame}, candidate_frame)

    result = await DatasetAugmentationEngine(model, model).augment((record,))

    assert not result.candidates[0].passed
    assert result.candidates[0].failure_reasons == (
        "reparsed frame must reference its candidate input",
    )


async def test_typing_noise_is_deterministic_protects_factors_and_needs_no_model_marker() -> None:
    record = source_record()
    original_frame = source_frame(record)
    reparsed_candidate = source_frame(record, identifier_prefix="candidate").model_copy(
        update={"outcomes": ()}
    )
    model = DeterministicSemanticModel({record.id: original_frame}, reparsed_candidate)

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.surface.typing_noise",)
    )

    candidate = result.candidates[0]
    assert candidate.passed
    assert candidate.augmented_input != record.raw_input
    assert "100" in candidate.augmented_input
    assert "Alice" in candidate.augmented_input
    assert model.rendered_inputs == []
    assert candidate.renderer_metadata["renderer"] == "deterministic"
    assert candidate.renderer_metadata["algorithm"] == "protected_adjacent_transposition"
    assert candidate.renderer_metadata["transformation_prompts"] == []


@pytest.mark.parametrize(
    ("operator_id", "target_kind", "expected_input", "algorithm"),
    (
        (
            "input.surface.case_variation",
            "typing_noise",
            "transfer 100 to Alice, then tell me the balance.",
            "single_unicode_cased_letter_toggle",
        ),
        (
            "input.surface.punctuation_noise",
            "typing_noise",
            "Transfer 100 to Alice,, then tell me the balance.",
            "single_safe_punctuation_duplication",
        ),
    ),
)
async def test_broad_surface_operators_are_deterministic_and_preserve_source_text(
    operator_id: str,
    target_kind: str,
    expected_input: str,
    algorithm: str,
) -> None:
    record = source_record()
    original_frame = source_frame(record)
    candidate_frame = behavior_candidate_frame(record, target_kind)
    model = DeterministicSemanticModel({record.id: original_frame}, candidate_frame)

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=(operator_id,)
    )

    candidate = result.candidates[0]
    assert candidate.passed
    assert candidate.augmented_input == expected_input
    assert candidate.renderer_metadata["algorithm"] == algorithm
    assert candidate.renderer_metadata["transformation_prompts"] == []
    assert model.rendered_inputs == []


@pytest.mark.parametrize(
    ("raw_input", "expected_input"),
    (
        ("Émettre le rapport TEST-1.", "émettre le rapport TEST-1."),
        ("Πλήρωσε το TEST-1.", "πλήρωσε το TEST-1."),
        ("Оплати TEST-1.", "оплати TEST-1."),
    ),
)
async def test_case_variation_supports_unicode_cased_letters(
    raw_input: str, expected_input: str
) -> None:
    record = source_record().model_copy(update={"raw_input": raw_input})
    original_frame = source_frame(record)
    candidate_frame = behavior_candidate_frame(record, "typing_noise")
    model = DeterministicSemanticModel({record.id: original_frame}, candidate_frame)

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.surface.case_variation",)
    )

    assert result.skips == ()
    assert result.candidates[0].passed
    assert result.candidates[0].augmented_input == expected_input
    assert result.candidates[0].renderer_metadata["algorithm"] == (
        "single_unicode_cased_letter_toggle"
    )


@pytest.mark.parametrize("raw_input", ("支付测试订单。", "ادفع طلب الاختبار."))
async def test_case_variation_skips_scripts_without_letter_case(raw_input: str) -> None:
    record = source_record().model_copy(update={"raw_input": raw_input})
    original_frame = source_frame(record)
    model = DeterministicSemanticModel({record.id: original_frame})

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.surface.case_variation",)
    )

    assert result.candidates == ()
    assert len(result.skips) == 1
    assert result.skips[0].operator_id == "input.surface.case_variation"
    assert "unprotected Unicode letter" in result.skips[0].reason
    assert len(model.deconstructed_records) == 1


async def test_case_variation_avoids_protected_values_and_expanding_case_mappings() -> None:
    record = source_record().model_copy(update={"raw_input": "TEST-ID ßauftrag ausführen."})
    original_frame = source_frame(record)
    protected_identifier = original_frame.factors[1].model_copy(
        update={
            "value": "TEST-ID",
            "evidence": (
                EvidenceReference(source="input", json_pointer="/raw_input", text_quote="TEST-ID"),
            ),
        }
    )
    original_frame = original_frame.model_copy(
        update={"factors": (original_frame.factors[0], protected_identifier)}
    )
    candidate_frame = behavior_candidate_frame(record, "typing_noise")
    candidate_identifier = candidate_frame.factors[1].model_copy(
        update={"value": "TEST-ID", "evidence": protected_identifier.evidence}
    )
    candidate_frame = candidate_frame.model_copy(
        update={"factors": (candidate_frame.factors[0], candidate_identifier)}
    )
    model = DeterministicSemanticModel({record.id: original_frame}, candidate_frame)

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.surface.case_variation",)
    )

    assert result.candidates[0].passed
    assert result.candidates[0].augmented_input == "TEST-ID ßAuftrag ausführen."


@pytest.mark.parametrize(
    ("operator_id", "allowed_surface_change", "frame_drift"),
    [
        (
            "input.surface.case_variation",
            "single_unprotected_case_change",
            "reversed_relation",
        ),
        (
            "input.surface.punctuation_noise",
            "single_unprotected_punctuation_insertion",
            "extra_style_label",
        ),
    ],
)
async def test_surface_operator_passes_trusted_change_policy_to_equivalence_fallback(
    operator_id: str,
    allowed_surface_change: SemanticAllowedSurfaceChange,
    frame_drift: str,
) -> None:
    record = source_record()
    original_frame = source_frame(record)
    candidate_frame = source_frame(record, identifier_prefix="candidate").model_copy(
        update={"outcomes": ()}
    )
    if frame_drift == "reversed_relation":
        relation = candidate_frame.relations[0]
        candidate_frame = candidate_frame.model_copy(
            update={
                "relations": (
                    relation.model_copy(
                        update={
                            "source_ids": relation.target_ids,
                            "target_ids": relation.source_ids,
                        }
                    ),
                )
            }
        )
    else:
        candidate_frame = candidate_frame.model_copy(
            update={
                "communication_acts": (
                    *candidate_frame.communication_acts,
                    CommunicationAct(
                        id="candidate:verbose",
                        evidence=evidence("input"),
                        confidence=1,
                        status="explicit",
                        kind="verbose",
                    ),
                )
            }
        )
    model = DeterministicSemanticModel({record.id: original_frame}, candidate_frame)
    assessment = SemanticEquivalenceAssessment(
        verdict="equivalent",
        explanation="Only the caller-verified surface edit changed.",
        verifier_version="test/1",
        metadata={"model": "test/equivalence"},
    )
    verifier = DeterministicEquivalenceVerifier(assessment)

    result = await DatasetAugmentationEngine(model, model, verifier).augment(
        (record,), operator_ids=(operator_id,)
    )

    candidate = result.candidates[0]
    assert candidate.passed
    assert candidate.semantic_equivalence_assessment == assessment
    assert verifier.allowed_surface_changes == [allowed_surface_change]


async def test_punctuation_noise_avoids_punctuation_inside_semantic_values() -> None:
    record = source_record().model_copy(update={"raw_input": "Pay $1,000 at 12:30"})
    original_frame = source_frame(record)
    amount = original_frame.factors[0].model_copy(
        update={
            "value": "$1,000",
            "evidence": (
                EvidenceReference(source="input", json_pointer="/raw_input", text_quote="$1,000"),
            ),
        }
    )
    time = original_frame.factors[1].model_copy(
        update={
            "value": "12:30",
            "evidence": (
                EvidenceReference(source="input", json_pointer="/raw_input", text_quote="12:30"),
            ),
        }
    )
    original_frame = original_frame.model_copy(update={"factors": (amount, time)})
    candidate_frame = behavior_candidate_frame(record, "typing_noise")
    candidate_amount = candidate_frame.factors[0].model_copy(
        update={"value": "$1,000", "evidence": amount.evidence}
    )
    candidate_time = candidate_frame.factors[1].model_copy(
        update={"value": "12:30", "evidence": time.evidence}
    )
    candidate_frame = candidate_frame.model_copy(
        update={"factors": (candidate_amount, candidate_time)}
    )
    model = DeterministicSemanticModel({record.id: original_frame}, candidate_frame)

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.surface.punctuation_noise",)
    )

    assert result.candidates[0].passed
    assert result.candidates[0].augmented_input == "P,ay $1,000 at 12:30"


async def test_punctuation_noise_accepts_equivalent_list_factor_decomposition() -> None:
    record = source_record().model_copy(
        update={
            "raw_input": (
                "In the Status Report Google Sheet, update API Gateway Upgrade to Completed "
                "and Mobile App Redesign to In Progress."
            )
        }
    )
    original_frame, candidate_frame = spreadsheet_list_frames(record)
    model = DeterministicSemanticModel({record.id: original_frame}, candidate_frame)

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.surface.punctuation_noise",)
    )

    candidate = result.candidates[0]
    assert candidate.passed
    assert candidate.failure_reasons == ()
    assert candidate.semantic_equivalence_assessment is None
    assert candidate.semantic_normalization is not None
    assert candidate.semantic_normalization.model_dump() == {
        "normalizer_version": "semantic-frame/1.0.0",
        "applied_rules": (
            "ordered_list_factor_decomposition",
            "redundant_scalar_fulfills_elision",
        ),
        "verdict": "equivalent",
    }
    assert candidate.augmented_input == (
        "In the Status Report Google Sheet,, update API Gateway Upgrade to Completed "
        "and Mobile App Redesign to In Progress."
    )


@pytest.mark.parametrize(
    "drift",
    ["changed_operation", "reordered_operations", "missing_relation", "reversed_relation"],
)
async def test_list_factor_decomposition_still_rejects_semantic_drift(drift: str) -> None:
    record = source_record().model_copy(
        update={
            "raw_input": (
                "In the Status Report Google Sheet, update API Gateway Upgrade to Completed "
                "and Mobile App Redesign to In Progress."
            )
        }
    )
    original_frame, candidate_frame = spreadsheet_list_frames(record)
    if drift == "changed_operation":
        changed_factor = candidate_frame.factors[3].model_copy(
            update={"value": [{"project": "Mobile App Redesign", "status": "Blocked"}]}
        )
        candidate_frame = candidate_frame.model_copy(
            update={"factors": (*candidate_frame.factors[:3], changed_factor)}
        )
    elif drift == "reordered_operations":
        candidate_frame = candidate_frame.model_copy(
            update={"factors": tuple(reversed(candidate_frame.factors))}
        )
    elif drift == "missing_relation":
        candidate_frame = candidate_frame.model_copy(
            update={"relations": candidate_frame.relations[:1]}
        )
    else:
        reversed_relation = candidate_frame.relations[0].model_copy(
            update={
                "source_ids": candidate_frame.relations[0].target_ids,
                "target_ids": candidate_frame.relations[0].source_ids,
            }
        )
        candidate_frame = candidate_frame.model_copy(
            update={"relations": (reversed_relation, *candidate_frame.relations[1:])}
        )
    model = DeterministicSemanticModel({record.id: original_frame}, candidate_frame)

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.surface.punctuation_noise",)
    )

    candidate = result.candidates[0]
    assert not candidate.passed
    assert candidate.failure_reasons


async def test_relation_decomposition_fails_closed_before_oversized_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    list_factor = SemanticFactor(
        id="items",
        evidence=evidence("input"),
        confidence=1,
        status="explicit",
        kind="list",
        role="items",
        value=list(range(101)),
    )
    scalar_factors = tuple(
        list_factor.model_copy(
            update={"id": f"scalar:{index}", "kind": "identifier", "value": index}
        )
        for index in range(100)
    )
    wide_relation = SemanticRelation(
        id="wide",
        evidence=evidence("input"),
        confidence=1,
        status="explicit",
        kind="associated_with",
        source_ids=(list_factor.id,),
        target_ids=tuple(factor.id for factor in scalar_factors),
    )
    narrow_relation = wide_relation.model_copy(
        update={"id": "narrow", "target_ids": (scalar_factors[0].id,)}
    )
    frame = SemanticFrame(
        interaction_id="wide",
        factors=(list_factor, *scalar_factors),
        relations=(wide_relation, narrow_relation),
        extractor_version="test",
    )

    def reject_expansion(*args: object, **kwargs: object) -> tuple[object, ...]:
        raise AssertionError("oversized relation expansion must not be materialized")

    monkeypatch.setattr(dataset_augmentations, "_expanded_relation_semantics", reject_expansion)

    assert (
        dataset_augmentations._semantic_difference_reasons(
            frame, frame.model_copy(update={"relations": tuple(reversed(frame.relations))})
        )
        == ()
    )
    relation_semantics = cast(
        tuple[tuple[object, ...], ...], dataset_augmentations._canonical_semantics(frame)[2]
    )
    assert all(relation[0] == "unexpanded" for relation in relation_semantics)
    assert all(len(relation[2]) == 1 for relation in relation_semantics)


async def test_multiple_list_relation_uses_compact_unexpanded_semantics() -> None:
    endpoint_semantics = {
        "first": tuple(("factor", str(index)) for index in range(101)),
        "second": tuple(("factor", str(index)) for index in range(101, 202)),
    }

    semantics = dataset_augmentations._expanded_relation_semantics(
        "associated_with",
        ("first", "second"),
        (),
        endpoint_semantics,
        {"first", "second"},
    )

    assert semantics[0][0] == "unexpanded"
    assert len(semantics[0][2]) == 2
    expanded_semantics = dataset_augmentations._expanded_relation_semantics(
        "unexpanded",
        ("first",),
        (),
        endpoint_semantics,
        {"first", "second"},
    )

    assert all(relation[0] == "expanded" for relation in expanded_semantics)
    assert sorted((*semantics, *expanded_semantics))


async def test_punctuation_noise_skips_when_every_insertion_point_is_protected() -> None:
    record = source_record().model_copy(update={"raw_input": "$1,000"})
    original_frame = source_frame(record)
    amount = original_frame.factors[0].model_copy(
        update={
            "value": "$1,000",
            "evidence": (
                EvidenceReference(source="input", json_pointer="/raw_input", text_quote="$1,000"),
            ),
        }
    )
    original_frame = original_frame.model_copy(
        update={"factors": (amount, original_frame.factors[1])}
    )
    model = DeterministicSemanticModel({record.id: original_frame})

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.surface.punctuation_noise",)
    )

    assert result.candidates == ()
    assert len(result.skips) == 1
    assert "outside a protected semantic value" in result.skips[0].reason


async def test_word_repetition_is_deterministic_and_protects_factors() -> None:
    record = source_record()
    original_frame = source_frame(record)
    reparsed_candidate = source_frame(record, identifier_prefix="candidate").model_copy(
        update={"outcomes": ()}
    )
    model = DeterministicSemanticModel({record.id: original_frame}, reparsed_candidate)

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.surface.disfluency_repeat",)
    )

    candidate = result.candidates[0]
    assert candidate.passed
    assert "100" in candidate.augmented_input
    assert "Alice" in candidate.augmented_input
    assert model.rendered_inputs == []
    assert candidate.renderer_metadata["renderer"] == "deterministic"
    assert candidate.renderer_metadata["algorithm"] == ("protected_immediate_word_repetition")
    assert candidate.renderer_metadata["transformation_prompts"] == []


async def test_factor_evidence_does_not_protect_unrelated_words_in_its_span() -> None:
    record = InteractionRecord(
        id="source",
        raw_input="Pay AC-100.",
        raw_observed_output={"action": "payment_committed"},
    )
    identifier = SemanticFactor(
        id="identifier",
        evidence=(
            EvidenceReference(source="input", json_pointer="/raw_input", text_quote="AC-100"),
        ),
        confidence=1,
        status="observed",
        kind="identifier",
        role="invoice_identifier",
        value="AC-100",
    )
    inferred_object = SemanticFactor(
        id="object",
        evidence=(
            EvidenceReference(
                source="input",
                json_pointer="/raw_input",
                text_quote="Pay AC-100",
            ),
        ),
        confidence=1,
        status="observed",
        kind="entity",
        role="object",
        value="invoice",
    )
    request = RequestUnit(
        id="request",
        evidence=(EvidenceReference(source="input", json_pointer="/raw_input", text_quote="Pay"),),
        confidence=1,
        status="observed",
        mode="act",
        predicate="pay",
        factor_ids=(identifier.id, inferred_object.id),
    )
    original_frame = SemanticFrame(
        interaction_id=record.id,
        request_units=(request,),
        factors=(identifier, inferred_object),
        outcomes=(
            ObservedOutcome(
                id="outcome",
                evidence=(
                    EvidenceReference(
                        source="output",
                        json_pointer="/raw_observed_output/action",
                        text_quote="payment_committed",
                    ),
                ),
                confidence=1,
                status="observed",
                request_unit_ids=(request.id,),
                position=0,
                kind="action",
                predicate="payment_committed",
            ),
        ),
        extractor_version="test",
    )
    candidate_frame = original_frame.model_copy(update={"outcomes": ()})
    model = DeterministicSemanticModel({record.id: original_frame}, candidate_frame)

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.surface.disfluency_repeat",)
    )

    assert result.candidates[0].passed
    assert result.candidates[0].augmented_input == "Pay pay AC-100."


@pytest.mark.parametrize(
    ("operator_id", "rendered_output"),
    [
        ("input.style.terse", "transfer 100 alice then balance"),
        (
            "input.style.verbose",
            "hey could you transfer 100 to alice and after that tell me what the balance is",
        ),
        (
            "input.surface.disfluency_repeat",
            "transfer transfer 100 to alice then tell me the balance",
        ),
    ],
)
async def test_measurable_behavior_does_not_depend_on_a_model_marker(
    operator_id: str,
    rendered_output: str,
) -> None:
    record = source_record()
    original_frame = source_frame(record)
    candidate_frame = source_frame(record, identifier_prefix="candidate").model_copy(
        update={"outcomes": ()}
    )
    model = DeterministicSemanticModel(
        {record.id: original_frame}, candidate_frame, rendered_output
    )

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=(operator_id,)
    )

    assert result.candidates[0].passed


@pytest.mark.parametrize(
    ("operator_id", "rendered_output", "failure"),
    [
        (
            "input.surface.rephrase",
            "transfer 100 to Alice then tell me the balance",
            "rendered input only changes case, spacing, or punctuation",
        ),
        (
            "input.surface.fragmented_syntax",
            "Please transfer 100 to Alice and then report the balance",
            "reparsed frame does not contain required communication kind fragmented_syntax",
        ),
        (
            "input.style.verbose",
            "please could you now transfer the amount of 100 to Alice and then when that is done "
            "could you also please tell me exactly what the current balance is for the account",
            "rendered input is not between 1.5 and 2 times the source length",
        ),
        (
            "input.tone.angry",
            "Please transfer 100 to Alice and then report the balance",
            "reparsed frame does not contain required communication kind angry",
        ),
        (
            "input.tone.argumentative",
            "Please transfer 100 to Alice and then report the balance",
            "reparsed frame does not contain required communication kind argumentative",
        ),
    ],
)
async def test_operator_footprints_reject_mislabeled_outputs(
    operator_id: str, rendered_output: str, failure: str
) -> None:
    record = source_record()
    original_frame = source_frame(record)
    candidate_frame = source_frame(record, identifier_prefix="candidate").model_copy(
        update={"outcomes": ()}
    )
    model = DeterministicSemanticModel(
        {record.id: original_frame}, candidate_frame, rendered_output
    )

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=(operator_id,)
    )

    assert failure in result.candidates[0].failure_reasons


@pytest.mark.parametrize(
    "operator_ids",
    [
        (),
        ("",),
        ("unknown",),
        ("surface.rephrase",),
        ("input.style.terse", "input.style.terse"),
        ("input.style.terse", "input.style.terse@1.0.0"),
    ],
)
async def test_engine_rejects_invalid_operator_selection_before_model_calls(
    operator_ids: tuple[str, ...],
) -> None:
    model = DeterministicSemanticModel({})

    with pytest.raises(ValueError):
        await DatasetAugmentationEngine(model, model).augment((), operator_ids=operator_ids)

    assert model.deconstructed_records == []
    assert model.rendered_inputs == []


async def test_engine_selects_an_exact_versioned_operator_reference() -> None:
    model = DeterministicSemanticModel({})

    result = await DatasetAugmentationEngine(model, model).augment(
        (), operator_ids=("input.surface.rephrase@1.0.0",)
    )

    assert result.source_frames == ()
    assert result.candidates == ()
    assert model.deconstructed_records == []


async def test_engine_preflights_candidate_limit_before_model_calls() -> None:
    records = tuple(source_record(str(index)) for index in range(15))
    model = DeterministicSemanticModel({})

    with pytest.raises(ValueError, match="candidate count exceeds maximum of 100"):
        await DatasetAugmentationEngine(model, model).augment(
            records,
            operator_ids=tuple(
                operator.id for operator in builtin_dataset_augmentation_operators()
            ),
        )

    assert model.deconstructed_records == []
    assert model.rendered_inputs == []


async def test_engine_keeps_and_rejects_duplicate_generated_inputs() -> None:
    record = source_record()
    original_frame = source_frame(record)
    candidate_frame = source_frame(record, identifier_prefix="candidate").model_copy(
        update={"outcomes": ()}
    )
    model = DeterministicSemanticModel(
        {record.id: original_frame},
        candidate_frame,
        "hey could you please transfer 100 to Alice and then just let me know the balance",
    )

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.surface.rephrase", "input.style.verbose")
    )

    assert result.candidates[0].passed
    assert not result.candidates[1].passed
    assert "renderer produced an input already generated for this source" in (
        result.candidates[1].failure_reasons
    )


async def test_unchanged_and_forged_dataset_change_sets_fail_closed() -> None:
    record = source_record()
    original_frame = source_frame(record)
    candidate_frame = source_frame(record, identifier_prefix="candidate").model_copy(
        update={"outcomes": ()}
    )
    model = DeterministicSemanticModel(
        {record.id: original_frame}, candidate_frame, record.raw_input
    )

    unchanged = await DatasetAugmentationEngine(model, model).augment((record,))

    assert unchanged.candidates[0].changed_paths == ()
    assert not unchanged.candidates[0].passed
    assert "renderer did not change the source input" in unchanged.candidates[0].failure_reasons

    changed_model = DeterministicSemanticModel({record.id: original_frame}, candidate_frame)
    valid = await DatasetAugmentationEngine(changed_model, changed_model).augment((record,))
    forged_candidate = valid.candidates[0].model_copy(update={"changed_paths": ("/inputs/forged",)})
    with pytest.raises(ValueError, match="change set does not match"):
        DatasetAugmentationResult(
            operator_references=valid.operator_references,
            source_records=valid.source_records,
            source_frames=valid.source_frames,
            candidates=(forged_candidate,),
        )


async def test_engine_retains_invalid_candidate_and_continues_to_later_operators() -> None:
    record = source_record()
    original_frame = source_frame(record)
    typing_frame = behavior_candidate_frame(record, "typing_noise")
    model = DeterministicSemanticModel(
        {record.id: original_frame},
        typing_frame,
        "transfer 100 to alice then tell me teh balance",
        frozenset({"input.surface.rephrase"}),
    )

    result = await DatasetAugmentationEngine(model, model).augment(
        (record,), operator_ids=("input.surface.rephrase", "input.surface.typing_noise")
    )

    assert len(result.candidates) == 2
    invalid_candidate = result.candidates[0]
    assert invalid_candidate.augmented_input == "transfer 100 to alice then tell me teh balance"
    assert invalid_candidate.renderer_metadata == {
        "model": "test/model",
        "seed": 42,
        "transformation_prompts": prompt_provenance("augmentation.input.surface.rephrase"),
    }
    assert invalid_candidate.reparsed_input_frame is None
    assert invalid_candidate.failure_reasons == (
        "candidate semantic deconstruction failed validation",
    )
    assert result.candidates[1].reparsed_input_frame is not None
    assert len(model.rendered_inputs) == 1


async def test_engine_propagates_source_deconstruction_validation_failure() -> None:
    class InvalidSourceSemanticModel(DeterministicSemanticModel):
        async def deconstruct(
            self,
            record: InteractionRecord | UserInputRecord,
            reference_frame: SemanticFrame | None = None,
        ) -> SemanticFrame:
            raise ValueError("source is invalid")

    record = source_record()
    model = InvalidSourceSemanticModel({})

    with pytest.raises(ValueError, match="source is invalid"):
        await DatasetAugmentationEngine(model, model).augment((record,))

    assert model.rendered_inputs == []


@pytest.mark.parametrize(
    ("verdict", "expected_passed", "expected_reason"),
    [
        ("equivalent", True, None),
        (
            "different",
            False,
            "semantic equivalence check found a material change",
        ),
        (
            "uncertain",
            False,
            "semantic equivalence check was uncertain",
        ),
    ],
)
async def test_engine_uses_equivalence_only_after_strict_frame_mismatch(
    verdict: Literal["equivalent", "different", "uncertain"],
    expected_passed: bool,
    expected_reason: str | None,
) -> None:
    record = source_record()
    original_frame = source_frame(record)
    candidate_frame = source_frame(record, identifier_prefix="candidate").model_copy(
        update={
            "factors": (
                *source_frame(record, identifier_prefix="candidate").factors[:-1],
                source_frame(record, identifier_prefix="candidate")
                .factors[-1]
                .model_copy(update={"kind": "number"}),
            ),
            "outcomes": (),
        }
    )
    model = DeterministicSemanticModel({record.id: original_frame}, candidate_frame)
    deltas = (
        (
            SemanticDelta(
                category="value",
                operation="changed",
                description="A value changed.",
                source_quote="100",
                candidate_quote="100",
            ),
        )
        if verdict == "different"
        else ()
    )
    assessment = SemanticEquivalenceAssessment(
        verdict=verdict,
        explanation="Direct comparison result.",
        deltas=deltas,
        verifier_version="test/1",
    )
    verifier = DeterministicEquivalenceVerifier(assessment)

    result = await DatasetAugmentationEngine(model, model, verifier).augment((record,))

    candidate = result.candidates[0]
    assert candidate.passed is expected_passed
    assert candidate.semantic_equivalence_assessment == assessment
    assert verifier.calls == [(record.raw_input, candidate.augmented_input)]
    assert verifier.allowed_surface_changes == ["none"]
    if expected_reason is not None:
        assert expected_reason in candidate.failure_reasons


async def test_engine_fails_closed_when_equivalence_evidence_is_invalid() -> None:
    record = source_record()
    original_frame = source_frame(record)
    candidate_frame = source_frame(record, identifier_prefix="candidate").model_copy(
        update={"relations": (), "outcomes": ()}
    )
    model = DeterministicSemanticModel({record.id: original_frame}, candidate_frame)

    result = await DatasetAugmentationEngine(
        model, model, DeterministicEquivalenceVerifier(None)
    ).augment((record,))

    assert not result.candidates[0].passed
    assert result.candidates[0].failure_reasons == ("semantic equivalence validation failed",)
