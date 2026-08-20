import json
from typing import TypedDict

import pytest
from pydantic import ValidationError
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
    SemanticRelation,
    UserInputRecord,
)


class FrameElementFields(TypedDict):
    id: str
    confidence: float
    status: str


def frame_element_fields(identifier: str, *, status: str = "explicit") -> FrameElementFields:
    return {"id": identifier, "confidence": 0.9, "status": status}


def test_rendered_user_input_preserves_replay_metadata() -> None:
    rendered = RenderedUserInput(
        text="transfer 100$ to alice",
        metadata={"model": "x-ai/grok", "seed": 7},
    )

    assert RenderedUserInput.model_validate_json(rendered.model_dump_json()) == rendered


def test_observed_agent_output_preserves_replay_metadata() -> None:
    observation = ObservedAgentOutput(
        raw_output={"actions": [{"kind": "transfer", "amount": 100}]},
        metadata={"endpoint_version": "test-v2", "latency_ms": 12},
    )

    assert ObservedAgentOutput.model_validate_json(observation.model_dump_json()) == observation

    with pytest.raises(ValidationError, match="require an output"):
        ObservedAgentOutput(raw_output=None)


def test_mixed_action_and_answer_frame_round_trips() -> None:
    record = InteractionRecord(
        id="interaction-1",
        raw_input="Update my address to 10 Main Street and tell me my balance.",
        raw_observed_output={
            "action": {"name": "update_address", "address": "10 Main Street"},
            "answer": "Your balance is $120.",
        },
        metadata={"source": "production"},
    )
    address = SemanticFactor(
        **frame_element_fields("factor-address"),
        kind="entity.location.postal",
        role="destination_address",
        value="10 Main Street",
        evidence=(
            EvidenceReference(
                source="input", json_pointer="/raw_input", text_quote="10 Main Street"
            ),
        ),
    )
    update_request = RequestUnit(
        **frame_element_fields("request-update"),
        mode="act",
        predicate="update_address",
        factor_ids=(address.id,),
    )
    balance_request = RequestUnit(
        **frame_element_fields("request-balance", status="understood"),
        mode="retrieve",
        predicate="account_balance",
    )
    frame = SemanticFrame(
        interaction_id=record.id,
        request_units=(update_request, balance_request),
        factors=(address,),
        relations=(
            SemanticRelation(
                **frame_element_fields("relation-sequence"),
                kind="sequence",
                source_ids=(update_request.id,),
                target_ids=(balance_request.id,),
            ),
        ),
        communication_acts=(
            CommunicationAct(
                **frame_element_fields("act-request"),
                kind="request.compound",
                factor_ids=(address.id,),
                attributes={"conjunction": "and"},
            ),
        ),
        outcomes=(
            ObservedOutcome(
                **frame_element_fields("outcome-action"),
                request_unit_ids=(update_request.id,),
                position=0,
                kind="action.api_call",
                predicate="update_address",
                fields={"address": "10 Main Street"},
            ),
            ObservedOutcome(
                **frame_element_fields("outcome-answer"),
                request_unit_ids=(balance_request.id,),
                position=1,
                kind="answer.accounting",
                predicate="report_balance",
                propositions=("account_balance_usd=120",),
            ),
        ),
        extractor_version="test-extractor/1",
    )

    assert InteractionRecord.model_validate_json(record.model_dump_json()) == record
    assert SemanticFrame.model_validate_json(frame.model_dump_json()) == frame
    assert json.loads(frame.model_dump_json())["outcomes"][1]["kind"] == "answer.accounting"


def test_open_vocabularies_and_unresolved_status_are_accepted() -> None:
    factor = SemanticFactor(
        **frame_element_fields("factor-novel", status="unresolved.pending_specialist"),
        kind="customer-defined.medical-code",
        role="triage_mystery_dimension",
        value=None,
    )

    frame = SemanticFrame(
        interaction_id="interaction-open-vocabulary",
        factors=(factor,),
        extractor_version="extractor-experimental",
    )

    assert frame.factors[0].status == "unresolved.pending_specialist"
    assert frame.factors[0].kind == "customer-defined.medical-code"


def test_models_are_frozen_and_strict() -> None:
    frame = SemanticFrame(
        interaction_id="interaction-strict",
        extractor_version="test-extractor/1",
    )

    with pytest.raises(ValidationError, match="frozen"):
        frame.interaction_id = "changed"
    with pytest.raises(ValidationError, match="tuple"):
        SemanticFrame.model_validate(
            {
                "interaction_id": "interaction-list",
                "factors": [],
                "extractor_version": "test-extractor/1",
            }
        )
    with pytest.raises(ValidationError, match="observed output"):
        InteractionRecord(
            id="missing-output",
            raw_input="Do the thing",
            raw_observed_output=None,
        )

    input_only = UserInputRecord(id="input-only", raw_input="Do the thing")
    assert input_only.raw_input == "Do the thing"


def test_frame_rejects_duplicate_and_dangling_identifiers() -> None:
    factor = SemanticFactor(
        **frame_element_fields("duplicate"), kind="entity", role="recipient", value="Ada"
    )

    with pytest.raises(ValidationError, match="globally unique"):
        SemanticFrame(
            interaction_id="interaction-duplicate",
            factors=(factor,),
            request_units=(
                RequestUnit(
                    **frame_element_fields("duplicate"),
                    mode="act",
                    predicate="send",
                    factor_ids=(factor.id,),
                ),
            ),
            extractor_version="test-extractor/1",
        )

    with pytest.raises(ValidationError, match="unknown reference"):
        SemanticFrame(
            interaction_id="interaction-dangling",
            request_units=(
                RequestUnit(
                    **frame_element_fields("request-1"),
                    mode="act",
                    predicate="send",
                    factor_ids=("missing-factor",),
                ),
            ),
            extractor_version="test-extractor/1",
        )


def test_frame_rejects_dangling_relations_request_links_and_duplicate_positions() -> None:
    request = RequestUnit(
        **frame_element_fields("request-1"), mode="decide", predicate="eligibility"
    )

    with pytest.raises(ValidationError, match="unknown reference"):
        SemanticFrame(
            interaction_id="interaction-relation",
            request_units=(request,),
            relations=(
                SemanticRelation(
                    **frame_element_fields("relation-1"),
                    kind="condition",
                    source_ids=(request.id,),
                    target_ids=("missing",),
                ),
            ),
            extractor_version="test-extractor/1",
        )

    with pytest.raises(ValidationError, match="unknown reference"):
        SemanticFrame(
            interaction_id="interaction-outcome-link",
            outcomes=(
                ObservedOutcome(
                    **frame_element_fields("outcome-1"),
                    request_unit_ids=("missing-request",),
                    position=0,
                    kind="decision",
                    predicate="eligible",
                ),
            ),
            extractor_version="test-extractor/1",
        )

    first_outcome = ObservedOutcome(
        **frame_element_fields("outcome-first"),
        request_unit_ids=(request.id,),
        position=0,
        kind="answer",
        predicate="first",
    )
    second_outcome = ObservedOutcome(
        **frame_element_fields("outcome-second"),
        request_unit_ids=(request.id,),
        position=0,
        kind="action",
        predicate="second",
    )
    with pytest.raises(ValidationError, match="positions must be unique"):
        SemanticFrame(
            interaction_id="interaction-position",
            request_units=(request,),
            outcomes=(first_outcome, second_outcome),
            extractor_version="test-extractor/1",
        )


def test_evidence_requires_valid_pointer_and_nonempty_quote() -> None:
    evidence = EvidenceReference(
        source="output",
        json_pointer="/raw_observed_output/action/arguments/~0key~1path",
        text_quote="evidence",
    )
    assert evidence.text_quote == "evidence"

    with pytest.raises(ValidationError, match="at least 1 character"):
        EvidenceReference(source="input", json_pointer="/raw_input", text_quote="")
    with pytest.raises(ValidationError, match="RFC 6901"):
        EvidenceReference(
            source="output",
            json_pointer="action/~2invalid",
            text_quote=None,
        )
