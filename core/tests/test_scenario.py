import json

import pytest
from pydantic import ValidationError
from ul_core.models import (
    Action,
    ActionEffect,
    Actor,
    Artifact,
    Resource,
    Scenario,
    ScenarioProvenance,
)


def test_scenario_is_json_safe_and_round_trips() -> None:
    scenario = Scenario(
        id="case-1",
        title="Execute an approved operation",
        objective="Complete the requested operation exactly once.",
        actors=(Actor(id="requester", role="requester"),),
        artifacts=(Artifact(id="document-1", kind="document", attributes={"amount": 10}),),
        resources=(Resource(id="account-1", kind="account", owner_actor_id="requester"),),
        actions=(
            Action(
                id="write-1",
                kind="execute",
                effect=ActionEffect.WRITE,
                actor_id="requester",
                artifact_ids=("document-1",),
                resource_ids=("account-1",),
            ),
        ),
        provenance=ScenarioProvenance(source="production_trace", source_reference="trace-1"),
        metadata={"semantic_tags": ["approved", "single"]},
    )

    serialized = scenario.model_dump_json()

    assert json.loads(serialized)["artifacts"][0]["attributes"] == {"amount": 10}
    assert Scenario.model_validate_json(serialized) == scenario


def test_scenario_rejects_dangling_references() -> None:
    with pytest.raises(ValidationError, match="unknown artifact referenced"):
        Scenario(
            id="invalid",
            title="Invalid",
            objective="Reject invalid references.",
            actions=(
                Action(
                    id="write-1",
                    kind="execute",
                    effect=ActionEffect.WRITE,
                    artifact_ids=("missing",),
                ),
            ),
            provenance=ScenarioProvenance(source="test"),
        )


def test_scenario_rejects_non_json_metadata() -> None:
    with pytest.raises(ValidationError):
        Scenario(
            id="invalid-json",
            title="Invalid JSON",
            objective="Reject non-JSON values.",
            provenance=ScenarioProvenance(source="test"),
            metadata={"unsafe": object()},
        )


def test_scenario_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValidationError):
        Scenario(
            id="invalid-number",
            title="Invalid number",
            objective="Reject non-finite values.",
            provenance=ScenarioProvenance(source="test"),
            metadata={"value": float("nan")},
        )
