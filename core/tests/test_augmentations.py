from ul_core.augmentation import builtin_augmentation_registry
from ul_core.models import (
    Action,
    ActionEffect,
    ActionItem,
    Actor,
    Artifact,
    ConversationRole,
    ConversationTurn,
    EnvironmentEvent,
    EventTiming,
    ItemValidity,
    Policy,
    PolicyBoundary,
    Resource,
    Scenario,
    ScenarioProvenance,
)


def example_scenario() -> Scenario:
    return Scenario(
        id="plausible-operation",
        title="Perform approved batch operation",
        objective="Perform the current approved work once.",
        actors=(Actor(id="requester", role="requester"),),
        artifacts=(Artifact(id="record-1", kind="business_record", label="August request"),),
        resources=(Resource(id="resource-1", kind="controlled_resource"),),
        actions=(
            Action(
                id="read-current-state",
                kind="read_current_state",
                effect=ActionEffect.READ,
                artifact_ids=("record-1",),
                resource_ids=("resource-1",),
            ),
            Action(
                id="execute-operation",
                kind="execute_operation",
                effect=ActionEffect.WRITE,
                actor_id="requester",
                artifact_ids=("record-1",),
                resource_ids=("resource-1",),
                parameters={"amount": 100, "destination": "primary"},
                batch_items=(
                    ActionItem(id="item-1", target_ids=("record-1",)),
                    ActionItem(id="item-2", target_ids=("resource-1",)),
                ),
            ),
        ),
        policies=(
            Policy(
                id="approval-policy",
                kind="approval",
                description="Additional approval is required above the threshold.",
                boundaries=(
                    PolicyBoundary(
                        action_id="execute-operation",
                        parameter="amount",
                        threshold=1000,
                        increment=1,
                    ),
                ),
            ),
        ),
        conversation=(
            ConversationTurn(
                id="request",
                role=ConversationRole.USER,
                actor_id="requester",
                content="Please complete the approved August request.",
            ),
        ),
        provenance=ScenarioProvenance(source="production_trace"),
    )


def test_builtin_library_is_available_without_customer_extensions() -> None:
    registry = builtin_augmentation_registry()

    assert [item.metadata.id for item in registry.list()] == [
        "conversation.ambiguity",
        "conversation.correction_after_first_response",
        "environment.state.change_between_read_write",
        "environment.state.existing_partial_operation",
        "environment.tool.stale_observation",
        "environment.tool.timeout_after_commit",
        "environment.tool.timeout_before_commit",
        "input.batch.mixed_validity",
        "input.policy.boundary_shift",
    ]
    assert registry.applicable(example_scenario()) == registry.list()


def test_every_builtin_produces_valid_derived_scenarios_with_lineage() -> None:
    scenario = example_scenario()

    for augmentation in builtin_augmentation_registry().list():
        results = augmentation.apply(scenario)

        assert results
        for result in results:
            expected_reference = f"{augmentation.metadata.id}@{augmentation.metadata.version}"
            assert result.augmentation_id == augmentation.metadata.id
            assert result.scenario.id.startswith(f"{scenario.id}::{expected_reference}:")
            assert result.scenario.provenance.parent_scenario_id == scenario.id
            assert (
                result.scenario.provenance.lineage[-1].augmentation_id == augmentation.metadata.id
            )
            assert augmentation.validate(scenario, result.scenario).valid
            assert result.changed_paths
            assert result.scenario != scenario
            Scenario.model_validate_json(result.scenario.model_dump_json())


def test_policy_boundary_generates_below_equal_and_above_variants() -> None:
    augmentation = builtin_augmentation_registry().get("input.policy.boundary_shift")

    results = augmentation.apply(example_scenario())

    assert [result.variant for result in results] == ["below", "equal", "above"]
    assert [result.scenario.actions[1].parameters["amount"] for result in results] == [
        999,
        1000,
        1001,
    ]


def test_later_correction_changes_semantics_and_conversation_together() -> None:
    scenario = example_scenario()
    augmentation = builtin_augmentation_registry().get(
        "conversation.correction_after_first_response"
    )

    result = augmentation.apply(scenario)[0]

    assert result.scenario.actions[1].parameters["amount"] == 101
    assert len(result.scenario.conversation) == len(scenario.conversation) + 1
    correction = result.scenario.conversation[-1]
    assert correction.metadata["corrects_action_id"] == "execute-operation"
    assert correction.metadata["value"] == 101
    assert "101" in correction.content


def test_later_correction_uses_adapter_supplied_domain_fact() -> None:
    scenario = example_scenario()
    write_action = scenario.actions[1].model_copy(
        update={"parameters": {"expected_payment_count": 1, "amount": 100}}
    )
    scenario = scenario.model_copy(
        update={
            "actions": (scenario.actions[0], write_action),
            "metadata": {
                "augmentation_hints": {
                    "later_correction": {
                        "action_id": "execute-operation",
                        "parameter": "amount",
                        "corrected_value": 87,
                        "message": "Correction: the remaining amount is 87.",
                    }
                }
            },
        }
    )
    augmentation = builtin_augmentation_registry().get(
        "conversation.correction_after_first_response"
    )

    result = augmentation.apply(Scenario.model_validate(scenario.model_dump()))[0]

    assert result.scenario.actions[1].parameters == {
        "expected_payment_count": 1,
        "amount": 87,
    }
    assert result.scenario.conversation[-1].content == "Correction: the remaining amount is 87."


def test_ambiguity_uses_adapter_supplied_artifact_overrides() -> None:
    scenario = example_scenario().model_copy(
        update={
            "metadata": {
                "augmentation_hints": {
                    "ambiguity": {
                        "source_artifact_id": "record-1",
                        "alternative_id": "record-2",
                        "label": "August request",
                        "state": "revised",
                        "attributes": {"amount": 87},
                        "replacement_text": "the August request",
                    }
                }
            }
        }
    )
    augmentation = builtin_augmentation_registry().get("conversation.ambiguity")

    result = augmentation.apply(Scenario.model_validate(scenario.model_dump()))[0]

    alternative = result.scenario.artifacts[-1]
    assert alternative.id == "record-2"
    assert alternative.state == "revised"
    assert alternative.attributes["amount"] == 87
    assert result.scenario.conversation[0].metadata["ambiguous_reference_ids"] == [
        "record-1",
        "record-2",
    ]


def test_timeout_variants_encode_commit_state_independently_of_domain() -> None:
    registry = builtin_augmentation_registry()

    before = registry.get("environment.tool.timeout_before_commit").apply(example_scenario())[0]
    after = registry.get("environment.tool.timeout_after_commit").apply(example_scenario())[0]

    assert before.scenario.environment_events[-1].payload["commit_state"] == "not_committed"
    assert after.scenario.environment_events[-1].payload["commit_state"] == "committed"
    assert after.scenario.environment_events[-1].payload["acknowledgement"] == "lost"


def test_timeout_augmentation_does_not_duplicate_existing_fault() -> None:
    scenario = example_scenario().model_copy(
        update={
            "environment_events": (
                EnvironmentEvent(
                    id="existing-timeout",
                    kind="timeout",
                    timing=EventTiming.AFTER_ACTION,
                    action_id="execute-operation",
                    payload={"commit_state": "committed"},
                ),
            )
        }
    )
    scenario = Scenario.model_validate(scenario.model_dump())
    augmentation = builtin_augmentation_registry().get("environment.tool.timeout_after_commit")

    assert not augmentation.applicability(scenario).applicable
    assert augmentation.apply(scenario) == ()


def test_state_change_includes_resources_used_only_by_write() -> None:
    scenario = example_scenario()
    write_only_resource = Resource(id="write-only", kind="destination")
    actions = list(scenario.actions)
    actions[1] = actions[1].model_copy(update={"resource_ids": ("write-only",)})
    scenario = scenario.model_copy(
        update={"resources": (*scenario.resources, write_only_resource), "actions": tuple(actions)}
    )
    scenario = Scenario.model_validate(scenario.model_dump())

    result = (
        builtin_augmentation_registry()
        .get("environment.state.change_between_read_write")
        .apply(scenario)[0]
    )

    assert "write-only" in result.scenario.environment_events[-1].target_ids


def test_boundary_shift_skips_semantic_no_op_variant() -> None:
    scenario = example_scenario()
    actions = list(scenario.actions)
    actions[1] = actions[1].model_copy(
        update={"parameters": {**actions[1].parameters, "amount": 1000}}
    )
    scenario = Scenario.model_validate(
        scenario.model_copy(update={"actions": tuple(actions)}).model_dump()
    )

    results = builtin_augmentation_registry().get("input.policy.boundary_shift").apply(scenario)

    assert [result.variant for result in results] == ["below", "above"]


def test_mixed_validity_batch_changes_only_one_item() -> None:
    result = (
        builtin_augmentation_registry()
        .get("input.batch.mixed_validity")
        .apply(example_scenario())[0]
    )

    assert [item.validity for item in result.scenario.actions[1].batch_items] == [
        ItemValidity.VALID,
        ItemValidity.INVALID,
    ]


def test_inapplicable_augmentation_explains_why_and_returns_no_candidates() -> None:
    scenario = Scenario(
        id="conversation-only",
        title="Ask a question",
        objective="Answer without taking action.",
        conversation=(
            ConversationTurn(id="question", role=ConversationRole.USER, content="Status?"),
        ),
        provenance=ScenarioProvenance(source="production_trace"),
    )
    augmentation = builtin_augmentation_registry().get("environment.tool.timeout_after_commit")

    applicability = augmentation.applicability(scenario)

    assert not applicability.applicable
    assert applicability.reasons == ("scenario has no write action",)
    assert augmentation.apply(scenario) == ()


def test_registry_resolves_latest_semantic_version() -> None:
    original = builtin_augmentation_registry().get("conversation.ambiguity")
    from ul_core.augmentation import AugmentationRegistry

    class NewVersion:
        metadata = original.metadata.model_copy(update={"version": "2.0.0"})

        def applicability(self, scenario: Scenario):
            return original.applicability(scenario)

        def apply(self, scenario: Scenario):
            return original.apply(scenario)

        def validate(self, source: Scenario, candidate: Scenario):
            return original.validate(source, candidate)

    registry = AugmentationRegistry((NewVersion(), original))

    assert registry.get("conversation.ambiguity").metadata.version == "2.0.0"
    assert len(registry.list()) == 1
    assert len(registry.list(latest_only=False)) == 2
