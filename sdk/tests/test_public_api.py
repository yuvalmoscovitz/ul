from ul import (
    DatasetAugmentationEngine,
    DatasetAugmentationOperator,
    DatasetEvaluationFinding,
    DatasetEvaluationRunner,
    DatasetTargetExecutor,
    ExecutionResult,
    ExecutionStatus,
    InteractionRecord,
    ObservedAgentOutput,
    OpenRouterDatasetSettings,
    OpenRouterSemanticDeconstructor,
    RenderedUserInput,
    Scenario,
    ScenarioProvenance,
    SemanticFrame,
    UserInputRecord,
    builtin_augmentation_registry,
    builtin_dataset_augmentation_operators,
)


def test_sdk_exposes_scenario_and_builtin_library() -> None:
    scenario = Scenario(
        id="sdk-case",
        title="SDK case",
        objective="Use the public API.",
        provenance=ScenarioProvenance(source="sdk-test"),
    )

    assert scenario.id == "sdk-case"
    assert len(builtin_augmentation_registry().list()) == 9


def test_sdk_execution_result_preserves_json_safe_provider_evidence() -> None:
    execution = ExecutionResult(
        scenario_id="sdk-case",
        status=ExecutionStatus.SUCCEEDED,
        cost_usd=0.001,
        metadata={
            "provider": "openrouter",
            "requested_model": "provider/model",
            "generation_id": "generation-1",
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            "config": {"max_steps": 8},
        },
    )

    restored = ExecutionResult.model_validate_json(execution.model_dump_json())

    assert restored == execution
    assert restored.metadata["generation_id"] == "generation-1"


def test_sdk_exposes_dataset_augmentation_api() -> None:
    record = InteractionRecord(
        id="observed-interaction",
        raw_input="Book a visit tomorrow.",
        raw_observed_output={"action": "visit_booked"},
    )

    assert record.raw_observed_output == {"action": "visit_booked"}
    assert DatasetAugmentationEngine is not None
    assert DatasetAugmentationOperator.__name__ == "DatasetAugmentationOperator"
    assert DatasetEvaluationFinding.__name__ == "DatasetEvaluationFinding"
    assert DatasetEvaluationRunner.__name__ == "DatasetEvaluationRunner"
    assert DatasetTargetExecutor.__name__ == "DatasetTargetExecutor"
    assert len(builtin_dataset_augmentation_operators()) == 7
    assert OpenRouterSemanticDeconstructor is not None
    assert OpenRouterDatasetSettings is not None
    assert ObservedAgentOutput(raw_output={"action": "visit_booked"}).raw_output == {
        "action": "visit_booked"
    }
    assert RenderedUserInput(text="hey").text == "hey"
    assert SemanticFrame.__name__ == "SemanticFrame"
    assert UserInputRecord.__name__ == "UserInputRecord"
