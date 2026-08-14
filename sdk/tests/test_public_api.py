from ul import (
    ExecutionResult,
    ExecutionStatus,
    Scenario,
    ScenarioProvenance,
    builtin_augmentation_registry,
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
