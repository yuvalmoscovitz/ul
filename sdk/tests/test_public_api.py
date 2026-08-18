from ul import (
    DatasetAugmentationEngine,
    DatasetAugmentationOperator,
    DatasetEvaluationBaseline,
    DatasetEvaluationFinding,
    DatasetEvaluationOutcomeGroup,
    DatasetEvaluationRunner,
    DatasetEvaluationTrial,
    DatasetEvaluationTrialSet,
    DatasetInvariantArmEvaluation,
    DatasetInvariantEvaluation,
    DatasetInvariantRuleEvaluation,
    DatasetInvariantSuite,
    DatasetInvariantTrialEvaluation,
    DatasetRegressionCase,
    DatasetRegressionResult,
    DatasetTargetExecutor,
    ExecutionResult,
    ExecutionStatus,
    InteractionRecord,
    JsonHttpDatasetTarget,
    JsonHttpDatasetTargetConfig,
    JsonValuesEqualInvariant,
    ObservedAgentOutput,
    OpenRouterDatasetSettings,
    OpenRouterSemanticDeconstructor,
    PromptManager,
    PromptTemplateInfo,
    RenderedUserInput,
    Scenario,
    ScenarioProvenance,
    SemanticDelta,
    SemanticEquivalenceAssessment,
    SemanticEquivalenceVerifier,
    SemanticFrame,
    UserInputRecord,
    builtin_augmentation_registry,
    builtin_dataset_augmentation_operators,
    create_dataset_regression_case,
    dataset_regression_target_config_sha256,
    evaluate_dataset_invariants,
    load_dataset_invariant_suite,
    load_dataset_regression_case,
    load_json_http_dataset_target_config,
    replay_dataset_regression,
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
    assert DatasetEvaluationBaseline.__name__ == "DatasetEvaluationBaseline"
    assert DatasetEvaluationFinding.__name__ == "DatasetEvaluationFinding"
    assert DatasetEvaluationOutcomeGroup.__name__ == "DatasetEvaluationOutcomeGroup"
    assert DatasetEvaluationRunner.__name__ == "DatasetEvaluationRunner"
    assert DatasetEvaluationTrial.__name__ == "DatasetEvaluationTrial"
    assert DatasetEvaluationTrialSet.__name__ == "DatasetEvaluationTrialSet"
    assert DatasetInvariantArmEvaluation.__name__ == "DatasetInvariantArmEvaluation"
    assert DatasetInvariantEvaluation.__name__ == "DatasetInvariantEvaluation"
    assert DatasetInvariantRuleEvaluation.__name__ == "DatasetInvariantRuleEvaluation"
    assert DatasetInvariantSuite.__name__ == "DatasetInvariantSuite"
    assert DatasetInvariantTrialEvaluation.__name__ == "DatasetInvariantTrialEvaluation"
    assert DatasetRegressionCase.__name__ == "DatasetRegressionCase"
    assert DatasetRegressionResult.__name__ == "DatasetRegressionResult"
    assert DatasetTargetExecutor.__name__ == "DatasetTargetExecutor"
    assert JsonHttpDatasetTarget.__name__ == "JsonHttpDatasetTarget"
    assert JsonHttpDatasetTargetConfig.__name__ == "JsonHttpDatasetTargetConfig"
    assert JsonValuesEqualInvariant.__name__ == "JsonValuesEqualInvariant"
    assert load_json_http_dataset_target_config.__name__ == ("load_json_http_dataset_target_config")
    assert evaluate_dataset_invariants.__name__ == "evaluate_dataset_invariants"
    assert create_dataset_regression_case.__name__ == "create_dataset_regression_case"
    assert dataset_regression_target_config_sha256.__name__ == (
        "dataset_regression_target_config_sha256"
    )
    assert load_dataset_invariant_suite.__name__ == "load_dataset_invariant_suite"
    assert load_dataset_regression_case.__name__ == "load_dataset_regression_case"
    assert replay_dataset_regression.__name__ == "replay_dataset_regression"
    assert len(builtin_dataset_augmentation_operators()) == 8
    assert OpenRouterSemanticDeconstructor is not None
    assert OpenRouterDatasetSettings is not None
    assert PromptManager.instance().list_templates()
    assert PromptTemplateInfo.__name__ == "PromptTemplateInfo"
    assert ObservedAgentOutput(raw_output={"action": "visit_booked"}).raw_output == {
        "action": "visit_booked"
    }
    assert RenderedUserInput(text="hey").text == "hey"
    assert SemanticFrame.__name__ == "SemanticFrame"
    assert SemanticDelta.__name__ == "SemanticDelta"
    assert SemanticEquivalenceAssessment.__name__ == "SemanticEquivalenceAssessment"
    assert SemanticEquivalenceVerifier.__name__ == "SemanticEquivalenceVerifier"
    assert UserInputRecord.__name__ == "UserInputRecord"
