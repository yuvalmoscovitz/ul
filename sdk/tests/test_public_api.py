from ul import (
    DatasetAugmentationEngine,
    DatasetAugmentationOperator,
    DatasetEvaluationBaseline,
    DatasetEvaluationFinding,
    DatasetEvaluationMode,
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
    EvaluationCaseResult,
    EvaluatorCalibrationExample,
    EvaluatorCalibrationReport,
    ExecutionResult,
    ExecutionStatus,
    FindingBundle,
    FindingOtlpEvent,
    InteractionRecord,
    JsonHttpEnvironmentConfig,
    JsonHttpEnvironmentConnection,
    JsonValuesEqualInvariant,
    LocalPseudonymStore,
    ObservedAgentOutput,
    OpenAICompatibleDatasetSettings,
    OpenRouterDatasetSettings,
    OtlpJsonHttpReceiver,
    OtlpObservationConfig,
    OtlpObservationSource,
    ProbeExecutionIdentity,
    ProjectionContract,
    ProjectionTarget,
    ProjectionTargetOperation,
    PromptManager,
    PromptTemplateInfo,
    RedactedSemanticPipeline,
    RedactionEngine,
    RedactionPolicy,
    RedactionRule,
    RenderedUserInput,
    Scenario,
    ScenarioProvenance,
    SemanticDelta,
    SemanticEquivalenceAssessment,
    SemanticEquivalenceVerifier,
    SemanticFrame,
    SemanticModelDeconstructor,
    UserInputRecord,
    WorkerTraceFlusher,
    append_finding_annotations,
    builtin_augmentation_registry,
    builtin_dataset_augmentation_operators,
    calibrate_evaluator,
    create_dataset_augmentation_projection,
    create_dataset_regression_case,
    create_semantic_model_deconstructor,
    dataset_regression_target_config_sha256,
    evaluate_case,
    evaluate_dataset_invariants,
    load_dataset_invariant_suite,
    load_dataset_regression_case,
    load_dataset_semantic_settings,
    load_json_http_environment_config,
    load_redaction_policy,
    replay_dataset_regression,
    safe_finding_bundle_json,
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
    assert ProjectionContract.__name__ == "ProjectionContract"
    assert ProjectionTarget.__name__ == "ProjectionTarget"
    assert ProjectionTargetOperation == ProjectionTargetOperation
    assert create_dataset_augmentation_projection.__name__ == (
        "create_dataset_augmentation_projection"
    )


def test_sdk_exposes_neutral_finding_export_api() -> None:
    assert FindingBundle.__name__ == "FindingBundle"
    assert FindingOtlpEvent.__name__ == "FindingOtlpEvent"
    assert append_finding_annotations.__name__ == "append_finding_annotations"
    assert safe_finding_bundle_json.__name__ == "safe_finding_bundle_json"


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
    assert DatasetEvaluationMode is not None
    assert DatasetEvaluationFinding.__name__ == "DatasetEvaluationFinding"
    assert DatasetEvaluationOutcomeGroup.__name__ == "DatasetEvaluationOutcomeGroup"
    assert DatasetEvaluationRunner.__name__ == "DatasetEvaluationRunner"
    assert DatasetEvaluationTrial.__name__ == "DatasetEvaluationTrial"
    assert DatasetEvaluationTrialSet.__name__ == "DatasetEvaluationTrialSet"
    assert EvaluationCaseResult.__name__ == "EvaluationCaseResult"
    assert EvaluatorCalibrationExample.__name__ == "EvaluatorCalibrationExample"
    assert EvaluatorCalibrationReport.__name__ == "EvaluatorCalibrationReport"
    assert calibrate_evaluator.__name__ == "calibrate_evaluator"
    assert evaluate_case.__name__ == "evaluate_case"
    assert DatasetInvariantArmEvaluation.__name__ == "DatasetInvariantArmEvaluation"
    assert DatasetInvariantEvaluation.__name__ == "DatasetInvariantEvaluation"
    assert DatasetInvariantRuleEvaluation.__name__ == "DatasetInvariantRuleEvaluation"
    assert DatasetInvariantSuite.__name__ == "DatasetInvariantSuite"
    assert DatasetInvariantTrialEvaluation.__name__ == "DatasetInvariantTrialEvaluation"
    assert DatasetRegressionCase.__name__ == "DatasetRegressionCase"
    assert DatasetRegressionResult.__name__ == "DatasetRegressionResult"
    assert JsonHttpEnvironmentConnection.__name__ == "JsonHttpEnvironmentConnection"
    assert JsonHttpEnvironmentConfig.__name__ == "JsonHttpEnvironmentConfig"
    assert JsonValuesEqualInvariant.__name__ == "JsonValuesEqualInvariant"
    assert LocalPseudonymStore.__name__ == "LocalPseudonymStore"
    assert load_json_http_environment_config.__name__ == ("load_json_http_environment_config")
    assert load_redaction_policy.__name__ == "load_redaction_policy"
    assert evaluate_dataset_invariants.__name__ == "evaluate_dataset_invariants"
    assert create_dataset_regression_case.__name__ == "create_dataset_regression_case"
    assert dataset_regression_target_config_sha256.__name__ == (
        "dataset_regression_target_config_sha256"
    )
    assert load_dataset_invariant_suite.__name__ == "load_dataset_invariant_suite"
    assert load_dataset_regression_case.__name__ == "load_dataset_regression_case"
    assert replay_dataset_regression.__name__ == "replay_dataset_regression"
    assert len(builtin_dataset_augmentation_operators()) == 11
    assert OpenRouterDatasetSettings is not None
    assert OpenAICompatibleDatasetSettings is not None
    assert OtlpObservationConfig.__name__ == "OtlpObservationConfig"
    assert OtlpObservationSource.__name__ == "OtlpObservationSource"
    assert OtlpJsonHttpReceiver.__name__ == "OtlpJsonHttpReceiver"
    assert ProbeExecutionIdentity.__name__ == "ProbeExecutionIdentity"
    assert WorkerTraceFlusher.__name__ == "WorkerTraceFlusher"
    assert SemanticModelDeconstructor is not None
    assert create_semantic_model_deconstructor is not None
    assert load_dataset_semantic_settings is not None
    assert PromptManager.instance().list_templates()
    assert PromptTemplateInfo.__name__ == "PromptTemplateInfo"
    assert RedactedSemanticPipeline.__name__ == "RedactedSemanticPipeline"
    assert RedactionEngine.__name__ == "RedactionEngine"
    assert RedactionPolicy.__name__ == "RedactionPolicy"
    assert RedactionRule.__name__ == "RedactionRule"
    assert ObservedAgentOutput(raw_output={"action": "visit_booked"}).raw_output == {
        "action": "visit_booked"
    }
    assert RenderedUserInput(text="hey").text == "hey"
    assert SemanticFrame.__name__ == "SemanticFrame"
    assert SemanticDelta.__name__ == "SemanticDelta"
    assert SemanticEquivalenceAssessment.__name__ == "SemanticEquivalenceAssessment"
    assert SemanticEquivalenceVerifier.__name__ == "SemanticEquivalenceVerifier"
    assert UserInputRecord.__name__ == "UserInputRecord"
