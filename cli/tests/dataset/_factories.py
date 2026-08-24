from __future__ import annotations

import hashlib
import json
import re
from types import SimpleNamespace
from typing import Any, cast

from pydantic import SecretStr
from typer.testing import CliRunner
from ul import (
    AugmentationTarget,
    CaseFixtureReference,
    DatasetAugmentationResult,
    DatasetEvaluationBaseline,
    DatasetEvaluationCase,
    DatasetEvaluationFinding,
    DatasetEvaluationOutcomeGroup,
    DatasetEvaluationResult,
    DatasetEvaluationTrial,
    DatasetEvaluationTrialSet,
    EvaluatorModelPreflight,
    InteractionRecord,
    JsonHttpEnvironmentConfig,
    JsonHttpIsolatedResponseConfig,
    ObservedAgentOutput,
    RichInteractionCase,
    SemanticFrame,
    create_dataset_augmentation_projection,
    project_rich_interaction_case,
)
from ul.augmentations.dataset import DatasetAugmentationCandidate
from ul.dataset_invariants import (
    DatasetInvariantArmEvaluation,
    DatasetInvariantEvaluation,
    DatasetInvariantRuleEvaluation,
)
from ul_cli.dataset.evidence import context as context_module

runner = CliRunner()
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _trial_set(
    *,
    requested_repetitions: int = 3,
    stability: str = "stable",
    outcome_group_repetitions: tuple[tuple[int, ...], ...] | None = None,
    representative_effect: object | None = None,
) -> SimpleNamespace:
    if outcome_group_repetitions is None:
        outcome_group_repetitions = (tuple(range(1, requested_repetitions + 1)),)
    grouped_repetitions = {
        repetition for group in outcome_group_repetitions for repetition in group
    }
    trials = tuple(
        SimpleNamespace(
            repetition=repetition,
            target_output=(
                SimpleNamespace(raw_output={"status": "ok"})
                if repetition in grouped_repetitions
                else None
            ),
            inconclusive_reasons=(
                () if repetition in grouped_repetitions else ("target execution failed",)
            ),
        )
        for repetition in range(1, requested_repetitions + 1)
    )
    effects = () if representative_effect is None else (representative_effect,)
    outcome_groups = tuple(
        SimpleNamespace(repetitions=repetitions, representative_effects=effects)
        for repetitions in outcome_group_repetitions
    )
    return SimpleNamespace(
        requested_repetitions=requested_repetitions,
        stability=stability,
        trials=trials,
        outcome_groups=outcome_groups,
    )


def _evaluation_result(
    identifier: str,
    *,
    has_review_finding: bool = False,
) -> DatasetEvaluationResult:
    source = InteractionRecord(
        id=identifier,
        raw_input="Transfer 100 to Alice.",
        raw_observed_output={
            "actions": [{"action": "transfer", "amount": 100, "recipient": "Alice"}]
        },
    )
    source_frame = SemanticFrame(interaction_id=identifier, extractor_version="test")
    baseline_frame = SemanticFrame(
        interaction_id=f"{identifier}:current_baseline:round-1",
        extractor_version="test",
    )
    trial_set = DatasetEvaluationTrialSet(
        requested_repetitions=1,
        stability="stable",
        trials=(
            DatasetEvaluationTrial(
                repetition=1,
                target_output=ObservedAgentOutput(raw_output={"status": "ok"}),
                observed_frame=baseline_frame,
            ),
        ),
        outcome_groups=(
            DatasetEvaluationOutcomeGroup(repetitions=(1,), representative_effects=()),
        ),
    )
    candidate = DatasetAugmentationCandidate(
        source_interaction_id=identifier,
        operator_id="input.surface.rephrase",
        projection=create_dataset_augmentation_projection(source),
        changed_paths=(source.augmentation_path,),
        augmented_input="Please transfer 100 to Alice.",
        expected_input_frame=source_frame,
        reparsed_input_frame=source_frame if has_review_finding else None,
        passed=has_review_finding,
        failure_reasons=() if has_review_finding else ("test rejection",),
    )
    if has_review_finding:
        variation_trial_set = DatasetEvaluationTrialSet(
            requested_repetitions=1,
            stability="stable",
            trials=(
                DatasetEvaluationTrial(
                    repetition=1,
                    target_output=ObservedAgentOutput(raw_output={"status": "changed"}),
                    observed_frame=SemanticFrame(
                        interaction_id=f"{identifier}:input.surface.rephrase:round-1",
                        extractor_version="test",
                    ),
                ),
            ),
            outcome_groups=(
                DatasetEvaluationOutcomeGroup(repetitions=(1,), representative_effects=()),
            ),
        )
        evaluation_case = DatasetEvaluationCase(
            candidate=candidate,
            verdict="divergence_needs_review",
            trial_set=variation_trial_set,
            findings=(
                DatasetEvaluationFinding(
                    category="unexpected_effect",
                    message="The variation changed observable behavior.",
                ),
            ),
        )
    else:
        evaluation_case = DatasetEvaluationCase(
            candidate=candidate,
            verdict="augmentation_rejected",
        )
    return DatasetEvaluationResult(
        source=source,
        augmentation=DatasetAugmentationResult(
            operator_references=({"id": candidate.operator_id, "version": "1.0.0"},),
            source_records=(source,),
            source_frames=(source_frame,),
            candidates=(candidate,),
        ),
        baseline=DatasetEvaluationBaseline(verdict="no_divergence", trial_set=trial_set),
        cases=(evaluation_case,),
    )


def _rich_evaluation_result() -> DatasetEvaluationResult:
    source = project_rich_interaction_case(
        RichInteractionCase(
            id="cancel-order",
            inputs={"customer_id": "cus-7", "message": "Cancel order ord-9."},
            augmentation_targets=(
                AugmentationTarget(
                    id="message", kind="input_field", json_pointer="/inputs/message"
                ),
            ),
            fixture=CaseFixtureReference(id="orders", version="9"),
            observed_output={"status": "cancelled"},
        )
    )[0]
    result = _evaluation_result(source.id)
    candidate = result.cases[0].candidate.model_copy(
        update={
            "source_record_id": source.source_interaction_id,
            "augmentation_target_id": source.augmentation_target.id,
            "projection": create_dataset_augmentation_projection(source),
            "changed_paths": (source.augmentation_path,),
        }
    )
    return result.model_copy(
        update={
            "source": source,
            "augmentation": result.augmentation.model_copy(
                update={"source_records": (source,), "candidates": (candidate,)}
            ),
            "cases": (result.cases[0].model_copy(update={"candidate": candidate}),),
        }
    )


def _settings(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "live_calls": True,
        "allow_external_data_processing": True,
        "api_key": SecretStr("test-key"),
        "model": "test/deconstructor",
        "render_model": "test/renderer",
        "equivalence_model": "test/equivalence",
        "max_input_chars": 50_000,
        "max_output_tokens": 4_096,
        "max_render_tokens": 512,
        "max_response_bytes": 1_000_000,
        "timeout_seconds": 60.0,
        "semantic_provider_id": "openrouter",
        "semantic_provider_type": "openrouter",
        "semantic_base_url": "https://openrouter.ai/api/v1",
        "semantic_endpoint_sha256": (
            "76ef4ad6f0c8a4ae66efb13875c107cee40c78997a212353d379acfbb2f45591"
        ),
        "api_key_required": True,
        "api_key_environment_variable": "OPEN_ROUTER_API_KEY",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _invariant_evaluation(
    baseline_status: str = "satisfied",
    variation_status: str | None = None,
    *,
    interaction_id: str = "case-1",
    suite_sha256: str = "a" * 64,
) -> DatasetInvariantEvaluation:
    def arm_rule(status: str) -> DatasetInvariantRuleEvaluation:
        if status == "satisfied":
            trial_reason = "values_equal"
            aggregate_reason = "all_trials_satisfied"
            resolved_values = {"left": 100, "right": 100}
        elif status == "violated":
            trial_reason = "values_differ"
            aggregate_reason = "one_or_more_trials_violated"
            resolved_values = {"left": 200, "right": 100}
        else:
            trial_reason = "left_pointer_missing"
            aggregate_reason = "one_or_more_trials_not_evaluable"
            resolved_values = {}
        return DatasetInvariantRuleEvaluation.model_validate(
            {
                "rule_type": "json_values_equal",
                "rule_id": "amount-matches-corrected",
                "rule_version": "1.0.0",
                "description": "Final amount equals the corrected amount.",
                "severity": "high",
                "status": status,
                "reason_code": aggregate_reason,
                "trials": (
                    {
                        "repetition": 1,
                        "status": status,
                        "reason_code": trial_reason,
                        "left_pointer": "/final_amount",
                        "right_pointer": "/corrected_amount",
                        "resolved_values": resolved_values,
                    },
                ),
            }
        )

    variations = (
        ()
        if variation_status is None
        else (
            DatasetInvariantArmEvaluation(
                arm="variation",
                operator_id="input.surface.rephrase",
                rules=(arm_rule(variation_status),),
            ),
        )
    )
    return DatasetInvariantEvaluation(
        interaction_id=interaction_id,
        suite_sha256=suite_sha256,
        observation_authority="committed_state_snapshot",
        baseline=DatasetInvariantArmEvaluation(
            arm="baseline",
            rules=(arm_rule(baseline_status),),
        ),
        variations=variations,
    )


def _isolated_response_target_config() -> JsonHttpIsolatedResponseConfig:
    return JsonHttpIsolatedResponseConfig.model_validate(
        {
            "version": 1,
            "adapter_tier": "isolated_response",
            "environment_id": "isolated-test",
            "request_isolation_attested": True,
            "safe_test_target_attested": True,
            "execute": {
                "url": "https://environment.example.test/execute",
                "request_json_template": {
                    "case_id": "{{case_id}}",
                    "turn_id": "{{turn_id}}",
                    "input": "{{input}}",
                },
                "response_json_pointer": "/response",
            },
        }
    )


def _run_context(
    records: tuple[InteractionRecord, ...],
    *,
    invariant_suite: object | None = None,
    target_config: object | None = None,
) -> object:
    return context_module.build_dataset_evidence_run_context(
        selected_records=records,
        selected_operator_ids=("input.surface.rephrase",),
        repetitions=1,
        invariant_suite=cast(Any, invariant_suite),
        target_config=cast(Any, target_config)
        if target_config is not None
        else JsonHttpEnvironmentConfig.model_validate(
            {
                "version": 5,
                "environment_id": "test-environment",
                "reset": {
                    "url": "https://environment.example.test/reset",
                    "request_json_template": {"case_id": "{{case_id}}"},
                    "case_id_json_pointer": "/case_id",
                    "generation_json_pointer": "/generation",
                    "clean_state_json_pointer": "/clean",
                    "clean_state_value": True,
                },
                "execute_turn": {
                    "url": "https://environment.example.test/execute",
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "turn_id": "{{turn_id}}",
                        "input": "{{input}}",
                    },
                    "case_id_json_pointer": "/case_id",
                    "turn_id_json_pointer": "/turn_id",
                },
                "snapshot": {
                    "url": "https://environment.example.test/snapshot",
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "turn_id": "{{turn_id}}",
                    },
                    "case_id_json_pointer": "/case_id",
                    "turn_id_json_pointer": "/turn_id",
                },
            }
        ),
        settings=cast(Any, _settings()),
    )


def _evaluator_preflight() -> EvaluatorModelPreflight:
    provider_options = {
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
    }

    def request_options_sha256(
        model: str,
        reasoning_effort: str,
        temperature: int | float,
        seed: int,
        *,
        max_tokens: int = 1_024,
        top_p: float | None = None,
    ) -> str:
        options: dict[str, object] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "reasoning": {"effort": reasoning_effort},
            "provider": provider_options,
        }
        if top_p is not None:
            options["top_p"] = top_p
        serialized = json.dumps(options, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()

    render_seed = (
        int.from_bytes(
            hashlib.sha256(b"UL evaluator preflight\0Check renderer compatibility.").digest()[:4],
            "big",
        )
        & 0x7FFF_FFFF
    )
    return EvaluatorModelPreflight.model_validate(
        {
            "provider": "openrouter",
            "endpoint_sha256": _settings().semantic_endpoint_sha256,
            "profiles": (
                {
                    "roles": ("deconstruct",),
                    "requested_model": "test/deconstructor",
                    "routed_model": "test/deconstructor",
                    "upstream_provider": "test-provider",
                    "required_parameters": (
                        "response_format",
                        "seed",
                        "temperature",
                        "max_tokens",
                        "reasoning",
                    ),
                    "request_options_sha256": request_options_sha256(
                        "test/deconstructor", "minimal", 0, 0
                    ),
                    "parameter_support": "routing_enforced",
                    "unverified_options": (),
                },
                {
                    "roles": ("render",),
                    "requested_model": "test/renderer",
                    "routed_model": "test/renderer",
                    "upstream_provider": "test-provider",
                    "required_parameters": (
                        "response_format",
                        "seed",
                        "temperature",
                        "max_tokens",
                        "reasoning",
                        "top_p",
                    ),
                    "request_options_sha256": request_options_sha256(
                        "test/renderer",
                        "none",
                        0.7,
                        render_seed,
                        max_tokens=512,
                        top_p=0.95,
                    ),
                    "parameter_support": "routing_enforced",
                    "unverified_options": (),
                },
                {
                    "roles": ("equivalence",),
                    "requested_model": "test/equivalence",
                    "routed_model": "test/equivalence",
                    "upstream_provider": "test-provider",
                    "required_parameters": (
                        "response_format",
                        "seed",
                        "temperature",
                        "max_tokens",
                        "reasoning",
                    ),
                    "request_options_sha256": request_options_sha256(
                        "test/equivalence", "low", 0, 0
                    ),
                    "parameter_support": "routing_enforced",
                    "unverified_options": (),
                },
            ),
            "verified_capabilities": (
                "routing",
                "structured_output",
                "required_parameters",
            ),
            "ignored_or_unsupported_options": (),
            "unverified_options": (),
            "data_policy": {
                "external_processing": True,
                "provider_policy_declared": True,
                "data_collection": "deny",
                "zero_data_retention_required": True,
                "implication": (
                    "The configured route requires data collection to be denied and zero data "
                    "retention; the evaluator request is still processed externally."
                ),
            },
        }
    )
