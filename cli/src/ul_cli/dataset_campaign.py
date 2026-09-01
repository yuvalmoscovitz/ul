from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from ul import (
    DatasetAugmentationResult,
    DatasetSemanticSettings,
    EvaluatorPreflightProfilePlan,
    InteractionRecord,
    plan_evaluator_preflight_profiles,
    resolve_dataset_augmentation_operator,
)
from ul_core.augmentations.definitions import (
    BuiltinAugmentationSpec,
    builtin_augmentation_catalog,
)

from ul_cli.dataset_run_config import DatasetRunConfig


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CampaignOperatorPlan(_StrictModel):
    id: str
    version: str
    status: Literal["eligible", "conditional", "ineligible"]
    selected: bool
    applicability_profile: Literal["broad", "conditional"]
    applicability_rule: str
    reasons: tuple[str, ...] = ()
    candidate_input_available: bool = False
    candidate_input: str | None = None


class CampaignExamplePlan(_StrictModel):
    interaction_id: str
    operators: tuple[CampaignOperatorPlan, ...]


class CampaignCallCounts(_StrictModel):
    basis: Literal["authorized_maximum"] = "authorized_maximum"
    baseline: int = Field(ge=0)
    variation: int = Field(ge=0)
    repetitions: int = Field(ge=1)
    repetition_executions: int = Field(ge=0)
    retries: int = Field(ge=0)
    preflight: int = Field(ge=0)
    evaluators: int = Field(ge=0)
    materiality: int = Field(ge=0)
    variation_generation: int = Field(ge=0)
    total_semantic_model: int = Field(ge=0)
    total_environment_api: int = Field(ge=0)


class CampaignTokenRange(_StrictModel):
    minimum: int = Field(ge=0)
    maximum: int = Field(ge=0)
    scope: Literal["completion_tokens"] = "completion_tokens"


class CampaignMoneyRange(_StrictModel):
    currency: Literal["USD"] = "USD"
    minimum: float = Field(ge=0)
    maximum: float = Field(ge=0)


class CampaignFixturePlan(_StrictModel):
    status: Literal["configured", "missing", "not_required"]
    id: str | None = None
    version: str | None = None


class CampaignTimingPlan(_StrictModel):
    target_trial_timeout_seconds: float = Field(gt=0, le=3_600)
    maximum_wall_time_seconds: float = Field(gt=0)


class DatasetCampaignPlan(_StrictModel):
    schema_version: Literal["1.3.0"] = "1.3.0"
    evaluation_mode: Literal["variance"] = "variance"
    fixture: CampaignFixturePlan | None = None
    examples: tuple[CampaignExamplePlan, ...]
    calls: CampaignCallCounts
    preflight_profiles: tuple[EvaluatorPreflightProfilePlan, ...]
    tokens: CampaignTokenRange
    money: CampaignMoneyRange | None = None
    timing: CampaignTimingPlan
    warnings: tuple[str, ...] = ()
    inspection_model_calls: Literal[0] = 0
    inspection_environment_calls: Literal[0] = 0


def create_dataset_campaign_plan(
    *,
    records: tuple[InteractionRecord, ...],
    selected_operator_ids: tuple[str, ...],
    run_config: DatasetRunConfig,
    settings: DatasetSemanticSettings,
    saved_augmentations: dict[str, DatasetAugmentationResult] | None = None,
    show_sensitive_values: bool = False,
    requires_preflight: bool = True,
    fixture_status: Literal["configured", "missing", "not_required"] | None = None,
    fixture_id: str | None = None,
    fixture_version: str | None = None,
) -> DatasetCampaignPlan:
    repetitions = run_config.repetitions
    target_config = run_config.target
    saved = saved_augmentations or {}
    selected_ids = {reference.partition("@")[0] for reference in selected_operator_ids}
    catalog = builtin_augmentation_catalog().list()
    examples = tuple(
        CampaignExamplePlan(
            interaction_id=record.id,
            operators=tuple(
                _operator_plan(
                    operator,
                    selected=operator.ref.id in selected_ids,
                    saved_augmentation=saved.get(record.id),
                    show_sensitive_values=show_sensitive_values,
                )
                for operator in catalog
            ),
        )
        for record in records
    )

    selected_count = len(records)
    operator_count = len(selected_operator_ids)
    baseline_calls = selected_count * repetitions
    variation_candidate_count = sum(
        _planned_variation_count(
            selected_operator_ids=selected_ids,
            saved_augmentation=saved.get(record.id),
        )
        for record in records
    )
    variation_calls = variation_candidate_count * repetitions
    execution_calls = baseline_calls + variation_calls
    records_without_saved_augmentation = tuple(
        record for record in records if record.id not in saved
    )
    materialization_record_count = len(records_without_saved_augmentation)
    semantic_generation_operator_count = sum(
        resolve_dataset_augmentation_operator(reference).generation_mechanism == "llm"
        for reference in selected_operator_ids
    )
    tone_safety_operator_count = sum(
        resolve_dataset_augmentation_operator(reference).target_communication_kind
        in {"angry", "argumentative"}
        for reference in selected_operator_ids
    )
    generation_calls = materialization_record_count * semantic_generation_operator_count
    tone_safety_calls = materialization_record_count * tone_safety_operator_count
    source_deconstruction_calls = materialization_record_count
    candidate_deconstruction_calls = materialization_record_count * operator_count
    equivalence_calls = materialization_record_count * operator_count
    trial_evaluator_calls = execution_calls
    materiality_calls = variation_candidate_count
    evaluator_calls = (
        source_deconstruction_calls
        + candidate_deconstruction_calls
        + equivalence_calls
        + tone_safety_calls
        + trial_evaluator_calls
    )
    preflight_profiles = plan_evaluator_preflight_profiles(settings) if requires_preflight else ()
    preflight_calls = len(preflight_profiles)
    total_semantic_calls = evaluator_calls + materiality_calls + generation_calls + preflight_calls
    maximum_wall_time_seconds = (
        max(1, execution_calls) * target_config.trial_timeout_seconds
        + total_semantic_calls * settings.timeout_seconds
    )

    deconstruction_calls = (
        source_deconstruction_calls + candidate_deconstruction_calls + trial_evaluator_calls
    )
    maximum_completion_tokens = (
        sum(profile.max_completion_tokens for profile in preflight_profiles)
        + deconstruction_calls * settings.max_output_tokens
        + generation_calls * settings.max_render_tokens
        + equivalence_calls * min(settings.max_output_tokens, 1_024)
        + tone_safety_calls * min(settings.max_output_tokens, 1_024)
        + materiality_calls * 512
    )
    warnings = list(_model_parameter_warnings(settings))
    candidate_inputs_available = any(
        operator.candidate_input_available for example in examples for operator in example.operators
    )
    if candidate_inputs_available and show_sensitive_values:
        warnings.append(
            "Candidate inputs come from the private augmentation ledger and may contain "
            "sensitive data."
        )
    elif candidate_inputs_available:
        warnings.append(
            "Private candidate inputs are available but omitted; use --show-sensitive-values "
            "only when terminal and JSON output may contain sensitive data."
        )
    if any(
        operator.ref.id in selected_ids
        and any(
            binding.mode == "dataset_variation" and binding.requirements.human_review
            for binding in operator.bindings
        )
        for operator in catalog
    ):
        warnings.append(
            "Selected operators can require human review; no automatic customer evaluator is "
            "configured."
        )
    warnings.append(
        "No trusted model pricing is configured, so a monetary estimate is unavailable."
    )
    if fixture_status == "missing":
        warnings.append(
            "stateful target has no fixture identity; add fixture_id and fixture_version so "
            "findings can be reproduced."
        )
    return DatasetCampaignPlan(
        evaluation_mode=run_config.evaluation_mode,
        fixture=(
            CampaignFixturePlan(
                status=fixture_status,
                id=fixture_id,
                version=fixture_version,
            )
            if fixture_status is not None
            else None
        ),
        examples=examples,
        calls=CampaignCallCounts(
            baseline=baseline_calls,
            variation=variation_calls,
            repetitions=repetitions,
            repetition_executions=execution_calls,
            retries=0,
            preflight=preflight_calls,
            evaluators=evaluator_calls,
            materiality=materiality_calls,
            variation_generation=generation_calls,
            total_semantic_model=total_semantic_calls,
            total_environment_api=(execution_calls * target_config.environment_api_calls_per_trial),
        ),
        preflight_profiles=preflight_profiles,
        tokens=CampaignTokenRange(minimum=0, maximum=maximum_completion_tokens),
        timing=CampaignTimingPlan(
            target_trial_timeout_seconds=target_config.trial_timeout_seconds,
            maximum_wall_time_seconds=maximum_wall_time_seconds,
        ),
        warnings=tuple(warnings),
    )


def _operator_plan(
    operator: BuiltinAugmentationSpec,
    *,
    selected: bool,
    saved_augmentation: DatasetAugmentationResult | None,
    show_sensitive_values: bool,
) -> CampaignOperatorPlan:
    dataset_binding = next(
        (binding for binding in operator.bindings if binding.mode == "dataset_variation"), None
    )
    if dataset_binding is None:
        return CampaignOperatorPlan(
            id=operator.ref.id,
            version=operator.ref.version,
            status="ineligible",
            selected=False,
            applicability_profile=operator.applicability_profile,
            applicability_rule=operator.applicability_rule,
            reasons=("operator is not available in dataset evaluation mode",),
        )
    candidate = (
        next(
            (
                candidate
                for candidate in saved_augmentation.candidates
                if candidate.operator_id == operator.ref.id
            ),
            None,
        )
        if saved_augmentation is not None
        else None
    )
    if candidate is not None:
        return CampaignOperatorPlan(
            id=operator.ref.id,
            version=operator.ref.version,
            status="eligible" if candidate.passed else "ineligible",
            selected=selected,
            applicability_profile=operator.applicability_profile,
            applicability_rule=operator.applicability_rule,
            reasons=(
                ("saved candidate passed semantic qualification",)
                if candidate.passed
                else candidate.failure_reasons
            ),
            candidate_input_available=True,
            candidate_input=(candidate.augmented_input if show_sensitive_values else None),
        )
    saved_skip = (
        next(
            (skip for skip in saved_augmentation.skips if skip.operator_id == operator.ref.id),
            None,
        )
        if saved_augmentation is not None
        else None
    )
    if saved_skip is not None:
        return CampaignOperatorPlan(
            id=operator.ref.id,
            version=operator.ref.version,
            status="ineligible",
            selected=selected,
            applicability_profile=operator.applicability_profile,
            applicability_rule=operator.applicability_rule,
            reasons=(saved_skip.reason,),
        )
    if saved_augmentation is not None and any(
        reference.id == operator.ref.id for reference in saved_augmentation.operator_references
    ):
        return CampaignOperatorPlan(
            id=operator.ref.id,
            version=operator.ref.version,
            status="ineligible",
            selected=selected,
            applicability_profile=operator.applicability_profile,
            applicability_rule=operator.applicability_rule,
            reasons=("saved semantic qualification produced no candidate",),
        )
    deterministic_reason = (
        "candidate materialization is deterministic and free after source semantics are known; "
        "no candidate was generated during this zero-call inspection"
        if resolve_dataset_augmentation_operator(operator.ref.id).generation_mechanism
        == "deterministic"
        else "candidate generation requires a semantic model call"
    )
    return CampaignOperatorPlan(
        id=operator.ref.id,
        version=operator.ref.version,
        status="conditional",
        selected=selected,
        applicability_profile=operator.applicability_profile,
        applicability_rule=operator.applicability_rule,
        reasons=(
            *(("operator was not selected",) if not selected else ()),
            (
                "broad applicability still requires semantic source qualification during execution"
                if operator.applicability_profile == "broad"
                else operator.applicability_rule
            ),
            deterministic_reason,
        ),
    )


def _planned_variation_count(
    *,
    selected_operator_ids: set[str],
    saved_augmentation: DatasetAugmentationResult | None,
) -> int:
    if saved_augmentation is None:
        return len(selected_operator_ids)
    return sum(
        candidate.passed and candidate.operator_id in selected_operator_ids
        for candidate in saved_augmentation.candidates
    )


def _model_parameter_warnings(settings: DatasetSemanticSettings) -> tuple[str, ...]:
    provider_type = getattr(
        settings,
        "semantic_provider_type",
        "openai-compatible" if settings.semantic_provider_id != "openrouter" else "openrouter",
    )
    if provider_type != "openai-compatible":
        return ()
    return (
        "The configured OpenAI-compatible provider has not declared support for seed, reasoning, "
        "top_p, or strict JSON-schema response parameters; verify compatibility before execution.",
    )
