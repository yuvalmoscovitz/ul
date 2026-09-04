from __future__ import annotations

from dataclasses import dataclass

from pydantic import JsonValue
from ul import (
    DatasetSemanticSettings,
    InteractionRecord,
    RedactionEngine,
    load_dataset_semantic_settings,
)
from ul.dataset_invariants import DatasetInvariantSuite, load_dataset_invariant_suite
from ul.http_environment import JsonHttpTargetConfig

from ul_cli.dataset.evidence.context import build_dataset_evidence_run_context
from ul_cli.dataset_augmentation_ledger import (
    DatasetAugmentationGenerationContext,
    create_dataset_augmentation_generation_context,
    dataset_augmentation_ledger_semantic_settings,
)
from ul_cli.dataset_review import (
    DatasetEvidenceRedactionCoverage,
    DatasetEvidenceRunContext,
)
from ul_cli.dataset_run_config import DatasetRunConfig
from ul_cli.dataset_trial_journal import DatasetRunManifest
from ul_cli.http_target_resolution import (
    ResolvedHttpTarget,
    http_target_evidence_receipt,
)
from ul_cli.local_target_resolution import (
    ResolvedLocalTarget,
    local_target_evidence_receipt,
)

from .operators import dataset_operator_identity, validate_operator_ids
from .records import load_interaction_records, validate_model_input_bounds
from .redaction import (
    calculate_redaction_coverage,
    load_redaction_engine,
    protect_interaction_records,
)
from .request import (
    DatasetEvaluationRequest,
    DatasetRequestError,
    NormalizedDatasetEvaluationRequest,
    normalize_dataset_evaluation_request,
)
from .resume_compatibility import restore_recorded_semantic_settings
from .target_preparation import prepare_evaluation_target


@dataclass(frozen=True)
class PreparedDatasetEvaluation:
    request: NormalizedDatasetEvaluationRequest
    records: tuple[InteractionRecord, ...]
    selected_records: tuple[InteractionRecord, ...]
    selected_operator_ids: tuple[str, ...]
    invariant_suite: DatasetInvariantSuite | None
    redaction_engine: RedactionEngine | None
    redaction_coverage: tuple[DatasetEvidenceRedactionCoverage, ...]
    local_target: ResolvedLocalTarget | None
    http_target: ResolvedHttpTarget | None
    target_config: JsonHttpTargetConfig | None
    run_config: DatasetRunConfig
    settings: DatasetSemanticSettings
    run_context: DatasetEvidenceRunContext | None
    augmentation_generation_context: DatasetAugmentationGenerationContext


def prepare_dataset_evaluation(
    raw_request: DatasetEvaluationRequest,
) -> PreparedDatasetEvaluation:
    request = normalize_dataset_evaluation_request(raw_request)
    requested = request.requested
    if requested.data is None:
        assert request.recorded_manifest is not None
        records = request.recorded_manifest.selected_records
    else:
        records = load_interaction_records(requested.data)
    selected_operator_ids = validate_operator_ids(
        list(request.operators) if request.operators is not None else None
    )
    invariant_suite = (
        load_dataset_invariant_suite(requested.invariants)
        if requested.invariants is not None
        else (
            request.recorded_manifest.effective_command.invariant_suite_snapshot
            if request.recorded_manifest is not None
            else None
        )
    )
    selected_records = records[: request.limit]
    redaction_engine = load_redaction_engine(
        requested.redaction_policy,
        request.redaction_state,
        state_required=not requested.dry_run or requested.resume is not None,
        policy_snapshot=(
            request.recorded_manifest.effective_command.redaction_policy_snapshot
            if requested.redaction_policy is None and request.recorded_manifest is not None
            else None
        ),
    )
    if requested.expected_redaction_policy_sha256 is not None and (
        redaction_engine is None
        or redaction_engine.policy.digest != requested.expected_redaction_policy_sha256
    ):
        raise ValueError(
            "redaction policy changed since 'ul init'; reinitialize the project before sending "
            "data to the semantic provider"
        )
    redaction_coverage = (
        request.recorded_manifest.run_context.redaction_coverage
        if requested.data is None and request.recorded_manifest is not None
        else calculate_redaction_coverage(selected_records, redaction_engine)
    )
    if (
        requested.data is not None
        and redaction_engine is not None
        and (not requested.dry_run or requested.resume is not None)
    ):
        selected_records = protect_interaction_records(selected_records, redaction_engine)

    prepared_target = prepare_evaluation_target(
        request,
        selected_records=selected_records,
        selected_operator_ids=selected_operator_ids,
        invariant_suite=invariant_suite,
    )
    if not requested.dry_run and requested.resume is None:
        if request.output is None:
            raise DatasetRequestError("execution requires --output", parameter="--output")
        if request.output.exists():
            raise DatasetRequestError(
                "output already exists; UL will not overwrite it", parameter="--output"
            )
        if request.augmentations_output is not None and request.augmentations_output.exists():
            raise DatasetRequestError(
                "augmentations output already exists; UL will not overwrite it",
                parameter="--augmentations-output",
            )

    settings = (
        restore_recorded_semantic_settings(request.recorded_manifest)
        if request.recorded_manifest is not None
        else load_dataset_semantic_settings()
    )
    validate_model_input_bounds(selected_records, settings.max_input_chars)
    direct_http_receipt = _direct_http_target_receipt(
        prepared_target.http_target, request.recorded_manifest
    )
    run_context = (
        build_dataset_evidence_run_context(
            selected_records=selected_records,
            selected_operator_ids=selected_operator_ids,
            run_config=prepared_target.run_config,
            invariant_suite=invariant_suite,
            target_config=(None if direct_http_receipt is not None else prepared_target.config),
            target_receipt=(
                local_target_evidence_receipt(prepared_target.local_target)
                if prepared_target.local_target is not None
                else direct_http_receipt
            ),
            settings=settings,
            redaction_policy_sha256=(
                redaction_engine.policy.digest if redaction_engine is not None else None
            ),
            redaction_coverage=redaction_coverage,
        )
        if prepared_target.config is not None or prepared_target.local_target is not None
        else None
    )
    augmentation_generation_context = create_dataset_augmentation_generation_context(
        selected_records=selected_records,
        operators=tuple(
            dataset_operator_identity(operator_reference)
            for operator_reference in selected_operator_ids
        ),
        semantic_settings=dataset_augmentation_ledger_semantic_settings(settings),
        redaction_policy_sha256=(
            redaction_engine.policy.digest if redaction_engine is not None else None
        ),
        source_outcome_projection_sha256=_source_outcome_projection_sha256(
            prepared_target.config,
            prepared_target.local_target,
        ),
    )
    return PreparedDatasetEvaluation(
        request=request,
        records=records,
        selected_records=selected_records,
        selected_operator_ids=selected_operator_ids,
        invariant_suite=invariant_suite,
        redaction_engine=redaction_engine,
        redaction_coverage=redaction_coverage,
        local_target=prepared_target.local_target,
        http_target=prepared_target.http_target,
        target_config=prepared_target.config,
        run_config=prepared_target.run_config,
        settings=settings,
        run_context=run_context,
        augmentation_generation_context=augmentation_generation_context,
    )


def _source_outcome_projection_sha256(
    target_config: JsonHttpTargetConfig | None,
    local_target: ResolvedLocalTarget | None,
) -> str | None:
    projection = local_target.config.outcome if local_target is not None else None
    if projection is None and target_config is not None:
        projection = target_config.outcome
    return projection.digest if projection is not None else None


def _direct_http_target_receipt(
    resolved_target: ResolvedHttpTarget | None,
    recorded_manifest: DatasetRunManifest | None,
) -> dict[str, JsonValue] | None:
    if resolved_target is not None:
        return http_target_evidence_receipt(resolved_target)
    if (
        recorded_manifest is not None
        and recorded_manifest.effective_command.http_target_config is not None
    ):
        return recorded_manifest.run_context.target.receipt
    return None
