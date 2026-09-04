from __future__ import annotations

import asyncio
import shlex
from dataclasses import dataclass
from pathlib import Path

import typer
from ul import EvaluatorModelPreflight
from ul.http_environment import (
    json_http_environment_capabilities,
    json_http_environment_config_urls,
)

from ul_cli.dataset_campaign import DatasetCampaignPlan, create_dataset_campaign_plan
from ul_cli.dataset_trial_journal import (
    DatasetRunManifest,
    create_dataset_run_manifest,
    private_file_sha256,
)

from ..evidence.persistence import load_evaluator_preflight
from ..presentation.evaluation import print_dataset_plan
from .durable_run import (
    DurableRunPreparationError,
    PreparedDurableRun,
    create_or_validate_durable_state,
    prepare_durable_run,
)
from .preparation import PreparedDatasetEvaluation
from .target_preparation import print_http_target_identity, print_local_target_identity


@dataclass(frozen=True)
class PreparedCampaign:
    evaluation: PreparedDatasetEvaluation
    durable: PreparedDurableRun
    manifest: DatasetRunManifest | None
    plan: DatasetCampaignPlan
    evaluator_preflight: EvaluatorModelPreflight | None
    evaluator_preflight_receipt: Path | None


def prepare_campaign(evaluation: PreparedDatasetEvaluation) -> PreparedCampaign | None:
    request = evaluation.request
    requested = request.requested
    try:
        durable = prepare_durable_run(
            resume=requested.resume,
            output=request.output,
            recorded_manifest=request.recorded_manifest,
            run_context=evaluation.run_context,
            selected_records=evaluation.selected_records,
            augmentation_generation_context=evaluation.augmentation_generation_context,
            augmentations_input=request.augmentations_input,
            augmentations_output=request.augmentations_output,
            invariant_suite=evaluation.invariant_suite,
        )
    except DurableRunPreparationError as error:
        prefix, parameter = {
            "journal": ("cannot safely lock durable run state", "--resume"),
            "augmentation_input": (
                "cannot safely reuse augmentation input",
                "--augmentations-input",
            ),
            "resume_evidence": ("cannot safely resume evidence", "--resume"),
        }[error.phase]
        raise typer.BadParameter(f"{prefix} ({error})", param_hint=parameter) from None
    ownership_transferred = False
    try:
        manifest = _prepare_manifest(evaluation, durable)
        evaluator_preflight: EvaluatorModelPreflight | None = None
        evaluator_preflight_receipt: Path | None = None
        if requested.resume is not None and durable.remaining_records:
            assert request.output is not None
            try:
                evaluator_preflight, evaluator_preflight_receipt = asyncio.run(
                    load_evaluator_preflight(request.output, evaluation.settings)
                )
            except ValueError as error:
                raise typer.BadParameter(
                    f"cannot reuse evaluator preflight receipt ({error}); restore the matching "
                    "receipt and semantic settings, or start a new run with a new --output",
                    param_hint="--resume",
                ) from None
        plan = create_dataset_campaign_plan(
            records=durable.remaining_records,
            selected_operator_ids=evaluation.selected_operator_ids,
            run_config=evaluation.run_config,
            settings=evaluation.settings,
            saved_augmentations=durable.saved_augmentations,
            show_sensitive_values=requested.show_sensitive_values,
            requires_preflight=evaluator_preflight is None and bool(durable.remaining_records),
            fixture_status=_fixture_value(evaluation, "status"),
            fixture_id=_fixture_value(evaluation, "id"),
            fixture_version=_fixture_value(evaluation, "version"),
        )
        potential_calls = plan.calls.total_environment_api
        if potential_calls > evaluation.run_config.target.max_environment_api_calls:
            raise typer.BadParameter(
                f"remaining selection would make up to {potential_calls} environment API calls, "
                "exceeding --max-environment-api-calls "
                f"{evaluation.run_config.target.max_environment_api_calls}; reduce --limit, "
                "--operator, or --repetitions, or explicitly raise the call budget"
            )
        campaign = PreparedCampaign(
            evaluation, durable, manifest, plan, evaluator_preflight, evaluator_preflight_receipt
        )
        if requested.dry_run:
            _print_dry_run(campaign)
            return None
        ownership_transferred = True
        return campaign
    finally:
        if not ownership_transferred:
            durable.close()


def _prepare_manifest(
    evaluation: PreparedDatasetEvaluation, durable: PreparedDurableRun
) -> DatasetRunManifest | None:
    request = evaluation.request
    requested = request.requested
    if requested.dry_run and (requested.resume is None or request.recorded_manifest is None):
        return None
    assert request.output is not None
    assert evaluation.run_context is not None
    if (
        request.augmentations_output is not None
        and not request.augmentations_output.parent.is_dir()
    ):
        raise typer.BadParameter(
            "cannot safely open augmentations output (FileNotFoundError)",
            param_hint="--augmentations-output",
        )
    try:
        redaction_state_sha256 = (
            private_file_sha256(request.redaction_state)
            if request.redaction_state is not None
            and (request.recorded_manifest is None or request.redaction_state_was_explicit)
            and request.redaction_state.exists()
            else _recorded_field(request.recorded_manifest, "redaction_state_sha256")
        )
        manifest = create_dataset_run_manifest(
            run_context=evaluation.run_context,
            selected_records=durable.all_records,
            selected_operator_ids=evaluation.selected_operator_ids,
            run_config=evaluation.run_config,
            save_augmentations=request.augmentations_output is not None,
            semantic_provider_type=evaluation.settings.semantic_provider_type,
            semantic_base_url=evaluation.settings.semantic_base_url,
            semantic_live_calls=evaluation.settings.live_calls,
            semantic_allow_external_data_processing=evaluation.settings.allow_external_data_processing,
            invariant_suite_snapshot=evaluation.invariant_suite,
            invariant_suite_source=_path_or_recorded(
                requested.invariants, request.recorded_manifest, "invariant_suite_source"
            ),
            redaction_policy_snapshot=(
                evaluation.redaction_engine.policy if evaluation.redaction_engine else None
            ),
            redaction_policy_source=_path_or_recorded(
                requested.redaction_policy, request.recorded_manifest, "redaction_policy_source"
            ),
            redaction_state_path=_path_or_recorded(
                request.redaction_state,
                request.recorded_manifest,
                "redaction_state_path",
                request.redaction_state_was_explicit,
            ),
            redaction_state_sha256=redaction_state_sha256,
            augmentations_output_path=_path_or_recorded(
                request.augmentations_output,
                request.recorded_manifest,
                "augmentations_output_path",
                request.augmentations_output_was_explicit,
            ),
            augmentations_input_path=_path_or_recorded(
                request.augmentations_input,
                request.recorded_manifest,
                "augmentations_input_path",
                request.augmentations_input_was_explicit,
            ),
            augmentations_input_sha256=durable.augmentations_input_sha256,
            http_target_confirmation=(
                evaluation.http_target.confirmation
                if evaluation.http_target is not None
                else _recorded_field(request.recorded_manifest, "http_target_confirmation")
            ),
            http_target_config=(
                evaluation.http_target.config
                if evaluation.http_target is not None
                else _recorded_field(request.recorded_manifest, "http_target_config")
            ),
        )
        create_or_validate_durable_state(
            prepared=durable,
            resume=requested.resume,
            output=request.output,
            expected_manifest=manifest,
            resolve_quarantine_after=requested.resolve_quarantine_after,
        )
        return manifest
    except (OSError, ValueError) as error:
        message = str(error) if isinstance(error, ValueError) else error.__class__.__name__
        if requested.resume is not None and message.startswith("resume_"):
            message = (
                f"{message}; diagnose with: ul dataset evaluate --resume "
                f"{shlex.quote(str(requested.resume))} --dry-run"
            )
        raise typer.BadParameter(
            f"cannot safely open durable run state ({message})",
            param_hint="--resume" if requested.resume is not None else "--output",
        ) from None


def _recorded_field(manifest: DatasetRunManifest | None, field: str):
    return getattr(manifest.effective_command, field) if manifest is not None else None


def _path_or_recorded(
    path: Path | None,
    manifest: DatasetRunManifest | None,
    field: str,
    explicit: bool = True,
) -> str | None:
    if path is not None and (manifest is None or explicit):
        return str(path.resolve())
    return _recorded_field(manifest, field)


def _fixture_value(evaluation: PreparedDatasetEvaluation, field: str):
    fixture = evaluation.run_context.fixture if evaluation.run_context is not None else None
    return getattr(fixture, field) if fixture is not None else None


def _print_dry_run(campaign: PreparedCampaign) -> None:
    evaluation = campaign.evaluation
    request = evaluation.request
    if evaluation.local_target is not None:
        print_local_target_identity(evaluation.local_target)
    if evaluation.http_target is not None:
        print_http_target_identity(evaluation.http_target)
    print_dataset_plan(
        record_count=len(evaluation.records),
        selected_count=len(campaign.durable.remaining_records),
        skipped_count=campaign.durable.skipped_count,
        operator_ids=evaluation.selected_operator_ids,
        evaluation_mode=request.evaluation_mode,
        target_configured=(
            request.requested.environment_config is not None or request.requested.target is not None
        ),
        target_endpoint=(
            json_http_environment_config_urls(evaluation.target_config)[0]
            if evaluation.target_config is not None
            else None
        ),
        target_header_environment_variables=(
            evaluation.target_config.headers_from_env
            if evaluation.target_config is not None
            else {}
        ),
        repetitions=evaluation.run_config.repetitions,
        target_timeout_seconds=evaluation.run_config.target.trial_timeout_seconds,
        max_environment_api_calls=evaluation.run_config.target.max_environment_api_calls,
        target_calls_per_execution=evaluation.run_config.target.environment_api_calls_per_trial,
        target_supports_state_observation=(
            json_http_environment_capabilities(evaluation.target_config).supports_state_observation
            if evaluation.target_config is not None
            else None
        ),
        fixture_status=_fixture_value(evaluation, "status"),
        fixture_id=_fixture_value(evaluation, "id"),
        fixture_version=_fixture_value(evaluation, "version"),
        invariant_suite=evaluation.invariant_suite,
        output=request.output,
        augmentations_input=request.augmentations_input,
        augmentations_output=request.augmentations_output,
        semantic_provider_id=evaluation.settings.semantic_provider_id,
        semantic_endpoint_sha256=evaluation.settings.semantic_endpoint_sha256,
        redaction_policy_sha256=(
            evaluation.redaction_engine.policy.digest if evaluation.redaction_engine else None
        ),
        redaction_coverage=evaluation.redaction_coverage,
        campaign_plan=campaign.plan,
        json_output=request.requested.json_output,
    )
