from __future__ import annotations

import asyncio
import inspect
import json
import os
import secrets
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, TextIO, cast

import httpx
import typer
from pydantic import ValidationError
from ul import (
    DatasetAugmentationResult,
    DatasetEvaluationMode,
    DatasetSemanticSettings,
    EvaluatorModelCompatibilityError,
    EvaluatorModelPreflight,
    OpenAICompatibleDatasetSettings,
    OpenRouterDatasetSettings,
    ProviderDiagnosticError,
    load_dataset_semantic_settings,
)
from ul.dataset_invariants import DatasetInvariantEvaluation, load_dataset_invariant_suite
from ul.http_environment import (
    JsonHttpEnvironmentConnection,
    json_http_environment_calls_per_execution,
    json_http_environment_capabilities,
    json_http_environment_config_sha256,
    json_http_environment_config_urls,
    json_http_environment_origin,
    load_json_http_environment_config,
    validate_json_http_environment_configuration,
)

from ul_cli.dataset_augmentation_ledger import (
    DatasetAugmentationLedger,
    DatasetAugmentationLedgerSemanticSettings,
    create_dataset_augmentation_generation_context,
    create_private_augmentation_ledger,
    open_augmentation_ledger_for_resume,
    read_augmentation_ledger,
)
from ul_cli.dataset_campaign import create_dataset_campaign_plan
from ul_cli.dataset_review import DatasetEvidenceRunContext, DatasetResumeEvidence
from ul_cli.dataset_trial_journal import (
    DatasetRunManifest,
    DatasetTrialJournal,
    create_dataset_run_manifest,
    create_dataset_trial_journal,
    create_quarantine_resolution,
    fsync_run_directory,
    journal_anchor_path,
    journal_path,
    manifest_path,
    open_dataset_trial_journal,
    persist_dataset_run_manifest,
    persist_quarantine_resolution,
    private_file_sha256,
    quarantine_resolution_path,
    read_dataset_run_manifest,
    read_quarantine_resolution,
)
from ul_cli.environment import TEST_ENVIRONMENT_CONFIRMATION_MESSAGE
from ul_cli.finding_adapters import FindingAdapterContext, adapt_dataset_finding_packages

from ..evidence.context import build_dataset_evidence_run_context
from ..evidence.persistence import (
    create_durable_evidence_output,
    default_augmentations_output,
    durable_evidence_marker_manifest_sha256,
    load_evaluator_preflight,
    open_resume_output,
    persist_evaluator_preflight,
    read_resume_evidence,
    write_provider_diagnostic,
)
from ..presentation.evaluation import (
    dataset_invariant_exit_code,
    print_dataset_plan,
    print_dataset_results,
    print_evaluator_preflight,
    print_fixture_identity,
    result_needs_review,
)
from ..presentation.runtime import console, print_dataset_plain
from ..storage.private_files import create_private_output, open_private_append_output
from .operators import dataset_operator_identity, validate_operator_ids
from .records import DatasetInputError, load_interaction_records, validate_model_input_bounds
from .redaction import (
    calculate_redaction_coverage,
    load_redaction_engine,
    protect_interaction_records,
)
from .runner import evaluate_interaction_records, preflight_evaluator

_MAXIMUM_DATASET_RECORDS = 100
_MAXIMUM_REPETITIONS = 100
_DEFAULT_MAXIMUM_ENVIRONMENT_API_CALLS = 100


def evaluate_dataset(
    data: Annotated[
        Path | None,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help=(
                'Interaction JSONL: shorthand {"id": ..., "input": ..., "output": ...} '
                "records or structured multi-turn cases."
            ),
        ),
    ] = None,
    environment_config: Annotated[
        Path | None,
        typer.Option(
            "--environment-config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Connection to the customer's agent environment API.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="New JSONL file for complete local evidence."),
    ] = None,
    augmentations_output: Annotated[
        Path | None,
        typer.Option(
            "--augmentations-output",
            help=(
                "Private resumable augmentation JSONL. Defaults beside --output as "
                "NAME.augmentations.jsonl."
            ),
        ),
    ] = None,
    no_save_augmentations: Annotated[
        bool,
        typer.Option(
            "--no-save-augmentations",
            help="Do not retain generated augmentations; interrupted work may be regenerated.",
        ),
    ] = False,
    invariants: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Strict declarative customer invariant configuration.",
        ),
    ] = None,
    evaluation_mode: Annotated[
        DatasetEvaluationMode,
        typer.Option(
            "--evaluation-mode",
            help=(
                "Evaluation intent. Variance compares fresh original replays with variations; "
                "correctness and preference evaluators are not implemented."
            ),
        ),
    ] = "variance",
    operator: Annotated[
        list[str] | None,
        typer.Option(
            "--operator",
            help=(
                "Augmentation ID. Run 'ul augmentations list --mode dataset_variation' "
                "for values; repeat as needed."
            ),
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(min=1, max=_MAXIMUM_DATASET_RECORDS, help="Interactions to evaluate."),
    ] = None,
    repetitions: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Fresh-state environment executions per original input and accepted variation.",
        ),
    ] = None,
    max_environment_api_calls: Annotated[
        int | None,
        typer.Option(
            "--max-environment-api-calls",
            min=1,
            help="Maximum customer environment API requests authorized for this evaluation.",
        ),
    ] = None,
    allow_environment_network: Annotated[
        bool,
        typer.Option(
            "--allow-environment-network",
            help="Allow UL to call the configured remote environment API.",
        ),
    ] = False,
    confirm_test_environment: Annotated[
        bool,
        typer.Option(help=("Confirm the environment is intended for testing and can be reset.")),
    ] = False,
    allow_insecure_http: Annotated[
        bool,
        typer.Option(help="Allow an HTTP environment API. Intended for local environments."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(help="Validate and show the execution plan without external calls."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the dry-run campaign plan as stable JSON."),
    ] = False,
    progress_json: Annotated[
        bool,
        typer.Option(
            "--progress-json",
            help="Emit versioned campaign progress JSON lines on stderr.",
        ),
    ] = False,
    show_sensitive_values: Annotated[
        bool,
        typer.Option(
            "--show-sensitive-values",
            help="Include private saved candidate inputs in dry-run human or JSON output.",
        ),
    ] = False,
    resume: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help=(
                "Existing evidence file to resume; validates run compatibility, skips completed "
                "interactions, and appends."
            ),
        ),
    ] = None,
    resolve_quarantine_after: Annotated[
        Literal["environment-reset", "environment-replacement"] | None,
        typer.Option(
            "--resolve-quarantine-after",
            help=(
                "Operator attestation that the recorded test environment was reset or replaced; "
                "UL records but cannot independently verify this cleanup."
            ),
        ),
    ] = None,
    redaction_policy: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Explicit literal/JSON-pointer policy for semantic-provider data.",
        ),
    ] = None,
    redaction_state: Annotated[
        Path | None,
        typer.Option(help="Private local reversible pseudonym mapping state."),
    ] = None,
    expected_environment_origin: Annotated[
        str | None,
        typer.Option(hidden=True),
    ] = None,
    expected_environment_config_sha256: Annotated[
        str | None,
        typer.Option(hidden=True),
    ] = None,
    expected_redaction_policy_sha256: Annotated[
        str | None,
        typer.Option(hidden=True),
    ] = None,
    show_report_guidance: Annotated[bool, typer.Option(hidden=True)] = True,
) -> None:
    """Explore behavioral differences against a black-box agent.

    UL_LIVE=true enables billed semantic-model calls and external processing together.
    UL_DATASET_LIVE_CALLS and UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING remain separate,
    higher-precedence controls. OpenRouter remains the default; set
    UL_DATASET_SEMANTIC_PROVIDER=openai-compatible for a customer-controlled endpoint.

    UL calls only the configured customer-managed environment API through an explicit
    reset/setup/execute/snapshot lifecycle. Production observations are passive source data and
    cannot select or configure the execution destination.

    Example: ul dataset evaluate interactions.jsonl --environment-config environment.json
    --allow-environment-network --confirm-test-environment
    --output results.jsonl

    Discover operators: ul augmentations list --mode dataset_variation
    Augmentation retention: --augmentations-output PATH or --no-save-augmentations
    """
    augmentations_output_was_explicit = augmentations_output is not None
    redaction_state_was_explicit = redaction_state is not None
    recorded_manifest_for_resume = None
    durable_path_presence = (False, False, False)
    durable_evidence_manifest_sha256 = None
    if resume is not None:
        durable_paths = (
            manifest_path(resume),
            journal_path(resume),
            journal_anchor_path(resume),
        )
        durable_path_presence = tuple(os.path.lexists(path) for path in durable_paths)
        try:
            durable_evidence_manifest_sha256 = durable_evidence_marker_manifest_sha256(resume)
        except (OSError, ValueError) as error:
            message = str(error) if isinstance(error, ValueError) else error.__class__.__name__
            raise typer.BadParameter(
                f"cannot safely inspect resume evidence ({message})",
                param_hint="--resume",
            ) from None
        if durable_evidence_manifest_sha256 is not None and not all(durable_path_presence):
            raise typer.BadParameter(
                "durable evidence requires its manifest, journal, and anchor sidecars; restore "
                "all three together because legacy replay is unsafe",
                param_hint="--resume",
            )
        if any(durable_path_presence) and not all(durable_path_presence):
            raise typer.BadParameter(
                "durable resume sidecars are incomplete; restore the manifest, journal, and "
                "anchor together",
                param_hint="--resume",
            )
    if resume is not None and all(durable_path_presence):
        try:
            recorded_manifest_for_resume = read_dataset_run_manifest(manifest_path(resume))
        except (OSError, ValueError) as error:
            message = str(error) if isinstance(error, ValueError) else error.__class__.__name__
            raise typer.BadParameter(
                f"cannot safely read recorded run manifest ({message})",
                param_hint="--resume",
            ) from None
        if (
            durable_evidence_manifest_sha256 is not None
            and durable_evidence_manifest_sha256 != recorded_manifest_for_resume.manifest_sha256
        ):
            raise typer.BadParameter(
                "primary evidence marker does not match its durable manifest",
                param_hint="--resume",
            )
        recorded_command = recorded_manifest_for_resume.effective_command
        repetitions = repetitions or recorded_command.repetitions
        max_environment_api_calls = (
            max_environment_api_calls or recorded_command.max_environment_api_calls
        )
        allow_environment_network = (
            allow_environment_network or recorded_command.allow_environment_network
        )
        confirm_test_environment = (
            confirm_test_environment or recorded_command.confirm_test_environment
        )
        allow_insecure_http = allow_insecure_http or recorded_command.allow_insecure_http
        if operator is None:
            operator = list(recorded_manifest_for_resume.selected_operator_ids)
        if data is None:
            limit = len(recorded_manifest_for_resume.selected_records)
        no_save_augmentations = not recorded_command.save_augmentations
        if augmentations_output is None and recorded_command.augmentations_output_path is not None:
            augmentations_output = Path(recorded_command.augmentations_output_path)
        if redaction_state is None and recorded_command.redaction_state_path is not None:
            redaction_state = Path(recorded_command.redaction_state_path)
    repetitions = repetitions or 3
    if repetitions > _MAXIMUM_REPETITIONS:
        raise typer.BadParameter(
            f"repetitions cannot exceed {_MAXIMUM_REPETITIONS}",
            param_hint="--repetitions",
        )
    max_environment_api_calls = max_environment_api_calls or _DEFAULT_MAXIMUM_ENVIRONMENT_API_CALLS
    limit = limit or 10
    if data is None and recorded_manifest_for_resume is None:
        raise typer.BadParameter(
            "DATA is required unless --resume has a durable run manifest",
            param_hint="DATA",
        )
    if (json_output or show_sensitive_values) and not dry_run:
        option = "--show-sensitive-values" if show_sensitive_values else "--json"
        raise typer.BadParameter(f"{option} requires --dry-run", param_hint=option)
    if evaluation_mode != "variance":
        raise typer.BadParameter(
            f"evaluation mode '{evaluation_mode}' is not implemented; use 'variance'. "
            "Historical dataset output is grounding evidence, not an expected answer.",
            param_hint="--evaluation-mode",
        )
    if augmentations_output is not None and no_save_augmentations:
        raise typer.BadParameter(
            "--augmentations-output cannot be used with --no-save-augmentations",
            param_hint="--augmentations-output",
        )
    if resume is not None:
        if output is not None and output.resolve() != resume.resolve():
            raise typer.BadParameter(
                "--output must point to the same file as --resume, or be omitted",
                param_hint="--output",
            )
        if output is None:
            output = resume
    if not no_save_augmentations and augmentations_output is None and output is not None:
        augmentations_output = default_augmentations_output(output)
    if (
        output is not None
        and augmentations_output is not None
        and output.resolve() == augmentations_output.resolve()
    ):
        raise typer.BadParameter(
            "--augmentations-output must differ from --output",
            param_hint="--augmentations-output",
        )
    try:
        if data is None:
            assert recorded_manifest_for_resume is not None
            records = recorded_manifest_for_resume.selected_records
        else:
            records = load_interaction_records(data)
        selected_operators = validate_operator_ids(operator)
        invariant_suite = (
            load_dataset_invariant_suite(invariants)
            if invariants is not None
            else (
                recorded_manifest_for_resume.effective_command.invariant_suite_snapshot
                if recorded_manifest_for_resume is not None
                else None
            )
        )
        selected_records = records[:limit]
        redaction_engine = load_redaction_engine(
            redaction_policy,
            redaction_state,
            state_required=not dry_run or resume is not None,
            policy_snapshot=(
                recorded_manifest_for_resume.effective_command.redaction_policy_snapshot
                if redaction_policy is None and recorded_manifest_for_resume is not None
                else None
            ),
        )
        if expected_redaction_policy_sha256 is not None and (
            redaction_engine is None
            or redaction_engine.policy.digest != expected_redaction_policy_sha256
        ):
            raise ValueError(
                "redaction policy changed since 'ul init'; reinitialize the project before "
                "sending data to the semantic provider"
            )
        redaction_coverage = (
            recorded_manifest_for_resume.run_context.redaction_coverage
            if data is None and recorded_manifest_for_resume is not None
            else calculate_redaction_coverage(selected_records, redaction_engine)
        )
        if (
            data is not None
            and redaction_engine is not None
            and (not dry_run or resume is not None)
        ):
            selected_records = protect_interaction_records(selected_records, redaction_engine)
        all_selected_records = selected_records
        if not dry_run and resume is None:
            if environment_config is None:
                raise typer.BadParameter(
                    "execution requires --environment-config",
                    param_hint="--environment-config",
                )
            if not allow_environment_network:
                raise typer.BadParameter(
                    "execution requires --allow-environment-network",
                    param_hint="--allow-environment-network",
                )
            if not confirm_test_environment:
                raise typer.BadParameter(
                    TEST_ENVIRONMENT_CONFIRMATION_MESSAGE,
                    param_hint="--confirm-test-environment",
                )
        loaded_target_config = (
            load_json_http_environment_config(environment_config)
            if environment_config is not None
            else (
                recorded_manifest_for_resume.run_context.target.config
                if recorded_manifest_for_resume is not None
                and recorded_manifest_for_resume.run_context.target.kind == "environment_http"
                else None
            )
        )
        if expected_environment_origin is not None:
            if loaded_target_config is None:
                raise ValueError("saved environment origin requires --environment-config")
            if json_http_environment_origin(loaded_target_config) != expected_environment_origin:
                raise ValueError(
                    "environment origin changed since 'ul init'; reinitialize the project and "
                    "repeat the environment safety acknowledgements"
                )
        if expected_environment_config_sha256 is not None:
            if loaded_target_config is None:
                raise ValueError("saved environment configuration requires --environment-config")
            if json_http_environment_config_sha256(loaded_target_config) != (
                expected_environment_config_sha256
            ):
                raise ValueError(
                    "environment configuration changed since 'ul init'; reinitialize the project "
                    "and repeat the environment safety acknowledgements"
                )
        if loaded_target_config is not None:
            validate_json_http_environment_configuration(
                loaded_target_config,
                test_environment_confirmed=confirm_test_environment
                or dry_run
                or resume is not None,
                allow_insecure_http=allow_insecure_http,
            )
            target_capabilities = json_http_environment_capabilities(loaded_target_config)
            if (
                invariant_suite is not None
                and invariant_suite.observation_authority == "committed_state_snapshot"
                and not target_capabilities.supports_state_observation
            ):
                raise ValueError(
                    "committed-state invariants require the stateful-lifecycle adapter tier; "
                    "isolated-response targets provide response evidence only"
                )
        normalized_target_config = loaded_target_config
        if resume is not None and normalized_target_config is None:
            raise ValueError("--resume requires a recorded or explicit environment configuration")
        target_calls_per_execution = (
            json_http_environment_calls_per_execution(normalized_target_config)
            if normalized_target_config is not None
            else 1
        )
        initial_target_calls = (
            len(selected_records)
            * repetitions
            * (1 + len(selected_operators))
            * target_calls_per_execution
        )
        if resume is None and initial_target_calls > max_environment_api_calls:
            raise ValueError(
                f"selection would make up to {initial_target_calls} environment API calls, "
                f"exceeding --max-environment-api-calls {max_environment_api_calls}; reduce "
                "--limit, --operator, or --repetitions, or explicitly raise the call budget"
            )
        if not dry_run and resume is None:
            if output is None:
                raise typer.BadParameter("execution requires --output", param_hint="--output")
            if output.exists():
                raise typer.BadParameter(
                    "output already exists; UL will not overwrite it",
                    param_hint="--output",
                )
            if augmentations_output is not None and augmentations_output.exists():
                raise typer.BadParameter(
                    "augmentations output already exists; UL will not overwrite it",
                    param_hint="--augmentations-output",
                )
        settings = (
            _restore_recorded_semantic_settings(recorded_manifest_for_resume)
            if recorded_manifest_for_resume is not None
            else load_dataset_semantic_settings()
        )
        validate_model_input_bounds(selected_records, settings.max_input_chars)
        run_context = (
            build_dataset_evidence_run_context(
                selected_records=selected_records,
                selected_operator_ids=selected_operators,
                evaluation_mode=evaluation_mode,
                repetitions=repetitions,
                invariant_suite=invariant_suite,
                target_config=normalized_target_config,
                settings=settings,
                redaction_policy_sha256=(
                    redaction_engine.policy.digest if redaction_engine is not None else None
                ),
                redaction_coverage=redaction_coverage,
            )
            if normalized_target_config is not None
            else None
        )
        augmentation_generation_context = create_dataset_augmentation_generation_context(
            selected_records=all_selected_records,
            operators=tuple(
                dataset_operator_identity(operator_reference)
                for operator_reference in selected_operators
            ),
            semantic_settings=DatasetAugmentationLedgerSemanticSettings(
                provider=settings.semantic_provider_id,
                endpoint_sha256=settings.semantic_endpoint_sha256,
                model=settings.model,
                render_model=settings.render_model,
                equivalence_model=settings.equivalence_model,
                max_input_chars=settings.max_input_chars,
                max_output_tokens=settings.max_output_tokens,
                max_render_tokens=settings.max_render_tokens,
                max_response_bytes=settings.max_response_bytes,
                timeout_seconds=settings.timeout_seconds,
            ),
            redaction_policy_sha256=(
                redaction_engine.policy.digest if redaction_engine is not None else None
            ),
        )
    except (DatasetInputError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from None

    trial_journal: DatasetTrialJournal | None = None
    if recorded_manifest_for_resume is not None:
        assert resume is not None
        try:
            trial_journal = open_dataset_trial_journal(
                journal_path(resume), recorded_manifest_for_resume
            )
        except (OSError, ValueError) as error:
            message = str(error) if isinstance(error, ValueError) else error.__class__.__name__
            raise typer.BadParameter(
                f"cannot safely lock durable run state ({message})",
                param_hint="--resume",
            ) from None

    resume_evidence: DatasetResumeEvidence | None = None
    saved_augmentations: dict[str, DatasetAugmentationResult] = {}
    skipped_count = 0
    if resume is not None:
        assert output is not None
        assert run_context is not None
        try:
            resume_evidence = read_resume_evidence(
                output,
                expected_context=run_context,
                selected_records=selected_records,
                invariant_suite=invariant_suite,
            )
            if augmentations_output is not None and augmentations_output.exists():
                augmentation_snapshot = read_augmentation_ledger(
                    augmentations_output,
                    expected_context=augmentation_generation_context,
                    selected_records=all_selected_records,
                )
                saved_augmentations = {
                    record.source.id: record.augmentation
                    for record in augmentation_snapshot.records
                }
                for prior_result in resume_evidence.technical_results:
                    if saved_augmentations.get(prior_result.source.id) != prior_result.augmentation:
                        raise ValueError(
                            "augmentation ledger does not match completed evaluation evidence"
                        )
        except (OSError, ValueError) as error:
            if trial_journal is not None:
                trial_journal.close()
            message = str(error) if isinstance(error, ValueError) else error.__class__.__name__
            raise typer.BadParameter(
                f"cannot safely resume evidence ({message})",
                param_hint="--resume",
            ) from None
        selected_records = tuple(
            record for record in selected_records if record.id not in resume_evidence.processed_ids
        )
        skipped_count = len(resume_evidence.processed_ids)

    if not dry_run or (resume is not None and recorded_manifest_for_resume is not None):
        assert output is not None
        assert run_context is not None
        if augmentations_output is not None and not augmentations_output.parent.is_dir():
            raise typer.BadParameter(
                "cannot safely open augmentations output (FileNotFoundError)",
                param_hint="--augmentations-output",
            )
        expected_manifest = create_dataset_run_manifest(
            run_context=run_context,
            selected_records=all_selected_records,
            selected_operator_ids=selected_operators,
            repetitions=repetitions,
            max_environment_api_calls=max_environment_api_calls,
            allow_environment_network=allow_environment_network,
            confirm_test_environment=confirm_test_environment,
            allow_insecure_http=allow_insecure_http,
            save_augmentations=not no_save_augmentations,
            semantic_provider_type=settings.semantic_provider_type,
            semantic_base_url=settings.semantic_base_url,
            semantic_live_calls=settings.live_calls,
            semantic_allow_external_data_processing=settings.allow_external_data_processing,
            invariant_suite_snapshot=invariant_suite,
            invariant_suite_source=(
                str(invariants.resolve())
                if invariants is not None
                else (
                    recorded_manifest_for_resume.effective_command.invariant_suite_source
                    if recorded_manifest_for_resume is not None
                    else None
                )
            ),
            redaction_policy_snapshot=(redaction_engine.policy if redaction_engine else None),
            redaction_policy_source=(
                str(redaction_policy.resolve())
                if redaction_policy is not None
                else (
                    recorded_manifest_for_resume.effective_command.redaction_policy_source
                    if recorded_manifest_for_resume is not None
                    else None
                )
            ),
            redaction_state_path=(
                str(redaction_state.resolve())
                if redaction_state is not None
                and (recorded_manifest_for_resume is None or redaction_state_was_explicit)
                else (
                    recorded_manifest_for_resume.effective_command.redaction_state_path
                    if recorded_manifest_for_resume is not None
                    else None
                )
            ),
            redaction_state_sha256=(
                private_file_sha256(redaction_state)
                if redaction_state is not None
                and (recorded_manifest_for_resume is None or redaction_state_was_explicit)
                and redaction_state.exists()
                else (
                    recorded_manifest_for_resume.effective_command.redaction_state_sha256
                    if recorded_manifest_for_resume is not None
                    else None
                )
            ),
            augmentations_output_path=(
                str(augmentations_output.resolve())
                if augmentations_output is not None
                and (recorded_manifest_for_resume is None or augmentations_output_was_explicit)
                else (
                    recorded_manifest_for_resume.effective_command.augmentations_output_path
                    if recorded_manifest_for_resume is not None
                    else None
                )
            ),
        )
        run_manifest_path = manifest_path(output)
        run_journal_path = journal_path(output)
        try:
            if resume is None:
                persist_dataset_run_manifest(run_manifest_path, expected_manifest)
                trial_journal = create_dataset_trial_journal(run_journal_path, expected_manifest)
                create_durable_evidence_output(output, expected_manifest.manifest_sha256)
                fsync_run_directory(output)
            elif run_manifest_path.exists():
                recorded_manifest = read_dataset_run_manifest(run_manifest_path)
                incompatibility = _manifest_incompatibility_reason(
                    recorded_manifest.run_context, expected_manifest.run_context
                )
                if incompatibility is not None:
                    raise ValueError(f"resume_incompatible:{incompatibility}")
                if recorded_manifest != expected_manifest:
                    effective_incompatibility = _effective_command_incompatibility_reason(
                        recorded_manifest, expected_manifest
                    )
                    raise ValueError(
                        f"resume_incompatible:{effective_incompatibility or 'effective_command'}"
                    )
                if trial_journal is None:
                    raise ValueError("resume journal lock was not acquired")
                if trial_journal.snapshot.quarantined_unit_ids:
                    quarantined_unit_ids = trial_journal.snapshot.quarantined_unit_ids
                    resolution_path = quarantine_resolution_path(output)
                    if resolution_path.exists():
                        resolution = read_quarantine_resolution(resolution_path)
                    elif resolve_quarantine_after is not None:
                        resolution = create_quarantine_resolution(
                            recorded_manifest,
                            quarantined_unit_ids,
                            resolve_quarantine_after,
                            datetime.now(UTC),
                        )
                        persist_quarantine_resolution(resolution_path, resolution)
                    else:
                        raise ValueError(
                            "resume_quarantined:target_delivery_or_cleanup_uncertain; after an "
                            "operator has reset or replaced the recorded test environment, attest "
                            "with --resolve-quarantine-after environment-reset (or "
                            "environment-replacement). UL records but cannot independently verify "
                            "this cleanup"
                        )
                    if (
                        resolution.manifest_sha256 != recorded_manifest.manifest_sha256
                        or resolution.target_sha256 != recorded_manifest.run_context.target.sha256
                        or frozenset(resolution.quarantined_unit_ids) != quarantined_unit_ids
                    ):
                        raise ValueError(
                            "resume_quarantined:cleanup_attestation_does_not_match_campaign"
                        )
                elif resolve_quarantine_after is not None:
                    raise ValueError("resume_incompatible:no_quarantined_trials_to_resolve")
                if (
                    not recorded_manifest.effective_command.save_augmentations
                    and trial_journal.snapshot.terminal_states
                ):
                    raise ValueError(
                        "resume_incompatible:augmentation_not_durable; start a new output with "
                        "augmentation retention enabled"
                    )
        except (OSError, ValueError) as error:
            if trial_journal is not None:
                trial_journal.close()
            message = str(error) if isinstance(error, ValueError) else error.__class__.__name__
            if resume is not None and message.startswith("resume_"):
                diagnose_command = (
                    f"ul dataset evaluate --resume {shlex.quote(str(resume))} --dry-run"
                )
                message = f"{message}; diagnose with: {diagnose_command}"
            raise typer.BadParameter(
                f"cannot safely open durable run state ({message})",
                param_hint="--resume" if resume is not None else "--output",
            ) from None

    evaluator_preflight: EvaluatorModelPreflight | None = None
    evaluator_preflight_receipt: Path | None = None
    if resume is not None and selected_records:
        assert output is not None
        try:
            evaluator_preflight, evaluator_preflight_receipt = asyncio.run(
                load_evaluator_preflight(output, settings)
            )
        except ValueError as error:
            raise typer.BadParameter(
                f"cannot reuse evaluator preflight receipt ({error}); restore the matching "
                "receipt and semantic settings, or start a new run with a new --output",
                param_hint="--resume",
            ) from None

    campaign_plan = create_dataset_campaign_plan(
        records=selected_records,
        selected_operator_ids=selected_operators,
        repetitions=repetitions,
        target_calls_per_execution=target_calls_per_execution,
        settings=settings,
        saved_augmentations=saved_augmentations,
        show_sensitive_values=show_sensitive_values,
        requires_preflight=evaluator_preflight is None and bool(selected_records),
        evaluation_mode=evaluation_mode,
        fixture_status=(
            run_context.fixture.status
            if run_context is not None and run_context.fixture is not None
            else None
        ),
        fixture_id=(
            run_context.fixture.id
            if run_context is not None and run_context.fixture is not None
            else None
        ),
        fixture_version=(
            run_context.fixture.version
            if run_context is not None and run_context.fixture is not None
            else None
        ),
    )
    potential_target_calls = campaign_plan.calls.total_environment_api
    if potential_target_calls > max_environment_api_calls:
        raise typer.BadParameter(
            f"remaining selection would make up to {potential_target_calls} environment API calls, "
            f"exceeding --max-environment-api-calls {max_environment_api_calls}; reduce --limit, "
            "--operator, "
            "or --repetitions, or explicitly raise the call budget"
        )

    if dry_run:
        print_dataset_plan(
            record_count=len(records),
            selected_count=len(selected_records),
            skipped_count=skipped_count,
            operator_ids=selected_operators,
            evaluation_mode=evaluation_mode,
            target_configured=environment_config is not None,
            target_endpoint=(
                json_http_environment_config_urls(loaded_target_config)[0]
                if loaded_target_config is not None
                else None
            ),
            target_header_environment_variables=(
                loaded_target_config.headers_from_env if loaded_target_config is not None else {}
            ),
            repetitions=repetitions,
            max_environment_api_calls=max_environment_api_calls,
            target_calls_per_execution=target_calls_per_execution,
            target_supports_state_observation=(
                json_http_environment_capabilities(loaded_target_config).supports_state_observation
                if loaded_target_config is not None
                else None
            ),
            fixture_status=(
                run_context.fixture.status
                if run_context is not None and run_context.fixture is not None
                else None
            ),
            fixture_id=(
                run_context.fixture.id
                if run_context is not None and run_context.fixture is not None
                else None
            ),
            fixture_version=(
                run_context.fixture.version
                if run_context is not None and run_context.fixture is not None
                else None
            ),
            invariant_suite=invariant_suite,
            output=output,
            augmentations_output=augmentations_output,
            semantic_provider_id=settings.semantic_provider_id,
            semantic_endpoint_sha256=settings.semantic_endpoint_sha256,
            redaction_policy_sha256=(
                redaction_engine.policy.digest if redaction_engine is not None else None
            ),
            redaction_coverage=redaction_coverage,
            campaign_plan=campaign_plan,
            json_output=json_output,
        )
        if trial_journal is not None:
            trial_journal.close()
        return

    if not selected_records and skipped_count > 0:
        assert output is not None
        assert resume_evidence is not None
        if augmentations_output is not None:
            try:
                if augmentations_output.exists():
                    for prior_result in resume_evidence.technical_results:
                        if (
                            saved_augmentations.get(prior_result.source.id)
                            != prior_result.augmentation
                        ):
                            raise ValueError(
                                "augmentation ledger does not match completed evaluation evidence"
                            )
                else:
                    with create_private_augmentation_ledger(
                        augmentations_output,
                        generation_context=augmentation_generation_context,
                        selected_records=all_selected_records,
                    ) as completed_ledger:
                        for prior_result in resume_evidence.technical_results:
                            completed_ledger.append(
                                source=prior_result.source,
                                augmentation=prior_result.augmentation,
                            )
            except (OSError, ValueError) as error:
                message = str(error) if isinstance(error, ValueError) else error.__class__.__name__
                raise typer.BadParameter(
                    f"cannot safely persist augmentations ({message})",
                    param_hint="--augmentations-output",
                ) from None
        console.print(
            f"Resume compatible: all {skipped_count} selected interaction(s) are complete in "
            f"{output}. Nothing to do."
        )
        if trial_journal is not None:
            trial_journal.close()
        previous_invariant_exit_code = dataset_invariant_exit_code(
            resume_evidence.invariant_evaluations
        )
        if previous_invariant_exit_code:
            raise typer.Exit(code=previous_invariant_exit_code)
        if resume_evidence.has_review_findings:
            raise typer.Exit(code=1)
        raise typer.Exit(code=0)

    if loaded_target_config is None:
        raise typer.BadParameter(
            "execution requires a recorded or explicit environment configuration",
            param_hint="--environment-config",
        )
    if not confirm_test_environment:
        raise typer.BadParameter(
            TEST_ENVIRONMENT_CONFIRMATION_MESSAGE,
            param_hint="--confirm-test-environment",
        )
    if output is None:
        raise typer.BadParameter("execution requires --output", param_hint="--output")
    assert run_context is not None
    assert run_context.fixture is not None
    print_fixture_identity(
        run_context.fixture.status,
        fixture_id=run_context.fixture.id,
        fixture_version=run_context.fixture.version,
    )
    if not settings.live_calls:
        raise typer.BadParameter(
            "set UL_LIVE=true (or UL_DATASET_LIVE_CALLS=true) to allow semantic model calls"
        )
    if not settings.allow_external_data_processing:
        raise typer.BadParameter(
            "set UL_LIVE=true (or UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING=true) "
            "to allow semantic model calls"
        )
    if settings.api_key_required and (
        settings.api_key is None or not settings.api_key.get_secret_value().strip()
    ):
        raise typer.BadParameter(
            f"set {settings.api_key_environment_variable} to run an evaluation"
        )

    try:
        assert loaded_target_config is not None
        if not allow_environment_network:
            raise ValueError("environment execution requires --allow-environment-network")
        target = JsonHttpEnvironmentConnection.from_config(
            loaded_target_config,
            test_environment_confirmed=True,
            allow_insecure_http=allow_insecure_http,
            max_environment_api_calls=max_environment_api_calls,
        )
    except ValueError as error:
        raise typer.BadParameter(
            str(error),
            param_hint="--environment-config",
        ) from None

    if evaluator_preflight is None:
        try:
            evaluator_preflight = asyncio.run(preflight_evaluator(settings))
            evaluator_preflight_receipt = persist_evaluator_preflight(output, evaluator_preflight)
        except EvaluatorModelCompatibilityError as error:
            if trial_journal is not None:
                trial_journal.close()
            asyncio.run(target.aclose())
            print_dataset_plain(f"Evaluation stopped before campaign execution: {error}")
            raise typer.Exit(code=2) from None
        except (OSError, ValueError) as error:
            if trial_journal is not None:
                trial_journal.close()
            asyncio.run(target.aclose())
            message = str(error) if isinstance(error, ValueError) else error.__class__.__name__
            raise typer.BadParameter(
                f"cannot safely persist evaluator preflight ({message})",
                param_hint="--output",
            ) from None
    assert evaluator_preflight_receipt is not None
    print_evaluator_preflight(evaluator_preflight, evaluator_preflight_receipt)

    augmentation_ledger: DatasetAugmentationLedger | None = None
    augmentation_ledger_was_created = False
    output_stream: TextIO | None = None
    finding_output_stream: TextIO | None = None
    finding_output = output.with_name(f"{output.name}.findings.jsonl")
    finding_reference_key = secrets.token_bytes(32)
    failure_parameter = "--augmentations-output"
    try:
        if augmentations_output is not None:
            if resume is not None and augmentations_output.exists():
                augmentation_ledger = open_augmentation_ledger_for_resume(
                    augmentations_output,
                    expected_context=augmentation_generation_context,
                    selected_records=all_selected_records,
                )
            else:
                augmentation_ledger = create_private_augmentation_ledger(
                    augmentations_output,
                    generation_context=augmentation_generation_context,
                    selected_records=all_selected_records,
                )
                augmentation_ledger_was_created = True
                if resume_evidence is not None:
                    for prior_result in resume_evidence.technical_results:
                        augmentation_ledger.append(
                            source=prior_result.source,
                            augmentation=prior_result.augmentation,
                        )
            saved_augmentations = {
                record.source.id: record.augmentation
                for record in augmentation_ledger.snapshot.records
            }
            if resume_evidence is not None:
                for prior_result in resume_evidence.technical_results:
                    if saved_augmentations.get(prior_result.source.id) != prior_result.augmentation:
                        raise ValueError(
                            "augmentation ledger does not match completed evaluation evidence"
                        )
        failure_parameter = "--resume" if resume is not None else "--output"
        if resume is None:
            assert expected_manifest is not None
            assert run_manifest_path is not None
            assert run_journal_path is not None
            persist_dataset_run_manifest(run_manifest_path, expected_manifest)
            trial_journal = create_dataset_trial_journal(run_journal_path, expected_manifest)
            create_private_output(output).close()
            output_stream, initial_evidence = open_resume_output(
                output,
                expected_context=run_context,
                selected_records=all_selected_records,
                invariant_suite=invariant_suite,
            )
            if initial_evidence.processed_ids:
                output_stream.close()
                raise ValueError("new evidence output is not empty")
            finding_output_stream = create_private_output(finding_output)
        else:
            assert resume_evidence is not None
            output_stream, locked_resume_evidence = open_resume_output(
                output,
                expected_context=run_context,
                selected_records=all_selected_records,
                invariant_suite=invariant_suite,
            )
            if locked_resume_evidence != resume_evidence:
                output_stream.close()
                raise ValueError("resume evidence changed after preflight")
            finding_output_stream = (
                open_private_append_output(finding_output)
                if finding_output.exists()
                else create_private_output(finding_output)
            )
    except (OSError, ValueError) as error:
        if augmentation_ledger is not None:
            if augmentation_ledger_was_created:
                assert augmentations_output is not None
                augmentation_ledger.discard_if_empty(augmentations_output)
            augmentation_ledger.close()
        if output_stream is not None and not output_stream.closed:
            output_stream.close()
        if finding_output_stream is not None and not finding_output_stream.closed:
            finding_output_stream.close()
        asyncio.run(target.aclose())
        message = str(error) if isinstance(error, ValueError) else error.__class__.__name__
        raise typer.BadParameter(
            f"cannot safely open persistence file ({message})",
            param_hint=failure_parameter,
        ) from None

    assert output_stream is not None
    assert finding_output_stream is not None
    has_review_findings = False
    invariant_evaluations: list[DatasetInvariantEvaluation] = []
    try:
        with output_stream, finding_output_stream:
            evaluation_parameters = inspect.signature(evaluate_interaction_records).parameters
            accepts_extra_arguments = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in evaluation_parameters.values()
            )
            durable_arguments: dict[str, object] = {}
            if "progress_plan" in evaluation_parameters or accepts_extra_arguments:
                durable_arguments["progress_plan"] = campaign_plan
            if trial_journal is not None and (
                "trial_journal" in evaluation_parameters or accepts_extra_arguments
            ):
                durable_arguments["trial_journal"] = trial_journal
            if progress_json and (
                "progress_json" in evaluation_parameters or accepts_extra_arguments
            ):
                durable_arguments["progress_json"] = True
            evaluation_runner = cast(Any, evaluate_interaction_records)
            if invariant_suite is not None:
                evaluation_coroutine = evaluation_runner(
                    selected_records,
                    selected_operators,
                    settings,
                    target,
                    output_stream,
                    repetitions=repetitions,
                    max_environment_api_calls=max_environment_api_calls,
                    planned_target_calls=(
                        (len(selected_records) + skipped_count)
                        * repetitions
                        * (1 + len(selected_operators))
                        * target_calls_per_execution
                    ),
                    run_context=run_context,
                    augmentation_ledger=augmentation_ledger,
                    saved_augmentations=saved_augmentations,
                    invariant_suite=invariant_suite,
                    invariant_evaluations=invariant_evaluations,
                    redaction_engine=redaction_engine,
                    evaluator_preflight=evaluator_preflight,
                    **durable_arguments,
                )
            else:
                evaluation_coroutine = evaluation_runner(
                    selected_records,
                    selected_operators,
                    settings,
                    target,
                    output_stream,
                    repetitions=repetitions,
                    max_environment_api_calls=max_environment_api_calls,
                    planned_target_calls=(
                        (len(selected_records) + skipped_count)
                        * repetitions
                        * (1 + len(selected_operators))
                        * target_calls_per_execution
                    ),
                    run_context=run_context,
                    augmentation_ledger=augmentation_ledger,
                    saved_augmentations=saved_augmentations,
                    redaction_engine=redaction_engine,
                    evaluator_preflight=evaluator_preflight,
                    **durable_arguments,
                )
            results = asyncio.run(evaluation_coroutine)
            invariant_evaluation_by_interaction = {
                evaluation.interaction_id: evaluation for evaluation in invariant_evaluations
            }
            for result in results:
                packages = adapt_dataset_finding_packages(
                    result,
                    invariant_evaluation=invariant_evaluation_by_interaction.get(result.source.id),
                    invariant_rules=invariant_suite.rules if invariant_suite is not None else (),
                    context=FindingAdapterContext(
                        campaign_id=run_context.context_sha256,
                        recorded_at=datetime.now(UTC),
                        reference_key=finding_reference_key,
                    ),
                )
                for package in packages:
                    finding_output_stream.write(
                        json.dumps(
                            package.model_dump(mode="json"),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                has_review_findings |= result_needs_review(result)
            finding_output_stream.flush()
            os.fsync(finding_output_stream.fileno())
    except (TimeoutError, RuntimeError, ValueError, httpx.HTTPError) as error:
        if isinstance(error, ProviderDiagnosticError):
            console.print(str(error))
            try:
                diagnostic_output = write_provider_diagnostic(output, error)
            except OSError as diagnostic_error:
                print_dataset_plain(
                    "Sanitized provider diagnostics could not be written "
                    f"({diagnostic_error.__class__.__name__})."
                )
            else:
                print_dataset_plain(f"Sanitized provider diagnostics: {diagnostic_output}")
        else:
            console.print(f"Evaluation stopped ({error.__class__.__name__}).")
        console.print(f"Complete results written before the error remain in {output}.")
        if augmentations_output is not None:
            print_dataset_plain(
                f"Generated augmentations remain in {augmentations_output} and will be reused "
                f"with --resume {output}."
            )
        raise typer.Exit(code=2) from None
    finally:
        if augmentation_ledger is not None:
            augmentation_ledger.close()
        if trial_journal is not None:
            trial_journal.close()

    if skipped_count > 0:
        console.print(
            f"Resumed: {skipped_count} interaction(s) skipped (already in evidence), "
            f"{len(results)} newly evaluated."
        )
    print_dataset_results(
        results,
        output,
        augmentations_output=augmentations_output,
        invariant_evaluations=tuple(invariant_evaluations),
        show_report_guidance=show_report_guidance,
    )
    prior_invariant_evaluations = (
        resume_evidence.invariant_evaluations if resume_evidence is not None else ()
    )
    evaluation_exit_code = dataset_invariant_exit_code(
        (*prior_invariant_evaluations, *invariant_evaluations)
    )
    if evaluation_exit_code == 1:
        raise typer.Exit(code=1)
    if evaluation_exit_code == 2:
        raise typer.Exit(code=2)
    if has_review_findings or (resume_evidence is not None and resume_evidence.has_review_findings):
        raise typer.Exit(code=1)


def _manifest_incompatibility_reason(
    recorded: DatasetEvidenceRunContext,
    requested: DatasetEvidenceRunContext,
) -> str | None:
    checks = (
        ("fixture", recorded.fixture, requested.fixture),
        ("target", recorded.target, requested.target),
        ("projection", recorded.invariant_suite_sha256, requested.invariant_suite_sha256),
        ("operators", recorded.operators, requested.operators),
        (
            "evaluator.provider",
            recorded.semantic_settings.provider,
            requested.semantic_settings.provider,
        ),
        (
            "evaluator.endpoint_sha256",
            recorded.semantic_settings.endpoint_sha256,
            requested.semantic_settings.endpoint_sha256,
        ),
        ("evaluator.model", recorded.semantic_settings.model, requested.semantic_settings.model),
        (
            "evaluator.render_model",
            recorded.semantic_settings.render_model,
            requested.semantic_settings.render_model,
        ),
        (
            "evaluator.equivalence_model",
            recorded.semantic_settings.equivalence_model,
            requested.semantic_settings.equivalence_model,
        ),
        (
            "evaluator.limits",
            (
                recorded.semantic_settings.max_input_chars,
                recorded.semantic_settings.max_output_tokens,
                recorded.semantic_settings.max_render_tokens,
                recorded.semantic_settings.max_response_bytes,
                recorded.semantic_settings.timeout_seconds,
            ),
            (
                requested.semantic_settings.max_input_chars,
                requested.semantic_settings.max_output_tokens,
                requested.semantic_settings.max_render_tokens,
                requested.semantic_settings.max_response_bytes,
                requested.semantic_settings.timeout_seconds,
            ),
        ),
        ("dataset", recorded.selected_dataset_sha256, requested.selected_dataset_sha256),
        ("redaction", recorded.redaction_policy_sha256, requested.redaction_policy_sha256),
    )
    return next((reason for reason, left, right in checks if left != right), None)


def _restore_recorded_semantic_settings(
    manifest: DatasetRunManifest,
) -> DatasetSemanticSettings:
    recorded = manifest.run_context.semantic_settings
    command = manifest.effective_command
    if command.semantic_provider_type == "openai-compatible":
        return OpenAICompatibleDatasetSettings(
            live_calls=command.semantic_live_calls,
            allow_external_data_processing=command.semantic_allow_external_data_processing,
            model=recorded.model,
            render_model=recorded.render_model,
            equivalence_model=recorded.equivalence_model,
            max_input_chars=recorded.max_input_chars,
            max_output_tokens=recorded.max_output_tokens,
            max_render_tokens=recorded.max_render_tokens,
            max_response_bytes=recorded.max_response_bytes,
            timeout_seconds=recorded.timeout_seconds,
            provider_id=recorded.provider,
            base_url=command.semantic_base_url,
        )
    return OpenRouterDatasetSettings(
        live_calls=command.semantic_live_calls,
        allow_external_data_processing=command.semantic_allow_external_data_processing,
        model=recorded.model,
        render_model=recorded.render_model,
        equivalence_model=recorded.equivalence_model,
        max_input_chars=recorded.max_input_chars,
        max_output_tokens=recorded.max_output_tokens,
        max_render_tokens=recorded.max_render_tokens,
        max_response_bytes=recorded.max_response_bytes,
        timeout_seconds=recorded.timeout_seconds,
    )


def _effective_command_incompatibility_reason(
    recorded: DatasetRunManifest,
    requested: DatasetRunManifest,
) -> str | None:
    left = recorded.effective_command
    right = requested.effective_command
    checks = (
        ("invariant_suite_source", left.invariant_suite_source, right.invariant_suite_source),
        ("redaction_policy_source", left.redaction_policy_source, right.redaction_policy_source),
        ("redaction_state_path", left.redaction_state_path, right.redaction_state_path),
        (
            "redaction_state_sha256",
            left.redaction_state_sha256,
            right.redaction_state_sha256,
        ),
        (
            "augmentations_output_path",
            left.augmentations_output_path,
            right.augmentations_output_path,
        ),
        ("repetitions", left.repetitions, right.repetitions),
        (
            "max_environment_api_calls",
            left.max_environment_api_calls,
            right.max_environment_api_calls,
        ),
    )
    return next((name for name, old, new in checks if old != new), None)
