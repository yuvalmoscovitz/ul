from __future__ import annotations

import asyncio
import inspect
import json
import os
import shlex
import sys
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, TextIO, cast

import httpx
import typer
from pydantic import JsonValue, ValidationError
from ul import (
    DatasetAugmentationResult,
    DatasetEvaluationMode,
    DatasetEvaluationResult,
    DatasetSemanticSettings,
    EvaluatorModelCompatibilityError,
    EvaluatorModelPreflight,
    OpenAICompatibleDatasetSettings,
    OpenRouterDatasetSettings,
    ProviderDiagnosticError,
    load_dataset_semantic_settings,
    semantic_deconstructor_identity,
)
from ul.dataset_invariants import (
    DatasetInvariantEvaluation,
    DatasetInvariantRule,
    load_dataset_invariant_suite,
)
from ul.http_environment import (
    JsonHttpEnvironmentConnection,
    JsonHttpTargetConfig,
    json_http_environment_calls_per_execution,
    json_http_environment_capabilities,
    json_http_environment_config_sha256,
    json_http_environment_config_urls,
    json_http_environment_origin,
    load_json_http_environment_config,
    validate_json_http_environment_configuration,
)

from ul_cli.dataset.progress import (
    CampaignControlRequested,
    create_campaign_next_commands,
    create_campaign_progress_runtime,
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
from ul_cli.dataset_review import (
    DatasetEvidenceRunContext,
    DatasetResumeEvidence,
    DatasetSourcePreparationFailureEvidence,
)
from ul_cli.dataset_trial_journal import (
    DatasetRunManifest,
    DatasetTrialJournal,
    DatasetTrialJournalSnapshot,
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
from ul_cli.finding_reference import (
    FindingReferenceContext,
    finding_reference_key_path,
    resolve_finding_reference_context,
)
from ul_cli.http_target_resolution import (
    HttpTargetConfirmation,
    ResolvedHttpTarget,
    http_target_evidence_receipt,
    resolve_http_target,
    resolve_http_target_config,
)
from ul_cli.local_target_resolution import (
    ResolvedLocalTarget,
    local_target_evidence_receipt,
    resolve_local_target,
)

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
from ..storage.private_files import open_resume_descriptor
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
_MAXIMUM_FINDING_SNAPSHOT_BYTES = 128_000_000
_SOURCE_PREPARATION_REASON_CODES = {
    "source_semantic_preparation_failed",
    "source_comparison_surface_incompatible",
}


def _source_outcome_projection_sha256(
    target_config: JsonHttpTargetConfig | None,
    local_target: ResolvedLocalTarget | None,
) -> str | None:
    projection = local_target.config.outcome if local_target is not None else None
    if projection is None and target_config is not None:
        projection = target_config.outcome
    return projection.digest if projection is not None else None


def _reconcile_source_preparation_failures(
    trial_journal: DatasetTrialJournal,
    resume_evidence: DatasetResumeEvidence,
) -> None:
    snapshot = trial_journal.snapshot
    failures_by_interaction_id = {
        failure.interaction_id: failure for failure in resume_evidence.source_preparation_failures
    }
    for unit in trial_journal.manifest.work_plan:
        failure = failures_by_interaction_id.get(unit.interaction_id)
        if failure is None:
            continue
        terminal_state = snapshot.terminal_states.get(unit.id)
        if terminal_state is None:
            trial_journal.terminal(unit, "errored", failure.reason_code)
            continue
        reason_code = snapshot.terminal_reason_codes.get(unit.id)
        if terminal_state == "errored" and reason_code == failure.reason_code:
            continue
        if unit.arm == "probe" and terminal_state in {"inapplicable", "rejected"}:
            continue
        raise ValueError("source failure evidence conflicts with durable trial state")


def _attempted_target_calls(snapshot: DatasetTrialJournalSnapshot) -> int:
    return sum(
        state in {"completed", "errored", "inconclusive", "quarantined"}
        and snapshot.terminal_reason_codes.get(unit_id) not in _SOURCE_PREPARATION_REASON_CODES
        for unit_id, state in snapshot.terminal_states.items()
    )


def _write_finding_package_snapshot(
    finding_output: Path,
    results: tuple[DatasetEvaluationResult, ...],
    invariant_evaluations: tuple[DatasetInvariantEvaluation, ...],
    invariant_rules: tuple[DatasetInvariantRule, ...],
    *,
    campaign_id: str,
    reference_context: FindingReferenceContext,
) -> bool:
    invariant_evaluation_by_interaction = {
        evaluation.interaction_id: evaluation for evaluation in invariant_evaluations
    }
    serialized_packages: list[str] = []
    for result in results:
        packages = adapt_dataset_finding_packages(
            result,
            invariant_evaluation=invariant_evaluation_by_interaction.get(result.source.id),
            invariant_rules=invariant_rules,
            context=FindingAdapterContext(
                campaign_id=campaign_id,
                recorded_at=reference_context.recorded_at,
                reference_key=reference_context.key,
            ),
        )
        serialized_packages.extend(
            json.dumps(
                package.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for package in packages
        )
    snapshot = "".join(serialized_packages).encode("utf-8")
    if len(snapshot) > _MAXIMUM_FINDING_SNAPSHOT_BYTES:
        raise ValueError("finding package snapshot exceeds the 128 MB limit")
    _replace_finding_package_snapshot(finding_output, snapshot)
    return any(result_needs_review(result) for result in results)


def _replace_finding_package_snapshot(finding_output: Path, snapshot: bytes) -> None:
    lock_output = finding_output.with_name(f".{finding_output.name}.lock")
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    with suppress(FileExistsError):
        lock_descriptor = os.open(
            lock_output,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow_flag,
            0o600,
        )
        os.close(lock_descriptor)
    lock_descriptor = open_resume_descriptor(lock_output, writable=True)
    temporary_output: Path | None = None
    try:
        temporary_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{finding_output.name}.tmp-",
            dir=finding_output.parent,
        )
        temporary_output = Path(temporary_name)
        try:
            if sys.platform != "win32":
                os.fchmod(temporary_descriptor, 0o600)
            remaining = memoryview(snapshot)
            while remaining:
                written = os.write(temporary_descriptor, remaining)
                if written == 0:
                    raise OSError("finding package snapshot write was incomplete")
                remaining = remaining[written:]
            os.fsync(temporary_descriptor)
        finally:
            os.close(temporary_descriptor)
        os.replace(temporary_output, finding_output)
        temporary_output = None
        _fsync_directory(finding_output.parent)
    finally:
        if temporary_output is not None:
            with suppress(OSError):
                temporary_output.unlink()
        os.close(lock_descriptor)


def _fsync_directory(directory: Path) -> None:
    if sys.platform == "win32":
        return
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_descriptor = os.open(directory, directory_flags)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


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
    target: Annotated[
        str | None,
        typer.Option(
            "--target",
            help=(
                "Isolated-response HTTP(S) URL, Python module:callable, or local/HTTP target "
                "configuration JSON."
            ),
        ),
    ] = None,
    target_artifact: Annotated[
        list[Path] | None,
        typer.Option(
            "--target-artifact",
            help="Additional command worker artifact to hash and bind; repeat as needed.",
        ),
    ] = None,
    http_preset: Annotated[
        Literal["generic-json", "openai-chat"] | None,
        typer.Option(help="Request/response shape for an HTTP URL; defaults to generic-json."),
    ] = None,
    request_json_template: Annotated[
        str | None,
        typer.Option(help="JSON containing one {{input}} value; overrides the direct HTTP preset."),
    ] = None,
    response_json_pointer: Annotated[
        str | None,
        typer.Option(help="RFC 6901 pointer to the direct HTTP response value."),
    ] = None,
    agent_model: Annotated[
        str | None,
        typer.Option(help="Model sent by the direct HTTP openai-chat preset."),
    ] = None,
    header_from_env: Annotated[
        list[str] | None,
        typer.Option(
            "--header-from-env",
            help="HTTP_HEADER=UL_ENVIRONMENT_VARIABLE; repeat for credentials or routing.",
        ),
    ] = None,
    confirm_target: Annotated[
        str | None,
        typer.Option(
            "--confirm-target",
            help="Confirm the exact local or HTTP target digest.",
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
                "for values; repeat as needed. Defaults to input.surface.typing_noise."
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
    confirm_request_isolation: Annotated[
        bool,
        typer.Option(
            help="Attest every direct HTTP request starts fresh and cannot affect another."
        ),
    ] = False,
    confirm_safe_test_target: Annotated[
        bool,
        typer.Option(help="Attest the direct HTTP target cannot cause real-world effects."),
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

    UL calls only the explicitly configured customer test target. Local callable and command
    targets run in a bounded child process. HTTP environments use their explicit
    reset/setup/execute/snapshot lifecycle. Production observations are passive source data and
    cannot select the execution destination.

    Pass --target URL for an existing response-only JSON agent without writing a UL config. Direct
    HTTP targets use the generic-json mapping by default, must isolate every request, and cannot
    provide committed-state evidence without a separate state observer.

    ul probe and this command both default to input.surface.typing_noise when --operator is omitted.

    Example: ul dataset evaluate interactions.jsonl --environment-config environment.json
    --allow-environment-network --confirm-test-environment
    --output results.jsonl

    Direct HTTP: ul dataset evaluate interactions.jsonl --target https://agent.test/invoke
    --confirm-request-isolation --confirm-safe-test-target --dry-run

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
    direct_http_options_used = (
        http_preset is not None
        or request_json_template is not None
        or response_json_pointer is not None
        or agent_model is not None
        or bool(header_from_env)
    )
    if environment_config is not None and target is not None:
        raise typer.BadParameter(
            "--target cannot be combined with --environment-config",
            param_hint="--target",
        )
    if target_artifact and target is None:
        raise typer.BadParameter(
            "--target-artifact requires --target", param_hint="--target-artifact"
        )
    if direct_http_options_used and target is None:
        raise typer.BadParameter(
            "direct HTTP mapping options require --target with an HTTP URL",
            param_hint="--target",
        )
    if confirm_target is not None and target is None:
        raise typer.BadParameter(
            "--confirm-target requires --target", param_hint="--confirm-target"
        )
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
            if environment_config is None and target is None:
                raise typer.BadParameter(
                    "execution requires --target or --environment-config",
                    param_hint="--target",
                )
            if environment_config is not None and not allow_environment_network:
                raise typer.BadParameter(
                    "execution requires --allow-environment-network",
                    param_hint="--allow-environment-network",
                )
            if not confirm_test_environment:
                raise typer.BadParameter(
                    TEST_ENVIRONMENT_CONFIRMATION_MESSAGE,
                    param_hint="--confirm-test-environment",
                )
        loaded_local_target: ResolvedLocalTarget | None = None
        resolved_http_target: ResolvedHttpTarget | None = None
        if target is not None:
            direct_http_target = target.casefold().startswith(("https://", "http://"))
            if direct_http_target and not confirm_request_isolation:
                raise typer.BadParameter(
                    "direct HTTP targets require --confirm-request-isolation",
                    param_hint="--confirm-request-isolation",
                )
            if direct_http_target and not confirm_safe_test_target:
                raise typer.BadParameter(
                    "direct HTTP targets require --confirm-safe-test-target",
                    param_hint="--confirm-safe-test-target",
                )
            try:
                resolved_http_target = resolve_http_target(
                    target,
                    allow_insecure_http=allow_insecure_http,
                    http_preset=http_preset,
                    request_json_template=request_json_template,
                    response_json_pointer=response_json_pointer,
                    agent_model=agent_model,
                    header_from_env=header_from_env,
                    request_isolation_attested=confirm_request_isolation,
                    safe_test_target_attested=confirm_safe_test_target,
                )
            except (OSError, ValidationError, ValueError):
                if (
                    target.casefold().startswith(("https://", "http://"))
                    or direct_http_options_used
                ):
                    raise
                loaded_local_target = resolve_local_target(
                    target,
                    explicit_artifacts=tuple(target_artifact or ()),
                )
            if resolved_http_target is not None and target_artifact:
                raise ValueError("--target-artifact applies only to local targets")
        loaded_target_config = (
            resolved_http_target.config
            if resolved_http_target is not None
            else (
                load_json_http_environment_config(environment_config)
                if environment_config is not None
                else _recorded_http_target_config(recorded_manifest_for_resume)
            )
        )
        recorded_http_confirmation = (
            recorded_manifest_for_resume.effective_command.http_target_confirmation
            if recorded_manifest_for_resume is not None
            else None
        )
        if (
            recorded_http_confirmation is not None
            and resolved_http_target is None
            and loaded_target_config is not None
        ):
            current_http_confirmation = resolve_http_target_config(
                recorded_http_confirmation.reference,
                loaded_target_config,
                allow_insecure_http=allow_insecure_http,
            ).confirmation
            if current_http_confirmation != recorded_http_confirmation:
                raise ValueError(
                    "HTTP target credential identity changed since this run was confirmed; "
                    "start a new evaluation and confirm the new target digest"
                )
        if (
            resume is not None
            and recorded_manifest_for_resume is not None
            and recorded_manifest_for_resume.run_context.target.kind == "probe_target"
            and recorded_manifest_for_resume.effective_command.http_target_config is None
            and loaded_local_target is None
        ):
            raise ValueError("local target resume requires the same explicit --target")
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
        if (
            resolved_http_target is not None
            and not dry_run
            and resume is None
            and not allow_environment_network
        ):
            raise ValueError("HTTP target execution requires --allow-environment-network")
        if (
            resolved_http_target is not None
            and not dry_run
            and resume is None
            and confirm_target != resolved_http_target.confirmation_sha256
        ):
            raise ValueError(
                "HTTP execution requires --confirm-target with the exact displayed digest"
            )
        if (
            loaded_local_target is not None
            and invariant_suite is not None
            and invariant_suite.observation_authority == "committed_state_snapshot"
        ):
            raise ValueError(
                "committed-state invariants require the stateful-lifecycle adapter tier; "
                "local targets provide response evidence only"
            )
        normalized_target_config = loaded_target_config
        if resume is not None and normalized_target_config is None and loaded_local_target is None:
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
        direct_http_target_receipt = _direct_http_target_receipt(
            resolved_http_target, recorded_manifest_for_resume
        )
        run_context = (
            build_dataset_evidence_run_context(
                selected_records=selected_records,
                selected_operator_ids=selected_operators,
                evaluation_mode=evaluation_mode,
                repetitions=repetitions,
                invariant_suite=invariant_suite,
                target_config=(
                    None if direct_http_target_receipt is not None else normalized_target_config
                ),
                target_receipt=(
                    local_target_evidence_receipt(loaded_local_target)
                    if loaded_local_target is not None
                    else direct_http_target_receipt
                ),
                settings=settings,
                redaction_policy_sha256=(
                    redaction_engine.policy.digest if redaction_engine is not None else None
                ),
                redaction_coverage=redaction_coverage,
            )
            if normalized_target_config is not None or loaded_local_target is not None
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
                deconstruct_reasoning=settings.deconstruct_reasoning,
                render_reasoning=settings.render_reasoning,
                equivalence_reasoning=settings.equivalence_reasoning,
                max_input_chars=settings.max_input_chars,
                max_output_tokens=settings.max_output_tokens,
                max_render_tokens=settings.max_render_tokens,
                max_response_bytes=settings.max_response_bytes,
                timeout_seconds=settings.timeout_seconds,
                deconstructor_identity=semantic_deconstructor_identity(settings),
            ),
            redaction_policy_sha256=(
                redaction_engine.policy.digest if redaction_engine is not None else None
            ),
            source_outcome_projection_sha256=_source_outcome_projection_sha256(
                normalized_target_config,
                loaded_local_target,
            ),
        )
    except (DatasetInputError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from None

    trial_journal: DatasetTrialJournal | None = None
    expected_manifest: DatasetRunManifest | None = None
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
            if trial_journal is not None:
                _reconcile_source_preparation_failures(trial_journal, resume_evidence)
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
            http_target_confirmation=_recorded_or_resolved_http_confirmation(
                resolved_http_target, recorded_manifest_for_resume
            ),
            http_target_config=_resolved_or_recorded_http_target_config(
                resolved_http_target, recorded_manifest_for_resume
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
                source_failure_ids: set[str] = (
                    {
                        failure.interaction_id
                        for failure in resume_evidence.source_preparation_failures
                    }
                    if resume_evidence is not None
                    else set()
                )
                terminal_interaction_ids = {
                    unit.interaction_id
                    for unit in recorded_manifest.work_plan
                    if unit.id in trial_journal.snapshot.terminal_states
                }
                if (
                    not recorded_manifest.effective_command.save_augmentations
                    and terminal_interaction_ids - source_failure_ids
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
        if loaded_local_target is not None:
            _print_local_target_identity(loaded_local_target)
        if resolved_http_target is not None:
            _print_http_target_identity(resolved_http_target)
        print_dataset_plan(
            record_count=len(records),
            selected_count=len(selected_records),
            skipped_count=skipped_count,
            operator_ids=selected_operators,
            evaluation_mode=evaluation_mode,
            target_configured=environment_config is not None or target is not None,
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
        assert run_context is not None
        finding_output = output.with_name(f"{output.name}.findings.jsonl")
        try:
            finding_reference_context = resolve_finding_reference_context(finding_output)
            _write_finding_package_snapshot(
                finding_output,
                resume_evidence.technical_results,
                resume_evidence.invariant_evaluations,
                invariant_suite.rules if invariant_suite is not None else (),
                campaign_id=run_context.context_sha256,
                reference_context=finding_reference_context,
            )
        except (OSError, ValueError) as error:
            message = str(error) if isinstance(error, ValueError) else error.__class__.__name__
            raise typer.BadParameter(
                f"cannot safely reconcile finding packages ({message})",
                param_hint="--resume",
            ) from None
        console.print(
            f"Resume compatible: all {skipped_count} selected interaction(s) are complete in "
            f"{output}. Nothing to do."
        )
        source_preparation_failure_count = len(resume_evidence.source_preparation_failures)
        if source_preparation_failure_count:
            console.print(
                f"Source preparation failures: {source_preparation_failure_count}; "
                "no target calls were made for those sources."
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
        if source_preparation_failure_count:
            raise typer.Exit(code=2)
        raise typer.Exit(code=0)

    if loaded_target_config is None and loaded_local_target is None:
        raise typer.BadParameter(
            "execution requires a recorded or explicit target",
            param_hint="--target",
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
        if loaded_local_target is not None:
            _print_local_target_identity(loaded_local_target)
            if confirm_target != loaded_local_target.confirmation_sha256:
                raise ValueError(
                    "local execution requires --confirm-target with the exact displayed digest"
                )
            loaded_local_target.revalidate_identity()
            execution_target = loaded_local_target.create_connection(
                campaign_plan.calls.total_environment_api,
                loaded_local_target.maximum_active_target_seconds,
            )
        else:
            assert loaded_target_config is not None
            if (
                resolved_http_target is not None
                and confirm_target != resolved_http_target.confirmation_sha256
            ):
                raise ValueError(
                    "HTTP execution requires --confirm-target with the exact displayed digest"
                )
            if not allow_environment_network:
                raise ValueError("environment execution requires --allow-environment-network")
            execution_target = JsonHttpEnvironmentConnection.from_config(
                loaded_target_config,
                test_environment_confirmed=True,
                allow_insecure_http=allow_insecure_http,
                max_environment_api_calls=max_environment_api_calls,
            )
    except ValueError as error:
        raise typer.BadParameter(
            str(error),
            param_hint=(
                "--target"
                if loaded_local_target is not None or resolved_http_target is not None
                else "--environment-config"
            ),
        ) from None

    resume_argv: tuple[str, ...] | None = None
    if loaded_local_target is not None:
        assert target is not None
        target_path = Path(target)
        action_target = str(target_path.resolve()) if target_path.is_file() else target
        local_resume_argv = [
            "ul",
            "dataset",
            "evaluate",
            "--resume",
            str(output.resolve()),
            "--target",
            action_target,
            "--confirm-target",
            loaded_local_target.confirmation_sha256,
        ]
        for artifact in target_artifact or ():
            local_resume_argv.extend(("--target-artifact", str(artifact.resolve())))
        resume_argv = tuple(local_resume_argv)

    progress_runtime = create_campaign_progress_runtime(
        case_count=len(all_selected_records),
        work_upper_bound=(
            len(expected_manifest.work_plan)
            if expected_manifest is not None
            else len(selected_records) * repetitions * (1 + len(selected_operators))
        ),
        target_call_budget=campaign_plan.calls.total_environment_api,
        semantic_call_budget=campaign_plan.calls.total_semantic_model,
        environment_call_budget=max_environment_api_calls,
        token_budget=campaign_plan.tokens.maximum,
        maximum_wall_time_seconds=(
            max(
                1,
                campaign_plan.calls.total_environment_api
                + campaign_plan.calls.total_semantic_model,
            )
            * settings.timeout_seconds
        ),
        next_commands=create_campaign_next_commands(output, resume_argv=resume_argv),
        json_output=progress_json,
    )
    if trial_journal is not None:
        journal_snapshot = trial_journal.snapshot
        terminal_states = journal_snapshot.terminal_states
        progress_runtime.tracker.hydrate_terminal_states(terminal_states)
        attempted_target_calls = _attempted_target_calls(journal_snapshot)
        progress_runtime.tracker.record_usage(
            target_calls=attempted_target_calls,
            semantic_calls=None,
            environment_calls=0 if not terminal_states else None,
            tokens=None,
        )

    def flush_progress_boundary() -> None:
        if trial_journal is not None:
            trial_journal.flush()

    progress_runtime.tracker.emit(status="running", stage="preflight")
    if evaluator_preflight is None:
        try:
            with progress_runtime.signal_control.installed():
                evaluator_preflight = asyncio.run(preflight_evaluator(settings))
            evaluator_preflight_receipt = persist_evaluator_preflight(output, evaluator_preflight)
            if not progress_runtime.tracker.safe_boundary(
                progress_runtime.control,
                flush_progress_boundary,
            ):
                asyncio.run(execution_target.aclose())
                if trial_journal is not None:
                    trial_journal.close()
                raise typer.Exit(code=130)
        except EvaluatorModelCompatibilityError as error:
            progress_runtime.tracker.emit(status="failed", stage="terminal")
            if trial_journal is not None:
                trial_journal.close()
            asyncio.run(execution_target.aclose())
            print_dataset_plain(f"Evaluation stopped before campaign execution: {error}")
            raise typer.Exit(code=2) from None
        except (OSError, ValueError) as error:
            progress_runtime.tracker.emit(status="failed", stage="terminal")
            if trial_journal is not None:
                trial_journal.close()
            asyncio.run(execution_target.aclose())
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
    finding_output = output.with_name(f"{output.name}.findings.jsonl")
    finding_reference_context: FindingReferenceContext | None = None
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
            assert trial_journal is not None
            output_stream, initial_evidence = open_resume_output(
                output,
                expected_context=run_context,
                selected_records=all_selected_records,
                invariant_suite=invariant_suite,
            )
            if initial_evidence.processed_ids:
                output_stream.close()
                raise ValueError("new evidence output is not empty")
            if (
                finding_output.exists()
                or finding_output.is_symlink()
                or finding_reference_key_path(finding_output).exists()
                or finding_reference_key_path(finding_output).is_symlink()
            ):
                raise ValueError("new finding package outputs already exist")
            finding_reference_context = resolve_finding_reference_context(finding_output)
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
            finding_reference_context = resolve_finding_reference_context(finding_output)
    except (OSError, ValueError) as error:
        if augmentation_ledger is not None:
            if augmentation_ledger_was_created:
                assert augmentations_output is not None
                augmentation_ledger.discard_if_empty(augmentations_output)
            augmentation_ledger.close()
        if output_stream is not None and not output_stream.closed:
            output_stream.close()
        asyncio.run(execution_target.aclose())
        message = str(error) if isinstance(error, ValueError) else error.__class__.__name__
        raise typer.BadParameter(
            f"cannot safely open persistence file ({message})",
            param_hint=failure_parameter,
        ) from None

    assert output_stream is not None
    assert finding_reference_context is not None
    has_review_findings = False
    invariant_evaluations: list[DatasetInvariantEvaluation] = []
    source_preparation_failures: list[DatasetSourcePreparationFailureEvidence] = []
    try:
        with output_stream:
            evaluation_parameters = inspect.signature(evaluate_interaction_records).parameters
            accepts_extra_arguments = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in evaluation_parameters.values()
            )
            durable_arguments: dict[str, object] = {}
            if "progress_plan" in evaluation_parameters or accepts_extra_arguments:
                durable_arguments["progress_plan"] = campaign_plan
            if "progress_runtime" in evaluation_parameters or accepts_extra_arguments:
                durable_arguments["progress_runtime"] = progress_runtime
            if "complete_progress" in evaluation_parameters or accepts_extra_arguments:
                durable_arguments["complete_progress"] = False
            if (
                "environment_calls_per_target_call" in evaluation_parameters
                or accepts_extra_arguments
            ):
                durable_arguments["environment_calls_per_target_call"] = target_calls_per_execution
            if trial_journal is not None and (
                "trial_journal" in evaluation_parameters or accepts_extra_arguments
            ):
                durable_arguments["trial_journal"] = trial_journal
            if progress_json and (
                "progress_json" in evaluation_parameters or accepts_extra_arguments
            ):
                durable_arguments["progress_json"] = True
            if "source_preparation_failures" in evaluation_parameters or accepts_extra_arguments:
                durable_arguments["source_preparation_failures"] = source_preparation_failures
            evaluation_runner = cast(Any, evaluate_interaction_records)
            if invariant_suite is not None:
                evaluation_coroutine = evaluation_runner(
                    selected_records,
                    selected_operators,
                    settings,
                    execution_target,
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
                    execution_target,
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
            prior_results = resume_evidence.technical_results if resume_evidence is not None else ()
            prior_invariant_evaluations = (
                resume_evidence.invariant_evaluations if resume_evidence is not None else ()
            )
            has_review_findings = _write_finding_package_snapshot(
                finding_output,
                (*prior_results, *results),
                (*prior_invariant_evaluations, *invariant_evaluations),
                invariant_suite.rules if invariant_suite is not None else (),
                campaign_id=run_context.context_sha256,
                reference_context=finding_reference_context,
            )
    except CampaignControlRequested:
        raise typer.Exit(code=130) from None
    except (TimeoutError, RuntimeError, ValueError, httpx.HTTPError) as error:
        if not progress_runtime.tracker.terminal_emitted:
            progress_runtime.tracker.emit(status="failed", stage="terminal")
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
    progress_runtime.tracker.emit(status="running", stage="report")
    all_source_preparation_failures = (
        resume_evidence.source_preparation_failures if resume_evidence is not None else ()
    ) + tuple(source_preparation_failures)
    try:
        print_dataset_results(
            results,
            output,
            augmentations_output=augmentations_output,
            invariant_evaluations=tuple(invariant_evaluations),
            show_report_guidance=show_report_guidance,
            source_preparation_failure_count=len(all_source_preparation_failures),
        )
    except Exception:
        progress_runtime.tracker.emit(status="failed", stage="terminal")
        raise
    progress_runtime.tracker.emit(
        status="failed" if all_source_preparation_failures else "completed",
        stage="terminal",
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
    if all_source_preparation_failures:
        raise typer.Exit(code=2)


def _recorded_http_target_config(
    recorded_manifest: DatasetRunManifest | None,
) -> JsonHttpTargetConfig | None:
    if recorded_manifest is None:
        return None
    direct_http_config = recorded_manifest.effective_command.http_target_config
    if direct_http_config is not None:
        return direct_http_config
    if recorded_manifest.run_context.target.kind == "environment_http":
        return recorded_manifest.run_context.target.config
    return None


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


def _recorded_or_resolved_http_confirmation(
    resolved_target: ResolvedHttpTarget | None,
    recorded_manifest: DatasetRunManifest | None,
) -> HttpTargetConfirmation | None:
    if resolved_target is not None:
        return resolved_target.confirmation
    if recorded_manifest is not None:
        return recorded_manifest.effective_command.http_target_confirmation
    return None


def _resolved_or_recorded_http_target_config(
    resolved_target: ResolvedHttpTarget | None,
    recorded_manifest: DatasetRunManifest | None,
) -> JsonHttpTargetConfig | None:
    if resolved_target is not None:
        return resolved_target.config
    if recorded_manifest is not None:
        return recorded_manifest.effective_command.http_target_config
    return None


def _print_local_target_identity(target: ResolvedLocalTarget) -> None:
    confirmation = target.confirmation
    print_dataset_plain("UL active-probe target")
    print_dataset_plain(f"  Kind: {target.kind}")
    print_dataset_plain(f"  Config sha256: {target.config_sha256}")
    print_dataset_plain(f"  Confirmation sha256: {target.confirmation_sha256}")
    print_dataset_plain(
        f"  Executable: {confirmation.executable.path} ({confirmation.executable.sha256})"
    )
    for artifact in confirmation.artifacts:
        print_dataset_plain(f"  Artifact: {artifact.path} ({artifact.sha256})")
    for environment in confirmation.environment:
        print_dataset_plain(
            f"  Environment: {environment.name} value sha256 {environment.value_sha256}"
        )
    if confirmation.callable is not None:
        print_dataset_plain(f"  Callable: {confirmation.callable}")
    print_dataset_plain("Use only a dedicated test target that cannot cause real-world effects.")


def _print_http_target_identity(target: ResolvedHttpTarget) -> None:
    print_dataset_plain("UL active-probe target")
    print_dataset_plain("  Kind: http")
    print_dataset_plain(f"  Config sha256: {target.config_sha256}")
    print_dataset_plain(f"  Confirmation sha256: {target.confirmation_sha256}")
    print_dataset_plain("Use only a dedicated test target that cannot cause real-world effects.")


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
            "evaluator.reasoning",
            (
                recorded.semantic_settings.deconstruct_reasoning,
                recorded.semantic_settings.render_reasoning,
                recorded.semantic_settings.equivalence_reasoning,
            ),
            (
                requested.semantic_settings.deconstruct_reasoning,
                requested.semantic_settings.render_reasoning,
                requested.semantic_settings.equivalence_reasoning,
            ),
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
            deconstruct_reasoning=recorded.deconstruct_reasoning,
            render_reasoning=recorded.render_reasoning,
            equivalence_reasoning=recorded.equivalence_reasoning,
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
        deconstruct_reasoning=recorded.deconstruct_reasoning,
        render_reasoning=recorded.render_reasoning,
        equivalence_reasoning=recorded.equivalence_reasoning,
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
