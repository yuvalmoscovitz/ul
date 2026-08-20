from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import sys
import unicodedata
from pathlib import Path
from typing import Annotated, TextIO, cast

import httpx
import typer
from pydantic import JsonValue, SecretStr, ValidationError
from rich.console import Console
from rich.table import Table
from ul import (
    DatasetAugmentationEngine,
    DatasetAugmentationOperator,
    DatasetAugmentationResult,
    DatasetEvaluationCase,
    DatasetEvaluationFinding,
    DatasetEvaluationResult,
    DatasetEvaluationRunner,
    DatasetEvaluationTrialSet,
    DatasetSemanticSettings,
    DatasetTargetLifecycleFailure,
    InteractionRecord,
    LocalPseudonymStore,
    RedactedSemanticPipeline,
    RedactionEngine,
    create_semantic_model_deconstructor,
    load_dataset_semantic_settings,
    load_redaction_policy,
    resolve_dataset_augmentation_operator,
)
from ul.dataset_invariants import (
    DatasetInvariantArrayUniqueTrialEvaluation,
    DatasetInvariantEvaluation,
    DatasetInvariantSuite,
    DatasetInvariantTrialEvaluation,
    DatasetInvariantValueEqualsTrialEvaluation,
    DatasetInvariantValueInSetTrialEvaluation,
    evaluate_dataset_invariants,
    load_dataset_invariant_suite,
)
from ul.http_sandbox import (
    JsonHttpSandboxConfig,
    JsonHttpSandboxConnection,
    json_http_sandbox_calls_per_execution,
    json_http_sandbox_config_urls,
    load_json_http_sandbox_config,
    validate_json_http_sandbox_configuration,
)
from ul_core.augmentation_catalog import builtin_augmentation_catalog
from ul_core.dataset import ObservedOutcome

from ul_cli.dataset_augmentation_ledger import (
    DatasetAugmentationLedger,
    DatasetAugmentationLedgerSemanticSettings,
    create_dataset_augmentation_generation_context,
    create_private_augmentation_ledger,
    open_augmentation_ledger_for_resume,
    read_augmentation_ledger,
)
from ul_cli.dataset_ingest import app as ingest_app
from ul_cli.dataset_review import (
    DatasetEvidenceRedactionCoverage,
    DatasetEvidenceRunContext,
    DatasetEvidenceSemanticSettings,
    DatasetResumeEvidence,
    create_dataset_evidence_run_context,
    report_dataset_evidence,
    review_dataset_finding,
    validate_dataset_resume_evidence,
)

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

app = typer.Typer(help="Explore behavioral differences in observed agent interactions.")
console = Console()

app.add_typer(ingest_app, name="ingest")
app.command("report")(report_dataset_evidence)
app.command("review")(review_dataset_finding)

_MAXIMUM_DATASET_BYTES = 10_000_000
_MAXIMUM_DATASET_RECORDS = 100
_MAXIMUM_EVIDENCE_BYTES = 128_000_000
_DEFAULT_MAXIMUM_SANDBOX_API_CALLS = 100
_REDACTION_KEY_ENVIRONMENT_VARIABLE = "UL_DATASET_REDACTION_KEY"
_CUSTOMER_STATUSES = {
    "augmentation_rejected": "VARIATION DISCARDED",
    "inconclusive": "COULDN'T DETERMINE",
    "no_divergence": "NO OBSERVED DIFFERENCE",
    "divergence_needs_review": "DIFFERENCE — REVIEW",
}
_FINDING_LABELS = {
    "duplicate_effect": "duplicate action",
    "unexpected_effect": "unexpected action",
    "missing_effect": "missing action",
    "changed_grounded_effect_argument": "changed action value",
}
_BEHAVIORAL_LIMITATIONS = (
    "UL compares observed action behavior only. It does not determine whether the original or "
    "variation is correct, prove that the variation caused a difference, or estimate a "
    "production failure rate."
)


class _DatasetInputError(ValueError):
    pass


@app.command("init")
def initialize_dataset_sandbox(
    sandbox_config: Annotated[
        Path,
        typer.Argument(
            dir_okay=False,
            help="New JSON file describing the customer-managed sandbox API.",
        ),
    ],
    url: Annotated[
        str,
        typer.Option(help="Base URL of the customer's isolated agent sandbox API."),
    ],
) -> None:
    """Create a private connection config for a customer-managed agent sandbox API."""
    try:
        base_url = url.rstrip("/")
        config = JsonHttpSandboxConfig.model_validate(
            {
                "version": 3,
                "sandbox_id": "replace-with-stable-sandbox-id",
                "headers_from_env": {},
                "reset": {
                    "url": f"{base_url}/reset",
                    "request_json_template": {"case_id": "{{case_id}}"},
                    "case_id_json_pointer": "/case_id",
                    "generation_json_pointer": "/generation",
                    "clean_state_json_pointer": "/clean",
                    "clean_state_value": True,
                    "sandbox_id_json_pointer": "/sandbox_id",
                },
                "setup": {
                    "url": f"{base_url}/setup",
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "fixture": "default",
                    },
                    "case_id_json_pointer": "/case_id",
                    "sandbox_id_json_pointer": "/sandbox_id",
                },
                "execute_turn": {
                    "url": f"{base_url}/execute",
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "turn_id": "{{turn_id}}",
                        "input": "{{input}}",
                    },
                    "response_json_pointer": "/response",
                    "case_id_json_pointer": "/case_id",
                    "turn_id_json_pointer": "/turn_id",
                    "sandbox_id_json_pointer": "/sandbox_id",
                },
                "snapshot": {
                    "url": f"{base_url}/snapshot",
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "turn_id": "{{turn_id}}",
                    },
                    "response_json_pointer": "/state",
                    "case_id_json_pointer": "/case_id",
                    "turn_id_json_pointer": "/turn_id",
                    "sandbox_id_json_pointer": "/sandbox_id",
                },
            }
        )
        output_stream = _create_private_output(sandbox_config)
    except (OSError, ValidationError, ValueError) as error:
        if isinstance(error, FileExistsError):
            message = "sandbox config already exists; UL will not overwrite it"
        elif isinstance(error, OSError):
            message = f"cannot create sandbox config ({error.__class__.__name__})"
        elif isinstance(error, ValidationError):
            message = "sandbox config is invalid"
        else:
            message = str(error)
        raise typer.BadParameter(message, param_hint="SANDBOX_CONFIG") from None

    with output_stream:
        json.dump(config.model_dump(mode="json", exclude_none=True), output_stream, indent=2)
        output_stream.write("\n")

    console.print(f"Created private sandbox connection config: {sandbox_config}")
    console.print(
        "Next: adjust the lifecycle request bodies and response pointers, add any "
        "headers_from_env, then validate the connection with 'ul sandbox check "
        f'{sandbox_config} --probe "Return sandbox health only; do not take action." '
        "--allow-sandbox-network-egress "
        "--confirm-isolated-sandbox --confirm-harmless-probe'. After that, validate a "
        "dataset plan with 'ul dataset evaluate DATASET --sandbox-config "
        f"{sandbox_config} --dry-run'."
    )
    console.print(
        "Setup uses one static fixture for the sandbox. Keep exactly one complete "
        "{{case_id}} value in every lifecycle request, {{turn_id}} in execute_turn and snapshot, "
        "and one {{input}} value in execute_turn. "
        "headers_from_env maps HTTP header names to "
        "dedicated UL_SANDBOX_* environment-variable names; secret values stay outside this file."
    )


@app.command("operators")
def list_dataset_operators() -> None:
    """List the dataset subset of UL's unified augmentation catalog."""
    console.print("Dataset augmentations (from 'ul augmentations list')")
    for augmentation in builtin_augmentation_catalog().list(mode="dataset_variation"):
        console.print(f"- {augmentation.ref.id}@{augmentation.ref.version}: {augmentation.summary}")


@app.command("evaluate")
def evaluate_dataset(
    data: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help='JSONL containing one {"id": ..., "input": ..., "output": ...} object per line.',
        ),
    ],
    sandbox_config: Annotated[
        Path | None,
        typer.Option(
            "--sandbox-config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Connection to the customer's isolated agent sandbox API.",
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
        int,
        typer.Option(min=1, max=_MAXIMUM_DATASET_RECORDS, help="Interactions to evaluate."),
    ] = 10,
    repetitions: Annotated[
        int,
        typer.Option(
            min=1,
            help="Fresh-state sandbox executions per original input and accepted variation.",
        ),
    ] = 3,
    max_sandbox_api_calls: Annotated[
        int,
        typer.Option(
            "--max-sandbox-api-calls",
            min=1,
            help="Maximum customer sandbox API requests authorized for this evaluation.",
        ),
    ] = _DEFAULT_MAXIMUM_SANDBOX_API_CALLS,
    allow_sandbox_network_egress: Annotated[
        bool,
        typer.Option(
            "--allow-sandbox-network-egress",
            help="Allow UL to call the configured remote sandbox API.",
        ),
    ] = False,
    confirm_isolated_sandbox: Annotated[
        bool,
        typer.Option(
            help=(
                "Attest that the configured endpoint is a customer-managed, isolated "
                "non-production sandbox. UL does not verify its isolation."
            )
        ),
    ] = False,
    allow_insecure_http: Annotated[
        bool,
        typer.Option(help="Allow an HTTP sandbox API. Intended for local sandboxes."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(help="Validate and show the execution plan without external calls."),
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
) -> None:
    """Explore behavioral differences against an isolated black-box agent.

    UL_LIVE=true enables billed semantic-model calls and external processing together.
    UL_DATASET_LIVE_CALLS and UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING remain separate,
    higher-precedence controls. OpenRouter remains the default; set
    UL_DATASET_SEMANTIC_PROVIDER=openai-compatible for a customer-controlled endpoint.

    UL calls only the configured customer-managed sandbox API through an explicit
    reset/setup/execute/snapshot lifecycle. Production observations are passive source data and
    cannot select or configure the execution destination.

    Example: ul dataset evaluate interactions.jsonl --sandbox-config sandbox.json
    --allow-sandbox-network-egress --confirm-isolated-sandbox
    --output results.jsonl

    Discover operators: ul augmentations list --mode dataset_variation
    Augmentation retention: --augmentations-output PATH or --no-save-augmentations
    """
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
        augmentations_output = _default_augmentations_output(output)
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
        records = _load_interaction_records(data)
        selected_operators = _validate_operator_ids(operator)
        invariant_suite = (
            load_dataset_invariant_suite(invariants) if invariants is not None else None
        )
        selected_records = records[:limit]
        redaction_engine = _load_redaction_engine(
            redaction_policy,
            redaction_state,
            state_required=not dry_run or resume is not None,
        )
        redaction_coverage = _redaction_coverage(selected_records, redaction_engine)
        if redaction_engine is not None and (not dry_run or resume is not None):
            selected_records = _protect_interaction_records(selected_records, redaction_engine)
        all_selected_records = selected_records
        if not dry_run and resume is None:
            if sandbox_config is None:
                raise typer.BadParameter(
                    "execution requires --sandbox-config",
                    param_hint="--sandbox-config",
                )
            if not allow_sandbox_network_egress:
                raise typer.BadParameter(
                    "execution requires --allow-sandbox-network-egress",
                    param_hint="--allow-sandbox-network-egress",
                )
            if not confirm_isolated_sandbox:
                raise typer.BadParameter(
                    "execution requires --confirm-isolated-sandbox",
                    param_hint="--confirm-isolated-sandbox",
                )
        loaded_target_config = (
            load_json_http_sandbox_config(sandbox_config) if sandbox_config is not None else None
        )
        if loaded_target_config is not None:
            validate_json_http_sandbox_configuration(
                loaded_target_config,
                sandbox_confirmed=confirm_isolated_sandbox or dry_run or resume is not None,
                allow_insecure_http=allow_insecure_http,
            )
        normalized_target_config = loaded_target_config
        if resume is not None and normalized_target_config is None:
            raise ValueError("--resume requires --sandbox-config")
        target_calls_per_execution = (
            json_http_sandbox_calls_per_execution(normalized_target_config)
            if normalized_target_config is not None
            else 1
        )
        initial_target_calls = (
            len(selected_records)
            * repetitions
            * (1 + len(selected_operators))
            * target_calls_per_execution
        )
        if resume is None and initial_target_calls > max_sandbox_api_calls:
            raise ValueError(
                f"selection would make up to {initial_target_calls} sandbox API calls, exceeding "
                f"--max-sandbox-api-calls {max_sandbox_api_calls}; reduce --limit, --operator, or "
                "--repetitions, or explicitly raise the call budget"
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
        settings = load_dataset_semantic_settings()
        _validate_model_input_bounds(selected_records, settings.max_input_chars)
        run_context = (
            _dataset_evidence_run_context(
                selected_records=selected_records,
                selected_operator_ids=selected_operators,
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
                _dataset_operator_identity(operator_reference)
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
    except (_DatasetInputError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from None

    resume_evidence: DatasetResumeEvidence | None = None
    saved_augmentations: dict[str, DatasetAugmentationResult] = {}
    skipped_count = 0
    if resume is not None:
        assert output is not None
        assert run_context is not None
        try:
            resume_evidence = _read_resume_evidence(
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
            message = str(error) if isinstance(error, ValueError) else error.__class__.__name__
            raise typer.BadParameter(
                f"cannot safely resume evidence ({message})",
                param_hint="--resume",
            ) from None
        selected_records = tuple(
            record for record in selected_records if record.id not in resume_evidence.processed_ids
        )
        skipped_count = len(resume_evidence.processed_ids)

    potential_target_calls = (
        len(selected_records)
        * repetitions
        * (1 + len(selected_operators))
        * target_calls_per_execution
    )
    if potential_target_calls > max_sandbox_api_calls:
        raise typer.BadParameter(
            f"remaining selection would make up to {potential_target_calls} sandbox API calls, "
            f"exceeding --max-sandbox-api-calls {max_sandbox_api_calls}; reduce --limit, "
            "--operator, "
            "or --repetitions, or explicitly raise the call budget"
        )

    if dry_run:
        _print_dataset_plan(
            record_count=len(records),
            selected_count=len(selected_records),
            skipped_count=skipped_count,
            operator_ids=selected_operators,
            target_configured=sandbox_config is not None,
            target_endpoint=(
                json_http_sandbox_config_urls(loaded_target_config)[0]
                if loaded_target_config is not None
                else None
            ),
            target_header_environment_variables=(
                loaded_target_config.headers_from_env if loaded_target_config is not None else {}
            ),
            repetitions=repetitions,
            max_sandbox_api_calls=max_sandbox_api_calls,
            target_calls_per_execution=target_calls_per_execution,
            invariant_suite=invariant_suite,
            output=output,
            augmentations_output=augmentations_output,
            semantic_provider_id=settings.semantic_provider_id,
            semantic_endpoint_sha256=settings.semantic_endpoint_sha256,
            redaction_policy_sha256=(
                redaction_engine.policy.digest if redaction_engine is not None else None
            ),
            redaction_coverage=redaction_coverage,
        )
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
        previous_invariant_exit_code = _invariant_exit_code(resume_evidence.invariant_evaluations)
        if previous_invariant_exit_code:
            raise typer.Exit(code=previous_invariant_exit_code)
        if resume_evidence.has_review_findings:
            raise typer.Exit(code=1)
        raise typer.Exit(code=0)

    if sandbox_config is None:
        raise typer.BadParameter(
            "execution requires --sandbox-config",
            param_hint="--sandbox-config",
        )
    if not confirm_isolated_sandbox:
        raise typer.BadParameter(
            "execution requires --confirm-isolated-sandbox",
            param_hint="--confirm-isolated-sandbox",
        )
    if output is None:
        raise typer.BadParameter("execution requires --output", param_hint="--output")
    if output.exists() and resume is None:
        raise typer.BadParameter(
            "output already exists; UL will not overwrite it",
            param_hint="--output",
        )

    assert run_context is not None
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
        if not allow_sandbox_network_egress:
            raise ValueError("sandbox execution requires --allow-sandbox-network-egress")
        target = JsonHttpSandboxConnection.from_config(
            loaded_target_config,
            sandbox_confirmed=True,
            allow_insecure_http=allow_insecure_http,
            max_sandbox_api_calls=max_sandbox_api_calls,
        )
    except ValueError as error:
        raise typer.BadParameter(
            str(error),
            param_hint="--sandbox-config",
        ) from None

    augmentation_ledger: DatasetAugmentationLedger | None = None
    augmentation_ledger_was_created = False
    output_stream: TextIO | None = None
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
            output_stream = _create_private_output(output)
        else:
            assert resume_evidence is not None
            output_stream, locked_resume_evidence = _open_resume_output(
                output,
                expected_context=run_context,
                selected_records=all_selected_records,
                invariant_suite=invariant_suite,
            )
            if locked_resume_evidence != resume_evidence:
                output_stream.close()
                raise ValueError("resume evidence changed after preflight")
    except (OSError, ValueError) as error:
        if augmentation_ledger is not None:
            if augmentation_ledger_was_created:
                assert augmentations_output is not None
                augmentation_ledger.discard_if_empty(augmentations_output)
            augmentation_ledger.close()
        if output_stream is not None and not output_stream.closed:
            output_stream.close()
        asyncio.run(target.aclose())
        message = str(error) if isinstance(error, ValueError) else error.__class__.__name__
        raise typer.BadParameter(
            f"cannot safely open persistence file ({message})",
            param_hint=failure_parameter,
        ) from None

    assert output_stream is not None
    has_review_findings = False
    invariant_evaluations: list[DatasetInvariantEvaluation] = []
    try:
        with output_stream:
            if invariant_suite is not None:
                evaluation_coroutine = _evaluate_interaction_records(
                    selected_records,
                    selected_operators,
                    settings,
                    target,
                    output_stream,
                    repetitions=repetitions,
                    max_sandbox_api_calls=max_sandbox_api_calls,
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
                )
            else:
                evaluation_coroutine = _evaluate_interaction_records(
                    selected_records,
                    selected_operators,
                    settings,
                    target,
                    output_stream,
                    repetitions=repetitions,
                    max_sandbox_api_calls=max_sandbox_api_calls,
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
                )
            results = asyncio.run(evaluation_coroutine)
            for result in results:
                has_review_findings |= _result_needs_review(result)
    except (TimeoutError, RuntimeError, ValueError, httpx.HTTPError) as error:
        console.print(
            f"Evaluation stopped ({error.__class__.__name__}). "
            f"Complete results written before the error remain in {output}."
        )
        if augmentations_output is not None:
            _print_dataset_plain(
                f"Generated augmentations remain in {augmentations_output} and will be reused "
                f"with --resume {output}."
            )
        raise typer.Exit(code=2) from None
    finally:
        if augmentation_ledger is not None:
            augmentation_ledger.close()

    if skipped_count > 0:
        console.print(
            f"Resumed: {skipped_count} interaction(s) skipped (already in evidence), "
            f"{len(results)} newly evaluated."
        )
    _print_dataset_results(
        results,
        output,
        augmentations_output=augmentations_output,
        invariant_evaluations=tuple(invariant_evaluations),
    )
    prior_invariant_evaluations = (
        resume_evidence.invariant_evaluations if resume_evidence is not None else ()
    )
    invariant_exit_code = _invariant_exit_code(
        (*prior_invariant_evaluations, *invariant_evaluations)
    )
    if invariant_exit_code == 1:
        raise typer.Exit(code=1)
    if invariant_exit_code == 2:
        raise typer.Exit(code=2)
    if has_review_findings or (resume_evidence is not None and resume_evidence.has_review_findings):
        raise typer.Exit(code=1)


def _load_interaction_records(path: Path) -> tuple[InteractionRecord, ...]:
    dataset_name = path.name
    try:
        with path.open("rb") as dataset_stream:
            encoded_dataset = dataset_stream.read(_MAXIMUM_DATASET_BYTES + 1)
        if len(encoded_dataset) > _MAXIMUM_DATASET_BYTES:
            raise _DatasetInputError(
                f"{dataset_name}: dataset exceeds {_MAXIMUM_DATASET_BYTES} bytes"
            )
        lines = encoded_dataset.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        raise _DatasetInputError(f"{dataset_name}: dataset must be UTF-8") from None
    except OSError as error:
        raise _DatasetInputError(
            f"{dataset_name}: cannot read dataset ({error.__class__.__name__})"
        ) from None

    records: list[InteractionRecord] = []
    known_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise _DatasetInputError(
                f"{dataset_name} line {line_number}: blank lines are not allowed"
            )
        if len(records) == _MAXIMUM_DATASET_RECORDS:
            raise _DatasetInputError(
                f"{dataset_name} line {line_number}: dataset exceeds "
                f"{_MAXIMUM_DATASET_RECORDS} records"
            )
        try:
            untyped_payload = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonstandard_json_constant,
            )
        except (json.JSONDecodeError, RecursionError, ValueError):
            raise _DatasetInputError(f"{dataset_name} line {line_number}: invalid JSON") from None
        if not isinstance(untyped_payload, dict):
            raise _DatasetInputError(f"{dataset_name} line {line_number}: record must be an object")
        payload = cast(dict[str, JsonValue], untyped_payload)
        expected_fields = {"id", "input", "output"}
        payload_fields = set(payload)
        if payload_fields != expected_fields:
            missing_fields = sorted(expected_fields - payload_fields)
            unknown_fields = sorted(payload_fields - expected_fields)
            details: list[str] = []
            if missing_fields:
                details.append(f"missing {', '.join(missing_fields)}")
            if unknown_fields:
                details.append("unknown field(s)")
            raise _DatasetInputError(
                f"{dataset_name} line {line_number}: expected exactly id, input, output "
                f"({'; '.join(details)})"
            )
        try:
            record = InteractionRecord.model_validate(
                {
                    "id": payload["id"],
                    "raw_input": payload["input"],
                    "raw_observed_output": payload["output"],
                }
            )
        except ValidationError as error:
            invalid_fields = sorted(
                {
                    {"raw_input": "input", "raw_observed_output": "output"}.get(
                        str(item["loc"][0]), str(item["loc"][0])
                    )
                    for item in error.errors(include_input=False)
                    if item["loc"]
                }
            )
            field_summary = ", ".join(invalid_fields) or "record"
            raise _DatasetInputError(
                f"{dataset_name} line {line_number}: invalid {field_summary}"
            ) from None
        if record.id in known_ids:
            raise _DatasetInputError(f"{dataset_name} line {line_number}: duplicate id")
        known_ids.add(record.id)
        records.append(record)
    if not records:
        raise _DatasetInputError(f"{dataset_name}: dataset contains no records")
    return tuple(records)


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _validate_model_input_bounds(
    records: tuple[InteractionRecord, ...],
    maximum_characters: int,
) -> None:
    for case_number, record in enumerate(records, start=1):
        serialized_record = json.dumps(
            {
                "raw_input": record.raw_input,
                "raw_observed_output": record.raw_observed_output,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(serialized_record) > maximum_characters:
            raise _DatasetInputError(
                f"selected interaction {case_number} exceeds the semantic model input limit"
            )


def _validate_operator_ids(operator_ids: list[str] | None) -> tuple[str, ...]:
    selected_references = tuple(operator_ids or ["input.surface.rephrase"])
    selected_operators: list[DatasetAugmentationOperator] = []
    for reference in selected_references:
        try:
            operator = resolve_dataset_augmentation_operator(reference)
        except ValueError:
            raise _DatasetInputError("unknown augmentation operator reference") from None
        selected_operators.append(operator)
    resolved_references = tuple((operator.id, operator.version) for operator in selected_operators)
    if len(resolved_references) != len(set(resolved_references)):
        raise _DatasetInputError("duplicate --operator values are not allowed")
    return tuple(
        reference if "@" in reference else operator.id
        for reference, operator in zip(selected_references, selected_operators, strict=True)
    )


def _dataset_operator_identity(reference: str) -> tuple[str, str]:
    operator = resolve_dataset_augmentation_operator(reference)
    return operator.id, operator.version


def _load_redaction_engine(
    policy_path: Path | None,
    state_path: Path | None,
    *,
    state_required: bool,
) -> RedactionEngine | None:
    if policy_path is None:
        if state_path is not None:
            raise ValueError("--redaction-state requires --redaction-policy")
        return None
    policy = load_redaction_policy(policy_path)
    if not state_required:
        return RedactionEngine(
            policy,
            LocalPseudonymStore(Path("unused-redaction-state"), SecretStr("0" * 32)),
        )
    if state_path is None:
        raise ValueError("--redaction-policy requires --redaction-state for execution and resume")
    key = os.environ.get(_REDACTION_KEY_ENVIRONMENT_VARIABLE, "")
    if len(key.encode()) < 32:
        raise ValueError(f"set {_REDACTION_KEY_ENVIRONMENT_VARIABLE} to at least 32 UTF-8 bytes")
    return RedactionEngine(policy, LocalPseudonymStore(state_path, SecretStr(key)))


def _redaction_coverage(
    records: tuple[InteractionRecord, ...],
    engine: RedactionEngine | None,
) -> tuple[DatasetEvidenceRedactionCoverage, ...]:
    if engine is None:
        return ()
    coverage_by_location: list[DatasetEvidenceRedactionCoverage] = []
    for location in ("input", "output"):
        matched_values = 0
        matched_paths: set[str] = set()
        matches_by_rule: dict[str, int] = {}
        for record in records:
            value: JsonValue = (
                record.raw_input if location == "input" else record.raw_observed_output
            )
            coverage = engine.transform(value, location=location, dry_run=True).coverage
            matched_values += coverage.matched_values
            matched_paths.update(coverage.matched_paths)
            for rule_name, count in coverage.matches_by_rule.items():
                matches_by_rule[rule_name] = matches_by_rule.get(rule_name, 0) + count
        coverage_by_location.append(
            DatasetEvidenceRedactionCoverage(
                location=location,
                matched_values=matched_values,
                matched_paths=tuple(sorted(matched_paths)),
                matches_by_rule=dict(sorted(matches_by_rule.items())),
            )
        )
    return tuple(coverage_by_location)


def _protect_interaction_records(
    records: tuple[InteractionRecord, ...], engine: RedactionEngine
) -> tuple[InteractionRecord, ...]:
    protected_records: list[InteractionRecord] = []
    for record in records:
        protected_input = engine.transform(record.raw_input, location="input").value
        if not isinstance(protected_input, str):
            raise ValueError("redaction policy did not preserve executable input as text")
        protected_records.append(
            record.model_copy(
                update={
                    "raw_input": protected_input,
                    "raw_observed_output": engine.transform(
                        record.raw_observed_output, location="output"
                    ).value,
                }
            )
        )
    return tuple(protected_records)


def _dataset_evidence_run_context(
    *,
    selected_records: tuple[InteractionRecord, ...],
    selected_operator_ids: tuple[str, ...],
    repetitions: int,
    invariant_suite: DatasetInvariantSuite | None,
    target_config: JsonHttpSandboxConfig | None,
    settings: DatasetSemanticSettings,
    redaction_policy_sha256: str | None = None,
    redaction_coverage: tuple[DatasetEvidenceRedactionCoverage, ...] = (),
) -> DatasetEvidenceRunContext:
    return create_dataset_evidence_run_context(
        selected_records=selected_records,
        operators=tuple(
            _dataset_operator_identity(reference) for reference in selected_operator_ids
        ),
        repetitions=repetitions,
        invariant_suite_sha256=(invariant_suite.sha256 if invariant_suite is not None else None),
        target_config=target_config,
        semantic_settings=DatasetEvidenceSemanticSettings(
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
        redaction_policy_sha256=redaction_policy_sha256,
        redaction_coverage=redaction_coverage,
    )


def _print_dataset_plan(
    *,
    record_count: int,
    selected_count: int,
    skipped_count: int,
    operator_ids: tuple[str, ...],
    target_configured: bool,
    target_endpoint: str | None,
    target_header_environment_variables: dict[str, str],
    repetitions: int,
    max_sandbox_api_calls: int,
    target_calls_per_execution: int,
    invariant_suite: DatasetInvariantSuite | None,
    output: Path | None,
    augmentations_output: Path | None,
    semantic_provider_id: str,
    semantic_endpoint_sha256: str,
    redaction_policy_sha256: str | None,
    redaction_coverage: tuple[DatasetEvidenceRedactionCoverage, ...],
) -> None:
    potential_target_calls = (
        selected_count * repetitions * (1 + len(operator_ids)) * target_calls_per_execution
    )
    potential_model_calls = selected_count * (
        1 + 3 * len(operator_ids) + repetitions * (1 + len(operator_ids))
    )
    console.print(f"Dataset valid: {record_count} interaction(s)")
    if skipped_count:
        console.print(
            f"Resume compatible: {skipped_count} complete interaction(s) skipped; "
            f"{selected_count} remaining"
        )
    else:
        console.print(f"Selected interactions: {selected_count}")
    console.print(f"Operators: {', '.join(operator_ids)}")
    console.print(f"Repetitions: {repetitions} per original and accepted variation")
    if invariant_suite is None:
        console.print("Customer invariants: none")
    else:
        console.print(f"Customer invariants: {len(invariant_suite.rules)} rule(s)")
        console.print(f"Declared observation authority: {invariant_suite.observation_authority}")
        console.print("Additional model calls for customer invariants: 0")
        console.print("Additional sandbox API calls for customer invariants: 0")
    console.print(f"Potential semantic model calls: up to {potential_model_calls}")
    console.print(
        f"Semantic provider: {semantic_provider_id} "
        f"(endpoint sha256: {semantic_endpoint_sha256[:12]})"
    )
    if redaction_policy_sha256 is not None:
        console.print(f"Redaction policy sha256: {redaction_policy_sha256}")
        for coverage in redaction_coverage:
            console.print(
                f"Redaction coverage ({coverage.location}): "
                f"{coverage.matched_values} selected value(s) across "
                f"{len(coverage.matched_paths)} path(s)"
            )
    console.print(
        f"Potential sandbox API calls: up to {potential_target_calls} "
        f"(authorized maximum: {max_sandbox_api_calls})"
    )
    if target_calls_per_execution > 1:
        console.print(
            f"Lifecycle calls per execution: {target_calls_per_execution} "
            "(reset, optional setup, execute_turn, snapshot, cleanup reset)"
        )
    console.print(
        f"Customer-managed sandbox API: {'configured' if target_configured else 'not configured'}"
    )
    if output is not None:
        console.print(f"Evidence destination: {output}")
    if augmentations_output is None:
        console.print(
            "Augmentations will not be saved. Interrupted generation may repeat model calls."
        )
    else:
        _print_dataset_plain(f"Augmentations destination: {augmentations_output}")
        _print_dataset_plain(
            "This private artifact may contain sensitive inputs and derived semantic data. It is "
            "not encrypted or automatically redacted; retain it only under your data policy."
        )
    if target_endpoint is not None:
        console.print(f"Sandbox API endpoint: {target_endpoint}")
        if target_header_environment_variables:
            mappings = ", ".join(
                f"{header_name}={environment_variable}"
                for header_name, environment_variable in sorted(
                    target_header_environment_variables.items()
                )
            )
            console.print(f"Sandbox API header environment mappings: {mappings}")
        else:
            console.print("Sandbox API header environment mappings: none")
    console.print(
        "Semantic models receive historical inputs and outputs, generated variations, "
        "live control responses, and variation responses on execution."
    )
    console.print(
        "Every test case invokes and validates the configured sandbox reset contract. Optional "
        "setup uses one static fixture from the sandbox config for the entire run."
    )
    console.print(
        "Target requests and semantic model calls may be billed separately. Repetitions only "
        "show observed behavioral consistency: they do not determine correctness, identify "
        "causality, or estimate a production failure rate."
    )
    console.print("No model or sandbox API requests sent.")


def _default_augmentations_output(evidence_output: Path) -> Path:
    if evidence_output.suffix:
        return evidence_output.with_name(f"{evidence_output.stem}.augmentations.jsonl")
    return evidence_output.with_name(f"{evidence_output.name}.augmentations.jsonl")


def _create_private_output(path: Path) -> TextIO:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.fchmod(descriptor, 0o600)
    return os.fdopen(descriptor, "w", encoding="utf-8")


def _read_resume_evidence(
    path: Path,
    *,
    expected_context: DatasetEvidenceRunContext,
    selected_records: tuple[InteractionRecord, ...],
    invariant_suite: DatasetInvariantSuite | None,
) -> DatasetResumeEvidence:
    descriptor = _open_resume_descriptor(path, writable=False)
    try:
        return _read_resume_descriptor(
            descriptor,
            expected_context=expected_context,
            selected_records=selected_records,
            invariant_suite=invariant_suite,
        )
    finally:
        os.close(descriptor)


def _open_resume_output(
    path: Path,
    *,
    expected_context: DatasetEvidenceRunContext,
    selected_records: tuple[InteractionRecord, ...],
    invariant_suite: DatasetInvariantSuite | None,
) -> tuple[TextIO, DatasetResumeEvidence]:
    descriptor = _open_resume_descriptor(path, writable=True)
    try:
        resume_evidence = _read_resume_descriptor(
            descriptor,
            expected_context=expected_context,
            selected_records=selected_records,
            invariant_suite=invariant_suite,
        )
        if sys.platform != "win32":
            os.fchmod(descriptor, 0o600)
        os.lseek(descriptor, 0, os.SEEK_END)
        return os.fdopen(descriptor, "a", encoding="utf-8"), resume_evidence
    except BaseException:
        os.close(descriptor)
        raise


def _open_resume_descriptor(path: Path, *, writable: bool) -> int:
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    binary_flag = os.O_BINARY if sys.platform == "win32" else 0
    path_status = os.lstat(path)
    if not stat.S_ISREG(path_status.st_mode):
        raise OSError("resume evidence is not a regular file")
    access_flags = os.O_RDWR | os.O_APPEND if writable else os.O_RDONLY
    descriptor = os.open(path, access_flags | no_follow_flag | binary_flag)
    try:
        descriptor_status = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_status.st_mode) or (
            path_status.st_dev,
            path_status.st_ino,
        ) != (descriptor_status.st_dev, descriptor_status.st_ino):
            raise OSError("resume evidence changed while opening")
        if sys.platform == "win32":
            os.lseek(descriptor, 0, os.SEEK_SET)
            lock_mode = msvcrt.LK_NBLCK if writable else msvcrt.LK_NBRLCK
            msvcrt.locking(descriptor, lock_mode, 1)
        else:
            lock_mode = fcntl.LOCK_EX if writable else fcntl.LOCK_SH
            fcntl.flock(descriptor, lock_mode | fcntl.LOCK_NB)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_resume_descriptor(
    descriptor: int,
    *,
    expected_context: DatasetEvidenceRunContext,
    selected_records: tuple[InteractionRecord, ...],
    invariant_suite: DatasetInvariantSuite | None,
) -> DatasetResumeEvidence:
    size = os.fstat(descriptor).st_size
    if size > _MAXIMUM_EVIDENCE_BYTES:
        raise ValueError("resume evidence exceeds the 128 MB limit")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, min(65_536, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    raw_evidence = b"".join(chunks)
    if raw_evidence and not raw_evidence.endswith(b"\n"):
        raise ValueError("resume evidence must end with a newline")
    return validate_dataset_resume_evidence(
        raw_evidence,
        expected_context=expected_context,
        selected_records=selected_records,
        invariant_suite=invariant_suite,
        evidence_projector=_customer_evidence_record,
    )


async def _evaluate_interaction_records(
    records: tuple[InteractionRecord, ...],
    operator_ids: tuple[str, ...],
    settings: DatasetSemanticSettings,
    target: JsonHttpSandboxConnection,
    output_stream: TextIO,
    *,
    repetitions: int,
    max_sandbox_api_calls: int,
    planned_target_calls: int,
    run_context: DatasetEvidenceRunContext | None = None,
    augmentation_ledger: DatasetAugmentationLedger | None = None,
    saved_augmentations: dict[str, DatasetAugmentationResult] | None = None,
    invariant_suite: DatasetInvariantSuite | None = None,
    invariant_evaluations: list[DatasetInvariantEvaluation] | None = None,
    redaction_engine: RedactionEngine | None = None,
    allow_network_egress: bool = True,
) -> tuple[DatasetEvaluationResult, ...]:
    results: list[DatasetEvaluationResult] = []
    async with create_semantic_model_deconstructor(settings) as deconstructor, target:
        semantic_pipeline = (
            RedactedSemanticPipeline(deconstructor, redaction_engine)
            if redaction_engine is not None
            else deconstructor
        )
        evaluation_target = (
            semantic_pipeline.wrap_sandbox(target)
            if isinstance(semantic_pipeline, RedactedSemanticPipeline)
            else target
        )
        runner = DatasetEvaluationRunner(
            DatasetAugmentationEngine(
                semantic_pipeline,
                semantic_pipeline,
                semantic_pipeline,
            ),
            semantic_pipeline,
            evaluation_target,
            allow_network_egress=allow_network_egress,
        )
        for record in records:
            precomputed_augmentation = (
                saved_augmentations.get(record.id) if saved_augmentations is not None else None
            )

            def checkpoint_augmentation(
                augmentation: DatasetAugmentationResult,
                source: InteractionRecord = record,
            ) -> None:
                if augmentation_ledger is None:
                    return
                augmentation_ledger.append(source=source, augmentation=augmentation)

            result = await runner.run(
                record,
                operator_ids=operator_ids,
                repetitions=repetitions,
                precomputed_augmentation=precomputed_augmentation,
                augmentation_checkpoint_callback=(
                    checkpoint_augmentation if augmentation_ledger is not None else None
                ),
            )
            invariant_evaluation = (
                evaluate_dataset_invariants(result, invariant_suite)
                if invariant_suite is not None
                else None
            )
            if invariant_evaluation is not None and invariant_evaluations is not None:
                invariant_evaluations.append(invariant_evaluation)
            output_stream.write(
                json.dumps(
                    _customer_evidence_record(
                        result,
                        repetitions=repetitions,
                        max_sandbox_api_calls=max_sandbox_api_calls,
                        planned_target_calls=planned_target_calls,
                        run_context=run_context,
                        invariant_evaluation=invariant_evaluation,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
            output_stream.flush()
            os.fsync(output_stream.fileno())
            results.append(result)
    return tuple(results)


def _print_dataset_results(
    results: tuple[DatasetEvaluationResult, ...],
    output: Path,
    *,
    augmentations_output: Path | None = None,
    invariant_evaluations: tuple[DatasetInvariantEvaluation, ...] = (),
) -> None:
    table = Table(title="Dataset evaluation")
    table.add_column("Case", style="cyan")
    table.add_column("Augmentation")
    table.add_column("Status")
    table.add_column("Stability")
    table.add_column("Trials / outcome groups")
    table.add_column("Finding")
    case_number = 0
    for result in results:
        case_number += 1
        table.add_row(
            str(case_number),
            "original replay",
            _baseline_customer_status(result),
            result.baseline.trial_set.stability,
            _trial_set_summary(result.baseline.trial_set),
            "—",
        )
        for case in result.cases:
            case_number += 1
            table.add_row(
                str(case_number),
                case.candidate.operator_id,
                _case_customer_status(result, case),
                case.trial_set.stability if case.trial_set is not None else "—",
                _trial_set_summary(case.trial_set),
                ", ".join(_FINDING_LABELS[finding.category] for finding in case.findings) or "—",
            )
    console.print(table)
    if invariant_evaluations:
        _print_invariant_results(invariant_evaluations)
    console.print(f"Complete evidence: {output}")
    if augmentations_output is not None:
        _print_dataset_plain(f"Saved augmentations: {augmentations_output}")
    console.print(f"Next: ul dataset report {output}")


def _result_needs_review(result: DatasetEvaluationResult) -> bool:
    return any(case.verdict == "divergence_needs_review" for case in result.cases)


def _invariant_exit_code(
    evaluations: tuple[DatasetInvariantEvaluation, ...],
) -> int:
    rules = tuple(
        rule
        for evaluation in evaluations
        for arm in (evaluation.baseline, *evaluation.variations)
        for rule in arm.rules
    )
    if any(rule.status == "violated" for rule in rules):
        return 1
    if any(rule.status == "not_evaluable" for rule in rules):
        return 2
    return 0


def _print_invariant_results(
    evaluations: tuple[DatasetInvariantEvaluation, ...],
) -> None:
    _print_dataset_plain("")
    _print_dataset_plain("Customer invariant evaluation")
    _print_dataset_plain(
        "Selected values remain in the private evidence file; terminal output shows pointers only."
    )
    for evaluation in evaluations:
        _print_dataset_plain(f"Interaction: {evaluation.interaction_id}")
        _print_dataset_plain(f"Declared observation authority: {evaluation.observation_authority}")
        for arm in (evaluation.baseline, *evaluation.variations):
            arm_name = "original" if arm.arm == "baseline" else f"variation ({arm.operator_id})"
            for rule in arm.rules:
                status_counts = {
                    status: sum(trial.status == status for trial in rule.trials)
                    for status in ("satisfied", "violated", "not_evaluable")
                }
                _print_dataset_plain(
                    f"Rule {rule.rule_id} ({rule.rule_version}); severity={rule.severity}; "
                    f"arm={arm_name}; status={rule.status}; reason={rule.reason_code}; trials="
                    + ", ".join(f"{status}={count}" for status, count in status_counts.items())
                )
                _print_dataset_plain(f"Description: {rule.description}")
                if rule.status == "violated":
                    _print_dataset_plain(
                        "Customer rule violated against declared "
                        f"{evaluation.observation_authority}."
                    )
                for trial in rule.trials:
                    _print_dataset_plain(
                        f"Trial {trial.repetition}: {trial.status}; "
                        f"{_invariant_trial_location(trial)}; "
                        f"reason={trial.reason_code}"
                    )


def _invariant_trial_location(
    trial: DatasetInvariantTrialEvaluation
    | DatasetInvariantValueEqualsTrialEvaluation
    | DatasetInvariantValueInSetTrialEvaluation
    | DatasetInvariantArrayUniqueTrialEvaluation,
) -> str:
    if isinstance(trial, DatasetInvariantTrialEvaluation):
        return f"left={trial.left_pointer}; right={trial.right_pointer}"
    if isinstance(
        trial,
        (DatasetInvariantValueEqualsTrialEvaluation, DatasetInvariantValueInSetTrialEvaluation),
    ):
        return f"value={trial.value_pointer}"
    location = (
        f"array={trial.array_pointer}; keys={','.join(trial.key_pointers)}; "
        f"items={trial.item_count}"
    )
    if trial.duplicate_indices:
        location += f"; duplicate_indices={trial.duplicate_indices}"
    if trial.failed_item_index is not None:
        location += (
            f"; failed_item={trial.failed_item_index}; failed_key={trial.failed_key_pointer}"
        )
    return location


def _print_dataset_plain(message: str) -> None:
    safe_message = "".join(
        character
        if (ord(character) >= 32 and not 0x7F <= ord(character) <= 0x9F)
        and unicodedata.category(character) not in {"Cf", "Cs"}
        else f"\\u{ord(character):04x}"
        for character in message
    )
    console.print(safe_message, markup=False, highlight=False, soft_wrap=True)


def _customer_evidence_record(
    result: DatasetEvaluationResult,
    *,
    repetitions: int,
    max_sandbox_api_calls: int,
    planned_target_calls: int,
    run_context: DatasetEvidenceRunContext | None = None,
    invariant_evaluation: DatasetInvariantEvaluation | None = None,
) -> dict[str, JsonValue]:
    cases: list[JsonValue] = []
    for case in result.cases:
        cases.append(
            {
                "operator_id": case.candidate.operator_id,
                "operator_version": case.candidate.operator_version,
                "augmented_input": case.candidate.augmented_input,
                "status": _case_customer_status(result, case),
                "variation_accepted": case.candidate.passed,
                "variation_rejection_reasons": list(case.candidate.failure_reasons),
                "observations": _customer_trial_set(case.trial_set),
                "findings": _customer_findings(
                    case.findings,
                    interaction_id=result.source.id,
                    original_input=result.source.raw_input,
                    operator_id=case.candidate.operator_id,
                    operator_version=case.candidate.operator_version,
                    augmented_input=case.candidate.augmented_input,
                ),
                "inconclusive_reasons": list(case.inconclusive_reasons),
            }
        )
    uses_extended_invariants = invariant_evaluation is not None and any(
        rule.rule_type != "json_values_equal"
        for arm in (invariant_evaluation.baseline, *invariant_evaluation.variations)
        for rule in arm.rules
    )
    if uses_extended_invariants and run_context is None:
        raise ValueError("extended invariant evidence requires a resumable run context")
    if uses_extended_invariants:
        evidence_schema_version = "1.6.0"
    elif run_context is not None:
        evidence_schema_version = "1.5.0"
    else:
        evidence_schema_version = "1.4.0"
    evidence: dict[str, JsonValue] = {
        "schema_version": evidence_schema_version,
        "interaction_id": result.source.id,
        "original_input": result.source.raw_input,
        "execution_plan": {
            "repetitions": repetitions,
            "max_target_calls": max_sandbox_api_calls,
            "dataset_planned_target_calls": planned_target_calls,
        },
        "limitations": _BEHAVIORAL_LIMITATIONS,
        "current_baseline": {
            "status": _baseline_customer_status(result),
            "observations": _customer_trial_set(result.baseline.trial_set),
            "inconclusive_reasons": list(result.baseline.inconclusive_reasons),
        },
        "cases": cases,
        "invariant_evaluation": cast(
            JsonValue,
            invariant_evaluation.model_dump(mode="json")
            if invariant_evaluation is not None
            else None,
        ),
        "technical_details": cast(JsonValue, result.model_dump(mode="json")),
    }
    if run_context is not None:
        evidence["run_context"] = cast(JsonValue, run_context.model_dump(mode="json"))
    return evidence


def _baseline_customer_status(result: DatasetEvaluationResult) -> str:
    trial_set = result.baseline.trial_set
    stability = trial_set.stability
    if stability == "unstable":
        return "UNSTABLE ORIGINAL — INCONCLUSIVE"
    if stability == "inconclusive":
        return "COULDN'T DETERMINE"
    repetitions = trial_set.requested_repetitions
    return f"ORIGINAL REPLAY STABLE ({repetitions}/{repetitions} OBSERVED)"


def _case_customer_status(result: DatasetEvaluationResult, case: DatasetEvaluationCase) -> str:
    if case.verdict == "augmentation_rejected":
        return _CUSTOMER_STATUSES["augmentation_rejected"]
    trial_set = case.trial_set
    if trial_set is not None and trial_set.stability == "inconclusive":
        return "COULDN'T DETERMINE"
    if result.baseline.trial_set.stability == "inconclusive":
        return "COULDN'T DETERMINE"
    if (
        result.baseline.trial_set.stability == "unstable"
        and trial_set is not None
        and trial_set.stability == "unstable"
    ):
        return "UNSTABLE ORIGINAL AND VARIATION — INCONCLUSIVE"
    if result.baseline.trial_set.stability == "unstable":
        return "UNSTABLE ORIGINAL — INCONCLUSIVE"
    if trial_set is not None and trial_set.stability == "unstable":
        return "UNSTABLE VARIATION — REVIEW"
    if case.verdict == "divergence_needs_review":
        if trial_set is not None and trial_set.requested_repetitions > 1:
            return "REPEATABLE DIFFERENCE — REVIEW"
        return "POTENTIAL DIFFERENCE — REVIEW"
    return _CUSTOMER_STATUSES[case.verdict]


def _trial_set_summary(trial_set: DatasetEvaluationTrialSet | None) -> str:
    if trial_set is None:
        return "—"
    return f"{trial_set.requested_repetitions} / {len(trial_set.outcome_groups)}"


def _customer_trial_set(trial_set: DatasetEvaluationTrialSet | None) -> JsonValue:
    if trial_set is None:
        return None
    trials = [
        {
            "repetition": trial.repetition,
            "status": "inconclusive" if trial.inconclusive_reasons else "observed",
            "inconclusive_reasons": list(trial.inconclusive_reasons),
            "lifecycle_failure": _customer_lifecycle_failure(trial),
        }
        for trial in trial_set.trials
    ]
    outcome_groups = [
        {
            "repetitions": list(group.repetitions),
            "count": len(group.repetitions),
            "representative_effects": [
                effect.model_dump(mode="json") for effect in group.representative_effects
            ],
        }
        for group in trial_set.outcome_groups
    ]
    return cast(
        JsonValue,
        {
            "requested_repetitions": trial_set.requested_repetitions,
            "stability": trial_set.stability,
            "observed_repetitions": sum(trial["status"] == "observed" for trial in trials),
            "inconclusive_repetitions": sum(trial["status"] == "inconclusive" for trial in trials),
            "outcome_group_count": len(outcome_groups),
            "outcome_groups": outcome_groups,
            "trials": trials,
        },
    )


def _customer_lifecycle_failure(trial: object) -> JsonValue:
    lifecycle_failure = getattr(trial, "lifecycle_failure", None)
    if lifecycle_failure is None:
        return None
    return cast(DatasetTargetLifecycleFailure, lifecycle_failure).model_dump(mode="json")


def _customer_findings(
    findings: tuple[DatasetEvaluationFinding, ...],
    *,
    interaction_id: str,
    original_input: str,
    operator_id: str,
    operator_version: str,
    augmented_input: str,
) -> list[JsonValue]:
    base_finding_ids = [
        _finding_id(
            interaction_id=interaction_id,
            original_input=original_input,
            operator_id=operator_id,
            operator_version=operator_version,
            augmented_input=augmented_input,
            finding=finding,
        )
        for finding in findings
    ]
    finding_id_counts: dict[str, int] = {}
    for finding_id in base_finding_ids:
        finding_id_counts[finding_id] = finding_id_counts.get(finding_id, 0) + 1
    finding_id_occurrences: dict[str, int] = {}
    customer_findings: list[JsonValue] = []
    for finding, base_finding_id in zip(findings, base_finding_ids, strict=True):
        duplicate_ordinal = None
        if finding_id_counts[base_finding_id] > 1:
            duplicate_ordinal = finding_id_occurrences.get(base_finding_id, 0) + 1
            finding_id_occurrences[base_finding_id] = duplicate_ordinal
        finding_id = (
            base_finding_id
            if duplicate_ordinal is None
            else _finding_id(
                interaction_id=interaction_id,
                original_input=original_input,
                operator_id=operator_id,
                operator_version=operator_version,
                augmented_input=augmented_input,
                finding=finding,
                duplicate_ordinal=duplicate_ordinal,
            )
        )
        customer_findings.append(
            cast(
                JsonValue,
                {
                    "finding_id": finding_id,
                    "category": finding.category,
                    "grounded_field_names": sorted(finding.grounded_field_names),
                    "severity": "unrated",
                    "review_status": "needs_review",
                    "summary": finding.message,
                    "reference_effects": [
                        effect.model_dump(mode="json") for effect in finding.expected_effects
                    ],
                    "observed_effects": [
                        effect.model_dump(mode="json") for effect in finding.observed_effects
                    ],
                },
            )
        )
    return customer_findings


def _finding_id(
    *,
    interaction_id: str,
    original_input: str,
    operator_id: str,
    operator_version: str,
    augmented_input: str,
    finding: DatasetEvaluationFinding,
    duplicate_ordinal: int | None = None,
) -> str:
    canonical_finding = cast(
        dict[str, JsonValue],
        {
            "interaction_id": interaction_id,
            "original_input": original_input,
            "operator_id": operator_id,
            "operator_version": operator_version,
            "augmented_input": augmented_input,
            "category": finding.category,
            "grounded_field_names": sorted(finding.grounded_field_names),
            "reference_action_semantics": _normalized_action_semantics(finding.expected_effects),
            "observed_action_semantics": _normalized_action_semantics(finding.observed_effects),
        },
    )
    if duplicate_ordinal is not None:
        canonical_finding["duplicate_ordinal"] = duplicate_ordinal
    canonical_json = json.dumps(
        canonical_finding,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"ulf_v1_{hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()}"


def _normalized_action_semantics(effects: tuple[ObservedOutcome, ...]) -> list[JsonValue]:
    signatures = [
        cast(
            JsonValue,
            {
                "kind": effect.kind,
                "predicate": effect.predicate,
                "fields": effect.fields,
                "propositions": sorted(effect.propositions),
            },
        )
        for effect in effects
    ]
    return sorted(
        signatures,
        key=lambda signature: json.dumps(
            signature,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )
