from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer
from pydantic import ValidationError
from ul import DatasetEvaluationMode

from .request import DatasetEvaluationRequest, DatasetRequestError
from .workflow import run_dataset_evaluation

_MAXIMUM_DATASET_RECORDS = 100
_MAXIMUM_TARGET_TIMEOUT_SECONDS = 3_600.0


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
    augmentations_input: Annotated[
        Path | None,
        typer.Option(
            "--augmentations-input",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help=(
                "Reuse a complete private augmentation JSONL from an earlier UL campaign "
                "without regenerating candidates."
            ),
        ),
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
    concurrency: Annotated[
        int | None,
        typer.Option(
            "--concurrency",
            min=1,
            max=100,
            help="Maximum isolated target requests in flight. Defaults to 1.",
        ),
    ] = None,
    target_timeout_seconds: Annotated[
        float | None,
        typer.Option(
            "--target-timeout-seconds",
            min=0,
            max=_MAXIMUM_TARGET_TIMEOUT_SECONDS,
            help="Maximum seconds allowed for each target trial.",
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
    Augmentation reuse: --augmentations-input PATH
    Augmentation retention: --augmentations-output PATH or --no-save-augmentations
    """
    raw_request = DatasetEvaluationRequest(
        data=data,
        environment_config=environment_config,
        target=target,
        target_artifacts=tuple(target_artifact or ()),
        http_preset=http_preset,
        request_json_template=request_json_template,
        response_json_pointer=response_json_pointer,
        agent_model=agent_model,
        headers_from_env=tuple(header_from_env or ()),
        confirm_target=confirm_target,
        output=output,
        augmentations_input=augmentations_input,
        augmentations_output=augmentations_output,
        no_save_augmentations=no_save_augmentations,
        invariants=invariants,
        evaluation_mode=evaluation_mode,
        operators=tuple(operator) if operator is not None else None,
        limit=limit,
        repetitions=repetitions,
        concurrency=concurrency,
        target_timeout_seconds=target_timeout_seconds,
        max_environment_api_calls=max_environment_api_calls,
        allow_environment_network=allow_environment_network,
        confirm_test_environment=confirm_test_environment,
        confirm_request_isolation=confirm_request_isolation,
        confirm_safe_test_target=confirm_safe_test_target,
        allow_insecure_http=allow_insecure_http,
        dry_run=dry_run,
        json_output=json_output,
        progress_json=progress_json,
        show_sensitive_values=show_sensitive_values,
        resume=resume,
        resolve_quarantine_after=resolve_quarantine_after,
        redaction_policy=redaction_policy,
        redaction_state=redaction_state,
        expected_environment_origin=expected_environment_origin,
        expected_environment_config_sha256=expected_environment_config_sha256,
        expected_redaction_policy_sha256=expected_redaction_policy_sha256,
        show_report_guidance=show_report_guidance,
    )
    try:
        run_dataset_evaluation(raw_request)
    except typer.Exit:
        raise
    except DatasetRequestError as error:
        raise typer.BadParameter(str(error), param_hint=error.parameter) from None
    except (ValidationError, ValueError) as error:
        raise typer.BadParameter(str(error)) from None
