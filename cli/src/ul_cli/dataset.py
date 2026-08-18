from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import stat
import sys
import unicodedata
from pathlib import Path
from typing import Annotated, TextIO, cast

import httpx
import typer
from pydantic import JsonValue, ValidationError
from rich.console import Console
from rich.table import Table
from ul import (
    DatasetAugmentationEngine,
    DatasetEvaluationCase,
    DatasetEvaluationFinding,
    DatasetEvaluationResult,
    DatasetEvaluationRunner,
    DatasetEvaluationTrialSet,
    InteractionRecord,
    OpenAICompatibleDatasetSettings,
    OpenAICompatibleSemanticDeconstructor,
    OpenRouterDatasetSettings,
    OpenRouterSemanticDeconstructor,
    builtin_dataset_augmentation_operators,
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
from ul.deconstruction import (
    DatasetSemanticProviderSelection,
    DatasetSemanticSettings,
    SemanticModelDeconstructor,
)
from ul.http_target import (
    JsonHttpDatasetTarget,
    JsonHttpDatasetTargetConfig,
    load_json_http_dataset_target_config,
    validate_json_http_dataset_target_configuration,
)
from ul_core.dataset import ObservedOutcome

from ul_cli.dataset_ingest import app as ingest_app
from ul_cli.dataset_review import (
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
_DEFAULT_MAXIMUM_TARGET_CALLS = 100
_HEADER_NAME_PATTERN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
_ENVIRONMENT_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DATASET_OPERATORS = builtin_dataset_augmentation_operators()
_DATASET_OPERATOR_IDS = tuple(operator.id for operator in _DATASET_OPERATORS)
_DATASET_OPERATORS_BY_ID = {operator.id: operator for operator in _DATASET_OPERATORS}


def _load_dataset_semantic_settings() -> DatasetSemanticSettings:
    selection = DatasetSemanticProviderSelection()
    if selection.provider == "openai-compatible":
        try:
            return OpenAICompatibleDatasetSettings()
        except ValidationError:
            raise ValueError(
                "OpenAI-compatible semantic provider configuration is invalid"
            ) from None
    return OpenRouterDatasetSettings()


def _create_semantic_model_deconstructor(
    settings: DatasetSemanticSettings,
) -> SemanticModelDeconstructor:
    if isinstance(settings, OpenAICompatibleDatasetSettings):
        return OpenAICompatibleSemanticDeconstructor(settings)
    return OpenRouterSemanticDeconstructor(settings)


def _semantic_provider_id(settings: object) -> str:
    return cast(str, getattr(settings, "semantic_provider_id", "openrouter"))


def _semantic_base_url(settings: object) -> str:
    return cast(
        str,
        getattr(settings, "semantic_base_url", "https://openrouter.ai/api/v1"),
    )


def _semantic_api_key_environment_variable(settings: object) -> str:
    return cast(
        str,
        getattr(settings, "api_key_environment_variable", "OPEN_ROUTER_API_KEY"),
    )


def _semantic_api_key_is_required(settings: object) -> bool:
    return not isinstance(settings, OpenAICompatibleDatasetSettings)


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
def initialize_dataset_target(
    target_config: Annotated[
        Path,
        typer.Argument(
            dir_okay=False,
            help="New JSON file describing the target request and response shape.",
        ),
    ],
    url: Annotated[
        str,
        typer.Option(help="Sandbox HTTP(S) endpoint that UL will evaluate."),
    ],
) -> None:
    """Create a private starter configuration for a JSON HTTP target.

    The generated template contains one complete {{input}} JSON value. A response
    JSON Pointer such as /choices/0/message/content selects the observable result.
    """
    try:
        config = JsonHttpDatasetTargetConfig(
            version=1,
            url=url,
            headers_from_env={},
            request_json_template={"input": "{{input}}"},
            response_json_pointer="",
        )
        output_stream = _create_private_output(target_config)
    except (OSError, ValidationError, ValueError) as error:
        if isinstance(error, FileExistsError):
            message = "target config already exists; UL will not overwrite it"
        elif isinstance(error, OSError):
            message = f"cannot create target config ({error.__class__.__name__})"
        elif isinstance(error, ValidationError):
            message = "target config is invalid"
        else:
            message = str(error)
        raise typer.BadParameter(message, param_hint="TARGET_CONFIG") from None

    with output_stream:
        json.dump(config.model_dump(mode="json"), output_stream, indent=2)
        output_stream.write("\n")

    console.print(f"Created private target config: {target_config}")
    console.print(
        "Next: adjust request_json_template and response_json_pointer, add any "
        "headers_from_env, then run 'ul dataset evaluate DATASET --target-config "
        f"{target_config} --dry-run'."
    )
    console.print(
        "Keep exactly one complete {{input}} value. headers_from_env maps HTTP header names "
        "to environment-variable names; secret values stay outside this file."
    )


@app.command("operators")
def list_dataset_operators() -> None:
    """List realistic-user transformations available for dataset evaluation."""
    console.print("Dataset augmentation operators")
    for operator in builtin_dataset_augmentation_operators():
        review_note = "; human review required" if operator.human_review_required else ""
        console.print(
            f"- {operator.id} ({operator.version}): "
            f"{operator.allowed_change.replace('_', ' ')}{review_note}"
        )


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
    target_url: Annotated[
        str | None,
        typer.Option(
            help="Simple sandbox endpoint: POST one JSON string field and return non-null JSON."
        ),
    ] = None,
    target_config: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSON target configuration created by 'ul dataset init'.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="New JSONL file for complete local evidence."),
    ] = None,
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
            help="Augmentation operator. Run 'ul dataset operators' for values; repeat as needed.",
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
            help="Fresh-state target executions per original input and accepted variation.",
        ),
    ] = 3,
    max_target_calls: Annotated[
        int,
        typer.Option(
            min=1,
            help="Maximum target requests authorized for this evaluation.",
        ),
    ] = _DEFAULT_MAXIMUM_TARGET_CALLS,
    request_field: Annotated[
        str | None,
        typer.Option(help="JSON request field that receives the augmented input."),
    ] = None,
    header_env: Annotated[
        list[str] | None,
        typer.Option(
            "--header-env",
            help="HTTP header and environment variable as HEADER=ENV. Repeat as needed.",
        ),
    ] = None,
    allow_target_network: Annotated[
        bool,
        typer.Option(help="Allow requests to the configured sandbox endpoint."),
    ] = False,
    confirm_isolated_sandbox: Annotated[
        bool,
        typer.Option(help="Confirm the target cannot cause real business effects."),
    ] = False,
    confirm_fresh_state: Annotated[
        bool,
        typer.Option(help="Confirm every target request starts from the same clean state."),
    ] = False,
    allow_insecure_http: Annotated[
        bool,
        typer.Option(help="Allow an HTTP target. Intended for local sandboxes."),
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
) -> None:
    """Explore behavioral differences against an isolated black-box agent.

    UL_LIVE=true enables billed semantic-model calls and external processing together.
    UL_DATASET_LIVE_CALLS and UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING remain separate,
    higher-precedence controls. OpenRouter remains the default; set
    UL_DATASET_SEMANTIC_PROVIDER=openai-compatible for a customer-controlled endpoint.

    Example: ul dataset evaluate interactions.jsonl --target-url https://sandbox/run
    --allow-target-network --confirm-isolated-sandbox --confirm-fresh-state
    --output results.jsonl
    """
    if resume is not None:
        if output is not None and output.resolve() != resume.resolve():
            raise typer.BadParameter(
                "--output must point to the same file as --resume, or be omitted",
                param_hint="--output",
            )
        if output is None:
            output = resume
    try:
        records = _load_interaction_records(data)
        selected_operators = _validate_operator_ids(operator)
        _validate_target_mode_options(
            target_url=target_url,
            target_config=target_config,
            request_field=request_field,
            header_env=header_env,
        )
        header_environment_variables = _parse_header_environment_variables(header_env)
        invariant_suite = (
            load_dataset_invariant_suite(invariants) if invariants is not None else None
        )
        selected_records = records[:limit]
        if resume is None:
            initial_target_calls = (
                len(selected_records) * repetitions * (1 + len(selected_operators))
            )
            if initial_target_calls > max_target_calls:
                raise typer.BadParameter(
                    f"selection would make up to {initial_target_calls} target calls, exceeding "
                    f"--max-target-calls {max_target_calls}; reduce --limit, --operator, or "
                    "--repetitions, or explicitly raise the call budget"
                )
        if not dry_run and resume is None:
            if target_url is None and target_config is None:
                raise typer.BadParameter(
                    "execution requires --target-url or --target-config",
                    param_hint="--target-url",
                )
            if not allow_target_network:
                raise typer.BadParameter(
                    "execution requires --allow-target-network",
                    param_hint="--allow-target-network",
                )
            if not confirm_isolated_sandbox:
                raise typer.BadParameter(
                    "execution requires --confirm-isolated-sandbox",
                    param_hint="--confirm-isolated-sandbox",
                )
            if not confirm_fresh_state:
                raise typer.BadParameter(
                    "execution requires --confirm-fresh-state",
                    param_hint="--confirm-fresh-state",
                )
            if output is None:
                raise typer.BadParameter("execution requires --output", param_hint="--output")
            if output.exists():
                raise typer.BadParameter(
                    "output already exists; UL will not overwrite it",
                    param_hint="--output",
                )
        settings = _load_dataset_semantic_settings()
        _validate_model_input_bounds(selected_records, settings.max_input_chars)
        loaded_target_config = (
            load_json_http_dataset_target_config(target_config)
            if target_config is not None
            else None
        )
        if target_url is not None:
            validate_json_http_dataset_target_configuration(
                target_url,
                sandbox_confirmed=confirm_isolated_sandbox or dry_run or resume is not None,
                fresh_state_confirmed=confirm_fresh_state or dry_run or resume is not None,
                request_field=request_field or "input",
                header_environment_variables=header_environment_variables,
                allow_insecure_http=allow_insecure_http,
            )
        if loaded_target_config is not None:
            validate_json_http_dataset_target_configuration(
                loaded_target_config.url,
                sandbox_confirmed=confirm_isolated_sandbox or dry_run or resume is not None,
                fresh_state_confirmed=confirm_fresh_state or dry_run or resume is not None,
                header_environment_variables=loaded_target_config.headers_from_env,
                request_json_template=loaded_target_config.request_json_template,
                response_json_pointer=loaded_target_config.response_json_pointer,
                allow_insecure_http=allow_insecure_http,
            )
        normalized_target_config = _normalized_target_config(
            target_url=target_url,
            loaded_target_config=loaded_target_config,
            request_field=request_field,
            header_environment_variables=header_environment_variables,
        )
        if resume is not None and normalized_target_config is None:
            raise ValueError("--resume requires --target-url or --target-config")
        run_context = (
            _dataset_evidence_run_context(
                selected_records=selected_records,
                selected_operator_ids=selected_operators,
                repetitions=repetitions,
                invariant_suite=invariant_suite,
                target_config=normalized_target_config,
                settings=settings,
            )
            if normalized_target_config is not None
            else None
        )
    except (_DatasetInputError, ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from None

    resume_evidence: DatasetResumeEvidence | None = None
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

    potential_target_calls = len(selected_records) * repetitions * (1 + len(selected_operators))
    if potential_target_calls > max_target_calls:
        raise typer.BadParameter(
            f"remaining selection would make up to {potential_target_calls} target calls, "
            f"exceeding --max-target-calls {max_target_calls}; reduce --limit, --operator, "
            "or --repetitions, or explicitly raise the call budget"
        )

    if dry_run:
        _print_dataset_plan(
            record_count=len(records),
            selected_count=len(selected_records),
            skipped_count=skipped_count,
            operator_ids=selected_operators,
            target_configured=target_url is not None or target_config is not None,
            target_endpoint=(
                loaded_target_config.url if loaded_target_config is not None else target_url
            ),
            target_header_environment_variables=(
                loaded_target_config.headers_from_env
                if loaded_target_config is not None
                else header_environment_variables
            ),
            repetitions=repetitions,
            max_target_calls=max_target_calls,
            invariant_suite=invariant_suite,
            output=output,
            semantic_provider_id=_semantic_provider_id(settings),
            semantic_base_url=_semantic_base_url(settings),
        )
        return

    if not selected_records and skipped_count > 0:
        assert output is not None
        assert resume_evidence is not None
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

    if target_url is None and target_config is None:
        raise typer.BadParameter(
            "execution requires --target-url or --target-config",
            param_hint="--target-url",
        )
    if not allow_target_network:
        raise typer.BadParameter(
            "execution requires --allow-target-network",
            param_hint="--allow-target-network",
        )
    if not confirm_isolated_sandbox:
        raise typer.BadParameter(
            "execution requires --confirm-isolated-sandbox",
            param_hint="--confirm-isolated-sandbox",
        )
    if not confirm_fresh_state:
        raise typer.BadParameter(
            "execution requires --confirm-fresh-state",
            param_hint="--confirm-fresh-state",
        )
    if output is None:
        raise typer.BadParameter("execution requires --output", param_hint="--output")
    if output.exists() and resume is None:
        raise typer.BadParameter(
            "output already exists; UL will not overwrite it",
            param_hint="--output",
        )

    assert target_url is not None or loaded_target_config is not None
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
    if _semantic_api_key_is_required(settings) and (
        settings.api_key is None or not settings.api_key.get_secret_value().strip()
    ):
        raise typer.BadParameter(
            f"set {_semantic_api_key_environment_variable(settings)} to run an evaluation"
        )

    try:
        if loaded_target_config is not None:
            target = JsonHttpDatasetTarget.from_config(
                loaded_target_config,
                sandbox_confirmed=True,
                fresh_state_confirmed=True,
                allow_insecure_http=allow_insecure_http,
            )
        else:
            assert target_url is not None
            target = JsonHttpDatasetTarget(
                target_url,
                sandbox_confirmed=True,
                fresh_state_confirmed=True,
                request_field=request_field or "input",
                header_environment_variables=header_environment_variables,
                allow_insecure_http=allow_insecure_http,
            )
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--target-url") from None

    try:
        if resume is None:
            output_stream = _create_private_output(output)
        else:
            assert resume_evidence is not None
            output_stream, locked_resume_evidence = _open_resume_output(
                output,
                expected_context=run_context,
                selected_records=tuple(records[:limit]),
                invariant_suite=invariant_suite,
            )
            if locked_resume_evidence != resume_evidence:
                output_stream.close()
                raise ValueError("resume evidence changed after preflight")
    except (OSError, ValueError) as error:
        asyncio.run(target.aclose())
        message = str(error) if isinstance(error, ValueError) else error.__class__.__name__
        raise typer.BadParameter(
            f"cannot safely open output file ({message})",
            param_hint="--resume" if resume is not None else "--output",
        ) from None

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
                    max_target_calls=max_target_calls,
                    planned_target_calls=(
                        (len(selected_records) + skipped_count)
                        * repetitions
                        * (1 + len(selected_operators))
                    ),
                    run_context=run_context,
                    invariant_suite=invariant_suite,
                    invariant_evaluations=invariant_evaluations,
                )
            else:
                evaluation_coroutine = _evaluate_interaction_records(
                    selected_records,
                    selected_operators,
                    settings,
                    target,
                    output_stream,
                    repetitions=repetitions,
                    max_target_calls=max_target_calls,
                    planned_target_calls=(
                        (len(selected_records) + skipped_count)
                        * repetitions
                        * (1 + len(selected_operators))
                    ),
                    run_context=run_context,
                )
            results = asyncio.run(evaluation_coroutine)
            for result in results:
                has_review_findings |= _result_needs_review(result)
    except (TimeoutError, RuntimeError, ValueError, httpx.HTTPError) as error:
        console.print(
            f"Evaluation stopped ({error.__class__.__name__}). "
            f"Complete results written before the error remain in {output}."
        )
        raise typer.Exit(code=2) from None

    if skipped_count > 0:
        console.print(
            f"Resumed: {skipped_count} interaction(s) skipped (already in evidence), "
            f"{len(results)} newly evaluated."
        )
    _print_dataset_results(results, output, invariant_evaluations=tuple(invariant_evaluations))
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
    selected_ids = tuple(operator_ids or ["surface.rephrase"])
    known_ids = set(_DATASET_OPERATOR_IDS)
    unknown_ids = sorted(set(selected_ids) - known_ids)
    if unknown_ids:
        raise _DatasetInputError(f"unknown operator(s): {', '.join(unknown_ids)}")
    if len(selected_ids) != len(set(selected_ids)):
        raise _DatasetInputError("duplicate --operator values are not allowed")
    return selected_ids


def _parse_header_environment_variables(
    header_options: list[str] | None,
) -> dict[str, str]:
    parsed_headers: dict[str, str] = {}
    normalized_names: set[str] = set()
    for option in header_options or []:
        header_name, separator, environment_name = option.partition("=")
        if (
            separator != "="
            or _HEADER_NAME_PATTERN.fullmatch(header_name) is None
            or _ENVIRONMENT_NAME_PATTERN.fullmatch(environment_name) is None
        ):
            raise _DatasetInputError("--header-env must use valid HEADER=ENV names")
        normalized_name = header_name.casefold()
        if normalized_name in normalized_names:
            raise _DatasetInputError(f"duplicate header name: {header_name}")
        normalized_names.add(normalized_name)
        parsed_headers[header_name] = environment_name
    return parsed_headers


def _validate_target_mode_options(
    *,
    target_url: str | None,
    target_config: Path | None,
    request_field: str | None,
    header_env: list[str] | None,
) -> None:
    if target_config is None:
        return
    conflicting_options: list[str] = []
    if target_url is not None:
        conflicting_options.append("--target-url")
    if request_field is not None:
        conflicting_options.append("--request-field")
    if header_env:
        conflicting_options.append("--header-env")
    if conflicting_options:
        raise _DatasetInputError(
            f"--target-config cannot be combined with {', '.join(conflicting_options)}"
        )


def _normalized_target_config(
    *,
    target_url: str | None,
    loaded_target_config: JsonHttpDatasetTargetConfig | None,
    request_field: str | None,
    header_environment_variables: dict[str, str],
) -> JsonHttpDatasetTargetConfig | None:
    if loaded_target_config is not None:
        return loaded_target_config
    if target_url is None:
        return None
    return JsonHttpDatasetTargetConfig(
        version=1,
        url=target_url,
        headers_from_env=header_environment_variables,
        request_json_template={(request_field or "input"): "{{input}}"},
        response_json_pointer="",
    )


def _dataset_evidence_run_context(
    *,
    selected_records: tuple[InteractionRecord, ...],
    selected_operator_ids: tuple[str, ...],
    repetitions: int,
    invariant_suite: DatasetInvariantSuite | None,
    target_config: JsonHttpDatasetTargetConfig,
    settings: DatasetSemanticSettings,
) -> DatasetEvidenceRunContext:
    return create_dataset_evidence_run_context(
        selected_records=selected_records,
        operators=tuple(
            (operator_id, _DATASET_OPERATORS_BY_ID[operator_id].version)
            for operator_id in selected_operator_ids
        ),
        repetitions=repetitions,
        invariant_suite_sha256=(invariant_suite.sha256 if invariant_suite is not None else None),
        target_config=target_config,
        semantic_settings=DatasetEvidenceSemanticSettings(
            provider=_semantic_provider_id(settings),
            base_url=_semantic_base_url(settings),
            model=settings.model,
            render_model=settings.render_model,
            equivalence_model=settings.equivalence_model,
            max_input_chars=settings.max_input_chars,
            max_output_tokens=settings.max_output_tokens,
            max_render_tokens=settings.max_render_tokens,
            max_response_bytes=settings.max_response_bytes,
            timeout_seconds=settings.timeout_seconds,
        ),
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
    max_target_calls: int,
    invariant_suite: DatasetInvariantSuite | None,
    output: Path | None,
    semantic_provider_id: str,
    semantic_base_url: str,
) -> None:
    potential_target_calls = selected_count * repetitions * (1 + len(operator_ids))
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
        console.print("Additional target calls for customer invariants: 0")
    console.print(f"Potential semantic model calls: up to {potential_model_calls}")
    console.print(f"Semantic provider: {semantic_provider_id} ({semantic_base_url})")
    console.print(
        f"Potential target calls: up to {potential_target_calls} "
        f"(authorized maximum: {max_target_calls})"
    )
    console.print(f"Target: {'configured' if target_configured else 'not configured'}")
    if output is not None:
        console.print(f"Evidence destination: {output}")
    if target_endpoint is not None:
        console.print(f"Target endpoint: {target_endpoint}")
        if target_header_environment_variables:
            mappings = ", ".join(
                f"{header_name}={environment_variable}"
                for header_name, environment_variable in sorted(
                    target_header_environment_variables.items()
                )
            )
            console.print(f"Target header environment mappings: {mappings}")
        else:
            console.print("Target header environment mappings: none")
    console.print(
        "Semantic models receive historical inputs and outputs, generated variations, "
        "live control responses, and variation responses on execution."
    )
    console.print(
        "Every target request must start from the same clean state. The target receives each "
        "selected original input and each accepted variation for every repetition."
    )
    console.print(
        "Target requests and semantic model calls may be billed separately. Repetitions only "
        "show observed behavioral consistency: they do not determine correctness, identify "
        "causality, or estimate a production failure rate."
    )
    console.print("No model or target requests sent.")


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
    target: JsonHttpDatasetTarget,
    output_stream: TextIO,
    *,
    repetitions: int,
    max_target_calls: int,
    planned_target_calls: int,
    run_context: DatasetEvidenceRunContext | None = None,
    invariant_suite: DatasetInvariantSuite | None = None,
    invariant_evaluations: list[DatasetInvariantEvaluation] | None = None,
) -> tuple[DatasetEvaluationResult, ...]:
    results: list[DatasetEvaluationResult] = []
    async with _create_semantic_model_deconstructor(settings) as deconstructor, target:
        runner = DatasetEvaluationRunner(
            DatasetAugmentationEngine(deconstructor, deconstructor, deconstructor),
            deconstructor,
            target,
            allow_network_egress=True,
        )
        for record in records:
            result = await runner.run(
                record,
                operator_ids=operator_ids,
                repetitions=repetitions,
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
                        max_target_calls=max_target_calls,
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
    console.print(safe_message, markup=False, highlight=False)


def _customer_evidence_record(
    result: DatasetEvaluationResult,
    *,
    repetitions: int,
    max_target_calls: int,
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
            "max_target_calls": max_target_calls,
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
