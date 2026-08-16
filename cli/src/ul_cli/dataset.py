from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Annotated, TextIO, cast

import httpx
import typer
from pydantic import JsonValue, ValidationError
from rich.console import Console
from rich.table import Table
from ul import (
    DatasetAugmentationEngine,
    DatasetEvaluationFinding,
    DatasetEvaluationResult,
    DatasetEvaluationRunner,
    InteractionRecord,
    OpenRouterDatasetSettings,
    OpenRouterSemanticDeconstructor,
    builtin_dataset_augmentation_operators,
)
from ul.http_target import (
    JsonHttpDatasetTarget,
    validate_json_http_dataset_target_configuration,
)

app = typer.Typer(help="Evaluate successful agent interactions.")
console = Console()

_MAXIMUM_DATASET_BYTES = 10_000_000
_MAXIMUM_DATASET_RECORDS = 100
_MAXIMUM_TARGET_CALLS = 100
_HEADER_NAME_PATTERN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
_ENVIRONMENT_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DATASET_OPERATOR_IDS = tuple(operator.id for operator in builtin_dataset_augmentation_operators())
_CUSTOMER_STATUSES = {
    "augmentation_rejected": "VARIATION DISCARDED",
    "inconclusive": "COULDN'T DETERMINE",
    "no_divergence": "NO OBSERVED DIFFERENCE",
    "divergence_needs_review": "DIFFERENCE — REVIEW",
}
_BASELINE_STATUSES = {
    "inconclusive": "COULDN'T DETERMINE",
    "no_divergence": "LIVE CONTROL MATCHES STORED RUN",
    "divergence_needs_review": "LIVE CONTROL DIFFERS — REVIEW",
}
_FINDING_LABELS = {
    "duplicate_effect": "duplicate action",
    "unexpected_effect": "unexpected action",
    "missing_effect": "missing action",
    "changed_grounded_effect_argument": "changed action value",
}


class _DatasetInputError(ValueError):
    pass


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
    output: Annotated[
        Path | None,
        typer.Option(help="New JSONL file for complete local evidence."),
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
    ] = 25,
    request_field: Annotated[
        str,
        typer.Option(help="JSON request field that receives the augmented input."),
    ] = "input",
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
) -> None:
    """Evaluate successful interactions against an isolated black-box agent.

    Execution requires UL_DATASET_LIVE_CALLS=true,
    UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING=true, and OPEN_ROUTER_API_KEY.

    Example: ul dataset evaluate interactions.jsonl --target-url https://sandbox/run
    --allow-target-network --confirm-isolated-sandbox --confirm-fresh-state
    --output results.jsonl
    """
    try:
        records = _load_interaction_records(data)
        selected_operators = _validate_operator_ids(operator)
        header_environment_variables = _parse_header_environment_variables(header_env)
    except _DatasetInputError as error:
        raise typer.BadParameter(str(error)) from None

    selected_records = records[:limit]
    potential_target_calls = len(selected_records) * (1 + len(selected_operators))
    if potential_target_calls > _MAXIMUM_TARGET_CALLS:
        raise typer.BadParameter(
            f"selection would make {potential_target_calls} target calls; maximum is "
            f"{_MAXIMUM_TARGET_CALLS}"
        )
    if not dry_run:
        if target_url is None:
            raise typer.BadParameter("execution requires --target-url", param_hint="--target-url")
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
    try:
        settings = OpenRouterDatasetSettings()
        _validate_model_input_bounds(selected_records, settings.max_input_chars)
        if target_url is not None:
            validate_json_http_dataset_target_configuration(
                target_url,
                sandbox_confirmed=confirm_isolated_sandbox or dry_run,
                fresh_state_confirmed=confirm_fresh_state or dry_run,
                request_field=request_field,
                header_environment_variables=header_environment_variables,
                allow_insecure_http=allow_insecure_http,
            )
    except (ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from None
    if dry_run:
        _print_dataset_plan(
            record_count=len(records),
            selected_count=len(selected_records),
            operator_ids=selected_operators,
            target_configured=target_url is not None,
        )
        return

    assert target_url is not None
    assert output is not None
    if not settings.live_calls:
        raise typer.BadParameter("set UL_DATASET_LIVE_CALLS=true to allow semantic model calls")
    if not settings.allow_external_data_processing:
        raise typer.BadParameter(
            "set UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING=true to allow semantic model calls"
        )
    if settings.api_key is None or not settings.api_key.get_secret_value().strip():
        raise typer.BadParameter("set OPEN_ROUTER_API_KEY to run an evaluation")

    try:
        target = JsonHttpDatasetTarget(
            target_url,
            sandbox_confirmed=True,
            fresh_state_confirmed=True,
            request_field=request_field,
            header_environment_variables=header_environment_variables,
            allow_insecure_http=allow_insecure_http,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--target-url") from None

    try:
        output_stream = _create_private_output(output)
    except OSError as error:
        asyncio.run(target.aclose())
        raise typer.BadParameter(
            f"cannot create output file ({error.__class__.__name__})",
            param_hint="--output",
        ) from None

    has_review_findings = False
    try:
        with output_stream:
            results = asyncio.run(
                _evaluate_interaction_records(
                    selected_records,
                    selected_operators,
                    settings,
                    target,
                    output_stream,
                )
            )
            for result in results:
                has_review_findings |= _result_needs_review(result)
    except (TimeoutError, RuntimeError, ValueError, httpx.HTTPError) as error:
        console.print(
            f"Evaluation stopped ({error.__class__.__name__}). "
            f"Complete results written before the error remain in {output}."
        )
        raise typer.Exit(code=2) from None

    _print_dataset_results(results, output)
    if has_review_findings:
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


def _print_dataset_plan(
    *,
    record_count: int,
    selected_count: int,
    operator_ids: tuple[str, ...],
    target_configured: bool,
) -> None:
    potential_target_calls = selected_count * (1 + len(operator_ids))
    potential_model_calls = selected_count * (2 + 4 * len(operator_ids))
    console.print(f"Dataset valid: {record_count} interaction(s)")
    console.print(f"Selected interactions: {selected_count}")
    console.print(f"Operators: {', '.join(operator_ids)}")
    console.print(f"Potential semantic model calls: up to {potential_model_calls}")
    console.print(f"Potential target calls: up to {potential_target_calls}")
    console.print(f"Target: {'configured' if target_configured else 'not configured'}")
    console.print(
        "Semantic models receive historical inputs and outputs, generated variations, "
        "live control responses, and variation responses on execution."
    )
    console.print(
        "The target receives each selected original input once, then each accepted variation."
    )
    console.print("No model or target requests sent.")


def _create_private_output(path: Path) -> TextIO:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.fchmod(descriptor, 0o600)
    return os.fdopen(descriptor, "w", encoding="utf-8")


async def _evaluate_interaction_records(
    records: tuple[InteractionRecord, ...],
    operator_ids: tuple[str, ...],
    settings: OpenRouterDatasetSettings,
    target: JsonHttpDatasetTarget,
    output_stream: TextIO,
) -> tuple[DatasetEvaluationResult, ...]:
    results: list[DatasetEvaluationResult] = []
    async with OpenRouterSemanticDeconstructor(settings) as deconstructor, target:
        runner = DatasetEvaluationRunner(
            DatasetAugmentationEngine(deconstructor, deconstructor, deconstructor),
            deconstructor,
            target,
            allow_network_egress=True,
        )
        for record in records:
            result = await runner.run(record, operator_ids=operator_ids)
            output_stream.write(
                json.dumps(_customer_evidence_record(result), ensure_ascii=False) + "\n"
            )
            output_stream.flush()
            results.append(result)
    return tuple(results)


def _print_dataset_results(
    results: tuple[DatasetEvaluationResult, ...],
    output: Path,
) -> None:
    table = Table(title="Dataset evaluation")
    table.add_column("Case", style="cyan")
    table.add_column("Augmentation")
    table.add_column("Status")
    table.add_column("Finding")
    case_number = 0
    for result in results:
        case_number += 1
        table.add_row(
            str(case_number),
            "original replay",
            _BASELINE_STATUSES[result.baseline.verdict],
            ", ".join(_FINDING_LABELS[finding.category] for finding in result.baseline.findings)
            or "—",
        )
        for case in result.cases:
            case_number += 1
            table.add_row(
                str(case_number),
                case.candidate.operator_id,
                _CUSTOMER_STATUSES[case.verdict],
                ", ".join(_FINDING_LABELS[finding.category] for finding in case.findings) or "—",
            )
    console.print(table)
    console.print(f"Complete evidence: {output}")


def _result_needs_review(result: DatasetEvaluationResult) -> bool:
    return result.baseline.verdict == "divergence_needs_review" or any(
        case.verdict == "divergence_needs_review" for case in result.cases
    )


def _customer_evidence_record(result: DatasetEvaluationResult) -> dict[str, JsonValue]:
    baseline_findings = _customer_findings(result.baseline.findings)
    cases: list[JsonValue] = []
    for case in result.cases:
        cases.append(
            {
                "operator_id": case.candidate.operator_id,
                "operator_version": case.candidate.operator_version,
                "augmented_input": case.candidate.augmented_input,
                "status": _CUSTOMER_STATUSES[case.verdict],
                "variation_accepted": case.candidate.passed,
                "variation_rejection_reasons": list(case.candidate.failure_reasons),
                "findings": _customer_findings(case.findings),
                "inconclusive_reasons": list(case.inconclusive_reasons),
            }
        )
    return {
        "schema_version": "1.1.0",
        "interaction_id": result.source.id,
        "original_input": result.source.raw_input,
        "current_baseline": {
            "status": _BASELINE_STATUSES[result.baseline.verdict],
            "findings": baseline_findings,
            "inconclusive_reasons": list(result.baseline.inconclusive_reasons),
        },
        "cases": cases,
        "technical_details": cast(JsonValue, result.model_dump(mode="json")),
    }


def _customer_findings(
    findings: tuple[DatasetEvaluationFinding, ...],
) -> list[JsonValue]:
    return [
        {
            "category": finding.category,
            "summary": finding.message,
            "expected_effects": [
                effect.model_dump(mode="json") for effect in finding.expected_effects
            ],
            "observed_effects": [
                effect.model_dump(mode="json") for effect in finding.observed_effects
            ],
        }
        for finding in findings
    ]
