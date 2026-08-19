from __future__ import annotations

import asyncio
import json
import os
import shlex
import stat
import sys
import unicodedata
from pathlib import Path
from typing import Annotated, TextIO

import typer
from pydantic import ValidationError
from ul import DatasetEvaluationResult
from ul.dataset_invariants import (
    DatasetInvariantRule,
    DatasetInvariantRuleEvaluation,
    DatasetInvariantRuleResult,
    DatasetInvariantValueEqualsRuleEvaluation,
    DatasetInvariantValueInSetRuleEvaluation,
    JsonArrayItemsUniqueByInvariant,
    JsonValueEqualsLiteralInvariant,
    JsonValueInAllowedSetInvariant,
    JsonValuesEqualInvariant,
)
from ul.dataset_regression import (
    DatasetRegressionCase,
    DatasetRegressionRunResult,
    create_dataset_regression_case,
    dataset_regression_target_config_sha256,
    load_dataset_regression_case,
    replay_dataset_regression,
    run_dataset_regressions,
)
from ul.http_sandbox import (
    JsonHttpSandboxConfig,
    JsonHttpSandboxConnection,
    json_http_sandbox_calls_per_execution,
    json_http_sandbox_config_urls,
    load_json_http_sandbox_config,
)

from ul_cli.dataset_review import (
    load_confirmed_dataset_finding,
)

app = typer.Typer(help="Save and replay confirmed dataset findings.")


@app.command("save")
def save_dataset_regression(
    evidence: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Evaluation evidence JSONL.",
        ),
    ],
    finding_id: Annotated[
        str,
        typer.Argument(help="Confirmed semantic or customer-invariant finding ID to save."),
    ],
    target_config: Annotated[
        Path,
        typer.Option(
            "--sandbox-config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Target configuration to snapshot.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(help="New private JSON regression case file."),
    ],
    rules: Annotated[
        list[str] | None,
        typer.Option(
            "--rule",
            help=(
                "Violated customer rule ID. Required for semantic findings; an invariant "
                "finding selects its rule automatically."
            ),
        ),
    ] = None,
    reviews: Annotated[
        Path | None,
        typer.Option(help="Review JSONL; defaults to EVIDENCE with .reviews.jsonl suffix."),
    ] = None,
    confirm_versioned_input: Annotated[
        bool,
        typer.Option(
            "--confirm-versioned-input",
            help=(
                "Confirm the exact raw input, literal sandbox-template values, and selected "
                "customer-rule definitions plus any per-record sandbox setup fixture, which may "
                "be sensitive and are not auto-redacted, are appropriate to store and version."
            ),
        ),
    ] = False,
) -> None:
    """Create a portable case from a confirmed semantic or invariant finding."""
    if not confirm_versioned_input:
        raise typer.BadParameter(
            "saving requires confirmation that the exact raw input and literal sandbox-template "
            "values plus selected customer-rule definitions and any per-record sandbox setup "
            "fixture may be sensitive, are not auto-redacted, and are appropriate to store and "
            "version",
            param_hint="--confirm-versioned-input",
        )
    if rules is not None and len(rules) != len(set(rules)):
        raise typer.BadParameter("duplicate --rule values are not allowed", param_hint="--rule")
    if output.exists():
        raise typer.BadParameter(
            "output already exists; UL will not overwrite it", param_hint="--output"
        )

    try:
        case = _build_regression_case(
            evidence=evidence,
            reviews=reviews or evidence.with_suffix(".reviews.jsonl"),
            finding_id=finding_id,
            rule_ids=tuple(rules or ()),
            target_config_path=target_config,
        )
        output_stream = _create_private_output(output)
    except ValidationError:
        raise typer.BadParameter("regression case fields are invalid") from None
    except (ValueError, RuntimeError) as error:
        raise typer.BadParameter(_terminal_safe(str(error))) from None
    except OSError as error:
        raise typer.BadParameter(
            f"cannot create regression case ({error.__class__.__name__})",
            param_hint="--output",
        ) from None

    with output_stream:
        json.dump(
            case.model_dump(mode="json"),
            output_stream,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        output_stream.flush()
        os.fsync(output_stream.fileno())

    _print_safe(f"Saved regression case {case.case_id}: {output}")
    _print_safe(
        "The case stores the exact raw input and any per-record sandbox setup fixture, which may "
        "be sensitive and are not auto-redacted, plus selected customer-rule definitions and the "
        "declared observation authority. Rule literals, allowed sets, fixture values, and literal "
        "request-template values are copied unredacted. "
        "Header authentication remains environment-backed. "
        "The embedded sandbox config is customer-declared at case creation, is not verified as "
        "the discovery target, and is never executed by replay."
    )
    insecure_http_option = (
        " --allow-insecure-http"
        if any(
            url.casefold().startswith("http:")
            for url in json_http_sandbox_config_urls(case.target.config)
        )
        else ""
    )
    target_calls = case.discovery_repetitions * json_http_sandbox_calls_per_execution(
        case.target.config
    )
    _print_safe(
        "Replay: ul regression replay "
        f"{shlex.quote(str(output))} --sandbox-config {shlex.quote(str(target_config))} "
        "--allow-sandbox-network-egress --confirm-isolated-sandbox"
        f" --max-sandbox-api-calls {target_calls}{insecure_http_option} "
        "--output replay.json"
    )


@app.command("replay")
def replay_saved_dataset_regression(
    case_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Saved regression case JSON.",
        ),
    ],
    target_config: Annotated[
        Path,
        typer.Option(
            "--sandbox-config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Separately trusted sandbox API config whose digest must match the case.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(help="New private JSON replay evidence file."),
    ],
    allow_target_network: Annotated[
        bool,
        typer.Option(
            "--allow-sandbox-network-egress",
            help="Allow requests to the configured sandbox endpoint.",
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
        bool, typer.Option(help="Allow an HTTP sandbox API. Intended for local sandboxes.")
    ] = False,
    max_target_calls: Annotated[
        int,
        typer.Option(
            "--max-sandbox-api-calls",
            min=1,
            help="Maximum sandbox API requests authorized for this replay.",
        ),
    ] = 100,
) -> None:
    """Replay the exact saved variation without semantic-model calls."""
    try:
        regression_case = load_dataset_regression_case(case_path)
        trusted_target_config = load_json_http_sandbox_config(target_config)
        if _target_config_sha256(trusted_target_config) != regression_case.target.config_sha256:
            raise ValueError("trusted sandbox config digest does not match the regression case")
    except (ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(_terminal_safe(str(error))) from None

    requested_target_calls = (
        regression_case.discovery_repetitions
        * json_http_sandbox_calls_per_execution(trusted_target_config)
    )
    if requested_target_calls > max_target_calls:
        raise typer.BadParameter(
            f"case requires {requested_target_calls} sandbox API calls, exceeding "
            f"--max-sandbox-api-calls {max_target_calls}; explicitly raise the call budget",
            param_hint="--max-sandbox-api-calls",
        )

    if not allow_target_network:
        raise typer.BadParameter(
            "replay requires --allow-sandbox-network-egress",
            param_hint="--allow-sandbox-network-egress",
        )
    if not confirm_isolated_sandbox:
        raise typer.BadParameter(
            "replay requires --confirm-isolated-sandbox",
            param_hint="--confirm-isolated-sandbox",
        )
    if output.exists():
        raise typer.BadParameter(
            "output already exists; UL will not overwrite it", param_hint="--output"
        )

    try:
        target = JsonHttpSandboxConnection.from_config(
            trusted_target_config,
            sandbox_confirmed=True,
            allow_insecure_http=allow_insecure_http,
            max_sandbox_api_calls=max_target_calls,
        )
    except (ValueError, RuntimeError) as error:
        raise typer.BadParameter(
            _terminal_safe(str(error)), param_hint="--sandbox-config"
        ) from None
    try:
        output_stream = _create_private_output(output)
    except OSError as error:
        asyncio.run(target.aclose())
        raise typer.BadParameter(
            f"cannot create replay output ({error.__class__.__name__})", param_hint="--output"
        ) from None

    try:
        with output_stream:
            replay_result = asyncio.run(
                replay_dataset_regression(
                    regression_case,
                    target,
                    allow_network_egress=True,
                    max_target_calls=max_target_calls,
                )
            )
            json.dump(
                replay_result.model_dump(mode="json"),
                output_stream,
                ensure_ascii=False,
                indent=2,
            )
            output_stream.write("\n")
            output_stream.flush()
            os.fsync(output_stream.fileno())
    finally:
        asyncio.run(target.aclose())

    _print_safe(f"Regression {regression_case.case_id}: {replay_result.status}")
    _print_safe(f"Complete replay evidence: {output}")
    if replay_result.status == "failed":
        raise typer.Exit(code=1)
    if replay_result.status == "inconclusive":
        raise typer.Exit(code=2)


@app.command("run")
def run_saved_dataset_regressions(
    cases_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            readable=True,
            help="Saved regression case JSON file or directory of case JSON files.",
        ),
    ],
    target_config: Annotated[
        Path,
        typer.Option(
            "--sandbox-config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Separately trusted sandbox API config whose digest must match every case.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(help="New private JSON regression run evidence file."),
    ],
    allow_target_network: Annotated[
        bool,
        typer.Option(
            "--allow-sandbox-network-egress",
            help="Allow requests to the configured sandbox endpoint.",
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
        bool, typer.Option(help="Allow an HTTP sandbox API. Intended for local sandboxes.")
    ] = False,
    max_target_calls: Annotated[
        int,
        typer.Option(
            "--max-sandbox-api-calls",
            min=1,
            help="Maximum total sandbox API requests authorized for this run.",
        ),
    ] = 100,
) -> None:
    """Replay saved regressions against the current black-box sandbox."""
    try:
        case_labels, regression_cases = _load_regression_cases(cases_path)
        trusted_target_config = load_json_http_sandbox_config(target_config)
        trusted_target_config_sha256 = _target_config_sha256(trusted_target_config)
        for regression_case in regression_cases:
            if regression_case.target.config_sha256 != trusted_target_config_sha256:
                raise ValueError(
                    f"case {regression_case.case_id} sandbox config digest does not match "
                    "the trusted sandbox config"
                )
        requested_target_calls = sum(
            regression_case.discovery_repetitions
            * json_http_sandbox_calls_per_execution(regression_case.target.config)
            for regression_case in regression_cases
        )
        if requested_target_calls > max_target_calls:
            raise ValueError(
                f"run requires {requested_target_calls} sandbox API calls, exceeding "
                f"--max-sandbox-api-calls {max_target_calls}; explicitly raise the call budget"
            )
    except (ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(_terminal_safe(str(error))) from None

    if not allow_target_network:
        raise typer.BadParameter(
            "regression run requires --allow-sandbox-network-egress",
            param_hint="--allow-sandbox-network-egress",
        )
    if not confirm_isolated_sandbox:
        raise typer.BadParameter(
            "regression run requires --confirm-isolated-sandbox",
            param_hint="--confirm-isolated-sandbox",
        )
    if output.exists():
        raise typer.BadParameter(
            "output already exists; UL will not overwrite it", param_hint="--output"
        )

    try:
        target = JsonHttpSandboxConnection.from_config(
            trusted_target_config,
            sandbox_confirmed=True,
            allow_insecure_http=allow_insecure_http,
            max_sandbox_api_calls=max_target_calls,
        )
    except (ValueError, RuntimeError) as error:
        raise typer.BadParameter(
            _terminal_safe(str(error)), param_hint="--sandbox-config"
        ) from None
    try:
        output_stream = _create_private_output(output)
    except OSError as error:
        asyncio.run(target.aclose())
        raise typer.BadParameter(
            f"cannot create regression run output ({error.__class__.__name__})",
            param_hint="--output",
        ) from None

    with output_stream:
        run_result = asyncio.run(
            _run_regressions_and_close(
                regression_cases,
                target,
                case_labels=case_labels,
                max_target_calls=max_target_calls,
            )
        )
        json.dump(
            run_result.model_dump(mode="json"),
            output_stream,
            ensure_ascii=False,
            indent=2,
        )
        output_stream.write("\n")
        output_stream.flush()
        os.fsync(output_stream.fileno())

    _print_safe(f"Regression run: {run_result.status}")
    _print_safe(
        "Cases: "
        f"passed={run_result.passed_case_count}, "
        f"failed={run_result.failed_case_count}, "
        f"inconclusive={run_result.inconclusive_case_count}"
    )
    _print_safe(f"Target calls requested: {run_result.requested_target_calls}")
    for case_result in run_result.cases:
        violated_rules = tuple(
            f"{rule.rule_id} ({rule.severity})"
            for rule in case_result.result.rules
            if rule.status == "violated"
        )
        violation_summary = (
            "; violated rules: " + ", ".join(violated_rules) if violated_rules else ""
        )
        _print_safe(
            f"{case_result.label}: {case_result.result.status}{violation_summary}; "
            f"case={case_result.result.case_id}"
        )
    _print_safe(f"Complete regression run evidence: {output}")
    if run_result.status == "failed":
        raise typer.Exit(code=1)
    if run_result.status == "inconclusive":
        raise typer.Exit(code=2)


async def _run_regressions_and_close(
    regression_cases: tuple[DatasetRegressionCase, ...],
    target: JsonHttpSandboxConnection,
    *,
    case_labels: tuple[str, ...],
    max_target_calls: int,
) -> DatasetRegressionRunResult:
    try:
        return await run_dataset_regressions(
            regression_cases,
            target,
            case_labels=case_labels,
            allow_network_egress=True,
            max_target_calls=max_target_calls,
        )
    finally:
        await target.aclose()


def _load_regression_cases(
    path: Path,
) -> tuple[tuple[str, ...], tuple[DatasetRegressionCase, ...]]:
    if path.is_symlink():
        raise ValueError("regression case input must not be a symbolic link")
    if path.is_file():
        case_paths = (path,)
    elif path.is_dir():
        try:
            case_paths = tuple(
                sorted(
                    (
                        candidate
                        for candidate in path.iterdir()
                        if candidate.name.casefold().endswith(".json")
                    ),
                    key=lambda candidate: candidate.name,
                )
            )
        except OSError:
            raise RuntimeError("regression case directory could not be read") from None
        if not case_paths:
            raise ValueError("regression case directory contains no JSON files")
    else:
        raise ValueError("regression case input must be a regular file or directory")
    if len(case_paths) > 100:
        raise ValueError("regression run supports at most 100 cases")
    regression_cases = tuple(load_dataset_regression_case(case_path) for case_path in case_paths)
    case_ids = tuple(regression_case.case_id for regression_case in regression_cases)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("regression run case IDs must be unique")
    return tuple(case_path.name for case_path in case_paths), regression_cases


def _build_regression_case(
    *,
    evidence: Path,
    reviews: Path,
    finding_id: str,
    rule_ids: tuple[str, ...],
    target_config_path: Path,
) -> DatasetRegressionCase:
    selected = load_confirmed_dataset_finding(evidence, reviews, finding_id)
    loaded_record = selected.evidence_record
    finding_case = selected.case
    active_review = selected.review
    evaluation = loaded_record.evidence.invariant_evaluation
    if evaluation is None:
        raise ValueError("finding evidence does not include customer invariant results")
    variation_evaluations = tuple(
        arm for arm in evaluation.variations if arm.operator_id == finding_case.operator_id
    )
    if len(variation_evaluations) != 1:
        raise ValueError("finding does not map to exactly one invariant variation arm")
    baseline_rules = {rule.rule_id: rule for rule in evaluation.baseline.rules}
    variation_rules = {rule.rule_id: rule for rule in variation_evaluations[0].rules}
    if selected.kind == "customer_invariant_violation":
        if selected.invariant_rule_id is None:
            raise ValueError("invariant finding does not identify its violated rule")
        if rule_ids and rule_ids != (selected.invariant_rule_id,):
            raise ValueError(
                "an invariant finding automatically selects its one violated customer rule"
            )
        selected_rule_ids = {selected.invariant_rule_id}
    else:
        if not rule_ids:
            raise ValueError("semantic findings require at least one --rule")
        selected_rule_ids = set(rule_ids)
    missing_rule_ids = selected_rule_ids - baseline_rules.keys()
    if missing_rule_ids:
        raise ValueError(f"rule {sorted(missing_rule_ids)[0]!r} was not evaluated for both arms")
    selected_rules: list[DatasetInvariantRule] = []
    for baseline_rule in evaluation.baseline.rules:
        rule_id = baseline_rule.rule_id
        if rule_id not in selected_rule_ids:
            continue
        variation_rule = variation_rules.get(rule_id)
        if variation_rule is None:
            raise ValueError(f"rule {rule_id!r} was not evaluated for both arms")
        if baseline_rule.status != "satisfied" or variation_rule.status != "violated":
            raise ValueError(
                f"rule {rule_id!r} must be satisfied on the original and violated on the variation"
            )
        baseline_definition = _invariant_rule_definition(baseline_rule)
        variation_definition = _invariant_rule_definition(variation_rule)
        if baseline_definition != variation_definition:
            raise ValueError(f"rule {rule_id!r} uses inconsistent definitions")
        selected_rules.append(baseline_definition)

    trusted_target_config = load_json_http_sandbox_config(target_config_path)
    technical_result = DatasetEvaluationResult.model_validate_json(
        json.dumps(loaded_record.evidence.technical_details)
    )
    if technical_result.source.id != loaded_record.evidence.interaction_id:
        raise ValueError("finding technical source does not match its evidence interaction")
    return create_dataset_regression_case(
        finding_id=finding_id,
        evidence_sha256=loaded_record.sha256,
        review_id=active_review.review_id,
        interaction_id=loaded_record.evidence.interaction_id,
        operator_id=finding_case.operator_id,
        operator_version=finding_case.operator_version,
        original_input=loaded_record.evidence.original_input,
        variation_input=finding_case.augmented_input,
        target_config=trusted_target_config,
        source_suite_sha256=evaluation.suite_sha256,
        observation_authority=evaluation.observation_authority,
        state_observation_authority=(
            "sandbox_self_reported"
            if evaluation.observation_authority == "committed_state_snapshot"
            else None
        ),
        selected_rules=tuple(selected_rules),
        discovery_repetitions=loaded_record.evidence.execution_plan.repetitions,
        sandbox_setup=technical_result.source.sandbox_setup,
    )


def _invariant_rule_definition(rule: DatasetInvariantRuleResult) -> DatasetInvariantRule:
    if isinstance(rule, DatasetInvariantRuleEvaluation):
        first_trial = rule.trials[0]
        return JsonValuesEqualInvariant(
            type="json_values_equal",
            id=rule.rule_id,
            version=rule.rule_version,
            description=rule.description,
            severity=rule.severity,
            left_pointer=first_trial.left_pointer,
            right_pointer=first_trial.right_pointer,
        )
    if isinstance(rule, DatasetInvariantValueEqualsRuleEvaluation):
        return JsonValueEqualsLiteralInvariant(
            type="json_value_equals_literal",
            id=rule.rule_id,
            version=rule.rule_version,
            description=rule.description,
            severity=rule.severity,
            value_pointer=rule.value_pointer,
            literal=rule.literal,
        )
    if isinstance(rule, DatasetInvariantValueInSetRuleEvaluation):
        return JsonValueInAllowedSetInvariant(
            type="json_value_in_allowed_set",
            id=rule.rule_id,
            version=rule.rule_version,
            description=rule.description,
            severity=rule.severity,
            value_pointer=rule.value_pointer,
            allowed_values=rule.allowed_values,
        )
    return JsonArrayItemsUniqueByInvariant(
        type="json_array_items_unique_by",
        id=rule.rule_id,
        version=rule.rule_version,
        description=rule.description,
        severity=rule.severity,
        array_pointer=rule.array_pointer,
        key_pointers=rule.key_pointers,
    )


def _target_config_sha256(config: JsonHttpSandboxConfig) -> str:
    return dataset_regression_target_config_sha256(config)


def _create_private_output(path: Path) -> TextIO:
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    binary_flag = os.O_BINARY if sys.platform == "win32" else 0
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow_flag | binary_flag,
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("output is not a regular file")
        if sys.platform != "win32":
            os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "w", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise


def _print_safe(message: str) -> None:
    typer.echo(_terminal_safe(message))


def _terminal_safe(message: str) -> str:
    return "".join(
        character
        if (ord(character) >= 32 and not 0x7F <= ord(character) <= 0x9F)
        and unicodedata.category(character) not in {"Cf", "Cs"}
        else f"\\u{ord(character):04x}"
        for character in message
    )
