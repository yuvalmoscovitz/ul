from __future__ import annotations

import asyncio
import json
import os
import shlex
import stat
import sys
import unicodedata
from pathlib import Path
from typing import Annotated, Literal, TextIO

import typer
from pydantic import ValidationError
from ul import DatasetEvaluationResult
from ul.dataset_invariants import (
    DatasetInvariantRule,
    DatasetInvariantRuleEvaluation,
    DatasetInvariantRuleResult,
    DatasetInvariantTransitionRuleEvaluation,
    DatasetInvariantValueEqualsRuleEvaluation,
    DatasetInvariantValueInSetRuleEvaluation,
    ExactlyOneNewEffectInvariant,
    JsonArrayItemsUniqueByInvariant,
    JsonValueEqualsLiteralInvariant,
    JsonValueInAllowedSetInvariant,
    JsonValuesEqualInvariant,
    NoNewEffectInvariant,
    UnchangedBetweenCheckpointsInvariant,
)
from ul.dataset_regression import (
    DatasetRegressionCase,
    DatasetRegressionResult,
    DatasetRegressionReviewSnapshot,
    DatasetRegressionRunResult,
    create_dataset_regression_case,
    dataset_regression_target_config_sha256,
    load_dataset_regression_case,
    replay_dataset_regression,
    run_dataset_regressions,
)
from ul.http_environment import (
    JsonHttpEnvironmentConnection,
    JsonHttpTargetConfig,
    json_http_environment_calls_per_execution,
    json_http_environment_config_urls,
    load_json_http_environment_config,
)
from ul.local_target import LocalTargetConnection

from ul_cli.dataset_review import (
    load_confirmed_dataset_finding,
)
from ul_cli.environment import TEST_ENVIRONMENT_CONFIRMATION_MESSAGE
from ul_cli.probe import (
    ProbeFailure,
    confirm_probe_target,
    probe_target_evidence_receipt,
    resolve_probe_target,
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
    output: Annotated[
        Path,
        typer.Option(help="New private JSON regression case file."),
    ],
    target_config: Annotated[
        Path | None,
        typer.Option(
            "--environment-config",
            dir_okay=False,
            readable=True,
            help="Legacy stateful environment configuration to snapshot.",
        ),
    ] = None,
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
                "Confirm the exact raw input, literal environment-template values, and selected "
                "customer-rule definitions plus any observed output and review context, which may "
                "be sensitive and are not auto-redacted, are appropriate to store and version."
            ),
        ),
    ] = False,
) -> None:
    """Create a portable case from a confirmed semantic or invariant finding."""
    if not confirm_versioned_input:
        raise typer.BadParameter(
            "saving requires confirmation that exact input/output evidence, review context, "
            "literal environment-template values, and selected customer-rule definitions may be "
            "sensitive, are not auto-redacted, and are appropriate to store and version",
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
    if case.target.kind == "probe_target":
        _print_safe(
            "The case stores the exact observed input/output, accepted variation, customer-rule "
            "definitions, human review snapshot, and secret-safe discovery target receipt. These "
            "private values may be sensitive and are not auto-redacted. Header authentication "
            "values remain environment-backed. Historical output remains reference evidence, not "
            "a correctness oracle."
        )
    else:
        _print_safe(
            "The case stores the exact raw input, which may be sensitive and is not auto-redacted, "
            "plus selected customer-rule definitions and the declared observation authority. Rule "
            "literals, allowed sets, and literal request-template values are copied unredacted. "
            "Header authentication remains environment-backed. The embedded environment config is "
            "customer-declared at case creation, is not verified as the discovery target, and is "
            "never executed by replay."
        )
    if case.target.kind == "probe_target":
        _print_safe(
            "Replay: ul regression replay "
            f"{shlex.quote(str(output))} --target TARGET --confirm-target TARGET_DIGEST "
            "--max-target-calls "
            f"{case.discovery_repetitions} --output replay.json"
        )
        return
    if case.target.config is None or target_config is None:
        raise AssertionError("validated environment regression requires its target config")
    insecure_http_option = (
        " --allow-insecure-http"
        if any(
            url.casefold().startswith("http:")
            for url in json_http_environment_config_urls(case.target.config)
        )
        else ""
    )
    target_calls = case.discovery_repetitions * json_http_environment_calls_per_execution(
        case.target.config
    )
    _print_safe(
        "Replay: ul regression replay "
        f"{shlex.quote(str(output))} --environment-config {shlex.quote(str(target_config))} "
        "--allow-environment-network --confirm-test-environment"
        f" --max-environment-api-calls {target_calls}{insecure_http_option} "
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
    output: Annotated[
        Path,
        typer.Option(help="New private JSON replay evidence file."),
    ],
    target_config: Annotated[
        Path | None,
        typer.Option(
            "--environment-config",
            dir_okay=False,
            readable=True,
            help="Separately trusted environment API config whose digest must match the case.",
        ),
    ] = None,
    target_reference: Annotated[
        str | None,
        typer.Option(
            "--target",
            help="The same callable, command/config, or HTTP target used by ul probe.",
        ),
    ] = None,
    confirm_target: Annotated[
        str | None,
        typer.Option(help="Exact target confirmation digest displayed by this command."),
    ] = None,
    target_artifact: Annotated[
        list[Path] | None,
        typer.Option(help="Local target artifact to bind; repeat as needed."),
    ] = None,
    target_working_directory: Annotated[
        Path | None,
        typer.Option(help="Working directory for a direct Python callable target."),
    ] = None,
    target_interpreter: Annotated[
        Path | None,
        typer.Option(help="Python interpreter for a direct callable target."),
    ] = None,
    target_environment_variable: Annotated[
        list[str] | None,
        typer.Option(help="Environment variable exposed to a local target; repeat as needed."),
    ] = None,
    http_preset: Annotated[
        Literal["generic-json", "openai-chat"] | None,
        typer.Option(help="Direct HTTP request/response preset."),
    ] = None,
    request_json_template: Annotated[
        str | None,
        typer.Option(help="Direct HTTP JSON request template."),
    ] = None,
    response_json_pointer: Annotated[
        str | None,
        typer.Option(help="JSON pointer selecting the direct HTTP response value."),
    ] = None,
    agent_model: Annotated[
        str | None,
        typer.Option(help="Model value for the openai-chat HTTP preset."),
    ] = None,
    header_from_env: Annotated[
        list[str] | None,
        typer.Option(help="HTTP Header=ENV_VAR mapping; repeat as needed."),
    ] = None,
    allow_probe_target_network: Annotated[
        bool,
        typer.Option("--allow-target-network", help="Allow requests to a probe HTTP target."),
    ] = False,
    allow_target_network: Annotated[
        bool,
        typer.Option(
            "--allow-environment-network",
            help="Allow requests to the configured environment endpoint.",
        ),
    ] = False,
    confirm_test_environment: Annotated[
        bool,
        typer.Option(help=("Confirm the environment is intended for testing and can be reset.")),
    ] = False,
    allow_insecure_http: Annotated[
        bool, typer.Option(help="Allow an HTTP environment API. Intended for local environments.")
    ] = False,
    max_target_calls: Annotated[
        int,
        typer.Option(
            "--max-target-calls",
            "--max-environment-api-calls",
            min=1,
            help="Maximum environment API requests authorized for this replay.",
        ),
    ] = 100,
) -> None:
    """Replay the exact saved variation without semantic-model calls."""
    try:
        regression_case = load_dataset_regression_case(case_path)
        if (target_config is None) == (target_reference is None):
            raise ValueError("pass exactly one of --target or --environment-config")
        if target_reference is not None:
            if regression_case.target.kind != "probe_target":
                raise ValueError("this regression requires --environment-config")
            resolved_target = resolve_probe_target(
                target_reference,
                allow_insecure_http=allow_insecure_http,
                explicit_artifacts=tuple(target_artifact or ()),
                http_preset=http_preset,
                request_json_template=request_json_template,
                response_json_pointer=response_json_pointer,
                agent_model=agent_model,
                header_from_env=header_from_env,
                target_working_directory=target_working_directory,
                target_interpreter=target_interpreter,
                target_environment_variables=tuple(target_environment_variable or ()),
            )
            if probe_target_evidence_receipt(resolved_target) != regression_case.target.receipt:
                raise ValueError("resolved target identity does not match the regression case")
            if resolved_target.kind == "http" and not allow_probe_target_network:
                raise ValueError("HTTP target replay requires --allow-target-network")
            confirm_probe_target(resolved_target, confirmed_digest=confirm_target)
            resolved_target.revalidate_identity()
            requested_target_calls = (
                regression_case.discovery_repetitions * resolved_target.calls_per_execution
            )
            if requested_target_calls > max_target_calls:
                raise ValueError(
                    f"case requires {requested_target_calls} target calls, exceeding "
                    f"--max-target-calls {max_target_calls}; explicitly raise the call budget"
                )
            target = resolved_target.create_connection(
                requested_target_calls,
                resolved_target.maximum_active_target_seconds,
            )
        else:
            if regression_case.target.kind != "environment_http":
                raise ValueError("this regression requires --target")
            if target_config is None:
                raise AssertionError("validated environment replay requires its config")
            trusted_target_config = load_json_http_environment_config(target_config)
            if _target_config_sha256(trusted_target_config) != regression_case.target.config_sha256:
                raise ValueError(
                    "trusted environment config digest does not match the regression case"
                )
            requested_target_calls = (
                regression_case.discovery_repetitions
                * json_http_environment_calls_per_execution(trusted_target_config)
            )
            if requested_target_calls > max_target_calls:
                raise ValueError(
                    f"case requires {requested_target_calls} environment API calls, exceeding "
                    f"--max-target-calls {max_target_calls}; explicitly raise the call budget"
                )
            if not allow_target_network:
                raise ValueError("environment replay requires --allow-environment-network")
            if not confirm_test_environment:
                raise ValueError(TEST_ENVIRONMENT_CONFIRMATION_MESSAGE)
            target = JsonHttpEnvironmentConnection.from_config(
                trusted_target_config,
                test_environment_confirmed=True,
                allow_insecure_http=allow_insecure_http,
                max_environment_api_calls=max_target_calls,
            )
    except ProbeFailure as error:
        raise typer.BadParameter(_terminal_safe(error.explanation)) from None
    except (ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(_terminal_safe(str(error))) from None
    if output.exists():
        raise typer.BadParameter(
            "output already exists; UL will not overwrite it", param_hint="--output"
        )

    try:
        output_stream = _create_private_output(output)
    except OSError as error:
        asyncio.run(target.aclose())
        raise typer.BadParameter(
            f"cannot create replay output ({error.__class__.__name__})", param_hint="--output"
        ) from None

    with output_stream:
        replay_result = asyncio.run(
            _replay_regression_and_close(
                regression_case,
                target,
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

    _print_safe(f"Regression {regression_case.case_id}: {replay_result.status}")
    if regression_case.target.kind == "probe_target":
        response_evidence = (
            "observed"
            if any(item.status == "observed" for item in replay_result.executions)
            else "unavailable"
        )
        _print_safe(f"Response evidence: {response_evidence}")
        _print_safe("Trajectory evidence: unavailable")
        _print_safe("Committed-state evidence: unavailable")
    _print_safe(f"Complete replay evidence: {output}")
    if replay_result.status == "failed":
        raise typer.Exit(code=1)
    if replay_result.status == "inconclusive":
        raise typer.Exit(code=2)


async def _replay_regression_and_close(
    regression_case: DatasetRegressionCase,
    target: JsonHttpEnvironmentConnection | LocalTargetConnection,
    *,
    max_target_calls: int,
) -> DatasetRegressionResult:
    try:
        return await replay_dataset_regression(
            regression_case,
            target,
            allow_network_egress=True,
            max_target_calls=max_target_calls,
            target_receipt=regression_case.target.receipt,
        )
    finally:
        await target.aclose()


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
            "--environment-config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Separately trusted environment API config whose digest must match every case.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(help="New private JSON regression run evidence file."),
    ],
    allow_target_network: Annotated[
        bool,
        typer.Option(
            "--allow-environment-network",
            help="Allow requests to the configured environment endpoint.",
        ),
    ] = False,
    confirm_test_environment: Annotated[
        bool,
        typer.Option(help=("Confirm the environment is intended for testing and can be reset.")),
    ] = False,
    allow_insecure_http: Annotated[
        bool, typer.Option(help="Allow an HTTP environment API. Intended for local environments.")
    ] = False,
    max_target_calls: Annotated[
        int,
        typer.Option(
            "--max-environment-api-calls",
            min=1,
            help="Maximum total environment API requests authorized for this run.",
        ),
    ] = 100,
) -> None:
    """Replay saved regressions against the current black-box environment."""
    try:
        case_labels, regression_cases = _load_regression_cases(cases_path)
        trusted_target_config = load_json_http_environment_config(target_config)
        trusted_target_config_sha256 = _target_config_sha256(trusted_target_config)
        for regression_case in regression_cases:
            if regression_case.target.kind != "environment_http":
                raise ValueError(
                    f"case {regression_case.case_id} uses a probe target; replay it with --target"
                )
            if regression_case.target.config_sha256 != trusted_target_config_sha256:
                raise ValueError(
                    f"case {regression_case.case_id} environment config digest does not match "
                    "the trusted environment config"
                )
        requested_target_calls = sum(
            regression_case.discovery_repetitions
            * json_http_environment_calls_per_execution(
                _environment_regression_target_config(regression_case)
            )
            for regression_case in regression_cases
        )
        if requested_target_calls > max_target_calls:
            raise ValueError(
                f"run requires {requested_target_calls} environment API calls, exceeding "
                f"--max-environment-api-calls {max_target_calls}; explicitly raise the call budget"
            )
    except (ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(_terminal_safe(str(error))) from None

    if not allow_target_network:
        raise typer.BadParameter(
            "regression run requires --allow-environment-network",
            param_hint="--allow-environment-network",
        )
    if not confirm_test_environment:
        raise typer.BadParameter(
            TEST_ENVIRONMENT_CONFIRMATION_MESSAGE,
            param_hint="--confirm-test-environment",
        )
    if output.exists():
        raise typer.BadParameter(
            "output already exists; UL will not overwrite it", param_hint="--output"
        )

    try:
        target = JsonHttpEnvironmentConnection.from_config(
            trusted_target_config,
            test_environment_confirmed=True,
            allow_insecure_http=allow_insecure_http,
            max_environment_api_calls=max_target_calls,
        )
    except (ValueError, RuntimeError) as error:
        raise typer.BadParameter(
            _terminal_safe(str(error)), param_hint="--environment-config"
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
    target: JsonHttpEnvironmentConnection,
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
    target_config_path: Path | None,
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

    run_context = loaded_record.evidence.run_context
    probe_target_receipt = (
        run_context.target.receipt
        if run_context is not None and run_context.target.kind == "probe_target"
        else None
    )
    if probe_target_receipt is not None and target_config_path is not None:
        raise ValueError("probe findings do not accept --environment-config")
    if probe_target_receipt is None and target_config_path is None:
        raise ValueError("legacy environment findings require --environment-config")
    trusted_target_config = (
        load_json_http_environment_config(target_config_path)
        if target_config_path is not None
        else None
    )
    technical_result = DatasetEvaluationResult.model_validate(
        loaded_record.evidence.technical_details,
        strict=False,
    )
    return create_dataset_regression_case(
        finding_id=finding_id,
        evidence_sha256=loaded_record.sha256,
        review_id=active_review.review_id,
        interaction_id=loaded_record.evidence.interaction_id,
        operator_id=finding_case.operator_id,
        operator_version=finding_case.operator_version,
        original_input=loaded_record.evidence.original_input,
        variation_input=finding_case.augmented_input,
        historical_reference_output=technical_result.source.raw_observed_output,
        augmentation_target=finding_case.augmentation_target,
        target_config=trusted_target_config,
        target_receipt=probe_target_receipt,
        review_snapshot=(
            DatasetRegressionReviewSnapshot(
                review_id=active_review.review_id,
                status="confirmed",
                severity=active_review.severity,
                reviewer=active_review.reviewer,
                reason=active_review.reason,
                reviewed_at=active_review.reviewed_at.isoformat(),
            )
            if probe_target_receipt is not None
            else None
        ),
        discovery_cross_examination=(
            finding_case.cross_examination.model_dump(mode="json")
            if probe_target_receipt is not None and finding_case.cross_examination is not None
            else None
        ),
        source_suite_sha256=evaluation.suite_sha256,
        observation_authority=evaluation.observation_authority,
        state_observation_authority=(
            "environment_self_reported"
            if evaluation.observation_authority == "committed_state_snapshot"
            else None
        ),
        selected_rules=tuple(selected_rules),
        discovery_repetitions=loaded_record.evidence.execution_plan.repetitions,
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
    if isinstance(rule, DatasetInvariantTransitionRuleEvaluation):
        transition_rule_types = {
            "no_new_effect": NoNewEffectInvariant,
            "exactly_one_new_effect": ExactlyOneNewEffectInvariant,
            "unchanged_between_checkpoints": UnchangedBetweenCheckpointsInvariant,
        }
        transition_rule_type = transition_rule_types[rule.rule_type]
        return transition_rule_type.model_validate(
            {
                "type": rule.rule_type,
                "id": rule.rule_id,
                "version": rule.rule_version,
                "description": rule.description,
                "severity": rule.severity,
                "before_checkpoint": rule.before_checkpoint,
                "after_checkpoint": rule.after_checkpoint,
                "observation_pointer": rule.observation_pointer,
            }
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


def _target_config_sha256(config: JsonHttpTargetConfig) -> str:
    return dataset_regression_target_config_sha256(config)


def _environment_regression_target_config(
    regression_case: DatasetRegressionCase,
) -> JsonHttpTargetConfig:
    if regression_case.target.kind != "environment_http" or regression_case.target.config is None:
        raise ValueError("regression case does not contain an environment target config")
    return regression_case.target.config


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
