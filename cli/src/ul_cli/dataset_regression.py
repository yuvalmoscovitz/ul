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
from ul.dataset_invariants import JsonValuesEqualInvariant
from ul.dataset_regression import (
    DatasetRegressionCase,
    create_dataset_regression_case,
    dataset_regression_target_config_sha256,
    load_dataset_regression_case,
    replay_dataset_regression,
)
from ul.http_target import (
    JsonHttpDatasetTarget,
    JsonHttpDatasetTargetConfig,
    load_json_http_dataset_target_config,
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
    finding_id: Annotated[str, typer.Argument(help="Confirmed finding ID to save.")],
    target_config: Annotated[
        Path,
        typer.Option(
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
        typer.Option("--rule", help="Violated customer rule ID. Repeat to select more than one."),
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
                "Confirm the exact raw input and literal target-template values, which may be "
                "sensitive and are not auto-redacted, are appropriate to store and version."
            ),
        ),
    ] = False,
) -> None:
    """Create a portable case from a confirmed, invariant-backed finding."""
    if not confirm_versioned_input:
        raise typer.BadParameter(
            "saving requires confirmation that the exact raw input and literal target-template "
            "values may be sensitive, are not auto-redacted, and are appropriate to store and "
            "version",
            param_hint="--confirm-versioned-input",
        )
    if not rules:
        raise typer.BadParameter("pass at least one --rule", param_hint="--rule")
    if len(rules) != len(set(rules)):
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
            rule_ids=tuple(rules),
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
        "The case stores the exact raw input, which may be sensitive and is not auto-redacted, "
        "plus selected rules and the declared observation authority. Header authentication "
        "remains environment-backed, but literal request-template values may also be sensitive. "
        "The embedded target config is customer-declared at case creation, is not verified as "
        "the discovery target, and is never executed by replay."
    )
    insecure_http_option = (
        " --allow-insecure-http" if case.target.config.url.casefold().startswith("http:") else ""
    )
    _print_safe(
        "Replay: ul regression replay "
        f"{shlex.quote(str(output))} --target-config {shlex.quote(str(target_config))} "
        "--allow-target-network --confirm-isolated-sandbox --confirm-fresh-state "
        f"--max-target-calls {case.discovery_repetitions}{insecure_http_option} "
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
            exists=True,
            dir_okay=False,
            readable=True,
            help="Separately trusted target config whose digest must match the case.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(help="New private JSON replay evidence file."),
    ],
    allow_target_network: Annotated[
        bool, typer.Option(help="Allow requests to the configured sandbox endpoint.")
    ] = False,
    confirm_isolated_sandbox: Annotated[
        bool, typer.Option(help="Confirm the target cannot cause real business effects.")
    ] = False,
    confirm_fresh_state: Annotated[
        bool, typer.Option(help="Confirm every target request starts from the same clean state.")
    ] = False,
    allow_insecure_http: Annotated[
        bool, typer.Option(help="Allow an HTTP target. Intended for local sandboxes.")
    ] = False,
    max_target_calls: Annotated[
        int,
        typer.Option(min=1, help="Maximum target requests authorized for this replay."),
    ] = 100,
) -> None:
    """Replay the exact saved variation without semantic-model calls."""
    try:
        regression_case = load_dataset_regression_case(case_path)
        trusted_target_config = load_json_http_dataset_target_config(target_config)
        if _target_config_sha256(trusted_target_config) != regression_case.target.config_sha256:
            raise ValueError("trusted target config digest does not match the regression case")
    except (ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(_terminal_safe(str(error))) from None

    if regression_case.discovery_repetitions > max_target_calls:
        raise typer.BadParameter(
            f"case requires {regression_case.discovery_repetitions} target calls, exceeding "
            f"--max-target-calls {max_target_calls}; explicitly raise the call budget",
            param_hint="--max-target-calls",
        )

    if not allow_target_network:
        raise typer.BadParameter(
            "replay requires --allow-target-network", param_hint="--allow-target-network"
        )
    if not confirm_isolated_sandbox:
        raise typer.BadParameter(
            "replay requires --confirm-isolated-sandbox",
            param_hint="--confirm-isolated-sandbox",
        )
    if not confirm_fresh_state:
        raise typer.BadParameter(
            "replay requires --confirm-fresh-state", param_hint="--confirm-fresh-state"
        )
    if output.exists():
        raise typer.BadParameter(
            "output already exists; UL will not overwrite it", param_hint="--output"
        )

    try:
        target = JsonHttpDatasetTarget.from_config(
            trusted_target_config,
            sandbox_confirmed=True,
            fresh_state_confirmed=True,
            allow_insecure_http=allow_insecure_http,
        )
    except (ValueError, RuntimeError) as error:
        raise typer.BadParameter(_terminal_safe(str(error)), param_hint="--target-config") from None
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
    selected_rule_ids = set(rule_ids)
    missing_rule_ids = selected_rule_ids - baseline_rules.keys()
    if missing_rule_ids:
        raise ValueError(f"rule {sorted(missing_rule_ids)[0]!r} was not evaluated for both arms")
    selected_rules: list[JsonValuesEqualInvariant] = []
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
        baseline_trial = baseline_rule.trials[0]
        variation_trial = variation_rule.trials[0]
        if (
            baseline_trial.left_pointer != variation_trial.left_pointer
            or baseline_trial.right_pointer != variation_trial.right_pointer
        ):
            raise ValueError(f"rule {rule_id!r} uses inconsistent JSON pointers")
        selected_rules.append(
            JsonValuesEqualInvariant(
                type="json_values_equal",
                id=baseline_rule.rule_id,
                version=baseline_rule.rule_version,
                description=baseline_rule.description,
                severity=baseline_rule.severity,
                left_pointer=baseline_trial.left_pointer,
                right_pointer=baseline_trial.right_pointer,
            )
        )

    trusted_target_config = load_json_http_dataset_target_config(target_config_path)
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
        selected_rules=tuple(selected_rules),
        discovery_repetitions=loaded_record.evidence.execution_plan.repetitions,
    )


def _target_config_sha256(config: JsonHttpDatasetTargetConfig) -> str:
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
