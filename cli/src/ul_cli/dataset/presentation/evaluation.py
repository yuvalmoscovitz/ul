from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import typer
from rich.table import Table
from ul import DatasetEvaluationResult, EvaluatorModelPreflight
from ul.dataset_invariants import (
    DatasetInvariantArrayUniqueTrialEvaluation,
    DatasetInvariantEvaluation,
    DatasetInvariantSuite,
    DatasetInvariantTransitionTrialEvaluation,
    DatasetInvariantTrialEvaluation,
    DatasetInvariantValueEqualsTrialEvaluation,
    DatasetInvariantValueInSetTrialEvaluation,
)

from ul_cli.dataset_campaign import DatasetCampaignPlan
from ul_cli.dataset_review import DatasetEvidenceRedactionCoverage

from ..evidence.customer import baseline_customer_status, case_customer_status, trial_set_summary
from .runtime import console, print_dataset_plain

_FINDING_LABELS = {
    "duplicate_effect": "duplicate action",
    "unexpected_effect": "unexpected action",
    "missing_effect": "missing action",
    "changed_grounded_effect_argument": "changed action value",
}


def print_dataset_plan(
    *,
    record_count: int,
    selected_count: int,
    skipped_count: int,
    operator_ids: tuple[str, ...],
    evaluation_mode: Literal["variance"],
    target_configured: bool,
    target_endpoint: str | None,
    target_header_environment_variables: dict[str, str],
    repetitions: int,
    max_environment_api_calls: int,
    target_calls_per_execution: int,
    target_supports_state_observation: bool | None,
    fixture_status: Literal["configured", "missing", "not_required"] | None,
    fixture_id: str | None,
    fixture_version: str | None,
    invariant_suite: DatasetInvariantSuite | None,
    output: Path | None,
    augmentations_output: Path | None,
    semantic_provider_id: str,
    semantic_endpoint_sha256: str,
    redaction_policy_sha256: str | None,
    redaction_coverage: tuple[DatasetEvidenceRedactionCoverage, ...],
    campaign_plan: DatasetCampaignPlan,
    json_output: bool,
) -> None:
    if json_output:
        typer.echo(json.dumps(campaign_plan.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return
    potential_target_calls = campaign_plan.calls.total_environment_api
    potential_model_calls = campaign_plan.calls.total_semantic_model
    console.print(f"Dataset valid: {record_count} interaction(s)")
    if skipped_count:
        console.print(
            f"Resume compatible: {skipped_count} complete interaction(s) skipped; "
            f"{selected_count} remaining"
        )
    else:
        console.print(f"Selected interactions: {selected_count}")
    console.print(f"Operators: {', '.join(operator_ids)}")
    console.print(
        f"Evaluation mode: {evaluation_mode} (historical output is not an expected answer; "
        "correctness not assessed)"
    )
    console.print(f"Repetitions: {repetitions} per original and accepted variation")
    if invariant_suite is None:
        console.print("Customer invariants: none")
    else:
        console.print(f"Customer invariants: {len(invariant_suite.rules)} rule(s)")
        console.print(f"Declared observation authority: {invariant_suite.observation_authority}")
        console.print("Additional model calls for customer invariants: 0")
        console.print("Additional environment API calls for customer invariants: 0")
    console.print(f"Potential semantic model calls: up to {potential_model_calls}")
    console.print(
        "Planned maximum calls: "
        f"baseline={campaign_plan.calls.baseline}, "
        f"variation={campaign_plan.calls.variation}, "
        f"repetition_executions={campaign_plan.calls.repetition_executions}, "
        f"repetition_rounds={campaign_plan.calls.repetitions}, "
        f"retries={campaign_plan.calls.retries}, "
        f"preflight={campaign_plan.calls.preflight}, "
        f"evaluators={campaign_plan.calls.evaluators}"
    )
    for profile in campaign_plan.preflight_profiles:
        print_dataset_plain(
            "Evaluator preflight profile: "
            f"roles={','.join(profile.roles)}, model={profile.requested_model}, "
            f"max_completion_tokens={profile.max_completion_tokens}"
        )
    console.print(
        "Estimated completion tokens: "
        f"{campaign_plan.tokens.minimum}..{campaign_plan.tokens.maximum}"
    )
    console.print("Estimated monetary cost: unavailable (no trusted pricing configured)")
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
        f"Potential environment API calls: up to {potential_target_calls} "
        f"(authorized maximum: {max_environment_api_calls})"
    )
    if target_calls_per_execution > 1:
        console.print(
            f"Lifecycle calls per execution: {target_calls_per_execution} "
            "(reset, optional setup, execute_turn, snapshot, cleanup reset)"
        )
    elif target_supports_state_observation is False:
        console.print("Adapter tier: isolated-response (response evidence only)")
    target_status = "configured" if target_configured else "not configured"
    console.print(f"Customer-managed environment API: {target_status}")
    for warning in campaign_plan.warnings:
        print_dataset_plain(f"Warning: {warning}")
    console.print("Selected operator applicability by interaction")
    for example in campaign_plan.examples:
        print_dataset_plain(f"  {example.interaction_id}")
        for planned_operator in (operator for operator in example.operators if operator.selected):
            print_dataset_plain(
                f"    {planned_operator.status.upper()} {planned_operator.id}@"
                f"{planned_operator.version} selected"
            )
            for reason in planned_operator.reasons:
                print_dataset_plain(f"      Reason: {reason}")
            if (
                planned_operator.candidate_input_available
                and planned_operator.candidate_input is None
            ):
                print_dataset_plain("      Candidate input: omitted (sensitive)")
            if planned_operator.candidate_input is not None:
                print_dataset_plain(f"      Candidate input: {planned_operator.candidate_input}")
    if campaign_plan.examples:
        unselected_operators = tuple(
            operator for operator in campaign_plan.examples[0].operators if not operator.selected
        )
        unselected_eligible = sum(
            operator.status == "eligible" for operator in unselected_operators
        )
        unselected_conditional = sum(
            operator.status == "conditional" for operator in unselected_operators
        )
        unselected_ineligible = sum(
            operator.status == "ineligible" for operator in unselected_operators
        )
        console.print(
            "Unselected catalog operators: "
            f"{unselected_eligible} eligible, {unselected_conditional} conditional, "
            f"{unselected_ineligible} ineligible "
            "(use --json for full detail)"
        )
    if fixture_status is not None and fixture_status != "missing":
        print_fixture_identity(
            fixture_status,
            fixture_id=fixture_id,
            fixture_version=fixture_version,
        )
    if output is not None:
        console.print(f"Evidence destination: {output}")
    if augmentations_output is None:
        console.print(
            "Augmentations will not be saved. Interrupted generation may repeat model calls."
        )
    else:
        print_dataset_plain(f"Augmentations destination: {augmentations_output}")
        print_dataset_plain(
            "This private artifact may contain sensitive inputs and derived semantic data. It is "
            "not encrypted or automatically redacted; retain it only under your data policy."
        )
    if target_endpoint is not None:
        console.print(f"Environment API endpoint: {target_endpoint}")
        if target_header_environment_variables:
            mappings = ", ".join(
                f"{header_name}={environment_variable}"
                for header_name, environment_variable in sorted(
                    target_header_environment_variables.items()
                )
            )
            console.print(f"Environment API header environment mappings: {mappings}")
        else:
            console.print("Environment API header environment mappings: none")
    console.print(
        "Semantic models receive historical inputs and outputs, generated variations, "
        "live control responses, and variation responses on execution."
    )
    if target_supports_state_observation is False:
        console.print(
            "Every test case sends one isolated request. UL does not observe committed state, "
            "cleanup, or behavior across turns at this tier."
        )
    else:
        console.print(
            "Every test case invokes and validates the configured environment reset contract. "
            "Optional setup uses one static fixture from the environment config for the entire run."
        )
    console.print(
        "Target requests and semantic model calls may be billed separately. Repetitions only "
        "show observed behavioral consistency: they do not determine correctness, identify "
        "causality, or estimate a production failure rate."
    )
    console.print("No model or environment API requests sent.")


def print_fixture_identity(
    status: Literal["configured", "missing", "not_required"],
    *,
    fixture_id: str | None,
    fixture_version: str | None,
) -> None:
    if status == "configured":
        print_dataset_plain(f"Fixture: {fixture_id}@{fixture_version}")
    elif status == "missing":
        console.print(
            "Warning: stateful target has no fixture identity; add fixture_id and "
            "fixture_version so findings can be reproduced."
        )
    else:
        console.print("Fixture: not required for isolated-response target")


def print_evaluator_preflight(result: EvaluatorModelPreflight, receipt: Path) -> None:
    roles = ", ".join(role for profile in result.profiles for role in profile.roles)
    print_dataset_plain(
        f"Evaluator preflight passed for {len(result.profiles)} distinct profile(s): {roles}."
    )
    if result.unverified_options:
        print_dataset_plain(
            "Configured endpoint accepted every sample, but parameter support is unverified: "
            f"{', '.join(result.unverified_options)}."
        )
    else:
        print_dataset_plain("Required semantic parameters were verified by enforced routing.")
    data_policy_implication = result.data_policy.get("implication")
    if isinstance(data_policy_implication, str):
        print_dataset_plain(f"Evaluator data policy: {data_policy_implication}")
    print_dataset_plain(f"Evaluator preflight receipt: {receipt}")


def print_dataset_results(
    results: tuple[DatasetEvaluationResult, ...],
    output: Path,
    *,
    augmentations_output: Path | None = None,
    invariant_evaluations: tuple[DatasetInvariantEvaluation, ...] = (),
    show_report_guidance: bool = True,
) -> None:
    evaluation_modes = {getattr(result, "evaluation_mode", "variance") for result in results}
    if len(evaluation_modes) > 1:
        raise ValueError("dataset results contain incompatible evaluation modes")
    evaluation_mode = next(iter(evaluation_modes), "variance")
    console.print(
        f"Evaluation mode: {evaluation_mode} (historical output is not an expected answer; "
        "correctness not assessed)"
    )
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
            baseline_customer_status(result),
            result.baseline.trial_set.stability,
            trial_set_summary(result.baseline.trial_set),
            "—",
        )
        for case in result.cases:
            case_number += 1
            table.add_row(
                str(case_number),
                case.candidate.operator_id,
                case_customer_status(result, case),
                case.trial_set.stability if case.trial_set is not None else "—",
                trial_set_summary(case.trial_set),
                ", ".join(_FINDING_LABELS[finding.category] for finding in case.findings) or "—",
            )
    console.print(table)
    actual_semantic_calls = sum(
        getattr(getattr(result, "semantic_calls", None), "actual_calls", 0) for result in results
    )
    semantic_cache_hits = sum(
        getattr(getattr(result, "semantic_calls", None), "cache_hits", 0) for result in results
    )
    console.print(
        "Semantic evaluator calls: "
        f"{actual_semantic_calls} actual; {semantic_cache_hits} private cache hit(s)"
    )
    if invariant_evaluations:
        _print_invariant_results(invariant_evaluations)
    console.print(f"Complete evidence: {output}")
    if augmentations_output is not None:
        print_dataset_plain(f"Saved augmentations: {augmentations_output}")
    if show_report_guidance:
        console.print(f"Next: ul dataset report {output}")


def result_needs_review(result: DatasetEvaluationResult) -> bool:
    return any(case.verdict == "divergence_needs_review" for case in result.cases)


def dataset_invariant_exit_code(
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
    print_dataset_plain("")
    print_dataset_plain("Customer invariant evaluation")
    print_dataset_plain(
        "Terminal output shows pointers and reason codes only; transition rules do not retain "
        "selected state values."
    )
    for evaluation in evaluations:
        print_dataset_plain(f"Interaction: {evaluation.interaction_id}")
        print_dataset_plain(f"Declared observation authority: {evaluation.observation_authority}")
        for arm in (evaluation.baseline, *evaluation.variations):
            arm_name = "original" if arm.arm == "baseline" else f"variation ({arm.operator_id})"
            for rule in arm.rules:
                status_counts = {
                    status: sum(trial.status == status for trial in rule.trials)
                    for status in ("satisfied", "violated", "not_evaluable")
                }
                print_dataset_plain(
                    f"Rule {rule.rule_id} ({rule.rule_version}); severity={rule.severity}; "
                    f"arm={arm_name}; status={rule.status}; reason={rule.reason_code}; trials="
                    + ", ".join(f"{status}={count}" for status, count in status_counts.items())
                )
                print_dataset_plain(f"Description: {rule.description}")
                if rule.status == "violated":
                    print_dataset_plain(
                        "Customer rule violated against declared "
                        f"{evaluation.observation_authority}."
                    )
                for trial in rule.trials:
                    print_dataset_plain(
                        f"Trial {trial.repetition}: {trial.status}; "
                        f"{_invariant_trial_location(trial)}; "
                        f"reason={trial.reason_code}"
                    )


def _invariant_trial_location(
    trial: DatasetInvariantTrialEvaluation
    | DatasetInvariantValueEqualsTrialEvaluation
    | DatasetInvariantValueInSetTrialEvaluation
    | DatasetInvariantArrayUniqueTrialEvaluation
    | DatasetInvariantTransitionTrialEvaluation,
) -> str:
    if isinstance(trial, DatasetInvariantTrialEvaluation):
        return f"left={trial.left_pointer}; right={trial.right_pointer}"
    if isinstance(
        trial,
        (DatasetInvariantValueEqualsTrialEvaluation, DatasetInvariantValueInSetTrialEvaluation),
    ):
        return f"value={trial.value_pointer}"
    if isinstance(trial, DatasetInvariantTransitionTrialEvaluation):
        location = (
            f"before={trial.before_checkpoint}; after={trial.after_checkpoint}; "
            f"value={trial.observation_pointer}"
        )
        if trial.new_effect_count is not None:
            location += f"; new_effects={trial.new_effect_count}"
        return location
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
