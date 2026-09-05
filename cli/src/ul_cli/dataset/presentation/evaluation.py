from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

import typer
from rich.table import Table
from ul import DatasetEvaluationCase, DatasetEvaluationResult, EvaluatorModelPreflight
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
from ul_cli.invariant_findings import reproduced_invariant_rule_pairs

from ..evidence.customer import baseline_customer_status, case_customer_status, trial_set_summary
from .runtime import console, print_dataset_plain

_FINDING_LABELS = {
    "duplicate_effect": "duplicate action",
    "unexpected_effect": "unexpected action",
    "missing_effect": "missing action",
    "changed_grounded_effect_argument": "changed action value",
    "changed_response": "changed response",
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
    target_timeout_seconds: float,
    max_environment_api_calls: int,
    target_calls_per_execution: int,
    target_supports_state_observation: bool | None,
    fixture_status: Literal["configured", "missing", "not_required"] | None,
    fixture_id: str | None,
    fixture_version: str | None,
    invariant_suite: DatasetInvariantSuite | None,
    output: Path | None,
    augmentations_input: Path | None,
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
    console.print(f"Concurrent target requests: {campaign_plan.timing.target_request_concurrency}")
    console.print(f"Target trial timeout: {target_timeout_seconds:g} seconds")
    console.print(
        f"Maximum planned wall time: {campaign_plan.timing.maximum_wall_time_seconds:g} seconds"
    )
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
        f"evaluators={campaign_plan.calls.evaluators}, "
        f"materiality={campaign_plan.calls.materiality}"
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
    if augmentations_input is not None:
        print_dataset_plain(f"Reusing accepted augmentations: {augmentations_input}")
    elif augmentations_output is None:
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
    source_preparation_failure_count: int = 0,
) -> None:
    invariant_evaluations_by_interaction = {
        evaluation.interaction_id: evaluation for evaluation in invariant_evaluations
    }
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
            finding_labels = [_FINDING_LABELS[finding.category] for finding in case.findings]
            invariant_evaluation = invariant_evaluations_by_interaction.get(result.source.id)
            if invariant_evaluation is not None:
                finding_labels.extend(
                    f"reproduced invariant: {variation_rule.rule_id}"
                    for _, variation_rule in reproduced_invariant_rule_pairs(
                        invariant_evaluation, case.candidate.operator_id
                    )
                )
            table.add_row(
                str(case_number),
                case.candidate.operator_id,
                case_customer_status(result, case),
                case.trial_set.stability if case.trial_set is not None else "—",
                trial_set_summary(case.trial_set),
                ", ".join(finding_labels) or "—",
            )
    console.print(table)
    compared_variation_count = sum(
        case_has_completed_comparison(case) for result in results for case in result.cases
    )
    inconclusive_variation_count = sum(
        case.trial_set is not None and not case_has_completed_comparison(case)
        for result in results
        for case in result.cases
    )
    rejected_variation_count = sum(
        case.verdict == "augmentation_rejected" for result in results for case in result.cases
    )
    skipped_variation_count = sum(
        len(getattr(getattr(result, "augmentation", None), "skips", ())) for result in results
    )
    print_dataset_plain(
        f"Coverage: {compared_variation_count} "
        f"{'variation' if compared_variation_count == 1 else 'variations'} compared across "
        f"{len(results)} {'interaction' if len(results) == 1 else 'interactions'}; "
        f"{inconclusive_variation_count} inconclusive; {rejected_variation_count} rejected; "
        f"{skipped_variation_count} skipped."
    )
    selected_interactions_by_operator = Counter(
        reference.id
        for result in results
        for reference in getattr(getattr(result, "augmentation", None), "operator_references", ())
    )
    skip_groups = Counter(
        (
            skip.operator_id,
            skip.reason_code,
            skip.reason,
            skip.next_action,
        )
        for result in results
        for skip in getattr(getattr(result, "augmentation", None), "skips", ())
    )
    for (operator_id, _reason_code, reason, next_action), count in sorted(skip_groups.items()):
        selected_count = selected_interactions_by_operator[operator_id]
        print_dataset_plain(
            f"Warning: {operator_id} skipped for {count} of {selected_count} interactions: "
            f"{reason} Next: {next_action}"
        )
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
    if source_preparation_failure_count:
        console.print(
            f"Source preparation failures: {source_preparation_failure_count}; "
            "no target calls were made for those sources."
        )
    if invariant_evaluations:
        _print_invariant_results(invariant_evaluations)
    console.print(f"Complete evidence: {output}")
    finding_output = output.with_name(f"{output.name}.findings.jsonl")
    has_decision_ready_findings = finding_output.is_file() and finding_output.stat().st_size > 0
    if has_decision_ready_findings:
        print_dataset_plain(f"Decision-ready findings: {finding_output}")
    if augmentations_output is not None:
        print_dataset_plain(f"Saved augmentations: {augmentations_output}")
    if show_report_guidance:
        console.print(f"Next: ul dataset report {output}")
        if has_decision_ready_findings:
            console.print(f"Actionable finding export: ul report {finding_output}")


def result_needs_review(result: DatasetEvaluationResult) -> bool:
    return dataset_result_exit_code(result) == 1


def case_has_completed_comparison(case: DatasetEvaluationCase) -> bool:
    trial_set = case.trial_set
    return trial_set is not None and any(
        not trial.inconclusive_reasons for trial in trial_set.trials
    )


def dataset_result_exit_code(result: DatasetEvaluationResult) -> int:
    has_inconclusive_materiality = False
    for case in result.cases:
        if case.verdict != "divergence_needs_review":
            continue
        material_variance = getattr(case, "material_variance", None)
        if material_variance is None or material_variance.decision == "material_variance":
            return 1
        if material_variance.decision == "insufficient_evidence":
            has_inconclusive_materiality = True
    if has_inconclusive_materiality:
        return 2
    if not any(case_has_completed_comparison(case) for case in result.cases):
        return 2
    return 0


def dataset_results_exit_code(results: tuple[DatasetEvaluationResult, ...]) -> int:
    if not results:
        return 0
    compared_results = tuple(
        result
        for result in results
        if any(case_has_completed_comparison(case) for case in result.cases)
    )
    if not compared_results:
        return 2
    return max(
        (dataset_result_exit_code(result) for result in compared_results),
        key=lambda code: {0: 0, 2: 1, 1: 2}[code],
    )


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
        for variation in evaluation.variations:
            if variation.operator_id is None:
                continue
            for baseline_rule, variation_rule in reproduced_invariant_rule_pairs(
                evaluation, variation.operator_id
            ):
                print_dataset_plain(
                    "Reproduced invariant finding: "
                    f"rule={variation_rule.rule_id}; operator={variation.operator_id}; "
                    f"original satisfied={len(baseline_rule.trials)}/{len(baseline_rule.trials)}; "
                    f"variation violated={len(variation_rule.trials)}/"
                    f"{len(variation_rule.trials)}; authority={evaluation.observation_authority}"
                )
                print_dataset_plain(
                    "Finding limitations: causality not established; production prevalence not "
                    "measured; whole-task correctness not established."
                )
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
