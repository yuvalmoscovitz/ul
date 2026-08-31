from __future__ import annotations

from ul.dataset_invariants import DatasetInvariantEvaluation, DatasetInvariantRuleResult


def is_reproduced_invariant_difference(
    baseline_rule: DatasetInvariantRuleResult,
    variation_rule: DatasetInvariantRuleResult,
) -> bool:
    baseline_repetitions = tuple(trial.repetition for trial in baseline_rule.trials)
    variation_repetitions = tuple(trial.repetition for trial in variation_rule.trials)
    return (
        baseline_rule.rule_id == variation_rule.rule_id
        and baseline_rule.rule_version == variation_rule.rule_version
        and baseline_rule.rule_type == variation_rule.rule_type
        and baseline_repetitions == variation_repetitions
        and bool(baseline_repetitions)
        and all(trial.status == "satisfied" for trial in baseline_rule.trials)
        and all(trial.status == "violated" for trial in variation_rule.trials)
    )


def reproduced_invariant_rule_pairs(
    evaluation: DatasetInvariantEvaluation,
    operator_id: str,
) -> tuple[tuple[DatasetInvariantRuleResult, DatasetInvariantRuleResult], ...]:
    baseline_rules = {rule.rule_id: rule for rule in evaluation.baseline.rules}
    return tuple(
        (baseline_rules[variation_rule.rule_id], variation_rule)
        for variation in evaluation.variations
        if variation.operator_id == operator_id
        for variation_rule in variation.rules
        if variation_rule.rule_id in baseline_rules
        and is_reproduced_invariant_difference(
            baseline_rules[variation_rule.rule_id], variation_rule
        )
    )
