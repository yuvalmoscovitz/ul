from __future__ import annotations

from ul.dataset_invariants import DatasetInvariantEvaluation
from ul_cli.invariant_findings import (
    is_reproduced_invariant_difference,
    reproduced_invariant_rule_pairs,
)

from ._factories import _invariant_evaluation


def _repeated_evaluation(
    baseline_second_status: str,
    variation_second_status: str,
) -> DatasetInvariantEvaluation:
    evaluation = _invariant_evaluation("satisfied", "violated")
    baseline_rule = evaluation.baseline.rules[0]
    variation_rule = evaluation.variations[0].rules[0]
    baseline_second = baseline_rule.trials[0].model_copy(
        update={"repetition": 2, "status": baseline_second_status}
    )
    variation_second = variation_rule.trials[0].model_copy(
        update={"repetition": 2, "status": variation_second_status}
    )
    if baseline_second_status == "violated":
        baseline_second = baseline_second.model_copy(
            update={"reason_code": "values_differ"}
        )
    if variation_second_status == "satisfied":
        variation_second = variation_second.model_copy(
            update={"reason_code": "values_equal"}
        )
    return evaluation.model_copy(
        update={
            "baseline": evaluation.baseline.model_copy(
                update={
                    "rules": (
                        baseline_rule.model_copy(
                            update={"trials": (*baseline_rule.trials, baseline_second)}
                        ),
                    )
                }
            ),
            "variations": (
                evaluation.variations[0].model_copy(
                    update={
                        "rules": (
                            variation_rule.model_copy(
                                update={"trials": (*variation_rule.trials, variation_second)}
                            ),
                        )
                    }
                ),
            ),
        }
    )


def test_reproduced_invariant_requires_every_aligned_trial_to_flip() -> None:
    reproduced = _repeated_evaluation("satisfied", "violated")
    intermittent_probe = _repeated_evaluation("satisfied", "satisfied")
    unstable_source = _repeated_evaluation("violated", "violated")

    assert is_reproduced_invariant_difference(
        reproduced.baseline.rules[0], reproduced.variations[0].rules[0]
    )
    assert reproduced_invariant_rule_pairs(reproduced, "input.surface.rephrase")
    assert not reproduced_invariant_rule_pairs(
        intermittent_probe, "input.surface.rephrase"
    )
    assert not reproduced_invariant_rule_pairs(unstable_source, "input.surface.rephrase")
