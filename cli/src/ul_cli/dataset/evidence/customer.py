from __future__ import annotations

import hashlib
import json
from typing import Literal, cast

from pydantic import JsonValue
from ul import (
    DatasetEvaluationCase,
    DatasetEvaluationFinding,
    DatasetEvaluationResult,
    DatasetEvaluationTrialSet,
    DatasetTargetLifecycleFailure,
)
from ul.dataset_invariants import DatasetInvariantEvaluation
from ul_core.dataset import ObservedOutcome

from ul_cli.dataset_review import DatasetEvidenceRunContext

_CUSTOMER_STATUSES = {
    "augmentation_rejected": "VARIATION DISCARDED",
    "inconclusive": "COULDN'T DETERMINE",
    "no_divergence": "NO OBSERVED DIFFERENCE",
    "divergence_needs_review": "DIFFERENCE — REVIEW",
}
_BEHAVIORAL_LIMITATIONS = (
    "Evaluation mode is variance. UL compares fresh original replays with generated variations. "
    "Historical output is grounding evidence, not an expected answer. UL does not determine "
    "whether the original or variation is correct, prove that the variation caused a difference, "
    "or estimate a production failure rate."
)


def build_customer_evidence_record(
    result: DatasetEvaluationResult,
    *,
    repetitions: int,
    max_environment_api_calls: int,
    planned_target_calls: int,
    run_context: DatasetEvidenceRunContext | None = None,
    invariant_evaluation: DatasetInvariantEvaluation | None = None,
) -> dict[str, JsonValue]:
    evaluation_mode = cast(Literal["variance"], getattr(result, "evaluation_mode", "variance"))
    cases: list[JsonValue] = []
    for case in result.cases:
        cases.append(
            {
                "operator_id": case.candidate.operator_id,
                "operator_version": case.candidate.operator_version,
                "augmented_input": case.candidate.augmented_input,
                "status": case_customer_status(result, case),
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
    evidence: dict[str, JsonValue] = {
        "schema_version": "1.8.0",
        "evaluation_mode": evaluation_mode,
        "interaction_id": result.source.id,
        "original_input": result.source.raw_input,
        "execution_plan": {
            "repetitions": repetitions,
            "max_target_calls": max_environment_api_calls,
            "dataset_planned_target_calls": planned_target_calls,
        },
        "limitations": _BEHAVIORAL_LIMITATIONS,
        "current_baseline": {
            "status": baseline_customer_status(result),
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


def create_customer_evidence_record(
    result: DatasetEvaluationResult,
    *,
    repetitions: int,
    max_environment_api_calls: int,
    planned_target_calls: int,
) -> dict[str, JsonValue]:
    return build_customer_evidence_record(
        result,
        repetitions=repetitions,
        max_environment_api_calls=max_environment_api_calls,
        planned_target_calls=planned_target_calls,
    )


def baseline_customer_status(result: DatasetEvaluationResult) -> str:
    trial_set = result.baseline.trial_set
    stability = trial_set.stability
    if stability == "unstable":
        return "UNSTABLE ORIGINAL — INCONCLUSIVE"
    if stability == "inconclusive":
        return "COULDN'T DETERMINE"
    repetitions = trial_set.requested_repetitions
    return f"ORIGINAL REPLAY STABLE ({repetitions}/{repetitions} OBSERVED)"


def case_customer_status(result: DatasetEvaluationResult, case: DatasetEvaluationCase) -> str:
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


def trial_set_summary(trial_set: DatasetEvaluationTrialSet | None) -> str:
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
            "lifecycle_failure": _customer_lifecycle_failure(trial),
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


def _customer_lifecycle_failure(trial: object) -> JsonValue:
    lifecycle_failure = getattr(trial, "lifecycle_failure", None)
    if lifecycle_failure is None:
        return None
    return cast(DatasetTargetLifecycleFailure, lifecycle_failure).model_dump(mode="json")


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
