from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, cast

from pydantic import JsonValue
from ul import (
    DatasetEvaluationCase,
    DatasetEvaluationFinding,
    DatasetEvaluationResult,
    DatasetEvaluationTrialSet,
    DatasetSourcePreparationError,
    DatasetTargetLifecycleFailure,
    InteractionRecord,
)
from ul.dataset_evaluation import compare_observed_outcomes
from ul.dataset_invariants import DatasetInvariantEvaluation
from ul_core.dataset import ObservedOutcome

from ul_cli.dataset_review import (
    DatasetEvidenceRunContext,
    DatasetSourcePreparationFailureEvidence,
)
from ul_cli.report_contract import (
    CrossExaminationEvidenceAvailability,
    EvidenceAuthority,
    PatternVerticalFacets,
)

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


def build_source_preparation_failure_record(
    source: InteractionRecord,
    error: DatasetSourcePreparationError,
    *,
    repetitions: int,
    max_environment_api_calls: int,
    planned_target_calls: int,
    run_context: DatasetEvidenceRunContext,
) -> dict[str, JsonValue]:
    augmentation_target = getattr(source, "augmentation_target", None)
    evidence = DatasetSourcePreparationFailureEvidence(
        interaction_id=source.id,
        source_record_id=(
            source.source_interaction_id if augmentation_target is not None else None
        ),
        reason_code=cast(
            Literal[
                "source_semantic_preparation_failed",
                "source_comparison_surface_incompatible",
            ],
            error.code,
        ),
        summary=error.explanation,
        remediation=error.remediation,
        execution_plan=cast(
            Any,
            {
                "repetitions": repetitions,
                "max_target_calls": max_environment_api_calls,
                "dataset_planned_target_calls": planned_target_calls,
            },
        ),
        run_context=run_context,
    )
    return cast(dict[str, JsonValue], evidence.model_dump(mode="json", exclude_none=True))


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
    augmentation_target = getattr(result.source, "augmentation_target", None)
    pattern_facets = _customer_pattern_facets(getattr(result.source, "metadata", {}))
    cases: list[JsonValue] = []
    for case in result.cases:
        cases.append(
            {
                "operator_id": case.candidate.operator_id,
                "operator_version": case.candidate.operator_version,
                **(
                    {
                        "source_record_id": result.source.source_interaction_id,
                        "augmentation_target": augmentation_target.model_dump(mode="json"),
                        "original_value": result.source.raw_input,
                    }
                    if augmentation_target is not None
                    else {}
                ),
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
                "cross_examination": _customer_cross_examination(result, case),
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
        "schema_version": "1.14.0",
        "evaluation_mode": evaluation_mode,
        "interaction_id": result.source.id,
        **(
            {
                "pattern_facets": cast(
                    JsonValue, pattern_facets.model_dump(mode="json", exclude_none=True)
                )
            }
            if pattern_facets is not None
            else {}
        ),
        **(
            {
                "source_record_id": result.source.source_interaction_id,
                "augmentation_target": augmentation_target.model_dump(mode="json"),
            }
            if augmentation_target is not None
            else {}
        ),
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


def _customer_pattern_facets(metadata: dict[str, JsonValue]) -> PatternVerticalFacets | None:
    raw_facets = metadata.get("ul_pattern_facets")
    if raw_facets is None:
        return None
    return PatternVerticalFacets.model_validate(raw_facets)


def _customer_cross_examination(
    result: DatasetEvaluationResult,
    case: DatasetEvaluationCase,
) -> JsonValue:
    baseline_trial_set = result.baseline.trial_set
    variation_trial_set = case.trial_set
    baseline_incomplete = baseline_trial_set.stability == "inconclusive"
    comparison_incomplete = baseline_incomplete or (
        variation_trial_set is None or variation_trial_set.stability == "inconclusive"
    )
    unstable = baseline_trial_set.stability == "unstable" or (
        variation_trial_set is not None and variation_trial_set.stability == "unstable"
    )
    current_baseline_frame = getattr(baseline_trial_set, "representative_frame", None)
    augmentation = getattr(result, "augmentation", None)
    source_frames = getattr(augmentation, "source_frames", ())
    if baseline_incomplete or current_baseline_frame is None or not source_frames:
        baseline_drift = "inconclusive"
    else:
        historical_frame = source_frames[0]
        try:
            baseline_deltas = compare_observed_outcomes(
                historical_frame,
                current_baseline_frame,
                result.source.raw_input,
                grounding_frame=historical_frame,
                comparison_surface=result.comparison_surface,
            )
        except ValueError:
            baseline_drift = "inconclusive"
        else:
            baseline_drift = "observed" if baseline_deltas else "not_observed"
    augmentation_sensitivity = (
        "inconclusive"
        if comparison_incomplete or unstable
        else "observed"
        if case.findings
        else "not_observed"
    )
    intrinsic_instability = (
        "inconclusive" if comparison_incomplete else "observed" if unstable else "not_observed"
    )
    return cast(
        JsonValue,
        {
            "historical_reference_available": True,
            "current_baseline": _cross_examination_run_summary(baseline_trial_set),
            "variation": _cross_examination_run_summary(variation_trial_set),
            "baseline_drift": baseline_drift,
            "augmentation_sensitivity": augmentation_sensitivity,
            "intrinsic_instability": intrinsic_instability,
            "material_delta_count": len(case.findings),
            "response_evidence": _cross_examination_evidence_availability(
                "response", baseline_trial_set, variation_trial_set
            ).model_dump(mode="json"),
            "trajectory_evidence": _cross_examination_evidence_availability(
                "trajectory", baseline_trial_set, variation_trial_set
            ).model_dump(mode="json"),
            "committed_state_evidence": _cross_examination_evidence_availability(
                "committed_state", baseline_trial_set, variation_trial_set
            ).model_dump(mode="json"),
            "limitations": [
                "causality_not_established",
                "correctness_not_verified",
                "historical_reference_not_an_oracle",
            ],
        },
    )


def _cross_examination_evidence_availability(
    fact: Literal["response", "trajectory", "committed_state"],
    baseline: DatasetEvaluationTrialSet,
    variation: DatasetEvaluationTrialSet | None,
) -> CrossExaminationEvidenceAvailability:
    achieved = "verified" if fact == "committed_state" else "observed"

    def arm(
        trial_set: DatasetEvaluationTrialSet | None,
    ) -> tuple[Literal["observed", "verified", "unavailable"], int, set[EvidenceAuthority]]:
        if trial_set is None:
            return "unavailable", 0, set()
        authorities: set[EvidenceAuthority] = set()
        available: list[bool] = []
        for trial in trial_set.trials:
            execution_evidence = getattr(trial, "execution_evidence", None)
            if fact == "response":
                present = (
                    execution_evidence is not None and execution_evidence.final_response is not None
                ) or getattr(trial, "target_output", None) is not None
                if present:
                    authorities.add(
                        "source_self_reported"
                        if execution_evidence is not None
                        else "invoker_self_reported"
                    )
            elif fact == "trajectory":
                observations = execution_evidence.observations if execution_evidence else ()
                observed = tuple(
                    observation
                    for observation in observations
                    if any(
                        (
                            observation.traces,
                            observation.tool_calls,
                            observation.handoffs,
                            observation.errors,
                        )
                    )
                )
                present = any(observation.status == "complete" for observation in observed)
                authorities.update(observation.authority for observation in observed)
                if execution_evidence is not None and execution_evidence.execution_events:
                    present = True
                    authorities.add("invoker_self_reported")
            else:
                initial_state = (
                    execution_evidence.initial_state if execution_evidence is not None else None
                )
                final_state = (
                    execution_evidence.final_state if execution_evidence is not None else None
                )
                present = bool(
                    execution_evidence is not None
                    and execution_evidence.evidence_scope == "response_and_state"
                    and initial_state is not None
                    and final_state is not None
                )
                if present and initial_state is not None and final_state is not None:
                    authorities.update((initial_state.authority, final_state.authority))
            available.append(present)
        return (
            achieved if available and all(available) else "unavailable",
            sum(available),
            authorities,
        )

    baseline_status, baseline_covered, baseline_authorities = arm(baseline)
    variation_status, variation_covered, variation_authorities = arm(variation)
    return CrossExaminationEvidenceAvailability(
        fact=fact,
        conclusion=(
            achieved
            if baseline_status == achieved and variation_status == achieved
            else "unavailable"
        ),
        current_baseline=baseline_status,
        variation=variation_status,
        current_baseline_covered_repetitions=baseline_covered,
        variation_covered_repetitions=variation_covered,
        current_baseline_authorities=tuple(sorted(baseline_authorities)),
        variation_authorities=tuple(sorted(variation_authorities)),
    )


def _cross_examination_run_summary(trial_set: DatasetEvaluationTrialSet | None) -> JsonValue:
    if trial_set is None:
        return cast(
            JsonValue,
            {
                "requested_repetitions": 0,
                "observed_repetitions": 0,
                "inconclusive_repetitions": 0,
                "stability": "inconclusive",
            },
        )
    inconclusive = sum(bool(trial.inconclusive_reasons) for trial in trial_set.trials)
    return cast(
        JsonValue,
        {
            "requested_repetitions": trial_set.requested_repetitions,
            "observed_repetitions": trial_set.requested_repetitions - inconclusive,
            "inconclusive_repetitions": inconclusive,
            "stability": trial_set.stability,
        },
    )


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
            **(
                {"comparison_surface": "response"}
                if finding.category == "changed_response"
                else {
                    "reference_action_semantics": _normalized_outcome_semantics(
                        finding.expected_effects
                    ),
                    "observed_action_semantics": _normalized_outcome_semantics(
                        finding.observed_effects
                    ),
                }
            ),
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


def _normalized_outcome_semantics(effects: tuple[ObservedOutcome, ...]) -> list[JsonValue]:
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
