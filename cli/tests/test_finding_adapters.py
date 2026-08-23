from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import JsonValue, ValidationError
from ul import (
    DatasetAugmentationResult,
    DatasetEvaluationBaseline,
    DatasetEvaluationCase,
    DatasetEvaluationFinding,
    DatasetEvaluationOutcomeGroup,
    DatasetEvaluationResult,
    DatasetEvaluationTrial,
    DatasetEvaluationTrialSet,
    InteractionRecord,
    ObservedAgentOutput,
    SemanticFrame,
)
from ul.dataset_augmentation import (
    DatasetAugmentationCandidate,
    DatasetAugmentationOperatorReference,
)
from ul.dataset_invariants import (
    DatasetInvariantRuleEvaluation,
    JsonValuesEqualInvariant,
)
from ul.event_stress import (
    CorrectionAfterFirstResponseCase,
    CorrectionDivergence,
    CorrectionStressResult,
    CorrectionStressTrial,
    CorrectionTurnObservation,
)
from ul_cli.finding_adapters import (
    FindingAdapterContext,
    adapt_dataset_behavior_finding,
    adapt_dataset_invariant_finding,
    adapt_stateful_invariant_finding,
)
from ul_cli.report_contract import (
    FindingEvidencePackage,
    build_finding_occurrence,
    serialize_run_receipt,
)
from ul_core.dataset import ObservedOutcome
from ul_core.evaluation import (
    EnvironmentLifecycleEvidence,
    EnvironmentResetEvidence,
    EnvironmentStateEvidence,
    EnvironmentTurnEvidence,
    ExecutionEvidence,
)
from ul_core.models import ConversationRole, ConversationTurn

_PRIVATE_SECRET = "private-customer-secret-871"
_SHA256 = "a" * 64


def _context() -> FindingAdapterContext:
    return FindingAdapterContext(
        campaign_id=f"campaign-{_PRIVATE_SECRET}",
        recorded_at=datetime(2026, 8, 23, 12, tzinfo=UTC),
        redaction_policy_sha256="b" * 64,
    )


def _dataset_result() -> DatasetEvaluationResult:
    source = InteractionRecord(
        id=f"source-{_PRIVATE_SECRET}",
        raw_input=f"Pay account {_PRIVATE_SECRET}.",
        raw_observed_output={"status": "ok"},
    )
    source_frame = SemanticFrame(interaction_id=source.id, extractor_version="test")
    candidate = DatasetAugmentationCandidate(
        source_interaction_id=source.id,
        operator_id="input.surface.rephrase",
        operator_version="1.0.0",
        augmented_input=f"Please pay account {_PRIVATE_SECRET}.",
        expected_input_frame=source_frame,
        reparsed_input_frame=source_frame,
        passed=True,
    )
    added_action = ObservedOutcome(
        id="created-payment",
        confidence=1,
        status="observed",
        position=0,
        kind="action",
        predicate="create_payment",
        fields={"account": _PRIVATE_SECRET},
    )

    def trial_set(
        arm: str,
        output: str,
        outcomes: tuple[ObservedOutcome, ...] = (),
    ) -> DatasetEvaluationTrialSet:
        return DatasetEvaluationTrialSet(
            requested_repetitions=2,
            stability="stable",
            trials=tuple(
                DatasetEvaluationTrial(
                    repetition=repetition,
                    target_output=ObservedAgentOutput(raw_output={"status": output}),
                    observed_frame=SemanticFrame(
                        interaction_id=f"{source.id}:{arm}:round-{repetition}",
                        outcomes=outcomes,
                        extractor_version="test",
                    ),
                )
                for repetition in (1, 2)
            ),
            outcome_groups=(
                DatasetEvaluationOutcomeGroup(
                    repetitions=(1, 2),
                    representative_effects=outcomes,
                ),
            ),
        )

    return DatasetEvaluationResult(
        source=source,
        augmentation=DatasetAugmentationResult(
            operator_references=(
                DatasetAugmentationOperatorReference(
                    id=candidate.operator_id,
                    version="1.0.0",
                ),
            ),
            source_records=(source,),
            source_frames=(source_frame,),
            candidates=(candidate,),
        ),
        baseline=DatasetEvaluationBaseline(
            verdict="no_divergence",
            trial_set=trial_set("current_baseline", "baseline"),
        ),
        cases=(
            DatasetEvaluationCase(
                candidate=candidate,
                verdict="divergence_needs_review",
                trial_set=trial_set(candidate.operator_id, "changed", (added_action,)),
                findings=(
                    DatasetEvaluationFinding(
                        category="unexpected_effect",
                        message="The probe added an action.",
                        observed_effects=(added_action,),
                    ),
                ),
            ),
        ),
    )


def _execution(case_id: str, turn_ids: tuple[str, ...], amount: int) -> ExecutionEvidence:
    reset = EnvironmentResetEvidence(
        reset_session_requested=True,
        reset_session_acknowledged=True,
        reset_env_requested=True,
        reset_env_acknowledged=True,
    )
    state: JsonValue = {"amount": amount, "private": _PRIVATE_SECRET}
    turns = tuple(
        EnvironmentTurnEvidence(
            turn_id=turn_id,
            response={"status": "ok", "amount": amount},
            state_snapshot=state,
            state_observation_authority="environment_self_reported",
        )
        for turn_id in turn_ids
    )
    return ExecutionEvidence(
        case_id=case_id,
        environment_id="payments-environment",
        environment_config_sha256=_SHA256,
        initial_state=EnvironmentStateEvidence(
            value={"amount": 0, "private": _PRIVATE_SECRET},
            authority="environment_self_reported",
        ),
        turns=turns,
        final_response=turns[-1].response,
        final_state=EnvironmentStateEvidence(
            value=state,
            authority="environment_self_reported",
        ),
        lifecycle=EnvironmentLifecycleEvidence(
            terminal_status="succeeded",
            completed_phases=("initial_reset", "execute_turn", "cleanup_reset"),
            delivery="certain",
            cleanup="succeeded",
            environment_state_uncertain=False,
            initial_reset=reset,
            cleanup_reset=reset,
        ),
    )


def _rule_evaluation(status: str, amount: int) -> DatasetInvariantRuleEvaluation:
    return DatasetInvariantRuleEvaluation.model_validate(
        {
            "rule_type": "json_values_equal",
            "rule_id": "amount-matches-request",
            "rule_version": "1.0.0",
            "description": "The committed amount must match the request.",
            "severity": "critical",
            "status": status,
            "reason_code": (
                "all_trials_satisfied" if status == "satisfied" else "one_or_more_trials_violated"
            ),
            "trials": (
                {
                    "repetition": 1,
                    "status": status,
                    "reason_code": "values_equal" if status == "satisfied" else "values_differ",
                    "left_pointer": "/amount",
                    "right_pointer": "/requested_amount",
                    "resolved_values": {
                        "left": amount,
                        "right": 100,
                    },
                },
            ),
        }
    )


def _stateful_result() -> CorrectionStressResult:
    first_turn = ConversationTurn(
        id="initial-payment",
        role=ConversationRole.USER,
        content=f"Pay 100 to {_PRIVATE_SECRET}.",
    )
    correction_turn = ConversationTurn(
        id="corrected-payment",
        role=ConversationRole.USER,
        content="Correction: pay 200.",
    )
    case = CorrectionAfterFirstResponseCase(
        id=f"correction-{_PRIVATE_SECRET}",
        conversation=(first_turn, correction_turn),
    )

    def observation(turn: ConversationTurn, amount: int) -> CorrectionTurnObservation:
        return CorrectionTurnObservation(
            turn=turn,
            target_output=ObservedAgentOutput(raw_output={"amount": amount}),
            committed_state_snapshot={"amount": amount},
        )

    divergence = CorrectionDivergence(
        variation_turn_id=correction_turn.id,
        compared_baseline_turn_id=first_turn.id,
        baseline_response={"amount": 100},
        variation_response={"amount": 200},
        baseline_committed_state={"amount": 100},
        variation_committed_state={"amount": 200},
        response_diverged=True,
        committed_state_diverged=True,
        response_changed_from_previous_turn=True,
        committed_state_changed_from_previous_turn=True,
    )
    trial = CorrectionStressTrial(
        repetition=1,
        baseline_execution_evidence=_execution(case.id, (first_turn.id,), 100),
        variation_execution_evidence=_execution(
            case.id,
            (first_turn.id, correction_turn.id),
            200,
        ),
        baseline=(observation(first_turn, 100),),
        variation=(observation(first_turn, 100), observation(correction_turn, 200)),
        divergences=(divergence, divergence),
    )
    return CorrectionStressResult(
        case=case,
        requested_repetitions=1,
        required_target_calls=3,
        status="failed",
        first_response_divergence_turn_id=correction_turn.id,
        first_committed_state_divergence_turn_id=correction_turn.id,
        response_divergence_stability="stable",
        committed_state_divergence_stability="stable",
        response_divergence_counts={first_turn.id: 0, correction_turn.id: 1},
        committed_state_divergence_counts={first_turn.id: 0, correction_turn.id: 1},
        baseline_drift_observed=False,
        trials=(trial,),
        baseline_invariant_rules=(_rule_evaluation("satisfied", 100),),
        corrected_invariant_rules=(_rule_evaluation("violated", 200),),
    )


def _rule() -> JsonValuesEqualInvariant:
    return JsonValuesEqualInvariant(
        type="json_values_equal",
        id="amount-matches-request",
        version="1.0.0",
        description="The committed amount must match the request.",
        severity="critical",
        left_pointer="/amount",
        right_pointer="/requested_amount",
    )


def _dataset_rule_evaluation() -> DatasetInvariantRuleEvaluation:
    evaluation = _rule_evaluation("violated", 200)
    trial = evaluation.trials[0]
    return evaluation.model_copy(
        update={
            "trials": (
                trial,
                trial.model_copy(update={"repetition": 2}),
            )
        }
    )


def test_dataset_and_stateful_adapters_emit_the_same_validated_package() -> None:
    packages = (
        adapt_dataset_behavior_finding(
            _dataset_result(),
            case_index=0,
            finding_index=0,
            context=_context(),
        ),
        adapt_dataset_invariant_finding(
            _dataset_result(),
            _dataset_rule_evaluation(),
            _rule(),
            case_index=0,
            context=_context(),
        ),
        adapt_stateful_invariant_finding(
            _stateful_result(),
            _rule(),
            context=_context(),
        ),
    )

    for package in packages:
        assert FindingEvidencePackage.model_validate_json(package.model_dump_json()) == package
        assert tuple(receipt.receipt_id for receipt in package.receipts) == tuple(
            sorted(receipt.receipt_id for receipt in package.receipts)
        )
        assert all(
            serialize_run_receipt(receipt) == serialize_run_receipt(receipt)
            for receipt in package.receipts
        )


def test_public_occurrences_do_not_disclose_private_workflow_values() -> None:
    dataset_package = adapt_dataset_behavior_finding(
        _dataset_result(),
        case_index=0,
        finding_index=0,
        context=_context(),
    )
    stateful_package = adapt_stateful_invariant_finding(
        _stateful_result(),
        _rule(),
        context=_context(),
    )

    for package in (dataset_package, stateful_package):
        public_json = package.occurrence.model_dump_json()
        assert _PRIVATE_SECRET not in public_json
        assert "Pay account" not in public_json
        assert _PRIVATE_SECRET in package.model_dump_json()


def test_adapters_bind_every_repetition_to_its_exact_arm_receipts() -> None:
    package = adapt_dataset_behavior_finding(
        _dataset_result(),
        case_index=0,
        finding_index=0,
        context=_context(),
    )
    occurrence_values = package.occurrence.model_dump()
    occurrence_values.pop("occurrence_id")
    repetitions = package.occurrence.repetitions
    occurrence_values["repetitions"] = (
        repetitions[0].model_copy(update={"probe_receipt_id": repetitions[1].probe_receipt_id}),
        repetitions[1],
    )
    mismatched_occurrence = build_finding_occurrence(**occurrence_values)

    with pytest.raises(ValidationError, match="exactly match repetition references"):
        FindingEvidencePackage(occurrence=mismatched_occurrence, receipts=package.receipts)


def test_stateful_adapter_rejects_a_rule_definition_from_another_authority() -> None:
    wrong_rule = _rule().model_copy(update={"id": "another-rule"})

    with pytest.raises(ValueError, match="one exact invariant rule"):
        adapt_stateful_invariant_finding(
            _stateful_result(),
            wrong_rule,
            context=_context(),
        )
