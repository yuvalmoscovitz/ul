from __future__ import annotations

import hashlib
import json
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError
from typer.testing import CliRunner
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
from ul.augmentations.conversation import (
    CorrectionAfterFirstResponseCase,
    CorrectionDivergence,
    CorrectionStressResult,
    CorrectionStressTrial,
    CorrectionTurnObservation,
)
from ul.augmentations.dataset import (
    DatasetAugmentationCandidate,
    DatasetAugmentationOperatorReference,
)
from ul.dataset_evaluation import compare_action_outcomes
from ul.dataset_invariants import (
    DatasetInvariantArmEvaluation,
    DatasetInvariantEvaluation,
    DatasetInvariantRuleEvaluation,
    JsonValuesEqualInvariant,
)
from ul_cli import event_stress as event_stress_module
from ul_cli.event_stress import write_stateful_finding_packages
from ul_cli.finding_adapters import (
    FindingAdapterContext,
    adapt_dataset_behavior_finding,
    adapt_dataset_finding_packages,
    adapt_dataset_invariant_finding,
    adapt_stateful_invariant_finding,
)
from ul_cli.finding_reference import (
    create_finding_reference_context,
    finding_reference_key_path,
)
from ul_cli.main import app
from ul_cli.report_contract import (
    FindingEvidencePackage,
    RedactionReceipt,
    build_finding_decision,
    build_finding_occurrence,
    build_run_receipt,
    capture_json,
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
runner = CliRunner()


def _context() -> FindingAdapterContext:
    return FindingAdapterContext(
        campaign_id=f"campaign-{_PRIVATE_SECRET}",
        recorded_at=datetime(2026, 8, 23, 12, tzinfo=UTC),
        reference_key=b"test-only-reference-key-32bytes!",
    )


def _write_finding_evidence(path: Path, package: FindingEvidencePackage) -> None:
    context = _context()
    path.write_text(package.model_dump_json() + "\n", encoding="utf-8")
    key_path = finding_reference_key_path(path)
    key_path.write_text(
        context.reference_key.hex() + "\n" + context.recorded_at.isoformat() + "\n",
        encoding="ascii",
    )
    if sys.platform != "win32":
        key_path.chmod(0o600)


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
                    target_output=ObservedAgentOutput(
                        raw_output={
                            "status": output,
                            "amount": 100 if arm == "current_baseline" else 200,
                            "requested_amount": 100,
                        },
                        metadata={
                            "committed_state_snapshot": {
                                "amount": 100 if arm == "current_baseline" else 200,
                                "requested_amount": 100,
                            },
                            "state_observation_authority": "environment_self_reported",
                        },
                    ),
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


def _dataset_category_result(category: str) -> DatasetEvaluationResult:
    result = _dataset_result()
    source_action = ObservedOutcome(
        id="source-payment",
        confidence=1,
        status="observed",
        position=0,
        kind="action",
        predicate="create_payment",
        fields={"account": _PRIVATE_SECRET},
    )
    changed_action = source_action.model_copy(
        update={"id": "changed-payment", "fields": {"account": "another-account"}}
    )
    duplicate_action = source_action.model_copy(update={"id": "duplicate-payment", "position": 1})
    actions_by_category = {
        "unexpected_effect": ((), (source_action,)),
        "missing_effect": ((source_action,), ()),
        "changed_grounded_effect_argument": ((source_action,), (changed_action,)),
        "duplicate_effect": ((source_action,), (source_action, duplicate_action)),
    }
    source_actions, probe_actions = actions_by_category[category]

    def trial_set(arm: str, actions: tuple[ObservedOutcome, ...]) -> DatasetEvaluationTrialSet:
        return DatasetEvaluationTrialSet(
            requested_repetitions=2,
            stability="stable",
            trials=tuple(
                DatasetEvaluationTrial(
                    repetition=repetition,
                    target_output=ObservedAgentOutput(raw_output={"actions": []}),
                    observed_frame=SemanticFrame(
                        interaction_id=f"{result.source.id}:{arm}:round-{repetition}",
                        outcomes=actions,
                        extractor_version="test",
                    ),
                )
                for repetition in (1, 2)
            ),
            outcome_groups=(
                DatasetEvaluationOutcomeGroup(
                    repetitions=(1, 2),
                    representative_effects=actions,
                ),
            ),
        )

    grounding_frame = SemanticFrame(
        interaction_id=result.source.id,
        outcomes=source_actions,
        extractor_version="test",
    )
    source_frame = trial_set("current_baseline", source_actions).representative_frame
    probe_frame = trial_set(
        result.cases[0].candidate.operator_id, probe_actions
    ).representative_frame
    assert source_frame is not None and probe_frame is not None
    finding = next(
        finding
        for finding in compare_action_outcomes(
            source_frame,
            probe_frame,
            result.source.raw_input,
            grounding_frame=grounding_frame,
        )
        if finding.category == category
    )
    candidate = result.cases[0].candidate
    return result.model_copy(
        update={
            "augmentation": result.augmentation.model_copy(
                update={"source_frames": (grounding_frame,)}
            ),
            "baseline": DatasetEvaluationBaseline(
                verdict="no_divergence",
                trial_set=trial_set("current_baseline", source_actions),
            ),
            "cases": (
                DatasetEvaluationCase(
                    candidate=candidate,
                    verdict="divergence_needs_review",
                    trial_set=trial_set(candidate.operator_id, probe_actions),
                    findings=(finding,),
                ),
            ),
        }
    )


def _execution(case_id: str, turn_ids: tuple[str, ...], amount: int) -> ExecutionEvidence:
    reset = EnvironmentResetEvidence(
        reset_session_requested=True,
        reset_session_acknowledged=True,
        reset_env_requested=True,
        reset_env_acknowledged=True,
    )
    state: JsonValue = {
        "amount": amount,
        "requested_amount": 100,
        "private": _PRIVATE_SECRET,
    }
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
            value={
                "amount": 0,
                "requested_amount": 100,
                "private": _PRIVATE_SECRET,
            },
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
        committed_state: JsonValue = {"amount": amount, "requested_amount": 100}
        return CorrectionTurnObservation(
            turn=turn,
            target_output=ObservedAgentOutput(
                raw_output=committed_state,
                metadata={
                    "committed_state_snapshot": committed_state,
                    "state_observation_authority": "environment_self_reported",
                },
            ),
            committed_state_snapshot=committed_state,
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


def _dataset_invariant_evaluation() -> DatasetInvariantEvaluation:
    rule = _dataset_rule_evaluation()
    return DatasetInvariantEvaluation(
        interaction_id=f"source-{_PRIVATE_SECRET}",
        suite_sha256="c" * 64,
        observation_authority="committed_state_snapshot",
        baseline=DatasetInvariantArmEvaluation(
            arm="baseline",
            rules=(
                rule.model_copy(
                    update={
                        "status": "satisfied",
                        "reason_code": "all_trials_satisfied",
                        "trials": tuple(
                            trial.model_copy(
                                update={
                                    "status": "satisfied",
                                    "reason_code": "values_equal",
                                    "resolved_values": {"left": 100, "right": 100},
                                }
                            )
                            for trial in rule.trials
                        ),
                    }
                ),
            ),
        ),
        variations=(
            DatasetInvariantArmEvaluation(
                arm="variation",
                operator_id="input.surface.rephrase",
                rules=(rule,),
            ),
        ),
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
            _dataset_invariant_evaluation(),
            _rule(),
            case_index=0,
            context=_context(),
        ),
        adapt_stateful_invariant_finding(
            _stateful_result(),
            _rule(),
            observation_authority="committed_state_snapshot",
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
        observation_authority="committed_state_snapshot",
        context=_context(),
    )

    for package in (dataset_package, stateful_package):
        public_json = package.occurrence.model_dump_json()
        assert _PRIVATE_SECRET not in public_json
        assert "Pay account" not in public_json
        assert _PRIVATE_SECRET in package.model_dump_json()


@pytest.mark.parametrize(
    "category",
    (
        "unexpected_effect",
        "missing_effect",
        "changed_grounded_effect_argument",
        "duplicate_effect",
    ),
)
def test_behavior_categories_are_recomputed_from_each_exact_arm(category: str) -> None:
    package = adapt_dataset_behavior_finding(
        _dataset_category_result(category),
        case_index=0,
        finding_index=0,
        context=_context(),
    )
    pointers = {
        pointer.pointer_id: pointer
        for receipt in package.receipts
        for pointer in receipt.content.evidence_pointers
    }

    assert package.occurrence.category == category
    assert all(
        {pointers[pointer_id].arm for pointer_id in repetition.evidence_pointer_ids}
        == {"source", "probe"}
        for repetition in package.occurrence.repetitions
    )


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
        FindingEvidencePackage(
            occurrence=mismatched_occurrence,
            private_references=package.private_references,
            receipts=package.receipts,
        )


def test_stateful_adapter_rejects_a_rule_definition_from_another_authority() -> None:
    wrong_rule = _rule().model_copy(update={"id": "another-rule"})

    with pytest.raises(ValueError, match="one exact invariant rule"):
        adapt_stateful_invariant_finding(
            _stateful_result(),
            wrong_rule,
            observation_authority="committed_state_snapshot",
            context=_context(),
        )


def test_packages_embed_every_publicly_cited_private_artifact() -> None:
    package = adapt_dataset_invariant_finding(
        _dataset_result(),
        _dataset_invariant_evaluation(),
        _rule(),
        case_index=0,
        context=_context(),
    )
    artifacts = {artifact.artifact_sha256 for artifact in package.artifacts}
    pointers = {
        pointer.pointer_id: pointer
        for receipt in package.receipts
        for pointer in receipt.content.evidence_pointers
    }

    assert package.artifact_retention == "embedded"
    assert {
        pointers[pointer_id].artifact_sha256
        for pointer_id in package.occurrence.evidence_pointer_ids
    }.issubset(artifacts)


def test_private_receipt_disclosure_includes_cited_embedded_artifacts(tmp_path: Path) -> None:
    package = adapt_dataset_invariant_finding(
        _dataset_result(),
        _dataset_invariant_evaluation(),
        _rule(),
        case_index=0,
        context=_context(),
    )
    evidence = tmp_path / "evidence.findings.jsonl"
    _write_finding_evidence(evidence, package)
    safe_result = runner.invoke(app, ["report", str(evidence)])

    result = runner.invoke(
        app,
        [
            "report",
            str(evidence),
            "--show-sensitive-values",
            "--finding",
            package.occurrence.occurrence_id,
        ],
    )

    assert result.exit_code == 1, result.output
    assert safe_result.exit_code == 1, safe_result.output
    assert package.private_references.operator_id not in safe_result.output
    assert package.private_references.rule_id is not None
    assert package.private_references.rule_id not in safe_result.output
    assert '"artifacts": [' in result.output
    assert package.private_references.operator_id in result.output
    assert package.private_references.rule_id in result.output
    assert all(artifact.artifact_sha256 in result.output for artifact in package.artifacts)


def test_embedded_receipt_values_must_match_their_cited_artifacts() -> None:
    package = adapt_dataset_behavior_finding(
        _dataset_result(),
        case_index=0,
        finding_index=0,
        context=_context(),
    )
    original_receipt = package.receipts[0]
    changed_content = original_receipt.content.model_copy(
        update={
            "input": original_receipt.content.input.model_copy(
                update={"value": capture_json({"forged": _PRIVATE_SECRET})}
            )
        }
    )
    changed_receipt = build_run_receipt(changed_content)
    occurrence_values = package.occurrence.model_dump()
    occurrence_values.pop("occurrence_id")
    occurrence_values["repetitions"] = tuple(
        repetition.model_copy(
            update={
                "source_receipt_id": (
                    changed_receipt.receipt_id
                    if repetition.source_receipt_id == original_receipt.receipt_id
                    else repetition.source_receipt_id
                ),
                "probe_receipt_id": (
                    changed_receipt.receipt_id
                    if repetition.probe_receipt_id == original_receipt.receipt_id
                    else repetition.probe_receipt_id
                ),
            }
        )
        for repetition in package.occurrence.repetitions
    )
    changed_occurrence = build_finding_occurrence(**occurrence_values)
    changed_receipts = tuple(
        sorted(
            (
                changed_receipt if receipt.receipt_id == original_receipt.receipt_id else receipt
                for receipt in package.receipts
            ),
            key=lambda receipt: receipt.receipt_id,
        )
    )

    with pytest.raises(ValidationError, match="receipt value must match"):
        FindingEvidencePackage(
            occurrence=changed_occurrence,
            private_references=package.private_references,
            receipts=changed_receipts,
            artifact_retention="embedded",
            artifacts=package.artifacts,
        )


def test_opaque_references_resist_plain_hash_dictionary_guesses() -> None:
    package = adapt_dataset_behavior_finding(
        _dataset_result(),
        case_index=0,
        finding_index=0,
        context=_context(),
    )
    public_json = package.occurrence.model_dump_json()
    dictionary_guesses = (
        _PRIVATE_SECRET,
        f"source-{_PRIVATE_SECRET}",
        f"campaign-{_PRIVATE_SECRET}",
        "input.surface.rephrase",
        "1.0.0",
    )

    assert all(
        hashlib.sha256(value.encode()).hexdigest() not in public_json
        for value in dictionary_guesses
    )
    other_context = FindingAdapterContext(
        campaign_id=_context().campaign_id,
        recorded_at=_context().recorded_at,
        reference_key=b"a-different-private-reference-key-32-bytes",
    )
    other = adapt_dataset_behavior_finding(
        _dataset_result(),
        case_index=0,
        finding_index=0,
        context=other_context,
    )
    assert other.occurrence.case_ref != package.occurrence.case_ref


def test_dataset_invariant_adapter_preserves_exact_mixed_repetitions() -> None:
    result = _dataset_result()
    evaluation = _dataset_invariant_evaluation()
    probe_rule = evaluation.variations[0].rules[0]
    satisfied_trial = probe_rule.trials[1].model_copy(
        update={
            "status": "satisfied",
            "reason_code": "values_equal",
            "resolved_values": {"left": 100, "right": 100},
        }
    )
    mixed_probe_rule = probe_rule.model_copy(
        update={"trials": (probe_rule.trials[0], satisfied_trial)}
    )
    mixed_evaluation = evaluation.model_copy(
        update={
            "variations": (
                evaluation.variations[0].model_copy(update={"rules": (mixed_probe_rule,)}),
            )
        }
    )
    probe_trial_set = result.cases[0].trial_set
    assert probe_trial_set is not None
    satisfied_output = probe_trial_set.trials[1].target_output
    assert satisfied_output is not None
    satisfied_output = satisfied_output.model_copy(
        update={
            "raw_output": {
                "status": "changed",
                "amount": 100,
                "requested_amount": 100,
            },
            "metadata": {
                "committed_state_snapshot": {"amount": 100, "requested_amount": 100},
                "state_observation_authority": "environment_self_reported",
            },
        }
    )
    mixed_result = result.model_copy(
        update={
            "cases": (
                result.cases[0].model_copy(
                    update={
                        "trial_set": probe_trial_set.model_copy(
                            update={
                                "trials": (
                                    probe_trial_set.trials[0],
                                    probe_trial_set.trials[1].model_copy(
                                        update={"target_output": satisfied_output}
                                    ),
                                )
                            }
                        )
                    }
                ),
            )
        }
    )
    mixed_result = DatasetEvaluationResult.model_validate_json(mixed_result.model_dump_json())
    mixed_evaluation = DatasetInvariantEvaluation.model_validate_json(
        mixed_evaluation.model_dump_json()
    )

    package = adapt_dataset_invariant_finding(
        mixed_result,
        mixed_evaluation,
        _rule(),
        case_index=0,
        context=_context(),
    )
    pointers = {
        pointer.pointer_id: pointer
        for receipt in package.receipts
        for pointer in receipt.content.evidence_pointers
    }

    assert tuple(item.outcome for item in package.occurrence.repetitions) == (
        "finding_observed",
        "finding_not_observed",
    )
    assert package.occurrence.repetition_summary.reproducibility == "intermittent"
    assert all(
        {pointers[pointer_id].arm for pointer_id in repetition.evidence_pointer_ids}
        == {"source", "probe"}
        for repetition in package.occurrence.repetitions
    )


def test_invariant_adapter_rejects_same_identity_with_changed_rule_semantics() -> None:
    changed_rule = _rule().model_copy(update={"right_pointer": "/another_value"})

    with pytest.raises(ValueError, match="full customer rule definition"):
        adapt_dataset_invariant_finding(
            _dataset_result(),
            _dataset_invariant_evaluation(),
            changed_rule,
            case_index=0,
            context=_context(),
        )


def test_adapter_rejects_oversized_values_before_package_construction() -> None:
    result = _dataset_result()
    oversized_input = "x" * 260_000
    oversized = result.model_copy(
        update={"source": result.source.model_copy(update={"raw_input": oversized_input})}
    )

    with pytest.raises(ValueError, match="captured JSON exceeds"):
        adapt_dataset_behavior_finding(
            oversized,
            case_index=0,
            finding_index=0,
            context=_context(),
        )


def test_adapter_rejects_repetition_overflow_before_receipt_construction() -> None:
    result = _dataset_result()
    probe_trial_set = result.cases[0].trial_set
    assert probe_trial_set is not None
    baseline_trial = result.baseline.trial_set.trials[0]
    probe_trial = probe_trial_set.trials[0]
    baseline_trials = tuple(
        baseline_trial.model_copy(update={"repetition": repetition})
        for repetition in range(1, 1_002)
    )
    probe_trials = tuple(
        probe_trial.model_copy(update={"repetition": repetition}) for repetition in range(1, 1_002)
    )
    oversized = result.model_copy(
        update={
            "baseline": result.baseline.model_copy(
                update={
                    "trial_set": result.baseline.trial_set.model_copy(
                        update={"trials": baseline_trials}
                    )
                }
            ),
            "cases": (
                result.cases[0].model_copy(
                    update={
                        "trial_set": probe_trial_set.model_copy(update={"trials": probe_trials})
                    }
                ),
            ),
        }
    )

    with pytest.raises(ValueError, match="1,000 repetition limit"):
        adapt_dataset_behavior_finding(
            oversized,
            case_index=0,
            finding_index=0,
            context=_context(),
        )


def test_receipts_never_claim_false_zero_redaction_accounting() -> None:
    unavailable = adapt_dataset_behavior_finding(
        _dataset_result(), case_index=0, finding_index=0, context=_context()
    )
    accounted_context = FindingAdapterContext(
        campaign_id=_context().campaign_id,
        recorded_at=_context().recorded_at,
        reference_key=_context().reference_key,
        redaction=RedactionReceipt(
            policy_sha256="d" * 64,
            matched_value_count=2,
            redacted_value_count=2,
            retained_private_value_count=0,
        ),
    )
    accounted = adapt_dataset_behavior_finding(
        _dataset_result(), case_index=0, finding_index=0, context=accounted_context
    )

    assert all(
        receipt.content.redaction is None
        and "redaction_accounting_unavailable" in receipt.content.limitations
        for receipt in unavailable.receipts
    )
    assert all(
        receipt.content.redaction == accounted_context.redaction
        and "redaction_accounting_unavailable" not in receipt.content.limitations
        for receipt in accounted.receipts
    )


def test_initial_and_final_state_authorities_are_preserved_independently() -> None:
    result = _stateful_result()
    trial = result.trials[0]

    def independent_before(evidence: ExecutionEvidence) -> ExecutionEvidence:
        assert evidence.initial_state is not None
        return evidence.model_copy(
            update={
                "initial_state": EnvironmentStateEvidence(
                    value=evidence.initial_state.value,
                    authority="independent_observer",
                    observer_id="before-state-observer",
                )
            }
        )

    assert trial.baseline_execution_evidence is not None
    assert trial.variation_execution_evidence is not None
    result = result.model_copy(
        update={
            "trials": (
                trial.model_copy(
                    update={
                        "baseline_execution_evidence": independent_before(
                            trial.baseline_execution_evidence
                        ),
                        "variation_execution_evidence": independent_before(
                            trial.variation_execution_evidence
                        ),
                    }
                ),
            )
        }
    )
    package = adapt_stateful_invariant_finding(
        result,
        _rule(),
        observation_authority="committed_state_snapshot",
        context=_context(),
    )

    for receipt in package.receipts:
        state_pointers = {
            pointer.json_pointer: pointer
            for pointer in receipt.content.evidence_pointers
            if pointer.kind == "state"
        }
        assert state_pointers["/initial_state/value"].authority == "independent_observer"
        assert state_pointers["/initial_state/value"].source_id == "before-state-observer"
        assert state_pointers["/final_state/value"].authority == "environment_self_reported"


def test_dataset_mixed_missing_probe_repetition_remains_persistable() -> None:
    result = _dataset_result()
    probe_trial_set = result.cases[0].trial_set
    assert probe_trial_set is not None
    mixed_result = result.model_copy(
        update={
            "cases": (
                result.cases[0].model_copy(
                    update={
                        "verdict": "inconclusive",
                        "findings": (),
                        "inconclusive_reasons": ("target_output_missing",),
                        "trial_set": probe_trial_set.model_copy(
                            update={
                                "stability": "inconclusive",
                                "trials": (
                                    probe_trial_set.trials[0],
                                    probe_trial_set.trials[1].model_copy(
                                        update={
                                            "target_output": None,
                                            "observed_frame": None,
                                            "inconclusive_reasons": ("target_output_missing",),
                                        }
                                    ),
                                ),
                                "outcome_groups": (
                                    probe_trial_set.outcome_groups[0].model_copy(
                                        update={"repetitions": (1,)}
                                    ),
                                ),
                            }
                        ),
                    }
                ),
            )
        }
    )
    evaluation = _dataset_invariant_evaluation()
    probe_rule = evaluation.variations[0].rules[0]
    missing_trial = probe_rule.trials[1].model_copy(
        update={
            "status": "not_evaluable",
            "reason_code": "target_output_missing",
            "resolved_values": {},
        }
    )
    mixed_evaluation = evaluation.model_copy(
        update={
            "variations": (
                evaluation.variations[0].model_copy(
                    update={
                        "rules": (
                            probe_rule.model_copy(
                                update={"trials": (probe_rule.trials[0], missing_trial)}
                            ),
                        )
                    }
                ),
            )
        }
    )
    mixed_result = DatasetEvaluationResult.model_validate_json(mixed_result.model_dump_json())
    mixed_evaluation = DatasetInvariantEvaluation.model_validate_json(
        mixed_evaluation.model_dump_json()
    )

    packages = adapt_dataset_finding_packages(
        mixed_result,
        invariant_evaluation=mixed_evaluation,
        invariant_rules=(_rule(),),
        context=_context(),
    )

    assert len(packages) == 1
    package = packages[0]
    assert package.occurrence.kind == "customer_invariant_violation"
    assert package.occurrence.repetition_summary.inconclusive == 1
    assert package.occurrence.repetitions[1].outcome == "inconclusive"
    assert package.occurrence.repetitions[1].inconclusive_reason == "target_output_missing"
    assert build_finding_decision(package).classification == "customer_rule_violation"


def test_stateful_mixed_missing_arms_write_canonical_sidecar(tmp_path: Path) -> None:
    result = _stateful_result()
    missing_trial = CorrectionStressTrial(
        repetition=2,
        inconclusive_reason="environment execution timed out",
    )
    source_rule = result.baseline_invariant_rules[0]
    probe_rule = result.corrected_invariant_rules[0]
    source_missing = source_rule.trials[0].model_copy(
        update={
            "repetition": 2,
            "status": "not_evaluable",
            "reason_code": "target_output_missing",
            "resolved_values": {},
        }
    )
    probe_missing = probe_rule.trials[0].model_copy(
        update={
            "repetition": 2,
            "status": "not_evaluable",
            "reason_code": "target_output_missing",
            "resolved_values": {},
        }
    )
    mixed_result = result.model_copy(
        update={
            "requested_repetitions": 2,
            "required_target_calls": 6,
            "status": "inconclusive",
            "trials": (result.trials[0], missing_trial),
            "baseline_invariant_rules": (
                source_rule.model_copy(
                    update={
                        "status": "not_evaluable",
                        "reason_code": "one_or_more_trials_not_evaluable",
                        "trials": (source_rule.trials[0], source_missing),
                    }
                ),
            ),
            "corrected_invariant_rules": (
                probe_rule.model_copy(update={"trials": (probe_rule.trials[0], probe_missing)}),
            ),
        }
    )
    mixed_result = CorrectionStressResult.model_validate_json(mixed_result.model_dump_json())
    evidence_output = tmp_path / "mixed-correction.json"
    evidence_output.write_text("{}\n", encoding="utf-8")

    finding_output = write_stateful_finding_packages(
        evidence_output,
        mixed_result,
        (_rule(),),
        observation_authority="committed_state_snapshot",
    )
    (package,) = tuple(
        FindingEvidencePackage.model_validate_json(line)
        for line in finding_output.read_text(encoding="utf-8").splitlines()
    )

    assert tuple(item.outcome for item in package.occurrence.repetitions) == (
        "finding_observed",
        "inconclusive",
    )
    second_repetition = package.occurrence.repetitions[1]
    assert second_repetition.inconclusive_reason == "environment_execution_timed_out"
    assert "source_execution_unavailable" in package.occurrence.limitations
    unavailable_receipts = tuple(
        receipt for receipt in package.receipts if receipt.content.repetition == 2
    )
    assert len(unavailable_receipts) == 2
    assert all(receipt.content.response is None for receipt in unavailable_receipts)


def test_dataset_adapter_rejects_supplied_evaluation_that_disagrees_with_output() -> None:
    result = _dataset_result()
    probe_trial_set = result.cases[0].trial_set
    assert probe_trial_set is not None
    output = probe_trial_set.trials[0].target_output
    assert output is not None
    adversarial_result = result.model_copy(
        update={
            "cases": (
                result.cases[0].model_copy(
                    update={
                        "trial_set": probe_trial_set.model_copy(
                            update={
                                "trials": (
                                    probe_trial_set.trials[0].model_copy(
                                        update={
                                            "target_output": output.model_copy(
                                                update={
                                                    "raw_output": {
                                                        "amount": 999,
                                                        "requested_amount": 100,
                                                    }
                                                }
                                            )
                                        }
                                    ),
                                    probe_trial_set.trials[1],
                                )
                            }
                        )
                    }
                ),
            )
        }
    )
    response_evaluation = _dataset_invariant_evaluation().model_copy(
        update={"observation_authority": "agent_response"}
    )

    with pytest.raises(ValueError, match="exact captured evidence"):
        adapt_dataset_invariant_finding(
            adversarial_result,
            response_evaluation,
            _rule(),
            case_index=0,
            context=_context(),
        )


def test_stateful_adapter_rejects_supplied_evaluation_that_disagrees_with_state() -> None:
    result = _stateful_result()
    trial = result.trials[0]
    changed_observation = trial.variation[-1].model_copy(
        update={
            "target_output": trial.variation[-1].target_output.model_copy(
                update={
                    "metadata": {
                        "committed_state_snapshot": {
                            "amount": 999,
                            "requested_amount": 100,
                        },
                        "state_observation_authority": "environment_self_reported",
                    }
                }
            ),
            "committed_state_snapshot": {"amount": 999, "requested_amount": 100},
        }
    )
    adversarial_result = result.model_copy(
        update={
            "trials": (
                trial.model_copy(update={"variation": (trial.variation[0], changed_observation)}),
            )
        }
    )

    with pytest.raises(ValueError, match="exact captured evidence"):
        adapt_stateful_invariant_finding(
            adversarial_result,
            _rule(),
            observation_authority="committed_state_snapshot",
            context=_context(),
        )


def test_real_workflow_collectors_emit_canonical_packages_without_calls(tmp_path: Path) -> None:
    dataset_packages = adapt_dataset_finding_packages(
        _dataset_result(),
        invariant_evaluation=_dataset_invariant_evaluation(),
        invariant_rules=(_rule(),),
        context=_context(),
    )
    evidence_output = tmp_path / "correction.json"
    evidence_output.write_text("{}\n", encoding="utf-8")
    stateful_output = write_stateful_finding_packages(
        evidence_output,
        _stateful_result(),
        (_rule(),),
        observation_authority="committed_state_snapshot",
    )
    stateful_packages = tuple(
        FindingEvidencePackage.model_validate_json(line)
        for line in stateful_output.read_text(encoding="utf-8").splitlines()
    )

    assert len(dataset_packages) == 2
    assert len(stateful_packages) == 1
    assert all(package.artifact_retention == "embedded" for package in dataset_packages)
    assert all(package.artifact_retention == "embedded" for package in stateful_packages)
    assert stat.S_IMODE(stateful_output.stat().st_mode) == 0o600
    all_packages = (*dataset_packages, *stateful_packages)
    assert all(json.loads(package.model_dump_json()) for package in all_packages)


def test_stateful_command_reconciles_findings_from_durable_primary_evidence(
    tmp_path: Path,
) -> None:
    result = _stateful_result().model_copy(update={"required_target_calls": 14})
    output = tmp_path / "correction-result.json"
    output.write_text(result.model_dump_json(), encoding="utf-8")
    if sys.platform != "win32":
        output.chmod(0o600)
    case_path = tmp_path / "case.json"
    case_path.write_text(result.case.model_dump_json(), encoding="utf-8")
    invariants_path = tmp_path / "invariants.json"
    invariants_path.write_text(
        json.dumps(
            {
                "schema_version": "1.1.0",
                "observation_source": "target_output",
                "observation_authority": "committed_state_snapshot",
                "rules": [_rule().model_dump(mode="json")],
            }
        ),
        encoding="utf-8",
    )
    target_path = tmp_path / "target.json"
    target_path.write_text(
        json.dumps(
            {
                "version": 5,
                "environment_id": "reconciliation-test",
                "reset": {
                    "url": "http://127.0.0.1:8765/reset",
                    "generation_json_pointer": "/generation",
                    "clean_state_json_pointer": "/clean",
                    "clean_state_value": True,
                },
                "setup": {"url": "http://127.0.0.1:8765/setup"},
                "execute_turn": {
                    "url": "http://127.0.0.1:8765/execute",
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "turn_id": "{{turn_id}}",
                        "input": "{{input}}",
                    },
                },
                "snapshot": {
                    "url": "http://127.0.0.1:8765/snapshot",
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "turn_id": "{{turn_id}}",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    finding_output = event_stress_module.stateful_finding_output(output)
    reference_context = create_finding_reference_context(finding_output)

    command = [
        "stress",
        "correction",
        str(case_path),
        "--environment-config",
        str(target_path),
        "--invariants",
        str(invariants_path),
        "--repetitions",
        "1",
        "--confirm-test-environment",
        "--allow-insecure-http",
        "--output",
        str(output),
    ]
    command_result = runner.invoke(app, command)

    assert command_result.exit_code == 1, command_result.output
    assert finding_output.is_file()
    assert finding_reference_key_path(finding_output).is_file()
    assert (
        event_stress_module.resolve_finding_reference_context(finding_output) == reference_context
    )
    report_result = runner.invoke(app, ["report", str(finding_output), "--json"])
    assert report_result.exit_code == 1, report_result.output

    output.write_text(
        json.dumps({"schema_version": "1.1.0", "private": _PRIVATE_SECRET}),
        encoding="utf-8",
    )
    malformed_result = runner.invoke(app, command)
    assert malformed_result.exit_code == 2
    assert "existing private output" in malformed_result.output
    assert "safely" in malformed_result.output
    assert "reconciled" in malformed_result.output
    assert _PRIVATE_SECRET not in malformed_result.output


def test_stateful_finding_snapshot_failure_preserves_published_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finding_output = tmp_path / "evidence.json.findings.jsonl"
    finding_output.write_bytes(b"prior snapshot\n")

    def fail_write(descriptor: int, value: object) -> int:
        del descriptor, value
        raise OSError("simulated disk full")

    monkeypatch.setattr(event_stress_module.os, "write", fail_write)

    with pytest.raises(OSError, match="simulated disk full"):
        event_stress_module._replace_stateful_finding_snapshot(
            finding_output, b"replacement snapshot\n"
        )
    assert finding_output.read_bytes() == b"prior snapshot\n"
