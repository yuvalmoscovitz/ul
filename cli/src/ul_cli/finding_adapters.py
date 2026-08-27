from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, cast

from pydantic import BaseModel, JsonValue
from ul import DatasetEvaluationResult
from ul.augmentations.conversation import (
    CorrectionStressResult,
    RetryAfterSuccessfulCommitStressResult,
)
from ul.dataset_evaluation import (
    DatasetEvaluationCase,
    DatasetEvaluationFinding,
    DatasetEvaluationTrial,
    compare_observed_outcomes,
)
from ul.dataset_invariants import (
    DatasetInvariantEvaluation,
    DatasetInvariantRule,
    DatasetInvariantRuleResult,
    evaluate_dataset_invariant_rule_trials,
    evaluate_dataset_invariant_rules,
)
from ul_core.dataset import ObservedAgentOutput, ObservedOutcome, SemanticFrame
from ul_core.evaluation import (
    EnvironmentStateEvidence,
    ExecutionEvidence,
    ProbeExecutionEvent,
    ProbeObservation,
)

from ul_cli.finding_reference import finding_public_reference
from ul_cli.report_contract import (
    CapturedJson,
    CrossExaminationArm,
    CrossExaminationEvidenceAvailability,
    EvidenceArtifact,
    EvidencePointer,
    FindingCategory,
    FindingCrossExamination,
    FindingEvidencePackage,
    FindingPrivateReferences,
    FindingRepetition,
    LifecycleReceipt,
    ObservedDelta,
    ProbeChange,
    ProvenanceReceipt,
    ReceiptEvidenceValue,
    RedactionReceipt,
    RepetitionSummary,
    RunReceipt,
    RunReceiptContent,
    StateReceipt,
    VersionedReference,
    build_finding_occurrence,
    build_run_receipt,
)

StatefulFindingResult = CorrectionStressResult | RetryAfterSuccessfulCommitStressResult
EvidencePointerKind = Literal[
    "input",
    "response",
    "action",
    "rule",
    "state",
    "tool_call",
    "tool_result",
    "lifecycle",
    "trace",
]
EvidenceAuthority = Literal[
    "customer_declared",
    "deterministic_evaluator",
    "invoker_self_reported",
    "source_self_reported",
    "environment_self_reported",
    "independent_observer",
]
_MAXIMUM_CAPTURED_JSON_BYTES = 250_000
_MAXIMUM_STATE_CAPTURED_JSON_BYTES = 1_000_000
_MAXIMUM_RUN_RECEIPT_BYTES = 1_000_000
_MAXIMUM_ADAPTED_PACKAGE_BYTES = 16_000_000
_MAXIMUM_ADAPTED_REPETITIONS = 1_000


@dataclass(frozen=True)
class FindingAdapterContext:
    campaign_id: str
    recorded_at: datetime
    reference_key: bytes
    redaction: RedactionReceipt | None = None
    customer_source_id: str = "customer-invariant-suite"
    evaluator_source_id: str = "ul-deterministic-evaluator"
    invoker_source_id: str = "ul-active-probe"

    def __post_init__(self) -> None:
        if len(self.reference_key) < 32:
            raise ValueError("opaque reference keys must contain at least 32 random bytes")


@dataclass(frozen=True)
class _ReceiptBuild:
    receipt: RunReceipt
    input_pointer_id: str
    response_pointer_id: str | None
    historical_reference_pointer_id: str | None
    category_pointer_ids: tuple[str, ...]
    rule_definition_pointer_id: str | None = None
    artifacts: tuple[EvidenceArtifact, ...] = ()
    execution_available: bool = True


@dataclass(frozen=True)
class _BehaviorRepetitionEvidence:
    source_value: JsonValue | None
    probe_value: JsonValue | None
    outcome: Literal["finding_observed", "finding_not_observed", "inconclusive"]
    inconclusive_reason: str | None = None


def adapt_dataset_behavior_finding(
    result: DatasetEvaluationResult,
    *,
    case_index: int,
    finding_index: int,
    context: FindingAdapterContext,
) -> FindingEvidencePackage:
    case = result.cases[case_index]
    finding = case.findings[finding_index]
    trial_set = _require_dataset_trial_set(case)
    baseline_trials = result.baseline.trial_set.trials
    _validate_repetition_bound(baseline_trials, trial_set.trials)
    source_receipts: list[_ReceiptBuild] = []
    probe_receipts: list[_ReceiptBuild] = []
    repetition_evidence: list[_BehaviorRepetitionEvidence] = []
    for receipt_index, (baseline_trial, probe_trial) in enumerate(
        zip(baseline_trials, trial_set.trials, strict=True)
    ):
        evidence = _behavior_repetition_evidence(
            finding,
            baseline_trial,
            probe_trial,
            result.source.raw_input,
            result.augmentation.source_frames[0],
            result.comparison_surface,
        )
        repetition_evidence.append(evidence)
        source_receipts.append(
            _dataset_receipt(
                repetition=baseline_trial.repetition,
                arm="source",
                input_value=result.source.raw_input,
                target_output=baseline_trial.target_output,
                execution_evidence=baseline_trial.execution_evidence,
                category=finding.category if evidence.source_value is not None else None,
                category_value=evidence.source_value,
                context=context,
                historical_reference_value=result.source.raw_observed_output,
                historical_reference_present=receipt_index == 0,
            )
        )
        probe_receipts.append(
            _dataset_receipt(
                repetition=probe_trial.repetition,
                arm="probe",
                input_value=case.candidate.augmented_input,
                target_output=probe_trial.target_output,
                execution_evidence=probe_trial.execution_evidence,
                category=finding.category if evidence.probe_value is not None else None,
                category_value=evidence.probe_value,
                context=context,
            )
        )
    return _build_behavior_package(
        category=finding.category,
        campaign_id=context.campaign_id,
        case_id=result.source.id,
        source_interaction_id=result.source.id,
        operator_id=case.candidate.operator_id,
        operator_version=case.candidate.operator_version,
        source_receipts=source_receipts,
        probe_receipts=probe_receipts,
        repetition_evidence=repetition_evidence,
        baseline_stability=result.baseline.trial_set.stability,
        variation_stability=trial_set.stability,
        baseline_drift=_baseline_drift_signal(result),
        context=context,
    )


def adapt_dataset_finding_packages(
    result: DatasetEvaluationResult,
    *,
    invariant_evaluation: DatasetInvariantEvaluation | None,
    invariant_rules: tuple[DatasetInvariantRule, ...],
    context: FindingAdapterContext,
) -> tuple[FindingEvidencePackage, ...]:
    packages: list[FindingEvidencePackage] = []
    rules_by_id = {rule.id: rule for rule in invariant_rules}
    variation_rules_by_operator = (
        {
            arm.operator_id: arm.rules
            for arm in invariant_evaluation.variations
            if arm.operator_id is not None
        }
        if invariant_evaluation is not None
        else {}
    )
    for case_index, case in enumerate(result.cases):
        packages.extend(
            adapt_dataset_behavior_finding(
                result,
                case_index=case_index,
                finding_index=finding_index,
                context=context,
            )
            for finding_index in range(len(case.findings))
        )
        for rule_evaluation in variation_rules_by_operator.get(case.candidate.operator_id, ()):
            if rule_evaluation.status != "violated":
                continue
            rule_definition = rules_by_id.get(rule_evaluation.rule_id)
            if rule_definition is None or invariant_evaluation is None:
                raise ValueError("violated invariant evidence requires its customer rule")
            packages.append(
                adapt_dataset_invariant_finding(
                    result,
                    invariant_evaluation,
                    rule_definition,
                    case_index=case_index,
                    context=context,
                )
            )
    return tuple(packages)


def adapt_dataset_invariant_finding(
    result: DatasetEvaluationResult,
    evaluation: DatasetInvariantEvaluation,
    rule_definition: DatasetInvariantRule,
    *,
    case_index: int,
    context: FindingAdapterContext,
) -> FindingEvidencePackage:
    case = result.cases[case_index]
    if evaluation.interaction_id != result.source.id:
        raise ValueError("dataset invariant evaluation must match its source interaction")
    source_evaluation, probe_evaluation = _dataset_rule_pair(
        evaluation,
        case.candidate.operator_id,
        rule_definition,
    )
    variation_index = next(
        index
        for index, arm in enumerate(evaluation.variations)
        if arm.operator_id == case.candidate.operator_id
    )
    baseline_rule_index = evaluation.baseline.rules.index(source_evaluation)
    probe_rule_index = evaluation.variations[variation_index].rules.index(probe_evaluation)
    full_evaluation_value = cast(JsonValue, evaluation.model_dump(mode="json"))
    trial_set = _require_dataset_trial_set(case)
    baseline_trials = result.baseline.trial_set.trials
    _validate_repetition_bound(
        baseline_trials,
        trial_set.trials,
        source_evaluation.trials,
        probe_evaluation.trials,
    )
    _require_dataset_recomputed_rule(
        rule_definition,
        baseline_trials,
        evaluation.observation_authority,
        source_evaluation,
    )
    _require_dataset_recomputed_rule(
        rule_definition,
        trial_set.trials,
        evaluation.observation_authority,
        probe_evaluation,
    )
    source_receipts: list[_ReceiptBuild] = []
    probe_receipts: list[_ReceiptBuild] = []
    for baseline_trial, probe_trial, source_rule_trial, probe_rule_trial in zip(
        baseline_trials,
        trial_set.trials,
        source_evaluation.trials,
        probe_evaluation.trials,
        strict=True,
    ):
        source_receipts.append(
            _dataset_receipt(
                repetition=baseline_trial.repetition,
                arm="source",
                input_value=result.source.raw_input,
                target_output=baseline_trial.target_output,
                execution_evidence=baseline_trial.execution_evidence,
                category="customer_invariant_violation",
                category_value=full_evaluation_value,
                category_json_pointer=(
                    f"/baseline/rules/{baseline_rule_index}/trials/"
                    f"{source_rule_trial.repetition - 1}"
                ),
                context=context,
                rule_definition=rule_definition if baseline_trial.repetition == 1 else None,
            )
        )
        probe_receipts.append(
            _dataset_receipt(
                repetition=probe_trial.repetition,
                arm="probe",
                input_value=case.candidate.augmented_input,
                target_output=probe_trial.target_output,
                execution_evidence=probe_trial.execution_evidence,
                category="customer_invariant_violation",
                category_value=full_evaluation_value,
                category_json_pointer=(
                    f"/variations/{variation_index}/rules/{probe_rule_index}/trials/"
                    f"{probe_rule_trial.repetition - 1}"
                ),
                context=context,
            )
        )
    return _build_invariant_package(
        campaign_id=context.campaign_id,
        case_id=result.source.id,
        source_interaction_id=result.source.id,
        operator_id=case.candidate.operator_id,
        operator_version=case.candidate.operator_version,
        source_evaluation=source_evaluation,
        probe_evaluation=probe_evaluation,
        rule_definition=rule_definition,
        source_receipts=source_receipts,
        probe_receipts=probe_receipts,
        context=context,
    )


def adapt_stateful_invariant_finding(
    result: StatefulFindingResult,
    rule_definition: DatasetInvariantRule,
    *,
    observation_authority: Literal["agent_response", "committed_state_snapshot"],
    context: FindingAdapterContext,
) -> FindingEvidencePackage:
    source_evaluation, probe_evaluation = _stateful_rule_pair(result, rule_definition)
    _validate_repetition_bound(
        result.trials,
        source_evaluation.trials,
        probe_evaluation.trials,
    )
    source_outputs, probe_outputs = _stateful_rule_outputs(result)
    _require_stateful_recomputed_rule(
        rule_definition, source_outputs, observation_authority, source_evaluation
    )
    _require_stateful_recomputed_rule(
        rule_definition, probe_outputs, observation_authority, probe_evaluation
    )
    source_receipts: list[_ReceiptBuild] = []
    probe_receipts: list[_ReceiptBuild] = []
    for trial, source_rule_trial, probe_rule_trial in zip(
        result.trials,
        source_evaluation.trials,
        probe_evaluation.trials,
        strict=True,
    ):
        source_receipts.append(
            _stateful_receipt(
                repetition=trial.repetition,
                arm="source",
                input_value=_stateful_input(result, "source"),
                execution_evidence=trial.baseline_execution_evidence,
                context=context,
                category="customer_invariant_violation",
                category_value=cast(JsonValue, source_evaluation.model_dump(mode="json")),
                category_json_pointer=f"/trials/{source_rule_trial.repetition - 1}",
                rule_definition=rule_definition if trial.repetition == 1 else None,
            )
        )
        probe_receipts.append(
            _stateful_receipt(
                repetition=trial.repetition,
                arm="probe",
                input_value=_stateful_input(result, "probe"),
                execution_evidence=trial.variation_execution_evidence,
                context=context,
                category="customer_invariant_violation",
                category_value=cast(JsonValue, probe_evaluation.model_dump(mode="json")),
                category_json_pointer=f"/trials/{probe_rule_trial.repetition - 1}",
            )
        )
    return _build_invariant_package(
        campaign_id=context.campaign_id,
        case_id=result.case.id,
        source_interaction_id=None,
        operator_id=result.case.operator_id,
        operator_version=result.case.operator_version,
        source_evaluation=source_evaluation,
        probe_evaluation=probe_evaluation,
        rule_definition=rule_definition,
        source_receipts=source_receipts,
        probe_receipts=probe_receipts,
        context=context,
        fixture_id=result.case.id,
        fixture_version=result.schema_version,
        inconclusive_reasons={
            trial.repetition: reason
            for trial in result.trials
            if (reason := _stateful_inconclusive_reason(trial.inconclusive_reason)) is not None
        },
    )


def _stateful_receipt(
    *,
    repetition: int,
    arm: Literal["source", "probe"],
    input_value: JsonValue,
    execution_evidence: ExecutionEvidence | None,
    context: FindingAdapterContext,
    category: FindingCategory,
    category_value: JsonValue,
    category_json_pointer: str,
    rule_definition: DatasetInvariantRule | None = None,
) -> _ReceiptBuild:
    if execution_evidence is None:
        return _unavailable_receipt(
            repetition=repetition,
            arm=arm,
            input_value=input_value,
            category=category,
            category_value=category_value,
            category_json_pointer=category_json_pointer,
            context=context,
            rule_definition=rule_definition,
        )
    return _execution_receipt(
        repetition=repetition,
        arm=arm,
        input_value=input_value,
        execution_evidence=execution_evidence,
        context=context,
        category=category,
        category_value=category_value,
        category_json_pointer=category_json_pointer,
        rule_definition=rule_definition,
    )


def adapt_stateful_finding_packages(
    result: StatefulFindingResult,
    *,
    invariant_rules: tuple[DatasetInvariantRule, ...],
    observation_authority: Literal["agent_response", "committed_state_snapshot"],
    context: FindingAdapterContext,
) -> tuple[FindingEvidencePackage, ...]:
    probe_rules = (
        result.corrected_invariant_rules
        if isinstance(result, CorrectionStressResult)
        else result.retried_invariant_rules
    )
    violated_rule_ids = {
        evaluation.rule_id for evaluation in probe_rules if evaluation.status == "violated"
    }
    rules_by_id = {rule.id: rule for rule in invariant_rules}
    if not violated_rule_ids.issubset(rules_by_id):
        raise ValueError("stateful violations require exact customer rule definitions")
    return tuple(
        adapt_stateful_invariant_finding(
            result,
            rules_by_id[rule_id],
            observation_authority=observation_authority,
            context=context,
        )
        for rule_id in sorted(violated_rule_ids)
    )


def _dataset_receipt(
    *,
    repetition: int,
    arm: Literal["source", "probe"],
    input_value: JsonValue,
    target_output: ObservedAgentOutput | None,
    execution_evidence: ExecutionEvidence | None,
    category: FindingCategory | None,
    category_value: JsonValue | None,
    category_json_pointer: str = "",
    context: FindingAdapterContext,
    rule_definition: DatasetInvariantRule | None = None,
    historical_reference_value: JsonValue | None = None,
    historical_reference_present: bool = False,
) -> _ReceiptBuild:
    if execution_evidence is not None:
        return _execution_receipt(
            repetition=repetition,
            arm=arm,
            input_value=input_value,
            execution_evidence=execution_evidence,
            context=context,
            category=category,
            category_value=category_value,
            category_json_pointer=category_json_pointer,
            rule_definition=rule_definition,
            historical_reference_value=historical_reference_value,
            historical_reference_present=historical_reference_present,
        )
    if target_output is None:
        return _unavailable_receipt(
            repetition=repetition,
            arm=arm,
            input_value=input_value,
            category=category,
            category_value=category_value,
            category_json_pointer=category_json_pointer,
            context=context,
            rule_definition=rule_definition,
            historical_reference_value=historical_reference_value,
            historical_reference_present=historical_reference_present,
        )
    return _receipt_from_values(
        repetition=repetition,
        arm=arm,
        evidence_scope="response_only",
        input_value=input_value,
        response_value=target_output.raw_output,
        state_before=None,
        state_after=None,
        lifecycle_status="unknown",
        lifecycle_limitation="lifecycle_provenance_unavailable",
        response_source_id=context.invoker_source_id,
        response_authority="invoker_self_reported",
        state_before_source_id=context.invoker_source_id,
        state_before_authority="invoker_self_reported",
        state_after_source_id=context.invoker_source_id,
        state_after_authority="invoker_self_reported",
        environment_source_id=context.invoker_source_id,
        artifact=target_output,
        lifecycle_json_pointer=None,
        context=context,
        category=category,
        category_value=category_value,
        category_json_pointer=category_json_pointer,
        rule_definition=rule_definition,
        historical_reference_value=historical_reference_value,
        historical_reference_present=historical_reference_present,
        limitations=("model_provenance_unavailable",),
    )


def _unavailable_receipt(
    *,
    repetition: int,
    arm: Literal["source", "probe"],
    input_value: JsonValue,
    category: FindingCategory | None,
    category_value: JsonValue | None,
    category_json_pointer: str,
    context: FindingAdapterContext,
    rule_definition: DatasetInvariantRule | None = None,
    historical_reference_value: JsonValue | None = None,
    historical_reference_present: bool = False,
) -> _ReceiptBuild:
    unavailable_limitation = (
        "source_execution_unavailable" if arm == "source" else "probe_execution_unavailable"
    )
    return _receipt_from_values(
        repetition=repetition,
        arm=arm,
        evidence_scope="response_only",
        input_value=input_value,
        response_value=None,
        state_before=None,
        state_after=None,
        lifecycle_status="not_attempted",
        lifecycle_limitation=unavailable_limitation,
        response_source_id=context.invoker_source_id,
        response_authority="invoker_self_reported",
        state_before_source_id=context.invoker_source_id,
        state_before_authority="invoker_self_reported",
        state_after_source_id=context.invoker_source_id,
        state_after_authority="invoker_self_reported",
        environment_source_id=context.invoker_source_id,
        artifact=input_value,
        context=context,
        category=category,
        category_value=category_value,
        category_json_pointer=category_json_pointer,
        rule_definition=rule_definition,
        lifecycle_json_pointer=None,
        limitations=("model_provenance_unavailable", unavailable_limitation),
        execution_available=False,
        historical_reference_value=historical_reference_value,
        historical_reference_present=historical_reference_present,
    )


def _execution_receipt(
    *,
    repetition: int,
    arm: Literal["source", "probe"],
    input_value: JsonValue,
    execution_evidence: ExecutionEvidence,
    context: FindingAdapterContext,
    category: FindingCategory | None = None,
    category_value: JsonValue | None = None,
    category_json_pointer: str = "",
    rule_definition: DatasetInvariantRule | None = None,
    historical_reference_value: JsonValue | None = None,
    historical_reference_present: bool = False,
) -> _ReceiptBuild:
    state_before = (
        execution_evidence.initial_state.value
        if execution_evidence.initial_state is not None
        else None
    )
    state_after = (
        execution_evidence.final_state.value if execution_evidence.final_state is not None else None
    )
    lifecycle_status: Literal["succeeded", "failed", "not_attempted", "unknown"] = (
        "succeeded" if execution_evidence.lifecycle.terminal_status == "succeeded" else "failed"
    )
    lifecycle_limitation = (
        None
        if lifecycle_status == "succeeded"
        else execution_evidence.lifecycle.failure_code or "execution_failed"
    )
    execution_available = (
        execution_evidence.lifecycle.terminal_status == "succeeded"
        and execution_evidence.final_response is not None
    )
    receipt_limitations = ["model_provenance_unavailable"]
    if not execution_available:
        receipt_limitations.append(
            "source_execution_unavailable" if arm == "source" else "probe_execution_unavailable"
        )
    state_before_source_id, state_before_authority = _state_provenance(
        execution_evidence.initial_state,
        execution_evidence.environment_id,
    )
    state_after_source_id, state_after_authority = _state_provenance(
        execution_evidence.final_state,
        execution_evidence.environment_id,
    )
    return _receipt_from_values(
        repetition=repetition,
        arm=arm,
        evidence_scope=(
            "response_and_state"
            if execution_evidence.evidence_scope == "response_and_state"
            and state_before is not None
            and state_after is not None
            else "response_only"
        ),
        input_value=input_value,
        response_value=execution_evidence.final_response,
        state_before=state_before,
        state_after=state_after,
        lifecycle_status=lifecycle_status,
        lifecycle_limitation=lifecycle_limitation,
        response_source_id=execution_evidence.environment_id,
        response_authority="source_self_reported",
        state_before_source_id=state_before_source_id,
        state_before_authority=state_before_authority,
        state_after_source_id=state_after_source_id,
        state_after_authority=state_after_authority,
        environment_source_id=execution_evidence.environment_id,
        artifact=execution_evidence,
        context=context,
        category=category,
        category_value=category_value,
        category_json_pointer=category_json_pointer,
        rule_definition=rule_definition,
        target_config_sha256=execution_evidence.environment_config_sha256,
        lifecycle_json_pointer="/lifecycle",
        limitations=tuple(receipt_limitations),
        execution_available=execution_available,
        trajectory_observations=execution_evidence.observations,
        trajectory_events=execution_evidence.execution_events,
        historical_reference_value=historical_reference_value,
        historical_reference_present=historical_reference_present,
    )


def _state_provenance(
    state: EnvironmentStateEvidence | None,
    environment_id: str,
) -> tuple[str, Literal["environment_self_reported", "independent_observer"]]:
    if state is not None and state.authority == "independent_observer":
        return state.observer_id or environment_id, "independent_observer"
    return environment_id, "environment_self_reported"


def _receipt_from_values(
    *,
    repetition: int,
    arm: Literal["source", "probe"],
    evidence_scope: Literal["response_only", "response_and_state"],
    input_value: JsonValue,
    response_value: JsonValue | None,
    state_before: JsonValue | None,
    state_after: JsonValue | None,
    lifecycle_status: Literal["succeeded", "failed", "not_attempted", "unknown"],
    lifecycle_limitation: str | None,
    response_source_id: str,
    response_authority: Literal[
        "invoker_self_reported",
        "source_self_reported",
        "environment_self_reported",
        "independent_observer",
    ],
    state_before_source_id: str,
    state_before_authority: Literal[
        "invoker_self_reported",
        "environment_self_reported",
        "independent_observer",
    ],
    state_after_source_id: str,
    state_after_authority: Literal[
        "invoker_self_reported",
        "environment_self_reported",
        "independent_observer",
    ],
    environment_source_id: str,
    artifact: object,
    context: FindingAdapterContext,
    category: FindingCategory | None,
    category_value: JsonValue | None,
    category_json_pointer: str,
    rule_definition: DatasetInvariantRule | None,
    target_config_sha256: str | None = None,
    lifecycle_json_pointer: str | None = "/lifecycle",
    limitations: tuple[str, ...] = (),
    execution_available: bool = True,
    trajectory_observations: tuple[ProbeObservation, ...] = (),
    trajectory_events: tuple[ProbeExecutionEvent, ...] = (),
    historical_reference_value: JsonValue | None = None,
    historical_reference_present: bool = False,
) -> _ReceiptBuild:
    pointers: list[EvidencePointer] = []
    retained_artifacts: dict[str, EvidenceArtifact] = {}
    default_artifact_value = _json_value(artifact)
    default_evidence_artifact: EvidenceArtifact | None = None

    def add_pointer(
        *,
        kind: EvidencePointerKind,
        json_pointer: str,
        pointer_arm: Literal["source", "probe", "shared"],
        authority: EvidenceAuthority,
        source_id: str,
        pointer_artifact: object = artifact,
        retain_artifact: bool = False,
        maximum_capture_bytes: int = _MAXIMUM_CAPTURED_JSON_BYTES,
    ) -> str:
        nonlocal default_evidence_artifact
        artifact_value = (
            default_artifact_value
            if pointer_artifact is artifact
            else _json_value(pointer_artifact)
        )
        if retain_artifact and pointer_artifact is artifact:
            if default_evidence_artifact is None:
                default_evidence_artifact = _evidence_artifact(default_artifact_value)
            retained_artifact = default_evidence_artifact
        else:
            retained_artifact = (
                _evidence_artifact(artifact_value, maximum_capture_bytes)
                if retain_artifact
                else None
            )
        artifact_sha256 = (
            retained_artifact.artifact_sha256
            if retained_artifact is not None
            else _sha256(artifact_value)
        )
        if retained_artifact is not None:
            retained_artifacts[artifact_sha256] = retained_artifact
        pointer_id = _pointer_id(
            context,
            artifact_sha256,
            repetition,
            pointer_arm,
            kind,
            json_pointer,
            source_id,
        )
        pointers.append(
            EvidencePointer(
                pointer_id=pointer_id,
                kind=kind,
                artifact_sha256=artifact_sha256,
                record_id=None,
                json_pointer=json_pointer,
                arm=pointer_arm,
                authority=authority,
                source_id=source_id,
            )
        )
        return pointer_id

    input_pointer_id = add_pointer(
        kind="input",
        json_pointer="",
        pointer_arm=arm,
        authority="invoker_self_reported",
        source_id=context.invoker_source_id,
        pointer_artifact=input_value,
        retain_artifact=True,
    )
    response_pointer_id = None
    if response_value is not None:
        response_pointer_id = add_pointer(
            kind="response",
            json_pointer="",
            pointer_arm=arm,
            authority=response_authority,
            source_id=response_source_id,
            pointer_artifact=response_value,
            retain_artifact=True,
        )
    historical_reference_pointer_id = None
    if historical_reference_present:
        historical_reference_pointer_id = add_pointer(
            kind="response",
            json_pointer="",
            pointer_arm="shared",
            authority="customer_declared",
            source_id=context.customer_source_id,
            pointer_artifact=historical_reference_value,
            retain_artifact=True,
        )
    lifecycle_pointer_ids: tuple[str, ...] = ()
    if lifecycle_json_pointer is not None:
        lifecycle_pointer_ids = (
            add_pointer(
                kind="lifecycle",
                json_pointer=lifecycle_json_pointer,
                pointer_arm=arm,
                authority="environment_self_reported",
                source_id=environment_source_id,
            ),
        )
    state_before_pointer_id = None
    state_after_pointer_id = None
    if evidence_scope == "response_and_state":
        if state_before is None or state_after is None:
            raise ValueError("stateful receipts require observed initial and final state")
        state_before_pointer_id = add_pointer(
            kind="state",
            json_pointer="/initial_state/value",
            pointer_arm=arm,
            authority=state_before_authority,
            source_id=state_before_source_id,
            pointer_artifact={
                "initial_state": {
                    "value": state_before,
                    "authority": state_before_authority,
                    "source_id": state_before_source_id,
                }
            },
            retain_artifact=True,
            maximum_capture_bytes=_MAXIMUM_STATE_CAPTURED_JSON_BYTES,
        )
        state_after_pointer_id = add_pointer(
            kind="state",
            json_pointer="/final_state/value",
            pointer_arm=arm,
            authority=state_after_authority,
            source_id=state_after_source_id,
            pointer_artifact={
                "final_state": {
                    "value": state_after,
                    "authority": state_after_authority,
                    "source_id": state_after_source_id,
                }
            },
            retain_artifact=True,
            maximum_capture_bytes=_MAXIMUM_STATE_CAPTURED_JSON_BYTES,
        )
    trace_evidence_pointer_ids = tuple(
        (
            *(
                add_pointer(
                    kind="trace",
                    json_pointer="/value",
                    pointer_arm=arm,
                    authority=observation.authority,
                    source_id=observation.source_id,
                    pointer_artifact={
                        "value": observation.model_dump(mode="json"),
                        "authority": observation.authority,
                        "source_id": observation.source_id,
                    },
                    retain_artifact=True,
                )
                for observation in trajectory_observations
                if any(
                    (
                        observation.traces,
                        observation.tool_calls,
                        observation.handoffs,
                        observation.errors,
                    )
                )
            ),
            *(
                add_pointer(
                    kind="trace",
                    json_pointer="/value",
                    pointer_arm=arm,
                    authority="invoker_self_reported",
                    source_id=context.invoker_source_id,
                    pointer_artifact={
                        "value": event.model_dump(mode="json"),
                        "authority": "invoker_self_reported",
                        "source_id": context.invoker_source_id,
                    },
                    retain_artifact=True,
                )
                for event in trajectory_events
            ),
        )
    )
    category_pointer_ids: tuple[str, ...] = ()
    if category is not None and category_value is not None:
        category_pointer_kind: EvidencePointerKind = (
            "rule"
            if category == "customer_invariant_violation"
            else "response"
            if category == "changed_response"
            else "action"
        )
        category_pointer_ids = (
            add_pointer(
                kind=category_pointer_kind,
                json_pointer=category_json_pointer,
                pointer_arm=arm,
                authority="deterministic_evaluator",
                source_id=context.evaluator_source_id,
                pointer_artifact=category_value,
                retain_artifact=True,
            ),
        )
    rule_definition_pointer_id = None
    if rule_definition is not None:
        rule_definition_pointer_id = add_pointer(
            kind="rule",
            json_pointer="",
            pointer_arm="shared",
            authority="customer_declared",
            source_id=context.customer_source_id,
            pointer_artifact=rule_definition,
            retain_artifact=True,
        )
    provenance = [
        ProvenanceReceipt(role="customer", id=context.customer_source_id),
        ProvenanceReceipt(role="environment", id=environment_source_id),
        ProvenanceReceipt(role="evaluator", id=context.evaluator_source_id),
        ProvenanceReceipt(role="invoker", id=context.invoker_source_id),
        ProvenanceReceipt(
            role="target",
            id=response_source_id,
            config_sha256=target_config_sha256,
        ),
    ]
    observer_source_ids = {
        source_id
        for source_id, authority in (
            (state_before_source_id, state_before_authority),
            (state_after_source_id, state_after_authority),
        )
        if authority == "independent_observer"
    }
    provenance.extend(
        ProvenanceReceipt(role="observer", id=source_id)
        for source_id in sorted(observer_source_ids)
    )
    trajectory_provenance = {
        (
            "observer" if observation.authority == "independent_observer" else "target",
            observation.source_id,
        )
        for observation in trajectory_observations
        if any(
            (
                observation.traces,
                observation.tool_calls,
                observation.handoffs,
                observation.errors,
            )
        )
    }
    provenance.extend(
        ProvenanceReceipt(role=cast(Literal["observer", "target"], role), id=source_id)
        for role, source_id in sorted(trajectory_provenance)
        if not any(item.role == role and item.id == source_id for item in provenance)
    )
    provenance.sort(key=lambda item: (item.role, item.id, item.version or ""))
    receipt_limitations = set(limitations)
    if response_value is None:
        receipt_limitations.add("response_missing")
    if context.redaction is None:
        receipt_limitations.add("redaction_accounting_unavailable")
    _validate_json_size(
        {
            "input": input_value,
            "response": response_value,
            "state_before": state_before,
            "state_after": state_after,
            "pointers": [pointer.model_dump(mode="json") for pointer in pointers],
            "provenance": [item.model_dump(mode="json") for item in provenance],
            "redaction": (
                context.redaction.model_dump(mode="json") if context.redaction is not None else None
            ),
            "limitations": sorted(receipt_limitations),
        },
        _MAXIMUM_RUN_RECEIPT_BYTES,
        "run receipt exceeds the 1 MB JSON limit",
    )
    trajectory_evidence_status: Literal["complete", "incomplete", "unavailable"] = (
        "complete"
        if any(
            observation.status == "complete"
            and any(
                (
                    observation.traces,
                    observation.tool_calls,
                    observation.handoffs,
                    observation.errors,
                )
            )
            for observation in trajectory_observations
        )
        or trajectory_events
        else "incomplete"
        if trace_evidence_pointer_ids
        else "unavailable"
    )
    content = RunReceiptContent(
        repetition=repetition,
        arm=arm,
        evidence_scope=evidence_scope,
        input=ReceiptEvidenceValue(
            evidence_pointer_id=input_pointer_id,
            value=_bounded_capture_json(input_value),
        ),
        historical_reference_response=(
            ReceiptEvidenceValue(
                evidence_pointer_id=historical_reference_pointer_id,
                value=_bounded_capture_json(historical_reference_value),
            )
            if historical_reference_pointer_id is not None
            else None
        ),
        response=(
            ReceiptEvidenceValue(
                evidence_pointer_id=response_pointer_id,
                value=_bounded_capture_json(response_value),
            )
            if response_pointer_id is not None and response_value is not None
            else None
        ),
        state_before=(
            StateReceipt(
                evidence=ReceiptEvidenceValue(
                    evidence_pointer_id=state_before_pointer_id,
                    value=_bounded_capture_json(state_before, _MAXIMUM_STATE_CAPTURED_JSON_BYTES),
                )
            )
            if state_before_pointer_id is not None
            else None
        ),
        state_after=(
            StateReceipt(
                evidence=ReceiptEvidenceValue(
                    evidence_pointer_id=state_after_pointer_id,
                    value=_bounded_capture_json(state_after, _MAXIMUM_STATE_CAPTURED_JSON_BYTES),
                )
            )
            if state_after_pointer_id is not None
            else None
        ),
        lifecycle=(
            LifecycleReceipt(
                phase="execution",
                status=lifecycle_status,
                evidence_pointer_ids=lifecycle_pointer_ids,
                limitation=lifecycle_limitation,
            ),
        ),
        provenance=tuple(provenance),
        trace_evidence_pointer_ids=trace_evidence_pointer_ids,
        trajectory_evidence_status=trajectory_evidence_status,
        redaction=context.redaction,
        evidence_pointers=tuple(sorted(pointers, key=lambda pointer: pointer.pointer_id)),
        limitations=tuple(sorted(receipt_limitations)),
        recorded_at=context.recorded_at,
    )
    return _ReceiptBuild(
        receipt=build_run_receipt(content),
        input_pointer_id=input_pointer_id,
        response_pointer_id=response_pointer_id,
        historical_reference_pointer_id=historical_reference_pointer_id,
        category_pointer_ids=category_pointer_ids,
        rule_definition_pointer_id=rule_definition_pointer_id,
        artifacts=tuple(sorted(retained_artifacts.values(), key=lambda item: item.artifact_sha256)),
        execution_available=execution_available,
    )


def _build_behavior_package(
    *,
    category: FindingCategory,
    campaign_id: str,
    case_id: str,
    source_interaction_id: str | None,
    operator_id: str,
    operator_version: str,
    source_receipts: list[_ReceiptBuild],
    probe_receipts: list[_ReceiptBuild],
    repetition_evidence: list[_BehaviorRepetitionEvidence],
    baseline_stability: Literal["stable", "unstable", "inconclusive"],
    variation_stability: Literal["stable", "unstable", "inconclusive"],
    baseline_drift: Literal["observed", "not_observed", "inconclusive"],
    context: FindingAdapterContext,
) -> FindingEvidencePackage:
    repetitions: list[FindingRepetition] = []
    delta_pointer_ids: list[str] = []
    for repetition, (source, probe, evidence) in enumerate(
        zip(source_receipts, probe_receipts, repetition_evidence, strict=True), start=1
    ):
        evidence_pointer_ids = tuple(
            sorted({*source.category_pointer_ids, *probe.category_pointer_ids})
        )
        if evidence.outcome == "finding_observed":
            delta_pointer_ids.extend(evidence_pointer_ids)
        repetitions.append(
            FindingRepetition(
                repetition=repetition,
                outcome=evidence.outcome,
                source_receipt_id=source.receipt.receipt_id,
                probe_receipt_id=probe.receipt.receipt_id,
                evidence_pointer_ids=evidence_pointer_ids,
                inconclusive_reason=evidence.inconclusive_reason,
            )
        )
    if not delta_pointer_ids:
        raise ValueError("behavior finding adapter requires one observed repetition")
    if category in {"duplicate_effect", "unexpected_effect"}:
        change: Literal["added", "removed", "changed"] = "added"
    elif category == "missing_effect":
        change = "removed"
    elif category in {"changed_grounded_effect_argument", "changed_response"}:
        change = "changed"
    else:
        raise ValueError("dataset behavior adapter requires a concrete behavior finding")
    observed_delta = ObservedDelta(
        kind="response" if category == "changed_response" else "action",
        change=change,
        subject_ref=_public_ref(
            context,
            "behavior-subject",
            operator_id,
            case_id,
            category,
        ),
        source_state="not_observed" if change == "added" else "observed",
        probe_state="not_observed" if change == "removed" else "observed",
        evidence_pointer_ids=tuple(sorted(set(delta_pointer_ids))),
    )
    return _package(
        kind="behavior_difference",
        category=category,
        campaign_id=campaign_id,
        case_id=case_id,
        source_interaction_id=source_interaction_id,
        operator_id=operator_id,
        operator_version=operator_version,
        source_receipts=source_receipts,
        probe_receipts=probe_receipts,
        repetitions=repetitions,
        observed_delta=observed_delta,
        context=context,
        violated_rule=None,
        rule_definition_pointer_ids=(),
        private_rule_id=None,
        private_rule_version=None,
        cross_examination=_dataset_cross_examination(
            operator_id=operator_id,
            operator_version=operator_version,
            source_receipts=source_receipts,
            probe_receipts=probe_receipts,
            material_delta_pointer_ids=observed_delta.evidence_pointer_ids,
            baseline_stability=baseline_stability,
            variation_stability=variation_stability,
            baseline_drift=baseline_drift,
            context=context,
        ),
    )


def _baseline_drift_signal(
    result: DatasetEvaluationResult,
) -> Literal["observed", "not_observed", "inconclusive"]:
    if result.baseline.trial_set.stability != "stable":
        return "inconclusive"
    current_baseline_frame = result.baseline.trial_set.representative_frame
    if current_baseline_frame is None:
        return "inconclusive"
    historical_frame = result.augmentation.source_frames[0]
    try:
        drift_deltas = compare_observed_outcomes(
            historical_frame,
            current_baseline_frame,
            result.source.raw_input,
            grounding_frame=historical_frame,
            comparison_surface=result.comparison_surface,
        )
    except ValueError:
        return "inconclusive"
    return "observed" if drift_deltas else "not_observed"


def _cross_examination_availability(
    *,
    fact: Literal["response", "trajectory", "committed_state"],
    source_receipts: list[_ReceiptBuild],
    probe_receipts: list[_ReceiptBuild],
) -> CrossExaminationEvidenceAvailability:
    achieved = "verified" if fact == "committed_state" else "observed"

    def arm(
        receipts: list[_ReceiptBuild],
    ) -> tuple[
        Literal["observed", "verified", "unavailable"],
        int,
        tuple[EvidenceAuthority, ...],
    ]:
        if fact == "response":
            covered = sum(receipt.response_pointer_id is not None for receipt in receipts)
            pointer_ids = {
                receipt.response_pointer_id
                for receipt in receipts
                if receipt.response_pointer_id is not None
            }
        elif fact == "trajectory":
            covered = sum(
                receipt.receipt.content.trajectory_evidence_status == "complete"
                for receipt in receipts
            )
            pointer_ids = {
                pointer.pointer_id
                for receipt in receipts
                for pointer in receipt.receipt.content.evidence_pointers
                if pointer.kind == "trace"
            }
        else:
            covered = sum(
                receipt.receipt.content.state_before is not None
                and receipt.receipt.content.state_after is not None
                for receipt in receipts
            )
            pointer_ids = {
                pointer.pointer_id
                for receipt in receipts
                for pointer in receipt.receipt.content.evidence_pointers
                if pointer.kind == "state"
            }
        authorities = cast(
            tuple[EvidenceAuthority, ...],
            tuple(
                sorted(
                    {
                        pointer.authority
                        for receipt in receipts
                        for pointer in receipt.receipt.content.evidence_pointers
                        if pointer.pointer_id in pointer_ids
                    }
                )
            ),
        )
        return (
            achieved if receipts and covered == len(receipts) else "unavailable",
            covered,
            authorities,
        )

    current_baseline, baseline_covered, baseline_authorities = arm(source_receipts)
    variation, variation_covered, variation_authorities = arm(probe_receipts)
    return CrossExaminationEvidenceAvailability(
        fact=fact,
        conclusion=(
            achieved if current_baseline == achieved and variation == achieved else "unavailable"
        ),
        current_baseline=current_baseline,
        variation=variation,
        current_baseline_covered_repetitions=baseline_covered,
        variation_covered_repetitions=variation_covered,
        current_baseline_authorities=baseline_authorities,
        variation_authorities=variation_authorities,
    )


def _dataset_cross_examination(
    *,
    operator_id: str,
    operator_version: str,
    source_receipts: list[_ReceiptBuild],
    probe_receipts: list[_ReceiptBuild],
    material_delta_pointer_ids: tuple[str, ...],
    baseline_stability: Literal["stable", "unstable", "inconclusive"],
    variation_stability: Literal["stable", "unstable", "inconclusive"],
    baseline_drift: Literal["observed", "not_observed", "inconclusive"],
    context: FindingAdapterContext,
) -> FindingCrossExamination:
    historical_pointer_ids = tuple(
        sorted(
            {
                pointer_id
                for receipt in source_receipts
                if receipt.historical_reference_pointer_id is not None
                for pointer_id in (receipt.historical_reference_pointer_id,)
            }
        )
    )
    current_pointer_ids = tuple(
        sorted(
            {
                pointer_id
                for receipt in source_receipts
                if receipt.response_pointer_id is not None
                for pointer_id in (receipt.response_pointer_id,)
            }
        )
    )
    variation_pointer_ids = tuple(
        sorted(
            {
                pointer_id
                for receipt in probe_receipts
                if receipt.response_pointer_id is not None
                for pointer_id in (receipt.response_pointer_id,)
            }
        )
    )
    current_trajectory_pointer_ids = tuple(
        sorted(
            pointer_id
            for receipt in source_receipts
            for pointer_id in receipt.receipt.content.trace_evidence_pointer_ids
        )
    )
    variation_trajectory_pointer_ids = tuple(
        sorted(
            pointer_id
            for receipt in probe_receipts
            for pointer_id in receipt.receipt.content.trace_evidence_pointer_ids
        )
    )
    current_state_pointer_ids = tuple(
        sorted(
            pointer.pointer_id
            for receipt in source_receipts
            for pointer in receipt.receipt.content.evidence_pointers
            if pointer.kind == "state"
        )
    )
    variation_state_pointer_ids = tuple(
        sorted(
            pointer.pointer_id
            for receipt in probe_receipts
            for pointer in receipt.receipt.content.evidence_pointers
            if pointer.kind == "state"
        )
    )
    if not historical_pointer_ids or not current_pointer_ids or not variation_pointer_ids:
        raise ValueError("response-level cross-examination requires all three evidence arms")
    baseline_observed = sum(receipt.execution_available for receipt in source_receipts)
    variation_observed = sum(receipt.execution_available for receipt in probe_receipts)
    requested = len(source_receipts)
    incomplete = "inconclusive" in {baseline_stability, variation_stability}
    unstable = "unstable" in {baseline_stability, variation_stability}
    return FindingCrossExamination(
        historical_reference=CrossExaminationArm(
            role="historical_reference",
            response_evidence_pointer_ids=historical_pointer_ids,
            requested_repetitions=0,
            observed_repetitions=0,
            inconclusive_repetitions=0,
            stability="not_applicable",
        ),
        current_baseline=CrossExaminationArm(
            role="current_baseline",
            response_evidence_pointer_ids=current_pointer_ids,
            trajectory_evidence_pointer_ids=current_trajectory_pointer_ids,
            committed_state_evidence_pointer_ids=current_state_pointer_ids,
            requested_repetitions=requested,
            observed_repetitions=baseline_observed,
            inconclusive_repetitions=requested - baseline_observed,
            stability=baseline_stability,
        ),
        variation=CrossExaminationArm(
            role="variation",
            response_evidence_pointer_ids=variation_pointer_ids,
            trajectory_evidence_pointer_ids=variation_trajectory_pointer_ids,
            committed_state_evidence_pointer_ids=variation_state_pointer_ids,
            requested_repetitions=len(probe_receipts),
            observed_repetitions=variation_observed,
            inconclusive_repetitions=len(probe_receipts) - variation_observed,
            stability=variation_stability,
        ),
        augmentation_relation=_versioned_ref(
            context,
            "operator",
            operator_id,
            operator_version,
        ),
        baseline_drift=("inconclusive" if baseline_stability == "inconclusive" else baseline_drift),
        augmentation_sensitivity=("inconclusive" if incomplete or unstable else "observed"),
        intrinsic_instability=(
            "inconclusive" if incomplete else "observed" if unstable else "not_observed"
        ),
        material_delta_evidence_pointer_ids=material_delta_pointer_ids,
        response_evidence=_cross_examination_availability(
            fact="response",
            source_receipts=source_receipts,
            probe_receipts=probe_receipts,
        ),
        trajectory_evidence=_cross_examination_availability(
            fact="trajectory",
            source_receipts=source_receipts,
            probe_receipts=probe_receipts,
        ),
        committed_state_evidence=_cross_examination_availability(
            fact="committed_state",
            source_receipts=source_receipts,
            probe_receipts=probe_receipts,
        ),
        limitations=(
            "causality_not_established",
            "correctness_not_verified",
            "historical_reference_not_an_oracle",
        ),
    )


def _build_invariant_package(
    *,
    campaign_id: str,
    case_id: str,
    source_interaction_id: str | None,
    operator_id: str,
    operator_version: str,
    source_evaluation: DatasetInvariantRuleResult,
    probe_evaluation: DatasetInvariantRuleResult,
    rule_definition: DatasetInvariantRule,
    source_receipts: list[_ReceiptBuild],
    probe_receipts: list[_ReceiptBuild],
    context: FindingAdapterContext,
    fixture_id: str | None = None,
    fixture_version: str | None = None,
    inconclusive_reasons: dict[int, str] | None = None,
) -> FindingEvidencePackage:
    repetitions: list[FindingRepetition] = []
    violated_pointer_ids: list[str] = []
    observed_source_statuses: set[str] = set()
    for source, probe, source_trial, probe_trial in zip(
        source_receipts,
        probe_receipts,
        source_evaluation.trials,
        probe_evaluation.trials,
        strict=True,
    ):
        category_pointer_ids = tuple(
            sorted({*source.category_pointer_ids, *probe.category_pointer_ids})
        )
        if source_trial.repetition != probe_trial.repetition:
            raise ValueError("source and probe invariant trials must align by repetition")
        if "not_evaluable" in {source_trial.status, probe_trial.status}:
            outcome: Literal["finding_observed", "finding_not_observed", "inconclusive"] = (
                "inconclusive"
            )
            inconclusive_reason = (
                inconclusive_reasons.get(probe_trial.repetition)
                if inconclusive_reasons is not None
                and probe_trial.repetition in inconclusive_reasons
                else source_trial.reason_code
                if source_trial.status == "not_evaluable"
                else probe_trial.reason_code
            )
        else:
            outcome = (
                "finding_observed" if probe_trial.status == "violated" else "finding_not_observed"
            )
            inconclusive_reason = None
        if outcome == "finding_observed":
            violated_pointer_ids.extend(category_pointer_ids)
            observed_source_statuses.add(source_trial.status)
        repetitions.append(
            FindingRepetition(
                repetition=probe_trial.repetition,
                outcome=outcome,
                source_receipt_id=source.receipt.receipt_id,
                probe_receipt_id=probe.receipt.receipt_id,
                evidence_pointer_ids=category_pointer_ids,
                inconclusive_reason=inconclusive_reason,
            )
        )
    if not violated_pointer_ids:
        raise ValueError("invariant finding adapter requires at least one violated repetition")
    versioned_rule = _versioned_ref(
        context,
        "rule",
        rule_definition.id,
        rule_definition.version,
    )
    source_state = (
        next(iter(observed_source_statuses)) if len(observed_source_statuses) == 1 else "unknown"
    )
    observed_delta = ObservedDelta(
        kind="rule",
        change="violated",
        subject_ref=_public_ref(
            context,
            "rule-subject",
            rule_definition.id,
            rule_definition.version,
        ),
        rule=versioned_rule,
        source_state=cast(Literal["satisfied", "violated", "unknown"], source_state),
        probe_state="violated",
        evidence_pointer_ids=tuple(sorted(set(violated_pointer_ids))),
    )
    rule_definition_pointer_ids = tuple(
        sorted(
            pointer_id
            for receipt in source_receipts
            if (pointer_id := receipt.rule_definition_pointer_id) is not None
        )
    )
    return _package(
        kind="customer_invariant_violation",
        category="customer_invariant_violation",
        campaign_id=campaign_id,
        case_id=case_id,
        source_interaction_id=source_interaction_id,
        operator_id=operator_id,
        operator_version=operator_version,
        source_receipts=source_receipts,
        probe_receipts=probe_receipts,
        repetitions=repetitions,
        observed_delta=observed_delta,
        context=context,
        violated_rule=versioned_rule,
        rule_definition_pointer_ids=rule_definition_pointer_ids,
        private_rule_id=rule_definition.id,
        private_rule_version=rule_definition.version,
        fixture_id=fixture_id,
        fixture_version=fixture_version,
    )


def _package(
    *,
    kind: Literal["behavior_difference", "customer_invariant_violation"],
    category: FindingCategory,
    campaign_id: str,
    case_id: str,
    source_interaction_id: str | None,
    operator_id: str,
    operator_version: str,
    source_receipts: list[_ReceiptBuild],
    probe_receipts: list[_ReceiptBuild],
    repetitions: list[FindingRepetition],
    observed_delta: ObservedDelta,
    context: FindingAdapterContext,
    violated_rule: VersionedReference | None,
    rule_definition_pointer_ids: tuple[str, ...],
    private_rule_id: str | None,
    private_rule_version: str | None,
    fixture_id: str | None = None,
    fixture_version: str | None = None,
    cross_examination: FindingCrossExamination | None = None,
) -> FindingEvidencePackage:
    requested = len(repetitions)
    observed = sum(item.outcome == "finding_observed" for item in repetitions)
    inconclusive = sum(item.outcome == "inconclusive" for item in repetitions)
    conclusive = requested - inconclusive
    reproducibility: Literal["reproduced", "intermittent", "not_reproduced", "not_established"] = (
        "not_established"
        if conclusive == 0
        else "not_reproduced"
        if observed == 0
        else "reproduced"
        if observed == conclusive
        else "intermittent"
    )
    stability: Literal["stable", "unstable", "inconclusive"] = (
        "inconclusive" if inconclusive else "unstable" if observed != conclusive else "stable"
    )
    change_source_pointer_ids = tuple(
        sorted(receipt.input_pointer_id for receipt in source_receipts)
    )
    change_probe_pointer_ids = tuple(sorted(receipt.input_pointer_id for receipt in probe_receipts))
    evidence_pointer_ids = tuple(
        sorted(
            {
                *change_source_pointer_ids,
                *change_probe_pointer_ids,
                *rule_definition_pointer_ids,
                *observed_delta.evidence_pointer_ids,
                *(
                    (
                        pointer_id
                        for arm in (
                            cross_examination.historical_reference,
                            cross_examination.current_baseline,
                            cross_examination.variation,
                        )
                        for pointer_id in (
                            *arm.response_evidence_pointer_ids,
                            *arm.trajectory_evidence_pointer_ids,
                            *arm.committed_state_evidence_pointer_ids,
                        )
                    )
                    if cross_examination is not None
                    else ()
                ),
                *(
                    pointer_id
                    for repetition in repetitions
                    for pointer_id in repetition.evidence_pointer_ids
                ),
            }
        )
    )
    limitations = {"production_prevalence_not_measured"}
    if kind == "behavior_difference":
        limitations.add("correctness_not_verified")
    if inconclusive:
        limitations.add("one_or_more_repetitions_inconclusive")
    if any(not receipt.execution_available for receipt in source_receipts):
        limitations.add("source_execution_unavailable")
    occurrence = build_finding_occurrence(
        kind=kind,
        category=category,
        campaign_ref=_public_ref(context, "campaign", campaign_id),
        source_interaction_ref=(
            _public_ref(context, "source-interaction", source_interaction_id)
            if source_interaction_id is not None
            else None
        ),
        fixture=(
            _versioned_ref(context, "fixture", fixture_id, fixture_version)
            if fixture_id is not None and fixture_version is not None
            else None
        ),
        case_ref=_public_ref(context, "case", case_id),
        operator=_versioned_ref(context, "operator", operator_id, operator_version),
        probe_change=ProbeChange(
            kind="input" if source_interaction_id is not None else "turn_sequence",
            source_descriptor=(
                "recorded_input" if source_interaction_id is not None else "baseline_turn_sequence"
            ),
            probe_descriptor=(
                "augmented_input"
                if source_interaction_id is not None
                else "augmented_turn_sequence"
            ),
            source_evidence_pointer_ids=change_source_pointer_ids,
            probe_evidence_pointer_ids=change_probe_pointer_ids,
        ),
        cross_examination=cross_examination,
        observed_deltas=(observed_delta,),
        violated_rule=violated_rule,
        rule_definition_evidence_pointer_ids=rule_definition_pointer_ids,
        evidence_pointer_ids=evidence_pointer_ids,
        repetitions=tuple(repetitions),
        repetition_summary=RepetitionSummary(
            requested=requested,
            conclusive=conclusive,
            observed=observed,
            violated=observed if kind == "customer_invariant_violation" else None,
            inconclusive=inconclusive,
            stability=stability,
            reproducibility=reproducibility,
        ),
        required_capabilities=("response_observation", "state_observation")
        if fixture_id is not None
        else ("response_observation",),
        limitations=tuple(sorted(limitations)),
        next_action=(
            "review_dataset_finding"
            if source_interaction_id is not None
            else "inspect_stateful_evidence"
        ),
    )
    receipts = tuple(
        sorted(
            (receipt.receipt for receipt in (*source_receipts, *probe_receipts)),
            key=lambda receipt: receipt.receipt_id,
        )
    )
    artifacts_by_digest = {
        artifact.artifact_sha256: artifact
        for receipt in (*source_receipts, *probe_receipts)
        for artifact in receipt.artifacts
    }
    artifacts = tuple(
        sorted(artifacts_by_digest.values(), key=lambda artifact: artifact.artifact_sha256)
    )
    package_values = {
        "schema_version": "1.2.0",
        "disclosure": "private",
        "occurrence": occurrence.model_dump(mode="json"),
        "private_references": {
            "disclosure": "private",
            "campaign_id": campaign_id,
            "case_id": case_id,
            "source_interaction_id": source_interaction_id,
            "operator_id": operator_id,
            "operator_version": operator_version,
            "rule_id": private_rule_id,
            "rule_version": private_rule_version,
        },
        "receipts": [receipt.model_dump(mode="json") for receipt in receipts],
        "artifact_retention": "embedded",
        "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
    }
    _validate_json_size(
        package_values,
        _MAXIMUM_ADAPTED_PACKAGE_BYTES,
        "finding evidence package exceeds the 16 MB JSON limit",
    )
    return FindingEvidencePackage(
        occurrence=occurrence,
        private_references=FindingPrivateReferences(
            campaign_id=campaign_id,
            case_id=case_id,
            source_interaction_id=source_interaction_id,
            operator_id=operator_id,
            operator_version=operator_version,
            rule_id=private_rule_id,
            rule_version=private_rule_version,
        ),
        receipts=receipts,
        artifact_retention="embedded",
        artifacts=artifacts,
    )


def _behavior_repetition_evidence(
    finding: DatasetEvaluationFinding,
    source_trial: DatasetEvaluationTrial,
    probe_trial: DatasetEvaluationTrial,
    source_input: str,
    grounding_frame: SemanticFrame,
    comparison_surface: Literal["action", "response"],
) -> _BehaviorRepetitionEvidence:
    if source_trial.observed_frame is None or probe_trial.observed_frame is None:
        unavailable_arm = "source" if source_trial.observed_frame is None else "probe"
        execution_missing = (
            source_trial.target_output is None
            if unavailable_arm == "source"
            else probe_trial.target_output is None
        )
        reason = (
            f"{unavailable_arm}_execution_unavailable"
            if execution_missing
            else f"{unavailable_arm}_evaluation_unavailable"
        )
        return _BehaviorRepetitionEvidence(
            source_value=_behavior_values(source_trial.observed_frame, finding.category),
            probe_value=_behavior_values(probe_trial.observed_frame, finding.category),
            outcome="inconclusive",
            inconclusive_reason=reason,
        )
    recomputed_findings = compare_observed_outcomes(
        source_trial.observed_frame,
        probe_trial.observed_frame,
        source_input,
        grounding_frame=grounding_frame,
        comparison_surface=comparison_surface,
    )
    finding_observed = any(
        _behavior_finding_signature(candidate) == _behavior_finding_signature(finding)
        for candidate in recomputed_findings
    )
    return _BehaviorRepetitionEvidence(
        source_value=_behavior_values(source_trial.observed_frame, finding.category),
        probe_value=_behavior_values(probe_trial.observed_frame, finding.category),
        outcome="finding_observed" if finding_observed else "finding_not_observed",
    )


def _behavior_values(frame: SemanticFrame | None, category: FindingCategory) -> JsonValue | None:
    if frame is None:
        return None
    return cast(
        JsonValue,
        [
            outcome.model_dump(mode="json")
            for outcome in frame.outcomes
            if (
                outcome.kind == "answer"
                if category == "changed_response"
                else outcome.kind == "action"
            )
        ],
    )


def _behavior_finding_signature(finding: DatasetEvaluationFinding) -> str:
    return _canonical_json(
        {
            "category": finding.category,
            "expected": [_outcome_semantics(outcome) for outcome in finding.expected_effects],
            "observed": [_outcome_semantics(outcome) for outcome in finding.observed_effects],
            "grounded_field_names": finding.grounded_field_names,
        }
    )


def _outcome_semantics(outcome: ObservedOutcome) -> JsonValue:
    return cast(
        JsonValue,
        {
            "predicate": outcome.predicate,
            "fields": outcome.fields,
            "position": outcome.position,
            "propositions": outcome.propositions,
        },
    )


def _dataset_rule_pair(
    evaluation: DatasetInvariantEvaluation,
    operator_id: str,
    rule_definition: DatasetInvariantRule,
) -> tuple[DatasetInvariantRuleResult, DatasetInvariantRuleResult]:
    if evaluation.interaction_id == "":
        raise ValueError("dataset invariant evaluation requires an interaction")
    matching_variations = tuple(
        arm for arm in evaluation.variations if arm.operator_id == operator_id
    )
    if len(matching_variations) != 1:
        raise ValueError("dataset invariant evaluation requires one exact probe arm")
    return (
        _matching_rule(evaluation.baseline.rules, rule_definition),
        _matching_rule(matching_variations[0].rules, rule_definition),
    )


def _stateful_rule_pair(
    result: StatefulFindingResult,
    rule_definition: DatasetInvariantRule,
) -> tuple[DatasetInvariantRuleResult, DatasetInvariantRuleResult]:
    if isinstance(result, CorrectionStressResult):
        source_rules = result.baseline_invariant_rules
        probe_rules = result.corrected_invariant_rules
    else:
        source_rules = result.successful_commit_invariant_rules
        probe_rules = result.retried_invariant_rules
    return (
        _matching_rule(source_rules, rule_definition),
        _matching_rule(probe_rules, rule_definition),
    )


def _require_dataset_recomputed_rule(
    rule_definition: DatasetInvariantRule,
    trials: tuple[DatasetEvaluationTrial, ...],
    observation_authority: Literal["agent_response", "committed_state_snapshot"],
    supplied_evaluation: DatasetInvariantRuleResult,
) -> None:
    recomputed = evaluate_dataset_invariant_rule_trials(
        rule_definition,
        trials,
        observation_authority=observation_authority,
    )
    if recomputed != supplied_evaluation:
        raise ValueError("invariant evaluation does not match its exact captured evidence")


def _stateful_rule_outputs(
    result: StatefulFindingResult,
) -> tuple[
    tuple[ObservedAgentOutput | None, ...],
    tuple[ObservedAgentOutput | None, ...],
]:
    if isinstance(result, CorrectionStressResult):
        source_outputs = tuple(
            trial.baseline[-1].target_output if trial.baseline else None for trial in result.trials
        )
    else:
        source_outputs = tuple(
            trial.variation[0].target_output
            if trial.inconclusive_reason is None and len(trial.variation) == 2
            else None
            for trial in result.trials
        )
    probe_outputs = tuple(
        trial.variation[-1].target_output
        if trial.inconclusive_reason is None and len(trial.variation) == 2
        else None
        for trial in result.trials
    )
    return source_outputs, probe_outputs


def _stateful_inconclusive_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    return {
        "environment lifecycle failed": "environment_lifecycle_failed",
        "environment execution timed out": "environment_execution_timed_out",
        "environment execution failed": "environment_execution_failed",
        (
            "environment not called because prior execution left state uncertain"
        ): "prior_execution_left_state_uncertain",
    }.get(reason, "workflow_execution_inconclusive")


def _require_stateful_recomputed_rule(
    rule_definition: DatasetInvariantRule,
    outputs: tuple[ObservedAgentOutput | None, ...],
    observation_authority: Literal["agent_response", "committed_state_snapshot"],
    supplied_evaluation: DatasetInvariantRuleResult,
) -> None:
    (recomputed_evaluation,) = evaluate_dataset_invariant_rules(
        (rule_definition,),
        outputs,
        observation_authority=observation_authority,
    )
    if supplied_evaluation != recomputed_evaluation:
        raise ValueError("invariant evaluation does not match its exact captured evidence")


def _matching_rule(
    rules: tuple[DatasetInvariantRuleResult, ...],
    rule_definition: DatasetInvariantRule,
) -> DatasetInvariantRuleResult:
    matches = tuple(
        rule
        for rule in rules
        if rule.rule_id == rule_definition.id and rule.rule_version == rule_definition.version
    )
    if len(matches) != 1:
        raise ValueError("workflow evidence must contain one exact invariant rule")
    evaluation = matches[0]
    if _rule_evaluation_semantics(evaluation) != _rule_definition_semantics(rule_definition):
        raise ValueError("invariant evaluation must match the full customer rule definition")
    return evaluation


def _stateful_input(
    result: StatefulFindingResult,
    arm: Literal["source", "probe"],
) -> JsonValue:
    turns = result.case.conversation
    selected_turns = turns[:1] if arm == "source" else turns
    return cast(JsonValue, [turn.model_dump(mode="json") for turn in selected_turns])


def _rule_definition_semantics(
    rule_definition: DatasetInvariantRule,
) -> dict[str, JsonValue]:
    values = cast(dict[str, JsonValue], rule_definition.model_dump(mode="json"))
    return {
        (
            "rule_type"
            if key == "type"
            else "rule_id"
            if key == "id"
            else "rule_version"
            if key == "version"
            else key
        ): value
        for key, value in values.items()
    }


def _rule_evaluation_semantics(
    evaluation: DatasetInvariantRuleResult,
) -> dict[str, JsonValue]:
    values = cast(
        dict[str, JsonValue],
        evaluation.model_dump(mode="json", exclude={"status", "reason_code", "trials"}),
    )
    if evaluation.rule_type == "json_values_equal":
        first_trial = evaluation.trials[0]
        values["left_pointer"] = first_trial.left_pointer
        values["right_pointer"] = first_trial.right_pointer
    return values


def _require_dataset_trial_set(case: DatasetEvaluationCase):
    if case.trial_set is None:
        raise ValueError("dataset finding adapter requires accepted variation trials")
    return case.trial_set


def _validate_repetition_bound(*repetition_collections: tuple[object, ...]) -> None:
    repetition_counts = {len(collection) for collection in repetition_collections}
    if len(repetition_counts) != 1:
        raise ValueError("finding evidence arms must have the same repetition count")
    repetition_count = next(iter(repetition_counts))
    if repetition_count > _MAXIMUM_ADAPTED_REPETITIONS:
        raise ValueError("finding evidence exceeds the 1,000 repetition limit")


def _versioned_ref(
    context: FindingAdapterContext,
    namespace: str,
    identifier: str,
    version: str,
) -> VersionedReference:
    return VersionedReference(
        id=_public_ref(context, namespace, identifier),
        version=_public_ref(context, f"{namespace}-version", version),
    )


def _public_ref(context: FindingAdapterContext, namespace: str, *values: str) -> str:
    return finding_public_reference(
        context.reference_key,
        context.campaign_id,
        namespace,
        *values,
    )


def _pointer_id(
    context: FindingAdapterContext,
    artifact_sha256: str,
    repetition: int,
    arm: str,
    kind: str,
    json_pointer: str,
    source_id: str,
) -> str:
    message = _canonical_json(
        [artifact_sha256, repetition, arm, kind, json_pointer, source_id]
    ).encode("utf-8")
    digest = hmac.digest(context.reference_key, message, "sha256").hex()
    return f"ulep_v1_{digest}"


def _sha256(value: JsonValue) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _bounded_capture_json(
    value: JsonValue,
    maximum_bytes: int = _MAXIMUM_CAPTURED_JSON_BYTES,
) -> CapturedJson:
    canonical_json = _bounded_canonical_json(
        value,
        maximum_bytes,
        "captured JSON exceeds its byte limit",
    )
    return CapturedJson(
        canonical_json=canonical_json,
        sha256=hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
    )


def _evidence_artifact(
    value: JsonValue,
    maximum_bytes: int = _MAXIMUM_CAPTURED_JSON_BYTES,
) -> EvidenceArtifact:
    capture = _bounded_capture_json(value, maximum_bytes)
    return EvidenceArtifact(artifact_sha256=capture.sha256, value=capture)


def _validate_json_size(value: object, maximum_bytes: int, message: str) -> None:
    _bounded_canonical_json(value, maximum_bytes, message)


def _bounded_canonical_json(value: object, maximum_bytes: int, message: str) -> str:
    chunks: list[str] = []
    encoded_bytes = 0
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    for chunk in encoder.iterencode(value):
        encoded_bytes += len(chunk.encode("utf-8"))
        if encoded_bytes > maximum_bytes:
            raise ValueError(message)
        chunks.append(chunk)
    return "".join(chunks)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def _json_value(value: object) -> JsonValue:
    if isinstance(value, BaseModel):
        return cast(JsonValue, value.model_dump(mode="json"))
    return cast(JsonValue, value)
