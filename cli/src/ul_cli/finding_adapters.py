from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, cast

from pydantic import BaseModel, JsonValue
from ul import DatasetEvaluationResult
from ul.dataset_evaluation import (
    DatasetEvaluationCase,
    DatasetEvaluationFinding,
    DatasetEvaluationTrial,
)
from ul.dataset_invariants import (
    DatasetInvariantRule,
    DatasetInvariantRuleEvaluation,
    DatasetInvariantRuleResult,
)
from ul.event_stress import (
    CorrectionStressResult,
    RetryAfterSuccessfulCommitStressResult,
)
from ul_core.dataset import ObservedAgentOutput
from ul_core.evaluation import ExecutionEvidence

from ul_cli.report_contract import (
    EvidencePointer,
    FindingCategory,
    FindingEvidencePackage,
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
    capture_json,
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


@dataclass(frozen=True)
class FindingAdapterContext:
    campaign_id: str
    recorded_at: datetime
    redaction_policy_sha256: str
    customer_source_id: str = "customer-invariant-suite"
    evaluator_source_id: str = "ul-deterministic-evaluator"
    invoker_source_id: str = "ul-active-probe"


@dataclass(frozen=True)
class _ReceiptBuild:
    receipt: RunReceipt
    input_pointer_id: str
    category_pointer_ids: tuple[str, ...]
    rule_definition_pointer_id: str | None = None


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
    source_receipts: list[_ReceiptBuild] = []
    probe_receipts: list[_ReceiptBuild] = []
    for baseline_trial, probe_trial in zip(baseline_trials, trial_set.trials, strict=True):
        source_receipts.append(
            _dataset_receipt(
                repetition=baseline_trial.repetition,
                arm="source",
                input_value=result.source.raw_input,
                target_output=baseline_trial.target_output,
                execution_evidence=baseline_trial.execution_evidence,
                category=finding.category,
                category_value=_behavior_trial_category_value(
                    finding,
                    baseline_trial,
                    "source",
                ),
                context=context,
            )
        )
        probe_receipts.append(
            _dataset_receipt(
                repetition=probe_trial.repetition,
                arm="probe",
                input_value=case.candidate.augmented_input,
                target_output=probe_trial.target_output,
                execution_evidence=probe_trial.execution_evidence,
                category=finding.category,
                category_value=_behavior_trial_category_value(
                    finding,
                    probe_trial,
                    "probe",
                ),
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
        context=context,
    )


def adapt_dataset_invariant_finding(
    result: DatasetEvaluationResult,
    evaluation: DatasetInvariantRuleEvaluation,
    rule_definition: DatasetInvariantRule,
    *,
    case_index: int,
    context: FindingAdapterContext,
) -> FindingEvidencePackage:
    _validate_rule_identity(evaluation, rule_definition)
    case = result.cases[case_index]
    trial_set = _require_dataset_trial_set(case)
    baseline_trials = result.baseline.trial_set.trials
    source_receipts: list[_ReceiptBuild] = []
    probe_receipts: list[_ReceiptBuild] = []
    for baseline_trial, probe_trial, rule_trial in zip(
        baseline_trials,
        trial_set.trials,
        evaluation.trials,
        strict=True,
    ):
        source_receipts.append(
            _dataset_receipt(
                repetition=baseline_trial.repetition,
                arm="source",
                input_value=result.source.raw_input,
                target_output=baseline_trial.target_output,
                execution_evidence=baseline_trial.execution_evidence,
                category=None,
                category_value=None,
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
                category_value=cast(JsonValue, rule_trial.model_dump(mode="json")),
                context=context,
            )
        )
    return _build_invariant_package(
        campaign_id=context.campaign_id,
        case_id=result.source.id,
        source_interaction_id=result.source.id,
        operator_id=case.candidate.operator_id,
        operator_version=case.candidate.operator_version,
        evaluation=evaluation,
        rule_definition=rule_definition,
        source_receipts=source_receipts,
        probe_receipts=probe_receipts,
        context=context,
    )


def adapt_stateful_invariant_finding(
    result: StatefulFindingResult,
    rule_definition: DatasetInvariantRule,
    *,
    context: FindingAdapterContext,
) -> FindingEvidencePackage:
    evaluation = _stateful_rule_evaluation(result, rule_definition)
    _validate_rule_identity(evaluation, rule_definition)
    source_receipts: list[_ReceiptBuild] = []
    probe_receipts: list[_ReceiptBuild] = []
    for trial, rule_trial in zip(result.trials, evaluation.trials, strict=True):
        if trial.baseline_execution_evidence is None or trial.variation_execution_evidence is None:
            raise ValueError("stateful invariant findings require both execution arms")
        source_receipts.append(
            _execution_receipt(
                repetition=trial.repetition,
                arm="source",
                input_value=_stateful_input(result, "source"),
                execution_evidence=trial.baseline_execution_evidence,
                context=context,
                rule_definition=rule_definition if trial.repetition == 1 else None,
            )
        )
        probe_receipts.append(
            _execution_receipt(
                repetition=trial.repetition,
                arm="probe",
                input_value=_stateful_input(result, "probe"),
                execution_evidence=trial.variation_execution_evidence,
                context=context,
                category="customer_invariant_violation",
                category_value=cast(JsonValue, rule_trial.model_dump(mode="json")),
            )
        )
    return _build_invariant_package(
        campaign_id=context.campaign_id,
        case_id=result.case.id,
        source_interaction_id=None,
        operator_id=result.case.operator_id,
        operator_version=result.case.operator_version,
        evaluation=evaluation,
        rule_definition=rule_definition,
        source_receipts=source_receipts,
        probe_receipts=probe_receipts,
        context=context,
        fixture_id=result.case.id,
        fixture_version=result.schema_version,
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
    context: FindingAdapterContext,
    rule_definition: DatasetInvariantRule | None = None,
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
            rule_definition=rule_definition,
        )
    if target_output is None:
        raise ValueError("conclusive dataset findings require execution output")
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
        state_source_id=context.invoker_source_id,
        state_authority="invoker_self_reported",
        environment_source_id=context.invoker_source_id,
        artifact=target_output,
        response_json_pointer="/raw_output",
        lifecycle_json_pointer=None,
        context=context,
        category=category,
        category_value=category_value,
        rule_definition=rule_definition,
        limitations=("model_provenance_unavailable",),
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
    rule_definition: DatasetInvariantRule | None = None,
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
    state_source_id = execution_evidence.environment_id
    state_authority: Literal["environment_self_reported", "independent_observer"] = (
        "environment_self_reported"
    )
    if execution_evidence.final_state is not None:
        state_authority = execution_evidence.final_state.authority
        if execution_evidence.final_state.authority == "independent_observer":
            state_source_id = (
                execution_evidence.final_state.observer_id or execution_evidence.environment_id
            )
    return _receipt_from_values(
        repetition=repetition,
        arm=arm,
        evidence_scope=execution_evidence.evidence_scope,
        input_value=input_value,
        response_value=execution_evidence.final_response,
        state_before=state_before,
        state_after=state_after,
        lifecycle_status=lifecycle_status,
        lifecycle_limitation=lifecycle_limitation,
        response_source_id=execution_evidence.environment_id,
        response_authority="source_self_reported",
        state_source_id=state_source_id,
        state_authority=state_authority,
        environment_source_id=execution_evidence.environment_id,
        artifact=execution_evidence,
        context=context,
        category=category,
        category_value=category_value,
        rule_definition=rule_definition,
        target_config_sha256=execution_evidence.environment_config_sha256,
        response_json_pointer="/final_response",
        state_before_json_pointer="/initial_state/value",
        state_after_json_pointer="/final_state/value",
        lifecycle_json_pointer="/lifecycle",
        limitations=("model_provenance_unavailable",),
    )


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
    state_source_id: str,
    state_authority: Literal[
        "invoker_self_reported",
        "environment_self_reported",
        "independent_observer",
    ],
    environment_source_id: str,
    artifact: object,
    context: FindingAdapterContext,
    category: FindingCategory | None,
    category_value: JsonValue | None,
    rule_definition: DatasetInvariantRule | None,
    target_config_sha256: str | None = None,
    response_json_pointer: str = "/response",
    state_before_json_pointer: str = "/state_before",
    state_after_json_pointer: str = "/state_after",
    lifecycle_json_pointer: str | None = "/lifecycle",
    limitations: tuple[str, ...] = (),
) -> _ReceiptBuild:
    pointers: list[EvidencePointer] = []

    def add_pointer(
        *,
        kind: EvidencePointerKind,
        json_pointer: str,
        pointer_arm: Literal["source", "probe", "shared"],
        authority: EvidenceAuthority,
        source_id: str,
        pointer_artifact: object = artifact,
    ) -> str:
        artifact_sha256 = _sha256(_json_value(pointer_artifact))
        pointer_id = _pointer_id(
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
    )
    response_pointer_id = None
    if response_value is not None:
        response_pointer_id = add_pointer(
            kind="response",
            json_pointer=response_json_pointer,
            pointer_arm=arm,
            authority=response_authority,
            source_id=response_source_id,
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
            json_pointer=state_before_json_pointer,
            pointer_arm=arm,
            authority=state_authority,
            source_id=state_source_id,
        )
        state_after_pointer_id = add_pointer(
            kind="state",
            json_pointer=state_after_json_pointer,
            pointer_arm=arm,
            authority=state_authority,
            source_id=state_source_id,
        )
    category_pointer_ids: tuple[str, ...] = ()
    if category is not None and category_value is not None:
        category_pointer_kind = "rule" if category == "customer_invariant_violation" else "action"
        category_pointer_ids = (
            add_pointer(
                kind=category_pointer_kind,
                json_pointer="",
                pointer_arm=arm,
                authority="deterministic_evaluator",
                source_id=context.evaluator_source_id,
                pointer_artifact=category_value,
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
    if state_authority == "independent_observer":
        provenance.append(ProvenanceReceipt(role="observer", id=state_source_id))
    provenance.sort(key=lambda item: (item.role, item.id, item.version or ""))
    receipt_limitations = set(limitations)
    if response_value is None:
        receipt_limitations.add("response_missing")
    content = RunReceiptContent(
        repetition=repetition,
        arm=arm,
        evidence_scope=evidence_scope,
        input=ReceiptEvidenceValue(
            evidence_pointer_id=input_pointer_id,
            value=capture_json(input_value),
        ),
        response=(
            ReceiptEvidenceValue(
                evidence_pointer_id=response_pointer_id,
                value=capture_json(response_value),
            )
            if response_pointer_id is not None and response_value is not None
            else None
        ),
        state_before=(
            StateReceipt(
                evidence=ReceiptEvidenceValue(
                    evidence_pointer_id=state_before_pointer_id,
                    value=capture_json(state_before),
                )
            )
            if state_before_pointer_id is not None
            else None
        ),
        state_after=(
            StateReceipt(
                evidence=ReceiptEvidenceValue(
                    evidence_pointer_id=state_after_pointer_id,
                    value=capture_json(state_after),
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
        redaction=RedactionReceipt(
            policy_sha256=context.redaction_policy_sha256,
            matched_value_count=0,
            redacted_value_count=0,
            retained_private_value_count=0,
        ),
        evidence_pointers=tuple(sorted(pointers, key=lambda pointer: pointer.pointer_id)),
        limitations=tuple(sorted(receipt_limitations)),
        recorded_at=context.recorded_at,
    )
    return _ReceiptBuild(
        receipt=build_run_receipt(content),
        input_pointer_id=input_pointer_id,
        category_pointer_ids=category_pointer_ids,
        rule_definition_pointer_id=rule_definition_pointer_id,
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
    context: FindingAdapterContext,
) -> FindingEvidencePackage:
    repetitions: list[FindingRepetition] = []
    delta_pointer_ids: list[str] = []
    for repetition, (source, probe) in enumerate(
        zip(source_receipts, probe_receipts, strict=True), start=1
    ):
        evidence_pointer_ids = tuple(
            sorted({*source.category_pointer_ids, *probe.category_pointer_ids})
        )
        delta_pointer_ids.extend(evidence_pointer_ids)
        repetitions.append(
            FindingRepetition(
                repetition=repetition,
                outcome="finding_observed",
                source_receipt_id=source.receipt.receipt_id,
                probe_receipt_id=probe.receipt.receipt_id,
                evidence_pointer_ids=evidence_pointer_ids,
            )
        )
    if category in {"duplicate_effect", "unexpected_effect"}:
        change: Literal["added", "removed", "changed"] = "added"
    elif category == "missing_effect":
        change = "removed"
    elif category == "changed_grounded_effect_argument":
        change = "changed"
    else:
        raise ValueError("dataset behavior adapter requires a concrete behavior finding")
    observed_delta = ObservedDelta(
        kind="action",
        change=change,
        subject_ref=_public_ref("behavior-subject", operator_id, case_id, category),
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
    )


def _build_invariant_package(
    *,
    campaign_id: str,
    case_id: str,
    source_interaction_id: str | None,
    operator_id: str,
    operator_version: str,
    evaluation: DatasetInvariantRuleResult,
    rule_definition: DatasetInvariantRule,
    source_receipts: list[_ReceiptBuild],
    probe_receipts: list[_ReceiptBuild],
    context: FindingAdapterContext,
    fixture_id: str | None = None,
    fixture_version: str | None = None,
) -> FindingEvidencePackage:
    repetitions: list[FindingRepetition] = []
    violated_pointer_ids: list[str] = []
    for source, probe, trial in zip(
        source_receipts,
        probe_receipts,
        evaluation.trials,
        strict=True,
    ):
        category_pointer_ids = probe.category_pointer_ids
        if trial.status == "not_evaluable":
            outcome: Literal["finding_observed", "finding_not_observed", "inconclusive"] = (
                "inconclusive"
            )
            inconclusive_reason = trial.reason_code
        else:
            outcome = "finding_observed" if trial.status == "violated" else "finding_not_observed"
            inconclusive_reason = None
        if outcome == "finding_observed":
            violated_pointer_ids.extend(category_pointer_ids)
        repetitions.append(
            FindingRepetition(
                repetition=trial.repetition,
                outcome=outcome,
                source_receipt_id=source.receipt.receipt_id,
                probe_receipt_id=probe.receipt.receipt_id,
                evidence_pointer_ids=category_pointer_ids,
                inconclusive_reason=inconclusive_reason,
            )
        )
    if not violated_pointer_ids:
        raise ValueError("invariant finding adapter requires at least one violated repetition")
    versioned_rule = _versioned_ref("rule", rule_definition.id, rule_definition.version)
    observed_delta = ObservedDelta(
        kind="rule",
        change="violated",
        subject_ref=_public_ref("rule-subject", rule_definition.id, rule_definition.version),
        rule=versioned_rule,
        source_state="satisfied",
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
    fixture_id: str | None = None,
    fixture_version: str | None = None,
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
    occurrence = build_finding_occurrence(
        kind=kind,
        category=category,
        campaign_ref=_public_ref("campaign", campaign_id),
        source_interaction_ref=(
            _public_ref("source-interaction", source_interaction_id)
            if source_interaction_id is not None
            else None
        ),
        fixture=(
            _versioned_ref("fixture", fixture_id, fixture_version)
            if fixture_id is not None and fixture_version is not None
            else None
        ),
        case_ref=_public_ref("case", case_id),
        operator=_versioned_ref("operator", operator_id, operator_version),
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
    return FindingEvidencePackage(occurrence=occurrence, receipts=receipts)


def _behavior_trial_category_value(
    finding: DatasetEvaluationFinding,
    trial: DatasetEvaluationTrial,
    arm: Literal["source", "probe"],
) -> JsonValue | None:
    if finding.category == "unexpected_effect" and arm == "source":
        return None
    if finding.category == "missing_effect" and arm == "probe":
        return None
    expected_outcomes = finding.expected_effects if arm == "source" else finding.observed_effects
    if not expected_outcomes or trial.observed_frame is None:
        raise ValueError("behavior findings require category evidence from every repetition")
    matching_outcomes = tuple(
        outcome
        for outcome in trial.observed_frame.outcomes
        if outcome.kind == "action"
        and any(
            outcome.id == expected.id
            or (
                outcome.predicate == expected.predicate
                and outcome.fields == expected.fields
                and outcome.position == expected.position
            )
            for expected in expected_outcomes
        )
    )
    if not matching_outcomes:
        raise ValueError("behavior finding evidence does not match its execution repetition")
    return cast(JsonValue, [outcome.model_dump(mode="json") for outcome in matching_outcomes])


def _stateful_rule_evaluation(
    result: StatefulFindingResult,
    rule_definition: DatasetInvariantRule,
) -> DatasetInvariantRuleResult:
    if isinstance(result, CorrectionStressResult):
        rules = result.corrected_invariant_rules
    else:
        rules = result.retried_invariant_rules
    matches = tuple(
        rule
        for rule in rules
        if rule.rule_id == rule_definition.id and rule.rule_version == rule_definition.version
    )
    if len(matches) != 1:
        raise ValueError("stateful result must contain one exact invariant rule")
    return matches[0]


def _stateful_input(
    result: StatefulFindingResult,
    arm: Literal["source", "probe"],
) -> JsonValue:
    turns = result.case.conversation
    selected_turns = turns[:1] if arm == "source" else turns
    return cast(JsonValue, [turn.model_dump(mode="json") for turn in selected_turns])


def _validate_rule_identity(
    evaluation: DatasetInvariantRuleResult,
    rule_definition: DatasetInvariantRule,
) -> None:
    if (
        evaluation.rule_id,
        evaluation.rule_version,
        evaluation.rule_type,
    ) != (rule_definition.id, rule_definition.version, rule_definition.type):
        raise ValueError("invariant evaluation must match its customer rule definition")


def _require_dataset_trial_set(case: DatasetEvaluationCase):
    if case.trial_set is None:
        raise ValueError("dataset finding adapter requires accepted variation trials")
    return case.trial_set


def _versioned_ref(namespace: str, identifier: str, version: str) -> VersionedReference:
    return VersionedReference(
        id=_public_ref(namespace, identifier),
        version=_public_ref(f"{namespace}-version", version),
    )


def _public_ref(namespace: str, *values: str) -> str:
    return f"ulref_v1_{_digest([namespace, *values])}"


def _pointer_id(
    artifact_sha256: str,
    repetition: int,
    arm: str,
    kind: str,
    json_pointer: str,
    source_id: str,
) -> str:
    return f"ulep_v1_{_digest([artifact_sha256, repetition, arm, kind, json_pointer, source_id])}"


def _sha256(value: JsonValue) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def _json_value(value: object) -> JsonValue:
    if isinstance(value, BaseModel):
        return cast(JsonValue, value.model_dump(mode="json"))
    return cast(JsonValue, value)
