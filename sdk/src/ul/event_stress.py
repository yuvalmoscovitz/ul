from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import secrets
import stat
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import ConfigDict, Field, JsonValue, ValidationError, field_validator, model_validator
from ul_core.contracts import SandboxExecutor
from ul_core.dataset import ObservedAgentOutput
from ul_core.evaluation import EvaluationCase, ExecutionEvidence
from ul_core.models import ConversationRole, ConversationTurn, ULModel

from ul.dataset_evaluation import DatasetTargetLifecycleFailure
from ul.dataset_invariants import (
    DatasetInvariantRule,
    DatasetInvariantRuleResult,
    ObservationAuthority,
    evaluate_dataset_invariant_rules,
)
from ul.dataset_regression import (
    DatasetRegressionInvariantSuite,
    DatasetRegressionTargetSnapshot,
    dataset_regression_target_config_sha256,
)
from ul.http_sandbox import JsonHttpSandboxConfig, json_http_sandbox_calls_per_conversation
from ul.sandbox import execution_evidence_requires_quarantine, validate_execution_evidence

_MAXIMUM_CASE_BYTES = 1_000_000
_MAXIMUM_JSON_DEPTH = 100
_CASE_ID_PATTERN = r"^ulmc_v1_[0-9a-f]{64}$"


class _StrictModel(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CorrectionAfterFirstResponseCase(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    id: str = Field(min_length=1, max_length=200)
    operator_id: Literal["conversation.correction_after_first_response"] = (
        "conversation.correction_after_first_response"
    )
    operator_version: Literal["1.0.0"] = "1.0.0"
    conversation: tuple[ConversationTurn, ConversationTurn]

    @field_validator("conversation", mode="before")
    @classmethod
    def accept_json_turn_array(cls, turns: object) -> object:
        return tuple(cast(list[object], turns)) if isinstance(turns, list) else turns

    @model_validator(mode="after")
    def validate_conversation(self) -> Self:
        initial_turn, correction_turn = self.conversation
        if (
            initial_turn.role != ConversationRole.USER
            or correction_turn.role != ConversationRole.USER
        ):
            raise ValueError("correction stress cases require exactly two user turns")
        if not initial_turn.content.strip() or not correction_turn.content.strip():
            raise ValueError("correction stress turns must contain non-whitespace text")
        if initial_turn.id == correction_turn.id:
            raise ValueError("correction stress turn identifiers must be unique")
        return self


class CorrectionTurnObservation(_StrictModel):
    turn: ConversationTurn
    target_output: ObservedAgentOutput
    committed_state_snapshot: JsonValue


class CorrectionDivergence(_StrictModel):
    variation_turn_id: str = Field(min_length=1)
    compared_baseline_turn_id: str = Field(min_length=1)
    baseline_response: JsonValue
    variation_response: JsonValue
    baseline_committed_state: JsonValue
    variation_committed_state: JsonValue
    response_diverged: bool
    committed_state_diverged: bool
    response_changed_from_previous_turn: bool
    committed_state_changed_from_previous_turn: bool


class CorrectionStressTrial(_StrictModel):
    repetition: int = Field(ge=1)
    baseline_execution_evidence: ExecutionEvidence | None = None
    variation_execution_evidence: ExecutionEvidence | None = None
    baseline: tuple[CorrectionTurnObservation, ...] = ()
    variation: tuple[CorrectionTurnObservation, ...] = ()
    divergences: tuple[CorrectionDivergence, ...] = ()
    inconclusive_reason: str | None = None
    lifecycle_failure: DatasetTargetLifecycleFailure | None = None

    @model_validator(mode="after")
    def validate_trial(self) -> Self:
        if self.inconclusive_reason is None:
            if len(self.baseline) != 1 or len(self.variation) != 2 or len(self.divergences) != 2:
                raise ValueError("conclusive correction trials require complete ordered arms")
        elif self.baseline and len(self.baseline) != 1:
            raise ValueError("partial baseline evidence is not allowed")
        return self


class CorrectionStressResult(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    case: CorrectionAfterFirstResponseCase
    requested_repetitions: int = Field(ge=1)
    required_target_calls: int = Field(ge=1)
    status: Literal["passed", "failed", "inconclusive"]
    first_response_divergence_turn_id: str | None = None
    first_committed_state_divergence_turn_id: str | None = None
    response_divergence_stability: Literal["stable", "unstable", "none", "inconclusive"]
    committed_state_divergence_stability: Literal["stable", "unstable", "none", "inconclusive"]
    response_divergence_counts: dict[str, int]
    committed_state_divergence_counts: dict[str, int]
    baseline_drift_observed: bool
    trials: tuple[CorrectionStressTrial, ...] = Field(min_length=1)
    baseline_invariant_rules: tuple[DatasetInvariantRuleResult, ...] = Field(min_length=1)
    corrected_invariant_rules: tuple[DatasetInvariantRuleResult, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if len(self.trials) != self.requested_repetitions or tuple(
            trial.repetition for trial in self.trials
        ) != tuple(range(1, self.requested_repetitions + 1)):
            raise ValueError("correction trials must preserve every repetition in order")
        baseline_statuses = {rule.status for rule in self.baseline_invariant_rules}
        corrected_statuses = {rule.status for rule in self.corrected_invariant_rules}
        expected_status: Literal["passed", "failed", "inconclusive"]
        if baseline_statuses == {"satisfied"} and "violated" in corrected_statuses:
            expected_status = "failed"
        elif (
            baseline_statuses != {"satisfied"}
            or "not_evaluable" in corrected_statuses
            or any(trial.inconclusive_reason is not None for trial in self.trials)
        ):
            expected_status = "inconclusive"
        else:
            expected_status = "passed"
        if self.status != expected_status:
            raise ValueError("correction result status must match its invariant evidence")
        return self


class CorrectionStressPlan(_StrictModel):
    operator_id: Literal["conversation.correction_after_first_response"]
    operator_version: Literal["1.0.0"]
    baseline_turn_count: Literal[1] = 1
    variation_turn_count: Literal[2] = 2
    repetitions: int = Field(ge=1)
    target_calls_per_pair: int = Field(ge=1)
    required_target_calls: int = Field(ge=1)


class RetryAfterSuccessfulCommitCase(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    id: str = Field(min_length=1, max_length=200)
    operator_id: Literal["conversation.retry_after_successful_commit"] = (
        "conversation.retry_after_successful_commit"
    )
    operator_version: Literal["1.0.0"] = "1.0.0"
    conversation: tuple[ConversationTurn, ConversationTurn]

    @field_validator("conversation", mode="before")
    @classmethod
    def accept_json_turn_array(cls, turns: object) -> object:
        return tuple(cast(list[object], turns)) if isinstance(turns, list) else turns

    @model_validator(mode="after")
    def validate_conversation(self) -> Self:
        initial_turn, retry_turn = self.conversation
        if initial_turn.role != ConversationRole.USER or retry_turn.role != ConversationRole.USER:
            raise ValueError("retry stress cases require exactly two user turns")
        if not initial_turn.content.strip() or not retry_turn.content.strip():
            raise ValueError("retry stress turns must contain non-whitespace text")
        if initial_turn.id == retry_turn.id:
            raise ValueError("retry stress turn identifiers must be unique")
        return self


class RetryAfterSuccessfulCommitStressResult(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    case: RetryAfterSuccessfulCommitCase
    requested_repetitions: int = Field(ge=1)
    required_target_calls: int = Field(ge=1)
    status: Literal["passed", "failed", "inconclusive"]
    baseline_drift_observed: bool
    trials: tuple[CorrectionStressTrial, ...] = Field(min_length=1)
    baseline_invariant_rules: tuple[DatasetInvariantRuleResult, ...] = Field(min_length=1)
    successful_commit_invariant_rules: tuple[DatasetInvariantRuleResult, ...] = Field(min_length=1)
    retried_invariant_rules: tuple[DatasetInvariantRuleResult, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if len(self.trials) != self.requested_repetitions or tuple(
            trial.repetition for trial in self.trials
        ) != tuple(range(1, self.requested_repetitions + 1)):
            raise ValueError("retry trials must preserve every repetition in order")
        rule_identities = tuple(
            (rule.rule_id, rule.rule_version, rule.rule_type)
            for rule in self.baseline_invariant_rules
        )
        if any(
            tuple((rule.rule_id, rule.rule_version, rule.rule_type) for rule in rules)
            != rule_identities
            for rules in (
                self.successful_commit_invariant_rules,
                self.retried_invariant_rules,
            )
        ):
            raise ValueError("retry checkpoint invariant rules must match")
        expected_status = _retry_after_successful_commit_status(
            self.trials,
            self.baseline_invariant_rules,
            self.successful_commit_invariant_rules,
            self.retried_invariant_rules,
            baseline_drift_observed=self.baseline_drift_observed,
        )
        if self.status != expected_status:
            raise ValueError("retry result status must match its checkpoint evidence")
        return self


class RetryAfterSuccessfulCommitStressPlan(_StrictModel):
    operator_id: Literal["conversation.retry_after_successful_commit"]
    operator_version: Literal["1.0.0"]
    baseline_turn_count: Literal[1] = 1
    variation_turn_count: Literal[2] = 2
    repetitions: int = Field(ge=1)
    target_calls_per_pair: int = Field(ge=1)
    required_target_calls: int = Field(ge=1)


class MultiTurnRegressionCase(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    case_id: str = Field(pattern=_CASE_ID_PATTERN)
    stress_case: CorrectionAfterFirstResponseCase
    target: DatasetRegressionTargetSnapshot
    invariant_suite: DatasetRegressionInvariantSuite
    repetitions: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_case_id(self) -> Self:
        content = self.model_dump(mode="json", exclude={"case_id"})
        if self.case_id != _multi_turn_case_id(cast(dict[str, JsonValue], content)):
            raise ValueError("multi-turn regression case ID must match its canonical content")
        if len(self.model_dump_json().encode("utf-8")) > _MAXIMUM_CASE_BYTES:
            raise ValueError("multi-turn regression case exceeds the size limit")
        return self


async def run_correction_stress_test(
    case: CorrectionAfterFirstResponseCase,
    sandbox: SandboxExecutor,
    *,
    invariant_rules: tuple[DatasetInvariantRule, ...],
    observation_authority: ObservationAuthority = "committed_state_snapshot",
    repetitions: int = 3,
    max_sandbox_api_calls: int = 100,
    allow_network_egress: bool = False,
) -> CorrectionStressResult:
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if not invariant_rules:
        raise ValueError("correction stress testing requires at least one invariant")
    if type(max_sandbox_api_calls) is not int or max_sandbox_api_calls < 1:
        raise ValueError("max_sandbox_api_calls must be a positive integer")
    if not allow_network_egress:
        raise ValueError("correction stress sandbox API access requires explicit network opt-in")
    if not sandbox.capabilities.supports_conversations:
        raise ValueError("correction stress testing requires conversation support")
    if not sandbox.capabilities.supports_state_observation:
        raise ValueError("correction stress testing requires state observation support")

    initial_turn, correction_turn = case.conversation
    planned_baseline = _evaluation_case(
        turns=(initial_turn,),
        max_sandbox_api_calls=max_sandbox_api_calls,
        sandbox=sandbox,
    )
    planned_variation = _evaluation_case(
        turns=(initial_turn, correction_turn),
        max_sandbox_api_calls=max_sandbox_api_calls,
        sandbox=sandbox,
    )
    baseline_calls = sandbox.api_calls_for_case(planned_baseline)
    variation_calls = sandbox.api_calls_for_case(planned_variation)
    if (
        type(baseline_calls) is not int
        or baseline_calls < 1
        or type(variation_calls) is not int
        or variation_calls < 1
    ):
        raise ValueError("correction stress sandbox returned an invalid API call count")
    required_target_calls = repetitions * (baseline_calls + variation_calls)
    if required_target_calls > max_sandbox_api_calls:
        raise ValueError("correction stress test exceeds the authorized target call budget")

    trials: list[CorrectionStressTrial] = []
    baseline_final_outputs: list[ObservedAgentOutput | None] = []
    final_outputs: list[ObservedAgentOutput | None] = []
    sandbox_state_uncertain = False
    for repetition in range(1, repetitions + 1):
        baseline: tuple[CorrectionTurnObservation, ...] = ()
        baseline_evidence: ExecutionEvidence | None = None
        variation_evidence: ExecutionEvidence | None = None
        if sandbox_state_uncertain:
            baseline_final_outputs.append(None)
            final_outputs.append(None)
            trials.append(
                CorrectionStressTrial(
                    repetition=repetition,
                    inconclusive_reason=(
                        "sandbox not called because prior execution left state uncertain"
                    ),
                )
            )
            continue
        try:
            baseline_case = _evaluation_case(
                turns=(initial_turn,),
                max_sandbox_api_calls=baseline_calls,
                sandbox=sandbox,
            )
            async with asyncio.timeout(baseline_case.timeout_seconds):
                baseline_evidence = await sandbox.execute(baseline_case)
            validate_execution_evidence(baseline_case, sandbox, baseline_evidence)
            sandbox_state_uncertain = execution_evidence_requires_quarantine(baseline_evidence)
            if baseline_evidence.lifecycle.terminal_status != "succeeded":
                baseline_final_outputs.append(None)
                final_outputs.append(None)
                trials.append(
                    CorrectionStressTrial(
                        repetition=repetition,
                        baseline_execution_evidence=baseline_evidence,
                        inconclusive_reason="sandbox lifecycle failed",
                        lifecycle_failure=_lifecycle_failure(baseline_evidence),
                    )
                )
                continue
            baseline = _observations((initial_turn,), baseline_case, baseline_evidence)

            variation_case = _evaluation_case(
                turns=(initial_turn, correction_turn),
                max_sandbox_api_calls=variation_calls,
                sandbox=sandbox,
            )
            async with asyncio.timeout(variation_case.timeout_seconds):
                variation_evidence = await sandbox.execute(variation_case)
            validate_execution_evidence(variation_case, sandbox, variation_evidence)
            sandbox_state_uncertain = execution_evidence_requires_quarantine(variation_evidence)
            if variation_evidence.lifecycle.terminal_status != "succeeded":
                baseline_final_outputs.append(baseline[-1].target_output)
                final_outputs.append(None)
                trials.append(
                    CorrectionStressTrial(
                        repetition=repetition,
                        baseline_execution_evidence=baseline_evidence,
                        variation_execution_evidence=variation_evidence,
                        baseline=baseline,
                        inconclusive_reason="sandbox lifecycle failed",
                        lifecycle_failure=_lifecycle_failure(variation_evidence),
                    )
                )
                continue
            variation = _observations(case.conversation, variation_case, variation_evidence)
        except TimeoutError:
            sandbox_state_uncertain = sandbox.capabilities.cancellation_guarantee != "guaranteed"
            baseline_final_outputs.append(baseline[-1].target_output if baseline else None)
            final_outputs.append(None)
            trials.append(
                CorrectionStressTrial(
                    repetition=repetition,
                    baseline_execution_evidence=baseline_evidence,
                    variation_execution_evidence=variation_evidence,
                    baseline=baseline,
                    inconclusive_reason="sandbox execution timed out",
                )
            )
            continue
        except RuntimeError:
            baseline_final_outputs.append(baseline[-1].target_output if baseline else None)
            final_outputs.append(None)
            trials.append(
                CorrectionStressTrial(
                    repetition=repetition,
                    baseline_execution_evidence=baseline_evidence,
                    variation_execution_evidence=variation_evidence,
                    baseline=baseline,
                    inconclusive_reason="sandbox execution failed",
                )
            )
            continue
        baseline_final_outputs.append(baseline[-1].target_output)
        final_outputs.append(variation[-1].target_output)
        trials.append(
            CorrectionStressTrial(
                repetition=repetition,
                baseline_execution_evidence=baseline_evidence,
                variation_execution_evidence=variation_evidence,
                baseline=baseline,
                variation=variation,
                divergences=_divergences(baseline[0], variation),
            )
        )

    baseline_invariant_results = evaluate_dataset_invariant_rules(
        invariant_rules,
        tuple(baseline_final_outputs),
        observation_authority=observation_authority,
    )
    corrected_invariant_results = evaluate_dataset_invariant_rules(
        invariant_rules,
        tuple(final_outputs),
        observation_authority=observation_authority,
    )
    baseline_statuses = {rule.status for rule in baseline_invariant_results}
    corrected_statuses = {rule.status for rule in corrected_invariant_results}
    status: Literal["passed", "failed", "inconclusive"]
    if baseline_statuses == {"satisfied"} and "violated" in corrected_statuses:
        status = "failed"
    elif (
        baseline_statuses != {"satisfied"}
        or "not_evaluable" in corrected_statuses
        or any(trial.inconclusive_reason is not None for trial in trials)
    ):
        status = "inconclusive"
    else:
        status = "passed"
    turn_ids = tuple(turn.id for turn in case.conversation)
    (
        first_response_divergence_turn_id,
        response_divergence_counts,
        response_divergence_stability,
    ) = _summarize_divergences(tuple(trials), turn_ids, "response_diverged")
    (
        first_committed_state_divergence_turn_id,
        committed_state_divergence_counts,
        committed_state_divergence_stability,
    ) = _summarize_divergences(tuple(trials), turn_ids, "committed_state_diverged")
    return CorrectionStressResult(
        case=case,
        requested_repetitions=repetitions,
        required_target_calls=required_target_calls,
        status=status,
        first_response_divergence_turn_id=first_response_divergence_turn_id,
        first_committed_state_divergence_turn_id=first_committed_state_divergence_turn_id,
        response_divergence_stability=response_divergence_stability,
        committed_state_divergence_stability=committed_state_divergence_stability,
        response_divergence_counts=response_divergence_counts,
        committed_state_divergence_counts=committed_state_divergence_counts,
        baseline_drift_observed=(
            response_divergence_counts.get(initial_turn.id, 0) > 0
            or committed_state_divergence_counts.get(initial_turn.id, 0) > 0
        ),
        trials=tuple(trials),
        baseline_invariant_rules=baseline_invariant_results,
        corrected_invariant_rules=corrected_invariant_results,
    )


def plan_correction_stress_test(
    case: CorrectionAfterFirstResponseCase,
    target_config: JsonHttpSandboxConfig,
    *,
    repetitions: int = 3,
    max_sandbox_api_calls: int = 100,
) -> CorrectionStressPlan:
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if type(max_sandbox_api_calls) is not int or max_sandbox_api_calls < 1:
        raise ValueError("max_sandbox_api_calls must be a positive integer")
    target_calls_per_pair = json_http_sandbox_calls_per_conversation(
        target_config, 1
    ) + json_http_sandbox_calls_per_conversation(target_config, 2)
    required_target_calls = repetitions * target_calls_per_pair
    if required_target_calls > max_sandbox_api_calls:
        raise ValueError("correction stress test exceeds the authorized target call budget")
    return CorrectionStressPlan(
        operator_id=case.operator_id,
        operator_version=case.operator_version,
        repetitions=repetitions,
        target_calls_per_pair=target_calls_per_pair,
        required_target_calls=required_target_calls,
    )


async def run_retry_after_successful_commit_stress_test(
    case: RetryAfterSuccessfulCommitCase,
    sandbox: SandboxExecutor,
    *,
    invariant_rules: tuple[DatasetInvariantRule, ...],
    observation_authority: ObservationAuthority = "committed_state_snapshot",
    repetitions: int = 3,
    max_sandbox_api_calls: int = 100,
    allow_network_egress: bool = False,
) -> RetryAfterSuccessfulCommitStressResult:
    if observation_authority != "committed_state_snapshot":
        raise ValueError("retry stress testing requires committed-state invariant observation")
    paired_result = await run_correction_stress_test(
        CorrectionAfterFirstResponseCase(id=case.id, conversation=case.conversation),
        sandbox,
        invariant_rules=invariant_rules,
        observation_authority=observation_authority,
        repetitions=repetitions,
        max_sandbox_api_calls=max_sandbox_api_calls,
        allow_network_egress=allow_network_egress,
    )
    successful_commit_outputs = tuple(
        trial.variation[0].target_output
        if trial.inconclusive_reason is None and len(trial.variation) == 2
        else None
        for trial in paired_result.trials
    )
    successful_commit_invariant_rules = evaluate_dataset_invariant_rules(
        invariant_rules,
        successful_commit_outputs,
        observation_authority=observation_authority,
    )
    status = _retry_after_successful_commit_status(
        paired_result.trials,
        paired_result.baseline_invariant_rules,
        successful_commit_invariant_rules,
        paired_result.corrected_invariant_rules,
        baseline_drift_observed=paired_result.baseline_drift_observed,
    )
    return RetryAfterSuccessfulCommitStressResult(
        case=case,
        requested_repetitions=repetitions,
        required_target_calls=paired_result.required_target_calls,
        status=status,
        baseline_drift_observed=paired_result.baseline_drift_observed,
        trials=paired_result.trials,
        baseline_invariant_rules=paired_result.baseline_invariant_rules,
        successful_commit_invariant_rules=successful_commit_invariant_rules,
        retried_invariant_rules=paired_result.corrected_invariant_rules,
    )


def plan_retry_after_successful_commit_stress_test(
    case: RetryAfterSuccessfulCommitCase,
    target_config: JsonHttpSandboxConfig,
    *,
    repetitions: int = 3,
    max_sandbox_api_calls: int = 100,
) -> RetryAfterSuccessfulCommitStressPlan:
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if type(max_sandbox_api_calls) is not int or max_sandbox_api_calls < 1:
        raise ValueError("max_sandbox_api_calls must be a positive integer")
    target_calls_per_pair = json_http_sandbox_calls_per_conversation(
        target_config, 1
    ) + json_http_sandbox_calls_per_conversation(target_config, 2)
    required_target_calls = repetitions * target_calls_per_pair
    if required_target_calls > max_sandbox_api_calls:
        raise ValueError("retry stress test exceeds the authorized target call budget")
    return RetryAfterSuccessfulCommitStressPlan(
        operator_id=case.operator_id,
        operator_version=case.operator_version,
        repetitions=repetitions,
        target_calls_per_pair=target_calls_per_pair,
        required_target_calls=required_target_calls,
    )


def _retry_after_successful_commit_status(
    trials: tuple[CorrectionStressTrial, ...],
    baseline_rules: tuple[DatasetInvariantRuleResult, ...],
    successful_commit_rules: tuple[DatasetInvariantRuleResult, ...],
    retried_rules: tuple[DatasetInvariantRuleResult, ...],
    *,
    baseline_drift_observed: bool,
) -> Literal["passed", "failed", "inconclusive"]:
    baseline_statuses = {rule.status for rule in baseline_rules}
    successful_commit_statuses = {rule.status for rule in successful_commit_rules}
    retried_statuses = {rule.status for rule in retried_rules}
    evidence_is_inconclusive = (
        baseline_drift_observed
        or any(trial.inconclusive_reason is not None for trial in trials)
        or baseline_statuses != {"satisfied"}
        or successful_commit_statuses != {"satisfied"}
        or "not_evaluable" in retried_statuses
    )
    if evidence_is_inconclusive:
        return "inconclusive"
    if any(all(trial.status == "violated" for trial in rule.trials) for rule in retried_rules):
        return "failed"
    if "violated" in retried_statuses:
        return "inconclusive"
    return "passed"


def create_multi_turn_regression_case(
    *,
    stress_case: CorrectionAfterFirstResponseCase,
    target_config: JsonHttpSandboxConfig,
    source_suite_sha256: str,
    observation_authority: ObservationAuthority,
    invariant_rules: tuple[DatasetInvariantRule, ...],
    repetitions: int,
) -> MultiTurnRegressionCase:
    target = DatasetRegressionTargetSnapshot(
        provenance="declared_at_case_creation",
        config=target_config,
        config_sha256=dataset_regression_target_config_sha256(target_config),
    )
    invariant_suite = DatasetRegressionInvariantSuite(
        source_suite_sha256=source_suite_sha256,
        observation_source="target_output",
        observation_authority=observation_authority,
        state_observation_authority=(
            "sandbox_self_reported" if observation_authority == "committed_state_snapshot" else None
        ),
        rules=invariant_rules,
    )
    content = cast(
        dict[str, JsonValue],
        {
            "schema_version": "1.0.0",
            "stress_case": stress_case.model_dump(mode="json"),
            "target": target.model_dump(mode="json"),
            "invariant_suite": invariant_suite.model_dump(mode="json"),
            "repetitions": repetitions,
        },
    )
    return MultiTurnRegressionCase(
        case_id=_multi_turn_case_id(content),
        stress_case=stress_case,
        target=target,
        invariant_suite=invariant_suite,
        repetitions=repetitions,
    )


async def replay_multi_turn_regression(
    case: MultiTurnRegressionCase,
    sandbox: SandboxExecutor,
    *,
    max_sandbox_api_calls: int = 100,
    allow_network_egress: bool = False,
) -> CorrectionStressResult:
    if case.invariant_suite.state_observation_authority is not None and (
        case.invariant_suite.state_observation_authority
        != sandbox.capabilities.state_observation_authority
        or case.invariant_suite.state_observer_id != sandbox.capabilities.state_observer_id
    ):
        raise ValueError("sandbox state authority does not match the multi-turn regression case")
    return await run_correction_stress_test(
        case.stress_case,
        sandbox,
        invariant_rules=case.invariant_suite.rules,
        observation_authority=case.invariant_suite.observation_authority,
        repetitions=case.repetitions,
        max_sandbox_api_calls=max_sandbox_api_calls,
        allow_network_egress=allow_network_egress,
    )


def load_multi_turn_regression_case(path: str | Path) -> MultiTurnRegressionCase:
    try:
        encoded = _read_bounded_regular_file(Path(path), _MAXIMUM_CASE_BYTES)
        raw = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonstandard_json_constant,
            parse_float=_parse_finite_float,
        )
        _reject_deep_json(raw)
        return MultiTurnRegressionCase.model_validate(raw)
    except OSError:
        raise RuntimeError("multi-turn regression case could not be read") from None
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValidationError, ValueError):
        raise ValueError("multi-turn regression case is invalid") from None


def load_correction_after_first_response_case(
    path: str | Path,
) -> CorrectionAfterFirstResponseCase:
    try:
        encoded = _read_bounded_regular_file(Path(path), _MAXIMUM_CASE_BYTES)
        raw = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonstandard_json_constant,
            parse_float=_parse_finite_float,
        )
        _reject_deep_json(raw)
        return CorrectionAfterFirstResponseCase.model_validate(raw)
    except OSError:
        raise RuntimeError("correction stress case could not be read") from None
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValidationError, ValueError):
        raise ValueError("correction stress case is invalid") from None


def load_retry_after_successful_commit_case(
    path: str | Path,
) -> RetryAfterSuccessfulCommitCase:
    try:
        encoded = _read_bounded_regular_file(Path(path), _MAXIMUM_CASE_BYTES)
        raw = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonstandard_json_constant,
            parse_float=_parse_finite_float,
        )
        _reject_deep_json(raw)
        return RetryAfterSuccessfulCommitCase.model_validate(raw)
    except OSError:
        raise RuntimeError("retry stress case could not be read") from None
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValidationError, ValueError):
        raise ValueError("retry stress case is invalid") from None


def _observations(
    turns: tuple[ConversationTurn, ...],
    evaluation_case: EvaluationCase,
    evidence: ExecutionEvidence,
) -> tuple[CorrectionTurnObservation, ...]:
    if len(turns) != len(evaluation_case.turns) or len(turns) != len(evidence.turns):
        raise RuntimeError("sandbox returned an invalid number of turn observations")
    observations: list[CorrectionTurnObservation] = []
    before_turn_state = evidence.initial_state.value if evidence.initial_state is not None else None
    before_turn_state_present = evidence.initial_state is not None
    for turn, sandbox_turn, turn_evidence in zip(
        turns, evaluation_case.turns, evidence.turns, strict=True
    ):
        if turn_evidence.turn_id != sandbox_turn.id:
            raise RuntimeError("sandbox returned turn evidence out of order")
        if turn_evidence.state_snapshot is None:
            raise RuntimeError("sandbox omitted a committed state snapshot")
        try:
            output = ObservedAgentOutput(
                raw_output=turn_evidence.response,
                metadata={
                    "committed_state_snapshot": turn_evidence.state_snapshot,
                    "state_observation_authority": turn_evidence.state_observation_authority,
                    **(
                        {"committed_state_before_turn": before_turn_state}
                        if before_turn_state_present
                        else {}
                    ),
                },
            )
        except ValidationError:
            raise RuntimeError("sandbox returned invalid turn evidence") from None
        observations.append(
            CorrectionTurnObservation(
                turn=turn,
                target_output=output,
                committed_state_snapshot=turn_evidence.state_snapshot,
            )
        )
        before_turn_state = turn_evidence.state_snapshot
        before_turn_state_present = True
    return tuple(observations)


def _divergences(
    baseline: CorrectionTurnObservation,
    variation: tuple[CorrectionTurnObservation, ...],
) -> tuple[CorrectionDivergence, ...]:
    divergences: list[CorrectionDivergence] = []
    previous = baseline
    for turn_index, observation in enumerate(variation):
        comparison = baseline
        divergences.append(
            CorrectionDivergence(
                variation_turn_id=observation.turn.id,
                compared_baseline_turn_id=comparison.turn.id,
                baseline_response=comparison.target_output.raw_output,
                variation_response=observation.target_output.raw_output,
                baseline_committed_state=comparison.committed_state_snapshot,
                variation_committed_state=observation.committed_state_snapshot,
                response_diverged=observation.target_output.raw_output
                != comparison.target_output.raw_output,
                committed_state_diverged=observation.committed_state_snapshot
                != comparison.committed_state_snapshot,
                response_changed_from_previous_turn=(
                    False
                    if turn_index == 0
                    else observation.target_output.raw_output != previous.target_output.raw_output
                ),
                committed_state_changed_from_previous_turn=(
                    False
                    if turn_index == 0
                    else observation.committed_state_snapshot != previous.committed_state_snapshot
                ),
            )
        )
        previous = observation
    return tuple(divergences)


def _summarize_divergences(
    trials: tuple[CorrectionStressTrial, ...],
    turn_ids: tuple[str, ...],
    field: Literal["response_diverged", "committed_state_diverged"],
) -> tuple[
    str | None,
    dict[str, int],
    Literal["stable", "unstable", "none", "inconclusive"],
]:
    conclusive_trials = tuple(trial for trial in trials if trial.inconclusive_reason is None)
    counts = {
        turn_id: sum(
            any(
                divergence.variation_turn_id == turn_id and getattr(divergence, field)
                for divergence in trial.divergences
            )
            for trial in conclusive_trials
        )
        for turn_id in turn_ids
    }
    first_turn_id = next((turn_id for turn_id in turn_ids if counts[turn_id] > 0), None)
    if not conclusive_trials:
        stability: Literal["stable", "unstable", "none", "inconclusive"] = "inconclusive"
    else:
        divergence_patterns = {
            tuple(
                turn_id
                for turn_id in turn_ids
                if any(
                    divergence.variation_turn_id == turn_id and getattr(divergence, field)
                    for divergence in trial.divergences
                )
            )
            for trial in conclusive_trials
        }
        if divergence_patterns == {()}:
            stability = "none"
        elif len(divergence_patterns) == 1:
            stability = "stable"
        else:
            stability = "unstable"
    return first_turn_id, counts, stability


def _evaluation_case(
    *,
    turns: tuple[ConversationTurn, ...],
    max_sandbox_api_calls: int,
    sandbox: SandboxExecutor,
) -> EvaluationCase:
    return EvaluationCase(
        id=f"ul-case-{secrets.token_hex(16)}",
        turns=tuple(
            ConversationTurn(
                id=f"turn-{index}",
                role=ConversationRole.USER,
                content=turn.content,
            )
            for index, turn in enumerate(turns, start=1)
        ),
        max_sandbox_api_calls=max_sandbox_api_calls,
        timeout_seconds=30,
        required_state_observation_authority=(sandbox.capabilities.state_observation_authority),
        required_state_observer_id=sandbox.capabilities.state_observer_id,
    )


def _lifecycle_failure(evidence: ExecutionEvidence) -> DatasetTargetLifecycleFailure:
    lifecycle = evidence.lifecycle
    return DatasetTargetLifecycleFailure(
        failed_phase=lifecycle.failed_phase or "unknown",
        completed_phases=lifecycle.completed_phases,
        cleanup_reset_failed=lifecycle.cleanup == "failed",
        sandbox_state_may_remain=lifecycle.sandbox_state_uncertain,
    )


def _multi_turn_case_id(content: dict[str, JsonValue]) -> str:
    encoded = json.dumps(
        content,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()
    return f"ulmc_v1_{hashlib.sha256(encoded).hexdigest()}"


def _read_bounded_regular_file(path: Path, maximum_bytes: int) -> bytes:
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    requires_identity_check = no_follow == 0
    if requires_identity_check and stat.S_ISLNK(os.lstat(path).st_mode):
        raise OSError("case input is a symbolic link")
    binary_flag = os.O_BINARY if os.name == "nt" else 0
    descriptor = os.open(path, os.O_RDONLY | no_follow | binary_flag)
    try:
        descriptor_status = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_status.st_mode):
            raise OSError("case input is not a regular file")
        if requires_identity_check:
            path_status = os.lstat(path)
            if stat.S_ISLNK(path_status.st_mode) or not os.path.samestat(
                descriptor_status, path_status
            ):
                raise OSError("case input changed while opening")
        encoded = os.read(descriptor, maximum_bytes + 1)
        if len(encoded) > maximum_bytes:
            raise ValueError("multi-turn regression case exceeds the size limit")
        return encoded
    finally:
        os.close(descriptor)


def _reject_duplicate_object_keys(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _reject_deep_json(value: object, depth: int = 0) -> None:
    if depth > _MAXIMUM_JSON_DEPTH:
        raise ValueError("JSON exceeds the nesting limit")
    if isinstance(value, dict):
        for child in cast(dict[str, object], value).values():
            _reject_deep_json(child, depth + 1)
    elif isinstance(value, list):
        for child in cast(list[object], value):
            _reject_deep_json(child, depth + 1)
