from __future__ import annotations

import asyncio
import json
import math
import os
import secrets
import stat
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import ConfigDict, Field, JsonValue, ValidationError, model_validator
from ul_core.contracts import EnvironmentExecutor
from ul_core.dataset import ObservedAgentOutput
from ul_core.evaluation import (
    EvaluationCase,
    ExecutionEvidence,
    TimeoutAfterCommitEventRequest,
)
from ul_core.models import ConversationRole, ConversationTurn, ULModel

from ul.dataset_evaluation import DatasetTargetLifecycleFailure
from ul.dataset_invariants import (
    DatasetInvariantRule,
    DatasetInvariantRuleResult,
    ObservationAuthority,
    evaluate_dataset_invariant_rules,
)
from ul.environment import execution_evidence_requires_quarantine, validate_execution_evidence
from ul.http_environment import (
    JsonHttpEnvironmentConfig,
    json_http_environment_calls_per_conversation,
)

_MAXIMUM_CASE_BYTES = 1_000_000
_MAXIMUM_JSON_DEPTH = 100


class _StrictModel(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TimeoutAfterCommitCase(_StrictModel):
    schema_version: Literal["1.1.0"] = "1.1.0"
    id: str = Field(min_length=1, max_length=200)
    operator_id: Literal["environment.tool.timeout_after_commit"] = (
        "environment.tool.timeout_after_commit"
    )
    operator_version: Literal["1.0.0"] = "1.0.0"
    event_id: str = Field(min_length=1, max_length=500)
    action_id: str = Field(min_length=1, max_length=500)
    turn: ConversationTurn

    @model_validator(mode="after")
    def validate_turn(self) -> Self:
        if self.turn.role != ConversationRole.USER:
            raise ValueError("timeout-after-commit cases require one user turn")
        if not self.turn.content.strip():
            raise ValueError("timeout-after-commit case input must contain non-whitespace text")
        return self


class TimeoutAfterCommitStressTrial(_StrictModel):
    repetition: int = Field(ge=1)
    execution_evidence: ExecutionEvidence | None = None
    inconclusive_reason: str | None = Field(default=None, min_length=1, max_length=500)
    lifecycle_failure: DatasetTargetLifecycleFailure | None = None

    @model_validator(mode="after")
    def validate_trial(self) -> Self:
        if self.inconclusive_reason is not None:
            return self
        if self.execution_evidence is None:
            raise ValueError("conclusive timeout-after-commit trials require execution evidence")
        event = self.execution_evidence.timeout_after_commit_event
        if (
            self.execution_evidence.lifecycle.terminal_status != "succeeded"
            or event is None
            or not event.armed
            or event.trigger_status != "fired"
            or not event.cleaned
        ):
            raise ValueError("conclusive timeout-after-commit trials require a fired clean event")
        return self


class TimeoutAfterCommitStressResult(_StrictModel):
    schema_version: Literal["1.1.0"] = "1.1.0"
    case: TimeoutAfterCommitCase
    requested_repetitions: int = Field(ge=1)
    target_calls_per_repetition: int = Field(ge=1)
    required_target_calls: int = Field(ge=1)
    status: Literal["passed", "failed", "inconclusive"]
    trials: tuple[TimeoutAfterCommitStressTrial, ...] = Field(min_length=1)
    invariant_rules: tuple[DatasetInvariantRuleResult, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if len(self.trials) != self.requested_repetitions or tuple(
            trial.repetition for trial in self.trials
        ) != tuple(range(1, self.requested_repetitions + 1)):
            raise ValueError("timeout-after-commit trials must preserve every repetition in order")
        if self.required_target_calls != (
            self.requested_repetitions * self.target_calls_per_repetition
        ):
            raise ValueError("timeout-after-commit target call totals are inconsistent")
        expected_repetitions = tuple(range(1, self.requested_repetitions + 1))
        if any(
            tuple(trial.repetition for trial in rule.trials) != expected_repetitions
            for rule in self.invariant_rules
        ):
            raise ValueError("timeout-after-commit invariant evidence must cover every repetition")
        expected_event_identity = (
            self.case.operator_id,
            self.case.operator_version,
            self.case.event_id,
            self.case.turn.id,
            self.case.action_id,
        )
        for trial in self.trials:
            evidence = trial.execution_evidence
            event = evidence.timeout_after_commit_event if evidence is not None else None
            if (
                event is not None
                and (
                    event.operator_id,
                    event.operator_version,
                    event.event_id,
                    event.turn_id,
                    event.action_id,
                )
                != expected_event_identity
            ):
                raise ValueError("timeout-after-commit trial evidence does not match its case")
        expected_status = _timeout_after_commit_status(self.invariant_rules, self.trials)
        if self.status != expected_status:
            raise ValueError("timeout-after-commit status must match its invariant evidence")
        return self


class TimeoutAfterCommitStressPlan(_StrictModel):
    operator_id: Literal["environment.tool.timeout_after_commit"] = (
        "environment.tool.timeout_after_commit"
    )
    operator_version: Literal["1.0.0"] = "1.0.0"
    repetitions: int = Field(ge=1)
    target_calls_per_repetition: int = Field(ge=1)
    required_target_calls: int = Field(ge=1)


def plan_timeout_after_commit_stress_test(
    case: TimeoutAfterCommitCase,
    target_config: JsonHttpEnvironmentConfig,
    *,
    repetitions: int = 3,
    max_environment_api_calls: int = 100,
) -> TimeoutAfterCommitStressPlan:
    _validate_run_inputs(repetitions, max_environment_api_calls)
    event_config = target_config.timeout_after_commit
    if event_config is None or event_config.version != case.operator_version:
        raise ValueError("environment does not support environment.tool.timeout_after_commit@1.0.0")
    target_calls_per_repetition = json_http_environment_calls_per_conversation(target_config, 1) + 3
    required_target_calls = repetitions * target_calls_per_repetition
    if required_target_calls > max_environment_api_calls:
        raise ValueError("timeout-after-commit test exceeds the authorized target call budget")
    return TimeoutAfterCommitStressPlan(
        repetitions=repetitions,
        target_calls_per_repetition=target_calls_per_repetition,
        required_target_calls=required_target_calls,
    )


async def run_timeout_after_commit_stress_test(
    case: TimeoutAfterCommitCase,
    environment: EnvironmentExecutor,
    *,
    invariant_rules: tuple[DatasetInvariantRule, ...],
    observation_authority: ObservationAuthority = "committed_state_snapshot",
    repetitions: int = 3,
    max_environment_api_calls: int = 100,
    allow_network_egress: bool = False,
) -> TimeoutAfterCommitStressResult:
    _validate_run_inputs(repetitions, max_environment_api_calls)
    if not invariant_rules:
        raise ValueError("timeout-after-commit testing requires at least one invariant")
    if observation_authority != "committed_state_snapshot":
        raise ValueError("timeout-after-commit testing requires committed-state invariants")
    if not allow_network_egress:
        raise ValueError(
            "timeout-after-commit environment API access requires explicit network opt-in"
        )
    if not environment.capabilities.supports_conversations:
        raise ValueError("timeout-after-commit testing requires conversation support")
    if not environment.capabilities.supports_state_observation:
        raise ValueError("timeout-after-commit testing requires state observation support")
    if environment.capabilities.timeout_after_commit_version != case.operator_version:
        raise ValueError("environment does not support environment.tool.timeout_after_commit@1.0.0")

    planned_case = _evaluation_case(case, max_environment_api_calls, environment)
    target_calls_per_repetition = environment.api_calls_for_case(planned_case)
    if type(target_calls_per_repetition) is not int or target_calls_per_repetition < 1:
        raise ValueError("timeout-after-commit environment returned an invalid API call count")
    required_target_calls = repetitions * target_calls_per_repetition
    if required_target_calls > max_environment_api_calls:
        raise ValueError("timeout-after-commit test exceeds the authorized target call budget")

    trials: list[TimeoutAfterCommitStressTrial] = []
    final_outputs: list[ObservedAgentOutput | None] = []
    environment_state_uncertain = False
    for repetition in range(1, repetitions + 1):
        if environment_state_uncertain:
            final_outputs.append(None)
            trials.append(
                TimeoutAfterCommitStressTrial(
                    repetition=repetition,
                    inconclusive_reason=(
                        "environment not called because prior execution left state uncertain"
                    ),
                )
            )
            continue
        evidence: ExecutionEvidence | None = None
        try:
            evaluation_case = _evaluation_case(case, target_calls_per_repetition, environment)
            async with asyncio.timeout(evaluation_case.timeout_seconds):
                evidence = await environment.execute(evaluation_case)
            validate_execution_evidence(evaluation_case, environment, evidence)
            environment_state_uncertain = execution_evidence_requires_quarantine(evidence)
        except TimeoutError:
            environment_state_uncertain = (
                environment.capabilities.cancellation_guarantee != "guaranteed"
            )
            final_outputs.append(None)
            trials.append(
                TimeoutAfterCommitStressTrial(
                    repetition=repetition,
                    execution_evidence=evidence,
                    inconclusive_reason="environment execution timed out",
                )
            )
            continue
        except (RuntimeError, ValueError):
            environment_state_uncertain = True
            final_outputs.append(None)
            trials.append(
                TimeoutAfterCommitStressTrial(
                    repetition=repetition,
                    execution_evidence=evidence,
                    inconclusive_reason="environment execution or evidence validation failed",
                )
            )
            continue

        if evidence.lifecycle.terminal_status != "succeeded":
            final_outputs.append(None)
            trials.append(
                TimeoutAfterCommitStressTrial(
                    repetition=repetition,
                    execution_evidence=evidence,
                    inconclusive_reason="environment lifecycle failed",
                    lifecycle_failure=_lifecycle_failure(evidence),
                )
            )
            continue
        event = evidence.timeout_after_commit_event
        if event is None or not event.armed:
            final_outputs.append(None)
            trials.append(
                TimeoutAfterCommitStressTrial(
                    repetition=repetition,
                    execution_evidence=evidence,
                    inconclusive_reason="timeout-after-commit event was not armed",
                )
            )
            continue
        if event.trigger_status != "fired":
            final_outputs.append(None)
            trials.append(
                TimeoutAfterCommitStressTrial(
                    repetition=repetition,
                    execution_evidence=evidence,
                    inconclusive_reason="timeout-after-commit event was requested but did not fire",
                )
            )
            continue
        if not event.cleaned:
            final_outputs.append(None)
            trials.append(
                TimeoutAfterCommitStressTrial(
                    repetition=repetition,
                    execution_evidence=evidence,
                    inconclusive_reason="timeout-after-commit event cleanup was not acknowledged",
                )
            )
            continue
        if evidence.final_state is None:
            final_outputs.append(None)
            trials.append(
                TimeoutAfterCommitStressTrial(
                    repetition=repetition,
                    execution_evidence=evidence,
                    inconclusive_reason="environment omitted authoritative post-state evidence",
                )
            )
            continue
        final_outputs.append(
            ObservedAgentOutput(
                raw_output=evidence.final_response,
                metadata={
                    "committed_state_snapshot": evidence.final_state.value,
                    "state_observation_authority": evidence.final_state.authority,
                },
            )
        )
        trials.append(
            TimeoutAfterCommitStressTrial(
                repetition=repetition,
                execution_evidence=evidence,
            )
        )

    invariant_results = evaluate_dataset_invariant_rules(
        invariant_rules,
        tuple(final_outputs),
        observation_authority=observation_authority,
    )
    status = _timeout_after_commit_status(invariant_results, tuple(trials))
    return TimeoutAfterCommitStressResult(
        case=case,
        requested_repetitions=repetitions,
        target_calls_per_repetition=target_calls_per_repetition,
        required_target_calls=required_target_calls,
        status=status,
        trials=tuple(trials),
        invariant_rules=invariant_results,
    )


def load_timeout_after_commit_case(path: str | Path) -> TimeoutAfterCommitCase:
    try:
        encoded = _read_bounded_regular_file(Path(path), _MAXIMUM_CASE_BYTES)
        raw = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonstandard_json_constant,
            parse_float=_parse_finite_float,
        )
        _reject_deep_json(raw)
        return TimeoutAfterCommitCase.model_validate(raw)
    except OSError:
        raise RuntimeError("timeout-after-commit case could not be read") from None
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValidationError, ValueError):
        raise ValueError("timeout-after-commit case is invalid") from None


def _evaluation_case(
    case: TimeoutAfterCommitCase,
    max_environment_api_calls: int,
    environment: EnvironmentExecutor,
) -> EvaluationCase:
    return EvaluationCase(
        id=f"ul-timeout-after-commit-{secrets.token_hex(16)}",
        turns=(case.turn,),
        max_environment_api_calls=max_environment_api_calls,
        timeout_seconds=30,
        required_state_observation_authority=environment.capabilities.state_observation_authority,
        required_state_observer_id=environment.capabilities.state_observer_id,
        timeout_after_commit_event=TimeoutAfterCommitEventRequest(
            event_id=case.event_id,
            turn_id=case.turn.id,
            action_id=case.action_id,
        ),
    )


def _validate_run_inputs(repetitions: int, max_environment_api_calls: int) -> None:
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if type(max_environment_api_calls) is not int or max_environment_api_calls < 1:
        raise ValueError("max_environment_api_calls must be a positive integer")


def _timeout_after_commit_status(
    invariant_rules: tuple[DatasetInvariantRuleResult, ...],
    trials: tuple[TimeoutAfterCommitStressTrial, ...],
) -> Literal["passed", "failed", "inconclusive"]:
    if any(trial.inconclusive_reason is not None for trial in trials) or any(
        rule.status == "not_evaluable" for rule in invariant_rules
    ):
        return "inconclusive"
    if any(all(trial.status == "violated" for trial in rule.trials) for rule in invariant_rules):
        return "failed"
    if any(rule.status == "violated" for rule in invariant_rules):
        return "inconclusive"
    return "passed"


def _lifecycle_failure(evidence: ExecutionEvidence) -> DatasetTargetLifecycleFailure:
    lifecycle = evidence.lifecycle
    return DatasetTargetLifecycleFailure(
        failed_phase=lifecycle.failed_phase or "unknown",
        completed_phases=lifecycle.completed_phases,
        cleanup_reset_failed=(
            lifecycle.cleanup == "failed" and "cleanup_reset" not in lifecycle.completed_phases
        ),
        environment_state_may_remain=lifecycle.environment_state_uncertain,
    )


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
            raise ValueError("timeout-after-commit case exceeds the size limit")
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
