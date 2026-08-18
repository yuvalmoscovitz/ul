from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import ConfigDict, Field, JsonValue, ValidationError, field_validator, model_validator
from ul_core.contracts import DatasetTargetLifecycleError, MultiTurnDatasetTargetExecutor
from ul_core.dataset import ObservedAgentOutput
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
from ul.http_target import JsonHttpDatasetTargetConfig, json_http_target_calls_per_conversation

_MAXIMUM_CASE_BYTES = 1_000_000
_MAXIMUM_JSON_DEPTH = 100
_CASE_ID_PATTERN = r"^ulmc_v1_[0-9a-f]{64}$"


class _StrictModel(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CorrectionAfterFirstResponseCase(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    id: str = Field(min_length=1, max_length=200)
    operator_id: Literal["event.correction_after_first_response"] = (
        "event.correction_after_first_response"
    )
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
    operator_id: Literal["event.correction_after_first_response"]
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
    target: MultiTurnDatasetTargetExecutor,
    *,
    invariant_rules: tuple[DatasetInvariantRule, ...],
    observation_authority: ObservationAuthority = "committed_state_snapshot",
    repetitions: int = 3,
    max_target_calls: int = 100,
    allow_network_egress: bool = False,
) -> CorrectionStressResult:
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if not invariant_rules:
        raise ValueError("correction stress testing requires at least one invariant")
    if type(max_target_calls) is not int or max_target_calls < 1:
        raise ValueError("max_target_calls must be a positive integer")
    if not target.safety_envelope.isolated:
        raise ValueError("dataset target must be isolated")
    if target.safety_envelope.allows_network_egress and not allow_network_egress:
        raise ValueError("dataset target network egress requires explicit opt-in")
    if target.safety_envelope.allows_business_side_effects:
        raise ValueError("dataset targets must not allow business side effects")
    if not target.fresh_state_per_execution:
        raise ValueError("dataset target must start from fresh state for every conversation")

    baseline_calls = target.target_calls_for_conversation(1)
    variation_calls = target.target_calls_for_conversation(2)
    required_target_calls = repetitions * (baseline_calls + variation_calls)
    if required_target_calls > max_target_calls:
        raise ValueError("correction stress test exceeds the authorized target call budget")

    initial_turn, correction_turn = case.conversation
    trials: list[CorrectionStressTrial] = []
    baseline_final_outputs: list[ObservedAgentOutput | None] = []
    final_outputs: list[ObservedAgentOutput | None] = []
    for repetition in range(1, repetitions + 1):
        baseline: tuple[CorrectionTurnObservation, ...] = ()
        try:
            baseline_outputs = await target.execute_conversation((initial_turn.content,))
            baseline = _observations((initial_turn,), baseline_outputs)
            variation_outputs = await target.execute_conversation(
                (initial_turn.content, correction_turn.content)
            )
            variation = _observations(case.conversation, variation_outputs)
        except DatasetTargetLifecycleError as error:
            baseline_final_outputs.append(baseline[-1].target_output if baseline else None)
            final_outputs.append(None)
            trials.append(
                CorrectionStressTrial(
                    repetition=repetition,
                    baseline=baseline,
                    inconclusive_reason="target lifecycle failed",
                    lifecycle_failure=_lifecycle_failure(error),
                )
            )
            continue
        except RuntimeError:
            baseline_final_outputs.append(baseline[-1].target_output if baseline else None)
            final_outputs.append(None)
            trials.append(
                CorrectionStressTrial(
                    repetition=repetition,
                    baseline=baseline,
                    inconclusive_reason="target execution failed",
                )
            )
            continue
        baseline_final_outputs.append(baseline_outputs[-1])
        final_outputs.append(variation_outputs[-1])
        trials.append(
            CorrectionStressTrial(
                repetition=repetition,
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
    conclusive_divergences = tuple(
        divergence
        for trial in trials
        if trial.inconclusive_reason is None
        for divergence in trial.divergences
    )
    return CorrectionStressResult(
        case=case,
        requested_repetitions=repetitions,
        required_target_calls=required_target_calls,
        status=status,
        first_response_divergence_turn_id=_first_consistent_divergence(
            conclusive_divergences, "response_diverged"
        ),
        first_committed_state_divergence_turn_id=_first_consistent_divergence(
            conclusive_divergences, "committed_state_diverged"
        ),
        trials=tuple(trials),
        baseline_invariant_rules=baseline_invariant_results,
        corrected_invariant_rules=corrected_invariant_results,
    )


def plan_correction_stress_test(
    case: CorrectionAfterFirstResponseCase,
    target_config: JsonHttpDatasetTargetConfig,
    *,
    repetitions: int = 3,
    max_target_calls: int = 100,
) -> CorrectionStressPlan:
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if type(max_target_calls) is not int or max_target_calls < 1:
        raise ValueError("max_target_calls must be a positive integer")
    target_calls_per_pair = json_http_target_calls_per_conversation(
        target_config, 1
    ) + json_http_target_calls_per_conversation(target_config, 2)
    required_target_calls = repetitions * target_calls_per_pair
    if required_target_calls > max_target_calls:
        raise ValueError("correction stress test exceeds the authorized target call budget")
    return CorrectionStressPlan(
        operator_id=case.operator_id,
        repetitions=repetitions,
        target_calls_per_pair=target_calls_per_pair,
        required_target_calls=required_target_calls,
    )


def create_multi_turn_regression_case(
    *,
    stress_case: CorrectionAfterFirstResponseCase,
    target_config: JsonHttpDatasetTargetConfig,
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
    target: MultiTurnDatasetTargetExecutor,
    *,
    max_target_calls: int = 100,
    allow_network_egress: bool = False,
) -> CorrectionStressResult:
    return await run_correction_stress_test(
        case.stress_case,
        target,
        invariant_rules=case.invariant_suite.rules,
        observation_authority=case.invariant_suite.observation_authority,
        repetitions=case.repetitions,
        max_target_calls=max_target_calls,
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


def _observations(
    turns: tuple[ConversationTurn, ...],
    outputs: tuple[ObservedAgentOutput, ...],
) -> tuple[CorrectionTurnObservation, ...]:
    if len(turns) != len(outputs):
        raise RuntimeError("target returned an invalid number of turn observations")
    observations: list[CorrectionTurnObservation] = []
    for turn, output in zip(turns, outputs, strict=True):
        if "committed_state_snapshot" not in output.metadata:
            raise RuntimeError("target omitted a committed state snapshot")
        observations.append(
            CorrectionTurnObservation(
                turn=turn,
                target_output=output,
                committed_state_snapshot=output.metadata["committed_state_snapshot"],
            )
        )
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


def _first_consistent_divergence(
    divergences: tuple[CorrectionDivergence, ...],
    field: Literal["response_diverged", "committed_state_diverged"],
) -> str | None:
    turn_ids = tuple(
        dict.fromkeys(
            divergence.variation_turn_id for divergence in divergences if getattr(divergence, field)
        )
    )
    return turn_ids[0] if turn_ids else None


def _lifecycle_failure(error: DatasetTargetLifecycleError) -> DatasetTargetLifecycleFailure:
    return DatasetTargetLifecycleFailure(
        failed_phase=error.failed_phase,
        completed_phases=error.completed_phases,
        cleanup_reset_failed=error.cleanup_reset_failed,
        sandbox_state_may_remain=error.target_state_uncertain,
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
