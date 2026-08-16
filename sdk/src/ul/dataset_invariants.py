from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import ConfigDict, Field, JsonValue, ValidationError, field_validator, model_validator
from ul_core.dataset import ObservedAgentOutput
from ul_core.models import ULModel

from ul.dataset_evaluation import DatasetEvaluationResult, DatasetEvaluationTrial

_MAXIMUM_SUITE_BYTES = 1_000_000
_MAXIMUM_JSON_DEPTH = 100
_MAXIMUM_RESOLVED_VALUE_BYTES = 4_096
_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$"
_VERSION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,49}$"
_JSON_POINTER_PATTERN = r"^(?:/(?:[^~/]|~[01])*)*$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"

InvariantStatus = Literal["satisfied", "violated", "not_evaluable"]
InvariantSeverity = Literal["low", "medium", "high", "critical"]
ObservationAuthority = Literal[
    "agent_response",
    "tool_result",
    "committed_state_snapshot",
]
TrialReasonCode = Literal[
    "values_equal",
    "values_differ",
    "target_output_missing",
    "left_pointer_missing",
    "right_pointer_missing",
    "left_value_not_scalar",
    "right_value_not_scalar",
    "left_non_integer_number_not_supported",
    "right_non_integer_number_not_supported",
    "left_value_exceeds_limit",
    "right_value_exceeds_limit",
    "operand_types_differ",
]
AggregateReasonCode = Literal[
    "all_trials_satisfied",
    "one_or_more_trials_violated",
    "one_or_more_trials_not_evaluable",
]


class _StrictModel(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class JsonValuesEqualInvariant(_StrictModel):
    type: Literal["json_values_equal"]
    id: str = Field(pattern=_IDENTIFIER_PATTERN)
    version: str = Field(pattern=_VERSION_PATTERN)
    description: str = Field(min_length=1, max_length=500)
    severity: InvariantSeverity
    left_pointer: str = Field(max_length=1_000, pattern=_JSON_POINTER_PATTERN)
    right_pointer: str = Field(max_length=1_000, pattern=_JSON_POINTER_PATTERN)

    @field_validator("description")
    @classmethod
    def validate_description(cls, description: str) -> str:
        if not description.strip():
            raise ValueError("description must contain non-whitespace text")
        return description

    @model_validator(mode="after")
    def validate_distinct_pointers(self) -> Self:
        if self.left_pointer == self.right_pointer:
            raise ValueError("left and right pointers must be different")
        return self


class DatasetInvariantSuite(_StrictModel):
    schema_version: Literal["1.0.0"]
    observation_source: Literal["target_output"]
    observation_authority: ObservationAuthority
    rules: tuple[JsonValuesEqualInvariant, ...] = Field(min_length=1, max_length=100)

    @field_validator("rules", mode="before")
    @classmethod
    def accept_json_rule_array(cls, rules: object) -> object:
        return tuple(cast(list[object], rules)) if isinstance(rules, list) else rules

    @model_validator(mode="after")
    def validate_rule_ids(self) -> Self:
        rule_ids = tuple(rule.id for rule in self.rules)
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("invariant rule identifiers must be unique")
        return self

    @property
    def sha256(self) -> str:
        canonical_suite = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return hashlib.sha256(canonical_suite).hexdigest()


def _empty_resolved_values() -> dict[Literal["left", "right"], JsonValue]:
    return {}


class DatasetInvariantTrialEvaluation(_StrictModel):
    repetition: int = Field(ge=1)
    status: InvariantStatus
    reason_code: TrialReasonCode
    left_pointer: str = Field(max_length=1_000, pattern=_JSON_POINTER_PATTERN)
    right_pointer: str = Field(max_length=1_000, pattern=_JSON_POINTER_PATTERN)
    resolved_values: dict[Literal["left", "right"], JsonValue] = Field(
        default_factory=_empty_resolved_values
    )

    @model_validator(mode="after")
    def validate_evaluation(self) -> Self:
        if self.left_pointer == self.right_pointer:
            raise ValueError("invariant evidence pointers must be different")
        if any(isinstance(value, (dict, list)) for value in self.resolved_values.values()):
            raise ValueError("resolved invariant values must be JSON scalars")
        if any(isinstance(value, float) for value in self.resolved_values.values()):
            raise ValueError("non-integer JSON numbers are not supported invariant evidence")
        if any(not _resolved_value_fits(value) for value in self.resolved_values.values()):
            raise ValueError("resolved invariant values exceed the evidence limit")
        expected_status_by_reason: dict[TrialReasonCode, InvariantStatus] = {
            "values_equal": "satisfied",
            "values_differ": "violated",
            "target_output_missing": "not_evaluable",
            "left_pointer_missing": "not_evaluable",
            "right_pointer_missing": "not_evaluable",
            "left_value_not_scalar": "not_evaluable",
            "right_value_not_scalar": "not_evaluable",
            "left_non_integer_number_not_supported": "not_evaluable",
            "right_non_integer_number_not_supported": "not_evaluable",
            "left_value_exceeds_limit": "not_evaluable",
            "right_value_exceeds_limit": "not_evaluable",
            "operand_types_differ": "not_evaluable",
        }
        if self.status != expected_status_by_reason[self.reason_code]:
            raise ValueError("trial invariant status must match its reason")
        expected_resolved_keys: set[str]
        if self.reason_code in {
            "target_output_missing",
            "left_pointer_missing",
            "left_value_not_scalar",
            "left_non_integer_number_not_supported",
            "left_value_exceeds_limit",
        }:
            expected_resolved_keys = set()
        elif self.reason_code in {
            "right_pointer_missing",
            "right_value_not_scalar",
            "right_non_integer_number_not_supported",
            "right_value_exceeds_limit",
        }:
            expected_resolved_keys = {"left"}
        else:
            expected_resolved_keys = {"left", "right"}
        if set(self.resolved_values) != expected_resolved_keys:
            raise ValueError("resolved invariant values must match the trial reason")
        if self.reason_code in {"values_equal", "values_differ", "operand_types_differ"}:
            comparable, equal = _json_scalars_equal(
                self.resolved_values["left"], self.resolved_values["right"]
            )
            if self.reason_code == "values_equal" and (not comparable or not equal):
                raise ValueError("equal-value evidence must contain equal comparable values")
            if self.reason_code == "values_differ" and (not comparable or equal):
                raise ValueError(
                    "different-value evidence must contain different comparable values"
                )
            if self.reason_code == "operand_types_differ" and comparable:
                raise ValueError("type-mismatch evidence must contain incomparable value types")
        return self


class DatasetInvariantRuleEvaluation(_StrictModel):
    rule_type: Literal["json_values_equal"]
    rule_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    rule_version: str = Field(pattern=_VERSION_PATTERN)
    description: str = Field(min_length=1, max_length=500)
    severity: InvariantSeverity
    status: InvariantStatus
    reason_code: AggregateReasonCode
    trials: tuple[DatasetInvariantTrialEvaluation, ...] = Field(min_length=1)

    @field_validator("description")
    @classmethod
    def validate_description(cls, description: str) -> str:
        if not description.strip():
            raise ValueError("description must contain non-whitespace text")
        return description

    @model_validator(mode="after")
    def validate_aggregate(self) -> Self:
        if tuple(trial.repetition for trial in self.trials) != tuple(
            range(1, len(self.trials) + 1)
        ):
            raise ValueError("invariant trials must preserve repetition order")
        statuses = {trial.status for trial in self.trials}
        if "violated" in statuses:
            expected_status: InvariantStatus = "violated"
            expected_reason: AggregateReasonCode = "one_or_more_trials_violated"
        elif "not_evaluable" in statuses:
            expected_status = "not_evaluable"
            expected_reason = "one_or_more_trials_not_evaluable"
        else:
            expected_status = "satisfied"
            expected_reason = "all_trials_satisfied"
        if self.status != expected_status or self.reason_code != expected_reason:
            raise ValueError("aggregate invariant result must match its trials")
        pointer_pairs = {(trial.left_pointer, trial.right_pointer) for trial in self.trials}
        if len(pointer_pairs) != 1:
            raise ValueError("all invariant trials must use the same pointers")
        return self


class DatasetInvariantArmEvaluation(_StrictModel):
    arm: Literal["baseline", "variation"]
    operator_id: str | None = Field(default=None, min_length=1)
    rules: tuple[DatasetInvariantRuleEvaluation, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_arm(self) -> Self:
        if (self.arm == "baseline") != (self.operator_id is None):
            raise ValueError("only variation invariant results have an operator ID")
        rule_ids = tuple(rule.rule_id for rule in self.rules)
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("arm invariant results must have unique rule IDs")
        return self


class DatasetInvariantEvaluation(_StrictModel):
    interaction_id: str = Field(min_length=1)
    suite_sha256: str = Field(pattern=_SHA256_PATTERN)
    observation_source: Literal["target_output"] = "target_output"
    observation_authority: ObservationAuthority
    baseline: DatasetInvariantArmEvaluation
    variations: tuple[DatasetInvariantArmEvaluation, ...] = ()

    @model_validator(mode="after")
    def validate_evaluation(self) -> Self:
        if self.baseline.arm != "baseline":
            raise ValueError("dataset invariant evaluation requires a baseline arm")
        if any(variation.arm != "variation" for variation in self.variations):
            raise ValueError("dataset invariant variations require variation arms")
        operator_ids = tuple(variation.operator_id for variation in self.variations)
        if len(operator_ids) != len(set(operator_ids)):
            raise ValueError("dataset invariant variation operator IDs must be unique")
        expected_rules = tuple(_rule_identity(rule) for rule in self.baseline.rules)
        if any(
            tuple(_rule_identity(rule) for rule in variation.rules) != expected_rules
            for variation in self.variations
        ):
            raise ValueError("all invariant arms must preserve suite rules and order")
        return self


def load_dataset_invariant_suite(path: str | Path) -> DatasetInvariantSuite:
    try:
        with Path(path).open("rb") as suite_file:
            encoded_suite = suite_file.read(_MAXIMUM_SUITE_BYTES + 1)
    except OSError:
        raise RuntimeError("dataset invariant suite could not be read") from None
    if len(encoded_suite) > _MAXIMUM_SUITE_BYTES:
        raise ValueError("dataset invariant suite exceeds the size limit")
    try:
        raw_suite = json.loads(
            encoded_suite.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonstandard_json_constant,
            parse_float=_parse_finite_float,
        )
        _reject_deep_json(raw_suite)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise ValueError("dataset invariant suite contains invalid JSON") from None
    try:
        return DatasetInvariantSuite.model_validate(raw_suite)
    except RecursionError:
        raise ValueError("dataset invariant suite is invalid") from None
    except ValidationError as error:
        reasons = [
            f"{'.'.join(str(part) for part in issue['loc'])}: "
            f"{str(issue['msg']).removeprefix('Value error, ')}"
            for issue in error.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            )
        ]
        raise ValueError(f"dataset invariant suite is invalid: {'; '.join(reasons)}") from None


def evaluate_dataset_invariants(
    result: DatasetEvaluationResult,
    suite: DatasetInvariantSuite,
) -> DatasetInvariantEvaluation:
    baseline = _evaluate_arm(suite, result.baseline.trial_set.trials, arm="baseline")
    variations = tuple(
        _evaluate_arm(
            suite,
            case.trial_set.trials,
            arm="variation",
            operator_id=case.candidate.operator_id,
        )
        for case in result.cases
        if case.candidate.passed and case.trial_set is not None
    )
    return DatasetInvariantEvaluation(
        interaction_id=result.source.id,
        suite_sha256=suite.sha256,
        observation_authority=suite.observation_authority,
        baseline=baseline,
        variations=variations,
    )


def evaluate_dataset_invariant_rules(
    rules: tuple[JsonValuesEqualInvariant, ...],
    outputs: tuple[ObservedAgentOutput | None, ...],
) -> tuple[DatasetInvariantRuleEvaluation, ...]:
    if not outputs:
        raise ValueError("invariant evaluation requires at least one target output")
    return tuple(_evaluate_rule_from_outputs(rule, outputs) for rule in rules)


def _evaluate_arm(
    suite: DatasetInvariantSuite,
    trials: tuple[DatasetEvaluationTrial, ...],
    *,
    arm: Literal["baseline", "variation"],
    operator_id: str | None = None,
) -> DatasetInvariantArmEvaluation:
    return DatasetInvariantArmEvaluation(
        arm=arm,
        operator_id=operator_id,
        rules=tuple(_evaluate_rule(rule, trials) for rule in suite.rules),
    )


def _evaluate_rule(
    rule: JsonValuesEqualInvariant,
    trials: tuple[DatasetEvaluationTrial, ...],
) -> DatasetInvariantRuleEvaluation:
    return _evaluate_rule_from_outputs(rule, tuple(trial.target_output for trial in trials))


def _evaluate_rule_from_outputs(
    rule: JsonValuesEqualInvariant,
    outputs: tuple[ObservedAgentOutput | None, ...],
) -> DatasetInvariantRuleEvaluation:
    trial_results = tuple(
        _evaluate_output(rule, output, repetition)
        for repetition, output in enumerate(outputs, start=1)
    )
    statuses = {trial.status for trial in trial_results}
    if "violated" in statuses:
        status: InvariantStatus = "violated"
        reason_code: AggregateReasonCode = "one_or_more_trials_violated"
    elif "not_evaluable" in statuses:
        status = "not_evaluable"
        reason_code = "one_or_more_trials_not_evaluable"
    else:
        status = "satisfied"
        reason_code = "all_trials_satisfied"
    return DatasetInvariantRuleEvaluation(
        rule_type=rule.type,
        rule_id=rule.id,
        rule_version=rule.version,
        description=rule.description,
        severity=rule.severity,
        status=status,
        reason_code=reason_code,
        trials=trial_results,
    )


def _evaluate_output(
    rule: JsonValuesEqualInvariant,
    target_output: ObservedAgentOutput | None,
    repetition: int,
) -> DatasetInvariantTrialEvaluation:
    if target_output is None:
        return _trial_result(rule, repetition, "not_evaluable", "target_output_missing")
    left_found, left_value = _resolve_json_pointer(
        target_output.raw_output,
        rule.left_pointer,
    )
    if not left_found:
        return _trial_result(rule, repetition, "not_evaluable", "left_pointer_missing")
    if not _is_json_scalar(left_value):
        return _trial_result(rule, repetition, "not_evaluable", "left_value_not_scalar")
    if isinstance(left_value, float):
        return _trial_result(
            rule,
            repetition,
            "not_evaluable",
            "left_non_integer_number_not_supported",
        )
    if not _resolved_value_fits(left_value):
        return _trial_result(
            rule,
            repetition,
            "not_evaluable",
            "left_value_exceeds_limit",
        )
    resolved_values: dict[Literal["left", "right"], JsonValue] = {
        "left": cast(JsonValue, left_value)
    }
    right_found, right_value = _resolve_json_pointer(
        target_output.raw_output,
        rule.right_pointer,
    )
    if not right_found:
        return _trial_result(
            rule,
            repetition,
            "not_evaluable",
            "right_pointer_missing",
            resolved_values,
        )
    if not _is_json_scalar(right_value):
        return _trial_result(
            rule,
            repetition,
            "not_evaluable",
            "right_value_not_scalar",
            resolved_values,
        )
    if isinstance(right_value, float):
        return _trial_result(
            rule,
            repetition,
            "not_evaluable",
            "right_non_integer_number_not_supported",
            resolved_values,
        )
    if not _resolved_value_fits(right_value):
        return _trial_result(
            rule,
            repetition,
            "not_evaluable",
            "right_value_exceeds_limit",
            resolved_values,
        )
    resolved_values["right"] = cast(JsonValue, right_value)
    comparable, equal = _json_scalars_equal(left_value, right_value)
    if not comparable:
        return _trial_result(
            rule,
            repetition,
            "not_evaluable",
            "operand_types_differ",
            resolved_values,
        )
    return _trial_result(
        rule,
        repetition,
        "satisfied" if equal else "violated",
        "values_equal" if equal else "values_differ",
        resolved_values,
    )


def _trial_result(
    rule: JsonValuesEqualInvariant,
    repetition: int,
    status: InvariantStatus,
    reason_code: TrialReasonCode,
    resolved_values: dict[Literal["left", "right"], JsonValue] | None = None,
) -> DatasetInvariantTrialEvaluation:
    return DatasetInvariantTrialEvaluation(
        repetition=repetition,
        status=status,
        reason_code=reason_code,
        left_pointer=rule.left_pointer,
        right_pointer=rule.right_pointer,
        resolved_values=resolved_values or {},
    )


def _resolve_json_pointer(document: JsonValue, pointer: str) -> tuple[bool, object]:
    current: object = document
    if pointer == "":
        return True, current
    for encoded_token in pointer[1:].split("/"):
        token = encoded_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
            continue
        valid_array_index = token == "0" or (
            token.isascii() and token.isdecimal() and not token.startswith("0")
        )
        if isinstance(current, list) and valid_array_index and int(token) < len(current):
            current = current[int(token)]
            continue
        return False, None
    return True, current


def _is_json_scalar(value: object) -> bool:
    return (
        value is None
        or isinstance(value, (str, bool, int))
        or (isinstance(value, float) and math.isfinite(value))
    )


def _json_scalars_equal(left: object, right: object) -> tuple[bool, bool]:
    if _is_json_number(left) and _is_json_number(right):
        return True, Decimal(str(left)) == Decimal(str(right))
    if type(left) is not type(right):
        return False, False
    return True, left == right


def _resolved_value_fits(value: object) -> bool:
    try:
        encoded_value = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError):
        return False
    return len(encoded_value) <= _MAXIMUM_RESOLVED_VALUE_BYTES


def _rule_identity(rule: DatasetInvariantRuleEvaluation) -> tuple[object, ...]:
    first_trial = rule.trials[0]
    return (
        rule.rule_type,
        rule.rule_id,
        rule.rule_version,
        rule.description,
        rule.severity,
        first_trial.left_pointer,
        first_trial.right_pointer,
    )


def _is_json_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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


def _reject_deep_json(value: object, *, depth: int = 0) -> None:
    if depth > _MAXIMUM_JSON_DEPTH:
        raise ValueError("JSON exceeds the nesting limit")
    if isinstance(value, dict):
        for item in cast(dict[str, object], value).values():
            _reject_deep_json(item, depth=depth + 1)
    elif isinstance(value, list):
        for item in cast(list[object], value):
            _reject_deep_json(item, depth=depth + 1)
