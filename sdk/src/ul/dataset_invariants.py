from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, Self, TypeGuard, cast

from pydantic import ConfigDict, Field, JsonValue, ValidationError, field_validator, model_validator
from ul_core.dataset import ObservedAgentOutput
from ul_core.models import ULModel

from ul.dataset_evaluation import DatasetEvaluationResult, DatasetEvaluationTrial

_MAXIMUM_SUITE_BYTES = 1_000_000
_MAXIMUM_JSON_DEPTH = 100
_MAXIMUM_RESOLVED_VALUE_BYTES = 4_096
_MAXIMUM_ALLOWED_VALUES = 100
_MAXIMUM_ARRAY_ITEMS = 10_000
_MAXIMUM_KEY_POINTERS = 10
_MAXIMUM_ARRAY_INVARIANT_WORK_UNITS = 250_000
_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$"
_VERSION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,49}$"
_JSON_POINTER_PATTERN = r"^(?:/(?:[^~/]|~[01])*)*$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"

InvariantStatus = Literal["satisfied", "violated", "not_evaluable"]
InvariantSeverity = Literal["low", "medium", "high", "critical"]
ObservationAuthority = Literal[
    "agent_response",
    "committed_state_snapshot",
]
EqualityTrialReasonCode = Literal[
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
TrialReasonCode = EqualityTrialReasonCode
ValueEqualsTrialReasonCode = Literal[
    "value_equals_literal",
    "value_differs_from_literal",
    "target_output_missing",
    "value_pointer_missing",
    "value_not_scalar",
    "value_non_integer_number_not_supported",
    "value_exceeds_limit",
]
ValueInSetTrialReasonCode = Literal[
    "value_in_allowed_set",
    "value_not_in_allowed_set",
    "target_output_missing",
    "value_pointer_missing",
    "value_not_scalar",
    "value_non_integer_number_not_supported",
    "value_exceeds_limit",
]
ArrayUniqueTrialReasonCode = Literal[
    "array_items_unique",
    "duplicate_array_items",
    "target_output_missing",
    "array_pointer_missing",
    "array_value_not_array",
    "array_exceeds_limit",
    "evaluation_work_limit_exceeded",
    "key_pointer_missing",
    "key_value_not_scalar",
    "key_non_integer_number_not_supported",
    "key_value_exceeds_limit",
]
TransitionTrialReasonCode = Literal[
    "no_new_effect",
    "new_effect_observed",
    "exactly_one_new_effect",
    "unexpected_new_effect_count",
    "value_unchanged",
    "value_changed",
    "before_checkpoint_missing",
    "after_checkpoint_missing",
    "before_pointer_missing",
    "after_pointer_missing",
    "before_value_not_array",
    "after_value_not_array",
    "effect_array_exceeds_limit",
    "effect_history_rewritten",
    "evaluation_work_limit_exceeded",
]
AggregateReasonCode = Literal[
    "all_trials_satisfied",
    "one_or_more_trials_violated",
    "one_or_more_trials_not_evaluable",
]
ScalarResolution = Literal[
    "resolved",
    "missing",
    "not_scalar",
    "non_integer_number",
    "exceeds_limit",
]


class _StrictModel(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _ArrayInvariantWorkBudget:
    def __init__(self, maximum_units: int) -> None:
        self.remaining_units = maximum_units

    def consume(self, units: int) -> bool:
        if units > self.remaining_units:
            return False
        self.remaining_units -= units
        return True


class _InvariantRule(_StrictModel):
    id: str = Field(pattern=_IDENTIFIER_PATTERN)
    version: str = Field(pattern=_VERSION_PATTERN)
    description: str = Field(min_length=1, max_length=500)
    severity: InvariantSeverity

    @field_validator("description")
    @classmethod
    def validate_description(cls, description: str) -> str:
        if not description.strip():
            raise ValueError("description must contain non-whitespace text")
        return description


class JsonValuesEqualInvariant(_InvariantRule):
    type: Literal["json_values_equal"]
    left_pointer: str = Field(max_length=1_000, pattern=_JSON_POINTER_PATTERN)
    right_pointer: str = Field(max_length=1_000, pattern=_JSON_POINTER_PATTERN)

    @model_validator(mode="after")
    def validate_distinct_pointers(self) -> Self:
        if self.left_pointer == self.right_pointer:
            raise ValueError("left and right pointers must be different")
        return self


class JsonValueEqualsLiteralInvariant(_InvariantRule):
    type: Literal["json_value_equals_literal"]
    value_pointer: str = Field(max_length=1_000, pattern=_JSON_POINTER_PATTERN)
    literal: JsonValue

    @field_validator("literal")
    @classmethod
    def validate_literal(cls, value: JsonValue) -> JsonValue:
        return _validate_configured_scalar(value, "literal")


class JsonValueInAllowedSetInvariant(_InvariantRule):
    type: Literal["json_value_in_allowed_set"]
    value_pointer: str = Field(max_length=1_000, pattern=_JSON_POINTER_PATTERN)
    allowed_values: tuple[JsonValue, ...] = Field(
        min_length=1,
        max_length=_MAXIMUM_ALLOWED_VALUES,
    )

    @field_validator("allowed_values", mode="before")
    @classmethod
    def accept_json_allowed_value_array(cls, values: object) -> object:
        return tuple(cast(list[object], values)) if isinstance(values, list) else values

    @field_validator("allowed_values")
    @classmethod
    def validate_allowed_values(cls, values: tuple[JsonValue, ...]) -> tuple[JsonValue, ...]:
        validated = tuple(_validate_configured_scalar(value, "allowed value") for value in values)
        if any(
            _json_scalars_equal(value, previous)[1]
            for index, value in enumerate(validated)
            for previous in validated[:index]
        ):
            raise ValueError("allowed values must be unique")
        return validated


class JsonArrayItemsUniqueByInvariant(_InvariantRule):
    type: Literal["json_array_items_unique_by"]
    array_pointer: str = Field(max_length=1_000, pattern=_JSON_POINTER_PATTERN)
    key_pointers: tuple[str, ...] = Field(
        min_length=1,
        max_length=_MAXIMUM_KEY_POINTERS,
    )

    @field_validator("key_pointers", mode="before")
    @classmethod
    def accept_json_key_pointer_array(cls, pointers: object) -> object:
        return tuple(cast(list[object], pointers)) if isinstance(pointers, list) else pointers

    @field_validator("key_pointers")
    @classmethod
    def validate_key_pointers(cls, pointers: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_key_pointers(pointers)


class _StateTransitionInvariant(_InvariantRule):
    before_checkpoint: Literal["before_turn"]
    after_checkpoint: Literal["after_turn"]
    observation_pointer: str = Field(max_length=1_000, pattern=_JSON_POINTER_PATTERN)


class NoNewEffectInvariant(_StateTransitionInvariant):
    type: Literal["no_new_effect"]


class ExactlyOneNewEffectInvariant(_StateTransitionInvariant):
    type: Literal["exactly_one_new_effect"]


class UnchangedBetweenCheckpointsInvariant(_StateTransitionInvariant):
    type: Literal["unchanged_between_checkpoints"]


DatasetInvariantRule = Annotated[
    JsonValuesEqualInvariant
    | JsonValueEqualsLiteralInvariant
    | JsonValueInAllowedSetInvariant
    | JsonArrayItemsUniqueByInvariant
    | NoNewEffectInvariant
    | ExactlyOneNewEffectInvariant
    | UnchangedBetweenCheckpointsInvariant,
    Field(discriminator="type"),
]


class DatasetInvariantSuite(_StrictModel):
    schema_version: Literal["1.0.0", "1.1.0", "1.2.0"]
    observation_source: Literal["target_output"]
    observation_authority: ObservationAuthority
    rules: tuple[DatasetInvariantRule, ...] = Field(min_length=1, max_length=100)

    @field_validator("rules", mode="before")
    @classmethod
    def accept_json_rule_array(cls, rules: object) -> object:
        return tuple(cast(list[object], rules)) if isinstance(rules, list) else rules

    @model_validator(mode="after")
    def validate_rule_ids(self) -> Self:
        rule_ids = tuple(rule.id for rule in self.rules)
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("invariant rule identifiers must be unique")
        if self.schema_version == "1.0.0" and any(
            not isinstance(rule, JsonValuesEqualInvariant) for rule in self.rules
        ):
            raise ValueError("invariant schema 1.0.0 supports only json_values_equal rules")
        transition_rules = tuple(rule for rule in self.rules if _is_transition_rule(rule))
        if transition_rules and self.schema_version != "1.2.0":
            raise ValueError("state-transition rules require invariant schema 1.2.0")
        if transition_rules and self.observation_authority != "committed_state_snapshot":
            raise ValueError("state-transition rules require committed-state observation")
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
    reason_code: EqualityTrialReasonCode
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
        expected_status_by_reason: dict[EqualityTrialReasonCode, InvariantStatus] = {
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


def _empty_actual_value() -> dict[Literal["actual"], JsonValue]:
    return {}


class DatasetInvariantValueEqualsTrialEvaluation(_StrictModel):
    repetition: int = Field(ge=1)
    status: InvariantStatus
    reason_code: ValueEqualsTrialReasonCode
    value_pointer: str = Field(max_length=1_000, pattern=_JSON_POINTER_PATTERN)
    resolved_values: dict[Literal["actual"], JsonValue] = Field(default_factory=_empty_actual_value)

    @model_validator(mode="after")
    def validate_evaluation(self) -> Self:
        expected_status_by_reason: dict[str, InvariantStatus] = {
            "value_equals_literal": "satisfied",
            "value_differs_from_literal": "violated",
            "target_output_missing": "not_evaluable",
            "value_pointer_missing": "not_evaluable",
            "value_not_scalar": "not_evaluable",
            "value_non_integer_number_not_supported": "not_evaluable",
            "value_exceeds_limit": "not_evaluable",
        }
        _validate_single_value_evidence(
            self.status,
            self.reason_code,
            self.resolved_values,
            expected_status_by_reason,
            {"value_equals_literal", "value_differs_from_literal"},
        )
        return self


class DatasetInvariantValueInSetTrialEvaluation(_StrictModel):
    repetition: int = Field(ge=1)
    status: InvariantStatus
    reason_code: ValueInSetTrialReasonCode
    value_pointer: str = Field(max_length=1_000, pattern=_JSON_POINTER_PATTERN)
    resolved_values: dict[Literal["actual"], JsonValue] = Field(default_factory=_empty_actual_value)

    @model_validator(mode="after")
    def validate_evaluation(self) -> Self:
        expected_status_by_reason: dict[str, InvariantStatus] = {
            "value_in_allowed_set": "satisfied",
            "value_not_in_allowed_set": "violated",
            "target_output_missing": "not_evaluable",
            "value_pointer_missing": "not_evaluable",
            "value_not_scalar": "not_evaluable",
            "value_non_integer_number_not_supported": "not_evaluable",
            "value_exceeds_limit": "not_evaluable",
        }
        _validate_single_value_evidence(
            self.status,
            self.reason_code,
            self.resolved_values,
            expected_status_by_reason,
            {"value_in_allowed_set", "value_not_in_allowed_set"},
        )
        return self


class DatasetInvariantArrayUniqueTrialEvaluation(_StrictModel):
    repetition: int = Field(ge=1)
    status: InvariantStatus
    reason_code: ArrayUniqueTrialReasonCode
    array_pointer: str = Field(max_length=1_000, pattern=_JSON_POINTER_PATTERN)
    key_pointers: tuple[str, ...] = Field(min_length=1, max_length=_MAXIMUM_KEY_POINTERS)
    item_count: int | None = Field(default=None, ge=0)
    failed_item_index: int | None = Field(default=None, ge=0)
    failed_key_pointer: str | None = Field(
        default=None,
        max_length=1_000,
        pattern=_JSON_POINTER_PATTERN,
    )
    duplicate_indices: tuple[int, ...] = Field(default=(), max_length=2)

    @field_validator("key_pointers")
    @classmethod
    def validate_key_pointers(cls, pointers: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_key_pointers(pointers)

    @model_validator(mode="after")
    def validate_evaluation(self) -> Self:
        expected_status_by_reason: dict[ArrayUniqueTrialReasonCode, InvariantStatus] = {
            "array_items_unique": "satisfied",
            "duplicate_array_items": "violated",
            "target_output_missing": "not_evaluable",
            "array_pointer_missing": "not_evaluable",
            "array_value_not_array": "not_evaluable",
            "array_exceeds_limit": "not_evaluable",
            "evaluation_work_limit_exceeded": "not_evaluable",
            "key_pointer_missing": "not_evaluable",
            "key_value_not_scalar": "not_evaluable",
            "key_non_integer_number_not_supported": "not_evaluable",
            "key_value_exceeds_limit": "not_evaluable",
        }
        if self.status != expected_status_by_reason[self.reason_code]:
            raise ValueError("trial invariant status must match its reason")
        if (
            self.reason_code != "array_exceeds_limit"
            and self.item_count is not None
            and self.item_count > _MAXIMUM_ARRAY_ITEMS
        ):
            raise ValueError("evaluated array invariant evidence exceeds the item limit")
        if self.reason_code in {
            "target_output_missing",
            "array_pointer_missing",
            "array_value_not_array",
        }:
            if (
                any(
                    value is not None
                    for value in (
                        self.item_count,
                        self.failed_item_index,
                        self.failed_key_pointer,
                    )
                )
                or self.duplicate_indices
            ):
                raise ValueError("array invariant evidence must match the trial reason")
        elif self.reason_code == "array_exceeds_limit":
            if (
                self.item_count is None
                or self.item_count <= _MAXIMUM_ARRAY_ITEMS
                or self.failed_item_index is not None
                or self.failed_key_pointer is not None
                or self.duplicate_indices
            ):
                raise ValueError("array invariant evidence must match the trial reason")
        elif self.reason_code == "duplicate_array_items":
            if (
                self.item_count is None
                or self.failed_item_index is not None
                or self.failed_key_pointer is not None
                or len(self.duplicate_indices) != 2
                or self.duplicate_indices[0] >= self.duplicate_indices[1]
                or self.duplicate_indices[1] >= self.item_count
            ):
                raise ValueError("duplicate item evidence must identify two array items")
        elif self.reason_code == "array_items_unique":
            if (
                self.item_count is None
                or self.failed_item_index is not None
                or self.failed_key_pointer is not None
                or self.duplicate_indices
            ):
                raise ValueError("unique item evidence must contain only the item count")
        elif (
            self.item_count is None
            or self.failed_item_index is None
            or self.failed_item_index >= self.item_count
            or self.failed_key_pointer is None
            or self.failed_key_pointer not in self.key_pointers
            or self.duplicate_indices
        ):
            raise ValueError("array key evidence must identify the unevaluable item and key")
        return self


class DatasetInvariantTransitionTrialEvaluation(_StrictModel):
    repetition: int = Field(ge=1)
    status: InvariantStatus
    reason_code: TransitionTrialReasonCode
    before_checkpoint: Literal["before_turn"]
    after_checkpoint: Literal["after_turn"]
    observation_pointer: str = Field(max_length=1_000, pattern=_JSON_POINTER_PATTERN)
    before_item_count: int | None = Field(default=None, ge=0)
    after_item_count: int | None = Field(default=None, ge=0)
    new_effect_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_evaluation(self) -> Self:
        expected_status_by_reason: dict[TransitionTrialReasonCode, InvariantStatus] = {
            "no_new_effect": "satisfied",
            "new_effect_observed": "violated",
            "exactly_one_new_effect": "satisfied",
            "unexpected_new_effect_count": "violated",
            "value_unchanged": "satisfied",
            "value_changed": "violated",
            "before_checkpoint_missing": "not_evaluable",
            "after_checkpoint_missing": "not_evaluable",
            "before_pointer_missing": "not_evaluable",
            "after_pointer_missing": "not_evaluable",
            "before_value_not_array": "not_evaluable",
            "after_value_not_array": "not_evaluable",
            "effect_array_exceeds_limit": "not_evaluable",
            "effect_history_rewritten": "not_evaluable",
            "evaluation_work_limit_exceeded": "not_evaluable",
        }
        if self.status != expected_status_by_reason[self.reason_code]:
            raise ValueError("trial invariant status must match its reason")
        count_reasons = {
            "no_new_effect",
            "new_effect_observed",
            "exactly_one_new_effect",
            "unexpected_new_effect_count",
        }
        if self.reason_code in count_reasons:
            before_item_count = self.before_item_count
            after_item_count = self.after_item_count
            new_effect_count = self.new_effect_count
            if before_item_count is None or after_item_count is None or new_effect_count is None:
                raise ValueError("evaluated effect transitions require item counts")
            if new_effect_count != after_item_count - before_item_count:
                raise ValueError("new-effect count must match the checkpoint item counts")
            expected_new_effect_counts: dict[TransitionTrialReasonCode, set[int]] = {
                "no_new_effect": {0},
                "exactly_one_new_effect": {1},
            }
            if (
                self.reason_code in expected_new_effect_counts
                and new_effect_count not in expected_new_effect_counts[self.reason_code]
            ):
                raise ValueError("new-effect count must match the transition reason")
            if self.reason_code == "new_effect_observed" and new_effect_count == 0:
                raise ValueError("new-effect count must match the transition reason")
            if self.reason_code == "unexpected_new_effect_count" and new_effect_count == 1:
                raise ValueError("new-effect count must match the transition reason")
        elif self.new_effect_count is not None:
            raise ValueError("unevaluable transitions cannot report a new-effect count")
        if self.reason_code in {
            "value_unchanged",
            "value_changed",
            "before_checkpoint_missing",
            "after_checkpoint_missing",
            "before_pointer_missing",
            "after_pointer_missing",
            "before_value_not_array",
            "after_value_not_array",
        } and any(count is not None for count in (self.before_item_count, self.after_item_count)):
            raise ValueError("transition evidence must match its reason")
        return self


class _DatasetInvariantRuleEvaluation(_StrictModel):
    rule_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    rule_version: str = Field(pattern=_VERSION_PATTERN)
    description: str = Field(min_length=1, max_length=500)
    severity: InvariantSeverity
    status: InvariantStatus
    reason_code: AggregateReasonCode

    @field_validator("description")
    @classmethod
    def validate_description(cls, description: str) -> str:
        if not description.strip():
            raise ValueError("description must contain non-whitespace text")
        return description


class DatasetInvariantRuleEvaluation(_DatasetInvariantRuleEvaluation):
    rule_type: Literal["json_values_equal"]
    trials: tuple[DatasetInvariantTrialEvaluation, ...] = Field(min_length=1)

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


class DatasetInvariantValueEqualsRuleEvaluation(_DatasetInvariantRuleEvaluation):
    rule_type: Literal["json_value_equals_literal"]
    value_pointer: str = Field(max_length=1_000, pattern=_JSON_POINTER_PATTERN)
    literal: JsonValue
    trials: tuple[DatasetInvariantValueEqualsTrialEvaluation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_aggregate(self) -> Self:
        _validate_rule_aggregate(self.status, self.reason_code, self.trials)
        if any(trial.value_pointer != self.value_pointer for trial in self.trials):
            raise ValueError("all invariant trials must use the rule value pointer")
        literal = _validate_configured_scalar(self.literal, "literal")
        for trial in self.trials:
            actual = trial.resolved_values.get("actual")
            if "actual" not in trial.resolved_values:
                continue
            comparable, equal = _json_scalars_equal(actual, literal)
            if trial.reason_code == "value_equals_literal" and (not comparable or not equal):
                raise ValueError("satisfied literal evidence must equal the literal")
            if trial.reason_code == "value_differs_from_literal" and comparable and equal:
                raise ValueError("violated literal evidence must differ from the literal")
        return self


class DatasetInvariantValueInSetRuleEvaluation(_DatasetInvariantRuleEvaluation):
    rule_type: Literal["json_value_in_allowed_set"]
    value_pointer: str = Field(max_length=1_000, pattern=_JSON_POINTER_PATTERN)
    allowed_values: tuple[JsonValue, ...] = Field(min_length=1, max_length=_MAXIMUM_ALLOWED_VALUES)
    trials: tuple[DatasetInvariantValueInSetTrialEvaluation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_aggregate(self) -> Self:
        _validate_rule_aggregate(self.status, self.reason_code, self.trials)
        if any(trial.value_pointer != self.value_pointer for trial in self.trials):
            raise ValueError("all invariant trials must use the rule value pointer")
        allowed_values = JsonValueInAllowedSetInvariant(
            type="json_value_in_allowed_set",
            id=self.rule_id,
            version=self.rule_version,
            description=self.description,
            severity=self.severity,
            value_pointer=self.value_pointer,
            allowed_values=self.allowed_values,
        ).allowed_values
        for trial in self.trials:
            if "actual" not in trial.resolved_values:
                continue
            allowed = _value_in_set(trial.resolved_values["actual"], allowed_values)
            if trial.reason_code == "value_in_allowed_set" and not allowed:
                raise ValueError("satisfied set evidence must contain an allowed value")
            if trial.reason_code == "value_not_in_allowed_set" and allowed:
                raise ValueError("violated set evidence must contain a disallowed value")
        return self


class DatasetInvariantArrayUniqueRuleEvaluation(_DatasetInvariantRuleEvaluation):
    rule_type: Literal["json_array_items_unique_by"]
    array_pointer: str = Field(max_length=1_000, pattern=_JSON_POINTER_PATTERN)
    key_pointers: tuple[str, ...] = Field(min_length=1, max_length=_MAXIMUM_KEY_POINTERS)
    trials: tuple[DatasetInvariantArrayUniqueTrialEvaluation, ...] = Field(min_length=1)

    @field_validator("key_pointers")
    @classmethod
    def validate_key_pointers(cls, pointers: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_key_pointers(pointers)

    @model_validator(mode="after")
    def validate_aggregate(self) -> Self:
        _validate_rule_aggregate(self.status, self.reason_code, self.trials)
        if any(
            (trial.array_pointer, trial.key_pointers) != (self.array_pointer, self.key_pointers)
            for trial in self.trials
        ):
            raise ValueError("all invariant trials must use the rule array and key pointers")
        return self


class DatasetInvariantTransitionRuleEvaluation(_DatasetInvariantRuleEvaluation):
    rule_type: Literal[
        "no_new_effect",
        "exactly_one_new_effect",
        "unchanged_between_checkpoints",
    ]
    before_checkpoint: Literal["before_turn"]
    after_checkpoint: Literal["after_turn"]
    observation_pointer: str = Field(max_length=1_000, pattern=_JSON_POINTER_PATTERN)
    trials: tuple[DatasetInvariantTransitionTrialEvaluation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_aggregate(self) -> Self:
        _validate_rule_aggregate(self.status, self.reason_code, self.trials)
        expected_location = (
            self.before_checkpoint,
            self.after_checkpoint,
            self.observation_pointer,
        )
        if any(
            (trial.before_checkpoint, trial.after_checkpoint, trial.observation_pointer)
            != expected_location
            for trial in self.trials
        ):
            raise ValueError("all invariant trials must use the rule transition location")
        rule_reasons = {
            "no_new_effect": {"no_new_effect", "new_effect_observed"},
            "exactly_one_new_effect": {
                "exactly_one_new_effect",
                "unexpected_new_effect_count",
            },
            "unchanged_between_checkpoints": {"value_unchanged", "value_changed"},
        }
        shared_reasons = {
            "before_checkpoint_missing",
            "after_checkpoint_missing",
            "before_pointer_missing",
            "after_pointer_missing",
            "evaluation_work_limit_exceeded",
        }
        effect_only_reasons = {
            "before_value_not_array",
            "after_value_not_array",
            "effect_array_exceeds_limit",
            "effect_history_rewritten",
        }
        allowed_reasons = rule_reasons[self.rule_type] | shared_reasons
        if self.rule_type != "unchanged_between_checkpoints":
            allowed_reasons |= effect_only_reasons
        if any(trial.reason_code not in allowed_reasons for trial in self.trials):
            raise ValueError("transition trial reason must match its rule type")
        return self


DatasetInvariantRuleResult = Annotated[
    DatasetInvariantRuleEvaluation
    | DatasetInvariantValueEqualsRuleEvaluation
    | DatasetInvariantValueInSetRuleEvaluation
    | DatasetInvariantArrayUniqueRuleEvaluation
    | DatasetInvariantTransitionRuleEvaluation,
    Field(discriminator="rule_type"),
]


class DatasetInvariantArmEvaluation(_StrictModel):
    arm: Literal["baseline", "variation"]
    operator_id: str | None = Field(default=None, min_length=1)
    rules: tuple[DatasetInvariantRuleResult, ...] = Field(min_length=1, max_length=100)

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
        rule_discriminators = {
            "json_values_equal",
            "json_value_equals_literal",
            "json_value_in_allowed_set",
            "json_array_items_unique_by",
            "no_new_effect",
            "exactly_one_new_effect",
            "unchanged_between_checkpoints",
        }
        reasons = [
            f"{'.'.join(str(part) for part in issue['loc'] if part not in rule_discriminators)}: "
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
    array_work_budget = _ArrayInvariantWorkBudget(_MAXIMUM_ARRAY_INVARIANT_WORK_UNITS)
    baseline = _evaluate_arm(
        suite,
        result.baseline.trial_set.trials,
        arm="baseline",
        array_work_budget=array_work_budget,
    )
    variations = tuple(
        _evaluate_arm(
            suite,
            case.trial_set.trials,
            arm="variation",
            operator_id=case.candidate.operator_id,
            array_work_budget=array_work_budget,
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
    rules: tuple[DatasetInvariantRule, ...],
    outputs: tuple[ObservedAgentOutput | None, ...],
    *,
    observation_authority: ObservationAuthority = "agent_response",
) -> tuple[DatasetInvariantRuleResult, ...]:
    if not outputs:
        raise ValueError("invariant evaluation requires at least one target output")
    if observation_authority != "committed_state_snapshot" and any(
        _is_transition_rule(rule) for rule in rules
    ):
        raise ValueError("state-transition rules require committed-state observation")
    array_work_budget = _ArrayInvariantWorkBudget(_MAXIMUM_ARRAY_INVARIANT_WORK_UNITS)
    selected_outputs = _outputs_for_observation_authority(outputs, observation_authority)
    return tuple(
        _evaluate_rule_from_outputs(
            rule,
            selected_outputs,
            array_work_budget=array_work_budget,
        )
        for rule in rules
    )


def evaluate_dataset_invariant_rule_trials(
    rule: DatasetInvariantRule,
    trials: tuple[DatasetEvaluationTrial, ...],
    *,
    observation_authority: ObservationAuthority = "agent_response",
) -> DatasetInvariantRuleResult:
    if not trials:
        raise ValueError("invariant evaluation requires at least one trial")
    return _evaluate_rule(
        rule,
        trials,
        observation_authority=observation_authority,
        array_work_budget=_ArrayInvariantWorkBudget(_MAXIMUM_ARRAY_INVARIANT_WORK_UNITS),
    )


def _evaluate_arm(
    suite: DatasetInvariantSuite,
    trials: tuple[DatasetEvaluationTrial, ...],
    *,
    arm: Literal["baseline", "variation"],
    array_work_budget: _ArrayInvariantWorkBudget,
    operator_id: str | None = None,
) -> DatasetInvariantArmEvaluation:
    return DatasetInvariantArmEvaluation(
        arm=arm,
        operator_id=operator_id,
        rules=tuple(
            _evaluate_rule(
                rule,
                trials,
                observation_authority=suite.observation_authority,
                array_work_budget=array_work_budget,
            )
            for rule in suite.rules
        ),
    )


def _evaluate_rule(
    rule: DatasetInvariantRule,
    trials: tuple[DatasetEvaluationTrial, ...],
    *,
    observation_authority: ObservationAuthority,
    array_work_budget: _ArrayInvariantWorkBudget,
) -> DatasetInvariantRuleResult:
    outputs = tuple(trial.target_output for trial in trials)
    if _is_transition_rule(rule):
        outputs = tuple(_output_with_before_turn_checkpoint(trial) for trial in trials)
    return _evaluate_rule_from_outputs(
        rule,
        _outputs_for_observation_authority(
            outputs,
            observation_authority,
        ),
        array_work_budget=array_work_budget,
    )


def _output_with_before_turn_checkpoint(
    trial: DatasetEvaluationTrial,
) -> ObservedAgentOutput | None:
    output = trial.target_output
    evidence = trial.execution_evidence
    if (
        output is None
        or evidence is None
        or evidence.lifecycle.terminal_status != "succeeded"
        or evidence.initial_state is None
        or evidence.final_state is None
        or evidence.initial_state.authority != evidence.final_state.authority
        or evidence.initial_state.observer_id != evidence.final_state.observer_id
    ):
        return None
    return output.model_copy(
        update={
            "metadata": {
                "committed_state_before_turn": evidence.initial_state.value,
                "committed_state_snapshot": evidence.final_state.value,
                "state_observation_authority": evidence.final_state.authority,
            }
        }
    )


def _outputs_for_observation_authority(
    outputs: tuple[ObservedAgentOutput | None, ...],
    observation_authority: ObservationAuthority,
) -> tuple[ObservedAgentOutput | None, ...]:
    if observation_authority == "agent_response":
        return outputs
    if observation_authority == "committed_state_snapshot":
        return tuple(
            None
            if output is None
            or "committed_state_snapshot" not in output.metadata
            or output.metadata.get("state_observation_authority")
            not in {"environment_self_reported", "independent_observer"}
            else ObservedAgentOutput(
                raw_output=output.metadata["committed_state_snapshot"],
                metadata={
                    "committed_state_before_turn": output.metadata["committed_state_before_turn"]
                }
                if "committed_state_before_turn" in output.metadata
                else {},
            )
            for output in outputs
        )
    raise ValueError("unsupported observation authority")


def _evaluate_rule_from_outputs(
    rule: DatasetInvariantRule,
    outputs: tuple[ObservedAgentOutput | None, ...],
    *,
    array_work_budget: _ArrayInvariantWorkBudget,
) -> DatasetInvariantRuleResult:
    if isinstance(
        rule,
        (
            NoNewEffectInvariant,
            ExactlyOneNewEffectInvariant,
            UnchangedBetweenCheckpointsInvariant,
        ),
    ):
        return _evaluate_transition_rule(rule, outputs, array_work_budget=array_work_budget)
    if isinstance(rule, JsonValueEqualsLiteralInvariant):
        return _evaluate_value_equals_rule(rule, outputs)
    if isinstance(rule, JsonValueInAllowedSetInvariant):
        return _evaluate_value_in_set_rule(rule, outputs)
    if isinstance(rule, JsonArrayItemsUniqueByInvariant):
        return _evaluate_array_unique_rule(rule, outputs, array_work_budget=array_work_budget)
    trial_results = tuple(
        _evaluate_equal_output(rule, output, repetition)
        for repetition, output in enumerate(outputs, start=1)
    )
    status, reason_code = _aggregate_trial_statuses(trial_results)
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


def _evaluate_transition_rule(
    rule: NoNewEffectInvariant
    | ExactlyOneNewEffectInvariant
    | UnchangedBetweenCheckpointsInvariant,
    outputs: tuple[ObservedAgentOutput | None, ...],
    *,
    array_work_budget: _ArrayInvariantWorkBudget,
) -> DatasetInvariantTransitionRuleEvaluation:
    trials = tuple(
        _evaluate_transition_output(
            rule,
            output,
            repetition,
            array_work_budget=array_work_budget,
        )
        for repetition, output in enumerate(outputs, start=1)
    )
    status, reason_code = _aggregate_trial_statuses(trials)
    return DatasetInvariantTransitionRuleEvaluation(
        rule_type=rule.type,
        rule_id=rule.id,
        rule_version=rule.version,
        description=rule.description,
        severity=rule.severity,
        before_checkpoint=rule.before_checkpoint,
        after_checkpoint=rule.after_checkpoint,
        observation_pointer=rule.observation_pointer,
        status=status,
        reason_code=reason_code,
        trials=trials,
    )


def _evaluate_transition_output(
    rule: NoNewEffectInvariant
    | ExactlyOneNewEffectInvariant
    | UnchangedBetweenCheckpointsInvariant,
    output: ObservedAgentOutput | None,
    repetition: int,
    *,
    array_work_budget: _ArrayInvariantWorkBudget,
) -> DatasetInvariantTransitionTrialEvaluation:
    def result(
        status: InvariantStatus,
        reason_code: TransitionTrialReasonCode,
        *,
        before_item_count: int | None = None,
        after_item_count: int | None = None,
        new_effect_count: int | None = None,
    ) -> DatasetInvariantTransitionTrialEvaluation:
        return DatasetInvariantTransitionTrialEvaluation(
            repetition=repetition,
            status=status,
            reason_code=reason_code,
            before_checkpoint=rule.before_checkpoint,
            after_checkpoint=rule.after_checkpoint,
            observation_pointer=rule.observation_pointer,
            before_item_count=before_item_count,
            after_item_count=after_item_count,
            new_effect_count=new_effect_count,
        )

    if output is None:
        return result("not_evaluable", "after_checkpoint_missing")
    if "committed_state_before_turn" not in output.metadata:
        return result("not_evaluable", "before_checkpoint_missing")
    before_found, before_value = _resolve_json_pointer(
        output.metadata["committed_state_before_turn"], rule.observation_pointer
    )
    if not before_found:
        return result("not_evaluable", "before_pointer_missing")
    after_found, after_value = _resolve_json_pointer(output.raw_output, rule.observation_pointer)
    if not after_found:
        return result("not_evaluable", "after_pointer_missing")
    if isinstance(rule, UnchangedBetweenCheckpointsInvariant):
        unchanged = _json_values_equal(
            before_value,
            after_value,
            work_budget=array_work_budget,
        )
        if unchanged is None:
            return result("not_evaluable", "evaluation_work_limit_exceeded")
        return result(
            "satisfied" if unchanged else "violated",
            "value_unchanged" if unchanged else "value_changed",
        )
    if not isinstance(before_value, list):
        return result("not_evaluable", "before_value_not_array")
    if not isinstance(after_value, list):
        return result("not_evaluable", "after_value_not_array")
    before_items = cast(list[JsonValue], before_value)
    after_items = cast(list[JsonValue], after_value)
    before_count = len(before_items)
    after_count = len(after_items)
    if max(before_count, after_count) > _MAXIMUM_ARRAY_ITEMS:
        return result(
            "not_evaluable",
            "effect_array_exceeds_limit",
            before_item_count=before_count,
            after_item_count=after_count,
        )
    if after_count < before_count:
        return result(
            "not_evaluable",
            "effect_history_rewritten",
            before_item_count=before_count,
            after_item_count=after_count,
        )
    history_rewritten = False
    for before_item, after_item in zip(before_items, after_items, strict=False):
        items_equal = _json_values_equal(
            before_item,
            after_item,
            work_budget=array_work_budget,
        )
        if items_equal is None:
            return result(
                "not_evaluable",
                "evaluation_work_limit_exceeded",
                before_item_count=before_count,
                after_item_count=after_count,
            )
        if not items_equal:
            history_rewritten = True
            break
    if history_rewritten:
        return result(
            "not_evaluable",
            "effect_history_rewritten",
            before_item_count=before_count,
            after_item_count=after_count,
        )
    new_effect_count = after_count - before_count
    expected_new_effects = 0 if isinstance(rule, NoNewEffectInvariant) else 1
    satisfied = new_effect_count == expected_new_effects
    reason_code: TransitionTrialReasonCode
    if isinstance(rule, NoNewEffectInvariant):
        reason_code = "no_new_effect" if satisfied else "new_effect_observed"
    else:
        reason_code = "exactly_one_new_effect" if satisfied else "unexpected_new_effect_count"
    return result(
        "satisfied" if satisfied else "violated",
        reason_code,
        before_item_count=before_count,
        after_item_count=after_count,
        new_effect_count=new_effect_count,
    )


def _evaluate_equal_output(
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


def _evaluate_value_equals_rule(
    rule: JsonValueEqualsLiteralInvariant,
    outputs: tuple[ObservedAgentOutput | None, ...],
) -> DatasetInvariantValueEqualsRuleEvaluation:
    trials = tuple(
        _evaluate_value_equals_output(rule, output, repetition)
        for repetition, output in enumerate(outputs, start=1)
    )
    status, reason_code = _aggregate_trial_statuses(trials)
    return DatasetInvariantValueEqualsRuleEvaluation(
        rule_type=rule.type,
        rule_id=rule.id,
        rule_version=rule.version,
        description=rule.description,
        severity=rule.severity,
        value_pointer=rule.value_pointer,
        literal=rule.literal,
        status=status,
        reason_code=reason_code,
        trials=trials,
    )


def _evaluate_value_equals_output(
    rule: JsonValueEqualsLiteralInvariant,
    target_output: ObservedAgentOutput | None,
    repetition: int,
) -> DatasetInvariantValueEqualsTrialEvaluation:
    if target_output is None:
        return DatasetInvariantValueEqualsTrialEvaluation(
            repetition=repetition,
            status="not_evaluable",
            reason_code="target_output_missing",
            value_pointer=rule.value_pointer,
        )
    resolution, value = _resolve_supported_scalar(target_output.raw_output, rule.value_pointer)
    if resolution != "resolved":
        reason_by_resolution: dict[ScalarResolution, ValueEqualsTrialReasonCode] = {
            "missing": "value_pointer_missing",
            "not_scalar": "value_not_scalar",
            "non_integer_number": "value_non_integer_number_not_supported",
            "exceeds_limit": "value_exceeds_limit",
            "resolved": "value_equals_literal",
        }
        return DatasetInvariantValueEqualsTrialEvaluation(
            repetition=repetition,
            status="not_evaluable",
            reason_code=reason_by_resolution[resolution],
            value_pointer=rule.value_pointer,
        )
    comparable, equal = _json_scalars_equal(value, rule.literal)
    if comparable and equal:
        status: InvariantStatus = "satisfied"
        reason_code: ValueEqualsTrialReasonCode = "value_equals_literal"
    else:
        status = "violated"
        reason_code = "value_differs_from_literal"
    return DatasetInvariantValueEqualsTrialEvaluation(
        repetition=repetition,
        status=status,
        reason_code=reason_code,
        value_pointer=rule.value_pointer,
        resolved_values={"actual": cast(JsonValue, value)},
    )


def _evaluate_value_in_set_rule(
    rule: JsonValueInAllowedSetInvariant,
    outputs: tuple[ObservedAgentOutput | None, ...],
) -> DatasetInvariantValueInSetRuleEvaluation:
    trials = tuple(
        _evaluate_value_in_set_output(rule, output, repetition)
        for repetition, output in enumerate(outputs, start=1)
    )
    status, reason_code = _aggregate_trial_statuses(trials)
    return DatasetInvariantValueInSetRuleEvaluation(
        rule_type=rule.type,
        rule_id=rule.id,
        rule_version=rule.version,
        description=rule.description,
        severity=rule.severity,
        value_pointer=rule.value_pointer,
        allowed_values=rule.allowed_values,
        status=status,
        reason_code=reason_code,
        trials=trials,
    )


def _evaluate_value_in_set_output(
    rule: JsonValueInAllowedSetInvariant,
    target_output: ObservedAgentOutput | None,
    repetition: int,
) -> DatasetInvariantValueInSetTrialEvaluation:
    if target_output is None:
        return DatasetInvariantValueInSetTrialEvaluation(
            repetition=repetition,
            status="not_evaluable",
            reason_code="target_output_missing",
            value_pointer=rule.value_pointer,
        )
    resolution, value = _resolve_supported_scalar(target_output.raw_output, rule.value_pointer)
    if resolution != "resolved":
        reason_by_resolution: dict[ScalarResolution, ValueInSetTrialReasonCode] = {
            "missing": "value_pointer_missing",
            "not_scalar": "value_not_scalar",
            "non_integer_number": "value_non_integer_number_not_supported",
            "exceeds_limit": "value_exceeds_limit",
            "resolved": "value_in_allowed_set",
        }
        return DatasetInvariantValueInSetTrialEvaluation(
            repetition=repetition,
            status="not_evaluable",
            reason_code=reason_by_resolution[resolution],
            value_pointer=rule.value_pointer,
        )
    allowed = _value_in_set(value, rule.allowed_values)
    return DatasetInvariantValueInSetTrialEvaluation(
        repetition=repetition,
        status="satisfied" if allowed else "violated",
        reason_code="value_in_allowed_set" if allowed else "value_not_in_allowed_set",
        value_pointer=rule.value_pointer,
        resolved_values={"actual": cast(JsonValue, value)},
    )


def _evaluate_array_unique_rule(
    rule: JsonArrayItemsUniqueByInvariant,
    outputs: tuple[ObservedAgentOutput | None, ...],
    *,
    array_work_budget: _ArrayInvariantWorkBudget,
) -> DatasetInvariantArrayUniqueRuleEvaluation:
    trials = tuple(
        _evaluate_array_unique_output(
            rule,
            output,
            repetition,
            array_work_budget=array_work_budget,
        )
        for repetition, output in enumerate(outputs, start=1)
    )
    status, reason_code = _aggregate_trial_statuses(trials)
    return DatasetInvariantArrayUniqueRuleEvaluation(
        rule_type=rule.type,
        rule_id=rule.id,
        rule_version=rule.version,
        description=rule.description,
        severity=rule.severity,
        array_pointer=rule.array_pointer,
        key_pointers=rule.key_pointers,
        status=status,
        reason_code=reason_code,
        trials=trials,
    )


def _evaluate_array_unique_output(
    rule: JsonArrayItemsUniqueByInvariant,
    target_output: ObservedAgentOutput | None,
    repetition: int,
    *,
    array_work_budget: _ArrayInvariantWorkBudget,
) -> DatasetInvariantArrayUniqueTrialEvaluation:
    if target_output is None:
        return DatasetInvariantArrayUniqueTrialEvaluation(
            repetition=repetition,
            array_pointer=rule.array_pointer,
            key_pointers=rule.key_pointers,
            status="not_evaluable",
            reason_code="target_output_missing",
        )
    found, array_value = _resolve_json_pointer(target_output.raw_output, rule.array_pointer)
    if not found:
        return DatasetInvariantArrayUniqueTrialEvaluation(
            repetition=repetition,
            array_pointer=rule.array_pointer,
            key_pointers=rule.key_pointers,
            status="not_evaluable",
            reason_code="array_pointer_missing",
        )
    if not isinstance(array_value, list):
        return DatasetInvariantArrayUniqueTrialEvaluation(
            repetition=repetition,
            array_pointer=rule.array_pointer,
            key_pointers=rule.key_pointers,
            status="not_evaluable",
            reason_code="array_value_not_array",
        )
    array_items = cast(list[JsonValue], array_value)
    item_count = len(array_items)
    if item_count > _MAXIMUM_ARRAY_ITEMS:
        return DatasetInvariantArrayUniqueTrialEvaluation(
            repetition=repetition,
            array_pointer=rule.array_pointer,
            key_pointers=rule.key_pointers,
            status="not_evaluable",
            reason_code="array_exceeds_limit",
            item_count=item_count,
        )
    first_index_by_key: dict[tuple[tuple[str, str], ...], int] = {}
    first_duplicate: tuple[int, int] | None = None
    first_error: tuple[ArrayUniqueTrialReasonCode, int, str] | None = None
    key_pointer_tokens = tuple(
        (key_pointer, _json_pointer_tokens(key_pointer)) for key_pointer in rule.key_pointers
    )
    work_limit_reached = False
    for index, item in enumerate(array_items):
        key_parts: list[tuple[str, str]] = []
        for key_pointer, pointer_tokens in key_pointer_tokens:
            work_units = sum(max(1, len(token)) for token in pointer_tokens) or 1
            if not array_work_budget.consume(work_units):
                first_error = ("evaluation_work_limit_exceeded", index, key_pointer)
                work_limit_reached = True
                break
            resolution, key_value = _resolve_supported_scalar_tokens(item, pointer_tokens)
            if resolution != "resolved":
                if first_error is None:
                    reason_by_resolution: dict[ScalarResolution, ArrayUniqueTrialReasonCode] = {
                        "missing": "key_pointer_missing",
                        "not_scalar": "key_value_not_scalar",
                        "non_integer_number": "key_non_integer_number_not_supported",
                        "exceeds_limit": "key_value_exceeds_limit",
                        "resolved": "array_items_unique",
                    }
                    first_error = (reason_by_resolution[resolution], index, key_pointer)
                break
            key_parts.append(_typed_scalar_key(key_value))
        else:
            composite_key = tuple(key_parts)
            previous_index = first_index_by_key.get(composite_key)
            if previous_index is not None and first_duplicate is None:
                first_duplicate = (previous_index, index)
            else:
                first_index_by_key[composite_key] = index
        if work_limit_reached:
            break
    if first_duplicate is not None:
        return DatasetInvariantArrayUniqueTrialEvaluation(
            repetition=repetition,
            array_pointer=rule.array_pointer,
            key_pointers=rule.key_pointers,
            status="violated",
            reason_code="duplicate_array_items",
            item_count=item_count,
            duplicate_indices=first_duplicate,
        )
    if first_error is not None:
        reason_code, failed_item_index, failed_key_pointer = first_error
        return DatasetInvariantArrayUniqueTrialEvaluation(
            repetition=repetition,
            array_pointer=rule.array_pointer,
            key_pointers=rule.key_pointers,
            status="not_evaluable",
            reason_code=reason_code,
            item_count=item_count,
            failed_item_index=failed_item_index,
            failed_key_pointer=failed_key_pointer,
        )
    return DatasetInvariantArrayUniqueTrialEvaluation(
        repetition=repetition,
        array_pointer=rule.array_pointer,
        key_pointers=rule.key_pointers,
        status="satisfied",
        reason_code="array_items_unique",
        item_count=item_count,
    )


def _trial_result(
    rule: JsonValuesEqualInvariant,
    repetition: int,
    status: InvariantStatus,
    reason_code: EqualityTrialReasonCode,
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


def _validate_single_value_evidence(
    status: InvariantStatus,
    reason_code: str,
    resolved_values: dict[Literal["actual"], JsonValue],
    expected_status_by_reason: dict[str, InvariantStatus],
    reasons_with_actual_value: set[str],
) -> None:
    if status != expected_status_by_reason[reason_code]:
        raise ValueError("trial invariant status must match its reason")
    expected_keys: set[str] = {"actual"} if reason_code in reasons_with_actual_value else set()
    if set(resolved_values) != expected_keys:
        raise ValueError("resolved invariant values must match the trial reason")
    if any(not _configured_scalar_supported(value) for value in resolved_values.values()):
        raise ValueError("resolved invariant values must contain supported JSON scalars")


def _validate_rule_aggregate(
    status: InvariantStatus,
    reason_code: AggregateReasonCode,
    trials: Sequence[
        DatasetInvariantTrialEvaluation
        | DatasetInvariantValueEqualsTrialEvaluation
        | DatasetInvariantValueInSetTrialEvaluation
        | DatasetInvariantArrayUniqueTrialEvaluation
        | DatasetInvariantTransitionTrialEvaluation
    ],
) -> None:
    if tuple(trial.repetition for trial in trials) != tuple(range(1, len(trials) + 1)):
        raise ValueError("invariant trials must preserve repetition order")
    expected_status, expected_reason = _aggregate_trial_statuses(trials)
    if status != expected_status or reason_code != expected_reason:
        raise ValueError("aggregate invariant result must match its trials")


def _aggregate_trial_statuses(
    trials: Sequence[
        DatasetInvariantTrialEvaluation
        | DatasetInvariantValueEqualsTrialEvaluation
        | DatasetInvariantValueInSetTrialEvaluation
        | DatasetInvariantArrayUniqueTrialEvaluation
        | DatasetInvariantTransitionTrialEvaluation
    ],
) -> tuple[InvariantStatus, AggregateReasonCode]:
    statuses = {trial.status for trial in trials}
    if "violated" in statuses:
        return "violated", "one_or_more_trials_violated"
    if "not_evaluable" in statuses:
        return "not_evaluable", "one_or_more_trials_not_evaluable"
    return "satisfied", "all_trials_satisfied"


def _resolve_supported_scalar(document: JsonValue, pointer: str) -> tuple[ScalarResolution, object]:
    return _resolve_supported_scalar_tokens(document, _json_pointer_tokens(pointer))


def _resolve_supported_scalar_tokens(
    document: JsonValue,
    pointer_tokens: tuple[str, ...],
) -> tuple[ScalarResolution, object]:
    found, value = _resolve_json_pointer_tokens(document, pointer_tokens)
    if not found:
        return "missing", None
    if not _is_json_scalar(value):
        return "not_scalar", None
    if isinstance(value, float):
        return "non_integer_number", None
    if not _resolved_value_fits(value):
        return "exceeds_limit", None
    return "resolved", value


def _resolve_json_pointer(document: JsonValue, pointer: str) -> tuple[bool, object]:
    return _resolve_json_pointer_tokens(document, _json_pointer_tokens(pointer))


def _json_pointer_tokens(pointer: str) -> tuple[str, ...]:
    if pointer == "":
        return ()
    return tuple(
        encoded_token.replace("~1", "/").replace("~0", "~")
        for encoded_token in pointer[1:].split("/")
    )


def _resolve_json_pointer_tokens(
    document: JsonValue,
    pointer_tokens: tuple[str, ...],
) -> tuple[bool, object]:
    current: object = document
    for token in pointer_tokens:
        if isinstance(current, dict):
            current_object = cast(dict[str, object], current)
            if token in current_object:
                current = current_object[token]
                continue
        if isinstance(current, list):
            current_array = cast(list[object], current)
            array_index = _bounded_array_index(token, len(current_array))
            if array_index is not None:
                current = current_array[array_index]
                continue
        return False, None
    return True, current


def _bounded_array_index(token: str, array_length: int) -> int | None:
    valid_syntax = token == "0" or (
        token.isascii() and token.isdecimal() and not token.startswith("0")
    )
    if not valid_syntax or array_length == 0:
        return None
    maximum_index_text = str(array_length - 1)
    if len(token) > len(maximum_index_text) or (
        len(token) == len(maximum_index_text) and token > maximum_index_text
    ):
        return None
    return int(token)


def _is_json_scalar(value: object) -> bool:
    return (
        value is None
        or isinstance(value, (str, bool, int))
        or (isinstance(value, float) and math.isfinite(value))
    )


def _json_scalars_equal(left: object, right: object) -> tuple[bool, bool]:
    if _is_json_number(left) and _is_json_number(right):
        return True, left == right
    if type(left) is not type(right):
        return False, False
    return True, left == right


def _json_values_equal(
    left: object,
    right: object,
    *,
    work_budget: _ArrayInvariantWorkBudget,
) -> bool | None:
    if _is_json_scalar(left) or _is_json_scalar(right):
        if not work_budget.consume(
            max(_json_scalar_work_units(left), _json_scalar_work_units(right))
        ):
            return None
        comparable, equal = _json_scalars_equal(left, right)
        return comparable and equal
    if not work_budget.consume(1):
        return None
    if isinstance(left, list) and isinstance(right, list):
        left_items = cast(list[object], left)
        right_items = cast(list[object], right)
        if len(left_items) != len(right_items):
            return False
        for left_item, right_item in zip(left_items, right_items, strict=True):
            items_equal = _json_values_equal(
                left_item,
                right_item,
                work_budget=work_budget,
            )
            if items_equal is not True:
                return items_equal
        return True
    if isinstance(left, dict) and isinstance(right, dict):
        left_object = cast(dict[str, object], left)
        right_object = cast(dict[str, object], right)
        if len(left_object) != len(right_object):
            return False
        for key in left_object:
            if not work_budget.consume(max(1, len(key))):
                return None
            if key not in right_object:
                return False
            values_equal = _json_values_equal(
                left_object[key],
                right_object[key],
                work_budget=work_budget,
            )
            if values_equal is not True:
                return values_equal
        return True
    return False


def _json_scalar_work_units(value: object) -> int:
    if isinstance(value, str):
        return max(1, len(value))
    if isinstance(value, int) and not isinstance(value, bool):
        return max(1, (value.bit_length() + 7) // 8)
    return 1


def _typed_scalar_key(value: object) -> tuple[str, str]:
    if value is None:
        value_type = "null"
    elif isinstance(value, bool):
        value_type = "boolean"
    elif isinstance(value, int):
        value_type = "integer"
    elif isinstance(value, str):
        value_type = "string"
    else:
        raise ValueError("typed scalar key requires a supported JSON scalar")
    return value_type, _canonical_json_value(cast(JsonValue, value))


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


def _configured_scalar_supported(value: object) -> bool:
    return _is_json_scalar(value) and not isinstance(value, float) and _resolved_value_fits(value)


def _validate_configured_scalar(value: JsonValue, label: str) -> JsonValue:
    if not _is_json_scalar(value):
        raise ValueError(f"{label} must be a JSON scalar")
    if isinstance(value, float):
        raise ValueError(f"{label} must not be a non-integer JSON number")
    if not _resolved_value_fits(value):
        raise ValueError(f"{label} exceeds the evidence limit")
    return value


def _validate_key_pointers(pointers: tuple[str, ...]) -> tuple[str, ...]:
    if len(pointers) != len(set(pointers)):
        raise ValueError("key pointers must be unique")
    if any(
        len(pointer) > 1_000 or re.fullmatch(_JSON_POINTER_PATTERN, pointer) is None
        for pointer in pointers
    ):
        raise ValueError("key pointers must be RFC 6901 JSON pointers")
    return pointers


def _value_in_set(value: object, allowed_values: tuple[JsonValue, ...]) -> bool:
    return any(
        _json_scalars_equal(value, allowed_value) == (True, True)
        for allowed_value in allowed_values
    )


def _rule_identity(rule: DatasetInvariantRuleResult) -> tuple[object, ...]:
    common = (
        rule.rule_type,
        rule.rule_id,
        rule.rule_version,
        rule.description,
        rule.severity,
    )
    if isinstance(rule, DatasetInvariantRuleEvaluation):
        first_trial = rule.trials[0]
        return (*common, first_trial.left_pointer, first_trial.right_pointer)
    if isinstance(rule, DatasetInvariantValueEqualsRuleEvaluation):
        return (*common, rule.value_pointer, _canonical_json_value(rule.literal))
    if isinstance(rule, DatasetInvariantValueInSetRuleEvaluation):
        return (
            *common,
            rule.value_pointer,
            tuple(_canonical_json_value(value) for value in rule.allowed_values),
        )
    if isinstance(rule, DatasetInvariantArrayUniqueRuleEvaluation):
        return (*common, rule.array_pointer, rule.key_pointers)
    return (
        *common,
        rule.before_checkpoint,
        rule.after_checkpoint,
        rule.observation_pointer,
    )


def _is_transition_rule(
    rule: DatasetInvariantRule,
) -> TypeGuard[
    NoNewEffectInvariant | ExactlyOneNewEffectInvariant | UnchangedBetweenCheckpointsInvariant
]:
    return isinstance(
        rule,
        (
            NoNewEffectInvariant,
            ExactlyOneNewEffectInvariant,
            UnchangedBetweenCheckpointsInvariant,
        ),
    )


def _canonical_json_value(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


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
