from __future__ import annotations

import hashlib
import json
import math
from typing import Annotated, Literal, Self, cast

from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from ul_core.models import ToolCall, ULModel


class _StrictModel(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _EvaluatorSpecModel(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    id: str = Field(min_length=1, max_length=200)


class ExactValueEvaluator(_EvaluatorSpecModel):
    type: Literal["exact_value"] = "exact_value"
    source: Literal["answer", "final_state", "http_body"]
    json_pointer: str = ""
    expected: JsonValue


class JsonPropertyEvaluator(_EvaluatorSpecModel):
    type: Literal["json_property"] = "json_property"
    source: Literal["answer", "final_state", "http_body"]
    json_pointer: str
    operator: Literal["exists", "not_exists", "equals", "type"]
    expected: JsonValue = None
    expected_type: Literal["null", "boolean", "number", "string", "array", "object"] | None = None

    @model_validator(mode="after")
    def validate_operator_parameters(self) -> Self:
        expected_was_set = "expected" in self.model_fields_set
        if self.operator == "type":
            if self.expected_type is None:
                raise ValueError("type checks require expected_type")
            if expected_was_set:
                raise ValueError("expected is not valid for type checks")
        elif self.operator == "equals":
            if not expected_was_set:
                raise ValueError("equals checks require expected")
            if self.expected_type is not None:
                raise ValueError("expected_type is only valid for type checks")
        elif expected_was_set or self.expected_type is not None:
            raise ValueError("existence checks do not accept expected values or types")
        return self

    @model_serializer(mode="wrap")
    def serialize_operator_parameters(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        serialized = cast(dict[str, object], handler(self))
        if self.operator != "equals":
            serialized.pop("expected", None)
        if self.operator != "type":
            serialized.pop("expected_type", None)
        return serialized


class ToolCallEvaluator(_EvaluatorSpecModel):
    type: Literal["tool_call"] = "tool_call"
    tool_name: str = Field(min_length=1, max_length=500)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    arguments_match: Literal["contains", "exact"] = "contains"
    minimum_calls: int = Field(default=1, ge=1)


class StateChangeEvaluator(_EvaluatorSpecModel):
    type: Literal["state_change"] = "state_change"
    json_pointer: str = ""
    operator: Literal["changed", "unchanged", "equals"]
    expected: JsonValue = None

    @model_validator(mode="after")
    def validate_operator_parameters(self) -> Self:
        expected_was_set = "expected" in self.model_fields_set
        if self.operator == "equals" and not expected_was_set:
            raise ValueError("equals state checks require expected")
        if self.operator != "equals" and expected_was_set:
            raise ValueError("changed and unchanged state checks do not accept expected")
        return self

    @model_serializer(mode="wrap")
    def serialize_operator_parameters(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        serialized = cast(dict[str, object], handler(self))
        if self.operator != "equals":
            serialized.pop("expected", None)
        return serialized


class HttpResultEvaluator(_EvaluatorSpecModel):
    type: Literal["http_result"] = "http_result"
    status_code: int | None = Field(default=None, ge=100, le=599)
    body_json_pointer: str | None = None
    expected_body_value: JsonValue = None

    @model_validator(mode="after")
    def validate_assertion(self) -> Self:
        expected_body_was_set = "expected_body_value" in self.model_fields_set
        if self.status_code is None and self.body_json_pointer is None:
            raise ValueError("HTTP evaluators require a status or body assertion")
        if (self.body_json_pointer is None) != (not expected_body_was_set):
            raise ValueError("HTTP body pointers and expected values must be provided together")
        return self

    @model_serializer(mode="wrap")
    def serialize_body_assertion(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        serialized = cast(dict[str, object], handler(self))
        if self.body_json_pointer is None:
            serialized.pop("expected_body_value", None)
        return serialized


class CallableEvaluator(_EvaluatorSpecModel):
    type: Literal["callable"] = "callable"
    callable_id: str = Field(min_length=1, max_length=500, pattern=r"[a-zA-Z0-9][a-zA-Z0-9._:-]*")


class RubricEvaluator(_EvaluatorSpecModel):
    type: Literal["rubric"] = "rubric"
    rubric: str = Field(min_length=1, max_length=20_000)
    minimum_score: float = Field(default=1, ge=0, le=1)
    include_sources: tuple[
        Literal["answer", "tool_calls", "initial_state", "final_state", "http_result"], ...
    ] = ("answer",)
    allow_private_data: bool = False


class PairwiseEvaluator(_EvaluatorSpecModel):
    type: Literal["pairwise"] = "pairwise"
    rubric: str = Field(min_length=1, max_length=20_000)
    allow_tie: bool = True
    include_sources: tuple[
        Literal["answer", "tool_calls", "initial_state", "final_state", "http_result"], ...
    ] = ("answer",)
    allow_private_data: bool = False


class HumanReviewEvaluator(_EvaluatorSpecModel):
    type: Literal["human_review"] = "human_review"
    instructions: str = Field(min_length=1, max_length=20_000)


EvaluatorSpec = Annotated[
    ExactValueEvaluator
    | JsonPropertyEvaluator
    | ToolCallEvaluator
    | StateChangeEvaluator
    | HttpResultEvaluator
    | CallableEvaluator
    | RubricEvaluator
    | PairwiseEvaluator
    | HumanReviewEvaluator,
    Field(discriminator="type"),
]


class HttpEvaluationResult(_StrictModel):
    status_code: int = Field(ge=100, le=599)
    body: JsonValue = None


class EvaluationSubject(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    agent_status: Literal["succeeded", "failed", "timed_out"]
    answer: JsonValue = None
    reference_answer: JsonValue = None
    tool_calls: tuple[ToolCall, ...] = ()
    initial_state: JsonValue = None
    final_state: JsonValue = None
    http_result: HttpEvaluationResult | None = None
    public_context: dict[str, JsonValue] = Field(default_factory=dict)
    private_data: dict[str, JsonValue] = Field(default_factory=dict, repr=False)
    private_json_pointers: tuple[str, ...] = ()
    agent_failure_reason: str | None = Field(default=None, min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def validate_agent_status(self) -> Self:
        if (self.agent_status == "succeeded") == (self.agent_failure_reason is not None):
            raise ValueError("agent failure detail must be present exactly when the agent failed")
        return self


class EvaluatorEvidence(_StrictModel):
    source: Literal[
        "answer",
        "tool_call",
        "initial_state",
        "final_state",
        "http_result",
        "judge_payload",
        "callable",
    ]
    json_pointer: str = Field(default="", max_length=2_000)
    description: str = Field(min_length=1, max_length=1_000)


class EvaluatorDecision(_StrictModel):
    passed: bool | None = None
    score: float | None = Field(default=None, ge=0, le=1)
    label: str | None = Field(default=None, min_length=1, max_length=500)
    explanation: str = Field(min_length=1, max_length=5_000)
    evidence: tuple[EvaluatorEvidence, ...] = Field(default=(), max_length=20)


class EvaluatorResult(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    evaluator_id: str = Field(min_length=1, max_length=200)
    evaluator_type: str = Field(min_length=1, max_length=100)
    status: Literal["passed", "failed", "needs_review", "evaluator_error", "agent_error"]
    score: float | None = Field(default=None, ge=0, le=1)
    label: str | None = Field(default=None, min_length=1, max_length=500)
    explanation: str = Field(min_length=1, max_length=5_000)
    evidence: tuple[EvaluatorEvidence, ...] = Field(default=(), max_length=20)


class EvaluatorJudgeVersion(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    prompt_version: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str = Field(min_length=1, max_length=200)
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvaluatorVersion(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    id: str = Field(pattern=r"^ulev_v1_[0-9a-f]{64}$")
    evaluator_id: str = Field(min_length=1, max_length=200)
    evaluator_type: str = Field(min_length=1, max_length=100)
    evaluator_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    judge: EvaluatorJudgeVersion | None = None


class EvaluatorCalibrationExample(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    id: str = Field(min_length=1, max_length=500)
    kind: Literal["known_good", "known_bad", "borderline"]
    subject: EvaluationSubject
    expected_passed: bool
    repetitions: int = Field(default=1, ge=1, le=20)
    human_labels: tuple[bool, ...] = Field(default=(), max_length=100)

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        if self.kind == "known_good" and not self.expected_passed:
            raise ValueError("known-good examples must expect a passing judgment")
        if self.kind == "known_bad" and self.expected_passed:
            raise ValueError("known-bad examples must expect a failing judgment")
        if self.kind == "borderline" and self.repetitions < 2:
            raise ValueError("borderline examples require at least two repetitions")
        return self


class EvaluatorCalibrationExampleResult(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    example_id: str = Field(min_length=1, max_length=500)
    kind: Literal["known_good", "known_bad", "borderline"]
    expected_passed: bool
    human_labels: tuple[bool, ...] = Field(default=(), max_length=100)
    results: tuple[EvaluatorResult, ...] = Field(min_length=1, max_length=20)
    false_positive: bool = False
    false_negative: bool = False
    unstable: bool = False
    human_disagreement: bool = False
    human_agreement: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if self.kind == "known_good" and not self.expected_passed:
            raise ValueError("known-good results must expect a passing judgment")
        if self.kind == "known_bad" and self.expected_passed:
            raise ValueError("known-bad results must expect a failing judgment")
        if self.kind == "borderline" and len(self.results) < 2:
            raise ValueError("borderline results require at least two judgments")
        judged_passes = tuple(
            result.status == "passed"
            for result in self.results
            if result.status in {"passed", "failed"}
        )
        expected_false_positive = self.kind == "known_bad" and any(judged_passes)
        expected_false_negative = self.kind == "known_good" and any(
            not passed for passed in judged_passes
        )
        observed_outcomes = {(result.status, result.score, result.label) for result in self.results}
        expected_unstable = self.kind == "borderline" and len(observed_outcomes) > 1
        expected_human_disagreement = len(set(self.human_labels)) > 1
        expected_human_agreement = _human_agreement(judged_passes, self.human_labels)
        if self.false_positive != expected_false_positive:
            raise ValueError("false-positive summary does not match judgments")
        if self.false_negative != expected_false_negative:
            raise ValueError("false-negative summary does not match judgments")
        if self.unstable != expected_unstable:
            raise ValueError("instability summary does not match repeated judgments")
        if self.human_disagreement != expected_human_disagreement:
            raise ValueError("human disagreement does not match human labels")
        if self.human_agreement is None or expected_human_agreement is None:
            if self.human_agreement != expected_human_agreement:
                raise ValueError("human agreement does not match judgments")
        elif not math.isclose(self.human_agreement, expected_human_agreement):
            raise ValueError("human agreement does not match judgments")
        return self


class EvaluatorCalibrationReport(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    id: str = Field(pattern=r"^ulec_v1_[0-9a-f]{64}$")
    evaluator_version: EvaluatorVersion
    status: Literal["reliable", "unreliable"]
    examples: tuple[EvaluatorCalibrationExampleResult, ...] = Field(min_length=3)
    false_positive_examples: tuple[str, ...] = ()
    false_negative_examples: tuple[str, ...] = ()
    unstable_examples: tuple[str, ...] = ()
    human_disagreement_examples: tuple[str, ...] = ()
    human_agreement: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        example_ids = tuple(example.example_id for example in self.examples)
        if len(example_ids) != len(set(example_ids)):
            raise ValueError("calibration result identifiers must be unique")
        if not {"known_good", "known_bad", "borderline"} <= {
            example.kind for example in self.examples
        }:
            raise ValueError("calibration requires known-good, known-bad, and borderline results")
        if any(
            result.evaluator_id != self.evaluator_version.evaluator_id
            or result.evaluator_type != self.evaluator_version.evaluator_type
            for example in self.examples
            for result in example.results
        ):
            raise ValueError("calibration judgments must match the evaluator version")
        expected_false_positives = tuple(
            example.example_id for example in self.examples if example.false_positive
        )
        expected_false_negatives = tuple(
            example.example_id for example in self.examples if example.false_negative
        )
        expected_unstable = tuple(
            example.example_id for example in self.examples if example.unstable
        )
        expected_human_disagreement = tuple(
            example.example_id for example in self.examples if example.human_disagreement
        )
        summaries = (
            (self.false_positive_examples, expected_false_positives),
            (self.false_negative_examples, expected_false_negatives),
            (self.unstable_examples, expected_unstable),
            (self.human_disagreement_examples, expected_human_disagreement),
        )
        if any(actual != expected for actual, expected in summaries):
            raise ValueError("calibration report summaries do not match example results")
        comparable_agreement = tuple(
            example.human_agreement
            for example in self.examples
            if example.human_agreement is not None
        )
        expected_agreement = (
            sum(comparable_agreement) / len(comparable_agreement) if comparable_agreement else None
        )
        if self.human_agreement is None or expected_agreement is None:
            if self.human_agreement != expected_agreement:
                raise ValueError("report human agreement does not match example results")
        elif not math.isclose(self.human_agreement, expected_agreement):
            raise ValueError("report human agreement does not match example results")
        expected_unreliable = any(
            (
                expected_false_positives,
                expected_false_negatives,
                expected_unstable,
                expected_human_disagreement,
                any(
                    result.status not in {"passed", "failed"}
                    for example in self.examples
                    for result in example.results
                ),
            )
        )
        if (self.status == "unreliable") != expected_unreliable:
            raise ValueError("calibration reliability does not match example results")
        report_payload: dict[str, JsonValue] = {
            "evaluator_version": self.evaluator_version.model_dump(mode="json"),
            "examples": [example.model_dump(mode="json") for example in self.examples],
        }
        if self.id != f"ulec_v1_{_sha256_json(report_payload)}":
            raise ValueError("calibration report identifier does not match its contents")
        return self


class EvaluatorReliability(_StrictModel):
    evaluator_id: str = Field(min_length=1, max_length=200)
    evaluator_version_id: str = Field(pattern=r"^ulev_v1_[0-9a-f]{64}$")
    status: Literal["uncalibrated", "reliable", "unreliable"]
    calibration_report_id: str | None = Field(default=None, pattern=r"^ulec_v1_[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_calibration_reference(self) -> Self:
        if (self.status == "uncalibrated") != (self.calibration_report_id is None):
            raise ValueError("only calibrated evaluator reliability can reference a report")
        return self


class EvaluationResults(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    results: tuple[EvaluatorResult, ...]
    reliability: tuple[EvaluatorReliability, ...] = ()


def _sha256_json(value: JsonValue) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def _human_agreement(
    judged_passes: tuple[bool, ...],
    human_labels: tuple[bool, ...],
) -> float | None:
    if not judged_passes or not human_labels:
        return None
    comparisons = tuple(
        judged_passed == human_label
        for judged_passed in judged_passes
        for human_label in human_labels
    )
    return sum(comparisons) / len(comparisons)
