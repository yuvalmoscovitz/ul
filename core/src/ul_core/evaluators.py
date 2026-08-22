from __future__ import annotations

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


class EvaluationResults(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    results: tuple[EvaluatorResult, ...]
