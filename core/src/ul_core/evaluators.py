from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, JsonValue, model_validator

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
        if self.operator == "type" and self.expected_type is None:
            raise ValueError("type checks require expected_type")
        if self.operator != "type" and self.expected_type is not None:
            raise ValueError("expected_type is only valid for type checks")
        return self


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


class HttpResultEvaluator(_EvaluatorSpecModel):
    type: Literal["http_result"] = "http_result"
    status_code: int | None = Field(default=None, ge=100, le=599)
    body_json_pointer: str | None = None
    expected_body_value: JsonValue = None

    @model_validator(mode="after")
    def validate_assertion(self) -> Self:
        if self.status_code is None and self.body_json_pointer is None:
            raise ValueError("HTTP evaluators require a status or body assertion")
        return self


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
        "callable",
        "judge",
    ]
    json_pointer: str = ""
    description: str = Field(min_length=1, max_length=1_000)


class EvaluatorDecision(_StrictModel):
    passed: bool | None = None
    score: float | None = Field(default=None, ge=0, le=1)
    label: str | None = Field(default=None, min_length=1, max_length=500)
    explanation: str = Field(min_length=1, max_length=5_000)
    evidence: tuple[EvaluatorEvidence, ...] = ()


class EvaluatorResult(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    evaluator_id: str = Field(min_length=1, max_length=200)
    evaluator_type: str = Field(min_length=1, max_length=100)
    status: Literal["passed", "failed", "needs_review", "evaluator_error", "agent_error"]
    score: float | None = Field(default=None, ge=0, le=1)
    label: str | None = Field(default=None, min_length=1, max_length=500)
    explanation: str = Field(min_length=1, max_length=5_000)
    evidence: tuple[EvaluatorEvidence, ...] = ()


class EvaluationResults(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    results: tuple[EvaluatorResult, ...]
