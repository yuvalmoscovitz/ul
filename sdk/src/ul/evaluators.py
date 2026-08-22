from __future__ import annotations

import inspect
import ipaddress
import json
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from types import TracebackType
from typing import Any, Literal, Protocol, Self, cast
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, SecretStr, model_validator
from ul_core.evaluators import (
    CallableEvaluator,
    EvaluationResults,
    EvaluationSubject,
    EvaluatorDecision,
    EvaluatorEvidence,
    EvaluatorResult,
    EvaluatorSpec,
    ExactValueEvaluator,
    HttpResultEvaluator,
    HumanReviewEvaluator,
    JsonPropertyEvaluator,
    PairwiseEvaluator,
    RubricEvaluator,
    StateChangeEvaluator,
    ToolCallEvaluator,
)
from ul_core.models import ULModel
from ul_core.prompts import PromptManager

_PROMPTS = PromptManager.instance()

EvaluatorCallable = Callable[[EvaluationSubject], EvaluatorDecision | Awaitable[EvaluatorDecision]]


class JudgeRequest(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    evaluator_id: str
    mode: Literal["rubric", "pairwise"]
    rubric: str
    payload: dict[str, JsonValue]
    allow_tie: bool = False


class EvaluatorJudge(Protocol):
    def evaluate(self, request: JudgeRequest) -> Awaitable[EvaluatorDecision]: ...


class OpenAICompatibleJudgeConfig(ULModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        hide_input_in_errors=True,
    )

    base_url: str = Field(min_length=1, max_length=2_000)
    model: str = Field(min_length=1, max_length=200)
    api_key: SecretStr | None = Field(default=None, repr=False)
    allow_external_data_processing: Literal[True]
    data_policy: Literal["provider_default", "openrouter_zdr"] = "provider_default"
    timeout_seconds: float = Field(default=60, gt=0, le=300)
    max_output_tokens: int = Field(default=1_024, ge=64, le=8_192)

    @model_validator(mode="after")
    def validate_and_normalize_base_url(self) -> Self:
        object.__setattr__(self, "base_url", _validated_judge_base_url(self.base_url))
        return self


def _validated_judge_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("judge base_url must use https or loopback http")
    if not parsed.hostname:
        raise ValueError("judge base_url must include a host")
    try:
        _ = parsed.port
    except ValueError:
        raise ValueError("judge base_url has an invalid port") from None
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("judge base_url must not include credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("judge base_url must not include a query or fragment")
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise ValueError("judge base_url only permits plaintext HTTP on loopback")
    normalized_path = parsed.path.rstrip("/")
    if normalized_path.endswith("/chat/completions"):
        raise ValueError("judge base_url must be an API root")
    return urlunsplit((parsed.scheme, parsed.netloc, normalized_path, "", ""))


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class _JudgeMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    content: str = Field(min_length=1)


class _JudgeChoice(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    message: _JudgeMessage


class _JudgeResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    choices: tuple[_JudgeChoice, ...] = Field(min_length=1)


class _JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    score: float | None = Field(ge=0, le=1)
    label: str | None = Field(min_length=1, max_length=500)
    explanation: str = Field(min_length=1, max_length=5_000)


class OpenAICompatibleEvaluatorJudge:
    def __init__(
        self,
        config: OpenAICompatibleJudgeConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=config.timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def evaluate(self, request: JudgeRequest) -> EvaluatorDecision:
        request_body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": _PROMPTS.get_prompt("evaluation.judge"),
                },
                {
                    "role": "user",
                    "content": json.dumps(request.model_dump(mode="json"), sort_keys=True),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ul_evaluator_decision",
                    "strict": True,
                    "schema": _JudgeOutput.model_json_schema(mode="validation"),
                },
            },
            "max_completion_tokens": self.config.max_output_tokens,
            "temperature": 0,
        }
        if self.config.data_policy == "openrouter_zdr":
            request_body["provider"] = {
                "data_collection": "deny",
                "require_parameters": True,
                "zdr": True,
            }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key is not None:
            headers["Authorization"] = f"Bearer {self.config.api_key.get_secret_value()}"
        response = await self._client.post(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=request_body,
        )
        response.raise_for_status()
        parsed_response = _JudgeResponse.model_validate(response.json())
        output = _JudgeOutput.model_validate_json(parsed_response.choices[0].message.content)
        return EvaluatorDecision(
            score=output.score,
            label=output.label,
            explanation=output.explanation,
            evidence=(
                EvaluatorEvidence(
                    source="judge",
                    description="Structured decision returned by the configured judge model.",
                ),
            ),
        )


async def evaluate(
    subject: EvaluationSubject,
    evaluators: tuple[EvaluatorSpec, ...],
    *,
    judge: EvaluatorJudge | None = None,
    callables: Mapping[str, EvaluatorCallable] | None = None,
) -> EvaluationResults:
    results: list[EvaluatorResult] = []
    for evaluator in evaluators:
        if subject.agent_status != "succeeded":
            results.append(
                EvaluatorResult(
                    evaluator_id=evaluator.id,
                    evaluator_type=evaluator.type,
                    status="agent_error",
                    explanation=subject.agent_failure_reason or "The agent did not complete.",
                )
            )
            continue
        try:
            results.append(
                await _evaluate_one(
                    subject,
                    evaluator,
                    judge=judge,
                    callables=callables or {},
                )
            )
        except Exception:
            results.append(
                EvaluatorResult(
                    evaluator_id=evaluator.id,
                    evaluator_type=evaluator.type,
                    status="evaluator_error",
                    explanation="The evaluator could not produce a valid result.",
                )
            )
    return EvaluationResults(results=tuple(results))


async def _evaluate_one(
    subject: EvaluationSubject,
    evaluator: EvaluatorSpec,
    *,
    judge: EvaluatorJudge | None,
    callables: Mapping[str, EvaluatorCallable],
) -> EvaluatorResult:
    if isinstance(evaluator, ExactValueEvaluator):
        source_value = _source_value(subject, evaluator.source)
        found, observed = _resolve_json_pointer(source_value, evaluator.json_pointer)
        passed = found and observed == evaluator.expected
        return _deterministic_result(
            evaluator,
            passed,
            "Observed value matched the expected value."
            if passed
            else "Observed value did not match the expected value.",
            _evidence_source(evaluator.source),
            evaluator.json_pointer,
        )
    if isinstance(evaluator, JsonPropertyEvaluator):
        source_value = _source_value(subject, evaluator.source)
        found, observed = _resolve_json_pointer(source_value, evaluator.json_pointer)
        if evaluator.operator == "exists":
            passed = found
        elif evaluator.operator == "not_exists":
            passed = not found
        elif evaluator.operator == "equals":
            passed = found and observed == evaluator.expected
        else:
            passed = found and _json_type(observed) == evaluator.expected_type
        return _deterministic_result(
            evaluator,
            passed,
            "JSON property satisfied the assertion."
            if passed
            else "JSON property did not satisfy the assertion.",
            _evidence_source(evaluator.source),
            evaluator.json_pointer,
        )
    if isinstance(evaluator, ToolCallEvaluator):
        matching_indices = tuple(
            index
            for index, tool_call in enumerate(subject.tool_calls)
            if tool_call.name == evaluator.tool_name
            and (
                tool_call.arguments == evaluator.arguments
                if evaluator.arguments_match == "exact"
                else all(
                    key in tool_call.arguments and tool_call.arguments[key] == value
                    for key, value in evaluator.arguments.items()
                )
            )
        )
        passed = len(matching_indices) >= evaluator.minimum_calls
        return _deterministic_result(
            evaluator,
            passed,
            "Expected tool call was observed."
            if passed
            else "Expected tool call was not observed.",
            "tool_call",
            f"/{matching_indices[0]}" if matching_indices else "",
        )
    if isinstance(evaluator, StateChangeEvaluator):
        before_found, before = _resolve_json_pointer(subject.initial_state, evaluator.json_pointer)
        after_found, after = _resolve_json_pointer(subject.final_state, evaluator.json_pointer)
        if evaluator.operator == "changed":
            passed = before_found and after_found and before != after
        elif evaluator.operator == "unchanged":
            passed = before_found and after_found and before == after
        else:
            passed = after_found and after == evaluator.expected
        return _deterministic_result(
            evaluator,
            passed,
            "Final state satisfied the assertion."
            if passed
            else "Final state did not satisfy the assertion.",
            "final_state",
            evaluator.json_pointer,
        )
    if isinstance(evaluator, HttpResultEvaluator):
        if subject.http_result is None:
            passed = False
        else:
            status_passed = (
                evaluator.status_code is None
                or subject.http_result.status_code == evaluator.status_code
            )
            if evaluator.body_json_pointer is None:
                body_passed = True
            else:
                body_found, body_value = _resolve_json_pointer(
                    subject.http_result.body, evaluator.body_json_pointer
                )
                body_passed = body_found and body_value == evaluator.expected_body_value
            passed = status_passed and body_passed
        return _deterministic_result(
            evaluator,
            passed,
            "HTTP result satisfied the assertion."
            if passed
            else "HTTP result did not satisfy the assertion.",
            "http_result",
            evaluator.body_json_pointer or "/status_code",
        )
    if isinstance(evaluator, CallableEvaluator):
        evaluator_callable = callables[evaluator.callable_id]
        decision_or_awaitable = evaluator_callable(subject)
        decision = (
            await decision_or_awaitable
            if inspect.isawaitable(decision_or_awaitable)
            else decision_or_awaitable
        )
        return _decision_result(evaluator, decision)
    if isinstance(evaluator, HumanReviewEvaluator):
        return EvaluatorResult(
            evaluator_id=evaluator.id,
            evaluator_type=evaluator.type,
            status="needs_review",
            label="human_review",
            explanation=evaluator.instructions,
        )
    if judge is None:
        raise ValueError("a configured judge is required")
    if isinstance(evaluator, PairwiseEvaluator) and subject.reference_answer is None:
        raise ValueError("pairwise evaluation requires a reference answer")
    request = _judge_request(subject, evaluator)
    decision = await judge.evaluate(request)
    if isinstance(evaluator, RubricEvaluator):
        if decision.score is None:
            raise ValueError("rubric judges must return a score")
        passed = decision.score >= evaluator.minimum_score
        decision = decision.model_copy(update={"passed": passed})
    elif decision.label not in ({"candidate", "reference", "tie"}):
        raise ValueError("pairwise judges must return candidate, reference, or tie")
    elif decision.label == "tie":
        decision = decision.model_copy(update={"passed": evaluator.allow_tie})
    elif decision.passed is None:
        decision = decision.model_copy(update={"passed": decision.label == "candidate"})
    return _decision_result(evaluator, decision)


def _judge_request(
    subject: EvaluationSubject,
    evaluator: RubricEvaluator | PairwiseEvaluator,
) -> JudgeRequest:
    source_values: dict[str, JsonValue] = {
        "answer": subject.answer,
        "tool_calls": cast(
            JsonValue, [call.model_dump(mode="json") for call in subject.tool_calls]
        ),
        "initial_state": subject.initial_state,
        "final_state": subject.final_state,
        "http_result": (
            cast(JsonValue, subject.http_result.model_dump(mode="json"))
            if subject.http_result is not None
            else None
        ),
    }
    payload: dict[str, JsonValue] = {
        source: deepcopy(source_values[source]) for source in evaluator.include_sources
    }
    payload["public_context"] = cast(JsonValue, subject.public_context)
    if isinstance(evaluator, PairwiseEvaluator):
        payload["reference_answer"] = subject.reference_answer
    if evaluator.allow_private_data:
        payload["private_data"] = cast(JsonValue, subject.private_data)
    else:
        for pointer in subject.private_json_pointers:
            _remove_json_pointer(payload, pointer)
    return JudgeRequest(
        evaluator_id=evaluator.id,
        mode="pairwise" if isinstance(evaluator, PairwiseEvaluator) else "rubric",
        rubric=evaluator.rubric,
        payload=payload,
        allow_tie=isinstance(evaluator, PairwiseEvaluator) and evaluator.allow_tie,
    )


def _decision_result(
    evaluator: EvaluatorSpec,
    decision: EvaluatorDecision,
) -> EvaluatorResult:
    if decision.passed is None:
        raise ValueError("evaluator decision must declare pass or fail")
    return EvaluatorResult(
        evaluator_id=evaluator.id,
        evaluator_type=evaluator.type,
        status="passed" if decision.passed else "failed",
        score=decision.score,
        label=decision.label,
        explanation=decision.explanation,
        evidence=decision.evidence,
    )


def _deterministic_result(
    evaluator: EvaluatorSpec,
    passed: bool,
    explanation: str,
    source: Literal[
        "answer", "tool_call", "initial_state", "final_state", "http_result", "callable", "judge"
    ],
    pointer: str,
) -> EvaluatorResult:
    return EvaluatorResult(
        evaluator_id=evaluator.id,
        evaluator_type=evaluator.type,
        status="passed" if passed else "failed",
        score=1 if passed else 0,
        explanation=explanation,
        evidence=(
            EvaluatorEvidence(
                source=source,
                json_pointer=pointer,
                description="Value used by the evaluator.",
            ),
        ),
    )


def _source_value(
    subject: EvaluationSubject,
    source: Literal["answer", "final_state", "http_body"],
) -> JsonValue:
    if source == "answer":
        return subject.answer
    if source == "final_state":
        return subject.final_state
    return subject.http_result.body if subject.http_result is not None else None


def _evidence_source(
    source: Literal["answer", "final_state", "http_body"],
) -> Literal["answer", "final_state", "http_result"]:
    return "http_result" if source == "http_body" else source


def _resolve_json_pointer(value: JsonValue, pointer: str) -> tuple[bool, JsonValue]:
    if pointer == "":
        return True, value
    if not pointer.startswith("/"):
        raise ValueError("JSON pointers must be empty or start with /")
    current = value
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False, None
    return True, cast(JsonValue, current)


def _remove_json_pointer(value: dict[str, JsonValue], pointer: str) -> None:
    if not pointer.startswith("/") or pointer == "/":
        raise ValueError("private JSON pointers must identify a nested value")
    parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]
    current: JsonValue = value
    for part in parts[:-1]:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return
    final = parts[-1]
    if isinstance(current, dict):
        current.pop(final, None)
    elif isinstance(current, list) and final.isdigit() and int(final) < len(current):
        current[int(final)] = None


def _json_type(value: JsonValue) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"
