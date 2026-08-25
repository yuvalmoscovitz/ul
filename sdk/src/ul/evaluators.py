from __future__ import annotations

import asyncio
import hashlib
import inspect
import ipaddress
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from copy import deepcopy
from types import TracebackType
from typing import Any, Literal, Protocol, Self, cast
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, SecretStr, model_validator
from ul_core.contracts import EnvironmentExecutor
from ul_core.evaluation import EvaluationCase, ExecutionEvidence
from ul_core.evaluators import (
    CallableEvaluator,
    EvaluationResults,
    EvaluationSubject,
    EvaluatorCalibrationExample,
    EvaluatorCalibrationExampleResult,
    EvaluatorCalibrationReport,
    EvaluatorDecision,
    EvaluatorEvidence,
    EvaluatorJudgeVersion,
    EvaluatorReliability,
    EvaluatorResult,
    EvaluatorSpec,
    EvaluatorVersion,
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

from ul.environment import validate_execution_evidence

_PROMPTS = PromptManager.instance()

EvaluatorCallable = Callable[[EvaluationSubject], EvaluatorDecision | Awaitable[EvaluatorDecision]]
EvaluationSubjectBuilder = Callable[
    [ExecutionEvidence], EvaluationSubject | Awaitable[EvaluationSubject]
]


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


class EvaluationCaseResult(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    case_id: str
    execution_evidence: ExecutionEvidence | None = None
    evaluation_results: EvaluationResults


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
    max_response_bytes: int = Field(default=1_000_000, ge=1_024, le=5_000_000)

    @model_validator(mode="after")
    def validate_and_normalize_base_url(self) -> Self:
        object.__setattr__(self, "base_url", _validated_judge_base_url(self.base_url))
        return self

    def evaluator_judge_version(self) -> EvaluatorJudgeVersion:
        prompt_version = _PROMPTS.get_template_info("evaluation.judge").version
        configuration = self.model_dump(
            mode="json",
            exclude={"api_key", "model"},
        )
        return EvaluatorJudgeVersion(
            prompt_version=prompt_version,
            model=self.model,
            configuration_sha256=_sha256_json(configuration),
        )


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
    citations: tuple[str, ...] = Field(min_length=1, max_length=20)


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

    @property
    def version(self) -> EvaluatorJudgeVersion:
        return self.config.evaluator_judge_version()

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
        endpoint = f"{self.config.base_url}/chat/completions"
        async with asyncio.timeout(self.config.timeout_seconds):
            async with self._client.stream(
                "POST",
                endpoint,
                headers={**headers, "Accept-Encoding": "identity"},
                json=request_body,
                timeout=self.config.timeout_seconds,
                follow_redirects=False,
            ) as response:
                if not _same_origin(response.url, httpx.URL(endpoint)):
                    raise ValueError("judge response changed request origin")
                if 300 <= response.status_code < 400:
                    raise ValueError("judge redirects are not allowed")
                response.raise_for_status()
                if response.headers.get("content-encoding", "identity").strip().lower() != (
                    "identity"
                ):
                    raise ValueError("judge response Content-Encoding is not allowed")
                response_body = await _read_response(
                    response,
                    maximum_bytes=self.config.max_response_bytes,
                )
        parsed_response = _JudgeResponse.model_validate_json(response_body)
        parsed_value = parsed_response.model_dump(mode="json")
        if _contains_secret(parsed_value, self.config.base_url):
            raise ValueError("judge response contains the configured endpoint URL")
        if self.config.api_key is not None and _contains_secret(
            parsed_value, self.config.api_key.get_secret_value()
        ):
            raise ValueError("judge response contains the configured credential")
        output = _JudgeOutput.model_validate_json(parsed_response.choices[0].message.content)
        return EvaluatorDecision(
            score=output.score,
            label=output.label,
            explanation=output.explanation,
            evidence=_judge_evidence(request, output.citations),
        )


async def evaluate_case(
    case: EvaluationCase,
    environment: EnvironmentExecutor,
    *,
    judge: EvaluatorJudge | None = None,
    callables: Mapping[str, EvaluatorCallable] | None = None,
    subject_builder: EvaluationSubjectBuilder | None = None,
    calibration_reports: Mapping[str, EvaluatorCalibrationReport] | None = None,
    judge_version: EvaluatorJudgeVersion | None = None,
) -> EvaluationCaseResult:
    environment_api_calls = environment.api_calls_for_case(case)
    if type(environment_api_calls) is not int or not (
        1 <= environment_api_calls <= case.max_environment_api_calls
    ):
        raise ValueError("environment API calls exceed the evaluation case budget")
    try:
        async with asyncio.timeout(case.timeout_seconds):
            execution_evidence = await environment.execute(case)
    except TimeoutError:
        return EvaluationCaseResult(
            case_id=case.id,
            evaluation_results=await evaluate(
                EvaluationSubject(
                    agent_status="timed_out",
                    agent_failure_reason="The agent execution timed out.",
                ),
                case.evaluators,
                judge=judge,
                callables=callables,
                calibration_reports=calibration_reports,
                judge_version=judge_version,
            ),
        )
    except RuntimeError:
        return EvaluationCaseResult(
            case_id=case.id,
            evaluation_results=await evaluate(
                EvaluationSubject(
                    agent_status="failed",
                    agent_failure_reason="The agent execution failed.",
                ),
                case.evaluators,
                judge=judge,
                callables=callables,
                calibration_reports=calibration_reports,
                judge_version=judge_version,
            ),
        )
    validate_execution_evidence(case, environment, execution_evidence)
    if execution_evidence.lifecycle.terminal_status == "succeeded":
        subject_or_awaitable = (
            subject_builder(execution_evidence)
            if subject_builder is not None
            else _subject_from_execution_evidence(execution_evidence)
        )
        subject = (
            await subject_or_awaitable
            if inspect.isawaitable(subject_or_awaitable)
            else subject_or_awaitable
        )
        if subject.agent_status != "succeeded":
            raise ValueError("successful execution evidence requires a successful subject")
    else:
        subject = EvaluationSubject(
            agent_status=(
                "timed_out"
                if execution_evidence.lifecycle.terminal_status == "timed_out"
                else "failed"
            ),
            agent_failure_reason=(
                f"The agent execution {execution_evidence.lifecycle.terminal_status}."
            ),
        )
    return EvaluationCaseResult(
        case_id=case.id,
        execution_evidence=execution_evidence,
        evaluation_results=await evaluate(
            subject,
            case.evaluators,
            judge=judge,
            callables=callables,
            calibration_reports=calibration_reports,
            judge_version=judge_version,
        ),
    )


def _subject_from_execution_evidence(evidence: ExecutionEvidence) -> EvaluationSubject:
    return EvaluationSubject(
        agent_status="succeeded",
        answer=(
            evidence.public_normalized_result
            if evidence.public_normalized_result is not None
            else evidence.final_response
        ),
        initial_state=evidence.initial_state.value if evidence.initial_state is not None else None,
        final_state=evidence.final_state.value if evidence.final_state is not None else None,
    )


async def evaluate(
    subject: EvaluationSubject,
    evaluators: tuple[EvaluatorSpec, ...],
    *,
    judge: EvaluatorJudge | None = None,
    callables: Mapping[str, EvaluatorCallable] | None = None,
    calibration_reports: Mapping[str, EvaluatorCalibrationReport] | None = None,
    judge_version: EvaluatorJudgeVersion | None = None,
) -> EvaluationResults:
    resolved_judge_version = judge_version or _judge_version(judge)
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
    reliability = tuple(
        _evaluator_reliability(
            evaluator,
            judge_version=resolved_judge_version,
            calibration_report=(calibration_reports or {}).get(evaluator.id),
        )
        for evaluator in evaluators
    )
    return EvaluationResults(results=tuple(results), reliability=reliability)


def create_evaluator_version(
    evaluator: EvaluatorSpec,
    *,
    judge_version: EvaluatorJudgeVersion | None = None,
) -> EvaluatorVersion:
    evaluator_payload = evaluator.model_dump(mode="json")
    evaluator_sha256 = _sha256_json(evaluator_payload)
    version_payload: dict[str, JsonValue] = {
        "evaluator": evaluator_payload,
        "judge": judge_version.model_dump(mode="json") if judge_version is not None else None,
    }
    return EvaluatorVersion(
        id=f"ulev_v1_{_sha256_json(version_payload)}",
        evaluator_id=evaluator.id,
        evaluator_type=evaluator.type,
        evaluator_sha256=evaluator_sha256,
        judge=judge_version,
    )


async def calibrate_evaluator(
    evaluator: EvaluatorSpec,
    examples: tuple[EvaluatorCalibrationExample, ...],
    *,
    judge: EvaluatorJudge | None = None,
    callables: Mapping[str, EvaluatorCallable] | None = None,
    judge_version: EvaluatorJudgeVersion | None = None,
) -> EvaluatorCalibrationReport:
    example_ids = tuple(example.id for example in examples)
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("calibration example identifiers must be unique")
    example_kinds = {example.kind for example in examples}
    required_kinds = {"known_good", "known_bad", "borderline"}
    if not required_kinds <= example_kinds:
        raise ValueError("calibration requires known-good, known-bad, and borderline examples")
    resolved_judge_version = judge_version or _judge_version(judge)
    if (
        isinstance(evaluator, (RubricEvaluator, PairwiseEvaluator))
        and resolved_judge_version is None
    ):
        raise ValueError("judge-backed calibration requires a versioned judge configuration")

    example_results: list[EvaluatorCalibrationExampleResult] = []
    for example in examples:
        judgments: list[EvaluatorResult] = []
        for _ in range(example.repetitions):
            evaluated = await evaluate(
                example.subject,
                (evaluator,),
                judge=judge,
                callables=callables,
                judge_version=resolved_judge_version,
            )
            judgments.append(evaluated.results[0])
        judged_passes = tuple(
            judgment.status == "passed"
            for judgment in judgments
            if judgment.status in {"passed", "failed"}
        )
        false_positive = example.kind == "known_bad" and any(judged_passes)
        false_negative = example.kind == "known_good" and any(
            not passed for passed in judged_passes
        )
        observed_outcomes = {
            (judgment.status, judgment.score, judgment.label) for judgment in judgments
        }
        unstable = example.kind == "borderline" and len(observed_outcomes) > 1
        human_disagreement = len(set(example.human_labels)) > 1
        human_agreement = (
            sum(passed == example.expected_passed for passed in judged_passes) / len(judged_passes)
            if judged_passes
            else None
        )
        example_results.append(
            EvaluatorCalibrationExampleResult(
                example_id=example.id,
                kind=example.kind,
                expected_passed=example.expected_passed,
                human_labels=example.human_labels,
                results=tuple(judgments),
                false_positive=false_positive,
                false_negative=false_negative,
                unstable=unstable,
                human_disagreement=human_disagreement,
                human_agreement=human_agreement,
            )
        )

    false_positive_examples = tuple(
        result.example_id for result in example_results if result.false_positive
    )
    false_negative_examples = tuple(
        result.example_id for result in example_results if result.false_negative
    )
    unstable_examples = tuple(result.example_id for result in example_results if result.unstable)
    human_disagreement_examples = tuple(
        result.example_id for result in example_results if result.human_disagreement
    )
    comparable_results = tuple(
        result.human_agreement for result in example_results if result.human_agreement is not None
    )
    human_agreement = (
        sum(comparable_results) / len(comparable_results) if comparable_results else None
    )
    unreliable = any(
        (
            false_positive_examples,
            false_negative_examples,
            unstable_examples,
            human_disagreement_examples,
            any(
                judgment.status not in {"passed", "failed"}
                for result in example_results
                for judgment in result.results
            ),
        )
    )
    evaluator_version = create_evaluator_version(
        evaluator,
        judge_version=resolved_judge_version,
    )
    report_payload: dict[str, JsonValue] = {
        "evaluator_version": evaluator_version.model_dump(mode="json"),
        "examples": [result.model_dump(mode="json") for result in example_results],
    }
    return EvaluatorCalibrationReport(
        id=f"ulec_v1_{_sha256_json(report_payload)}",
        evaluator_version=evaluator_version,
        status="unreliable" if unreliable else "reliable",
        examples=tuple(example_results),
        false_positive_examples=false_positive_examples,
        false_negative_examples=false_negative_examples,
        unstable_examples=unstable_examples,
        human_disagreement_examples=human_disagreement_examples,
        human_agreement=human_agreement,
    )


def _evaluator_reliability(
    evaluator: EvaluatorSpec,
    *,
    judge_version: EvaluatorJudgeVersion | None,
    calibration_report: EvaluatorCalibrationReport | None,
) -> EvaluatorReliability:
    version = create_evaluator_version(evaluator, judge_version=judge_version)
    if calibration_report is None or calibration_report.evaluator_version.id != version.id:
        return EvaluatorReliability(
            evaluator_id=evaluator.id,
            evaluator_version_id=version.id,
            status="uncalibrated",
        )
    return EvaluatorReliability(
        evaluator_id=evaluator.id,
        evaluator_version_id=version.id,
        status=calibration_report.status,
        calibration_report_id=calibration_report.id,
    )


def _judge_version(judge: EvaluatorJudge | None) -> EvaluatorJudgeVersion | None:
    version = getattr(judge, "version", None)
    return version if isinstance(version, EvaluatorJudgeVersion) else None


def _sha256_json(value: JsonValue) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


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
    _validate_judge_evidence(request, decision.evidence)
    if isinstance(evaluator, RubricEvaluator):
        if decision.score is None:
            raise ValueError("rubric judges must return a score")
        passed = decision.score >= evaluator.minimum_score
        decision = decision.model_copy(update={"passed": passed})
    elif decision.label not in ({"candidate", "reference", "tie"}):
        raise ValueError("pairwise judges must return candidate, reference, or tie")
    elif decision.label == "tie":
        decision = decision.model_copy(update={"passed": evaluator.allow_tie})
    else:
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
        if subject.private_json_pointers:
            raise ValueError("private data opt-in cannot ignore private JSON pointers")
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
        "answer", "tool_call", "initial_state", "final_state", "http_result", "callable"
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
    parts = _json_pointer_parts(pointer)
    current = value
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False, None
    return True, cast(JsonValue, current)


def _remove_json_pointer(value: dict[str, JsonValue], pointer: str) -> None:
    parts = _json_pointer_parts(pointer)
    if not parts:
        raise ValueError("private JSON pointers must identify a nested value")
    current: JsonValue = value
    for part in parts[:-1]:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise ValueError("private JSON pointer does not resolve")
    final = parts[-1]
    if isinstance(current, dict) and final in current:
        current.pop(final)
    elif isinstance(current, list) and final.isdigit() and int(final) < len(current):
        current[int(final)] = None
    else:
        raise ValueError("private JSON pointer does not resolve")


def _json_pointer_parts(pointer: str) -> tuple[str, ...]:
    if pointer == "":
        return ()
    if not pointer.startswith("/"):
        raise ValueError("JSON pointers must be empty or start with /")
    parts: list[str] = []
    for raw_part in pointer[1:].split("/"):
        index = 0
        while index < len(raw_part):
            if raw_part[index] == "~":
                if index + 1 >= len(raw_part) or raw_part[index + 1] not in {"0", "1"}:
                    raise ValueError("JSON pointer contains an invalid escape")
                index += 2
            else:
                index += 1
        parts.append(raw_part.replace("~1", "/").replace("~0", "~"))
    return tuple(parts)


def _judge_evidence(
    request: JudgeRequest,
    citations: tuple[str, ...],
) -> tuple[EvaluatorEvidence, ...]:
    evidence = tuple(
        EvaluatorEvidence(
            source="judge_payload",
            json_pointer=pointer,
            description="Judge-cited value from the submitted payload.",
        )
        for pointer in citations
    )
    _validate_judge_evidence(request, evidence)
    return evidence


def _validate_judge_evidence(
    request: JudgeRequest,
    evidence: tuple[EvaluatorEvidence, ...],
) -> None:
    if not evidence:
        raise ValueError("judge decisions require cited evidence")
    submitted_request = cast(JsonValue, request.model_dump(mode="json"))
    for citation in evidence:
        if citation.source != "judge_payload" or not citation.json_pointer.startswith("/payload/"):
            raise ValueError("judge evidence must cite the submitted payload")
        found, _ = _resolve_json_pointer(submitted_request, citation.json_pointer)
        if not found:
            raise ValueError("judge evidence pointer does not resolve")


async def _read_response(response: httpx.Response, *, maximum_bytes: int) -> bytes:
    chunks: list[bytes] = []
    response_size = 0
    response_chunks = (
        _single_chunk(response.content) if response.is_stream_consumed else response.aiter_raw()
    )
    async for chunk in response_chunks:
        response_size += len(chunk)
        if response_size > maximum_bytes:
            raise ValueError("judge response exceeds max_response_bytes")
        chunks.append(chunk)
    return b"".join(chunks)


async def _single_chunk(content: bytes) -> AsyncIterator[bytes]:
    yield content


def _same_origin(left: httpx.URL, right: httpx.URL) -> bool:
    return (left.scheme, left.host, left.port) == (right.scheme, right.host, right.port)


def _contains_secret(value: object, secret: str) -> bool:
    if isinstance(value, str):
        return secret in value
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        return any(_contains_secret(item, secret) for item in mapping.values())
    if isinstance(value, list):
        sequence = cast(list[object], value)
        return any(_contains_secret(item, secret) for item in sequence)
    return False


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
