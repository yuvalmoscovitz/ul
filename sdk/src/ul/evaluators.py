from __future__ import annotations

import asyncio
import hashlib
import inspect
import ipaddress
import json
import math
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from types import TracebackType
from typing import Annotated, Any, Literal, Protocol, Self, cast
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
from ul.llm import (
    LLMClient,
    LLMClientConfig,
    LLMRoleConfig,
)

_PROMPTS = PromptManager.instance()
_MAXIMUM_CALIBRATION_EXAMPLES = 100
_MAXIMUM_CALIBRATION_JUDGE_CALLS = 100
_MAXIMUM_CALIBRATION_TIMEOUT_SECONDS = 300.0

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
    allowed_labels: tuple[str, ...] = ()
    label_scores: dict[str, float] = Field(default_factory=dict)


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
    upstream_provider: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]*$",
    )
    timeout_seconds: float = Field(default=60, gt=0, le=300)
    max_output_tokens: int = Field(default=1_024, ge=64, le=8_192)
    token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_completion_tokens"
    max_response_bytes: int = Field(default=1_000_000, ge=1_024, le=5_000_000)

    @model_validator(mode="after")
    def validate_and_normalize_base_url(self) -> Self:
        object.__setattr__(self, "base_url", _validated_judge_base_url(self.base_url))
        if self.data_policy == "openrouter_zdr" and self.upstream_provider is None:
            raise ValueError("openrouter_zdr requires upstream_provider")
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

    def llm_client_config(self) -> LLMClientConfig:
        provider_type = (
            "openrouter" if self.data_policy == "openrouter_zdr" else "openai-compatible"
        )
        return LLMClientConfig(
            provider_id=provider_type,
            provider_type=provider_type,
            base_url=self.base_url,
            api_key=self.api_key,
            api_key_environment_variable=(
                "OPEN_ROUTER_API_KEY"
                if provider_type == "openrouter"
                else "UL_DATASET_OPENAI_API_KEY"
            ),
            api_key_required=provider_type == "openrouter",
            live_calls=True,
            allow_external_data_processing=self.allow_external_data_processing,
            upstream_provider=self.upstream_provider,
            roles=tuple(
                LLMRoleConfig(
                    role=role,
                    model=self.model,
                    max_output_tokens=self.max_output_tokens,
                    token_parameter=self.token_parameter,
                    reasoning_mode="omitted",
                )
                for role in ("deconstruct", "render", "equivalence", "materiality")
            ),
            timeout_seconds=self.timeout_seconds,
            max_response_bytes=self.max_response_bytes,
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


class _JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    score: float | None = Field(ge=0, le=1)
    label: str | None = Field(min_length=1, max_length=500)
    explanation: str = Field(min_length=1, max_length=5_000)
    citations: tuple[
        Annotated[
            str,
            Field(
                max_length=4_096,
                pattern=r"^/payload/(?:[^~/]|~[01])*(?:/(?:[^~/]|~[01])*)*$",
            ),
        ],
        ...,
    ] = Field(min_length=1, max_length=20)


class OpenAICompatibleEvaluatorJudge:
    def __init__(
        self,
        config: OpenAICompatibleJudgeConfig | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        if (config is None) == (llm_client is None):
            raise ValueError("provide exactly one judge config or shared LLM client")
        if client is not None and llm_client is not None:
            raise ValueError("an HTTP client cannot replace the shared LLM client transport")
        self.config = config
        self._owns_llm_client = llm_client is None
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            assert config is not None
            self.llm_client = LLMClient(config.llm_client_config(), client=client)

    @property
    def version(self) -> EvaluatorJudgeVersion:
        if self.config is not None:
            return self.config.evaluator_judge_version()
        return evaluator_judge_version_from_llm_config(self.llm_client.config)

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
        if self._owns_llm_client:
            await self.llm_client.aclose()

    async def evaluate(self, request: JudgeRequest) -> EvaluatorDecision:
        output_schema = _judge_output_schema(request)
        allowed_labels = _allowed_judge_labels(request)
        completion = await self.llm_client.complete(
            role="materiality",
            seed=0,
            top_p=None,
            schema_name="ul_evaluator_decision",
            schema=output_schema,
            strict_schema=True,
            system_prompt=_PROMPTS.get_prompt("evaluation.judge"),
            user_payload=json.dumps(request.model_dump(mode="json"), sort_keys=True),
        )
        output = _JudgeOutput.model_validate_json(completion.choices[0].message.content)
        if allowed_labels and output.label not in allowed_labels:
            raise ValueError("judge label is outside the structured output contract")
        if request.mode == "pairwise" and output.label is None:
            raise ValueError("pairwise judge must choose a label")
        if request.mode == "pairwise" and output.score is not None:
            raise ValueError("pairwise judge must not return a score")
        if request.label_scores and output.score is not None:
            raise ValueError("label-scored rubric judge must not return a score")
        if request.mode == "rubric" and not request.label_scores and output.score is None:
            raise ValueError("rubric judge must return a score")
        score = (
            request.label_scores[output.label]
            if output.label is not None and request.label_scores
            else output.score
        )
        return EvaluatorDecision(
            score=score,
            label=output.label,
            explanation=output.explanation,
            evidence=_judge_evidence(request, output.citations),
        )


def _judge_output_schema(request: JudgeRequest) -> dict[str, Any]:
    schema = _JudgeOutput.model_json_schema(mode="validation")
    properties = cast(dict[str, dict[str, object]], schema["properties"])
    if request.mode == "pairwise":
        properties["label"] = {
            "enum": list(_allowed_judge_labels(request)),
            "type": "string",
        }
        properties["score"] = {"type": "null"}
    elif request.allowed_labels:
        properties["label"] = {
            "enum": list(request.allowed_labels),
            "type": "string",
        }
        if request.label_scores:
            properties["score"] = {"type": "null"}
        else:
            properties["score"] = {"maximum": 1, "minimum": 0, "type": "number"}
    else:
        properties["score"] = {"maximum": 1, "minimum": 0, "type": "number"}
    return schema


def _allowed_judge_labels(request: JudgeRequest) -> tuple[str, ...]:
    if request.mode == "pairwise":
        return (
            ("candidate", "reference", "tie") if request.allow_tie else ("candidate", "reference")
        )
    return request.allowed_labels


def evaluator_judge_version_from_llm_config(
    config: LLMClientConfig,
) -> EvaluatorJudgeVersion:
    prompt_version = _PROMPTS.get_template_info("evaluation.judge").version
    role_config = config.role_config("materiality")
    return EvaluatorJudgeVersion(
        prompt_version=prompt_version,
        model=role_config.model,
        configuration_sha256=_sha256_json(config.evaluator_judge_configuration("materiality")),
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
    has_judge_backed_evaluator = any(
        isinstance(evaluator, (RubricEvaluator, PairwiseEvaluator)) for evaluator in evaluators
    )
    resolved_judge_version = (
        _resolve_judge_version(judge, judge_version) if has_judge_backed_evaluator else None
    )
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
    resolved_judge_version = (
        judge_version if isinstance(evaluator, (RubricEvaluator, PairwiseEvaluator)) else None
    )
    version_payload: dict[str, JsonValue] = {
        "evaluator": evaluator_payload,
        "judge": (
            resolved_judge_version.model_dump(mode="json")
            if resolved_judge_version is not None
            else None
        ),
    }
    return EvaluatorVersion(
        id=f"ulev_v1_{_sha256_json(version_payload)}",
        evaluator_id=evaluator.id,
        evaluator_type=evaluator.type,
        evaluator_sha256=evaluator_sha256,
        judge=resolved_judge_version,
    )


async def calibrate_evaluator(
    evaluator: EvaluatorSpec,
    examples: tuple[EvaluatorCalibrationExample, ...],
    *,
    judge: EvaluatorJudge | None = None,
    callables: Mapping[str, EvaluatorCallable] | None = None,
    judge_version: EvaluatorJudgeVersion | None = None,
    maximum_judge_calls: int | None = None,
    timeout_seconds: float = _MAXIMUM_CALIBRATION_TIMEOUT_SECONDS,
) -> EvaluatorCalibrationReport:
    if len(examples) > _MAXIMUM_CALIBRATION_EXAMPLES:
        raise ValueError(f"calibration exceeds {_MAXIMUM_CALIBRATION_EXAMPLES} examples")
    example_ids = tuple(example.id for example in examples)
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("calibration example identifiers must be unique")
    example_kinds = {example.kind for example in examples}
    required_kinds = {"known_good", "known_bad", "borderline"}
    if not required_kinds <= example_kinds:
        raise ValueError("calibration requires known-good, known-bad, and borderline examples")
    judge_backed = isinstance(evaluator, (RubricEvaluator, PairwiseEvaluator))
    resolved_judge_version = _resolve_judge_version(judge, judge_version) if judge_backed else None
    if judge_backed and resolved_judge_version is None:
        raise ValueError("judge-backed calibration requires a versioned judge configuration")
    planned_judge_calls = sum(example.repetitions for example in examples) if judge_backed else 0
    if judge_backed:
        if type(maximum_judge_calls) is not int or not (
            1 <= maximum_judge_calls <= _MAXIMUM_CALIBRATION_JUDGE_CALLS
        ):
            raise ValueError(
                "judge-backed calibration requires maximum_judge_calls between "
                f"1 and {_MAXIMUM_CALIBRATION_JUDGE_CALLS}"
            )
        if planned_judge_calls > maximum_judge_calls:
            raise ValueError("calibration exceeds the authorized judge call budget")
    elif maximum_judge_calls is not None:
        raise ValueError("deterministic calibration does not accept a judge call budget")
    if (
        type(timeout_seconds) not in {int, float}
        or not math.isfinite(timeout_seconds)
        or not (0 < timeout_seconds <= _MAXIMUM_CALIBRATION_TIMEOUT_SECONDS)
    ):
        raise ValueError(
            "calibration timeout must be positive and at most "
            f"{_MAXIMUM_CALIBRATION_TIMEOUT_SECONDS:g} seconds"
        )

    async with asyncio.timeout(timeout_seconds):
        example_results = await _run_calibration_examples(
            evaluator,
            examples,
            judge=judge,
            callables=callables,
            judge_version=resolved_judge_version,
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
        examples=example_results,
        false_positive_examples=false_positive_examples,
        false_negative_examples=false_negative_examples,
        unstable_examples=unstable_examples,
        human_disagreement_examples=human_disagreement_examples,
        human_agreement=human_agreement,
    )


async def _run_calibration_examples(
    evaluator: EvaluatorSpec,
    examples: tuple[EvaluatorCalibrationExample, ...],
    *,
    judge: EvaluatorJudge | None,
    callables: Mapping[str, EvaluatorCallable] | None,
    judge_version: EvaluatorJudgeVersion | None,
) -> tuple[EvaluatorCalibrationExampleResult, ...]:
    example_results: list[EvaluatorCalibrationExampleResult] = []
    for example in examples:
        judgments: list[EvaluatorResult] = []
        for _ in range(example.repetitions):
            evaluated = await evaluate(
                example.subject,
                (evaluator,),
                judge=judge,
                callables=callables,
                judge_version=judge_version,
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
        human_agreement = _human_agreement(judged_passes, example.human_labels)
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

    return tuple(example_results)


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


def _resolve_judge_version(
    judge: EvaluatorJudge | None,
    explicit_version: EvaluatorJudgeVersion | None,
) -> EvaluatorJudgeVersion | None:
    exposed_version = _judge_version(judge)
    if (
        exposed_version is not None
        and explicit_version is not None
        and exposed_version != explicit_version
    ):
        raise ValueError("explicit judge version does not match the configured judge")
    return exposed_version or explicit_version


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
        allowed_labels=(
            (
                ("candidate", "reference", "tie")
                if evaluator.allow_tie
                else ("candidate", "reference")
            )
            if isinstance(evaluator, PairwiseEvaluator)
            else evaluator.allowed_labels
        ),
        label_scores=(evaluator.label_scores if isinstance(evaluator, RubricEvaluator) else {}),
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
