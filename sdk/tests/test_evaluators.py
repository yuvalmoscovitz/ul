from __future__ import annotations

import asyncio
import json
from typing import cast

import httpx
import pytest
from pydantic import JsonValue, ValidationError
from ul.evaluators import (
    EvaluationCaseResult,
    JudgeRequest,
    OpenAICompatibleEvaluatorJudge,
    OpenAICompatibleJudgeConfig,
    evaluate,
    evaluate_case,
)
from ul_core.evaluation import (
    EnvironmentCapabilities,
    EnvironmentLifecycleEvidence,
    EnvironmentTurnEvidence,
    EvaluationCase,
    ExecutionEvidence,
)
from ul_core.evaluators import (
    CallableEvaluator,
    EvaluationSubject,
    EvaluatorDecision,
    EvaluatorEvidence,
    ExactValueEvaluator,
    HttpEvaluationResult,
    HttpResultEvaluator,
    HumanReviewEvaluator,
    JsonPropertyEvaluator,
    PairwiseEvaluator,
    RubricEvaluator,
    StateChangeEvaluator,
    ToolCallEvaluator,
)
from ul_core.models import ConversationRole, ConversationTurn, ToolCall

pytestmark = pytest.mark.asyncio


class RecordingJudge:
    def __init__(self, decisions: tuple[EvaluatorDecision, ...]) -> None:
        self.decisions = list(decisions)
        self.requests: list[JudgeRequest] = []

    async def evaluate(self, request: JudgeRequest) -> EvaluatorDecision:
        self.requests.append(request)
        return self.decisions.pop(0)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://models.example.test/v1",
        "ftp://localhost:8000/v1",
        "https://user:password@models.example.test/v1",
        "https://models.example.test/v1?tenant=secret",
        "https://models.example.test/v1#fragment",
        "https://models.example.test:invalid/v1",
        "https://models.example.test/v1/chat/completions",
    ],
)
async def test_judge_config_rejects_unsafe_base_urls(base_url: str) -> None:
    with pytest.raises(ValidationError):
        OpenAICompatibleJudgeConfig(
            base_url=base_url,
            model="customer/judge",
            allow_external_data_processing=True,
        )


async def test_judge_config_hides_rejected_url_credentials_and_queries() -> None:
    credential_sentinel = "credential-sentinel"
    query_sentinel = "query-sentinel"
    rejected_url = (
        f"https://user:{credential_sentinel}@models.example.test/v1?token={query_sentinel}"
    )

    with pytest.raises(ValidationError) as error:
        OpenAICompatibleJudgeConfig(
            base_url=rejected_url,
            model="customer/judge",
            allow_external_data_processing=True,
        )

    rendered_error = str(error.value)
    assert credential_sentinel not in rendered_error
    assert query_sentinel not in rendered_error
    assert rejected_url not in rendered_error


async def test_judge_config_allows_and_normalizes_loopback_http() -> None:
    config = OpenAICompatibleJudgeConfig(
        base_url="http://[::1]:8000/v1/",
        model="local/judge",
        allow_external_data_processing=True,
    )

    assert config.base_url == "http://[::1]:8000/v1"


def _subject() -> EvaluationSubject:
    return EvaluationSubject(
        agent_status="succeeded",
        answer={"message": "Payment scheduled", "internal_token": "answer-secret"},
        reference_answer="The payment was scheduled.",
        tool_calls=(
            ToolCall(name="schedule_payment", arguments={"invoice_id": "INV-42", "amount": 10}),
        ),
        initial_state={"invoice": {"status": "approved"}},
        final_state={"invoice": {"status": "scheduled"}},
        http_result=HttpEvaluationResult(status_code=202, body={"accepted": True}),
        public_context={"policy": "Only schedule approved invoices."},
        private_data={"fixture_password": "private-fixture-secret"},
        private_json_pointers=("/answer/internal_token",),
    )


async def test_composes_deterministic_evaluators_with_common_results() -> None:
    results = await evaluate(
        _subject(),
        (
            ExactValueEvaluator(
                id="answer", source="answer", json_pointer="/message", expected="Payment scheduled"
            ),
            JsonPropertyEvaluator(
                id="shape",
                source="answer",
                json_pointer="/message",
                operator="type",
                expected_type="string",
            ),
            ToolCallEvaluator(
                id="tool",
                tool_name="schedule_payment",
                arguments={"invoice_id": "INV-42"},
            ),
            StateChangeEvaluator(
                id="state", json_pointer="/invoice/status", operator="equals", expected="scheduled"
            ),
            HttpResultEvaluator(
                id="http", status_code=202, body_json_pointer="/accepted", expected_body_value=True
            ),
        ),
    )

    assert tuple(result.status for result in results.results) == ("passed",) * 5
    assert tuple(result.evaluator_id for result in results.results) == (
        "answer",
        "shape",
        "tool",
        "state",
        "http",
    )
    assert all(result.score == 1 for result in results.results)
    assert all(result.evidence for result in results.results)


async def test_callable_registry_runs_customer_code_without_command_execution() -> None:
    def approved_invoice(subject: EvaluationSubject) -> EvaluatorDecision:
        passed = cast(dict[str, JsonValue], subject.initial_state)["invoice"] == {
            "status": "approved"
        }
        return EvaluatorDecision(
            passed=passed,
            label="approved_invoice",
            explanation="The fixture started from an approved invoice.",
            evidence=(
                EvaluatorEvidence(
                    source="initial_state",
                    json_pointer="/invoice/status",
                    description="Initial invoice status.",
                ),
            ),
        )

    results = await evaluate(
        _subject(),
        (CallableEvaluator(id="business-rule", callable_id="customer.approved_invoice"),),
        callables={"customer.approved_invoice": approved_invoice},
    )

    assert results.results[0].status == "passed"
    assert results.results[0].label == "approved_invoice"


async def test_rubric_judge_defaults_to_answer_and_redacts_declared_private_values() -> None:
    judge = RecordingJudge(
        (
            EvaluatorDecision(
                score=0.9,
                explanation="The answer clearly confirms the requested action.",
                evidence=(
                    EvaluatorEvidence(
                        source="judge_payload",
                        json_pointer="/payload/answer/message",
                        description="Judge assessment of the answer.",
                    ),
                ),
            ),
        )
    )

    results = await evaluate(
        _subject(),
        (
            RubricEvaluator(
                id="clear-answer",
                rubric="Score whether the answer clearly states the outcome.",
                minimum_score=0.8,
            ),
        ),
        judge=judge,
    )

    assert results.results[0].status == "passed"
    assert judge.requests[0].payload == {
        "answer": {"message": "Payment scheduled"},
        "public_context": {"policy": "Only schedule approved invoices."},
    }
    serialized_request = judge.requests[0].model_dump_json()
    assert "answer-secret" not in serialized_request
    assert "private-fixture-secret" not in serialized_request


async def test_private_fixture_data_requires_explicit_judge_opt_in() -> None:
    judge = RecordingJudge(
        (
            EvaluatorDecision(
                score=1,
                explanation="Allowed private fixture matched.",
                evidence=(
                    EvaluatorEvidence(
                        source="judge_payload",
                        json_pointer="/payload/private_data/fixture_password",
                        description="Explicitly allowed fixture value.",
                    ),
                ),
            ),
        )
    )

    await evaluate(
        _subject().model_copy(update={"private_json_pointers": ()}),
        (
            RubricEvaluator(
                id="private-opt-in",
                rubric="Use the explicitly allowed private fixture.",
                allow_private_data=True,
            ),
        ),
        judge=judge,
    )

    assert judge.requests[0].payload["private_data"] == {
        "fixture_password": "private-fixture-secret"
    }


async def test_private_data_opt_in_cannot_silently_ignore_private_pointers() -> None:
    judge = RecordingJudge(())

    results = await evaluate(
        _subject(),
        (
            RubricEvaluator(
                id="conflicting-privacy",
                rubric="Judge the answer.",
                allow_private_data=True,
            ),
        ),
        judge=judge,
    )

    assert results.results[0].status == "evaluator_error"
    assert judge.requests == []


async def test_pairwise_and_human_review_are_first_class_evaluators() -> None:
    judge = RecordingJudge(
        (
            EvaluatorDecision(
                passed=False,
                label="candidate",
                explanation="The candidate is more direct while preserving the outcome.",
                evidence=(
                    EvaluatorEvidence(
                        source="judge_payload",
                        json_pointer="/payload/answer/message",
                        description="Candidate answer wording.",
                    ),
                ),
            ),
        )
    )

    results = await evaluate(
        _subject(),
        (
            PairwiseEvaluator(id="preference", rubric="Prefer the clearer answer."),
            HumanReviewEvaluator(
                id="safety-review", instructions="Confirm the payment can be reversed."
            ),
        ),
        judge=judge,
    )

    assert results.results[0].status == "passed"
    assert results.results[0].label == "candidate"
    assert judge.requests[0].payload["reference_answer"] == "The payment was scheduled."
    assert results.results[1].status == "needs_review"


async def test_agent_failure_is_distinct_from_evaluator_failure_and_does_not_call_judge() -> None:
    judge = RecordingJudge(())
    failed_subject = EvaluationSubject(
        agent_status="failed",
        agent_failure_reason="The target returned a safe execution failure.",
    )

    agent_results = await evaluate(
        failed_subject,
        (RubricEvaluator(id="rubric", rubric="Judge the answer."),),
        judge=judge,
    )
    evaluator_results = await evaluate(
        _subject(),
        (CallableEvaluator(id="missing", callable_id="customer.missing"),),
    )

    assert agent_results.results[0].status == "agent_error"
    assert judge.requests == []
    assert evaluator_results.results[0].status == "evaluator_error"
    assert "customer.missing" not in evaluator_results.results[0].explanation


async def test_openai_compatible_judge_uses_structured_output_and_explicit_data_policy() -> None:
    captured_request: httpx.Request | None = None

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"score":0.95,"label":"clear",'
                                '"explanation":"The outcome is explicit.",'
                                '"citations":["/payload/answer/message"]}'
                            )
                        }
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        judge = OpenAICompatibleEvaluatorJudge(
            OpenAICompatibleJudgeConfig(
                base_url="https://models.example.test/v1",
                model="customer/judge",
                api_key="test-secret",
                allow_external_data_processing=True,
                data_policy="openrouter_zdr",
            ),
            client=client,
        )
        results = await evaluate(
            _subject(),
            (
                RubricEvaluator(
                    id="rubric", rubric="The outcome must be explicit.", minimum_score=0.9
                ),
            ),
            judge=judge,
        )

    assert results.results[0].status == "passed"
    assert captured_request is not None
    request_body = cast(dict[str, object], json.loads(captured_request.content))
    assert cast(dict[str, object], request_body["response_format"])["type"] == "json_schema"
    assert request_body["provider"] == {
        "data_collection": "deny",
        "require_parameters": True,
        "zdr": True,
    }
    assert captured_request.headers["authorization"] == "Bearer test-secret"


@pytest.mark.parametrize(
    ("private_pointer", "include_sources"),
    [
        ("answer/internal_token", ("answer",)),
        ("/answer/internal~2token", ("answer",)),
        ("/answer/internal_tokne", ("answer",)),
        ("/tool_calls/8/name", ("tool_calls",)),
        ("/tool_calls/0/nmae", ("tool_calls",)),
    ],
)
async def test_invalid_private_pointers_fail_before_judge_call(
    private_pointer: str,
    include_sources: tuple[str, ...],
) -> None:
    judge = RecordingJudge(())
    subject = _subject().model_copy(update={"private_json_pointers": (private_pointer,)})
    evaluator = RubricEvaluator.model_validate(
        {
            "id": "privacy",
            "rubric": "Judge the answer.",
            "include_sources": include_sources,
        }
    )

    results = await evaluate(subject, (evaluator,), judge=judge)

    assert results.results[0].status == "evaluator_error"
    assert judge.requests == []


async def test_judge_evidence_must_resolve_against_submitted_payload() -> None:
    judge = RecordingJudge(
        (
            EvaluatorDecision(
                score=1,
                explanation="Unsupported citation.",
                evidence=(
                    EvaluatorEvidence(
                        source="judge_payload",
                        json_pointer="/payload/answer/missing",
                        description="Missing answer value.",
                    ),
                ),
            ),
        )
    )

    results = await evaluate(
        _subject(),
        (RubricEvaluator(id="citation", rubric="Judge the answer."),),
        judge=judge,
    )

    assert results.results[0].status == "evaluator_error"


async def test_openai_compatible_judge_rejects_oversized_streamed_responses() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 1_025)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        judge = OpenAICompatibleEvaluatorJudge(
            OpenAICompatibleJudgeConfig(
                base_url="https://models.example.test/v1",
                model="customer/judge",
                allow_external_data_processing=True,
                max_response_bytes=1_024,
            ),
            client=client,
        )
        with pytest.raises(ValueError, match="max_response_bytes"):
            await judge.evaluate(
                JudgeRequest(
                    evaluator_id="bounded",
                    mode="rubric",
                    rubric="Judge the answer.",
                    payload={"answer": "hello"},
                )
            )


async def test_openai_compatible_judge_rejects_credential_echoes() -> None:
    credential = "provider-credential-sentinel"

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "score": 1,
                                    "label": "clear",
                                    "explanation": credential,
                                    "citations": ["/payload/answer"],
                                }
                            )
                        }
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        judge = OpenAICompatibleEvaluatorJudge(
            OpenAICompatibleJudgeConfig(
                base_url="https://models.example.test/v1",
                model="customer/judge",
                api_key=credential,
                allow_external_data_processing=True,
            ),
            client=client,
        )
        with pytest.raises(ValueError, match="configured credential"):
            await judge.evaluate(
                JudgeRequest(
                    evaluator_id="secret-echo",
                    mode="rubric",
                    rubric="Judge the answer.",
                    payload={"answer": "hello"},
                )
            )


async def test_openai_compatible_judge_has_an_absolute_response_deadline() -> None:
    class SlowResponseStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            while True:
                await asyncio.sleep(0.02)
                yield b" "

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=SlowResponseStream())

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        judge = OpenAICompatibleEvaluatorJudge(
            OpenAICompatibleJudgeConfig(
                base_url="https://models.example.test/v1",
                model="customer/judge",
                allow_external_data_processing=True,
                timeout_seconds=0.01,
            ),
            client=client,
        )
        with pytest.raises(TimeoutError):
            await judge.evaluate(
                JudgeRequest(
                    evaluator_id="deadline",
                    mode="rubric",
                    rubric="Judge the answer.",
                    payload={"answer": "hello"},
                )
            )


class ResponseOnlyEnvironment:
    environment_id = "response-only"
    config_sha256 = "a" * 64
    capabilities = EnvironmentCapabilities(
        supports_conversations=False,
        supports_state_observation=False,
        cancellation_guarantee="guaranteed",
    )

    def api_calls_for_case(self, case: EvaluationCase) -> int:
        return len(case.turns)

    async def execute(self, case: EvaluationCase) -> ExecutionEvidence:
        response = {"status": "scheduled"}
        return ExecutionEvidence(
            evidence_scope="response_only",
            case_id=case.id,
            environment_id=self.environment_id,
            environment_config_sha256=self.config_sha256,
            turns=(EnvironmentTurnEvidence(turn_id=case.turns[0].id, response=response),),
            final_response=response,
            lifecycle=EnvironmentLifecycleEvidence(
                terminal_status="succeeded",
                completed_phases=("execute_turn",),
                delivery="certain",
                cleanup="not_attempted",
                environment_state_uncertain=False,
            ),
        )


async def test_evaluate_case_executes_attached_evaluators_after_the_environment() -> None:
    case = EvaluationCase(
        id="case-with-evaluators",
        turns=(
            ConversationTurn(
                id="turn-1", role=ConversationRole.USER, content="Schedule the payment."
            ),
        ),
        max_environment_api_calls=1,
        timeout_seconds=5,
        evaluators=(
            ExactValueEvaluator(
                id="scheduled",
                source="answer",
                json_pointer="/status",
                expected="scheduled",
            ),
        ),
    )

    result = await evaluate_case(case, ResponseOnlyEnvironment())

    assert isinstance(result, EvaluationCaseResult)
    assert result.execution_evidence is not None
    assert result.evaluation_results.results[0].status == "passed"
