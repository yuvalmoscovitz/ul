from __future__ import annotations

import json
from typing import cast

import httpx
import pytest
from pydantic import JsonValue, ValidationError
from ul.evaluators import (
    JudgeRequest,
    OpenAICompatibleEvaluatorJudge,
    OpenAICompatibleJudgeConfig,
    evaluate,
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
from ul_core.models import ToolCall

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
                        source="judge", description="Judge assessment of the answer."
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
        (EvaluatorDecision(score=1, explanation="Allowed private fixture matched."),)
    )

    await evaluate(
        _subject(),
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


async def test_pairwise_and_human_review_are_first_class_evaluators() -> None:
    judge = RecordingJudge(
        (
            EvaluatorDecision(
                label="candidate",
                explanation="The candidate is more direct while preserving the outcome.",
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
                                '"explanation":"The outcome is explicit."}'
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
