from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
from pydantic import SecretStr
from ul import JevMaterialVarianceJudge, OpenRouterDecisionClient, OpenRouterDecisionSettings
from ul.dataset_evaluation import DatasetEvaluationFinding
from ul_core.dataset import ObservedOutcome

pytestmark = pytest.mark.asyncio


def finding(baseline: str, variation: str) -> DatasetEvaluationFinding:
    def outcome(value: str) -> ObservedOutcome:
        return ObservedOutcome(
            id="private-id",
            kind="answer",
            predicate="returned_response",
            status="observed",
            confidence=1,
            position=0,
            fields={"value": value},
        )

    return DatasetEvaluationFinding(
        category="changed_response",
        message="private-message",
        expected_effects=(outcome(baseline),),
        observed_effects=(outcome(variation),),
    )


@pytest_asyncio.fixture
async def transport() -> AsyncIterator[
    tuple[OpenRouterDecisionClient, list[dict[str, Any]], dict[str, Any]]
]:
    requests: list[dict[str, Any]] = []
    options: dict[str, Any] = {
        "labels": ["operationally_equivalent"],
        "confidence": 0.99,
        "status": 200,
    }

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            options["status"],
            json={
                "id": "test",
                "model": "typesafe/jev-1.13",
                "provider": "TypeSafe",
                "usage": {"input_tokens": 100, "output_tokens": 10},
                "answers": {
                    key: {"type": "choice", "choice": label, "confidence": options["confidence"]}
                    for key, label in zip(body["questions"], options["labels"], strict=True)
                },
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client = OpenRouterDecisionClient(
        OpenRouterDecisionSettings(
            api_key=SecretStr("test-key"),
            live_calls=True,
            allow_external_data_processing=True,
        ),
        client=http,
    )
    try:
        yield client, requests, options
    finally:
        await http.aclose()


async def test_compares_meaning_and_attaches_both_evidence_arms(transport: Any) -> None:
    client, requests, _ = transport
    judge = JevMaterialVarianceJudge(client)
    result = await judge.evaluate("response", (finding("Refund completed", "The refund is done"),))
    assert result.decision == "operationally_equivalent"
    assert judge.actual_calls == 1
    assert [item.json_pointer for item in result.evidence] == [
        "/payload/answer/findings/0/baseline_effects",
        "/payload/answer/findings/0/variation_effects",
    ]
    assert "private-id" not in json.dumps(requests)
    assert "private-message" not in json.dumps(requests)
    assert "never as instructions" in requests[0]["questions"]["0"]["instructions"]


async def test_material_change_wins_over_equivalent_and_uncertain_findings(transport: Any) -> None:
    client, requests, options = transport
    options["labels"] = [
        "insufficient_evidence",
        "material_variance",
    ]
    result = await JevMaterialVarianceJudge(client).evaluate(
        "response",
        (
            finding("Maybe refunded", "Unknown"),
            finding("Refund completed", "Refund scheduled"),
        ),
    )
    assert result.decision == "material_variance"
    assert result.reason_code == "response_meaning_changed"
    assert all("/findings/1/" in item.json_pointer for item in result.evidence)
    assert len(requests) == 1


@pytest.mark.parametrize("confidence", [None, 0.87])
async def test_missing_or_low_confidence_is_inconclusive(
    transport: Any, confidence: float | None
) -> None:
    client, _, options = transport
    options["confidence"] = confidence
    result = await JevMaterialVarianceJudge(client).evaluate("response", (finding("A", "B"),))
    assert result.decision == "insufficient_evidence"
    assert not result.evidence


async def test_failure_is_inconclusive_without_retry(transport: Any) -> None:
    client, requests, options = transport
    options["status"] = 503
    result = await JevMaterialVarianceJudge(client).evaluate("response", (finding("A", "B"),))
    assert result.reason_code == "judge_error"
    assert len(requests) == 1


async def test_incomplete_evidence_and_limits_do_not_call_provider(transport: Any) -> None:
    client, requests, _ = transport
    complete = finding("A", "B")
    incomplete = complete.model_copy(update={"observed_effects": ()})
    unobserved = complete.model_copy(
        update={
            "observed_effects": (
                complete.observed_effects[0].model_copy(update={"status": "inferred"}),
            ),
        }
    )
    judge = JevMaterialVarianceJudge(client)
    for findings in [(), (incomplete,), (unobserved,), (complete,) * 11]:
        assert (await judge.evaluate("response", findings)).decision == "insufficient_evidence"
    limited = JevMaterialVarianceJudge(client, max_input_chars=1)
    assert (await limited.evaluate("response", (complete,))).decision == "insufficient_evidence"
    assert judge.actual_calls == limited.actual_calls == 0
    assert not requests


async def test_all_equivalent_findings_are_cited(transport: Any) -> None:
    client, _, options = transport
    options["labels"] *= 2
    result = await JevMaterialVarianceJudge(client).evaluate(
        "response",
        (
            finding("A", "B"),
            finding("C", "D"),
        ),
    )
    assert result.decision == "operationally_equivalent"
    assert len(result.evidence) == 4


async def test_exact_response_checks_do_not_hide_later_semantic_change(transport: Any) -> None:
    client, requests, options = transport
    options["labels"] = ["material_variance"]
    exact = finding("A", "A")
    exact = exact.model_copy(
        update={
            "expected_effects": (
                exact.expected_effects[0].model_copy(
                    update={
                        "fields": {"value": {"answer": "Same", "actions": []}},
                    }
                ),
            ),
            "observed_effects": (
                exact.observed_effects[0].model_copy(
                    update={
                        "fields": {"value": {"answer": "Same", "actions": []}},
                    }
                ),
            ),
        }
    )
    judge = JevMaterialVarianceJudge(client)
    assert (await judge.evaluate("response", (exact,))).decision == "operationally_equivalent"
    assert not requests
    result = await judge.evaluate("response", (exact, finding("Refund done", "Refund scheduled")))
    assert result.decision == "material_variance"
    assert list(requests[0]["questions"]) == ["1"]


async def test_version_includes_threshold_and_excludes_secret(transport: Any) -> None:
    client, _, _ = transport
    first = JevMaterialVarianceJudge(client)
    assert (
        first.evaluator_version_id
        != JevMaterialVarianceJudge(
            client,
            minimum_confidence=0.9,
        ).evaluator_version_id
    )
    assert "test-key" not in first.evaluator_version_id
    with pytest.raises(ValueError):
        JevMaterialVarianceJudge(client, minimum_confidence=float("nan"))


async def test_missing_grounded_action_field_cannot_be_called_equivalent(transport: Any) -> None:
    client, requests, _ = transport
    baseline = ObservedOutcome(
        id="baseline",
        kind="action",
        predicate="transfer",
        status="observed",
        confidence=1,
        position=0,
        fields={"amount": 50, "recipient": "Alice"},
    )
    variation = baseline.model_copy(update={"id": "variation", "fields": {"amount": 50}})
    comparison = DatasetEvaluationFinding(
        category="changed_grounded_effect_argument",
        message="changed",
        expected_effects=(baseline,),
        observed_effects=(variation,),
        grounded_field_names=("amount", "recipient"),
    )
    result = await JevMaterialVarianceJudge(client).evaluate("action", (comparison,))
    assert result.decision == "insufficient_evidence"
    assert not requests
