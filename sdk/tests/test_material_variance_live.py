from __future__ import annotations

import os

import pytest
from pydantic import JsonValue, SecretStr
from ul.dataset_evaluation import DatasetEvaluationFinding
from ul.evaluators import OpenAICompatibleEvaluatorJudge, OpenAICompatibleJudgeConfig
from ul.material_variance import DatasetMaterialVarianceJudge
from ul_core.dataset import ObservedOutcome, SemanticFrame


def _email(identifier: str, recipient: str, subject: str, body: str) -> dict[str, JsonValue]:
    return {
        "operation": "add",
        "path": f"/messages/id={identifier}",
        "value": {
            "id": identifier,
            "date": 1_700_000_000,
            "to": [recipient],
            "subject": subject,
            "body_plain": body,
        },
    }


def _status(row: int) -> dict[str, JsonValue]:
    return {
        "operation": "replace",
        "path": f"/rows/row_id={row}/Status",
        "before": "Pending",
        "value": "Processing",
    }


def _finding(
    baseline_actions: list[JsonValue], variation_actions: list[JsonValue]
) -> DatasetEvaluationFinding:
    def outcome(identifier: str, actions: list[JsonValue]) -> ObservedOutcome:
        return ObservedOutcome(
            id=identifier,
            kind="answer",
            predicate="returned_response",
            status="observed",
            confidence=1,
            position=0,
            fields={"value": {"actions": actions}},
        )

    return DatasetEvaluationFinding(
        category="changed_response",
        message="changed",
        expected_effects=(outcome("baseline", baseline_actions),),
        observed_effects=(outcome("variation", variation_actions),),
    )


def _source_frame(actions: list[JsonValue]) -> SemanticFrame:
    return SemanticFrame(
        interaction_id="source",
        extractor_version="live-test",
        outcomes=(
            ObservedOutcome(
                id="source-response",
                kind="answer",
                predicate="returned_response",
                status="observed",
                confidence=1,
                position=0,
                fields={"value": {"actions": actions}},
            ),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.live_llm
@pytest.mark.skipif(
    os.environ.get("UL_LIVE", "").lower() != "true"
    or not os.environ.get("OPEN_ROUTER_API_KEY")
    or not os.environ.get("UL_DATASET_MODEL")
    or not os.environ.get("UL_DATASET_OPENROUTER_PROVIDER"),
    reason="requires explicit live-call opt-in, OpenRouter credentials, and model",
)
async def test_live_materiality_distinguishes_business_attributes_from_prose() -> None:
    baseline = [
        _email("baseline-a", "a@example.com", "Payment $100", "Payment is processing."),
        _email("baseline-b", "b@example.com", "Payment $200", "Payment is processing."),
        _email(
            "baseline-c",
            "c@example.com",
            "[PRIORITY] Payment $300",
            "Payment is processing urgently.",
        ),
        _status(2),
        _status(3),
        _status(5),
    ]
    missing_priority = [
        _email("terse-a", "a@example.com", "Payment $100", "Processing payment."),
        _email("terse-b", "b@example.com", "Payment $200", "Processing payment."),
        _email("terse-c", "c@example.com", "Payment $300", "Processing payment."),
        _status(2),
        _status(3),
        _status(5),
    ]
    prose_only = [
        _email(
            "grammar-a",
            "a@example.com",
            "Payment $100",
            "Payment is processing. No further action is required.",
        ),
        _email(
            "grammar-b",
            "b@example.com",
            "Payment $200",
            "Payment is processing. Contact us with questions.",
        ),
        _email(
            "grammar-c",
            "c@example.com",
            "[PRIORITY] Payment $300",
            "Payment is processing urgently. Thank you.",
        ),
        _status(2),
        _status(3),
        _status(5),
    ]

    config = OpenAICompatibleJudgeConfig(
        base_url="https://openrouter.ai/api/v1",
        model=os.environ["UL_DATASET_MODEL"],
        api_key=SecretStr(os.environ["OPEN_ROUTER_API_KEY"]),
        allow_external_data_processing=True,
        data_policy="openrouter_zdr",
        upstream_provider=os.environ["UL_DATASET_OPENROUTER_PROVIDER"],
        token_parameter="max_tokens",
    )
    async with OpenAICompatibleEvaluatorJudge(config) as judge:
        evaluator = DatasetMaterialVarianceJudge(judge, max_input_chars=50_000)
        priority_assessment = await evaluator.evaluate(
            "response",
            (_finding(baseline, missing_priority),),
            source_frame=_source_frame(baseline),
        )
        prose_assessment = await evaluator.evaluate(
            "response",
            (_finding(baseline, prose_only),),
            source_frame=_source_frame(baseline),
        )

    assert priority_assessment.decision == "material_variance"
    assert prose_assessment.decision == "operationally_equivalent"
    assert evaluator.actual_calls == 1
