from __future__ import annotations

import os

import pytest
from pydantic import SecretStr
from ul.evaluators import (
    OpenAICompatibleEvaluatorJudge,
    OpenAICompatibleJudgeConfig,
    evaluate,
)
from ul_core.evaluators import EvaluationSubject, RubricEvaluator


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("UL_LIVE", "").lower() != "true"
    or not os.environ.get("OPEN_ROUTER_API_KEY")
    or not os.environ.get("UL_DATASET_MODEL"),
    reason="requires explicit live-call opt-in, OpenRouter credentials, and model",
)
async def test_live_rubric_judge_scores_answer() -> None:
    api_key = os.environ["OPEN_ROUTER_API_KEY"]
    model = os.environ["UL_DATASET_MODEL"]
    subject = EvaluationSubject(
        agent_status="succeeded",
        answer="The payment was scheduled for tomorrow and no funds were sent today.",
    )

    async with OpenAICompatibleEvaluatorJudge(
        OpenAICompatibleJudgeConfig(
            base_url="https://openrouter.ai/api/v1",
            model=model,
            api_key=SecretStr(api_key),
            allow_external_data_processing=True,
            data_policy="openrouter_zdr",
        )
    ) as judge:
        results = await evaluate(
            subject,
            (
                RubricEvaluator(
                    id="live-clarity",
                    rubric=(
                        "Score 1 when the answer clearly distinguishes scheduling from an "
                        "already-completed payment, otherwise score 0."
                    ),
                    minimum_score=0.8,
                ),
            ),
            judge=judge,
        )

    assert results.results[0].status == "passed"
    assert results.results[0].score is not None
