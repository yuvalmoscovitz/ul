from __future__ import annotations

import asyncio
import os

import pytest
from ul import (
    FailureSearchGenerationRequest,
    SemanticRendererFailureSearchGenerator,
    create_semantic_model_deconstructor,
    load_dataset_semantic_settings,
)


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("UL_LIVE", "").lower() != "true"
    or not os.environ.get("OPEN_ROUTER_API_KEY")
    or not os.environ.get("UL_DATASET_MODEL")
    or not os.environ.get("UL_DATASET_OPENROUTER_PROVIDER"),
    reason="requires explicit live-call opt-in, OpenRouter credentials, and model",
)
async def test_live_generator_produces_meaning_preserving_plain_text() -> None:
    source_input = (
        "Update opportunity 006003 (DataStream Analytics License) to $45,000 in Salesforce."
    )
    settings = load_dataset_semantic_settings()
    async with create_semantic_model_deconstructor(settings) as semantic_pipeline:
        candidates = await SemanticRendererFailureSearchGenerator(
            semantic_pipeline, concurrency=2
        ).generate(
            FailureSearchGenerationRequest(
                source_input=source_input,
                round_number=1,
                candidate_count=2,
            )
        )
        assessments = await asyncio.gather(
            *(semantic_pipeline.verify(source_input, candidate.text) for candidate in candidates)
        )

    assert len(candidates) == 2
    assert all(candidate.text != source_input for candidate in candidates)
    assert all(assessment.verdict == "equivalent" for assessment in assessments)
