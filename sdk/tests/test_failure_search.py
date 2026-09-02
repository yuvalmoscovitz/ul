from __future__ import annotations

import asyncio
from typing import Literal

import pytest
from pydantic import JsonValue
from ul import (
    FailureSearchBusinessAssessment,
    FailureSearchGenerationRequest,
    FailureSearchSettings,
    HiddenFailureSearch,
    SemanticRendererFailureSearchGenerator,
)
from ul_core.dataset import (
    RenderedUserInput,
    SemanticAllowedSurfaceChange,
    SemanticEquivalenceAssessment,
)

pytestmark = pytest.mark.asyncio


def _meaning(
    verdict: Literal["equivalent", "different", "uncertain"] = "equivalent",
) -> SemanticEquivalenceAssessment:
    return SemanticEquivalenceAssessment(
        verdict=verdict,
        explanation="The requested business action and constraints are unchanged.",
        verifier_version="test",
    )


class MeaningVerifier:
    async def verify(
        self,
        source_input: str,
        candidate_input: str,
        *,
        allowed_surface_change: SemanticAllowedSurfaceChange = "none",
    ) -> SemanticEquivalenceAssessment:
        assert allowed_surface_change == "none"
        return _meaning("uncertain" if "different goal" in candidate_input else "equivalent")


class AdaptiveGenerator:
    def __init__(self, rounds: tuple[tuple[str, ...], ...]) -> None:
        self.rounds = rounds
        self.requests: list[FailureSearchGenerationRequest] = []

    async def generate(
        self, request: FailureSearchGenerationRequest
    ) -> tuple[RenderedUserInput, ...]:
        self.requests.append(request)
        return tuple(
            RenderedUserInput(text=value) for value in self.rounds[request.round_number - 1]
        )


class RecordingTarget:
    def __init__(self) -> None:
        self.inputs: list[str] = []
        self.active = 0
        self.maximum_active = 0

    async def execute(self, input: str) -> JsonValue:
        self.inputs.append(input)
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return {"correct": "break this" not in input, "input": input}


class BusinessEvaluator:
    def evaluate(self, target_outcome: JsonValue) -> FailureSearchBusinessAssessment:
        assert isinstance(target_outcome, dict)
        passed = target_outcome["correct"] is True
        return FailureSearchBusinessAssessment(
            passed=passed,
            summary="expected state preserved" if passed else "required action was omitted",
            evidence={"correct": passed},
        )


async def test_search_uses_feedback_and_returns_first_valid_counterexample() -> None:
    generator = AdaptiveGenerator(
        (
            ("please do the same task", "same task but terser"),
            ("break this while asking for the same task", "another harmless version"),
        )
    )
    target = RecordingTarget()
    search = HiddenFailureSearch(
        generator,
        MeaningVerifier(),
        target,
        BusinessEvaluator(),
        settings=FailureSearchSettings(
            baseline_repetitions=5,
            candidates_per_round=2,
            maximum_rounds=2,
            maximum_candidate_executions=4,
            target_concurrency=3,
        ),
    )

    result = await search.run("perform the task")

    assert result.status == "counterexample_found"
    assert result.counterexample is not None
    assert result.counterexample.input == "break this while asking for the same task"
    assert result.counterexample.meaning_assessment.verdict == "equivalent"
    assert result.counterexample.business_assessment is not None
    assert result.counterexample.business_assessment.passed is False
    assert len(result.baseline_trials) == 5
    assert len(generator.requests) == 2
    assert len(generator.requests[1].prior_results) == 2
    assert target.maximum_active == 3


async def test_search_rejects_an_unstable_baseline_before_generation() -> None:
    generator = AdaptiveGenerator((("unused",),))
    target = RecordingTarget()
    call_count = 0

    class UnstableBusinessEvaluator:
        def evaluate(self, target_outcome: JsonValue) -> FailureSearchBusinessAssessment:
            nonlocal call_count
            call_count += 1
            return FailureSearchBusinessAssessment(
                passed=call_count != 3,
                summary="baseline failed" if call_count == 3 else "baseline passed",
            )

    search = HiddenFailureSearch(
        generator,
        MeaningVerifier(),
        target,
        UnstableBusinessEvaluator(),
        settings=FailureSearchSettings(baseline_repetitions=5),
    )

    result = await search.run("perform the task")

    assert result.status == "baseline_rejected"
    assert result.rounds == ()
    assert result.candidate_executions == 0
    assert generator.requests == []


async def test_search_does_not_execute_meaning_changing_candidates() -> None:
    generator = AdaptiveGenerator((("different goal", "safe rewrite"),))
    target = RecordingTarget()
    search = HiddenFailureSearch(
        generator,
        MeaningVerifier(),
        target,
        BusinessEvaluator(),
        settings=FailureSearchSettings(
            baseline_repetitions=2,
            candidates_per_round=2,
            maximum_rounds=1,
            maximum_candidate_executions=2,
            target_concurrency=2,
        ),
    )

    result = await search.run("perform the task")

    assert result.status == "rounds_exhausted"
    assert result.candidate_executions == 1
    assert "different goal" not in target.inputs
    rejected_candidate = result.rounds[0].candidates[0]
    assert rejected_candidate.meaning_assessment.verdict == "uncertain"
    assert rejected_candidate.business_assessment is None


async def test_semantic_generator_returns_plain_rewrites_and_uses_feedback() -> None:
    instructions: list[str] = []

    class Renderer:
        async def render(
            self,
            raw_input: str,
            instruction: str,
            *,
            allow_temporary_value: bool = False,
        ) -> RenderedUserInput:
            assert raw_input == "perform the task"
            assert allow_temporary_value is False
            instructions.append(instruction)
            return RenderedUserInput(text=f"rewrite {len(instructions)}")

    generator = SemanticRendererFailureSearchGenerator(Renderer(), concurrency=2)
    first_round = await generator.generate(
        FailureSearchGenerationRequest(
            source_input="perform the task",
            round_number=1,
            candidate_count=2,
        )
    )

    assert tuple(candidate.text for candidate in first_round) == ("rewrite 1", "rewrite 2")
    assert all("Output only the rewrite" in instruction for instruction in instructions)
