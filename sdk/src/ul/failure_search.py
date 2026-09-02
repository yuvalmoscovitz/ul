from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Protocol, runtime_checkable

from pydantic import Field, JsonValue, model_validator
from ul_core.contracts import SemanticEquivalenceVerifier, SemanticRenderer
from ul_core.dataset import RenderedUserInput, SemanticEquivalenceAssessment
from ul_core.models import ULModel


class FailureSearchSettings(ULModel):
    baseline_repetitions: int = Field(default=5, ge=1, le=100)
    candidates_per_round: int = Field(default=8, ge=1, le=100)
    maximum_rounds: int = Field(default=3, ge=1, le=100)
    maximum_candidate_executions: int = Field(default=24, ge=1, le=10_000)
    target_concurrency: int = Field(default=1, ge=1, le=100)


class FailureSearchBusinessAssessment(ULModel):
    passed: bool
    summary: str = Field(min_length=1, max_length=1_000)
    evidence: JsonValue = None


class FailureSearchBaselineTrial(ULModel):
    repetition: int = Field(ge=1)
    target_outcome: JsonValue
    business_assessment: FailureSearchBusinessAssessment


class FailureSearchCandidateResult(ULModel):
    id: str = Field(min_length=1)
    round_number: int = Field(ge=1)
    input: str = Field(min_length=1)
    generator_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    meaning_assessment: SemanticEquivalenceAssessment
    target_outcome: JsonValue = None
    business_assessment: FailureSearchBusinessAssessment | None = None

    @model_validator(mode="after")
    def validate_execution(self) -> FailureSearchCandidateResult:
        meaning_preserved = self.meaning_assessment.verdict == "equivalent"
        if meaning_preserved != (self.business_assessment is not None):
            raise ValueError("only meaning-preserving candidates may have a business assessment")
        if not meaning_preserved and self.target_outcome is not None:
            raise ValueError("meaning-changing candidates cannot have a target outcome")
        return self


class FailureSearchRound(ULModel):
    number: int = Field(ge=1)
    candidates: tuple[FailureSearchCandidateResult, ...]


class FailureSearchResult(ULModel):
    status: str = Field(
        pattern=(
            r"^(baseline_rejected|counterexample_found|budget_exhausted|generator_exhausted|"
            r"rounds_exhausted)$"
        )
    )
    source_input: str = Field(min_length=1)
    settings: FailureSearchSettings
    baseline_trials: tuple[FailureSearchBaselineTrial, ...]
    rounds: tuple[FailureSearchRound, ...] = ()
    candidate_executions: int = Field(ge=0)
    counterexample: FailureSearchCandidateResult | None = None

    @model_validator(mode="after")
    def validate_result(self) -> FailureSearchResult:
        baseline_qualified = all(trial.business_assessment.passed for trial in self.baseline_trials)
        if self.status == "baseline_rejected":
            if baseline_qualified or self.rounds or self.counterexample is not None:
                raise ValueError("a rejected baseline cannot produce search rounds")
            return self
        if not baseline_qualified:
            raise ValueError("failure search requires a qualified baseline")
        if self.status == "counterexample_found":
            if self.counterexample is None or self.counterexample.business_assessment is None:
                raise ValueError("counterexample status requires a failed candidate")
            if self.counterexample.business_assessment.passed:
                raise ValueError("counterexample must fail the business check")
        elif self.counterexample is not None:
            raise ValueError("only a successful search can contain a counterexample")
        return self


class FailureSearchGenerationRequest(ULModel):
    source_input: str = Field(min_length=1)
    round_number: int = Field(ge=1)
    candidate_count: int = Field(ge=1, le=100)
    prior_results: tuple[FailureSearchCandidateResult, ...] = ()


@runtime_checkable
class FailureSearchCandidateGenerator(Protocol):
    def generate(
        self, request: FailureSearchGenerationRequest
    ) -> Awaitable[tuple[RenderedUserInput, ...]]: ...


@runtime_checkable
class FailureSearchTarget(Protocol):
    def execute(self, input: str) -> Awaitable[JsonValue]: ...


@runtime_checkable
class FailureSearchBusinessEvaluator(Protocol):
    def evaluate(
        self, target_outcome: JsonValue
    ) -> FailureSearchBusinessAssessment | Awaitable[FailureSearchBusinessAssessment]: ...


class SemanticRendererFailureSearchGenerator:
    def __init__(self, renderer: SemanticRenderer, *, concurrency: int = 4) -> None:
        if type(concurrency) is not int or not 1 <= concurrency <= 100:
            raise ValueError("concurrency must be between 1 and 100")
        self._renderer = renderer
        self._concurrency = concurrency

    async def generate(
        self, request: FailureSearchGenerationRequest
    ) -> tuple[RenderedUserInput, ...]:
        semaphore = asyncio.Semaphore(self._concurrency)
        feedback = _generation_feedback(request.prior_results)

        async def generate_one(position: int) -> RenderedUserInput:
            instruction = (
                "Rewrite like a real person talking to an agent. Keep every action, name, value, "
                "and constraint. Make it clearly different. Output only the rewrite. "
                f"Attempt {position}."
            )
            if feedback:
                instruction = (
                    "Try a new realistic way to say the same request. Keep every action, name, "
                    "value, and constraint. Use the prior results to explore different wording. "
                    f"Output only the rewrite. Attempt {position}. Prior results: {feedback}"
                )
            async with semaphore:
                return await self._renderer.render(request.source_input, instruction)

        return tuple(
            await asyncio.gather(
                *(generate_one(position) for position in range(1, request.candidate_count + 1))
            )
        )


class HiddenFailureSearch:
    def __init__(
        self,
        generator: FailureSearchCandidateGenerator,
        meaning_verifier: SemanticEquivalenceVerifier,
        target: FailureSearchTarget,
        business_evaluator: FailureSearchBusinessEvaluator,
        *,
        settings: FailureSearchSettings | None = None,
    ) -> None:
        self._generator = generator
        self._meaning_verifier = meaning_verifier
        self._target = target
        self._business_evaluator = business_evaluator
        self._settings = settings or FailureSearchSettings()

    async def run(self, source_input: str) -> FailureSearchResult:
        if not source_input.strip():
            raise ValueError("source_input must not be empty")
        semaphore = asyncio.Semaphore(self._settings.target_concurrency)

        async def execute(input: str) -> JsonValue:
            async with semaphore:
                return await self._target.execute(input)

        async def assess(outcome: JsonValue) -> FailureSearchBusinessAssessment:
            assessment = self._business_evaluator.evaluate(outcome)
            if isinstance(assessment, Awaitable):
                return await assessment
            return assessment

        async def run_baseline(repetition: int) -> FailureSearchBaselineTrial:
            outcome = await execute(source_input)
            return FailureSearchBaselineTrial(
                repetition=repetition,
                target_outcome=outcome,
                business_assessment=await assess(outcome),
            )

        baseline_trials = tuple(
            await asyncio.gather(
                *(
                    run_baseline(repetition)
                    for repetition in range(1, self._settings.baseline_repetitions + 1)
                )
            )
        )
        if any(not trial.business_assessment.passed for trial in baseline_trials):
            return FailureSearchResult(
                status="baseline_rejected",
                source_input=source_input,
                settings=self._settings,
                baseline_trials=baseline_trials,
                candidate_executions=0,
            )

        prior_results: list[FailureSearchCandidateResult] = []
        rounds: list[FailureSearchRound] = []
        seen_inputs = {source_input.strip().casefold()}
        candidate_executions = 0
        generator_exhausted = False

        for round_number in range(1, self._settings.maximum_rounds + 1):
            remaining_budget = self._settings.maximum_candidate_executions - candidate_executions
            if remaining_budget == 0:
                break
            requested_count = min(self._settings.candidates_per_round, remaining_budget)
            generated = await self._generator.generate(
                FailureSearchGenerationRequest(
                    source_input=source_input,
                    round_number=round_number,
                    candidate_count=requested_count,
                    prior_results=tuple(prior_results),
                )
            )
            if len(generated) > requested_count:
                raise ValueError("generator returned more candidates than requested")
            novel_candidates: list[RenderedUserInput] = []
            for candidate in generated:
                normalized_input = candidate.text.strip().casefold()
                if normalized_input in seen_inputs:
                    continue
                seen_inputs.add(normalized_input)
                novel_candidates.append(candidate)
            if not novel_candidates:
                generator_exhausted = True
                break

            meaning_assessments = await asyncio.gather(
                *(
                    self._meaning_verifier.verify(source_input, candidate.text)
                    for candidate in novel_candidates
                )
            )

            async def run_candidate(
                position: int,
                candidate: RenderedUserInput,
                meaning_assessment: SemanticEquivalenceAssessment,
                current_round_number: int,
            ) -> FailureSearchCandidateResult:
                nonlocal candidate_executions
                candidate_id = f"round-{current_round_number}-candidate-{position}"
                if meaning_assessment.verdict != "equivalent":
                    return FailureSearchCandidateResult(
                        id=candidate_id,
                        round_number=current_round_number,
                        input=candidate.text,
                        generator_metadata=candidate.metadata,
                        meaning_assessment=meaning_assessment,
                    )
                outcome = await execute(candidate.text)
                candidate_executions += 1
                return FailureSearchCandidateResult(
                    id=candidate_id,
                    round_number=current_round_number,
                    input=candidate.text,
                    generator_metadata=candidate.metadata,
                    meaning_assessment=meaning_assessment,
                    target_outcome=outcome,
                    business_assessment=await assess(outcome),
                )

            round_results = tuple(
                await asyncio.gather(
                    *(
                        run_candidate(position, candidate, meaning_assessment, round_number)
                        for position, (candidate, meaning_assessment) in enumerate(
                            zip(novel_candidates, meaning_assessments, strict=True), start=1
                        )
                    )
                )
            )
            rounds.append(FailureSearchRound(number=round_number, candidates=round_results))
            prior_results.extend(round_results)
            counterexample = next(
                (
                    candidate
                    for candidate in round_results
                    if candidate.business_assessment is not None
                    and not candidate.business_assessment.passed
                ),
                None,
            )
            if counterexample is not None:
                return FailureSearchResult(
                    status="counterexample_found",
                    source_input=source_input,
                    settings=self._settings,
                    baseline_trials=baseline_trials,
                    rounds=tuple(rounds),
                    candidate_executions=candidate_executions,
                    counterexample=counterexample,
                )

        return FailureSearchResult(
            status=(
                "generator_exhausted"
                if generator_exhausted
                else (
                    "budget_exhausted"
                    if candidate_executions >= self._settings.maximum_candidate_executions
                    else "rounds_exhausted"
                )
            ),
            source_input=source_input,
            settings=self._settings,
            baseline_trials=baseline_trials,
            rounds=tuple(rounds),
            candidate_executions=candidate_executions,
        )


def _generation_feedback(results: tuple[FailureSearchCandidateResult, ...]) -> str:
    summaries: list[str] = []
    for result in results[-8:]:
        if result.business_assessment is None:
            outcome = "meaning changed"
        else:
            outcome = result.business_assessment.summary
        summaries.append(f"{result.input[:300]} => {outcome[:300]}")
    return " | ".join(summaries)[:4_000]
