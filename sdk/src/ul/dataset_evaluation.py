from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from typing import Literal, Self

from pydantic import ConfigDict, Field, JsonValue, model_validator
from ul_core.contracts import DatasetTargetExecutor, SemanticDeconstructor
from ul_core.dataset import InteractionRecord, ObservedAgentOutput, ObservedOutcome, SemanticFrame
from ul_core.models import ULModel

from ul.dataset_augmentation import (
    DatasetAugmentationCandidate,
    DatasetAugmentationEngine,
    DatasetAugmentationResult,
)

FindingCategory = Literal[
    "duplicate_effect",
    "unexpected_effect",
    "missing_effect",
    "changed_grounded_effect_argument",
]
FindingSeverity = Literal["high", "critical"]
CaseVerdict = Literal[
    "augmentation_rejected",
    "no_divergence",
    "divergence_needs_review",
]


class _StrictULModel(ULModel):
    model_config = ConfigDict(strict=True)


class DatasetEvaluationFinding(_StrictULModel):
    category: FindingCategory
    severity: FindingSeverity
    review_status: Literal["needs_review"] = "needs_review"
    message: str = Field(min_length=1)
    expected_effects: tuple[ObservedOutcome, ...] = ()
    observed_effects: tuple[ObservedOutcome, ...] = ()
    grounded_field_names: tuple[str, ...] = ()


class DatasetEvaluationCase(_StrictULModel):
    candidate: DatasetAugmentationCandidate
    verdict: CaseVerdict
    target_output: ObservedAgentOutput | None = None
    observed_frame: SemanticFrame | None = None
    findings: tuple[DatasetEvaluationFinding, ...] = ()

    @model_validator(mode="after")
    def validate_execution_state(self) -> Self:
        execution_present = self.target_output is not None and self.observed_frame is not None
        if self.candidate.passed != execution_present:
            raise ValueError("only accepted candidates may have an observed execution")
        if not execution_present and self.findings:
            raise ValueError("unexecuted candidates cannot have findings")
        expected_verdict: CaseVerdict
        if not self.candidate.passed:
            expected_verdict = "augmentation_rejected"
        elif self.findings:
            expected_verdict = "divergence_needs_review"
        else:
            expected_verdict = "no_divergence"
        if self.verdict != expected_verdict:
            raise ValueError("case verdict must match augmentation and evaluation results")
        return self


class DatasetEvaluationResult(_StrictULModel):
    source: InteractionRecord
    augmentation: DatasetAugmentationResult
    cases: tuple[DatasetEvaluationCase, ...]

    @model_validator(mode="after")
    def validate_lineage(self) -> Self:
        if len(self.augmentation.source_frames) != 1:
            raise ValueError("dataset evaluation requires exactly one source frame")
        if self.augmentation.source_frames[0].interaction_id != self.source.id:
            raise ValueError("source frame must reference the source interaction")
        if tuple(case.candidate for case in self.cases) != self.augmentation.candidates:
            raise ValueError("evaluation cases must preserve every augmentation candidate")
        return self


class DatasetEvaluationRunner:
    def __init__(
        self,
        augmentation_engine: DatasetAugmentationEngine,
        deconstructor: SemanticDeconstructor,
        target: DatasetTargetExecutor,
    ) -> None:
        self._augmentation_engine = augmentation_engine
        self._deconstructor = deconstructor
        self._target = target

    async def run(
        self,
        source: InteractionRecord,
        *,
        operator_ids: Iterable[str] = ("surface.rephrase",),
    ) -> DatasetEvaluationResult:
        augmentation = await self._augmentation_engine.augment((source,), operator_ids=operator_ids)
        source_frame = augmentation.source_frames[0]
        if not any(outcome.kind == "action" for outcome in source_frame.outcomes):
            raise ValueError("source frame requires at least one observable action outcome")
        cases: list[DatasetEvaluationCase] = []
        for candidate in augmentation.candidates:
            if not candidate.passed:
                cases.append(
                    DatasetEvaluationCase(
                        candidate=candidate,
                        verdict="augmentation_rejected",
                    )
                )
                continue
            target_output = await self._target.execute(candidate.augmented_input)
            candidate_record = InteractionRecord(
                id=f"{source.id}:{candidate.operator_id}",
                raw_input=candidate.augmented_input,
                raw_observed_output=target_output.raw_output,
            )
            observed_frame = await self._deconstructor.deconstruct(candidate_record, source_frame)
            if observed_frame.interaction_id != candidate_record.id:
                raise ValueError("observed frame must reference its candidate interaction")
            findings = _compare_action_outcomes(source_frame, observed_frame)
            cases.append(
                DatasetEvaluationCase(
                    candidate=candidate,
                    verdict=("divergence_needs_review" if findings else "no_divergence"),
                    target_output=target_output,
                    observed_frame=observed_frame,
                    findings=findings,
                )
            )
        return DatasetEvaluationResult(source=source, augmentation=augmentation, cases=tuple(cases))


def _compare_action_outcomes(
    expected_frame: SemanticFrame,
    observed_frame: SemanticFrame,
) -> tuple[DatasetEvaluationFinding, ...]:
    expected_by_key = _action_outcomes_by_key(expected_frame)
    observed_by_key = _action_outcomes_by_key(observed_frame)
    input_factor_values = {
        _json_key(factor.value)
        for factor in expected_frame.factors
        if any(evidence.source == "input" for evidence in factor.evidence)
    }
    findings: list[DatasetEvaluationFinding] = []
    for key in sorted(expected_by_key.keys() | observed_by_key.keys()):
        expected = expected_by_key.get(key, ())
        observed = observed_by_key.get(key, ())
        if not expected:
            findings.append(
                DatasetEvaluationFinding(
                    category="unexpected_effect",
                    severity="critical",
                    message=(
                        f"Needs review: the augmented input produced an unexpected {key[1]} "
                        "action effect."
                    ),
                    observed_effects=observed,
                )
            )
            continue

        unmatched_expected, unmatched_observed = _remove_grounded_matches(
            expected, observed, input_factor_values
        )
        changed_count = min(len(unmatched_expected), len(unmatched_observed))
        for expected_effect, observed_effect in zip(
            unmatched_expected[:changed_count],
            unmatched_observed[:changed_count],
            strict=True,
        ):
            grounded_field_names = _changed_grounded_field_names(
                expected_effect, observed_effect, input_factor_values
            )
            findings.append(
                DatasetEvaluationFinding(
                    category="changed_grounded_effect_argument",
                    severity="critical",
                    message=(
                        f"Needs review: the augmented input changed a grounded argument of the "
                        f"{key[1]} action effect."
                    ),
                    expected_effects=(expected_effect,),
                    observed_effects=(observed_effect,),
                    grounded_field_names=grounded_field_names,
                )
            )

        missing = unmatched_expected[changed_count:]
        if missing:
            findings.append(
                DatasetEvaluationFinding(
                    category="missing_effect",
                    severity="high",
                    message=(
                        f"Needs review: the augmented input produced {len(observed)} {key[1]} "
                        f"action effects instead of {len(expected)}."
                    ),
                    expected_effects=expected,
                    observed_effects=observed,
                )
            )
        duplicate = unmatched_observed[changed_count:]
        if duplicate:
            findings.append(
                DatasetEvaluationFinding(
                    category="duplicate_effect",
                    severity="critical",
                    message=(
                        f"Needs review: the augmented input produced {len(observed)} {key[1]} "
                        f"action effects instead of {len(expected)}."
                    ),
                    expected_effects=expected,
                    observed_effects=observed,
                )
            )
    return tuple(findings)


def _action_outcomes_by_key(
    frame: SemanticFrame,
) -> dict[tuple[str, str], tuple[ObservedOutcome, ...]]:
    grouped: defaultdict[tuple[str, str], list[ObservedOutcome]] = defaultdict(list)
    for outcome in frame.outcomes:
        if outcome.kind == "action":
            grouped[(outcome.kind, outcome.predicate)].append(outcome)
    return {key: tuple(outcomes) for key, outcomes in grouped.items()}


def _remove_grounded_matches(
    expected: tuple[ObservedOutcome, ...],
    observed: tuple[ObservedOutcome, ...],
    input_factor_values: set[str],
) -> tuple[tuple[ObservedOutcome, ...], tuple[ObservedOutcome, ...]]:
    unmatched_observed = list(observed)
    unmatched_expected: list[ObservedOutcome] = []
    for expected_effect in expected:
        grounded_fields = {
            name: value
            for name, value in expected_effect.fields.items()
            if _json_key(value) in input_factor_values
        }
        match_index = next(
            (
                index
                for index, observed_effect in enumerate(unmatched_observed)
                if all(
                    name in observed_effect.fields
                    and _json_key(observed_effect.fields[name]) == _json_key(value)
                    for name, value in grounded_fields.items()
                )
            ),
            None,
        )
        if match_index is None:
            unmatched_expected.append(expected_effect)
        else:
            unmatched_observed.pop(match_index)
    return tuple(unmatched_expected), tuple(unmatched_observed)


def _changed_grounded_field_names(
    expected: ObservedOutcome,
    observed: ObservedOutcome,
    input_factor_values: set[str],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name, value in expected.fields.items()
            if _json_key(value) in input_factor_values
            and (
                name not in observed.fields or _json_key(observed.fields[name]) != _json_key(value)
            )
        )
    )


def _json_key(value: JsonValue) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
