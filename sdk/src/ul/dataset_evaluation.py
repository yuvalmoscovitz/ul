from __future__ import annotations

import asyncio
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from decimal import Decimal, InvalidOperation
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
_NUMBER_PATTERN = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?"
CaseVerdict = Literal[
    "augmentation_rejected",
    "inconclusive",
    "no_divergence",
    "divergence_needs_review",
]
BaselineVerdict = Literal["inconclusive", "no_divergence", "divergence_needs_review"]


class _StrictULModel(ULModel):
    model_config = ConfigDict(strict=True)


class DatasetEvaluationFinding(_StrictULModel):
    category: FindingCategory
    severity: Literal["unrated"] = "unrated"
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
    inconclusive_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_execution_state(self) -> Self:
        if self.observed_frame is not None and self.target_output is None:
            raise ValueError("an observed frame requires target output")
        if not self.candidate.passed and (
            self.target_output is not None
            or self.observed_frame is not None
            or self.findings
            or self.inconclusive_reasons
        ):
            raise ValueError("rejected candidates cannot have evaluation results")
        if self.candidate.passed and self.target_output is None and not self.inconclusive_reasons:
            raise ValueError("accepted candidates require target output or an inconclusive reason")
        if self.candidate.passed and self.observed_frame is None and not self.inconclusive_reasons:
            raise ValueError("missing observed frames require an inconclusive reason")
        if self.observed_frame is None and self.findings:
            raise ValueError("findings require an observed frame")
        if self.findings and self.inconclusive_reasons:
            raise ValueError("inconclusive cases cannot have findings")
        expected_verdict: CaseVerdict
        if not self.candidate.passed:
            expected_verdict = "augmentation_rejected"
        elif self.inconclusive_reasons:
            expected_verdict = "inconclusive"
        elif self.findings:
            expected_verdict = "divergence_needs_review"
        else:
            expected_verdict = "no_divergence"
        if self.verdict != expected_verdict:
            raise ValueError("case verdict must match augmentation and evaluation results")
        return self


class DatasetEvaluationBaseline(_StrictULModel):
    verdict: BaselineVerdict
    target_output: ObservedAgentOutput | None = None
    observed_frame: SemanticFrame | None = None
    findings: tuple[DatasetEvaluationFinding, ...] = ()
    inconclusive_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_execution_state(self) -> Self:
        if self.observed_frame is not None and self.target_output is None:
            raise ValueError("a baseline observed frame requires target output")
        if self.observed_frame is None and self.findings:
            raise ValueError("baseline findings require an observed frame")
        if self.findings and self.inconclusive_reasons:
            raise ValueError("an inconclusive baseline cannot have findings")
        if self.target_output is None and not self.inconclusive_reasons:
            raise ValueError("a baseline without target output requires an inconclusive reason")
        if (
            self.observed_frame is None
            and self.target_output is not None
            and not self.inconclusive_reasons
        ):
            raise ValueError("a missing baseline frame requires an inconclusive reason")
        expected_verdict: BaselineVerdict
        if self.inconclusive_reasons:
            expected_verdict = "inconclusive"
        elif self.findings:
            expected_verdict = "divergence_needs_review"
        else:
            expected_verdict = "no_divergence"
        if self.verdict != expected_verdict:
            raise ValueError("baseline verdict must match its evaluation results")
        return self


class DatasetEvaluationResult(_StrictULModel):
    source: InteractionRecord
    augmentation: DatasetAugmentationResult
    baseline: DatasetEvaluationBaseline
    cases: tuple[DatasetEvaluationCase, ...]

    @model_validator(mode="after")
    def validate_lineage(self) -> Self:
        if len(self.augmentation.source_frames) != 1:
            raise ValueError("dataset evaluation requires exactly one source frame")
        if self.augmentation.source_frames[0].interaction_id != self.source.id:
            raise ValueError("source frame must reference the source interaction")
        if tuple(case.candidate for case in self.cases) != self.augmentation.candidates:
            raise ValueError("evaluation cases must preserve every augmentation candidate")
        if (
            self.baseline.observed_frame is not None
            and self.baseline.observed_frame.interaction_id != f"{self.source.id}:current_baseline"
        ):
            raise ValueError("baseline frame must reference the current baseline interaction")
        if self.baseline.verdict == "inconclusive" and any(
            case.candidate.passed
            and (
                case.verdict != "inconclusive"
                or case.target_output is not None
                or case.observed_frame is not None
                or case.findings
            )
            for case in self.cases
        ):
            raise ValueError("accepted candidates cannot be evaluated without a valid baseline")
        return self


class DatasetEvaluationRunner:
    def __init__(
        self,
        augmentation_engine: DatasetAugmentationEngine,
        deconstructor: SemanticDeconstructor,
        target: DatasetTargetExecutor,
        *,
        target_timeout_seconds: float = 30,
        allow_network_egress: bool = False,
    ) -> None:
        if not math.isfinite(target_timeout_seconds) or target_timeout_seconds <= 0:
            raise ValueError("target_timeout_seconds must be positive and finite")
        safety_envelope = target.safety_envelope
        if not safety_envelope.isolated:
            raise ValueError("dataset target must be isolated")
        if safety_envelope.allows_network_egress and not allow_network_egress:
            raise ValueError("dataset target network egress requires explicit opt-in")
        if safety_envelope.allows_business_side_effects:
            raise ValueError("dataset targets must not allow business side effects")
        if not target.fresh_state_per_execution:
            raise ValueError("dataset target must start from fresh state for every execution")
        self._augmentation_engine = augmentation_engine
        self._deconstructor = deconstructor
        self._target = target
        self._target_timeout_seconds = target_timeout_seconds

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
        source_action_issues = _action_outcome_reliability_issues(
            source_frame,
            source.raw_observed_output,
            source.raw_input,
            require_input_grounded_fields=True,
        )
        if source_action_issues:
            raise ValueError(f"source action outcomes are inconclusive: {source_action_issues[0]}")
        baseline = await self._evaluate_baseline(source, source_frame)
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
            if baseline.verdict == "inconclusive":
                cases.append(
                    DatasetEvaluationCase(
                        candidate=candidate,
                        verdict="inconclusive",
                        inconclusive_reasons=(
                            f"current baseline is inconclusive: {baseline.inconclusive_reasons[0]}",
                        ),
                    )
                )
                continue
            baseline_frame = baseline.observed_frame
            if baseline_frame is None:
                raise AssertionError("conclusive baseline requires an observed frame")
            try:
                async with asyncio.timeout(self._target_timeout_seconds):
                    target_output = await self._target.execute(candidate.augmented_input)
            except TimeoutError:
                cases.append(
                    DatasetEvaluationCase(
                        candidate=candidate,
                        verdict="inconclusive",
                        inconclusive_reasons=("target execution timed out",),
                    )
                )
                continue
            except RuntimeError:
                cases.append(
                    DatasetEvaluationCase(
                        candidate=candidate,
                        verdict="inconclusive",
                        inconclusive_reasons=("target execution failed",),
                    )
                )
                continue
            candidate_record = InteractionRecord(
                id=f"{source.id}:{candidate.operator_id}",
                raw_input=candidate.augmented_input,
                raw_observed_output=target_output.raw_output,
            )
            try:
                observed_frame = await self._deconstructor.deconstruct(
                    candidate_record,
                    baseline_frame,
                )
            except ValueError:
                cases.append(
                    DatasetEvaluationCase(
                        candidate=candidate,
                        verdict="inconclusive",
                        target_output=target_output,
                        inconclusive_reasons=(
                            "target output could not be semantically deconstructed",
                        ),
                    )
                )
                continue
            if observed_frame.interaction_id != candidate_record.id:
                raise ValueError("observed frame must reference its candidate interaction")
            inconclusive_reasons = _action_outcome_reliability_issues(
                observed_frame,
                target_output.raw_output,
                source.raw_input,
                reference_frame=source_frame,
            )
            if inconclusive_reasons:
                cases.append(
                    DatasetEvaluationCase(
                        candidate=candidate,
                        verdict="inconclusive",
                        target_output=target_output,
                        observed_frame=observed_frame,
                        inconclusive_reasons=inconclusive_reasons,
                    )
                )
                continue
            findings = _compare_action_outcomes(
                baseline_frame,
                observed_frame,
                source.raw_input,
                grounding_frame=source_frame,
            )
            cases.append(
                DatasetEvaluationCase(
                    candidate=candidate,
                    verdict=("divergence_needs_review" if findings else "no_divergence"),
                    target_output=target_output,
                    observed_frame=observed_frame,
                    findings=findings,
                )
            )
        return DatasetEvaluationResult(
            source=source,
            augmentation=augmentation,
            baseline=baseline,
            cases=tuple(cases),
        )

    async def _evaluate_baseline(
        self,
        source: InteractionRecord,
        source_frame: SemanticFrame,
    ) -> DatasetEvaluationBaseline:
        try:
            async with asyncio.timeout(self._target_timeout_seconds):
                target_output = await self._target.execute(source.raw_input)
        except TimeoutError:
            return DatasetEvaluationBaseline(
                verdict="inconclusive",
                inconclusive_reasons=("current baseline execution timed out",),
            )
        except RuntimeError:
            return DatasetEvaluationBaseline(
                verdict="inconclusive",
                inconclusive_reasons=("current baseline execution failed",),
            )
        baseline_record = InteractionRecord(
            id=f"{source.id}:current_baseline",
            raw_input=source.raw_input,
            raw_observed_output=target_output.raw_output,
        )
        try:
            observed_frame = await self._deconstructor.deconstruct(
                baseline_record,
                source_frame,
            )
        except ValueError:
            return DatasetEvaluationBaseline(
                verdict="inconclusive",
                target_output=target_output,
                inconclusive_reasons=(
                    "current baseline output could not be semantically deconstructed",
                ),
            )
        if observed_frame.interaction_id != baseline_record.id:
            raise ValueError("baseline frame must reference its current baseline interaction")
        inconclusive_reasons = _action_outcome_reliability_issues(
            observed_frame,
            target_output.raw_output,
            source.raw_input,
            reference_frame=source_frame,
        )
        if inconclusive_reasons:
            return DatasetEvaluationBaseline(
                verdict="inconclusive",
                target_output=target_output,
                observed_frame=observed_frame,
                inconclusive_reasons=inconclusive_reasons,
            )
        findings = _compare_action_outcomes(
            source_frame,
            observed_frame,
            source.raw_input,
            subject="current baseline",
        )
        return DatasetEvaluationBaseline(
            verdict=("divergence_needs_review" if findings else "no_divergence"),
            target_output=target_output,
            observed_frame=observed_frame,
            findings=findings,
        )


def _action_outcome_reliability_issues(
    frame: SemanticFrame,
    raw_observed_output: JsonValue,
    source_input: str,
    *,
    reference_frame: SemanticFrame | None = None,
    require_input_grounded_fields: bool = False,
) -> tuple[str, ...]:
    issues: list[str] = []
    input_grounded_field_names_by_outcome, association_issues = (
        _input_grounded_action_field_names_by_outcome(
            frame,
            source_input,
            reference_frame=reference_frame,
        )
    )
    issues.extend(association_issues)
    for outcome in frame.outcomes:
        if outcome.kind != "action":
            continue
        input_grounded_field_names = input_grounded_field_names_by_outcome.get(outcome.id, set())
        if outcome.status.casefold() != "observed":
            issues.append(f"action outcome {outcome.id} is not affirmatively observed")
        if outcome.confidence < 1:
            issues.append(f"action outcome {outcome.id} has confidence below 1")
        evidenced_action_objects: list[dict[str, JsonValue]] = []
        for evidence in outcome.evidence:
            if evidence.source != "output":
                continue
            try:
                evidence_value = _resolve_output_pointer(
                    raw_observed_output,
                    evidence.json_pointer,
                )
            except ValueError:
                issues.append(f"action outcome {outcome.id} has invalid output evidence")
                continue
            if isinstance(evidence_value, dict):
                action_value = evidence_value.get("action")
                if _json_key(action_value) == _json_key(outcome.predicate):
                    evidenced_action_objects.append(evidence_value)
                    continue
                issues.append(f"action outcome {outcome.id} has non-action object evidence")
                continue
            if isinstance(evidence_value, list):
                issues.append(f"action outcome {outcome.id} has non-primitive output evidence")
                continue
            if _json_key(evidence_value) == _json_key(outcome.predicate):
                try:
                    parent_value = _resolve_output_pointer(
                        raw_observed_output,
                        evidence.json_pointer.rsplit("/", 1)[0],
                    )
                except ValueError:
                    continue
                if isinstance(parent_value, dict):
                    evidenced_action_objects.append(parent_value)
        grounded_fields = {
            name: value
            for name, value in outcome.fields.items()
            if name in input_grounded_field_names
        }
        if not evidenced_action_objects:
            issues.append(f"action outcome {outcome.id} predicate lacks coherent action evidence")
        elif grounded_fields and not any(
            all(
                name in action_object and _observable_values_equal(action_object[name], value)
                for name, value in grounded_fields.items()
            )
            for action_object in evidenced_action_objects
        ):
            issues.append(
                f"action outcome {outcome.id} grounded fields lack one coherent action record: "
                f"{', '.join(sorted(grounded_fields))}"
            )
        if require_input_grounded_fields and not grounded_fields:
            issues.append(f"action outcome {outcome.id} has no input-grounded fields")
    return tuple(issues)


def _input_grounded_action_field_names_by_outcome(
    frame: SemanticFrame,
    source_input: str,
    *,
    reference_frame: SemanticFrame | None = None,
) -> tuple[dict[str, set[str]], tuple[str, ...]]:
    grounding_frame = reference_frame or frame
    grounded_names_by_reference_id = {
        outcome.id: {
            name
            for name, value in outcome.fields.items()
            if _value_appears_in_input(value, source_input)
        }
        for outcome in grounding_frame.outcomes
        if outcome.kind == "action"
    }
    if reference_frame is None or frame is reference_frame:
        return grounded_names_by_reference_id, ()

    reference_by_predicate = _action_outcomes_by_key(grounding_frame)
    grounded_names_by_outcome: dict[str, set[str]] = {}
    association_issues: list[str] = []
    frame_by_predicate = _action_outcomes_by_key(frame)
    for key, outcomes in frame_by_predicate.items():
        reference_outcomes = reference_by_predicate.get(key, ())
        if not reference_outcomes:
            grounded_names_by_outcome.update(
                (
                    outcome.id,
                    {
                        name
                        for name, value in outcome.fields.items()
                        if _value_appears_in_input(value, source_input)
                    },
                )
                for outcome in outcomes
            )
            continue
        if len(reference_outcomes) == 1:
            grounded_names_by_outcome.update(
                (outcome.id, grounded_names_by_reference_id[reference_outcomes[0].id])
                for outcome in outcomes
            )
            continue
        remaining_outcomes = list(outcomes)
        remaining_references = list(reference_outcomes)
        while remaining_outcomes and remaining_references:
            proposals: list[tuple[ObservedOutcome, ObservedOutcome]] = []
            for outcome in remaining_outcomes:
                scored_references = [
                    (
                        sum(
                            name in outcome.fields
                            and _observable_values_equal(
                                outcome.fields[name], reference.fields[name]
                            )
                            for name in grounded_names_by_reference_id[reference.id]
                        ),
                        reference,
                    )
                    for reference in remaining_references
                ]
                highest_score = max(score for score, _ in scored_references)
                best_references = [
                    reference for score, reference in scored_references if score == highest_score
                ]
                if highest_score > 0 and len(best_references) == 1:
                    proposals.append((outcome, best_references[0]))
            uniquely_proposed_references = {
                reference.id
                for _, reference in proposals
                if sum(candidate.id == reference.id for _, candidate in proposals) == 1
            }
            accepted_proposals = [
                (outcome, reference)
                for outcome, reference in proposals
                if reference.id in uniquely_proposed_references
            ]
            if not accepted_proposals:
                break
            for outcome, reference in accepted_proposals:
                grounded_names_by_outcome[outcome.id] = grounded_names_by_reference_id[reference.id]
            accepted_outcome_ids = {outcome.id for outcome, _ in accepted_proposals}
            accepted_reference_ids = {reference.id for _, reference in accepted_proposals}
            remaining_outcomes = [
                outcome for outcome in remaining_outcomes if outcome.id not in accepted_outcome_ids
            ]
            remaining_references = [
                reference
                for reference in remaining_references
                if reference.id not in accepted_reference_ids
            ]
        for outcome in remaining_outcomes:
            association_issues.append(
                f"action outcome {outcome.id} cannot be safely associated with an "
                "input-grounded source action"
            )
    return grounded_names_by_outcome, tuple(association_issues)


def _value_appears_in_input(value: JsonValue, source_input: str) -> bool:
    if value is None or isinstance(value, (dict, list)):
        return False
    if isinstance(value, str) and not value.strip():
        return False
    numeric_text: str | None = None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric_text = str(value)
    elif (
        isinstance(value, str)
        and re.fullmatch(_NUMBER_PATTERN, value.strip(), re.IGNORECASE)
        and any(marker in value.casefold() for marker in (".", ",", "e"))
    ):
        numeric_text = value.strip()
    if numeric_text is not None:
        try:
            normalized_value = Decimal(numeric_text.replace(",", ""))
        except InvalidOperation:
            return False
        bounded_number_pattern = rf"(?<![\w.]){_NUMBER_PATTERN}(?!\w|\.\d)"
        for match in re.finditer(bounded_number_pattern, source_input, re.IGNORECASE):
            try:
                candidate_value = Decimal(match.group().replace(",", ""))
            except InvalidOperation:
                continue
            if candidate_value == normalized_value:
                return True
        return False
    value_text = str(value).casefold()
    return re.search(rf"(?<!\w){re.escape(value_text)}(?!\w)", source_input.casefold()) is not None


def _resolve_output_pointer(raw_observed_output: JsonValue, pointer: str) -> JsonValue:
    if pointer == "/raw_observed_output":
        return raw_observed_output
    prefix = "/raw_observed_output/"
    if not pointer.startswith(prefix):
        raise ValueError("action evidence must point below raw_observed_output")
    current: object = raw_observed_output
    for encoded_token in pointer[len(prefix) :].split("/"):
        token = encoded_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
            continue
        valid_array_index = token == "0" or (token.isdecimal() and not token.startswith("0"))
        if isinstance(current, list) and valid_array_index and int(token) < len(current):
            current = current[int(token)]
            continue
        raise ValueError("action evidence pointer does not resolve")
    return current


def _compare_action_outcomes(
    expected_frame: SemanticFrame,
    observed_frame: SemanticFrame,
    source_input: str,
    *,
    subject: str = "augmented input",
    grounding_frame: SemanticFrame | None = None,
) -> tuple[DatasetEvaluationFinding, ...]:
    expected_by_key = _action_outcomes_by_key(expected_frame)
    observed_by_key = _action_outcomes_by_key(observed_frame)
    input_grounded_field_names_by_outcome, association_issues = (
        _input_grounded_action_field_names_by_outcome(
            expected_frame,
            source_input,
            reference_frame=grounding_frame,
        )
    )
    if association_issues:
        raise AssertionError("expected action grounding must be unambiguous")
    findings: list[DatasetEvaluationFinding] = []
    for key in sorted(expected_by_key.keys() | observed_by_key.keys()):
        expected = expected_by_key.get(key, ())
        observed = observed_by_key.get(key, ())
        if not expected:
            findings.append(
                DatasetEvaluationFinding(
                    category="unexpected_effect",
                    message=(
                        f"Needs review: the {subject} produced an unexpected {key[1]} "
                        "action effect."
                    ),
                    observed_effects=observed,
                )
            )
            continue

        unmatched_expected, unmatched_observed = _remove_grounded_matches(
            expected,
            observed,
            input_grounded_field_names_by_outcome,
        )
        changed_count = min(len(unmatched_expected), len(unmatched_observed))
        for expected_effect, observed_effect in zip(
            unmatched_expected[:changed_count],
            unmatched_observed[:changed_count],
            strict=True,
        ):
            grounded_field_names = _changed_grounded_field_names(
                expected_effect,
                observed_effect,
                input_grounded_field_names_by_outcome[expected_effect.id],
            )
            findings.append(
                DatasetEvaluationFinding(
                    category="changed_grounded_effect_argument",
                    message=(
                        f"Needs review: the {subject} changed a grounded argument of the "
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
                    message=(
                        f"Needs review: the {subject} produced {len(observed)} {key[1]} "
                        f"action effects instead of {len(expected)}."
                    ),
                    expected_effects=expected,
                    observed_effects=observed,
                )
            )
        remaining_observed = unmatched_observed[changed_count:]
        duplicate = tuple(
            effect
            for effect in remaining_observed
            if any(
                _grounded_effect_matches(
                    expected_effect,
                    effect,
                    input_grounded_field_names_by_outcome[expected_effect.id],
                )
                for expected_effect in expected
            )
        )
        if duplicate:
            findings.append(
                DatasetEvaluationFinding(
                    category="duplicate_effect",
                    message=(f"Needs review: the {subject} repeated a {key[1]} action effect."),
                    expected_effects=expected,
                    observed_effects=duplicate,
                )
            )
        unexpected = tuple(effect for effect in remaining_observed if effect not in duplicate)
        if unexpected:
            findings.append(
                DatasetEvaluationFinding(
                    category="unexpected_effect",
                    message=(
                        f"Needs review: the {subject} produced an unexpected {key[1]} "
                        "action effect."
                    ),
                    expected_effects=expected,
                    observed_effects=unexpected,
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
    input_grounded_field_names_by_outcome: dict[str, set[str]],
) -> tuple[tuple[ObservedOutcome, ...], tuple[ObservedOutcome, ...]]:
    compatible_observed_indexes: list[tuple[int, ...]] = []
    for expected_effect in expected:
        compatible_observed_indexes.append(
            tuple(
                index
                for index, observed_effect in enumerate(observed)
                if _grounded_effect_matches(
                    expected_effect,
                    observed_effect,
                    input_grounded_field_names_by_outcome[expected_effect.id],
                )
            )
        )

    observed_matches: list[int | None] = [None] * len(observed)

    def match(expected_index: int, visited_observed_indexes: set[int]) -> bool:
        for observed_index in compatible_observed_indexes[expected_index]:
            if observed_index in visited_observed_indexes:
                continue
            visited_observed_indexes.add(observed_index)
            previous_expected_index = observed_matches[observed_index]
            if previous_expected_index is None or match(
                previous_expected_index, visited_observed_indexes
            ):
                observed_matches[observed_index] = expected_index
                return True
        return False

    for expected_index in range(len(expected)):
        match(expected_index, set())
    matched_expected_indexes = {
        expected_index for expected_index in observed_matches if expected_index is not None
    }
    return (
        tuple(
            effect for index, effect in enumerate(expected) if index not in matched_expected_indexes
        ),
        tuple(effect for index, effect in enumerate(observed) if observed_matches[index] is None),
    )


def _grounded_effect_matches(
    expected: ObservedOutcome,
    observed: ObservedOutcome,
    input_grounded_field_names: set[str],
) -> bool:
    expected_grounded_fields = {
        name: value for name, value in expected.fields.items() if name in input_grounded_field_names
    }
    return all(
        name in observed.fields and _json_key(observed.fields[name]) == _json_key(value)
        for name, value in expected_grounded_fields.items()
    )


def _changed_grounded_field_names(
    expected: ObservedOutcome,
    observed: ObservedOutcome,
    input_grounded_field_names: set[str],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name, value in expected.fields.items()
            if name in input_grounded_field_names
            and (
                name not in observed.fields or _json_key(observed.fields[name]) != _json_key(value)
            )
        )
    )


def _json_key(value: JsonValue) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _observable_values_equal(left: JsonValue, right: JsonValue) -> bool:
    if _json_key(left) == _json_key(right):
        return True
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    number: int | float
    numeric_text: str
    if isinstance(left, (int, float)) and isinstance(right, str):
        number, numeric_text = left, right
    elif isinstance(right, (int, float)) and isinstance(left, str):
        number, numeric_text = right, left
    else:
        return False
    if (
        re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:e[-+]?\d+)?", numeric_text, re.IGNORECASE)
        is None
    ):
        return False
    try:
        return Decimal(str(number)) == Decimal(numeric_text)
    except InvalidOperation:
        return False
