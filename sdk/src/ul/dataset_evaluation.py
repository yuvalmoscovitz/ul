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
        if self.candidate.passed and self.target_output is None:
            raise ValueError("accepted candidates require target output")
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
            async with asyncio.timeout(self._target_timeout_seconds):
                target_output = await self._target.execute(candidate.augmented_input)
            candidate_record = InteractionRecord(
                id=f"{source.id}:{candidate.operator_id}",
                raw_input=candidate.augmented_input,
                raw_observed_output=target_output.raw_output,
            )
            try:
                observed_frame = await self._deconstructor.deconstruct(
                    candidate_record,
                    source_frame,
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
                source_frame,
                observed_frame,
                source.raw_input,
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
        return DatasetEvaluationResult(source=source, augmentation=augmentation, cases=tuple(cases))


def _action_outcome_reliability_issues(
    frame: SemanticFrame,
    raw_observed_output: JsonValue,
    source_input: str,
    *,
    reference_frame: SemanticFrame | None = None,
    require_input_grounded_fields: bool = False,
) -> tuple[str, ...]:
    issues: list[str] = []
    input_grounded_values = _input_grounded_action_values(
        reference_frame or frame,
        source_input,
    )
    for outcome in frame.outcomes:
        if outcome.kind != "action":
            continue
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
            if _json_key(value) in input_grounded_values
        }
        if not evidenced_action_objects:
            issues.append(f"action outcome {outcome.id} predicate lacks coherent action evidence")
        elif grounded_fields and not any(
            all(
                name in action_object and _json_key(action_object[name]) == _json_key(value)
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


def _input_grounded_action_values(frame: SemanticFrame, source_input: str) -> set[str]:
    return {
        _json_key(value)
        for outcome in frame.outcomes
        if outcome.kind == "action"
        for value in outcome.fields.values()
        if _value_appears_in_input(value, source_input)
    }


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
) -> tuple[DatasetEvaluationFinding, ...]:
    expected_by_key = _action_outcomes_by_key(expected_frame)
    observed_by_key = _action_outcomes_by_key(observed_frame)
    input_grounded_values = _input_grounded_action_values(expected_frame, source_input)
    findings: list[DatasetEvaluationFinding] = []
    for key in sorted(expected_by_key.keys() | observed_by_key.keys()):
        expected = expected_by_key.get(key, ())
        observed = observed_by_key.get(key, ())
        if not expected:
            findings.append(
                DatasetEvaluationFinding(
                    category="unexpected_effect",
                    message=(
                        f"Needs review: the augmented input produced an unexpected {key[1]} "
                        "action effect."
                    ),
                    observed_effects=observed,
                )
            )
            continue

        unmatched_expected, unmatched_observed = _remove_grounded_matches(
            expected, observed, input_grounded_values
        )
        changed_count = min(len(unmatched_expected), len(unmatched_observed))
        for expected_effect, observed_effect in zip(
            unmatched_expected[:changed_count],
            unmatched_observed[:changed_count],
            strict=True,
        ):
            grounded_field_names = _changed_grounded_field_names(
                expected_effect, observed_effect, input_grounded_values
            )
            findings.append(
                DatasetEvaluationFinding(
                    category="changed_grounded_effect_argument",
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
                    message=(
                        f"Needs review: the augmented input produced {len(observed)} {key[1]} "
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
                _grounded_effect_matches(expected_effect, effect, input_grounded_values)
                for expected_effect in expected
            )
        )
        if duplicate:
            findings.append(
                DatasetEvaluationFinding(
                    category="duplicate_effect",
                    message=(
                        f"Needs review: the augmented input repeated a {key[1]} action effect."
                    ),
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
                        f"Needs review: the augmented input produced an unexpected {key[1]} "
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
    input_grounded_values: set[str],
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
                    input_grounded_values,
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
    input_grounded_values: set[str],
) -> bool:
    expected_grounded_fields = {
        name: value
        for name, value in expected.fields.items()
        if _json_key(value) in input_grounded_values
    }
    return all(
        name in observed.fields and _json_key(observed.fields[name]) == _json_key(value)
        for name, value in expected_grounded_fields.items()
    )


def _changed_grounded_field_names(
    expected: ObservedOutcome,
    observed: ObservedOutcome,
    input_grounded_values: set[str],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name, value in expected.fields.items()
            if _json_key(value) in input_grounded_values
            and (
                name not in observed.fields or _json_key(observed.fields[name]) != _json_key(value)
            )
        )
    )


def _json_key(value: JsonValue) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
