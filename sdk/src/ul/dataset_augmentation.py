from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from itertools import islice, pairwise
from typing import Any, Literal, Self

from pydantic import Field, JsonValue, model_validator
from ul_core.contracts import (
    SemanticDeconstructor,
    SemanticEquivalenceVerifier,
    SemanticRenderer,
)
from ul_core.dataset import (
    InteractionRecord,
    RenderedUserInput,
    SemanticEquivalenceAssessment,
    SemanticFrame,
    UserInputRecord,
)
from ul_core.models import ULModel

OperatorId = Literal[
    "surface.rephrase",
    "surface.typing_noise",
    "surface.fragmented_syntax",
    "surface.disfluency_repeat",
    "style.terse",
    "style.verbose",
    "tone.frustrated",
]
AllowedChange = Literal["surface_form_only", "declared_communication_form"]


class DatasetAugmentationOperator(ULModel):
    id: OperatorId
    version: Literal["1.0.0"] = "1.0.0"
    instruction: str = Field(min_length=1)
    allowed_change: AllowedChange
    target_communication_kind: str | None = Field(default=None, min_length=1)
    target_marker_required: bool = False
    human_review_required: bool = False

    @model_validator(mode="after")
    def validate_change_contract(self) -> Self:
        requires_target = self.allowed_change == "declared_communication_form"
        if requires_target != (self.target_communication_kind is not None):
            raise ValueError("target communication kind must match the allowed change")
        if self.target_marker_required and self.target_communication_kind is None:
            raise ValueError("required target marker needs a communication kind")
        return self


_BUILTIN_OPERATORS = (
    DatasetAugmentationOperator(
        id="surface.rephrase",
        instruction=(
            "Noticeably rephrase the user input as a natural everyday message; never return the "
            "exact input or a benchmark-polished sentence. Do not change any meaning, request "
            "order, or communication behavior. Keep semantic noun phrases and wording directly "
            "around values intact; change only the request verb or harmless surrounding wording. "
            "Preserve its language, every request, fact, value, constraint, identifier, and "
            "relationship. Do not add context."
        ),
        allowed_change="surface_form_only",
    ),
    DatasetAugmentationOperator(
        id="surface.typing_noise",
        instruction=(
            "Keep the wording almost identical but make exactly one ordinary word contain a "
            "plausible human typo. For example only, 'book 2 seats for AB-12' could become 'book "
            "2 seets for AB-12'. Never copy the example. Do not corrupt names, identifiers, "
            "amounts, dates, addresses, negation, or any other meaning, and do not change request "
            "order or other communication behavior."
        ),
        allowed_change="declared_communication_form",
        target_communication_kind="typing_noise",
    ),
    DatasetAugmentationOperator(
        id="surface.fragmented_syntax",
        instruction=(
            "Rewrite the user input in natural chat shorthand or sentence fragments, like a real "
            "person typing quickly rather than a polished benchmark example. For example only, "
            "'please book the red item for tomorrow' could become 'pls book red item. tomorrow'. "
            "Never copy the example. Keep it unambiguous and preserve every request, fact, value, "
            "constraint, identifier, relationship, and request order. Do not add context."
        ),
        allowed_change="declared_communication_form",
        target_communication_kind="fragmented_syntax",
        target_marker_required=True,
    ),
    DatasetAugmentationOperator(
        id="surface.disfluency_repeat",
        instruction=(
            "Keep the message natural and unpolished, and repeat exactly one ordinary word "
            "immediately as a small hesitation. For example only, 'please book it' could become "
            "'please please book it'. Never copy the example. Do not introduce a correction or "
            "alternative. Preserve every request, fact, value, constraint, identifier, "
            "relationship, and request order. Do not add context."
        ),
        allowed_change="declared_communication_form",
        target_communication_kind="repetition",
    ),
    DatasetAugmentationOperator(
        id="style.terse",
        instruction=(
            "Rewrite the user input as a visibly shorter, terse but natural message from a busy "
            "person, not a polished benchmark sentence. Preserve every request, fact, value, "
            "constraint, identifier, relationship, and request order. Do not make it ambiguous or "
            "add context."
        ),
        allowed_change="declared_communication_form",
        target_communication_kind="terse",
    ),
    DatasetAugmentationOperator(
        id="style.verbose",
        instruction=(
            "Rewrite the user input as a natural, visibly wordier everyday message, roughly one "
            "and a half to two times as many words. Expand only through harmless restatement, "
            "never new facts, motivations, requests, constraints, or context. Preserve every "
            "value, identifier, relationship, and request order, and mention each value and "
            "identifier exactly once. Use only casual filler and pronouns referring back to the "
            "same request; do not add statements about what the user needs or why. For example "
            "only, 'book 2 seats' could become 'hey could you just book 2 seats, just put them in "
            "there for me'. Never copy the example or make the result more formal."
        ),
        allowed_change="declared_communication_form",
        target_communication_kind="verbose",
    ),
    DatasetAugmentationOperator(
        id="tone.frustrated",
        instruction=(
            "Add one short, clearly visible expression of mild natural frustration while keeping "
            "the request otherwise almost identical. For example only, 'add 2 items' could become "
            "'ugh just add 2 items'. Never copy the example. Preserve every request, fact, value, "
            "constraint, identifier, relationship, and request order. Never invent urgency, "
            "authority, prior history, threats, deadlines, consequences, or any other facts. Do "
            "not add insults or abuse."
        ),
        allowed_change="declared_communication_form",
        target_communication_kind="frustrated",
        target_marker_required=True,
        human_review_required=True,
    ),
)
_BUILTIN_OPERATORS_BY_ID = {operator.id: operator for operator in _BUILTIN_OPERATORS}


def builtin_dataset_augmentation_operators() -> tuple[DatasetAugmentationOperator, ...]:
    return _BUILTIN_OPERATORS


class DatasetAugmentationCandidate(ULModel):
    source_interaction_id: str = Field(min_length=1)
    operator_id: OperatorId = "surface.rephrase"
    operator_version: Literal["1.0.0"] = "1.0.0"
    allowed_change: AllowedChange = "surface_form_only"
    human_review_required: bool = False
    augmented_input: str = Field(min_length=1)
    renderer_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    expected_input_frame: SemanticFrame
    reparsed_input_frame: SemanticFrame | None
    semantic_equivalence_assessment: SemanticEquivalenceAssessment | None = None
    passed: bool
    failure_reasons: tuple[str, ...] = ()


class DatasetAugmentationResult(ULModel):
    source_frames: tuple[SemanticFrame, ...]
    candidates: tuple[DatasetAugmentationCandidate, ...]


class DatasetAugmentationEngine:
    maximum_records = 100
    maximum_candidates = 100

    def __init__(
        self,
        deconstructor: SemanticDeconstructor,
        renderer: SemanticRenderer,
        equivalence_verifier: SemanticEquivalenceVerifier | None = None,
    ) -> None:
        self._deconstructor = deconstructor
        self._renderer = renderer
        self._equivalence_verifier = equivalence_verifier

    async def augment(
        self,
        records: Iterable[InteractionRecord],
        *,
        max_records: int = 25,
        operator_ids: Iterable[str] = ("surface.rephrase",),
    ) -> DatasetAugmentationResult:
        selected_operators = _select_operators(operator_ids)
        if not 1 <= max_records <= self.maximum_records:
            raise ValueError(f"max_records must be between 1 and {self.maximum_records}")
        source_records = tuple(islice(records, max_records + 1))
        if len(source_records) > max_records:
            raise ValueError("record count exceeds max_records")
        record_ids = tuple(record.id for record in source_records)
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("interaction record identifiers must be unique")
        candidate_count = len(source_records) * len(selected_operators)
        if candidate_count > self.maximum_candidates:
            raise ValueError(f"candidate count exceeds maximum of {self.maximum_candidates}")
        source_frames: list[SemanticFrame] = []
        candidates: list[DatasetAugmentationCandidate] = []
        for record in source_records:
            source_frame = await self._deconstructor.deconstruct(record)
            if source_frame.interaction_id != record.id:
                raise ValueError("deconstructed frame must reference its source interaction")
            source_frames.append(source_frame)
            expected_input_frame = _input_only_frame(source_frame)
            if (
                not expected_input_frame.request_units
                or not source_frame.outcomes
                or _has_unresolved_nodes(source_frame)
                or _has_ambiguous_nodes(expected_input_frame)
            ):
                continue
            generated_inputs: set[str] = set()
            for operator in selected_operators:
                if operator.id == "surface.typing_noise":
                    rendered_input = _add_typing_noise(record, expected_input_frame, operator)
                elif operator.id == "surface.disfluency_repeat":
                    rendered_input = _add_word_repetition(record, expected_input_frame, operator)
                else:
                    rendered_input = await self._renderer.render(
                        record.raw_input, operator.instruction
                    )
                augmented_input = rendered_input.text
                candidate_record = UserInputRecord(
                    id=f"{record.id}:{operator.id}",
                    raw_input=augmented_input,
                )
                try:
                    reparsed_frame = await self._deconstructor.deconstruct(
                        candidate_record, expected_input_frame
                    )
                except ValueError:
                    reparsed_frame = None
                    equivalence_assessment = None
                    failure_reasons = ["candidate semantic deconstruction failed validation"]
                else:
                    equivalence_assessment = None
                    if reparsed_frame.interaction_id != candidate_record.id:
                        failure_reasons = ["reparsed frame must reference its candidate input"]
                    elif operator.allowed_change == "surface_form_only":
                        failure_reasons = list(
                            _semantic_difference_reasons(expected_input_frame, reparsed_frame)
                        )
                    else:
                        failure_reasons = list(
                            _declared_communication_form_difference_reasons(
                                expected_input_frame,
                                reparsed_frame,
                                operator.target_communication_kind,
                                operator.target_marker_required,
                            )
                        )
                    if _has_unresolved_nodes(reparsed_frame):
                        failure_reasons.append(
                            "reparsed frame contains unresolved semantic elements"
                        )
                    if (
                        self._equivalence_verifier is not None
                        and failure_reasons
                        and all(
                            reason.endswith("differ from the expected frame")
                            for reason in failure_reasons
                        )
                    ):
                        try:
                            equivalence_assessment = await self._equivalence_verifier.verify(
                                record.raw_input,
                                augmented_input,
                            )
                        except ValueError:
                            failure_reasons = ["semantic equivalence validation failed"]
                        else:
                            if equivalence_assessment.verdict == "equivalent":
                                failure_reasons = []
                            elif equivalence_assessment.verdict == "different":
                                failure_reasons = [
                                    "semantic equivalence check found a material change"
                                ]
                            else:
                                failure_reasons = ["semantic equivalence check was uncertain"]
                failure_reasons.extend(
                    _surface_footprint_reasons(operator.id, record.raw_input, augmented_input)
                )
                if augmented_input == record.raw_input:
                    failure_reasons.append("renderer did not change the source input")
                generated_input_key = _text_key(augmented_input)
                if generated_input_key in generated_inputs:
                    failure_reasons.append(
                        "renderer produced an input already generated for this source"
                    )
                generated_inputs.add(generated_input_key)
                candidates.append(
                    DatasetAugmentationCandidate(
                        source_interaction_id=record.id,
                        operator_id=operator.id,
                        operator_version=operator.version,
                        allowed_change=operator.allowed_change,
                        human_review_required=operator.human_review_required,
                        augmented_input=augmented_input,
                        renderer_metadata=rendered_input.metadata,
                        expected_input_frame=expected_input_frame,
                        reparsed_input_frame=reparsed_frame,
                        semantic_equivalence_assessment=equivalence_assessment,
                        passed=not failure_reasons,
                        failure_reasons=tuple(failure_reasons),
                    )
                )
        return DatasetAugmentationResult(
            source_frames=tuple(source_frames), candidates=tuple(candidates)
        )


def _input_only_frame(frame: SemanticFrame) -> SemanticFrame:
    factors = tuple(
        factor.model_copy(update={"evidence": _input_evidence(factor)})
        for factor in frame.factors
        if _has_input_evidence(factor)
    )
    factor_ids = {factor.id for factor in factors}
    request_units = tuple(
        request_unit.model_copy(update={"evidence": _input_evidence(request_unit)})
        for request_unit in frame.request_units
        if _has_input_evidence(request_unit) and set(request_unit.factor_ids) <= factor_ids
    )
    communication_acts = tuple(
        communication_act.model_copy(update={"evidence": _input_evidence(communication_act)})
        for communication_act in frame.communication_acts
        if _has_input_evidence(communication_act)
        and set(communication_act.factor_ids) <= factor_ids
    )
    retained_ids = {
        *factor_ids,
        *(request_unit.id for request_unit in request_units),
        *(communication_act.id for communication_act in communication_acts),
    }
    relations = tuple(
        relation.model_copy(update={"evidence": _input_evidence(relation)})
        for relation in frame.relations
        if _has_input_evidence(relation)
        and set((*relation.source_ids, *relation.target_ids)) <= retained_ids
    )
    return SemanticFrame.model_validate(
        {
            **frame.model_dump(mode="python"),
            "request_units": request_units,
            "factors": factors,
            "relations": relations,
            "communication_acts": communication_acts,
            "outcomes": (),
            "metadata": {},
        }
    )


def _has_input_evidence(element: Any) -> bool:
    return any(evidence.source == "input" for evidence in element.evidence)


def _input_evidence(element: Any) -> tuple[Any, ...]:
    return tuple(evidence for evidence in element.evidence if evidence.source == "input")


def _semantic_difference_reasons(
    expected: SemanticFrame, reparsed: SemanticFrame
) -> tuple[str, ...]:
    expected_semantics = _canonical_semantics(expected)
    reparsed_semantics = _canonical_semantics(reparsed)
    labels = ("factors", "request units", "relations", "communication acts")
    return tuple(
        f"{label} differ from the expected frame"
        for label, expected_part, reparsed_part in zip(
            labels, expected_semantics, reparsed_semantics, strict=True
        )
        if expected_part != reparsed_part
    )


def _declared_communication_form_difference_reasons(
    expected: SemanticFrame,
    reparsed: SemanticFrame,
    target_communication_kind: str | None,
    target_marker_required: bool,
) -> tuple[str, ...]:
    target_acts = tuple(
        act for act in reparsed.communication_acts if act.kind == target_communication_kind
    )
    if not target_acts:
        if target_marker_required:
            return (
                f"reparsed frame does not contain required communication kind "
                f"{target_communication_kind}",
            )
        return _semantic_difference_reasons(expected, reparsed)
    candidate_differences: list[tuple[str, ...]] = []
    for target_act in target_acts:
        if target_act.factor_ids or target_act.attributes:
            candidate_differences.append(
                ("declared communication marker contains unsupported semantics",)
            )
            continue
        if any(
            target_act.id in (*relation.source_ids, *relation.target_ids)
            for relation in reparsed.relations
        ):
            candidate_differences.append(
                ("declared communication marker has unsupported relations",)
            )
            continue
        filtered_frame = reparsed.model_copy(
            update={
                "communication_acts": tuple(
                    act for act in reparsed.communication_acts if act.id != target_act.id
                ),
            }
        )
        differences = _semantic_difference_reasons(expected, filtered_frame)
        if not differences:
            return ()
        candidate_differences.append(differences)
    return min(candidate_differences, key=len)


def _surface_footprint_reasons(
    operator_id: OperatorId,
    source_input: str,
    augmented_input: str,
) -> tuple[str, ...]:
    source_word_count = len(re.findall(r"\w+", source_input, flags=re.UNICODE))
    augmented_words = re.findall(r"\w+", augmented_input, flags=re.UNICODE)
    augmented_word_count = len(augmented_words)
    if operator_id == "surface.rephrase" and _word_key(source_input) == _word_key(augmented_input):
        return ("rendered input only changes case, spacing, or punctuation",)
    if operator_id == "style.terse" and augmented_word_count * 10 > source_word_count * 9:
        return ("rendered input is not visibly shorter than the source",)
    if operator_id == "style.verbose" and not (
        source_word_count * 15 <= augmented_word_count * 10
        and augmented_word_count <= source_word_count * 2
    ):
        return ("rendered input is not between 1.5 and 2 times the source length",)
    if operator_id == "surface.disfluency_repeat":
        source_repetition_count = sum(
            first.casefold() == second.casefold()
            for first, second in pairwise(re.findall(r"\w+", source_input, flags=re.UNICODE))
        )
        augmented_repetition_count = sum(
            first.casefold() == second.casefold() for first, second in pairwise(augmented_words)
        )
        if augmented_repetition_count != source_repetition_count + 1:
            return ("rendered input must contain exactly one immediate word repetition",)
    return ()


def _add_typing_noise(
    record: InteractionRecord,
    frame: SemanticFrame,
    operator: DatasetAugmentationOperator,
) -> RenderedUserInput:
    seed = int.from_bytes(
        hashlib.sha256(f"{record.id}\0{operator.id}\0{operator.version}".encode()).digest()[:4],
        "big",
    )
    protected_words = _protected_factor_words(frame)
    eligible_words: list[tuple[int, int, tuple[int, ...]]] = []
    for match in re.finditer(r"[A-Za-z]+", record.raw_input):
        if len(match.group()) < 4 or match.group().casefold() in protected_words:
            continue
        swap_positions = tuple(
            index
            for index in range(1, len(match.group()) - 1)
            if match.group()[index] != match.group()[index + 1]
        )
        if swap_positions:
            eligible_words.append((match.start(), match.end(), swap_positions))
    if not eligible_words:
        rendered_text = record.raw_input
    else:
        start, end, swap_positions = eligible_words[seed % len(eligible_words)]
        characters = list(record.raw_input[start:end])
        swap_index = swap_positions[seed % len(swap_positions)]
        characters[swap_index], characters[swap_index + 1] = (
            characters[swap_index + 1],
            characters[swap_index],
        )
        rendered_text = f"{record.raw_input[:start]}{''.join(characters)}{record.raw_input[end:]}"
    return RenderedUserInput(
        text=rendered_text,
        metadata={
            "renderer": "deterministic",
            "algorithm": "protected_adjacent_transposition",
            "seed": seed,
        },
    )


def _add_word_repetition(
    record: InteractionRecord,
    frame: SemanticFrame,
    operator: DatasetAugmentationOperator,
) -> RenderedUserInput:
    seed = int.from_bytes(
        hashlib.sha256(f"{record.id}\0{operator.id}\0{operator.version}".encode()).digest()[:4],
        "big",
    )
    protected_words = _protected_factor_words(frame)
    eligible_words = tuple(
        match
        for match in re.finditer(r"[A-Za-z]+", record.raw_input)
        if len(match.group()) >= 3 and match.group().casefold() not in protected_words
    )
    if not eligible_words:
        rendered_text = record.raw_input
    else:
        match = eligible_words[seed % len(eligible_words)]
        repeated_word = match.group()
        if repeated_word[:1].isupper() and repeated_word[1:].islower():
            repeated_word = repeated_word.lower()
        rendered_text = (
            f"{record.raw_input[: match.end()]} {repeated_word}{record.raw_input[match.end() :]}"
        )
    return RenderedUserInput(
        text=rendered_text,
        metadata={
            "renderer": "deterministic",
            "algorithm": "protected_immediate_word_repetition",
            "seed": seed,
        },
    )


def _protected_factor_words(frame: SemanticFrame) -> set[str]:
    protected_words: set[str] = set()
    for factor in frame.factors:
        if not isinstance(factor.value, str):
            continue
        value_words = {word.casefold() for word in re.findall(r"[A-Za-z]+", factor.value)}
        for evidence in factor.evidence:
            if evidence.source == "input" and evidence.text_quote is not None:
                evidence_words = {
                    word.casefold() for word in re.findall(r"[A-Za-z]+", evidence.text_quote)
                }
                protected_words.update(value_words & evidence_words)
    return protected_words


def _has_ambiguous_nodes(frame: SemanticFrame) -> bool:
    factors_by_id = {
        factor.id: (factor.kind, factor.role, _json_key(factor.value)) for factor in frame.factors
    }
    factor_semantics = list(factors_by_id.values())
    request_semantics = [
        (
            request.mode,
            request.predicate,
            tuple(sorted(factors_by_id[factor_id] for factor_id in request.factor_ids)),
        )
        for request in frame.request_units
    ]
    communication_semantics = [
        (
            act.kind,
            tuple(sorted(factors_by_id[factor_id] for factor_id in act.factor_ids)),
            _json_key(act.attributes),
        )
        for act in frame.communication_acts
    ]
    return any(
        len(semantics) != len(set(semantics))
        for semantics in (factor_semantics, request_semantics, communication_semantics)
    )


def _has_unresolved_nodes(frame: SemanticFrame) -> bool:
    return any(
        element.status.casefold().startswith("unresolved")
        for element in (
            *frame.request_units,
            *frame.factors,
            *frame.relations,
            *frame.communication_acts,
            *frame.outcomes,
        )
    )


def _canonical_semantics(frame: SemanticFrame) -> tuple[object, ...]:
    factor_semantics = {
        factor.id: (factor.kind, factor.role, _json_key(factor.value)) for factor in frame.factors
    }
    request_semantics = {
        request.id: (
            request.mode,
            request.predicate,
            tuple(sorted(factor_semantics[factor_id] for factor_id in request.factor_ids)),
        )
        for request in frame.request_units
    }
    communication_semantics = {
        act.id: (
            act.kind,
            tuple(sorted(factor_semantics[factor_id] for factor_id in act.factor_ids)),
            _json_key(act.attributes),
        )
        for act in frame.communication_acts
    }
    endpoint_semantics = {
        **{identifier: ("factor", value) for identifier, value in factor_semantics.items()},
        **{identifier: ("request", value) for identifier, value in request_semantics.items()},
        **{
            identifier: ("communication", value)
            for identifier, value in communication_semantics.items()
        },
    }
    relation_semantics = tuple(
        sorted(
            (
                relation.kind,
                tuple(sorted(endpoint_semantics[item] for item in relation.source_ids)),
                tuple(sorted(endpoint_semantics[item] for item in relation.target_ids)),
            )
            for relation in frame.relations
        )
    )
    return (
        tuple(sorted(factor_semantics.values())),
        tuple(request_semantics[request.id] for request in frame.request_units),
        relation_semantics,
        tuple(communication_semantics[act.id] for act in frame.communication_acts),
    )


def _select_operators(
    operator_ids: Iterable[str],
) -> tuple[DatasetAugmentationOperator, ...]:
    selected_ids = tuple(islice(operator_ids, len(_BUILTIN_OPERATORS) + 1))
    if not selected_ids:
        raise ValueError("operator_ids must contain at least one operator")
    if any(not operator_id for operator_id in selected_ids):
        raise ValueError("operator identifiers must not be empty")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("operator identifiers must be unique")
    if len(selected_ids) > len(_BUILTIN_OPERATORS):
        raise ValueError("operator count exceeds the built-in library")
    unknown_ids = set(selected_ids) - _BUILTIN_OPERATORS_BY_ID.keys()
    if unknown_ids:
        raise ValueError(f"unknown operator identifiers: {sorted(unknown_ids)}")
    return tuple(_BUILTIN_OPERATORS_BY_ID[operator_id] for operator_id in selected_ids)


def _text_key(text: str) -> str:
    return " ".join(text.split()).casefold()


def _word_key(text: str) -> tuple[str, ...]:
    return tuple(word.casefold() for word in re.findall(r"\w+", text, flags=re.UNICODE))


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
