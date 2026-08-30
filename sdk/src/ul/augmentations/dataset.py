"""Dataset augmentation bindings and generation runtime."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from decimal import Decimal
from itertools import islice, pairwise
from typing import Any, Literal, Self, cast

from pydantic import Field, JsonValue, model_validator
from ul_core.augmentations.definitions import builtin_augmentation_catalog
from ul_core.augmentations.projections import AugmentationProjection, ProjectionTarget
from ul_core.contracts import (
    SemanticDeconstructor,
    SemanticEquivalenceVerifier,
    SemanticRenderer,
)
from ul_core.dataset import (
    InteractionRecord,
    RenderedUserInput,
    SemanticEquivalenceAssessment,
    SemanticFactor,
    SemanticFrame,
    UserInputRecord,
)
from ul_core.models import ULModel
from ul_core.prompts import PromptManager, prompt_provenance

_PROMPTS = PromptManager.instance()
_MAX_DECOMPOSED_RELATION_PARTS = 10_000


def _is_none(value: object) -> bool:
    return value is None


OperatorId = Literal[
    "input.surface.rephrase",
    "input.surface.typing_noise",
    "input.surface.case_variation",
    "input.surface.punctuation_noise",
    "input.surface.grammar_error",
    "input.surface.fragmented_syntax",
    "input.surface.disfluency_repeat",
    "input.style.terse",
    "input.style.verbose",
    "input.tone.frustrated",
    "input.intent.self_correction",
]
AllowedChange = Literal[
    "surface_form_only",
    "declared_communication_form",
    "structured_self_correction",
]
OperatorApplicabilityProfile = Literal["broad", "conditional"]

_OPERATOR_PROMPT_NAMES: dict[OperatorId, str] = {
    "input.surface.rephrase": "augmentation.input.surface.rephrase",
    "input.surface.typing_noise": "augmentation.input.surface.typing_noise",
    "input.surface.case_variation": "augmentation.input.surface.case_variation",
    "input.surface.punctuation_noise": "augmentation.input.surface.punctuation_noise",
    "input.surface.grammar_error": "augmentation.input.surface.grammar_error",
    "input.surface.fragmented_syntax": "augmentation.input.surface.fragmented_syntax",
    "input.surface.disfluency_repeat": "augmentation.input.surface.disfluency_repeat",
    "input.style.terse": "augmentation.input.style.terse",
    "input.style.verbose": "augmentation.input.style.verbose",
    "input.tone.frustrated": "augmentation.input.tone.frustrated",
    "input.intent.self_correction": "augmentation.input.intent.self_correction",
}


class DatasetAugmentationOperator(ULModel):
    id: OperatorId
    version: str = Field(
        default="1.0.0", pattern=r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
    )
    instruction: str = Field(min_length=1)
    applicability_profile: OperatorApplicabilityProfile = "broad"
    applicability_rule: str = Field(
        default="Applies to any nonempty user input with recorded source semantics.",
        min_length=1,
    )
    allowed_change: AllowedChange
    target_communication_kind: str | None = Field(default=None, min_length=1)
    target_marker_required: bool = False
    human_review_required: bool = False

    @model_validator(mode="after")
    def validate_change_contract(self) -> Self:
        requires_target = self.allowed_change in {
            "declared_communication_form",
            "structured_self_correction",
        }
        if requires_target != (self.target_communication_kind is not None):
            raise ValueError("target communication kind must match the allowed change")
        if self.target_marker_required and self.target_communication_kind is None:
            raise ValueError("required target marker needs a communication kind")
        return self


def _builtin_operator(
    operator_id: OperatorId,
    *,
    allowed_change: AllowedChange,
    target_communication_kind: str | None = None,
    target_marker_required: bool = False,
) -> DatasetAugmentationOperator:
    definition = builtin_augmentation_catalog().get(operator_id)
    binding = next(item for item in definition.bindings if item.mode == "dataset_variation")
    return DatasetAugmentationOperator(
        id=operator_id,
        version=definition.ref.version,
        instruction=_PROMPTS.get_prompt(_OPERATOR_PROMPT_NAMES[operator_id]),
        applicability_profile=definition.applicability_profile,
        applicability_rule=definition.applicability_rule,
        allowed_change=allowed_change,
        target_communication_kind=target_communication_kind,
        target_marker_required=target_marker_required,
        human_review_required=binding.requirements.human_review,
    )


_BUILTIN_OPERATORS = (
    _builtin_operator(
        operator_id="input.surface.rephrase",
        allowed_change="surface_form_only",
    ),
    _builtin_operator(
        operator_id="input.surface.typing_noise",
        allowed_change="declared_communication_form",
        target_communication_kind="typing_noise",
    ),
    _builtin_operator(
        operator_id="input.surface.case_variation",
        allowed_change="declared_communication_form",
        target_communication_kind="typing_noise",
    ),
    _builtin_operator(
        operator_id="input.surface.punctuation_noise",
        allowed_change="declared_communication_form",
        target_communication_kind="typing_noise",
    ),
    _builtin_operator(
        operator_id="input.surface.grammar_error",
        allowed_change="declared_communication_form",
        target_communication_kind="fragmented_syntax",
    ),
    _builtin_operator(
        operator_id="input.surface.fragmented_syntax",
        allowed_change="declared_communication_form",
        target_communication_kind="fragmented_syntax",
        target_marker_required=True,
    ),
    _builtin_operator(
        operator_id="input.surface.disfluency_repeat",
        allowed_change="declared_communication_form",
        target_communication_kind="repetition",
    ),
    _builtin_operator(
        operator_id="input.style.terse",
        allowed_change="declared_communication_form",
        target_communication_kind="terse",
    ),
    _builtin_operator(
        operator_id="input.style.verbose",
        allowed_change="declared_communication_form",
        target_communication_kind="verbose",
    ),
    _builtin_operator(
        operator_id="input.tone.frustrated",
        allowed_change="declared_communication_form",
        target_communication_kind="frustrated",
        target_marker_required=True,
    ),
    _builtin_operator(
        operator_id="input.intent.self_correction",
        allowed_change="structured_self_correction",
        target_communication_kind="self_correction",
        target_marker_required=True,
    ),
)
_BUILTIN_OPERATORS_BY_REFERENCE = {
    (operator.id, operator.version): operator for operator in _BUILTIN_OPERATORS
}


def builtin_dataset_augmentation_operators() -> tuple[DatasetAugmentationOperator, ...]:
    return _BUILTIN_OPERATORS


def resolve_dataset_augmentation_operator(reference: str) -> DatasetAugmentationOperator:
    if not reference or len(reference) > 251 or reference.count("@") > 1:
        raise ValueError("unknown dataset augmentation reference")
    operator_id, separator, version = reference.partition("@")
    if separator:
        operator = _BUILTIN_OPERATORS_BY_REFERENCE.get((operator_id, version))
        if operator is None:
            raise ValueError("unknown dataset augmentation reference")
        return operator
    matching = tuple(operator for operator in _BUILTIN_OPERATORS if operator.id == operator_id)
    if not matching:
        raise ValueError("unknown dataset augmentation reference")
    return max(matching, key=lambda operator: _version_tuple(operator.version))


class DatasetAugmentationCandidate(ULModel):
    source_interaction_id: str = Field(min_length=1)
    source_record_id: str | None = cast(Any, Field)(default=None, min_length=1, exclude_if=_is_none)
    augmentation_target_id: str | None = cast(Any, Field)(
        default=None, min_length=1, exclude_if=_is_none
    )
    operator_id: OperatorId = "input.surface.rephrase"
    operator_version: str = Field(
        default="1.0.0",
        pattern=r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$",
    )
    allowed_change: AllowedChange = "surface_form_only"
    human_review_required: bool = False
    projection: AugmentationProjection
    changed_paths: tuple[str, ...] = ()
    changed_events: tuple[str, ...] = ()
    augmented_input: str = Field(min_length=1)
    renderer_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    expected_input_frame: SemanticFrame
    reparsed_input_frame: SemanticFrame | None
    semantic_equivalence_assessment: SemanticEquivalenceAssessment | None = None
    passed: bool
    failure_reasons: tuple[str, ...] = ()


class DatasetAugmentationOperatorReference(ULModel):
    id: OperatorId
    version: str = Field(
        default="1.0.0",
        pattern=r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$",
    )


class DatasetAugmentationSkip(ULModel):
    source_interaction_id: str = Field(min_length=1)
    operator_id: OperatorId
    operator_version: str = Field(
        default="1.0.0",
        pattern=r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$",
    )
    reason: str = Field(min_length=1)


class DatasetAugmentationResult(ULModel):
    operator_references: tuple[DatasetAugmentationOperatorReference, ...] = Field(min_length=1)
    source_records: tuple[InteractionRecord, ...]
    source_frames: tuple[SemanticFrame, ...]
    candidates: tuple[DatasetAugmentationCandidate, ...]
    skips: tuple[DatasetAugmentationSkip, ...] = ()

    @model_validator(mode="after")
    def validate_operator_plan(self) -> Self:
        if tuple(record.id for record in self.source_records) != tuple(
            frame.interaction_id for frame in self.source_frames
        ):
            raise ValueError("augmentation source records and frames must match")
        references = tuple(
            (reference.id, reference.version) for reference in self.operator_references
        )
        if len(references) != len(set(references)):
            raise ValueError("augmentation result contains duplicate operator references")
        positions = {reference: index for index, reference in enumerate(references)}
        candidate_references = tuple(
            (candidate.operator_id, candidate.operator_version) for candidate in self.candidates
        )
        try:
            candidate_positions = tuple(positions[reference] for reference in candidate_references)
        except KeyError:
            raise ValueError(
                "augmentation candidate is outside the requested operator plan"
            ) from None
        if candidate_positions != tuple(sorted(candidate_positions)):
            raise ValueError("augmentation candidates must follow requested operator order")
        source_ids = {record.id for record in self.source_records}
        skip_references = tuple((skip.operator_id, skip.operator_version) for skip in self.skips)
        if any(reference not in positions for reference in skip_references):
            raise ValueError("augmentation skip is outside the requested operator plan")
        if any(skip.source_interaction_id not in source_ids for skip in self.skips):
            raise ValueError("augmentation skip references an unknown source interaction")
        candidate_keys = {
            (candidate.source_interaction_id, candidate.operator_id, candidate.operator_version)
            for candidate in self.candidates
        }
        skip_keys = tuple(
            (skip.source_interaction_id, skip.operator_id, skip.operator_version)
            for skip in self.skips
        )
        if len(skip_keys) != len(set(skip_keys)):
            raise ValueError("augmentation result contains duplicate skips")
        if candidate_keys & set(skip_keys):
            raise ValueError("augmentation result cannot both generate and skip an operator")
        records_by_id = {record.id: record for record in self.source_records}
        for candidate in self.candidates:
            record = records_by_id.get(candidate.source_interaction_id)
            if record is None:
                raise ValueError("augmentation candidate references an unknown source interaction")
            binding = next(
                binding
                for binding in builtin_augmentation_catalog()
                .get(candidate.operator_id, candidate.operator_version)
                .bindings
                if binding.mode == "dataset_variation"
            )
            binding.projection.validate_projection(candidate.projection)
            try:
                changes = candidate.projection.validate_candidate(
                    record.augmentation_document(),
                    record.augmentation_document(candidate.augmented_input),
                )
            except ValueError as error:
                if str(error) != "augmentation candidate does not change its source":
                    raise
                changes = None
            expected_paths = changes.changed_paths if changes is not None else ()
            expected_events = changes.changed_events if changes is not None else ()
            if (
                candidate.changed_paths != expected_paths
                or candidate.changed_events != expected_events
            ):
                raise ValueError("augmentation candidate change set does not match its projection")
            if changes is None and candidate.passed:
                raise ValueError("unchanged augmentation candidates cannot pass")
        return self


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
        operator_ids: Iterable[str] = ("input.surface.rephrase",),
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
        skips: list[DatasetAugmentationSkip] = []
        for record in source_records:
            source_frame = await self._deconstructor.deconstruct(record)
            if source_frame.interaction_id != record.id:
                raise ValueError("deconstructed frame must reference its source interaction")
            source_frames.append(source_frame)
            expected_input_frame = _input_only_frame(source_frame)
            source_skip_reason = _source_skip_reason(source_frame, expected_input_frame)
            if source_skip_reason is not None:
                skips.extend(
                    DatasetAugmentationSkip(
                        source_interaction_id=record.id,
                        operator_id=operator.id,
                        operator_version=operator.version,
                        reason=source_skip_reason,
                    )
                    for operator in selected_operators
                )
                continue
            generated_inputs: set[str] = set()
            for operator in selected_operators:
                transformation_prompt_names: tuple[str, ...] = ()
                selected_correction_factor: SemanticFactor | None = None
                planned_provisional_quote: str | None = None
                if (
                    operator.id == "input.surface.case_variation"
                    and _first_cased_letter(record.raw_input, expected_input_frame) is None
                ):
                    skips.append(
                        DatasetAugmentationSkip(
                            source_interaction_id=record.id,
                            operator_id=operator.id,
                            operator_version=operator.version,
                            reason=operator.applicability_rule,
                        )
                    )
                    continue
                if (
                    operator.id == "input.surface.punctuation_noise"
                    and _punctuation_insertion(record.raw_input, expected_input_frame) is None
                ):
                    skips.append(
                        DatasetAugmentationSkip(
                            source_interaction_id=record.id,
                            operator_id=operator.id,
                            operator_version=operator.version,
                            reason=operator.applicability_rule,
                        )
                    )
                    continue
                if operator.allowed_change == "structured_self_correction":
                    selected_correction_factor = _select_self_correction_factor(
                        record, source_frame
                    )
                    if selected_correction_factor is None:
                        skips.append(
                            DatasetAugmentationSkip(
                                source_interaction_id=record.id,
                                operator_id=operator.id,
                                operator_version=operator.version,
                                reason=operator.applicability_rule,
                            )
                        )
                        continue
                    planned_provisional_quote = _planned_provisional_quote(
                        selected_correction_factor, record.raw_input
                    )
                    if planned_provisional_quote is None:
                        skips.append(
                            DatasetAugmentationSkip(
                                source_interaction_id=record.id,
                                operator_id=operator.id,
                                operator_version=operator.version,
                                reason=(
                                    "The selected source value cannot be safely changed into a "
                                    "distinct temporary value."
                                ),
                            )
                        )
                        continue
                if operator.id == "input.surface.typing_noise":
                    rendered_input = _add_typing_noise(record, expected_input_frame, operator)
                elif operator.id == "input.surface.case_variation":
                    rendered_input = _add_case_variation(record, expected_input_frame, operator)
                elif operator.id == "input.surface.punctuation_noise":
                    rendered_input = _add_punctuation_noise(record, expected_input_frame, operator)
                elif operator.id == "input.surface.grammar_error":
                    rendered_input = _add_grammar_error(record, operator)
                elif operator.id == "input.surface.disfluency_repeat":
                    rendered_input = _add_word_repetition(record, expected_input_frame, operator)
                elif operator.id == "input.tone.frustrated":
                    rendered_input = _add_frustrated_tone(record, operator)
                elif operator.allowed_change == "structured_self_correction":
                    transformation_prompt_names = (
                        _OPERATOR_PROMPT_NAMES[operator.id],
                        "augmentation.input.intent.self_correction_argument",
                    )
                    if selected_correction_factor is None:
                        raise AssertionError("self-correction requires a selected factor")
                    correction_quote = _unique_input_quote(selected_correction_factor)
                    if correction_quote is None:
                        raise AssertionError("selected correction factor requires a unique quote")
                    argument_instruction = _PROMPTS.get_prompt(
                        "augmentation.input.intent.self_correction_argument",
                        source_text=json.dumps(correction_quote, ensure_ascii=False),
                        temporary_text=json.dumps(planned_provisional_quote, ensure_ascii=False),
                    )
                    rendered_input = await self._renderer.render(
                        record.raw_input,
                        f"{operator.instruction} {argument_instruction}",
                        allow_temporary_value=True,
                    )
                else:
                    transformation_prompt_names = (_OPERATOR_PROMPT_NAMES[operator.id],)
                    rendered_input = await self._renderer.render(
                        record.raw_input, operator.instruction
                    )
                renderer_metadata: dict[str, JsonValue] = {
                    **rendered_input.metadata,
                    "transformation_prompts": prompt_provenance(*transformation_prompt_names),
                }
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
                    elif operator.allowed_change == "structured_self_correction":
                        if selected_correction_factor is None:
                            raise AssertionError("self-correction requires a selected factor")
                        if planned_provisional_quote is None:
                            raise AssertionError("self-correction requires a provisional value")
                        failure_reasons = list(
                            _structured_self_correction_difference_reasons(
                                expected_input_frame,
                                reparsed_frame,
                                selected_correction_factor,
                                planned_provisional_quote,
                                record.raw_input,
                                augmented_input,
                            )
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
                        and operator.allowed_change != "structured_self_correction"
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
                projection = create_dataset_augmentation_projection(record)
                binding = next(
                    binding
                    for binding in builtin_augmentation_catalog()
                    .get(operator.id, operator.version)
                    .bindings
                    if binding.mode == "dataset_variation"
                )
                binding.projection.validate_projection(projection)
                try:
                    changes = projection.validate_candidate(
                        record.augmentation_document(),
                        record.augmentation_document(augmented_input),
                    )
                except ValueError as error:
                    if str(error) != "augmentation candidate does not change its source":
                        raise
                    changes = None
                candidates.append(
                    DatasetAugmentationCandidate(
                        source_interaction_id=record.id,
                        source_record_id=(
                            record.source_interaction_id
                            if record.augmentation_target is not None
                            else None
                        ),
                        augmentation_target_id=(
                            record.augmentation_target.id
                            if record.augmentation_target is not None
                            else None
                        ),
                        operator_id=operator.id,
                        operator_version=operator.version,
                        allowed_change=operator.allowed_change,
                        human_review_required=operator.human_review_required,
                        projection=projection,
                        changed_paths=changes.changed_paths if changes is not None else (),
                        changed_events=changes.changed_events if changes is not None else (),
                        augmented_input=augmented_input,
                        renderer_metadata=renderer_metadata,
                        expected_input_frame=expected_input_frame,
                        reparsed_input_frame=reparsed_frame,
                        semantic_equivalence_assessment=equivalence_assessment,
                        passed=not failure_reasons,
                        failure_reasons=tuple(failure_reasons),
                    )
                )
        return DatasetAugmentationResult(
            operator_references=tuple(
                DatasetAugmentationOperatorReference(id=operator.id, version=operator.version)
                for operator in selected_operators
            ),
            source_records=source_records,
            source_frames=tuple(source_frames),
            candidates=tuple(candidates),
            skips=tuple(skips),
        )


def create_dataset_augmentation_projection(
    record: InteractionRecord,
) -> AugmentationProjection:
    surface = (
        "conversation"
        if record.augmentation_target is not None
        and record.augmentation_target.kind == "conversation_turn"
        else "structured_input"
    )
    return AugmentationProjection(
        reads=(
            ProjectionTarget(
                id="source-augmentation-target",
                surface=surface,
                path=record.augmentation_path,
            ),
        ),
        writes=(
            ProjectionTarget(
                id="candidate-augmentation-target",
                surface=surface,
                path=record.augmentation_path,
            ),
        ),
    )


def _source_skip_reason(
    source_frame: SemanticFrame, expected_input_frame: SemanticFrame
) -> str | None:
    if not expected_input_frame.request_units:
        return "Source semantics contain no actionable request units."
    if not source_frame.outcomes:
        return "Source interaction contains no observed outcome to preserve."
    unresolved_reason = _unresolved_node_reason(source_frame)
    if unresolved_reason is not None:
        return unresolved_reason
    ambiguous_reason = _ambiguous_node_reason(expected_input_frame)
    if ambiguous_reason is not None:
        return ambiguous_reason
    return None


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


_SELF_CORRECTION_FACTOR_KINDS = ("money", "number", "date_time", "duration")


def _select_self_correction_factor(
    record: InteractionRecord, frame: SemanticFrame
) -> SemanticFactor | None:
    if any(act.kind == "self_correction" for act in frame.communication_acts) or any(
        relation.kind == "superseded_by" for relation in frame.relations
    ):
        return None
    action_outcomes_by_request_id = {
        request_id: tuple(
            outcome
            for outcome in frame.outcomes
            if outcome.kind == "action"
            and outcome.status == "observed"
            and request_id in outcome.request_unit_ids
        )
        for request_id in (request.id for request in frame.request_units)
    }
    action_requests_by_factor_id = {
        factor.id: tuple(
            request
            for request in frame.request_units
            if request.mode == "act"
            and action_outcomes_by_request_id[request.id]
            and factor.id in request.factor_ids
        )
        for factor in frame.factors
    }
    eligible_factors = tuple(
        factor
        for factor in frame.factors
        if len(action_requests_by_factor_id[factor.id]) == 1
        and factor.kind in _SELF_CORRECTION_FACTOR_KINDS
        and factor.status in {"explicit", "observed"}
        and _is_self_correction_value(factor.value)
        and any(
            _json_key(outcome.fields.get(factor.role)) == _json_key(factor.value)
            for outcome in action_outcomes_by_request_id[
                action_requests_by_factor_id[factor.id][0].id
            ]
        )
        and (quote := _unique_input_quote(factor)) is not None
        and record.raw_input.count(quote) == 1
    )
    if not eligible_factors:
        return None
    factor_kind_priority = {
        kind: priority for priority, kind in enumerate(_SELF_CORRECTION_FACTOR_KINDS)
    }
    return min(
        eligible_factors,
        key=lambda factor: (factor_kind_priority[factor.kind], frame.factors.index(factor)),
    )


def _is_self_correction_value(value: JsonValue) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)


def _unique_input_quote(factor: SemanticFactor) -> str | None:
    quotes = tuple(
        dict.fromkeys(
            evidence.text_quote
            for evidence in factor.evidence
            if evidence.source == "input" and evidence.text_quote is not None
        )
    )
    if len(quotes) != 1:
        return None
    return quotes[0]


def _planned_provisional_quote(factor: SemanticFactor, source_input: str) -> str | None:
    source_quote = _unique_input_quote(factor)
    if source_quote is None:
        return None
    match = re.search(r"(?<![\w.])(?P<number>\d+(?:\.\d+)?)(?![\w.])", source_quote)
    if match is None:
        return None
    source_number_text = match.group("number")
    if len(source_number_text) > 64:
        return None
    source_number = Decimal(source_number_text)
    if source_number != source_number.to_integral_value():
        provisional_number = source_number + 1
    else:
        integer_value = int(source_number)
        if integer_value < 10:
            provisional_number = Decimal(integer_value + 2)
        elif integer_value < 100:
            provisional_number = Decimal(integer_value - 1)
        else:
            provisional_number = Decimal(integer_value + 10 ** (len(str(integer_value)) - 2))
    provisional_text = (
        str(int(provisional_number))
        if provisional_number == provisional_number.to_integral_value()
        else format(provisional_number, "f").rstrip("0").rstrip(".")
    )
    provisional_quote = (
        f"{source_quote[: match.start()]}{provisional_text}{source_quote[match.end() :]}"
    )
    if provisional_quote in source_input or source_quote in provisional_quote:
        return None
    return provisional_quote


def _structured_self_correction_difference_reasons(
    expected: SemanticFrame,
    reparsed: SemanticFrame,
    selected_source_factor: SemanticFactor,
    planned_provisional_quote: str,
    source_input: str,
    augmented_input: str,
) -> tuple[str, ...]:
    correction_acts = tuple(
        act for act in reparsed.communication_acts if act.kind == "self_correction"
    )
    correction_relations = tuple(
        relation for relation in reparsed.relations if relation.kind == "superseded_by"
    )
    if len(correction_acts) != 1:
        return ("reparsed frame must contain exactly one self-correction act",)
    if len(correction_relations) != 1:
        return ("reparsed frame must contain exactly one superseded-by relation",)
    correction_act = correction_acts[0]
    correction_relation = correction_relations[0]
    if correction_act.attributes:
        return ("self-correction act cannot contain attributes",)
    if len(correction_act.factor_ids) != 2:
        return ("self-correction act must reference provisional and final factors",)
    provisional_factor_id, final_factor_id = correction_act.factor_ids
    if correction_relation.source_ids != (
        provisional_factor_id,
    ) or correction_relation.target_ids != (final_factor_id,):
        return ("superseded-by relation must connect provisional to final factor",)
    factors_by_id = {factor.id: factor for factor in reparsed.factors}
    provisional_factor = factors_by_id.get(provisional_factor_id)
    final_factor = factors_by_id.get(final_factor_id)
    if provisional_factor is None or final_factor is None:
        return ("self-correction references an unknown factor",)
    if provisional_factor.status != "superseded":
        return ("provisional factor must be marked superseded",)
    if any(provisional_factor_id in request.factor_ids for request in reparsed.request_units):
        return ("request units must reference only the final correction factor",)
    if provisional_factor.kind != final_factor.kind or provisional_factor.role != final_factor.role:
        return ("provisional and final factors must have the same kind and role",)
    if _json_key(provisional_factor.value) == _json_key(final_factor.value):
        return ("provisional and final factors must have different values",)
    provisional_quote = _unique_input_quote(provisional_factor)
    final_quote = _unique_input_quote(final_factor)
    selected_source_quote = _unique_input_quote(selected_source_factor)
    if provisional_quote is None or final_quote is None or selected_source_quote is None:
        return ("provisional and final factors require one direct input quote",)
    if provisional_quote != planned_provisional_quote:
        return ("provisional correction must match the planned temporary value",)
    if not all(
        _factor_value_is_supported_by_quote(factor, quote)
        for factor, quote in (
            (provisional_factor, provisional_quote),
            (final_factor, final_quote),
            (selected_source_factor, selected_source_quote),
        )
    ):
        return ("correction factor values must be supported by their exact text",)
    if final_quote != selected_source_quote:
        return ("final correction must retain the exact selected source text",)
    if (final_factor.kind, final_factor.role) != (
        selected_source_factor.kind,
        selected_source_factor.role,
    ):
        return ("final correction factor must preserve the selected source kind and role",)
    if _json_key(final_factor.value) != _json_key(selected_source_factor.value):
        return ("final correction factor must preserve the selected source value",)
    expected_factor_semantics = {
        (factor.kind, factor.role, _json_key(factor.value)) for factor in expected.factors
    }
    provisional_semantics = (
        provisional_factor.kind,
        provisional_factor.role,
        _json_key(provisional_factor.value),
    )
    if provisional_semantics in expected_factor_semantics:
        return ("provisional value cannot be an active source value",)
    if any(
        provisional_factor_id in (*relation.source_ids, *relation.target_ids)
        for relation in reparsed.relations
        if relation.id != correction_relation.id
    ) or any(
        provisional_factor_id in act.factor_ids
        for act in reparsed.communication_acts
        if act.id != correction_act.id
    ):
        return ("provisional factor can only appear in correction artifacts",)
    artifact_elements = (provisional_factor, correction_act, correction_relation)
    if any(not _has_input_evidence(element) for element in artifact_elements):
        return ("self-correction artifacts require direct input evidence",)
    if provisional_quote in source_input:
        return ("provisional value must be new to the augmented input",)
    if augmented_input.count(provisional_quote) != 1 or augmented_input.count(final_quote) != 1:
        return ("provisional and final values must each appear exactly once",)
    if augmented_input.index(provisional_quote) >= augmented_input.index(final_quote):
        return ("provisional value must appear before the final value",)
    between_values = augmented_input[
        augmented_input.index(provisional_quote) + len(provisional_quote) : augmented_input.index(
            final_quote
        )
    ]
    if not any(character.isalpha() for character in between_values):
        return ("correction language must appear between provisional and final values",)
    if not all(
        _element_evidence_spans_values(element, provisional_quote, final_quote)
        for element in (correction_act, correction_relation)
    ):
        return ("correction act and relation evidence must span both values in order",)
    stripped_frame = reparsed.model_copy(
        update={
            "factors": tuple(
                factor for factor in reparsed.factors if factor.id != provisional_factor_id
            ),
            "relations": tuple(
                relation for relation in reparsed.relations if relation.id != correction_relation.id
            ),
            "communication_acts": tuple(
                act for act in reparsed.communication_acts if act.id != correction_act.id
            ),
        }
    )
    return _semantic_difference_reasons(expected, stripped_frame)


def _element_evidence_spans_values(element: Any, provisional_quote: str, final_quote: str) -> bool:
    return any(
        evidence.source == "input"
        and evidence.text_quote is not None
        and provisional_quote in evidence.text_quote
        and final_quote in evidence.text_quote
        and evidence.text_quote.index(provisional_quote) < evidence.text_quote.index(final_quote)
        for evidence in element.evidence
    )


def _factor_value_is_supported_by_quote(factor: SemanticFactor, quote: str) -> bool:
    if not isinstance(factor.value, (str, int, float)) or isinstance(factor.value, bool):
        return False
    normalized_value = "".join(
        character for character in str(factor.value).casefold() if character.isalnum()
    )
    normalized_quote = "".join(character for character in quote.casefold() if character.isalnum())
    return bool(normalized_value) and normalized_value == normalized_quote


def _surface_footprint_reasons(
    operator_id: OperatorId,
    source_input: str,
    augmented_input: str,
) -> tuple[str, ...]:
    source_word_count = len(re.findall(r"\w+", source_input, flags=re.UNICODE))
    augmented_words = re.findall(r"\w+", augmented_input, flags=re.UNICODE)
    augmented_word_count = len(augmented_words)
    if operator_id == "input.surface.rephrase" and _word_key(source_input) == _word_key(
        augmented_input
    ):
        return ("rendered input only changes case, spacing, or punctuation",)
    if operator_id == "input.surface.case_variation" and not _is_single_case_change(
        source_input, augmented_input
    ):
        return ("rendered input must contain exactly one single-code-point case change",)
    if operator_id == "input.surface.punctuation_noise" and not _is_single_punctuation_insertion(
        source_input, augmented_input
    ):
        return ("rendered input must insert exactly one punctuation character",)
    if operator_id == "input.style.terse" and augmented_word_count * 10 > source_word_count * 9:
        return ("rendered input is not visibly shorter than the source",)
    if operator_id == "input.style.verbose" and not (
        source_word_count * 15 <= augmented_word_count * 10
        and augmented_word_count <= source_word_count * 2
    ):
        return ("rendered input is not between 1.5 and 2 times the source length",)
    if operator_id == "input.surface.disfluency_repeat":
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


def _deterministic_renderer_metadata(
    record: InteractionRecord,
    operator: DatasetAugmentationOperator,
    algorithm: str,
) -> dict[str, JsonValue]:
    return {
        "renderer": "deterministic",
        "algorithm": algorithm,
        "seed": int.from_bytes(
            hashlib.sha256(f"{record.id}\0{operator.id}\0{operator.version}".encode()).digest()[:4],
            "big",
        ),
    }


def _add_case_variation(
    record: InteractionRecord,
    frame: SemanticFrame,
    operator: DatasetAugmentationOperator,
) -> RenderedUserInput:
    cased_letter = _first_cased_letter(record.raw_input, frame)
    if cased_letter is None:
        raise AssertionError("case variation requires a Unicode cased letter")
    index, letter = cased_letter
    changed_letter = letter.lower() if letter.isupper() else letter.upper()
    rendered_text = f"{record.raw_input[:index]}{changed_letter}{record.raw_input[index + 1 :]}"
    return RenderedUserInput(
        text=rendered_text,
        metadata=_deterministic_renderer_metadata(
            record, operator, "single_unicode_cased_letter_toggle"
        ),
    )


def _first_cased_letter(text: str, frame: SemanticFrame) -> tuple[int, str] | None:
    protected_spans = _protected_input_spans(text, frame)
    return next(
        (
            (index, character)
            for index, character in enumerate(text)
            if character.isalpha()
            and character.lower() != character.upper()
            and len(character.lower()) == 1
            and len(character.upper()) == 1
            and not _position_is_protected(index, protected_spans)
        ),
        None,
    )


def _add_punctuation_noise(
    record: InteractionRecord,
    frame: SemanticFrame,
    operator: DatasetAugmentationOperator,
) -> RenderedUserInput:
    insertion = _punctuation_insertion(record.raw_input, frame)
    if insertion is None:
        raise AssertionError("punctuation noise requires an unprotected insertion point")
    position, punctuation, algorithm = insertion
    rendered_text = f"{record.raw_input[:position]}{punctuation}{record.raw_input[position:]}"
    return RenderedUserInput(
        text=rendered_text,
        metadata=_deterministic_renderer_metadata(record, operator, algorithm),
    )


def _punctuation_insertion(text: str, frame: SemanticFrame) -> tuple[int, str, str] | None:
    protected_spans = _protected_input_spans(text, frame)
    safe_punctuation = next(
        (
            match
            for match in re.finditer(r"[,.!?;:]", text)
            if not _position_is_protected(match.start(), protected_spans)
            and (
                match.end() == len(text)
                or text[match.end()].isspace()
                or (match.start() > 0 and text[match.start() - 1].isspace())
            )
        ),
        None,
    )
    if safe_punctuation is not None:
        return (
            safe_punctuation.end(),
            safe_punctuation.group(),
            "single_safe_punctuation_duplication",
        )
    word = next(
        (
            match
            for match in re.finditer(r"[^\W\d_]{3,}", text, flags=re.UNICODE)
            if not _insertion_is_protected(match.start() + 1, protected_spans)
        ),
        None,
    )
    if word is None:
        return None
    return word.start() + 1, ",", "single_unprotected_word_punctuation_insertion"


def _protected_input_spans(text: str, frame: SemanticFrame) -> tuple[tuple[int, int], ...]:
    spans: set[tuple[int, int]] = set()
    for factor in frame.factors:
        for evidence in factor.evidence:
            quote = evidence.text_quote
            if evidence.source != "input" or quote is None:
                continue
            start = 0
            while (position := text.find(quote, start)) >= 0:
                spans.add((position, position + len(quote)))
                start = position + 1
    return tuple(sorted(spans))


def _position_is_protected(position: int, spans: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= position < end for start, end in spans)


def _insertion_is_protected(position: int, spans: tuple[tuple[int, int], ...]) -> bool:
    return any(start < position < end for start, end in spans)


def _is_single_case_change(source_input: str, augmented_input: str) -> bool:
    if len(source_input) != len(augmented_input):
        return False
    changed_positions = tuple(
        index
        for index, (source_character, augmented_character) in enumerate(
            zip(source_input, augmented_input, strict=True)
        )
        if source_character != augmented_character
    )
    if len(changed_positions) != 1:
        return False
    position = changed_positions[0]
    source_character = source_input[position]
    augmented_character = augmented_input[position]
    return (
        source_character.lower() == augmented_character.lower()
        and source_character.upper() == augmented_character.upper()
    )


def _is_single_punctuation_insertion(source_input: str, augmented_input: str) -> bool:
    if len(augmented_input) != len(source_input) + 1:
        return False
    return any(
        augmented_input[index] in ",.!?;:"
        and f"{augmented_input[:index]}{augmented_input[index + 1 :]}" == source_input
        for index in range(len(augmented_input))
    )


def _add_grammar_error(
    record: InteractionRecord, operator: DatasetAugmentationOperator
) -> RenderedUserInput:
    return RenderedUserInput(
        text=f"Me need you to: {record.raw_input}",
        metadata=_deterministic_renderer_metadata(
            record, operator, "pronoun_case_error_request_prefix"
        ),
    )


def _add_frustrated_tone(
    record: InteractionRecord, operator: DatasetAugmentationOperator
) -> RenderedUserInput:
    return RenderedUserInput(
        text=f"Ugh, {record.raw_input}",
        metadata=_deterministic_renderer_metadata(
            record, operator, "frustration_interjection_prefix"
        ),
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


def _ambiguous_node_reason(frame: SemanticFrame) -> str | None:
    factors_by_id = {
        factor.id: (factor.kind, factor.role, _json_key(factor.value)) for factor in frame.factors
    }
    factor_semantics = tuple((factor.id, factors_by_id[factor.id]) for factor in frame.factors)
    request_semantics = tuple(
        (
            request.id,
            (
                request.mode,
                request.predicate,
                tuple(sorted(factors_by_id[factor_id] for factor_id in request.factor_ids)),
            ),
        )
        for request in frame.request_units
    )
    communication_semantics = tuple(
        (
            act.id,
            (
                act.kind,
                tuple(sorted(factors_by_id[factor_id] for factor_id in act.factor_ids)),
                _json_key(act.attributes),
            ),
        )
        for act in frame.communication_acts
    )
    for label, elements in (
        ("factors", factor_semantics),
        ("request units", request_semantics),
        ("communication acts", communication_semantics),
    ):
        ids_by_semantics: dict[object, str] = {}
        for element_id, semantics in elements:
            previous_id = ids_by_semantics.get(semantics)
            if previous_id is not None:
                return (
                    f"Source {label} {previous_id!r} and {element_id!r} have indistinguishable "
                    "semantics; clarify their evidence or roles."
                )
            ids_by_semantics[semantics] = element_id
    return None


def _has_unresolved_nodes(frame: SemanticFrame) -> bool:
    return _unresolved_node_reason(frame) is not None


def _unresolved_node_reason(frame: SemanticFrame) -> str | None:
    element_groups = (
        ("request unit", frame.request_units),
        ("factor", frame.factors),
        ("relation", frame.relations),
        ("communication act", frame.communication_acts),
        ("outcome", frame.outcomes),
    )
    for label, elements in element_groups:
        for element in elements:
            if element.status.casefold().startswith("unresolved"):
                return (
                    f"Source {label} {element.id!r} is unresolved; clarify its evidence or "
                    "improve semantic extraction."
                )
    return None


def _canonical_semantics(frame: SemanticFrame) -> tuple[object, ...]:
    factor_semantics: dict[str, tuple[tuple[str, ...], ...]] = {}
    list_offsets: dict[tuple[str, str], int] = {}
    for factor in frame.factors:
        list_key = (factor.kind, factor.role)
        list_offset = list_offsets.get(list_key, 0)
        parts = _factor_semantic_parts(factor, list_offset=list_offset)
        factor_semantics[factor.id] = parts
        if isinstance(factor.value, list):
            list_offsets[list_key] = list_offset + len(parts)
    request_semantics = {
        request.id: (
            request.mode,
            request.predicate,
            tuple(
                sorted(
                    part for factor_id in request.factor_ids for part in factor_semantics[factor_id]
                )
            ),
        )
        for request in frame.request_units
    }
    communication_semantics = {
        act.id: (
            act.kind,
            tuple(
                sorted(part for factor_id in act.factor_ids for part in factor_semantics[factor_id])
            ),
            _json_key(act.attributes),
        )
        for act in frame.communication_acts
    }
    endpoint_semantics = {
        **{
            identifier: tuple(("factor", part) for part in parts)
            for identifier, parts in factor_semantics.items()
        },
        **{identifier: (("request", value),) for identifier, value in request_semantics.items()},
        **{
            identifier: (("communication", value),)
            for identifier, value in communication_semantics.items()
        },
    }
    list_factor_ids = {factor.id for factor in frame.factors if isinstance(factor.value, list)}
    canonical_relations: list[tuple[object, ...]] = []
    decomposed_relation_parts = 0
    for relation in frame.relations:
        expanded_relations = _expanded_relation_semantics(
            relation.kind,
            relation.source_ids,
            relation.target_ids,
            endpoint_semantics,
            list_factor_ids,
        )
        if decomposed_relation_parts + len(expanded_relations) > _MAX_DECOMPOSED_RELATION_PARTS:
            canonical_relations.append(
                _unexpanded_relation_semantics(
                    relation.kind,
                    relation.source_ids,
                    relation.target_ids,
                    endpoint_semantics,
                )
            )
        else:
            canonical_relations.extend(expanded_relations)
            decomposed_relation_parts += len(expanded_relations)
    relation_semantics = tuple(sorted(canonical_relations))
    return (
        tuple(sorted(part for parts in factor_semantics.values() for part in parts)),
        tuple(request_semantics[request.id] for request in frame.request_units),
        relation_semantics,
        tuple(communication_semantics[act.id] for act in frame.communication_acts),
    )


def _factor_semantic_parts(
    factor: SemanticFactor, *, list_offset: int
) -> tuple[tuple[str, ...], ...]:
    if isinstance(factor.value, list):
        if not factor.value:
            return ((factor.kind, factor.role, "list", "[]"),)
        return tuple(
            (
                factor.kind,
                factor.role,
                "list_item",
                str(list_offset + index),
                _json_key(item),
            )
            for index, item in enumerate(factor.value)
        )
    return ((factor.kind, factor.role, "value", _json_key(factor.value)),)


def _expanded_relation_semantics(
    kind: str,
    source_ids: tuple[str, ...],
    target_ids: tuple[str, ...],
    endpoint_semantics: Mapping[str, tuple[tuple[object, ...], ...]],
    list_factor_ids: set[str],
) -> tuple[tuple[object, ...], ...]:
    decomposed_endpoint_ids = tuple(
        item for item in (*source_ids, *target_ids) if item in list_factor_ids
    )
    if len(decomposed_endpoint_ids) != 1:
        return (_unexpanded_relation_semantics(kind, source_ids, target_ids, endpoint_semantics),)
    decomposed_endpoint_id = decomposed_endpoint_ids[0]
    return tuple(
        (
            kind,
            _relation_endpoint_group(
                source_ids,
                endpoint_semantics,
                decomposed_endpoint_id=decomposed_endpoint_id,
                decomposed_part=part,
            ),
            _relation_endpoint_group(
                target_ids,
                endpoint_semantics,
                decomposed_endpoint_id=decomposed_endpoint_id,
                decomposed_part=part,
            ),
        )
        for part in endpoint_semantics[decomposed_endpoint_id]
    )


def _unexpanded_relation_semantics(
    kind: str,
    source_ids: tuple[str, ...],
    target_ids: tuple[str, ...],
    endpoint_semantics: Mapping[str, tuple[tuple[object, ...], ...]],
) -> tuple[object, ...]:
    return (
        kind,
        _relation_endpoint_group(source_ids, endpoint_semantics),
        _relation_endpoint_group(target_ids, endpoint_semantics),
    )


def _relation_endpoint_group(
    endpoint_ids: tuple[str, ...],
    endpoint_semantics: Mapping[str, tuple[tuple[object, ...], ...]],
    *,
    decomposed_endpoint_id: str | None = None,
    decomposed_part: tuple[object, ...] | None = None,
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        sorted(
            part
            for endpoint_id in endpoint_ids
            for part in (
                (decomposed_part,)
                if endpoint_id == decomposed_endpoint_id and decomposed_part is not None
                else endpoint_semantics[endpoint_id]
            )
        )
    )


def _select_operators(
    operator_ids: Iterable[str],
) -> tuple[DatasetAugmentationOperator, ...]:
    selected_references = tuple(islice(operator_ids, len(_BUILTIN_OPERATORS) + 1))
    if not selected_references:
        raise ValueError("operator_ids must contain at least one operator")
    if any(not reference for reference in selected_references):
        raise ValueError("operator identifiers must not be empty")
    if len(selected_references) > len(_BUILTIN_OPERATORS):
        raise ValueError("operator count exceeds the built-in library")
    selected_operators = tuple(
        resolve_dataset_augmentation_operator(reference) for reference in selected_references
    )
    resolved_references = tuple((operator.id, operator.version) for operator in selected_operators)
    if len(resolved_references) != len(set(resolved_references)):
        raise ValueError("operator identifiers must be unique")
    return selected_operators


def _version_tuple(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)


def _text_key(text: str) -> str:
    return " ".join(text.split()).casefold()


def _word_key(text: str) -> tuple[str, ...]:
    return tuple(word.casefold() for word in re.findall(r"\w+", text, flags=re.UNICODE))


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
