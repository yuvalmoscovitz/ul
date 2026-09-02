"""Dataset augmentation bindings and generation runtime."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from itertools import islice
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
    CommunicationAct,
    EvidenceReference,
    InteractionRecord,
    RenderedUserInput,
    RequestUnit,
    SemanticAllowedSurfaceChange,
    SemanticEquivalenceAssessment,
    SemanticFactor,
    SemanticFrame,
    SemanticRelation,
    UserInputRecord,
)
from ul_core.models import ULModel
from ul_core.prompts import PromptManager, prompt_provenance

_PROMPTS = PromptManager.instance()
_MAX_DECOMPOSED_RELATION_ENDPOINTS = 10_000
_TONE_SAFETY_KINDS = {"angry", "argumentative"}


@dataclass(frozen=True)
class _SelfCorrectionPlan:
    factor: SemanticFactor
    provisional_quote: str
    provisional_value: JsonValue
    grounding: dict[str, JsonValue]


def _is_none(value: object) -> bool:
    return value is None


OperatorId = Literal[
    "input.surface.rephrase",
    "input.surface.typing_noise",
    "input.surface.punctuation_noise",
    "input.surface.grammar_error",
    "input.surface.fragmented_syntax",
    "input.surface.disfluency_repeat",
    "input.style.terse",
    "input.style.verbose",
    "input.tone.angry",
    "input.tone.argumentative",
    "input.intent.self_correction",
]
AllowedChange = Literal[
    "surface_form_only",
    "declared_communication_form",
    "structured_self_correction",
]
OperatorApplicabilityProfile = Literal["broad", "conditional"]
OperatorGenerationMechanism = Literal["deterministic", "llm"]

_OPERATOR_PROMPT_NAMES: dict[OperatorId, str] = {
    "input.surface.rephrase": "augmentation.input.surface.rephrase",
    "input.surface.typing_noise": "augmentation.input.surface.typing_noise",
    "input.surface.punctuation_noise": "augmentation.input.surface.punctuation_noise",
    "input.surface.grammar_error": "augmentation.input.surface.grammar_error",
    "input.surface.fragmented_syntax": "augmentation.input.surface.fragmented_syntax",
    "input.surface.disfluency_repeat": "augmentation.input.surface.disfluency_repeat",
    "input.style.terse": "augmentation.input.style.terse",
    "input.style.verbose": "augmentation.input.style.verbose",
    "input.tone.angry": "augmentation.input.tone.angry",
    "input.tone.argumentative": "augmentation.input.tone.argumentative",
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
    generation_mechanism: OperatorGenerationMechanism
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
    generation_mechanism: OperatorGenerationMechanism,
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
        generation_mechanism=generation_mechanism,
        allowed_change=allowed_change,
        target_communication_kind=target_communication_kind,
        target_marker_required=target_marker_required,
        human_review_required=binding.requirements.human_review,
    )


_BUILTIN_OPERATORS = (
    _builtin_operator(
        operator_id="input.surface.rephrase",
        generation_mechanism="llm",
        allowed_change="surface_form_only",
    ),
    _builtin_operator(
        operator_id="input.surface.typing_noise",
        generation_mechanism="deterministic",
        allowed_change="declared_communication_form",
        target_communication_kind="typing_noise",
    ),
    _builtin_operator(
        operator_id="input.surface.punctuation_noise",
        generation_mechanism="deterministic",
        allowed_change="declared_communication_form",
        target_communication_kind="typing_noise",
    ),
    _builtin_operator(
        operator_id="input.surface.grammar_error",
        generation_mechanism="llm",
        allowed_change="declared_communication_form",
        target_communication_kind="grammar_error",
        target_marker_required=True,
    ),
    _builtin_operator(
        operator_id="input.surface.fragmented_syntax",
        generation_mechanism="llm",
        allowed_change="declared_communication_form",
        target_communication_kind="fragmented_syntax",
        target_marker_required=True,
    ),
    _builtin_operator(
        operator_id="input.surface.disfluency_repeat",
        generation_mechanism="llm",
        allowed_change="declared_communication_form",
        target_communication_kind="repetition",
    ),
    _builtin_operator(
        operator_id="input.style.terse",
        generation_mechanism="llm",
        allowed_change="declared_communication_form",
        target_communication_kind="terse",
    ),
    _builtin_operator(
        operator_id="input.style.verbose",
        generation_mechanism="llm",
        allowed_change="declared_communication_form",
        target_communication_kind="verbose",
    ),
    _builtin_operator(
        operator_id="input.tone.angry",
        generation_mechanism="llm",
        allowed_change="declared_communication_form",
        target_communication_kind="angry",
        target_marker_required=True,
    ),
    _builtin_operator(
        operator_id="input.tone.argumentative",
        generation_mechanism="llm",
        allowed_change="declared_communication_form",
        target_communication_kind="argumentative",
        target_marker_required=True,
    ),
    _builtin_operator(
        operator_id="input.intent.self_correction",
        generation_mechanism="deterministic",
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


class SemanticNormalizationAssessment(ULModel):
    normalizer_version: Literal["semantic-frame/1.0.0"] = "semantic-frame/1.0.0"
    applied_rules: tuple[
        Literal["ordered_list_factor_decomposition", "redundant_scalar_fulfills_elision"], ...
    ]
    verdict: Literal["equivalent", "different"]


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
    semantic_normalization: SemanticNormalizationAssessment | None = cast(Any, Field)(
        default=None, exclude_if=_is_none
    )
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
                self_correction_plan: _SelfCorrectionPlan | None = None
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
                    self_correction_plan = _self_correction_plan(record, source_frame)
                    if self_correction_plan is None:
                        skips.append(
                            DatasetAugmentationSkip(
                                source_interaction_id=record.id,
                                operator_id=operator.id,
                                operator_version=operator.version,
                                reason=operator.applicability_rule,
                            )
                        )
                        continue
                if operator.id == "input.surface.typing_noise":
                    rendered_input = _add_typing_noise(record, expected_input_frame, operator)
                elif operator.id == "input.surface.punctuation_noise":
                    rendered_input = _add_punctuation_noise(record, expected_input_frame, operator)
                elif operator.allowed_change == "structured_self_correction":
                    if self_correction_plan is None:
                        raise AssertionError("self-correction requires a selected factor")
                    rendered_input = _add_self_correction(
                        record,
                        operator,
                        self_correction_plan,
                    )
                else:
                    transformation_prompt_names = (_OPERATOR_PROMPT_NAMES[operator.id],)
                    rendered_input = await self._renderer.render(
                        record.raw_input, operator.instruction
                    )
                augmented_input = rendered_input.text
                if operator.generation_mechanism == "llm":
                    augmented_input = augmented_input.replace("—", " ")
                surface_footprint_reasons = _surface_footprint_reasons(
                    operator.id, record.raw_input, augmented_input
                )
                retry_reasons = surface_footprint_reasons
                if augmented_input == record.raw_input:
                    retry_reasons = (*retry_reasons, "renderer did not change the source input")
                if operator.generation_mechanism == "llm" and retry_reasons:
                    rendered_input = await self._renderer.render(
                        record.raw_input,
                        f"{operator.instruction}\n\n"
                        "Your previous attempt was invalid for these reasons: "
                        f"{'; '.join(retry_reasons)}. Return a corrected transformation that "
                        "satisfies the original instruction.",
                    )
                    augmented_input = rendered_input.text.replace("—", " ")
                    surface_footprint_reasons = _surface_footprint_reasons(
                        operator.id, record.raw_input, augmented_input
                    )
                renderer_metadata: dict[str, JsonValue] = {
                    **rendered_input.metadata,
                    "transformation_prompts": prompt_provenance(*transformation_prompt_names),
                }
                if operator.generation_mechanism == "llm" and retry_reasons:
                    renderer_metadata["retry_reasons"] = list(retry_reasons)
                if self_correction_plan is not None:
                    renderer_metadata["self_correction_grounding"] = self_correction_plan.grounding
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
                        if self_correction_plan is None:
                            raise AssertionError("self-correction requires a selected factor")
                        reparsed_frame = _normalize_planned_self_correction_frame(
                            reparsed_frame,
                            self_correction_plan.factor,
                            self_correction_plan.provisional_quote,
                            self_correction_plan.provisional_value,
                            record.raw_input,
                            augmented_input,
                        )
                        failure_reasons = list(
                            _structured_self_correction_difference_reasons(
                                expected_input_frame,
                                reparsed_frame,
                                self_correction_plan.factor,
                                self_correction_plan.provisional_quote,
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
                        and not surface_footprint_reasons
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
                                allowed_surface_change=_allowed_surface_change(operator.id),
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
                    if (
                        operator.target_communication_kind in _TONE_SAFETY_KINDS
                        and not surface_footprint_reasons
                        and not failure_reasons
                    ):
                        if self._equivalence_verifier is None:
                            failure_reasons = ["tone safety verifier is unavailable"]
                        else:
                            try:
                                equivalence_assessment = await self._equivalence_verifier.verify(
                                    record.raw_input,
                                    augmented_input,
                                    allowed_surface_change=_allowed_surface_change(operator.id),
                                )
                            except ValueError:
                                failure_reasons = ["tone safety validation failed"]
                            else:
                                if equivalence_assessment.verdict == "different":
                                    failure_reasons = [
                                        "tone safety check found a forbidden communication change"
                                    ]
                                elif equivalence_assessment.verdict == "uncertain":
                                    failure_reasons = ["tone safety check was uncertain"]
                failure_reasons.extend(surface_footprint_reasons)
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
                        semantic_normalization=(
                            _semantic_normalization_assessment(expected_input_frame, reparsed_frame)
                            if reparsed_frame is not None
                            else None
                        ),
                        passed=not failure_reasons,
                        failure_reasons=tuple(failure_reasons),
                    )
                )
        operator_positions = {
            (operator.id, operator.version): position
            for position, operator in enumerate(selected_operators)
        }
        source_positions = {record.id: position for position, record in enumerate(source_records)}
        candidates.sort(
            key=lambda candidate: (
                operator_positions[(candidate.operator_id, candidate.operator_version)],
                source_positions[candidate.source_interaction_id],
            )
        )
        skips.sort(
            key=lambda skip: (
                operator_positions[(skip.operator_id, skip.operator_version)],
                source_positions[skip.source_interaction_id],
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


def _semantic_normalization_assessment(
    expected: SemanticFrame, reparsed: SemanticFrame
) -> SemanticNormalizationAssessment | None:
    applied_rules: list[
        Literal["ordered_list_factor_decomposition", "redundant_scalar_fulfills_elision"]
    ] = []
    if any(
        isinstance(factor.value, list) for frame in (expected, reparsed) for factor in frame.factors
    ):
        applied_rules.append("ordered_list_factor_decomposition")
    if any(
        _is_redundant_scalar_fulfills(relation, frame)
        for frame in (expected, reparsed)
        for relation in frame.relations
    ):
        applied_rules.append("redundant_scalar_fulfills_elision")
    if not applied_rules:
        return None
    return SemanticNormalizationAssessment(
        applied_rules=tuple(applied_rules),
        verdict=(
            "equivalent" if not _semantic_difference_reasons(expected, reparsed) else "different"
        ),
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


_SELF_CORRECTION_FACTOR_KINDS = ("money", "number", "date_time", "duration", "enum")
_ACTION_OPERATION_WORDS = frozenset(
    {
        "add",
        "approve",
        "cancel",
        "create",
        "delete",
        "modify",
        "pay",
        "post",
        "reject",
        "remove",
        "schedule",
        "send",
        "set",
        "transfer",
        "update",
        "write",
    }
)
_OBSERVATION_OPERATION_WORDS = frozenset({"get", "list", "read", "retrieve", "search"})


def _self_correction_plan(
    record: InteractionRecord, frame: SemanticFrame
) -> _SelfCorrectionPlan | None:
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
    act_requests_by_factor_id = {
        factor.id: tuple(
            request
            for request in frame.request_units
            if request.mode == "act" and _factor_is_associated_with_request(factor, request)
        )
        for factor in frame.factors
    }
    plans: list[_SelfCorrectionPlan] = []
    for factor in frame.factors:
        requests = act_requests_by_factor_id[factor.id]
        source_quote = _unique_input_quote(factor)
        if (
            len(requests) != 1
            or factor.kind not in _SELF_CORRECTION_FACTOR_KINDS
            or factor.status not in {"confirmed", "explicit", "observed"}
            or not _is_self_correction_value(factor.value)
            or source_quote is None
            or record.raw_input.count(source_quote) != 1
        ):
            continue
        request = requests[0]
        recorded_action_match = _recorded_action_factor_match(record, frame, request, factor)
        is_grounded = recorded_action_match is not None or any(
            _json_key(outcome.fields.get(factor.role)) == _json_key(factor.value)
            for outcome in action_outcomes_by_request_id[request.id]
        )
        if not is_grounded:
            continue
        if factor.kind == "enum":
            if recorded_action_match is None:
                continue
            field_name = recorded_action_match[2]
            active_field_factor_ids = tuple(
                candidate.id
                for candidate in frame.factors
                if candidate.status != "superseded"
                and candidate.kind == "enum"
                and _factor_is_associated_with_request(candidate, request)
                and _semantic_role_matches_action_field(
                    candidate.role,
                    field_name,
                    request.predicate,
                )
            )
            if active_field_factor_ids != (factor.id,):
                continue
            grounded_plan = _grounded_prior_enum_plan(
                record, frame, request, factor, recorded_action_match
            )
            if grounded_plan is not None:
                plans.append(grounded_plan)
            continue
        provisional_quote = _planned_provisional_quote(factor, record.raw_input)
        if provisional_quote is None:
            continue
        plans.append(
            _SelfCorrectionPlan(
                factor=factor,
                provisional_quote=provisional_quote,
                provisional_value=_planned_provisional_value(factor, provisional_quote),
                grounding={"origin": "deterministic_numeric"},
            )
        )
    if not plans:
        return None
    factor_kind_priority = {
        kind: priority for priority, kind in enumerate(_SELF_CORRECTION_FACTOR_KINDS)
    }
    return min(
        plans,
        key=lambda plan: (
            factor_kind_priority[plan.factor.kind],
            frame.factors.index(plan.factor),
        ),
    )


def _recorded_action_factor_match(
    record: InteractionRecord,
    frame: SemanticFrame,
    request: RequestUnit,
    factor: SemanticFactor,
) -> tuple[int, Mapping[str, JsonValue], str] | None:
    action_records = _recorded_action_records(record)
    request_identifier_factors = _request_identifier_factors(frame, request)
    if not request_identifier_factors:
        return None
    matches: list[tuple[int, Mapping[str, JsonValue], str]] = []
    for action_index, action in enumerate(action_records):
        if not _action_matches_request_operation(str(action["action"]), request.predicate):
            continue
        if not _action_contains_exact_identifiers(
            action, frame, request, request_identifier_factors
        ):
            continue
        matching_fields = tuple(
            field_name
            for field_name, value in action.items()
            if not _is_action_identifier_field(field_name)
            and field_name != "action"
            and _semantic_role_matches_action_field(factor.role, field_name, request.predicate)
            and _action_matches_request_resource(str(action["action"]), frame, request, field_name)
            and not isinstance(value, (dict, list, bool))
            and _self_correction_values_equal(value, factor.value, factor_kind=factor.kind)
        )
        if len(matching_fields) == 1:
            matches.append((action_index, action, matching_fields[0]))
    return matches[0] if len(matches) == 1 else None


def _recorded_action_records(
    record: InteractionRecord,
) -> tuple[Mapping[str, JsonValue], ...]:
    observed_output = record.raw_observed_output
    if isinstance(observed_output, dict):
        if isinstance(observed_output.get("action"), str):
            return (observed_output,)
        raw_actions = observed_output.get("actions")
        if isinstance(raw_actions, list) and len(raw_actions) <= 10_000:
            return tuple(
                action
                for action in raw_actions
                if isinstance(action, dict) and isinstance(action.get("action"), str)
            )
    return ()


def _request_identifier_factors(
    frame: SemanticFrame, request: RequestUnit
) -> tuple[SemanticFactor, ...]:
    return tuple(
        identifier_factor
        for identifier_factor in frame.factors
        if _factor_is_associated_with_request(identifier_factor, request)
        and identifier_factor.kind == "identifier"
        and isinstance(identifier_factor.value, (str, int, float))
        and not isinstance(identifier_factor.value, bool)
    )


def _factor_is_associated_with_request(factor: SemanticFactor, request: RequestUnit) -> bool:
    if request.factor_ids:
        return factor.id in request.factor_ids
    factor_quote = _unique_input_quote(factor)
    if factor_quote is None:
        return False
    return any(
        evidence.source == "input"
        and evidence.text_quote is not None
        and factor_quote in evidence.text_quote
        for evidence in request.evidence
    )


def _is_action_identifier_field(field_name: str) -> bool:
    return field_name == "id" or field_name.endswith("_id")


def _action_contains_exact_identifiers(
    action: Mapping[str, JsonValue],
    frame: SemanticFrame,
    request: RequestUnit,
    request_identifier_factors: tuple[SemanticFactor, ...],
) -> bool:
    action_identifiers = tuple(
        (field_name, value)
        for field_name, value in action.items()
        if _is_action_identifier_field(field_name)
        and isinstance(value, (str, int, float))
        and not isinstance(value, bool)
    )
    if len(action_identifiers) != len(request_identifier_factors):
        return False
    factors_match_once = all(
        sum(
            _identifier_factor_matches_action_field(
                factor, field_name, str(action["action"]), frame, request
            )
            and type(factor.value) is type(value)
            and factor.value == value
            for field_name, value in action_identifiers
        )
        == 1
        for factor in request_identifier_factors
    )
    fields_match_once = all(
        sum(
            _identifier_factor_matches_action_field(
                factor, field_name, str(action["action"]), frame, request
            )
            and type(factor.value) is type(value)
            and factor.value == value
            for factor in request_identifier_factors
        )
        == 1
        for field_name, value in action_identifiers
    )
    return factors_match_once and fields_match_once


def _identifier_factor_matches_action_field(
    factor: SemanticFactor,
    field_name: str,
    action_name: str,
    frame: SemanticFrame,
    request: RequestUnit,
) -> bool:
    field_tokens = _semantic_name_tokens(field_name, ignored={"id", "identifier"})
    if not field_tokens:
        return field_name == "id" and factor.role in {"id", "identifier"}
    role_tokens = _semantic_name_tokens(factor.role, ignored={"id", "identifier", "object"})
    if role_tokens:
        return role_tokens == field_tokens
    request_tokens = _request_resource_tokens(frame, request)
    action_resource_tokens = set(_action_resource_key(action_name))
    return set(field_tokens) <= request_tokens and set(field_tokens) <= action_resource_tokens


def _semantic_role_matches_action_field(role: str, field_name: str, request_predicate: str) -> bool:
    ignored = {
        "current",
        "desired",
        "final",
        "from",
        "object",
        "name",
        "new",
        "requested",
        "target",
        "to",
        "value",
    }
    role_tokens = _semantic_name_tokens(role, ignored=ignored)
    field_tokens = _semantic_name_tokens(field_name, ignored=ignored)
    if not field_tokens:
        return False
    if role_tokens:
        return role_tokens == field_tokens
    request_tokens = set(_semantic_name_tokens(request_predicate, ignored=_ACTION_OPERATION_WORDS))
    return set(field_tokens) <= request_tokens


def _semantic_name_tokens(name: str, *, ignored: Collection[str]) -> tuple[str, ...]:
    return tuple(
        token for token in re.findall(r"[a-z0-9]+", name.casefold()) if token not in ignored
    )


def _request_resource_tokens(
    frame: SemanticFrame,
    request: RequestUnit,
    *,
    ignored: Collection[str] = (),
) -> set[str]:
    predicate_tokens = set(
        _semantic_name_tokens(request.predicate, ignored=_ACTION_OPERATION_WORDS.union(ignored))
    )
    tokens = set(predicate_tokens)
    for factor in frame.factors:
        if not _factor_is_associated_with_request(factor, request):
            continue
        if factor.kind == "entity" and isinstance(factor.value, str):
            entity_tokens = set(_semantic_name_tokens(factor.value, ignored=ignored))
            tokens.update(
                entity_tokens
                if request.factor_ids
                else entity_tokens.intersection(predicate_tokens)
            )
        if factor.kind == "identifier":
            identifier_role_tokens = set(
                _semantic_name_tokens(factor.role, ignored={"id", "identifier", "object", *ignored})
            )
            tokens.update(
                identifier_role_tokens
                if request.factor_ids
                else identifier_role_tokens.intersection(predicate_tokens)
            )
    return tokens


def _action_matches_request_resource(
    action_name: str,
    frame: SemanticFrame,
    request: RequestUnit,
    semantic_field_name: str,
) -> bool:
    semantic_field_tokens = set(
        _semantic_name_tokens(semantic_field_name, ignored={"name", "value"})
    )
    action_resource = tuple(
        token for token in _action_resource_key(action_name) if token not in semantic_field_tokens
    )
    request_resource_tokens = _request_resource_tokens(
        frame, request, ignored=semantic_field_tokens
    )
    return bool(action_resource) and action_resource[-1] in request_resource_tokens


def _grounded_prior_enum_plan(
    record: InteractionRecord,
    frame: SemanticFrame,
    request: RequestUnit,
    factor: SemanticFactor,
    final_match: tuple[int, Mapping[str, JsonValue], str],
) -> _SelfCorrectionPlan | None:
    if not isinstance(factor.value, str):
        return None
    final_action_index, final_action, field_name = final_match
    final_identifiers = tuple(
        sorted(
            (name, value)
            for name, value in final_action.items()
            if _is_action_identifier_field(name)
            and isinstance(value, (str, int, float))
            and not isinstance(value, bool)
        )
    )
    if not final_identifiers:
        return None
    final_resource = _action_resource_key(str(final_action["action"]))
    if not final_resource:
        return None
    for later_action in _recorded_action_records(record)[final_action_index + 1 :]:
        later_action_name = str(later_action["action"])
        later_identifiers = tuple(
            sorted(
                (name, value)
                for name, value in later_action.items()
                if _is_action_identifier_field(name)
                and isinstance(value, (str, int, float))
                and not isinstance(value, bool)
            )
        )
        later_action_tokens = set(re.findall(r"[a-z0-9]+", later_action_name.casefold()))
        if (
            later_identifiers == final_identifiers
            and _action_resource_key(later_action_name) == final_resource
            and field_name in later_action
            and later_action_tokens.intersection(_ACTION_OPERATION_WORDS)
        ):
            return None
    prior_observations: list[tuple[int, str]] = []
    prior_matches: list[tuple[int, str]] = []
    for action_index, action in enumerate(_recorded_action_records(record)[:final_action_index]):
        if not _is_observation_action(str(action["action"])):
            continue
        if _action_resource_key(str(action["action"])) != final_resource:
            continue
        prior_identifiers = tuple(
            sorted(
                (name, value)
                for name, value in action.items()
                if _is_action_identifier_field(name)
                and isinstance(value, (str, int, float))
                and not isinstance(value, bool)
            )
        )
        prior_value = action.get(field_name)
        if (
            prior_identifiers == final_identifiers
            and isinstance(prior_value, str)
            and prior_value != factor.value
        ):
            prior_observations.append((action_index, prior_value))
        if (
            prior_identifiers == final_identifiers
            and isinstance(prior_value, str)
            and prior_value != factor.value
            and _safe_enum_provisional_quote(prior_value)
            and record.raw_input.count(prior_value) == 1
        ):
            prior_matches.append((action_index, prior_value))
    if len(prior_observations) != 1 or len(prior_matches) != 1:
        return None
    prior_action_index, prior_value = prior_matches[0]
    return _SelfCorrectionPlan(
        factor=factor,
        provisional_quote=prior_value,
        provisional_value=prior_value,
        grounding={
            "origin": "prior_observed_action",
            "field_name": field_name,
            "source_occurrences": 1,
            "prior_action_index": prior_action_index,
            "final_action_index": final_action_index,
            "identifier_fields": [name for name, _ in final_identifiers],
        },
    )


def _action_resource_key(action_name: str) -> tuple[str, ...]:
    operation_words = _ACTION_OPERATION_WORDS | _OBSERVATION_OPERATION_WORDS
    return tuple(
        token
        for token in re.findall(r"[a-z0-9]+", action_name.casefold())
        if token not in operation_words
    )


def _is_observation_action(action_name: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", action_name.casefold()))
    observation_operations = {token for token in tokens if token in _OBSERVATION_OPERATION_WORDS}
    return len(observation_operations) == 1 and not tokens.intersection(_ACTION_OPERATION_WORDS)


def _safe_enum_provisional_quote(value: str) -> bool:
    return re.fullmatch(r"[\w][\w /&().'-]{0,63}", value) is not None


def _action_matches_request_operation(action_name: str, request_predicate: str) -> bool:
    action_operations = {
        token
        for token in re.findall(r"[a-z0-9]+", action_name.casefold())
        if token in _ACTION_OPERATION_WORDS
    }
    request_operations = {
        token
        for token in re.findall(r"[a-z0-9]+", request_predicate.casefold())
        if token in _ACTION_OPERATION_WORDS
    }
    return len(action_operations) == 1 and action_operations <= request_operations


def _self_correction_values_equal(first: JsonValue, second: JsonValue, *, factor_kind: str) -> bool:
    if factor_kind in {"money", "number"}:
        first_number = _self_correction_numeric_value(first, factor_kind=factor_kind)
        second_number = _self_correction_numeric_value(second, factor_kind=factor_kind)
        if first_number is not None and second_number is not None:
            return first_number == second_number
    if (
        isinstance(first, (int, float))
        and not isinstance(first, bool)
        and isinstance(second, (int, float))
        and not isinstance(second, bool)
    ):
        return first == second
    return type(first) is type(second) and first == second


def _self_correction_numeric_value(value: JsonValue, *, factor_kind: str) -> Decimal | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric_value = Decimal(str(value))
        return numeric_value if numeric_value.is_finite() else None
    if not isinstance(value, str) or len(value) > 128:
        return None
    if factor_kind == "money":
        match = re.fullmatch(
            r"\s*(?:[A-Za-z]{3}\s*)?[$€£¥]?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*(?:[A-Za-z]{3})?\s*",
            value,
        )
    else:
        match = re.fullmatch(r"\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*", value)
    if match is None:
        return None
    numeric_value = Decimal(match.group(1).replace(",", ""))
    return numeric_value if numeric_value.is_finite() else None


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
    if not _self_correction_values_equal(
        final_factor.value,
        selected_source_factor.value,
        factor_kind=selected_source_factor.kind,
    ):
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
    expected_provisional_source_count = 1 if selected_source_factor.kind == "enum" else 0
    if source_input.count(provisional_quote) != expected_provisional_source_count:
        return ("provisional value has unsupported source occurrences",)
    if (
        augmented_input.count(provisional_quote) != expected_provisional_source_count + 1
        or augmented_input.count(final_quote) != 1
    ):
        return ("provisional and final values must each appear exactly once",)
    final_start = augmented_input.index(final_quote)
    provisional_start = augmented_input.rfind(provisional_quote, 0, final_start)
    if provisional_start < 0:
        return ("provisional value must appear before the final value",)
    between_values = augmented_input[provisional_start + len(provisional_quote) : final_start]
    if (
        re.fullmatch(
            r"\s*[,;:.\u2013\u2014-]*\s*(?:actually|correction|i\s+mean|rather|sorry)"
            r"\s*[,;:.\u2013\u2014-]*\s*",
            between_values,
            flags=re.IGNORECASE,
        )
        is None
    ):
        return ("correction language must contain only a recognized correction cue",)
    if not all(
        _element_evidence_spans_values(element, provisional_quote, final_quote)
        for element in (correction_act, correction_relation)
    ):
        return ("correction act and relation evidence must span both values in order",)
    final_end = augmented_input.index(final_quote) + len(final_quote)
    reconstructed_source_input = (
        augmented_input[:provisional_start] + final_quote + augmented_input[final_end:]
    )
    if reconstructed_source_input != source_input:
        return ("self-correction must preserve all source text outside the correction",)
    return ()


def _normalize_planned_self_correction_frame(
    frame: SemanticFrame,
    selected_source_factor: SemanticFactor,
    planned_provisional_quote: str,
    planned_provisional_value: JsonValue,
    source_input: str,
    augmented_input: str,
) -> SemanticFrame:
    correction_acts = tuple(
        act for act in frame.communication_acts if act.kind == "self_correction"
    )
    if len(correction_acts) > 1 or any(
        relation.kind == "superseded_by" for relation in frame.relations
    ):
        return frame
    correction_act = correction_acts[0] if correction_acts else None
    if correction_act is not None and (correction_act.attributes or correction_act.factor_ids):
        return frame
    selected_source_quote = _unique_input_quote(selected_source_factor)
    if (
        selected_source_quote is None
        or augmented_input.count(planned_provisional_quote)
        != source_input.count(planned_provisional_quote) + 1
        or augmented_input.count(selected_source_quote) != 1
        or augmented_input.index(planned_provisional_quote)
        >= augmented_input.index(selected_source_quote)
    ):
        return frame
    final_factors = tuple(
        factor
        for factor in frame.factors
        if (factor.kind, factor.role) == (selected_source_factor.kind, selected_source_factor.role)
        and _self_correction_values_equal(
            factor.value,
            selected_source_factor.value,
            factor_kind=selected_source_factor.kind,
        )
        and _unique_input_quote(factor) == selected_source_quote
        and sum(
            request.mode == "act" and _factor_is_associated_with_request(factor, request)
            for request in frame.request_units
        )
        == 1
    )
    if len(final_factors) != 1:
        return frame
    if planned_provisional_value is None:
        return frame
    final_factor = final_factors[0]
    existing_provisional_factors = tuple(
        factor
        for factor in frame.factors
        if factor.status == "superseded"
        and (factor.kind, factor.role) == (final_factor.kind, final_factor.role)
        and _self_correction_values_equal(
            factor.value,
            planned_provisional_value,
            factor_kind=selected_source_factor.kind,
        )
        and _unique_input_quote(factor) == planned_provisional_quote
        and not any(factor.id in request.factor_ids for request in frame.request_units)
    )
    if len(existing_provisional_factors) > 1:
        return frame
    correction_act_id = (
        correction_act.id if correction_act is not None else f"{final_factor.id}:self-correction"
    )
    provisional_id = (
        existing_provisional_factors[0].id
        if existing_provisional_factors
        else f"{correction_act_id}:provisional"
    )
    relation_id = f"{correction_act_id}:superseded-by"
    existing_ids = {
        *(request.id for request in frame.request_units),
        *(factor.id for factor in frame.factors),
        *(relation.id for relation in frame.relations),
        *(act.id for act in frame.communication_acts),
        *(outcome.id for outcome in frame.outcomes),
    }
    new_ids = {relation_id}
    if correction_act is None:
        new_ids.add(correction_act_id)
    if not existing_provisional_factors:
        new_ids.add(provisional_id)
    if new_ids & existing_ids:
        return frame
    final_start = augmented_input.index(selected_source_quote)
    repair_start = augmented_input.rfind(planned_provisional_quote, 0, final_start)
    if repair_start < 0:
        return frame
    repair_end = final_start + len(selected_source_quote)
    repair_evidence = (
        EvidenceReference(
            source="input",
            json_pointer="/raw_input",
            text_quote=augmented_input[repair_start:repair_end],
        ),
    )
    confidence = (
        correction_act.confidence if correction_act is not None else final_factor.confidence
    )
    provisional_factor = (
        existing_provisional_factors[0]
        if existing_provisional_factors
        else SemanticFactor(
            id=provisional_id,
            evidence=(
                EvidenceReference(
                    source="input",
                    json_pointer="/raw_input",
                    text_quote=planned_provisional_quote,
                ),
            ),
            confidence=confidence,
            status="superseded",
            kind=selected_source_factor.kind,
            role=selected_source_factor.role,
            value=planned_provisional_value,
        )
    )
    normalized_correction_act = CommunicationAct(
        id=correction_act_id,
        evidence=repair_evidence,
        confidence=confidence,
        status=correction_act.status if correction_act is not None else "explicit",
        kind="self_correction",
        factor_ids=(provisional_factor.id, final_factor.id),
    )
    correction_relation = SemanticRelation(
        id=relation_id,
        evidence=repair_evidence,
        confidence=confidence,
        status=normalized_correction_act.status,
        kind="superseded_by",
        source_ids=(provisional_factor.id,),
        target_ids=(final_factor.id,),
    )
    return frame.model_copy(
        update={
            "factors": (
                frame.factors
                if existing_provisional_factors
                else (*frame.factors, provisional_factor)
            ),
            "relations": (*frame.relations, correction_relation),
            "communication_acts": (
                tuple(
                    normalized_correction_act if act.id == correction_act_id else act
                    for act in frame.communication_acts
                )
                if correction_act is not None
                else (*frame.communication_acts, normalized_correction_act)
            ),
        }
    )


def _planned_provisional_value(
    selected_source_factor: SemanticFactor, planned_provisional_quote: str
) -> JsonValue:
    if selected_source_factor.kind not in {"money", "number"}:
        return None
    numeric_matches = re.findall(
        r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?![\w.])", planned_provisional_quote
    )
    if len(numeric_matches) != 1:
        return None
    provisional_number = Decimal(numeric_matches[0].replace(",", ""))
    if provisional_number == provisional_number.to_integral_value():
        return int(provisional_number)
    return float(provisional_number)


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
    changed_word_count = _changed_word_count(source_input, augmented_input)
    if operator_id == "input.surface.rephrase" and changed_word_count < max(
        2, source_word_count // 4
    ):
        return ("rendered input does not change enough wording",)
    if operator_id == "input.surface.grammar_error" and changed_word_count < 2:
        return ("rendered input must contain at least two visible word errors",)
    if (
        operator_id == "input.surface.fragmented_syntax"
        and len(tuple(part for part in re.split(r"[.;\n]+", augmented_input) if part.strip())) < 2
    ):
        return ("rendered input must contain at least two natural fragments",)
    if operator_id == "input.surface.punctuation_noise" and not _is_disruptive_punctuation_noise(
        source_input, augmented_input
    ):
        return ("rendered input must add mixed exclamation, period, newline, and spacing noise",)
    if operator_id == "input.style.terse" and augmented_word_count * 5 > source_word_count * 4:
        return ("rendered input is not visibly shorter than the source",)
    if operator_id == "input.style.verbose" and not (
        source_word_count * 2 <= augmented_word_count
        and augmented_word_count * 2 <= source_word_count * 7
    ):
        return ("rendered input is not between 2 and 3.5 times the source length",)
    if operator_id == "input.surface.disfluency_repeat":
        source_repetition_count = _immediate_phrase_repetition_count(source_input)
        augmented_repetition_count = _immediate_phrase_repetition_count(augmented_input)
        if augmented_repetition_count != source_repetition_count + 1:
            return ("rendered input must repeat exactly one short phrase immediately",)
    if (
        operator_id == "input.tone.angry"
        and re.search(
            r"(?i)\b(?:fuck\w*|shit\w*|stupid|idiot|moron|damn\w*|asshole|useless|dumb)\b",
            augmented_input,
        )
        is None
    ):
        return ("rendered input must sound unmistakably angry and hostile",)
    if (
        operator_id == "input.tone.argumentative"
        and re.search(
            r"(?i)\b(?:last time|again|wrong|stop|ever|once in your life|get anything right|"
            r"do one thing right|can(?:not|'t) you (?:get|do|ever))\b",
            augmented_input,
        )
        is None
    ):
        return ("rendered input must accuse or challenge an agent that keeps failing",)
    return ()


def _changed_word_count(source_input: str, augmented_input: str) -> int:
    source_words = Counter(re.findall(r"\w+", source_input.casefold(), flags=re.UNICODE))
    augmented_words = Counter(re.findall(r"\w+", augmented_input.casefold(), flags=re.UNICODE))
    shared_word_count = sum((source_words & augmented_words).values())
    return max(sum(source_words.values()), sum(augmented_words.values())) - shared_word_count


def _allowed_surface_change(operator_id: OperatorId) -> SemanticAllowedSurfaceChange:
    if operator_id == "input.surface.punctuation_noise":
        return "unprotected_punctuation_noise"
    if operator_id == "input.tone.angry":
        return "hostile_angry_tone"
    if operator_id == "input.tone.argumentative":
        return "hostile_argumentative_tone"
    return "none"


def _add_typing_noise(
    record: InteractionRecord,
    frame: SemanticFrame,
    operator: DatasetAugmentationOperator,
) -> RenderedUserInput:
    seed = int.from_bytes(
        hashlib.sha256(f"{record.id}\0{operator.id}\0{operator.version}".encode()).digest()[:4],
        "big",
    )
    protected_spans = _protected_input_spans(record.raw_input, frame)
    eligible_words: list[tuple[int, int, str]] = []
    for match in re.finditer(r"[A-Za-z]+", record.raw_input):
        word = match.group()
        if len(word) < 3 or any(
            _position_is_protected(position, protected_spans)
            for position in range(match.start(), match.end())
        ):
            continue
        eligible_words.append((match.start(), match.end(), word))
    if not eligible_words:
        rendered_text = record.raw_input
        edit_count = 0
    else:
        ranked_words = sorted(
            eligible_words,
            key=lambda item: hashlib.sha256(f"{seed}\0{item[0]}".encode()).digest(),
        )
        edit_count = 4 + seed % 2
        mutations = {start: word for start, _end, word in ranked_words}
        for edit_index in range(edit_count):
            start, _end, _word = ranked_words[edit_index % len(ranked_words)]
            word = mutations[start]
            operation = (seed + edit_index) % 3
            position = 1 + (seed + edit_index) % max(1, len(word) - 2)
            if operation == 0 and len(word) >= 3:
                swap_position = min(position, len(word) - 2)
                characters = list(word)
                characters[swap_position], characters[swap_position + 1] = (
                    characters[swap_position + 1],
                    characters[swap_position],
                )
                mutated = "".join(characters)
                if mutated == word:
                    mutated = f"{word[:swap_position]}{word[swap_position + 1 :]}"
            elif operation == 1 and len(word) > 3:
                mutated = f"{word[:position]}{word[position + 1 :]}"
            else:
                mutated = f"{word[:position]}{word[position]}{word[position:]}"
            mutations[start] = mutated
        pieces: list[str] = []
        cursor = 0
        for start, end, _word in eligible_words:
            pieces.extend((record.raw_input[cursor:start], mutations[start]))
            cursor = end
        pieces.append(record.raw_input[cursor:])
        rendered_text = "".join(pieces)
    return RenderedUserInput(
        text=rendered_text,
        metadata={
            "renderer": "deterministic",
            "algorithm": "four_or_five_unprotected_typing_edits",
            "seed": seed,
            "edit_count": edit_count,
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


def _add_self_correction(
    record: InteractionRecord,
    operator: DatasetAugmentationOperator,
    plan: _SelfCorrectionPlan,
) -> RenderedUserInput:
    correction_quote = _unique_input_quote(plan.factor)
    if correction_quote is None or record.raw_input.count(correction_quote) != 1:
        raise AssertionError("selected correction factor requires one unique quote")
    correction = f"{plan.provisional_quote}, sorry {correction_quote}"
    return RenderedUserInput(
        text=record.raw_input.replace(correction_quote, correction),
        metadata=_deterministic_renderer_metadata(
            record,
            operator,
            "grounded_self_correction_insertion",
        ),
    )


def _add_punctuation_noise(
    record: InteractionRecord,
    frame: SemanticFrame,
    operator: DatasetAugmentationOperator,
) -> RenderedUserInput:
    seed = int.from_bytes(
        hashlib.sha256(f"{record.id}\0{operator.id}\0{operator.version}".encode()).digest()[:4],
        "big",
    )
    randomizer = random.Random(seed)
    protected_spans = _protected_input_spans(record.raw_input, frame)
    insertion_points = list(
        index
        for index, character in enumerate(record.raw_input)
        if character.isspace()
        and not _position_is_protected(index, protected_spans)
        and not _space_splits_title_words(record.raw_input, index)
    )
    if not insertion_points:
        fallback = _punctuation_insertion(record.raw_input, frame)
        if fallback is None:
            raise AssertionError("punctuation noise requires an unprotected insertion point")
        insertion_points = [fallback[0]]
    randomizer.shuffle(insertion_points)
    noise = ["!", ".", "\n", " ", "!", ".", "\n", " ", "!", "."]
    randomizer.shuffle(noise)
    insertions: dict[int, list[str]] = {}
    for index, character in enumerate(noise):
        position = insertion_points[index % len(insertion_points)]
        insertions.setdefault(position, []).append(character)
    rendered_text = "".join(
        f"{''.join(insertions.get(index, ()))}{character}"
        for index, character in enumerate(record.raw_input)
    )
    rendered_text = f"{rendered_text}{''.join(insertions.get(len(record.raw_input), ()))}"
    return RenderedUserInput(
        text=rendered_text,
        metadata=_deterministic_renderer_metadata(
            record, operator, "seeded_mixed_punctuation_spacing_noise"
        ),
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


def _space_splits_title_words(text: str, index: int) -> bool:
    left_word = re.search(r"[^\W\d_]+$", text[:index], flags=re.UNICODE)
    right_word = re.match(r"\s*([^\W\d_]+)", text[index + 1 :], flags=re.UNICODE)
    return (
        left_word is not None
        and right_word is not None
        and left_word.group()[0].isupper()
        and right_word.group(1)[0].isupper()
    )


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


def _is_disruptive_punctuation_noise(source_input: str, augmented_input: str) -> bool:
    source_index = 0
    inserted: list[str] = []
    for character in augmented_input:
        if source_index < len(source_input) and character == source_input[source_index]:
            source_index += 1
        elif character in {"!", "."} or character.isspace():
            inserted.append(character)
        else:
            return False
    return (
        source_index == len(source_input)
        and len(inserted) >= 10
        and "!" in inserted
        and "." in inserted
        and "\n" in inserted
        and " " in inserted
    )


def _immediate_phrase_repetition_count(text: str) -> int:
    words = tuple(word.casefold() for word in re.findall(r"\w+", text, flags=re.UNICODE))
    return sum(
        words[index : index + phrase_length]
        == words[index + phrase_length : index + (2 * phrase_length)]
        for phrase_length in (2, 3)
        for index in range(len(words) - (2 * phrase_length) + 1)
    )


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
    retained_relations = tuple(
        relation
        for relation in frame.relations
        if not _is_redundant_scalar_fulfills(relation, frame)
    )
    decomposed_relation_endpoints = 0
    for relation in retained_relations:
        decomposed_relation_endpoints += _relation_expansion_endpoint_count(
            relation.source_ids,
            relation.target_ids,
            endpoint_semantics,
            list_factor_ids,
        )
        if decomposed_relation_endpoints > _MAX_DECOMPOSED_RELATION_ENDPOINTS:
            break
    expand_relations = decomposed_relation_endpoints <= _MAX_DECOMPOSED_RELATION_ENDPOINTS
    canonical_relations: list[tuple[object, ...]] = []
    for relation in retained_relations:
        if expand_relations:
            canonical_relations.extend(
                _expanded_relation_semantics(
                    relation.kind,
                    relation.source_ids,
                    relation.target_ids,
                    endpoint_semantics,
                    list_factor_ids,
                )
            )
        else:
            canonical_relations.append(
                _unexpanded_relation_semantics(
                    relation.kind,
                    relation.source_ids,
                    relation.target_ids,
                    endpoint_semantics,
                )
            )
    relation_semantics = tuple(sorted(canonical_relations))
    return (
        tuple(sorted(part for parts in factor_semantics.values() for part in parts)),
        tuple(request_semantics[request.id] for request in frame.request_units),
        relation_semantics,
        tuple(communication_semantics[act.id] for act in frame.communication_acts),
    )


def _is_redundant_scalar_fulfills(relation: SemanticRelation, frame: SemanticFrame) -> bool:
    if relation.kind != "fulfills" or len(relation.source_ids) != 1:
        return False
    request_factor_ids = {request.id: set(request.factor_ids) for request in frame.request_units}
    referenced_factor_ids = request_factor_ids.get(relation.source_ids[0])
    if referenced_factor_ids is None or not set(relation.target_ids) <= referenced_factor_ids:
        return False
    factors_by_id = {factor.id: factor for factor in frame.factors}
    return bool(relation.target_ids) and all(
        target_id in factors_by_id and not isinstance(factors_by_id[target_id].value, list)
        for target_id in relation.target_ids
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
            "expanded",
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


def _relation_expansion_endpoint_count(
    source_ids: tuple[str, ...],
    target_ids: tuple[str, ...],
    endpoint_semantics: Mapping[str, tuple[tuple[object, ...], ...]],
    list_factor_ids: set[str],
) -> int:
    decomposed_endpoint_ids = tuple(
        item for item in (*source_ids, *target_ids) if item in list_factor_ids
    )
    if len(decomposed_endpoint_ids) != 1:
        return 0
    decomposed_endpoint_id = decomposed_endpoint_ids[0]
    expansion_count = len(endpoint_semantics[decomposed_endpoint_id])
    endpoints_per_expansion = sum(
        1 if endpoint_id == decomposed_endpoint_id else len(endpoint_semantics[endpoint_id])
        for endpoint_id in (*source_ids, *target_ids)
    )
    return expansion_count * endpoints_per_expansion


def _unexpanded_relation_semantics(
    kind: str,
    source_ids: tuple[str, ...],
    target_ids: tuple[str, ...],
    endpoint_semantics: Mapping[str, tuple[tuple[object, ...], ...]],
) -> tuple[object, ...]:
    return (
        "unexpanded",
        kind,
        tuple(sorted(endpoint_semantics[endpoint_id] for endpoint_id in source_ids)),
        tuple(sorted(endpoint_semantics[endpoint_id] for endpoint_id in target_ids)),
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


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
