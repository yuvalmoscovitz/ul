from __future__ import annotations

from collections.abc import Iterable
from itertools import islice
from typing import Annotated, Self

from pydantic import AfterValidator, Field, WithJsonSchema, model_validator
from ul_core.augmentations.definitions import (
    AugmentationApplicabilityProfile,
    AugmentationBinding,
    BuiltinAugmentationSpec,
    DatasetAllowedChange,
    DatasetGenerationMechanism,
    DatasetVariationRuntime,
    builtin_augmentation_catalog,
)
from ul_core.models import ULModel
from ul_core.prompts import PromptManager


def _dataset_binding(specification: BuiltinAugmentationSpec) -> AugmentationBinding:
    return next(
        binding for binding in specification.bindings if binding.mode == "dataset_variation"
    )


def _dataset_runtime(specification: BuiltinAugmentationSpec) -> DatasetVariationRuntime:
    runtime = _dataset_binding(specification).dataset_runtime
    if runtime is None:
        raise AssertionError("dataset variation binding requires runtime metadata")
    return runtime


_DATASET_SPECIFICATIONS = tuple(
    sorted(
        builtin_augmentation_catalog().list(mode="dataset_variation", latest_only=False),
        key=lambda specification: (
            _dataset_runtime(specification).order,
            specification.ref.id,
            specification.ref.version,
        ),
    )
)
_DATASET_OPERATOR_IDS = tuple(
    dict.fromkeys(specification.ref.id for specification in _DATASET_SPECIFICATIONS)
)


def _validate_operator_id(operator_id: str) -> str:
    if operator_id not in _DATASET_OPERATOR_IDS:
        raise ValueError("unknown dataset augmentation reference")
    return operator_id


def _version_tuple(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)


OperatorId = Annotated[
    str,
    AfterValidator(_validate_operator_id),
    WithJsonSchema({"type": "string", "enum": list(_DATASET_OPERATOR_IDS)}),
]
AllowedChange = DatasetAllowedChange
OperatorApplicabilityProfile = AugmentationApplicabilityProfile
OperatorGenerationMechanism = DatasetGenerationMechanism


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
    human_review_required: bool = False

    @model_validator(mode="after")
    def validate_change_contract(self) -> Self:
        requires_target = self.allowed_change in {
            "declared_communication_form",
            "structured_self_correction",
        }
        if requires_target != (self.target_communication_kind is not None):
            raise ValueError("target communication kind must match the allowed change")
        return self


def _specification_prompt_name(specification: BuiltinAugmentationSpec) -> str:
    return _dataset_runtime(specification).prompt_name or f"augmentation.{specification.ref.id}"


def _builtin_operator(specification: BuiltinAugmentationSpec) -> DatasetAugmentationOperator:
    binding = _dataset_binding(specification)
    runtime = _dataset_runtime(specification)
    return DatasetAugmentationOperator(
        id=specification.ref.id,
        version=specification.ref.version,
        instruction=(
            runtime.instruction
            or PromptManager.instance().get_prompt(_specification_prompt_name(specification))
        ),
        applicability_profile=specification.applicability_profile,
        applicability_rule=specification.applicability_rule,
        generation_mechanism=runtime.generation_mechanism,
        allowed_change=runtime.allowed_change,
        target_communication_kind=runtime.target_communication_kind,
        human_review_required=binding.requirements.human_review,
    )


def _validate_versioned_prompt_identities(
    specifications: tuple[BuiltinAugmentationSpec, ...],
) -> None:
    specifications_by_id: dict[str, list[BuiltinAugmentationSpec]] = {}
    for specification in specifications:
        specifications_by_id.setdefault(specification.ref.id, []).append(specification)
    for versioned_specifications in specifications_by_id.values():
        if len(versioned_specifications) > 1 and any(
            _dataset_runtime(specification).prompt_name is None
            for specification in versioned_specifications
        ):
            raise ValueError("versioned dataset augmentations require explicit prompt identities")


def _latest_specifications(
    specifications: tuple[BuiltinAugmentationSpec, ...],
) -> tuple[BuiltinAugmentationSpec, ...]:
    latest_by_id: dict[str, BuiltinAugmentationSpec] = {}
    for specification in specifications:
        previous = latest_by_id.get(specification.ref.id)
        if previous is None or _version_tuple(specification.ref.version) > _version_tuple(
            previous.ref.version
        ):
            latest_by_id[specification.ref.id] = specification
    return tuple(
        sorted(
            latest_by_id.values(),
            key=lambda specification: (
                _dataset_runtime(specification).order,
                specification.ref.id,
                specification.ref.version,
            ),
        )
    )


def _operators_from_specifications(
    specifications: tuple[BuiltinAugmentationSpec, ...],
    *,
    latest_only: bool,
) -> tuple[DatasetAugmentationOperator, ...]:
    _validate_versioned_prompt_identities(specifications)
    selected_specifications = (
        _latest_specifications(specifications) if latest_only else specifications
    )
    return tuple(_builtin_operator(specification) for specification in selected_specifications)


_ALL_BUILTIN_OPERATORS = _operators_from_specifications(
    _DATASET_SPECIFICATIONS,
    latest_only=False,
)
_BUILTIN_OPERATORS = _operators_from_specifications(
    _DATASET_SPECIFICATIONS,
    latest_only=True,
)
_BUILTIN_OPERATORS_BY_REFERENCE = {
    (operator.id, operator.version): operator for operator in _ALL_BUILTIN_OPERATORS
}
_DATASET_RUNTIMES_BY_REFERENCE = {
    (specification.ref.id, specification.ref.version): _dataset_runtime(specification)
    for specification in _DATASET_SPECIFICATIONS
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
    matching = tuple(operator for operator in _ALL_BUILTIN_OPERATORS if operator.id == operator_id)
    if not matching:
        raise ValueError("unknown dataset augmentation reference")
    return max(matching, key=lambda operator: _version_tuple(operator.version))


def dataset_operator_runtime(operator: DatasetAugmentationOperator) -> DatasetVariationRuntime:
    return _DATASET_RUNTIMES_BY_REFERENCE[(operator.id, operator.version)]


def dataset_operator_prompt_name(operator: DatasetAugmentationOperator) -> str:
    runtime = dataset_operator_runtime(operator)
    return runtime.prompt_name or f"augmentation.{operator.id}"


def select_dataset_augmentation_operators(
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
