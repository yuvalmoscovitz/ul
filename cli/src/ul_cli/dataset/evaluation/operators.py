from __future__ import annotations

from ul import DatasetAugmentationOperator, resolve_dataset_augmentation_operator
from ul_core.augmentations.definitions import builtin_augmentation_catalog

from ..presentation.runtime import console
from .records import DatasetInputError


def list_dataset_operators() -> None:
    """List the dataset subset of UL's unified augmentation catalog."""
    console.print("Dataset augmentations (from 'ul augmentations list')")
    for augmentation in builtin_augmentation_catalog().list(mode="dataset_variation"):
        console.print(f"- {augmentation.ref.id}@{augmentation.ref.version}: {augmentation.summary}")


def validate_operator_ids(operator_ids: list[str] | None) -> tuple[str, ...]:
    selected_references = tuple(operator_ids or ["input.surface.rephrase"])
    selected_operators: list[DatasetAugmentationOperator] = []
    for reference in selected_references:
        try:
            operator = resolve_dataset_augmentation_operator(reference)
        except ValueError:
            raise DatasetInputError("unknown augmentation operator reference") from None
        selected_operators.append(operator)
    resolved_references = tuple((operator.id, operator.version) for operator in selected_operators)
    if len(resolved_references) != len(set(resolved_references)):
        raise DatasetInputError("duplicate --operator values are not allowed")
    return tuple(
        reference if "@" in reference else operator.id
        for reference, operator in zip(selected_references, selected_operators, strict=True)
    )


def validate_dataset_operator_ids(operator_ids: list[str] | None) -> tuple[str, ...]:
    """Resolve and validate dataset augmentation operator references."""
    return validate_operator_ids(operator_ids)


def dataset_operator_identity(reference: str) -> tuple[str, str]:
    operator = resolve_dataset_augmentation_operator(reference)
    return operator.id, operator.version
