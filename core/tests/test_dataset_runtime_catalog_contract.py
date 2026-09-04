import pytest
from pydantic import ValidationError
from ul_core.augmentations.definitions import (
    BuiltinAugmentationCatalog,
    builtin_augmentation_catalog,
)


def test_only_dataset_bindings_declare_dataset_runtime_metadata() -> None:
    for specification in builtin_augmentation_catalog().list(latest_only=False):
        for binding in specification.bindings:
            if binding.mode == "dataset_variation":
                assert binding.dataset_runtime is not None, specification.ref
            else:
                assert binding.dataset_runtime is None, specification.ref


def test_dataset_runtime_order_is_unambiguous() -> None:
    catalog = builtin_augmentation_catalog()
    first, second, *remaining = catalog.list(mode="dataset_variation", latest_only=False)
    first_runtime = first.bindings[0].dataset_runtime
    second_runtime = second.bindings[0].dataset_runtime
    assert first_runtime is not None
    assert second_runtime is not None
    duplicate_order_binding = second.bindings[0].model_copy(
        update={"dataset_runtime": second_runtime.model_copy(update={"order": first_runtime.order})}
    )
    duplicate_order_specification = second.model_copy(
        update={"bindings": (duplicate_order_binding,)}
    )

    with pytest.raises(ValidationError, match="dataset variation runtime order must be unique"):
        BuiltinAugmentationCatalog(augmentations=(first, duplicate_order_specification, *remaining))
