"""Definitions and runtime registry for UL's built-in augmentations.

See ``README.md`` beside this file for the complete operator index.
"""

from ul_core.augmentations.definitions import (
    AugmentationBinding,
    AugmentationRef,
    AugmentationRequirements,
    BuiltinAugmentationCatalog,
    BuiltinAugmentationSpec,
    builtin_augmentation_catalog,
)
from ul_core.augmentations.registry import (
    AugmentationRegistry,
    builtin_augmentation_registry,
)

__all__ = [
    "AugmentationBinding",
    "AugmentationRef",
    "AugmentationRegistry",
    "AugmentationRequirements",
    "BuiltinAugmentationCatalog",
    "BuiltinAugmentationSpec",
    "builtin_augmentation_catalog",
    "builtin_augmentation_registry",
]
