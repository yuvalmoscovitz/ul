from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from pydantic import Field

from ul_core.models import OracleRelation, Scenario, ShrinkMetadata, ULModel


class AugmentationMetadata(ULModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_.-]+$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    category: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    required_features: tuple[str, ...] = ()
    oracle_relation: OracleRelation
    shrink: ShrinkMetadata = ShrinkMetadata()


class Applicability(ULModel):
    applicable: bool
    reasons: tuple[str, ...] = ()


class ValidationResult(ULModel):
    valid: bool
    issues: tuple[str, ...] = ()


class AugmentationResult(ULModel):
    scenario: Scenario
    augmentation_id: str
    augmentation_version: str
    variant: str = "default"
    changed_paths: tuple[str, ...]
    oracle_relation: OracleRelation
    shrink: ShrinkMetadata


@runtime_checkable
class Augmentation(Protocol):
    @property
    def metadata(self) -> AugmentationMetadata: ...

    def applicability(self, scenario: Scenario) -> Applicability: ...

    def apply(self, scenario: Scenario) -> tuple[AugmentationResult, ...]: ...

    def validate(self, source: Scenario, candidate: Scenario) -> ValidationResult: ...


class AugmentationRegistry:
    def __init__(self, augmentations: Iterable[Augmentation] = ()) -> None:
        self._augmentations: dict[tuple[str, str], Augmentation] = {}
        for augmentation in augmentations:
            self.register(augmentation)

    def register(self, augmentation: Augmentation) -> None:
        key = (augmentation.metadata.id, augmentation.metadata.version)
        if key in self._augmentations:
            raise ValueError(f"augmentation already registered: {key[0]}@{key[1]}")
        self._augmentations[key] = augmentation

    def get(self, augmentation_id: str, version: str | None = None) -> Augmentation:
        matching = [
            augmentation
            for (registered_id, _), augmentation in self._augmentations.items()
            if registered_id == augmentation_id
        ]
        if not matching:
            raise KeyError(augmentation_id)
        if version is not None:
            for augmentation in matching:
                if augmentation.metadata.version == version:
                    return augmentation
            raise KeyError(f"{augmentation_id}@{version}")
        return max(matching, key=lambda item: _version_tuple(item.metadata.version))

    def list(self, *, latest_only: bool = True) -> tuple[Augmentation, ...]:
        if not latest_only:
            return tuple(
                self._augmentations[key]
                for key in sorted(
                    self._augmentations,
                    key=lambda item: (item[0], _version_tuple(item[1])),
                )
            )
        augmentation_ids = sorted({key[0] for key in self._augmentations})
        return tuple(self.get(augmentation_id) for augmentation_id in augmentation_ids)

    def applicable(self, scenario: Scenario) -> tuple[Augmentation, ...]:
        return tuple(
            augmentation
            for augmentation in self.list()
            if augmentation.applicability(scenario).applicable
        )


def _version_tuple(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)


def builtin_augmentation_registry() -> AugmentationRegistry:
    from ul_core.operators import BUILTIN_AUGMENTATIONS

    return AugmentationRegistry(BUILTIN_AUGMENTATIONS)
