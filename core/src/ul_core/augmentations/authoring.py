"""Public authoring contracts for built-in and private augmentations."""

from __future__ import annotations

from collections.abc import Awaitable, Iterable
from typing import Annotated, Literal, Protocol, Self, runtime_checkable

from pydantic import ConfigDict, Field, model_validator

from ul_core.augmentations.definitions import (
    AugmentationApplicabilityProfile,
    AugmentationRef,
    AugmentationScope,
    AugmentationSurface,
)
from ul_core.augmentations.projections import AugmentationProjection, ProjectionContract
from ul_core.augmentations.registry import ValidationResult
from ul_core.models import ConversationTurn, EnvironmentEvent, Scenario, ULModel

AugmentationRuntimeKind = Literal[
    "deterministic_transform",
    "semantic_renderer",
    "conversation_modifier",
    "environment_schedule",
    "fault_control",
    "validator",
]


class _AuthoringModel(ULModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        strict=True,
    )


class AugmentationDefinition(_AuthoringModel):
    """Customer-visible product metadata with no runtime implementation details."""

    ref: AugmentationRef
    surface: AugmentationSurface
    scope: AugmentationScope
    summary: str = Field(min_length=1, max_length=500)
    expected_relation: str = Field(min_length=1, max_length=1_000)
    applicability_profile: AugmentationApplicabilityProfile
    applicability_rule: str = Field(min_length=1, max_length=500)


@runtime_checkable
class DeterministicTransformRuntime(Protocol):
    @property
    def ref(self) -> AugmentationRef: ...

    def transform(self, source: Scenario, projection: AugmentationProjection) -> Scenario: ...


@runtime_checkable
class SemanticRendererRuntime(Protocol):
    @property
    def ref(self) -> AugmentationRef: ...

    def render(
        self,
        source: Scenario,
        projection: AugmentationProjection,
        instruction: str,
    ) -> Awaitable[Scenario]: ...


@runtime_checkable
class ConversationModifierRuntime(Protocol):
    @property
    def ref(self) -> AugmentationRef: ...

    def modify_conversation(
        self, conversation: tuple[ConversationTurn, ...]
    ) -> tuple[ConversationTurn, ...]: ...


@runtime_checkable
class EnvironmentScheduleRuntime(Protocol):
    @property
    def ref(self) -> AugmentationRef: ...

    def schedule(self, source: Scenario) -> tuple[EnvironmentEvent, ...]: ...


@runtime_checkable
class FaultControlRuntime(Protocol):
    @property
    def ref(self) -> AugmentationRef: ...

    def control_fault(self, source: Scenario) -> tuple[EnvironmentEvent, ...]: ...


@runtime_checkable
class AugmentationValidatorRuntime(Protocol):
    @property
    def ref(self) -> AugmentationRef: ...

    def validate(self, source: Scenario, candidate: Scenario) -> ValidationResult: ...


class _ProjectionRuntimeBinding(_AuthoringModel):
    ref: AugmentationRef
    projection: ProjectionContract


class DeterministicTransformBinding(_ProjectionRuntimeBinding):
    kind: Literal["deterministic_transform"] = "deterministic_transform"
    runtime: DeterministicTransformRuntime


class SemanticRendererBinding(_ProjectionRuntimeBinding):
    kind: Literal["semantic_renderer"] = "semantic_renderer"
    runtime: SemanticRendererRuntime
    instruction: str = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_projection_surface(self) -> Self:
        if not set(self.projection.writes).intersection({"structured_input", "conversation"}):
            raise ValueError("semantic renderers must write structured input or conversation")
        return self


class ConversationModifierBinding(_ProjectionRuntimeBinding):
    kind: Literal["conversation_modifier"] = "conversation_modifier"
    runtime: ConversationModifierRuntime

    @model_validator(mode="after")
    def validate_projection_surface(self) -> Self:
        if "conversation" not in self.projection.writes:
            raise ValueError("conversation modifiers must write conversation")
        return self


class EnvironmentScheduleBinding(_ProjectionRuntimeBinding):
    kind: Literal["environment_schedule"] = "environment_schedule"
    runtime: EnvironmentScheduleRuntime

    @model_validator(mode="after")
    def validate_projection_surface(self) -> Self:
        if "environment" not in self.projection.writes:
            raise ValueError("environment schedules must write environment")
        return self


class FaultControlBinding(_ProjectionRuntimeBinding):
    kind: Literal["fault_control"] = "fault_control"
    runtime: FaultControlRuntime

    @model_validator(mode="after")
    def validate_projection_surface(self) -> Self:
        if "environment" not in self.projection.writes:
            raise ValueError("fault controls must write environment")
        return self


class ValidatorBinding(_AuthoringModel):
    kind: Literal["validator"] = "validator"
    ref: AugmentationRef
    runtime: AugmentationValidatorRuntime


AugmentationRuntimeBinding = Annotated[
    DeterministicTransformBinding
    | SemanticRendererBinding
    | ConversationModifierBinding
    | EnvironmentScheduleBinding
    | FaultControlBinding
    | ValidatorBinding,
    Field(discriminator="kind"),
]


class RegisteredAugmentation(_AuthoringModel):
    definition: AugmentationDefinition
    bindings: tuple[AugmentationRuntimeBinding, ...] = Field(min_length=1)


class AugmentationLibrary:
    """Explicit, process-local augmentation registry with atomic registration."""

    def __init__(self) -> None:
        self._registrations: dict[tuple[str, str], RegisteredAugmentation] = {}

    def register(
        self,
        definition: AugmentationDefinition,
        *bindings: AugmentationRuntimeBinding,
    ) -> RegisteredAugmentation:
        key = (definition.ref.id, definition.ref.version)
        if key in self._registrations:
            raise ValueError(f"augmentation already registered: {key[0]}@{key[1]}")
        if not bindings:
            raise ValueError("augmentation registration requires at least one runtime binding")
        kinds = tuple(binding.kind for binding in bindings)
        if len(kinds) != len(set(kinds)):
            raise ValueError("augmentation runtime binding kinds must be unique")
        for binding in bindings:
            if binding.ref != definition.ref:
                raise ValueError("runtime binding reference does not match its product definition")
            runtime_ref = binding.runtime.ref
            if runtime_ref != binding.ref:
                raise ValueError("runtime implementation reference does not match its binding")
        if "semantic_renderer" in kinds and "validator" not in kinds:
            raise ValueError("semantic renderer registrations require a reusable validator")
        registration = RegisteredAugmentation(definition=definition, bindings=bindings)
        self._registrations[key] = registration
        return registration

    def get(self, augmentation_id: str, version: str | None = None) -> RegisteredAugmentation:
        if version is not None:
            try:
                return self._registrations[(augmentation_id, version)]
            except KeyError:
                raise KeyError(f"{augmentation_id}@{version}") from None
        matching = tuple(
            registration
            for (registered_id, _), registration in self._registrations.items()
            if registered_id == augmentation_id
        )
        if not matching:
            raise KeyError(augmentation_id)
        return max(matching, key=lambda item: _version_tuple(item.definition.ref.version))

    def get_binding(
        self,
        augmentation_id: str,
        kind: AugmentationRuntimeKind,
        version: str | None = None,
    ) -> AugmentationRuntimeBinding:
        registration = self.get(augmentation_id, version)
        for binding in registration.bindings:
            if binding.kind == kind:
                return binding
        resolved_ref = registration.definition.ref
        raise KeyError(f"{resolved_ref.id}@{resolved_ref.version}:{kind}")

    def list(self, *, latest_only: bool = True) -> tuple[RegisteredAugmentation, ...]:
        registrations: Iterable[RegisteredAugmentation]
        if latest_only:
            registrations = (
                self.get(augmentation_id)
                for augmentation_id in sorted({key[0] for key in self._registrations})
            )
        else:
            registrations = self._registrations.values()
        return tuple(
            sorted(
                registrations,
                key=lambda item: (
                    item.definition.ref.id,
                    _version_tuple(item.definition.ref.version),
                ),
            )
        )


def _version_tuple(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)
