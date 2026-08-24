"""Public authoring contracts for built-in and private augmentations."""

from __future__ import annotations

from collections.abc import Awaitable, Iterable
from inspect import Parameter, signature
from threading import RLock
from typing import Annotated, Literal, Protocol, Self, runtime_checkable

from pydantic import ConfigDict, Field, model_validator

from ul_core.augmentations.definitions import (
    AugmentationApplicabilityProfile,
    AugmentationBinding,
    AugmentationRef,
    AugmentationScope,
    AugmentationSurface,
    BuiltinAugmentationSpec,
    builtin_augmentation_catalog,
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

    @model_validator(mode="after")
    def validate_runtime(self) -> Self:
        _validate_runtime_method(self.runtime, "transform", positional_parameters=2)
        return self


class SemanticRendererBinding(_ProjectionRuntimeBinding):
    kind: Literal["semantic_renderer"] = "semantic_renderer"
    runtime: SemanticRendererRuntime
    instruction: str = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_projection_surface(self) -> Self:
        if not set(self.projection.writes).issubset({"structured_input", "conversation"}):
            raise ValueError("semantic renderers must write structured input or conversation")
        _validate_runtime_method(self.runtime, "render", positional_parameters=3)
        return self


class ConversationModifierBinding(_ProjectionRuntimeBinding):
    kind: Literal["conversation_modifier"] = "conversation_modifier"
    runtime: ConversationModifierRuntime

    @model_validator(mode="after")
    def validate_projection_surface(self) -> Self:
        if set(self.projection.writes) != {"conversation"}:
            raise ValueError("conversation modifiers may only write conversation")
        _validate_runtime_method(self.runtime, "modify_conversation", positional_parameters=1)
        return self


class EnvironmentScheduleBinding(_ProjectionRuntimeBinding):
    kind: Literal["environment_schedule"] = "environment_schedule"
    runtime: EnvironmentScheduleRuntime

    @model_validator(mode="after")
    def validate_projection_surface(self) -> Self:
        if set(self.projection.writes) != {"environment"}:
            raise ValueError("environment schedules may only write environment")
        _validate_runtime_method(self.runtime, "schedule", positional_parameters=1)
        return self


class FaultControlBinding(_ProjectionRuntimeBinding):
    kind: Literal["fault_control"] = "fault_control"
    runtime: FaultControlRuntime

    @model_validator(mode="after")
    def validate_projection_surface(self) -> Self:
        if set(self.projection.writes) != {"environment"}:
            raise ValueError("fault controls may only write environment")
        _validate_runtime_method(self.runtime, "control_fault", positional_parameters=1)
        return self


class ValidatorBinding(_AuthoringModel):
    kind: Literal["validator"] = "validator"
    ref: AugmentationRef
    runtime: AugmentationValidatorRuntime

    @model_validator(mode="after")
    def validate_runtime(self) -> Self:
        _validate_runtime_method(self.runtime, "validate", positional_parameters=2)
        return self


AugmentationRuntimeBinding = Annotated[
    DeterministicTransformBinding
    | SemanticRendererBinding
    | ConversationModifierBinding
    | EnvironmentScheduleBinding
    | FaultControlBinding
    | ValidatorBinding,
    Field(discriminator="kind"),
]


class InstalledRuntimeBinding(_AuthoringModel):
    """Typed descriptor for a trusted runtime already shipped with UL."""

    kind: AugmentationRuntimeKind
    ref: AugmentationRef
    runtime_path: str = Field(min_length=3, max_length=500)
    projection: ProjectionContract | None = None

    @model_validator(mode="after")
    def validate_projection(self) -> Self:
        if (self.kind == "validator") != (self.projection is None):
            raise ValueError("only validator runtime descriptors omit a projection")
        return self


RegisteredRuntimeBinding = AugmentationRuntimeBinding | InstalledRuntimeBinding


class RegisteredAugmentation(_AuthoringModel):
    definition: AugmentationDefinition
    bindings: tuple[RegisteredRuntimeBinding, ...] = Field(min_length=1)


class AugmentationLibrary:
    """Explicit, process-local augmentation registry with atomic registration."""

    def __init__(self) -> None:
        self._registrations: dict[tuple[str, str], RegisteredAugmentation] = {}
        self._lock = RLock()

    def register(
        self,
        definition: AugmentationDefinition,
        *bindings: RegisteredRuntimeBinding,
    ) -> RegisteredAugmentation:
        key = (definition.ref.id, definition.ref.version)
        if not bindings:
            raise ValueError("augmentation registration requires at least one runtime binding")
        kinds = tuple(binding.kind for binding in bindings)
        if len(kinds) != len(set(kinds)):
            raise ValueError("augmentation runtime binding kinds must be unique")
        for binding in bindings:
            if binding.ref != definition.ref:
                raise ValueError("runtime binding reference does not match its product definition")
            if not isinstance(binding, InstalledRuntimeBinding):
                runtime_ref = binding.runtime.ref
                if runtime_ref != binding.ref:
                    raise ValueError("runtime implementation reference does not match its binding")
        if "semantic_renderer" in kinds and "validator" not in kinds:
            raise ValueError("semantic renderer registrations require a reusable validator")
        registration = RegisteredAugmentation(definition=definition, bindings=bindings)
        with self._lock:
            if key in self._registrations:
                raise ValueError(f"augmentation already registered: {key[0]}@{key[1]}")
            self._registrations[key] = registration
        return registration

    def get(self, augmentation_id: str, version: str | None = None) -> RegisteredAugmentation:
        with self._lock:
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
    ) -> RegisteredRuntimeBinding:
        registration = self.get(augmentation_id, version)
        for binding in registration.bindings:
            if binding.kind == kind:
                return binding
        resolved_ref = registration.definition.ref
        raise KeyError(f"{resolved_ref.id}@{resolved_ref.version}:{kind}")

    def list(self, *, latest_only: bool = True) -> tuple[RegisteredAugmentation, ...]:
        with self._lock:
            registrations: Iterable[RegisteredAugmentation]
            if latest_only:
                registrations = (
                    self.get(augmentation_id)
                    for augmentation_id in sorted({key[0] for key in self._registrations})
                )
            else:
                registrations = tuple(self._registrations.values())
            return tuple(
                sorted(
                    registrations,
                    key=lambda item: (
                        item.definition.ref.id,
                        _version_tuple(item.definition.ref.version),
                    ),
                )
            )


def _validate_runtime_method(
    runtime: object, method_name: str, *, positional_parameters: int
) -> None:
    method = getattr(runtime, method_name, None)
    if not callable(method):
        raise ValueError(f"runtime {method_name} member must be callable")
    parameters = tuple(signature(method).parameters.values())
    if any(parameter.kind == Parameter.VAR_POSITIONAL for parameter in parameters):
        return
    positional = tuple(
        parameter
        for parameter in parameters
        if parameter.kind in {Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD}
    )
    required_positional = tuple(
        parameter for parameter in positional if parameter.default is Parameter.empty
    )
    required_keyword_only = tuple(
        parameter
        for parameter in parameters
        if parameter.kind == Parameter.KEYWORD_ONLY and parameter.default is Parameter.empty
    )
    if (
        len(required_positional) > positional_parameters
        or len(positional) < positional_parameters
        or required_keyword_only
    ):
        raise ValueError(
            f"runtime {method_name} member must accept {positional_parameters} positional arguments"
        )


def builtin_augmentation_library() -> AugmentationLibrary:
    """Expose shipped and private augmentations through the same library contract."""

    library = AugmentationLibrary()
    for specification in builtin_augmentation_catalog().augmentations:
        definition = _definition_from_builtin(specification)
        installed_bindings = tuple(
            _installed_binding(specification, binding) for binding in specification.bindings
        )
        if any(binding.kind == "semantic_renderer" for binding in installed_bindings):
            installed_bindings = (
                *installed_bindings,
                InstalledRuntimeBinding(
                    kind="validator",
                    ref=definition.ref,
                    runtime_path="ul.augmentations.dataset:DatasetAugmentationEngine",
                ),
            )
        library.register(definition, *installed_bindings)
    return library


def _definition_from_builtin(specification: BuiltinAugmentationSpec) -> AugmentationDefinition:
    return AugmentationDefinition(
        ref=specification.ref,
        surface=specification.surface,
        scope=specification.scope,
        summary=specification.summary,
        expected_relation=specification.expected_relation,
        applicability_profile=specification.applicability_profile,
        applicability_rule=specification.applicability_rule,
    )


def _installed_binding(
    specification: BuiltinAugmentationSpec, binding: AugmentationBinding
) -> InstalledRuntimeBinding:
    if binding.mode == "dataset_variation":
        kind: AugmentationRuntimeKind = "semantic_renderer"
    elif binding.mode == "conversation_stress":
        kind = "conversation_modifier"
    elif binding.mode == "environment_fault":
        kind = "fault_control"
    elif specification.scope == "environment":
        kind = "environment_schedule"
    else:
        kind = "deterministic_transform"
    return InstalledRuntimeBinding(
        kind=kind,
        ref=specification.ref,
        runtime_path=binding.runtime,
        projection=binding.projection,
    )


def _version_tuple(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)
