from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import cast

import pytest
from pydantic import ValidationError
from ul_core.augmentations import (
    AugmentationDefinition,
    AugmentationLibrary,
    AugmentationProjection,
    AugmentationRef,
    ConversationModifierBinding,
    DeterministicTransformBinding,
    EnvironmentScheduleBinding,
    FaultControlBinding,
    ProjectionContract,
    SemanticRendererBinding,
    ValidatorBinding,
    builtin_augmentation_catalog,
    builtin_augmentation_library,
)
from ul_core.augmentations.authoring import (
    AugmentationRuntimeBinding,
    DeterministicTransformRuntime,
)
from ul_core.augmentations.registry import ValidationResult
from ul_core.models import ConversationTurn, EnvironmentEvent, Scenario


class PrivateRuntimes:
    def __init__(self, ref: AugmentationRef) -> None:
        self.ref = ref

    def transform(self, source: Scenario, projection: AugmentationProjection) -> Scenario:
        return source

    async def render(
        self,
        source: Scenario,
        projection: AugmentationProjection,
        instruction: str,
    ) -> Scenario:
        return source

    def modify_conversation(
        self, conversation: tuple[ConversationTurn, ...]
    ) -> tuple[ConversationTurn, ...]:
        return conversation

    def schedule(self, source: Scenario) -> tuple[EnvironmentEvent, ...]:
        return source.environment_events

    def control_fault(self, source: Scenario) -> tuple[EnvironmentEvent, ...]:
        return source.environment_events

    def validate(self, source: Scenario, candidate: Scenario) -> ValidationResult:
        return ValidationResult(valid=source != candidate)


def definition(ref: AugmentationRef | None = None) -> AugmentationDefinition:
    return AugmentationDefinition(
        ref=ref or AugmentationRef(id="private.customer_operator", version="1.0.0"),
        surface="conversation_workflow",
        scope="conversation",
        summary="Exercise a private customer workflow.",
        expected_relation="The declared workflow relation must hold.",
        applicability_profile="conditional",
        applicability_rule="Applies to a customer-selected conversation.",
    )


def bindings(ref: AugmentationRef) -> tuple[AugmentationRuntimeBinding, ...]:
    runtimes = PrivateRuntimes(ref)
    input_projection = ProjectionContract(reads=("structured_input",), writes=("structured_input",))
    conversation_projection = ProjectionContract(reads=("conversation",), writes=("conversation",))
    environment_projection = ProjectionContract(reads=("state",), writes=("environment",))
    return (
        DeterministicTransformBinding(
            ref=ref,
            projection=input_projection,
            runtime=runtimes,
        ),
        SemanticRendererBinding(
            ref=ref,
            projection=input_projection,
            runtime=runtimes,
            instruction="Preserve meaning while applying the declared variation.",
        ),
        ConversationModifierBinding(
            ref=ref,
            projection=conversation_projection,
            runtime=runtimes,
        ),
        EnvironmentScheduleBinding(
            ref=ref,
            projection=environment_projection,
            runtime=runtimes,
        ),
        FaultControlBinding(
            ref=ref,
            projection=environment_projection,
            runtime=runtimes,
        ),
        ValidatorBinding(ref=ref, runtime=runtimes),
    )


def test_private_author_registers_and_resolves_every_runtime_kind() -> None:
    product_definition = definition()
    library = AugmentationLibrary()

    registration = library.register(product_definition, *bindings(product_definition.ref))

    assert registration.definition == product_definition
    assert tuple(binding.kind for binding in registration.bindings) == (
        "deterministic_transform",
        "semantic_renderer",
        "conversation_modifier",
        "environment_schedule",
        "fault_control",
        "validator",
    )
    for binding in registration.bindings:
        assert library.get_binding(product_definition.ref.id, binding.kind) is binding


def test_builtin_and_private_operators_share_the_same_library_contract() -> None:
    library = builtin_augmentation_library()
    catalog = builtin_augmentation_catalog()

    assert len(library.list(latest_only=False)) == len(catalog.augmentations)
    rephrase = library.get("input.surface.rephrase")
    assert rephrase.definition.ref == catalog.get("input.surface.rephrase").ref
    assert {binding.kind for binding in rephrase.bindings} == {
        "semantic_renderer",
        "validator",
    }
    correction = library.get("conversation.correction_after_first_response")
    assert {binding.kind for binding in correction.bindings} == {
        "deterministic_transform",
        "conversation_modifier",
    }

    private_definition = definition()
    library.register(private_definition, bindings(private_definition.ref)[0])
    assert library.get(private_definition.ref.id).definition == private_definition


def test_registration_is_atomic_when_a_late_binding_has_a_mismatched_reference() -> None:
    product_definition = definition()
    candidate_bindings = list(bindings(product_definition.ref))
    other_ref = AugmentationRef(id="private.other_operator", version="1.0.0")
    candidate_bindings[-1] = ValidatorBinding(
        ref=other_ref,
        runtime=PrivateRuntimes(other_ref),
    )
    library = AugmentationLibrary()

    with pytest.raises(ValueError, match="does not match its product definition"):
        library.register(product_definition, *candidate_bindings)

    assert library.list() == ()


def test_registration_rejects_duplicate_definition_and_binding_kinds() -> None:
    product_definition = definition()
    candidate_bindings = bindings(product_definition.ref)
    library = AugmentationLibrary()
    library.register(product_definition, candidate_bindings[0])

    with pytest.raises(ValueError, match="already registered"):
        library.register(product_definition, candidate_bindings[0])

    second_definition = definition(AugmentationRef(id="private.second_operator", version="1.0.0"))
    duplicate_kind = DeterministicTransformBinding(
        ref=second_definition.ref,
        projection=ProjectionContract(reads=("structured_input",), writes=("structured_input",)),
        runtime=PrivateRuntimes(second_definition.ref),
    )
    with pytest.raises(ValueError, match="binding kinds must be unique"):
        library.register(second_definition, duplicate_kind, duplicate_kind)


def test_semantic_renderer_requires_a_validator_in_the_same_registration() -> None:
    product_definition = definition()
    semantic_binding = bindings(product_definition.ref)[1]

    with pytest.raises(ValueError, match="require a reusable validator"):
        AugmentationLibrary().register(product_definition, semantic_binding)


def test_binding_rejects_runtime_with_the_wrong_protocol() -> None:
    product_definition = definition()

    with pytest.raises(ValidationError, match="DeterministicTransformRuntime"):
        DeterministicTransformBinding(
            ref=product_definition.ref,
            projection=ProjectionContract(
                reads=("structured_input",), writes=("structured_input",)
            ),
            runtime=cast(DeterministicTransformRuntime, object()),
        )


def test_binding_rejects_non_callable_and_wrong_shape_runtime_members() -> None:
    product_definition = definition()

    class NonCallableTransform:
        ref = product_definition.ref
        transform = 42

    class WrongShapeTransform:
        ref = product_definition.ref

        def transform(self) -> Scenario:
            raise AssertionError

    for runtime, message in (
        (NonCallableTransform(), "transform member must be callable"),
        (WrongShapeTransform(), "transform member must accept 2 positional arguments"),
    ):
        with pytest.raises(ValidationError, match=message):
            DeterministicTransformBinding(
                ref=product_definition.ref,
                projection=ProjectionContract(
                    reads=("structured_input",), writes=("structured_input",)
                ),
                runtime=runtime,  # type: ignore[arg-type]
            )


def test_specialized_bindings_reject_mismatched_projection_surfaces() -> None:
    product_definition = definition()
    runtime = PrivateRuntimes(product_definition.ref)
    input_projection = ProjectionContract(reads=("structured_input",), writes=("structured_input",))

    with pytest.raises(ValidationError, match="may only write conversation"):
        ConversationModifierBinding(
            ref=product_definition.ref,
            projection=input_projection,
            runtime=runtime,
        )
    with pytest.raises(ValidationError, match="may only write environment"):
        EnvironmentScheduleBinding(
            ref=product_definition.ref,
            projection=input_projection,
            runtime=runtime,
        )
    with pytest.raises(ValidationError, match="may only write environment"):
        FaultControlBinding(
            ref=product_definition.ref,
            projection=input_projection,
            runtime=runtime,
        )


def test_specialized_bindings_reject_extra_write_surfaces() -> None:
    product_definition = definition()
    runtime = PrivateRuntimes(product_definition.ref)

    with pytest.raises(ValidationError, match="structured input or conversation"):
        SemanticRendererBinding(
            ref=product_definition.ref,
            projection=ProjectionContract(
                reads=("structured_input",),
                writes=("structured_input", "environment"),
            ),
            runtime=runtime,
            instruction="Preserve meaning.",
        )
    with pytest.raises(ValidationError, match="may only write conversation"):
        ConversationModifierBinding(
            ref=product_definition.ref,
            projection=ProjectionContract(
                reads=("conversation",), writes=("conversation", "environment")
            ),
            runtime=runtime,
        )
    for binding_type in (EnvironmentScheduleBinding, FaultControlBinding):
        with pytest.raises(ValidationError, match="may only write environment"):
            binding_type(
                ref=product_definition.ref,
                projection=ProjectionContract(reads=("state",), writes=("environment", "policy")),
                runtime=runtime,
            )


def test_concurrent_duplicate_registration_has_one_winner() -> None:
    product_definition = definition()
    rendezvous = Barrier(2)

    class RacingRuntime(PrivateRuntimes):
        def __init__(self) -> None:
            pass

        @property
        def ref(self) -> AugmentationRef:
            rendezvous.wait(timeout=5)
            return product_definition.ref

    def register(runtime: RacingRuntime) -> object:
        binding = DeterministicTransformBinding(
            ref=product_definition.ref,
            projection=ProjectionContract(
                reads=("structured_input",), writes=("structured_input",)
            ),
            runtime=runtime,
        )
        try:
            return library.register(product_definition, binding)
        except ValueError as error:
            return error

    library = AugmentationLibrary()
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(register, (RacingRuntime(), RacingRuntime())))

    assert sum(not isinstance(outcome, ValueError) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, ValueError) for outcome in outcomes) == 1
    assert len(library.list()) == 1
