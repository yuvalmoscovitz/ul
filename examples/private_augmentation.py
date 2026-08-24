from __future__ import annotations

from ul import (
    AugmentationDefinition,
    AugmentationLibrary,
    AugmentationProjection,
    AugmentationRef,
    ConversationModifierBinding,
    ConversationTurn,
    DeterministicTransformBinding,
    EnvironmentEvent,
    EnvironmentScheduleBinding,
    FaultControlBinding,
    ProjectionContract,
    Scenario,
    SemanticRendererBinding,
    ValidationResult,
    ValidatorBinding,
)


class CustomerWorkflowOperator:
    ref = AugmentationRef(id="private.customer_workflow", version="1.0.0")

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


runtime = CustomerWorkflowOperator()
definition = AugmentationDefinition(
    ref=runtime.ref,
    surface="conversation_workflow",
    scope="conversation",
    summary="Exercise a private customer workflow.",
    expected_relation="Only the declared workflow fields may change.",
    applicability_profile="conditional",
    applicability_rule="Applies to customer-selected conversations.",
)
input_projection = ProjectionContract(reads=("structured_input",), writes=("structured_input",))
conversation_projection = ProjectionContract(reads=("conversation",), writes=("conversation",))
environment_projection = ProjectionContract(reads=("state",), writes=("environment",))
semantic_instruction = "Preserve task meaning while applying the requested communication change."

library = AugmentationLibrary()
registration = library.register(
    definition,
    DeterministicTransformBinding(
        ref=runtime.ref,
        projection=input_projection,
        runtime=runtime,
    ),
    SemanticRendererBinding(
        ref=runtime.ref,
        projection=input_projection,
        runtime=runtime,
        instruction=semantic_instruction,
    ),
    ConversationModifierBinding(
        ref=runtime.ref,
        projection=conversation_projection,
        runtime=runtime,
    ),
    EnvironmentScheduleBinding(
        ref=runtime.ref,
        projection=environment_projection,
        runtime=runtime,
    ),
    FaultControlBinding(
        ref=runtime.ref,
        projection=environment_projection,
        runtime=runtime,
    ),
    ValidatorBinding(ref=runtime.ref, runtime=runtime),
)

print(
    registration.definition.ref.id,
    *(binding.kind for binding in registration.bindings),
)
