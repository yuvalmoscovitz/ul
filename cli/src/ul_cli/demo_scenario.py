from __future__ import annotations

from typing import ClassVar

from pydantic import JsonValue
from ul.augmentations.dataset import (
    DatasetAugmentationCandidate,
    DatasetAugmentationEngine,
    DatasetAugmentationOperatorReference,
    DatasetAugmentationResult,
)
from ul.dataset_evaluation import DatasetEvaluationResult, DatasetEvaluationRunner
from ul_core.dataset import (
    EvidenceReference,
    InteractionRecord,
    ObservedOutcome,
    RenderedUserInput,
    SemanticFrame,
    UserInputRecord,
)
from ul_core.evaluation import (
    EnvironmentCapabilities,
    EnvironmentLifecycleEvidence,
    EnvironmentResetEvidence,
    EnvironmentStateEvidence,
    EnvironmentTurnEvidence,
    EvaluationCase,
    ExecutionEvidence,
)

_REPETITIONS = 3
_ENVIRONMENT_CONFIG_SHA256 = "d" * 64


def _action_evidence() -> tuple[EvidenceReference, ...]:
    return (
        EvidenceReference(
            source="output",
            json_pointer="/raw_observed_output",
            text_quote=None,
        ),
    )


def _outcome(interaction_id: str, predicate: str, **fields: JsonValue) -> ObservedOutcome:
    return ObservedOutcome(
        id=f"{interaction_id}:action",
        evidence=_action_evidence(),
        confidence=1,
        status="observed",
        position=0,
        kind="action",
        predicate=predicate,
        fields=fields,
    )


def _frame_from_record(record: InteractionRecord) -> SemanticFrame:
    raw_output = record.raw_observed_output
    outcomes: tuple[ObservedOutcome, ...] = ()
    if isinstance(raw_output, dict):
        action = raw_output.get("action")
        if isinstance(action, str):
            fields = {name: value for name, value in raw_output.items() if name != "action"}
            outcomes = (_outcome(record.id, action, **fields),)
    return SemanticFrame(
        interaction_id=record.id,
        outcomes=outcomes,
        extractor_version="ul-demo-deterministic-1.0.0",
    )


class DemoSemanticPipeline:
    async def deconstruct(
        self,
        record: InteractionRecord | UserInputRecord,
        reference_frame: SemanticFrame | None = None,
    ) -> SemanticFrame:
        del reference_frame
        if isinstance(record, InteractionRecord):
            return _frame_from_record(record)
        return SemanticFrame(
            interaction_id=record.id,
            extractor_version="ul-demo-deterministic-1.0.0",
        )

    async def render(
        self,
        raw_input: str,
        instruction: str,
        *,
        allow_temporary_value: bool = False,
    ) -> RenderedUserInput:
        del raw_input, instruction, allow_temporary_value
        raise AssertionError("demo uses prebuilt model-free augmentations")


class DemoAgentEnvironment:
    environment_id = "ul-demo-customer-support-agent"
    config_sha256 = _ENVIRONMENT_CONFIG_SHA256
    capabilities = EnvironmentCapabilities(
        supports_conversations=False,
        supports_state_observation=True,
        state_observation_authority="environment_self_reported",
        cancellation_guarantee="guaranteed",
    )

    _responses: ClassVar[dict[str, JsonValue]] = {
        "Please cancel my subscription at the end of this billing period.": {
            "action": "subscription_cancellation_scheduled",
            "when": "end",
        },
        "Plase cancle my subscription at the end of this biling period.": {
            "message": "I could not understand that request.",
            "action": None,
        },
        "I'm fed up with this. Cancel my subscription at the end of this billing period.": {
            "action": "subscription_cancellation_scheduled",
            "when": "immediately",
        },
        "Please change my delivery address to 18 Oak Street.": {
            "action": "customer_address_changed",
            "address_type": "delivery",
            "address": "18 Oak Street",
        },
        "delivery address 18 Oak St": {
            "action": "customer_address_changed",
            "address_type": "billing",
            "address": "18 Oak Street",
        },
    }

    def api_calls_for_case(self, case: EvaluationCase) -> int:
        return len(case.turns)

    async def execute(self, case: EvaluationCase) -> ExecutionEvidence:
        turn = case.turns[0]
        response = self._responses.get(turn.content)
        if response is None:
            raise RuntimeError("unknown demo input")
        initial_state = EnvironmentStateEvidence(
            value={"actions": []},
            authority="environment_self_reported",
        )
        final_state = EnvironmentStateEvidence(
            value={"last_response": response},
            authority="environment_self_reported",
        )
        reset_receipt = EnvironmentResetEvidence(
            reset_session_requested=True,
            reset_session_acknowledged=True,
            reset_env_requested=True,
            reset_env_acknowledged=True,
        )
        return ExecutionEvidence(
            case_id=case.id,
            environment_id=self.environment_id,
            environment_config_sha256=self.config_sha256,
            initial_state=initial_state,
            turns=(
                EnvironmentTurnEvidence(
                    turn_id=turn.id,
                    response=response,
                    state_snapshot=final_state.value,
                    state_observation_authority=final_state.authority,
                ),
            ),
            final_response=response,
            final_state=final_state,
            lifecycle=EnvironmentLifecycleEvidence(
                initial_reset=reset_receipt,
                cleanup_reset=reset_receipt,
                terminal_status="succeeded",
                completed_phases=("reset", "execute_turn", "cleanup_reset"),
                delivery="certain",
                cleanup="succeeded",
                environment_state_uncertain=False,
            ),
        )


def _candidate(
    source: InteractionRecord,
    source_frame: SemanticFrame,
    operator_id: str,
    augmented_input: str,
) -> DatasetAugmentationCandidate:
    return DatasetAugmentationCandidate.model_validate(
        {
            "source_interaction_id": source.id,
            "operator_id": operator_id,
            "operator_version": "1.0.0",
            "allowed_change": "declared_communication_form",
            "human_review_required": operator_id == "input.tone.frustrated",
            "changed_paths": (source.augmentation_path,),
            "augmented_input": augmented_input,
            "renderer_metadata": {"renderer": "ul-demo-deterministic"},
            "expected_input_frame": source_frame,
            "reparsed_input_frame": source_frame.model_copy(
                update={"interaction_id": f"{source.id}:{operator_id}"}
            ),
            "passed": True,
            "failure_reasons": (),
        }
    )


def _precomputed_augmentation(
    source: InteractionRecord,
    variations: tuple[tuple[str, str], ...],
) -> DatasetAugmentationResult:
    source_frame = _frame_from_record(source)
    candidates = tuple(
        _candidate(source, source_frame, operator_id, augmented_input)
        for operator_id, augmented_input in variations
    )
    return DatasetAugmentationResult(
        operator_references=tuple(
            DatasetAugmentationOperatorReference(id=candidate.operator_id)
            for candidate in candidates
        ),
        source_records=(source,),
        source_frames=(source_frame,),
        candidates=candidates,
    )


async def _evaluate(
    source: InteractionRecord,
    variations: tuple[tuple[str, str], ...],
) -> DatasetEvaluationResult:
    semantic_pipeline = DemoSemanticPipeline()
    augmentation = _precomputed_augmentation(source, variations)
    runner = DatasetEvaluationRunner(
        DatasetAugmentationEngine(semantic_pipeline, semantic_pipeline),
        semantic_pipeline,
        DemoAgentEnvironment(),
        allow_network_egress=True,
    )
    return await runner.run(
        source,
        operator_ids=tuple(operator_id for operator_id, _ in variations),
        repetitions=_REPETITIONS,
        precomputed_augmentation=augmentation,
    )


async def run_demo_evaluations() -> tuple[DatasetEvaluationResult, ...]:
    cancellation = InteractionRecord(
        id="cancel-subscription",
        raw_input="Please cancel my subscription at the end of this billing period.",
        raw_observed_output={
            "action": "subscription_cancellation_scheduled",
            "when": "end",
        },
    )
    address = InteractionRecord(
        id="change-delivery-address",
        raw_input="Please change my delivery address to 18 Oak Street.",
        raw_observed_output={
            "action": "customer_address_changed",
            "address_type": "delivery",
            "address": "18 Oak Street",
        },
    )
    cancellation_result = await _evaluate(
        cancellation,
        (
            (
                "input.surface.typing_noise",
                "Plase cancle my subscription at the end of this biling period.",
            ),
            (
                "input.tone.frustrated",
                "I'm fed up with this. Cancel my subscription at the end of this billing period.",
            ),
        ),
    )
    address_result = await _evaluate(
        address,
        (("input.style.terse", "delivery address 18 Oak St"),),
    )
    return cancellation_result, address_result
