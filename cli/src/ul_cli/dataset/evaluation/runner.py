from __future__ import annotations

import json
import os
from typing import TextIO

from ul import (
    DatasetAugmentationEngine,
    DatasetAugmentationResult,
    DatasetEvaluationResult,
    DatasetEvaluationRunner,
    DatasetSemanticSettings,
    EvaluatorModelPreflight,
    InteractionRecord,
    RedactedSemanticPipeline,
    RedactionEngine,
    create_semantic_model_deconstructor,
)
from ul.dataset_invariants import (
    DatasetInvariantEvaluation,
    DatasetInvariantSuite,
    evaluate_dataset_invariants,
)
from ul.http_environment import JsonHttpEnvironmentConnection
from ul.local_target import LocalTargetConnection

from ul_cli.dataset_augmentation_ledger import DatasetAugmentationLedger
from ul_cli.dataset_review import DatasetEvidenceRunContext

from ..evidence.customer import build_customer_evidence_record


async def preflight_evaluator(settings: DatasetSemanticSettings) -> EvaluatorModelPreflight:
    async with create_semantic_model_deconstructor(settings) as deconstructor:
        return await deconstructor.preflight()


async def evaluate_interaction_records(
    records: tuple[InteractionRecord, ...],
    operator_ids: tuple[str, ...],
    settings: DatasetSemanticSettings,
    target: JsonHttpEnvironmentConnection | LocalTargetConnection,
    output_stream: TextIO,
    *,
    repetitions: int,
    max_environment_api_calls: int,
    planned_target_calls: int,
    run_context: DatasetEvidenceRunContext | None = None,
    augmentation_ledger: DatasetAugmentationLedger | None = None,
    saved_augmentations: dict[str, DatasetAugmentationResult] | None = None,
    invariant_suite: DatasetInvariantSuite | None = None,
    invariant_evaluations: list[DatasetInvariantEvaluation] | None = None,
    redaction_engine: RedactionEngine | None = None,
    allow_network_egress: bool = True,
    evaluator_preflight: EvaluatorModelPreflight,
) -> tuple[DatasetEvaluationResult, ...]:
    results: list[DatasetEvaluationResult] = []
    deconstructor = create_semantic_model_deconstructor(settings)
    deconstructor.reuse_preflight(evaluator_preflight)
    async with deconstructor, target:
        semantic_pipeline = (
            RedactedSemanticPipeline(deconstructor, redaction_engine)
            if redaction_engine is not None
            else deconstructor
        )
        evaluation_target = (
            semantic_pipeline.wrap_environment(target)
            if isinstance(semantic_pipeline, RedactedSemanticPipeline)
            else target
        )
        runner = DatasetEvaluationRunner(
            DatasetAugmentationEngine(
                semantic_pipeline,
                semantic_pipeline,
                semantic_pipeline,
            ),
            semantic_pipeline,
            evaluation_target,
            allow_network_egress=allow_network_egress,
            evaluation_mode=(
                run_context.evaluation_mode
                if run_context is not None and run_context.evaluation_mode is not None
                else "variance"
            ),
        )
        for record in records:
            precomputed_augmentation = (
                saved_augmentations.get(record.id) if saved_augmentations is not None else None
            )

            def checkpoint_augmentation(
                augmentation: DatasetAugmentationResult,
                source: InteractionRecord = record,
            ) -> None:
                if augmentation_ledger is None:
                    return
                augmentation_ledger.append(source=source, augmentation=augmentation)

            result = await runner.run(
                record,
                operator_ids=operator_ids,
                repetitions=repetitions,
                precomputed_augmentation=precomputed_augmentation,
                augmentation_checkpoint_callback=(
                    checkpoint_augmentation if augmentation_ledger is not None else None
                ),
            )
            invariant_evaluation = (
                evaluate_dataset_invariants(result, invariant_suite)
                if invariant_suite is not None
                else None
            )
            if invariant_evaluation is not None and invariant_evaluations is not None:
                invariant_evaluations.append(invariant_evaluation)
            output_stream.write(
                json.dumps(
                    build_customer_evidence_record(
                        result,
                        repetitions=repetitions,
                        max_environment_api_calls=max_environment_api_calls,
                        planned_target_calls=planned_target_calls,
                        run_context=run_context,
                        invariant_evaluation=invariant_evaluation,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
            output_stream.flush()
            os.fsync(output_stream.fileno())
            results.append(result)
    return tuple(results)
