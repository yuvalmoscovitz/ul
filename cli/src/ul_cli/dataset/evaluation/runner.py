from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import TextIO

from rich.console import Console
from ul import (
    DatasetAugmentationEngine,
    DatasetAugmentationResult,
    DatasetEvaluationResult,
    DatasetEvaluationRunner,
    DatasetEvaluationTrial,
    DatasetSemanticSettings,
    DatasetTargetDeliveryUncertain,
    DatasetTrialUnit,
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
from ul_cli.dataset_campaign import DatasetCampaignPlan
from ul_cli.dataset_review import DatasetEvidenceRunContext
from ul_cli.dataset_trial_journal import DatasetTrialJournal

from ..evidence.customer import build_customer_evidence_record
from ..progress import (
    CampaignControl,
    CampaignControlRequested,
    CampaignProgressTracker,
    CampaignSignalControl,
    JsonCampaignProgressRenderer,
    SafeCampaignProgressPublisher,
    TerminalCampaignProgressRenderer,
)


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
    trial_journal: DatasetTrialJournal | None = None,
    progress_json: bool = False,
    progress_plan: DatasetCampaignPlan | None = None,
) -> tuple[DatasetEvaluationResult, ...]:
    results: list[DatasetEvaluationResult] = []
    work_upper_bound = len(records) * repetitions * (1 + len(operator_ids))
    semantic_call_budget = progress_plan.calls.total_semantic_model if progress_plan else 0
    token_budget = progress_plan.tokens.maximum if progress_plan else 0
    timeout_seconds = settings.timeout_seconds if progress_plan else 1
    progress_renderer = (
        JsonCampaignProgressRenderer(sys.stderr)
        if progress_json
        else TerminalCampaignProgressRenderer(Console(stderr=True))
    )
    progress_tracker = CampaignProgressTracker(
        case_count=len(records),
        work_upper_bound=work_upper_bound,
        target_call_budget=planned_target_calls,
        semantic_call_budget=semantic_call_budget,
        environment_call_budget=max_environment_api_calls,
        token_budget=token_budget,
        maximum_wall_time_seconds=max(
            1,
            planned_target_calls + semantic_call_budget,
        )
        * timeout_seconds,
        publish=SafeCampaignProgressPublisher(progress_renderer).publish,
    )
    campaign_control = CampaignControl()
    signal_control = CampaignSignalControl(campaign_control)
    actual_target_calls = 0
    active_trial: tuple[int, DatasetTrialUnit] | None = None

    def durable_flush() -> None:
        if trial_journal is not None:
            trial_journal.flush()
        output_stream.flush()
        os.fsync(output_stream.fileno())

    def stop_at_requested_boundary() -> None:
        if progress_tracker.safe_boundary(campaign_control, durable_flush):
            return
        action = campaign_control.requested_action()
        assert action is not None
        raise CampaignControlRequested(action)

    progress_tracker.emit(status="running", stage="augmentation")
    deconstructor = create_semantic_model_deconstructor(settings)
    deconstructor.reuse_preflight(evaluator_preflight)
    with signal_control.installed():
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
            for case_number, record in enumerate(records, start=1):
                precomputed_augmentation = (
                    saved_augmentations.get(record.id) if saved_augmentations is not None else None
                )

                def checkpoint_plan_outcomes(
                    augmentation: DatasetAugmentationResult,
                    source: InteractionRecord = record,
                    case_position: int = case_number,
                ) -> None:
                    candidates = {
                        candidate.operator_id: candidate for candidate in augmentation.candidates
                    }
                    for operator_reference in operator_ids:
                        operator_id = operator_reference.partition("@")[0]
                        candidate = candidates.get(operator_id)
                        if candidate is not None and candidate.passed:
                            continue
                        state = "inapplicable" if candidate is None else "rejected"
                        reason_code = (
                            "operator_not_materialized"
                            if candidate is None
                            else "augmentation_rejected"
                        )
                        for repetition in range(1, repetitions + 1):
                            unit = DatasetTrialUnit(
                                interaction_id=source.id,
                                operator_id=operator_id,
                                arm="probe",
                                repetition=repetition,
                            )
                            if trial_journal is not None and trial_journal.is_terminal(unit):
                                continue
                            if trial_journal is not None:
                                trial_journal.terminal(unit, state, reason_code)
                            progress_tracker.trial_skipped(
                                case_number=case_position,
                                unit=unit,
                            )

                def checkpoint_augmentation(
                    augmentation: DatasetAugmentationResult,
                    source: InteractionRecord = record,
                ) -> None:
                    if augmentation_ledger is not None:
                        augmentation_ledger.append(source=source, augmentation=augmentation)
                    checkpoint_plan_outcomes(augmentation)

                if precomputed_augmentation is not None:
                    checkpoint_plan_outcomes(precomputed_augmentation)

                def trial_started(
                    unit: DatasetTrialUnit,
                    case_position: int = case_number,
                ) -> None:
                    nonlocal active_trial, actual_target_calls
                    stop_at_requested_boundary()
                    if trial_journal is not None:
                        trial_journal.start(unit)
                    progress_tracker.trial_started(case_number=case_position, unit=unit)
                    actual_target_calls += 1
                    active_trial = (case_position, unit)
                    task = asyncio.current_task()
                    if task is not None:
                        signal_control.target_call_started(task)

                def trial_terminal(
                    unit: DatasetTrialUnit,
                    trial: DatasetEvaluationTrial,
                    case_position: int = case_number,
                ) -> None:
                    nonlocal active_trial
                    signal_control.target_call_finished()
                    active_trial = None
                    if trial_journal is not None:
                        trial_journal.finish(unit, trial)
                    progress_tracker.trial_terminal(
                        case_number=case_position,
                        unit=unit,
                        trial=trial,
                    )
                    stop_at_requested_boundary()

                try:
                    result = await runner.run(
                        record,
                        operator_ids=operator_ids,
                        repetitions=repetitions,
                        precomputed_augmentation=precomputed_augmentation,
                        augmentation_checkpoint_callback=checkpoint_augmentation,
                        prior_trials=(
                            trial_journal.snapshot.recovered_trials
                            if trial_journal is not None
                            else None
                        ),
                        trial_started_callback=trial_started,
                        trial_terminal_callback=trial_terminal,
                    )
                except DatasetTargetDeliveryUncertain:
                    signal_control.target_call_finished()
                    durable_flush()
                    assert active_trial is not None
                    active_case_number, active_unit = active_trial
                    progress_tracker.trial_delivery_uncertain(
                        case_number=active_case_number,
                        unit=active_unit,
                    )
                    active_trial = None
                    raise
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
                durable_flush()
                progress_tracker.record_usage(
                    target_calls=actual_target_calls,
                    semantic_calls=sum(
                        getattr(getattr(item, "semantic_calls", None), "actual_calls", 0)
                        for item in (*results, result)
                    ),
                    environment_calls=None,
                    tokens=None,
                )
                results.append(result)
    progress_tracker.emit(status="completed", stage="terminal")
    return tuple(results)
