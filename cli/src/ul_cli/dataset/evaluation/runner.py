from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TextIO

from ul import (
    DatasetAugmentationEngine,
    DatasetAugmentationResult,
    DatasetEvaluationResult,
    DatasetEvaluationRunner,
    DatasetEvaluationTrial,
    DatasetMaterialVarianceJudge,
    DatasetSemanticSettings,
    DatasetSourcePreparationError,
    DatasetTargetDeliveryUncertain,
    DatasetTrialUnit,
    EvaluatorModelPreflight,
    InteractionRecord,
    OpenAICompatibleEvaluatorJudge,
    ProjectedResponseSemanticDeconstructor,
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

from ul_cli.dataset.source_preparation import (
    DatasetSourcePreparationFailureEvent,
    build_source_preparation_failure_event,
    persist_source_preparation_failure_event,
)
from ul_cli.dataset_augmentation_ledger import DatasetAugmentationLedger
from ul_cli.dataset_campaign import DatasetCampaignPlan
from ul_cli.dataset_review import DatasetEvidenceRunContext
from ul_cli.dataset_run_config import DatasetRunConfig
from ul_cli.dataset_trial_journal import DatasetTrialJournal

from ..evidence.customer import build_customer_evidence_record
from ..progress import (
    CampaignControlRequested,
    CampaignNextCommands,
    CampaignProgressRuntime,
    create_campaign_next_commands,
    create_campaign_progress_runtime,
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
    run_config: DatasetRunConfig,
    run_context: DatasetEvidenceRunContext | None = None,
    augmentation_ledger: DatasetAugmentationLedger | None = None,
    saved_augmentations: dict[str, DatasetAugmentationResult] | None = None,
    invariant_suite: DatasetInvariantSuite | None = None,
    invariant_evaluations: list[DatasetInvariantEvaluation] | None = None,
    redaction_engine: RedactionEngine | None = None,
    evaluator_preflight: EvaluatorModelPreflight,
    trial_journal: DatasetTrialJournal | None = None,
    progress_json: bool = False,
    progress_plan: DatasetCampaignPlan | None = None,
    progress_next_commands: CampaignNextCommands | None = None,
    progress_runtime: CampaignProgressRuntime | None = None,
    complete_progress: bool = True,
    isolate_source_preparation_failures: bool = True,
    source_preparation_events: list[DatasetSourcePreparationFailureEvent] | None = None,
) -> tuple[DatasetEvaluationResult, ...]:
    repetitions = run_config.repetitions
    target_config = run_config.target
    results: list[DatasetEvaluationResult] = []
    work_upper_bound = len(records) * repetitions * (1 + len(operator_ids))
    semantic_call_budget = progress_plan.calls.total_semantic_model if progress_plan else 0
    token_budget = progress_plan.tokens.maximum if progress_plan else 0
    semantic_timeout_seconds = settings.timeout_seconds if progress_plan else 1
    if progress_next_commands is None:
        output_name = str(getattr(output_stream, "name", "dataset-evidence.jsonl"))
        progress_next_commands = create_campaign_next_commands(Path(output_name))
    if progress_runtime is None:
        progress_runtime = create_campaign_progress_runtime(
            case_count=len(records),
            work_upper_bound=work_upper_bound,
            target_call_budget=target_config.planned_environment_api_calls,
            semantic_call_budget=semantic_call_budget,
            environment_call_budget=target_config.max_environment_api_calls,
            token_budget=token_budget,
            maximum_wall_time_seconds=(
                max(1, work_upper_bound) * target_config.trial_timeout_seconds
                + semantic_call_budget * semantic_timeout_seconds
            ),
            next_commands=progress_next_commands,
            json_output=progress_json,
        )
    progress_tracker = progress_runtime.tracker
    campaign_control = progress_runtime.control
    signal_control = progress_runtime.signal_control
    initial_target_calls, _, initial_environment_calls, _ = progress_tracker.actual_usage
    actual_target_calls = 0
    active_trials: dict[asyncio.Task[object], tuple[int, DatasetTrialUnit]] = {}
    recorded_source_preparation_events = (
        source_preparation_events if source_preparation_events is not None else []
    )

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
    if not settings.allow_external_data_processing:
        raise ValueError("material variance judging requires external data processing approval")
    materiality_judge = OpenAICompatibleEvaluatorJudge(llm_client=deconstructor.llm_client)
    material_variance_evaluator = DatasetMaterialVarianceJudge(
        materiality_judge,
        max_input_chars=settings.max_input_chars,
    )
    with signal_control.installed():
        async with deconstructor, materiality_judge, target:
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
            source_outcome_projection = getattr(target, "outcome_projection", None)
            evaluation_deconstructor = (
                ProjectedResponseSemanticDeconstructor(semantic_pipeline)
                if source_outcome_projection is not None
                else semantic_pipeline
            )
            runner = DatasetEvaluationRunner(
                DatasetAugmentationEngine(
                    evaluation_deconstructor,
                    semantic_pipeline,
                    semantic_pipeline,
                ),
                evaluation_deconstructor,
                evaluation_target,
                target_timeout_seconds=target_config.trial_timeout_seconds,
                allow_network_egress=target_config.allow_network_egress,
                evaluation_mode=run_config.evaluation_mode,
                source_outcome_projection=source_outcome_projection,
                material_variance_evaluator=material_variance_evaluator,
                max_concurrent_target_requests=run_config.concurrency,
            )
            for case_number, record in enumerate(records, start=1):
                plan_outcome_terminal_ids: set[str] = set()
                precomputed_augmentation = (
                    saved_augmentations.get(record.id) if saved_augmentations is not None else None
                )

                def checkpoint_plan_outcomes(
                    augmentation: DatasetAugmentationResult,
                    source: InteractionRecord = record,
                    case_position: int = case_number,
                    terminal_ids: set[str] = plan_outcome_terminal_ids,
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
                            terminal_ids.add(unit.id)
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
                    nonlocal actual_target_calls
                    if run_config.concurrency == 1:
                        stop_at_requested_boundary()
                    if trial_journal is not None:
                        trial_journal.start(unit)
                    progress_tracker.trial_started(case_number=case_position, unit=unit)
                    actual_target_calls += 1
                    task = asyncio.current_task()
                    if task is not None:
                        active_trials[task] = (case_position, unit)
                        signal_control.target_call_started(task)

                def trial_terminal(
                    unit: DatasetTrialUnit,
                    trial: DatasetEvaluationTrial,
                    case_position: int = case_number,
                ) -> None:
                    task = asyncio.current_task()
                    trial_was_started = task is not None and task in active_trials
                    if task is not None:
                        signal_control.target_call_finished(task)
                        active_trials.pop(task, None)
                    if trial_journal is not None:
                        trial_journal.finish(unit, trial)
                    if trial_was_started:
                        progress_tracker.trial_terminal(
                            case_number=case_position,
                            unit=unit,
                            trial=trial,
                        )
                    else:
                        progress_tracker.trial_skipped(
                            case_number=case_position,
                            unit=unit,
                        )
                    if run_config.concurrency == 1:
                        stop_at_requested_boundary()

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
                except DatasetSourcePreparationError as error:
                    if not isolate_source_preparation_failures:
                        raise
                    if active_trials:
                        raise AssertionError(
                            "source preparation failures must precede target delivery"
                        ) from error
                    if run_context is None:
                        raise AssertionError(
                            "source preparation failure evidence requires a run context"
                        ) from error
                    failure_event = build_source_preparation_failure_event(
                        record,
                        error,
                        repetitions=repetitions,
                        max_environment_api_calls=target_config.max_environment_api_calls,
                        planned_target_calls=target_config.planned_environment_api_calls,
                        run_context=run_context,
                    )
                    if any(
                        event.interaction_id == failure_event.interaction_id
                        for event in recorded_source_preparation_events
                    ):
                        raise ValueError("source preparation failure was recorded twice") from error
                    failure_event = persist_source_preparation_failure_event(
                        failure_event,
                        output_stream,
                        durable_flush,
                    )
                    failed_units = 0
                    units = [
                        DatasetTrialUnit(
                            interaction_id=record.id,
                            operator_id="current_baseline",
                            arm="original",
                            repetition=repetition,
                        )
                        for repetition in range(1, repetitions + 1)
                    ]
                    units.extend(
                        DatasetTrialUnit(
                            interaction_id=record.id,
                            operator_id=operator_reference.partition("@")[0],
                            arm="probe",
                            repetition=repetition,
                        )
                        for operator_reference in operator_ids
                        for repetition in range(1, repetitions + 1)
                    )
                    for unit in units:
                        if unit.id in plan_outcome_terminal_ids or (
                            trial_journal is not None and trial_journal.is_terminal(unit)
                        ):
                            continue
                        if trial_journal is not None:
                            trial_journal.terminal(
                                unit,
                                "errored",
                                failure_event.failure_category,
                            )
                        failed_units += 1
                    if failed_units:
                        progress_tracker.source_preparation_failed(
                            case_number=case_number,
                            failed_units=failed_units,
                            event=failure_event,
                        )
                    recorded_source_preparation_events.append(failure_event)
                    continue
                except (DatasetTargetDeliveryUncertain, asyncio.CancelledError):
                    signal_control.target_call_finished()
                    assert active_trials
                    for active_case_number, active_unit in active_trials.values():
                        if trial_journal is not None and not trial_journal.is_terminal(active_unit):
                            trial_journal.terminal(
                                active_unit,
                                "quarantined",
                                "target_delivery_or_cleanup_uncertain",
                            )
                        progress_tracker.trial_delivery_uncertain(
                            case_number=active_case_number,
                            unit=active_unit,
                        )
                    durable_flush()
                    active_trials.clear()
                    raise DatasetTargetDeliveryUncertain(
                        "target delivery is uncertain; environment quarantined and trial not "
                        "retried"
                    ) from None
                if run_config.concurrency > 1:
                    stop_at_requested_boundary()
                invariant_evaluation = (
                    evaluate_dataset_invariants(result, invariant_suite)
                    if invariant_suite is not None
                    else None
                )
                if invariant_evaluation is not None and invariant_evaluations is not None:
                    invariant_evaluations.append(invariant_evaluation)
                progress_tracker.record_usage(
                    target_calls=(initial_target_calls or 0) + actual_target_calls,
                    semantic_calls=None,
                    environment_calls=(
                        None
                        if initial_environment_calls is None
                        else initial_environment_calls
                        + actual_target_calls * target_config.environment_api_calls_per_trial
                    ),
                    tokens=None,
                )
                progress_tracker.emit(
                    status="running",
                    stage="evidence",
                    case_number=case_number,
                )
                output_stream.write(
                    json.dumps(
                        build_customer_evidence_record(
                            result,
                            repetitions=repetitions,
                            max_environment_api_calls=target_config.max_environment_api_calls,
                            planned_target_calls=target_config.planned_environment_api_calls,
                            run_context=run_context,
                            invariant_evaluation=invariant_evaluation,
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                durable_flush()
                results.append(result)
    if complete_progress:
        progress_tracker.emit(
            status="failed" if recorded_source_preparation_events else "completed",
            stage="terminal",
        )
    return tuple(results)
