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
    DatasetSemanticSettings,
    DatasetSourcePreparationError,
    DatasetTargetDeliveryUncertain,
    DatasetTrialUnit,
    EvaluatorModelPreflight,
    InteractionRecord,
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

from ul_cli.dataset_augmentation_ledger import DatasetAugmentationLedger
from ul_cli.dataset_campaign import DatasetCampaignPlan
from ul_cli.dataset_review import (
    DatasetEvidenceRunContext,
    DatasetSourcePreparationFailureEvidence,
)
from ul_cli.dataset_trial_journal import DatasetTrialJournal

from ..evidence.customer import (
    build_customer_evidence_record,
    build_source_preparation_failure_evidence,
)
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
    repetitions: int,
    max_environment_api_calls: int,
    planned_target_calls: int,
    target_timeout_seconds: float = 30.0,
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
    progress_next_commands: CampaignNextCommands | None = None,
    progress_runtime: CampaignProgressRuntime | None = None,
    complete_progress: bool = True,
    environment_calls_per_target_call: int = 1,
    isolate_source_preparation_failures: bool = True,
    source_preparation_failures: list[DatasetSourcePreparationFailureEvidence] | None = None,
) -> tuple[DatasetEvaluationResult, ...]:
    if environment_calls_per_target_call < 1:
        raise ValueError("environment calls per target call must be positive")
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
            target_call_budget=planned_target_calls,
            semantic_call_budget=semantic_call_budget,
            environment_call_budget=max_environment_api_calls,
            token_budget=token_budget,
            maximum_wall_time_seconds=(
                max(1, work_upper_bound) * target_timeout_seconds
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
    active_trial: tuple[int, DatasetTrialUnit] | None = None
    had_source_preparation_failure = False

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
                target_timeout_seconds=target_timeout_seconds,
                allow_network_egress=allow_network_egress,
                evaluation_mode=(
                    run_context.evaluation_mode
                    if run_context is not None and run_context.evaluation_mode is not None
                    else "variance"
                ),
                source_outcome_projection=source_outcome_projection,
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
                except DatasetSourcePreparationError as error:
                    had_source_preparation_failure = True
                    if not isolate_source_preparation_failures:
                        raise
                    if active_trial is not None:
                        raise AssertionError(
                            "source preparation failures must precede target delivery"
                        ) from error
                    if run_context is None:
                        raise AssertionError(
                            "source preparation failure evidence requires a run context"
                        ) from error
                    failure_evidence = build_source_preparation_failure_evidence(
                        record,
                        error,
                        repetitions=repetitions,
                        max_environment_api_calls=max_environment_api_calls,
                        planned_target_calls=planned_target_calls,
                        run_context=run_context,
                    )
                    output_stream.write(
                        json.dumps(
                            failure_evidence.model_dump(mode="json", exclude_none=True),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    durable_flush()
                    if source_preparation_failures is not None:
                        source_preparation_failures.append(failure_evidence)
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
                            trial_journal.terminal(unit, "errored", error.code)
                        failed_units += 1
                    if failed_units:
                        progress_tracker.source_preparation_failed(
                            case_number=case_number,
                            failed_units=failed_units,
                        )
                    continue
                except (DatasetTargetDeliveryUncertain, asyncio.CancelledError):
                    signal_control.target_call_finished()
                    assert active_trial is not None
                    active_case_number, active_unit = active_trial
                    if trial_journal is not None and not trial_journal.is_terminal(active_unit):
                        trial_journal.terminal(
                            active_unit,
                            "quarantined",
                            "target_delivery_or_cleanup_uncertain",
                        )
                    durable_flush()
                    progress_tracker.trial_delivery_uncertain(
                        case_number=active_case_number,
                        unit=active_unit,
                    )
                    active_trial = None
                    raise DatasetTargetDeliveryUncertain(
                        "target delivery is uncertain; environment quarantined and trial not "
                        "retried"
                    ) from None
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
                        + actual_target_calls * environment_calls_per_target_call
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
                results.append(result)
    if complete_progress:
        progress_tracker.emit(
            status="failed" if had_source_preparation_failure else "completed",
            stage="terminal",
        )
    return tuple(results)
