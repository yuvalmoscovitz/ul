from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
import typer
from ul import (
    DatasetEvaluationResult,
    EvaluatorModelCompatibilityError,
    EvaluatorModelPreflight,
    ProviderDiagnosticError,
)
from ul.dataset_invariants import DatasetInvariantEvaluation
from ul.http_environment import JsonHttpEnvironmentConnection

from ul_cli.dataset.progress import (
    CampaignControlRequested,
    CampaignProgressRuntime,
    create_campaign_next_commands,
    create_campaign_progress_runtime,
)
from ul_cli.dataset.source_preparation import DatasetSourcePreparationFailureEvent
from ul_cli.environment import TEST_ENVIRONMENT_CONFIRMATION_MESSAGE

from ..evidence.persistence import (
    persist_evaluator_preflight,
    write_provider_diagnostic,
)
from ..presentation.evaluation import print_evaluator_preflight, print_fixture_identity
from ..presentation.runtime import console, print_dataset_plain
from .campaign import PreparedCampaign
from .campaign_persistence import open_campaign_persistence
from .durable_run import attempted_target_calls
from .reporting import write_finding_package_snapshot
from .runner import evaluate_interaction_records, preflight_evaluator
from .target_preparation import print_local_target_identity


@dataclass(frozen=True)
class CampaignExecutionOutcome:
    progress_runtime: CampaignProgressRuntime
    results: tuple[DatasetEvaluationResult, ...]
    invariant_evaluations: tuple[DatasetInvariantEvaluation, ...]
    source_preparation_events: tuple[DatasetSourcePreparationFailureEvent, ...]


def execute_campaign(campaign: PreparedCampaign) -> CampaignExecutionOutcome:
    evaluation = campaign.evaluation
    durable = campaign.durable
    request = evaluation.request
    requested = request.requested
    output = request.output
    if evaluation.target_config is None and evaluation.local_target is None:
        raise typer.BadParameter(
            "execution requires a recorded or explicit target", param_hint="--target"
        )
    if not request.confirm_test_environment:
        raise typer.BadParameter(
            TEST_ENVIRONMENT_CONFIRMATION_MESSAGE, param_hint="--confirm-test-environment"
        )
    if output is None:
        raise typer.BadParameter("execution requires --output", param_hint="--output")
    assert evaluation.run_context is not None
    assert evaluation.run_context.fixture is not None
    print_fixture_identity(
        evaluation.run_context.fixture.status,
        fixture_id=evaluation.run_context.fixture.id,
        fixture_version=evaluation.run_context.fixture.version,
    )
    if not evaluation.settings.live_calls:
        raise typer.BadParameter(
            "set UL_LIVE=true (or UL_DATASET_LIVE_CALLS=true) to allow semantic model calls"
        )
    if not evaluation.settings.allow_external_data_processing:
        raise typer.BadParameter(
            "set UL_LIVE=true (or UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING=true) "
            "to allow semantic model calls"
        )
    if evaluation.settings.api_key_required and (
        evaluation.settings.api_key is None
        or not evaluation.settings.api_key.get_secret_value().strip()
    ):
        raise typer.BadParameter(
            f"set {evaluation.settings.api_key_environment_variable} to run an evaluation"
        )
    try:
        if evaluation.local_target is not None:
            print_local_target_identity(evaluation.local_target)
            if requested.confirm_target != evaluation.local_target.confirmation_sha256:
                raise ValueError(
                    "local execution requires --confirm-target with the exact displayed digest"
                )
            evaluation.local_target.revalidate_identity()
            execution_target = evaluation.local_target.create_connection(
                campaign.plan.calls.total_environment_api,
                evaluation.local_target.maximum_active_target_seconds,
            )
        else:
            assert evaluation.target_config is not None
            if (
                evaluation.http_target is not None
                and requested.confirm_target != evaluation.http_target.confirmation_sha256
            ):
                raise ValueError(
                    "HTTP execution requires --confirm-target with the exact displayed digest"
                )
            if not evaluation.run_config.target.allow_network_egress:
                raise ValueError("environment execution requires --allow-environment-network")
            execution_target = JsonHttpEnvironmentConnection.from_config(
                evaluation.target_config,
                test_environment_confirmed=evaluation.run_config.target.test_environment_confirmed,
                allow_insecure_http=evaluation.run_config.target.allow_insecure_http,
                timeout_seconds=evaluation.run_config.target.trial_timeout_seconds,
                max_environment_api_calls=evaluation.run_config.target.max_environment_api_calls,
            )
    except ValueError as error:
        raise typer.BadParameter(
            str(error),
            param_hint=(
                "--target"
                if evaluation.local_target is not None or evaluation.http_target is not None
                else "--environment-config"
            ),
        ) from None
    invariant_evaluations: list[DatasetInvariantEvaluation] = []
    source_preparation_events: list[DatasetSourcePreparationFailureEvent] = []
    progress_runtime: CampaignProgressRuntime | None = None
    try:
        progress_runtime = _create_progress_runtime(campaign)
        evaluator_preflight, _evaluator_preflight_receipt = _run_preflight(
            campaign, progress_runtime
        )
        with open_campaign_persistence(campaign) as persistence:
            output_stream = persistence.output_stream
            parameters = inspect.signature(evaluate_interaction_records).parameters
            accepts_extra = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
            )
            extra: dict[str, object] = {}
            for name, value in (
                ("progress_plan", campaign.plan),
                ("progress_runtime", progress_runtime),
                ("complete_progress", False),
                ("trial_journal", durable.trial_journal),
                ("progress_json", True if requested.progress_json else None),
                ("source_preparation_events", source_preparation_events),
            ):
                if value is not None and (name in parameters or accepts_extra):
                    extra[name] = value
            runner = cast(Any, evaluate_interaction_records)
            arguments = dict(
                run_config=evaluation.run_config,
                run_context=evaluation.run_context,
                augmentation_ledger=persistence.augmentation_ledger,
                saved_augmentations=durable.saved_augmentations,
                redaction_engine=evaluation.redaction_engine,
                evaluator_preflight=evaluator_preflight,
                **extra,
            )
            if evaluation.invariant_suite is not None:
                arguments.update(
                    invariant_suite=evaluation.invariant_suite,
                    invariant_evaluations=invariant_evaluations,
                )
            results = asyncio.run(
                runner(
                    durable.remaining_records,
                    evaluation.selected_operator_ids,
                    evaluation.settings,
                    execution_target,
                    output_stream,
                    **arguments,
                )
            )
            prior = durable.resume_evidence
            write_finding_package_snapshot(
                output.with_name(f"{output.name}.findings.jsonl"),
                (*(prior.technical_results if prior is not None else ()), *results),
                (
                    *(prior.invariant_evaluations if prior is not None else ()),
                    *invariant_evaluations,
                ),
                evaluation.invariant_suite.rules if evaluation.invariant_suite is not None else (),
                campaign_id=evaluation.run_context.context_sha256,
                reference_context=persistence.finding_reference_context,
            )
    except typer.Exit:
        raise
    except CampaignControlRequested:
        raise typer.Exit(code=130) from None
    except (TimeoutError, RuntimeError, ValueError, httpx.HTTPError) as error:
        if progress_runtime is not None and not progress_runtime.tracker.terminal_emitted:
            progress_runtime.tracker.emit(status="failed", stage="terminal")
        if isinstance(error, ProviderDiagnosticError):
            console.print(str(error))
            try:
                diagnostic_output = write_provider_diagnostic(output, error)
            except OSError as diagnostic_error:
                print_dataset_plain(
                    "Sanitized provider diagnostics could not be written "
                    f"({diagnostic_error.__class__.__name__})."
                )
            else:
                print_dataset_plain(f"Sanitized provider diagnostics: {diagnostic_output}")
        else:
            console.print(f"Evaluation stopped ({error.__class__.__name__}).")
        console.print(f"Complete results written before the error remain in {output}.")
        if request.augmentations_output is not None:
            print_dataset_plain(
                f"Generated augmentations remain in {request.augmentations_output} and will be "
                f"reused with --resume {output}."
            )
        raise typer.Exit(code=2) from None
    finally:
        asyncio.run(execution_target.aclose())
    assert progress_runtime is not None
    return CampaignExecutionOutcome(
        progress_runtime,
        results,
        tuple(invariant_evaluations),
        tuple(source_preparation_events),
    )


def _create_progress_runtime(campaign: PreparedCampaign) -> CampaignProgressRuntime:
    evaluation = campaign.evaluation
    request = evaluation.request
    assert request.output is not None
    resume_argv = None
    if evaluation.local_target is not None:
        assert request.requested.target is not None
        target_path = Path(request.requested.target)
        action_target = (
            str(target_path.resolve()) if target_path.is_file() else request.requested.target
        )
        argv = [
            "ul",
            "dataset",
            "evaluate",
            "--resume",
            str(request.output.resolve()),
            "--target",
            action_target,
            "--confirm-target",
            evaluation.local_target.confirmation_sha256,
        ]
        for artifact in request.requested.target_artifacts:
            argv.extend(("--target-artifact", str(artifact.resolve())))
        resume_argv = tuple(argv)
    runtime = create_campaign_progress_runtime(
        case_count=len(campaign.durable.all_records),
        work_upper_bound=(
            len(campaign.manifest.work_plan)
            if campaign.manifest is not None
            else len(campaign.durable.remaining_records)
            * request.repetitions
            * (1 + len(evaluation.selected_operator_ids))
        ),
        target_call_budget=campaign.plan.calls.total_environment_api,
        semantic_call_budget=campaign.plan.calls.total_semantic_model,
        environment_call_budget=evaluation.run_config.target.max_environment_api_calls,
        token_budget=campaign.plan.tokens.maximum,
        maximum_wall_time_seconds=campaign.plan.timing.maximum_wall_time_seconds,
        next_commands=create_campaign_next_commands(request.output, resume_argv=resume_argv),
        json_output=request.requested.progress_json,
    )
    if campaign.durable.trial_journal is not None:
        snapshot = campaign.durable.trial_journal.snapshot
        runtime.tracker.hydrate_terminal_states(snapshot.terminal_states)
        runtime.tracker.record_usage(
            target_calls=attempted_target_calls(snapshot),
            semantic_calls=None,
            environment_calls=0 if not snapshot.terminal_states else None,
            tokens=None,
        )
    return runtime


def _run_preflight(
    campaign: PreparedCampaign, runtime: CampaignProgressRuntime
) -> tuple[EvaluatorModelPreflight, Path]:
    preflight = campaign.evaluator_preflight
    receipt = campaign.evaluator_preflight_receipt
    assert campaign.evaluation.request.output is not None
    runtime.tracker.emit(status="running", stage="preflight")
    if preflight is None:
        try:
            with runtime.signal_control.installed():
                preflight = asyncio.run(preflight_evaluator(campaign.evaluation.settings))
            receipt = persist_evaluator_preflight(campaign.evaluation.request.output, preflight)
            if not runtime.tracker.safe_boundary(
                runtime.control,
                lambda: (
                    campaign.durable.trial_journal.flush()
                    if campaign.durable.trial_journal
                    else None
                ),
            ):
                raise typer.Exit(code=130)
        except EvaluatorModelCompatibilityError as error:
            runtime.tracker.emit(status="failed", stage="terminal")
            print_dataset_plain(f"Evaluation stopped before campaign execution: {error}")
            raise typer.Exit(code=2) from None
        except (OSError, ValueError) as error:
            runtime.tracker.emit(status="failed", stage="terminal")
            message = str(error) if isinstance(error, ValueError) else error.__class__.__name__
            raise typer.BadParameter(
                f"cannot safely persist evaluator preflight ({message})", param_hint="--output"
            ) from None
    assert receipt is not None
    print_evaluator_preflight(preflight, receipt)
    return preflight, receipt
