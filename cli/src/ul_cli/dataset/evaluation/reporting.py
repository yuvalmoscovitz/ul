from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

import typer
from ul import DatasetEvaluationResult
from ul.dataset_invariants import DatasetInvariantEvaluation, DatasetInvariantRule

from ul_cli.dataset.progress import CampaignProgressRuntime
from ul_cli.dataset.source_preparation import DatasetSourcePreparationFailureEvent
from ul_cli.dataset_augmentation_ledger import create_private_augmentation_ledger
from ul_cli.finding_adapters import FindingAdapterContext, adapt_dataset_finding_packages
from ul_cli.finding_reference import FindingReferenceContext, resolve_finding_reference_context

from ..presentation.evaluation import (
    dataset_invariant_exit_code,
    dataset_results_exit_code,
    print_dataset_results,
    result_needs_review,
)
from ..presentation.runtime import console
from ..storage.private_files import open_resume_descriptor
from .durable_run import PreparedDurableRun
from .preparation import PreparedDatasetEvaluation

_MAXIMUM_FINDING_SNAPSHOT_BYTES = 128_000_000


def write_finding_package_snapshot(
    finding_output: Path,
    results: tuple[DatasetEvaluationResult, ...],
    invariant_evaluations: tuple[DatasetInvariantEvaluation, ...],
    invariant_rules: tuple[DatasetInvariantRule, ...],
    *,
    campaign_id: str,
    reference_context: FindingReferenceContext,
) -> bool:
    invariant_evaluation_by_interaction = {
        evaluation.interaction_id: evaluation for evaluation in invariant_evaluations
    }
    serialized_packages: list[str] = []
    for result in results:
        packages = adapt_dataset_finding_packages(
            result,
            invariant_evaluation=invariant_evaluation_by_interaction.get(result.source.id),
            invariant_rules=invariant_rules,
            context=FindingAdapterContext(
                campaign_id=campaign_id,
                recorded_at=reference_context.recorded_at,
                reference_key=reference_context.key,
            ),
        )
        serialized_packages.extend(
            json.dumps(
                package.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for package in packages
        )
    snapshot = "".join(serialized_packages).encode("utf-8")
    if len(snapshot) > _MAXIMUM_FINDING_SNAPSHOT_BYTES:
        raise ValueError("finding package snapshot exceeds the 128 MB limit")
    replace_finding_package_snapshot(finding_output, snapshot)
    return any(result_needs_review(result) for result in results)


def replace_finding_package_snapshot(finding_output: Path, snapshot: bytes) -> None:
    lock_output = finding_output.with_name(f".{finding_output.name}.lock")
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    with suppress(FileExistsError):
        lock_descriptor = os.open(
            lock_output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow_flag, 0o600
        )
        os.close(lock_descriptor)
    lock_descriptor = open_resume_descriptor(lock_output, writable=True)
    temporary_output: Path | None = None
    try:
        temporary_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{finding_output.name}.tmp-", dir=finding_output.parent
        )
        temporary_output = Path(temporary_name)
        try:
            if sys.platform != "win32":
                os.fchmod(temporary_descriptor, 0o600)
            remaining = memoryview(snapshot)
            while remaining:
                written = os.write(temporary_descriptor, remaining)
                if written == 0:
                    raise OSError("finding package snapshot write was incomplete")
                remaining = remaining[written:]
            os.fsync(temporary_descriptor)
        finally:
            os.close(temporary_descriptor)
        os.replace(temporary_output, finding_output)
        temporary_output = None
        _fsync_directory(finding_output.parent)
    finally:
        if temporary_output is not None:
            with suppress(OSError):
                temporary_output.unlink()
        os.close(lock_descriptor)


def report_completed_resume(
    evaluation: PreparedDatasetEvaluation,
    durable: PreparedDurableRun,
) -> None:
    request = evaluation.request
    output = request.output
    resume_evidence = durable.resume_evidence
    assert output is not None
    assert resume_evidence is not None
    if request.augmentations_output is not None:
        try:
            if request.augmentations_output.exists():
                for prior_result in resume_evidence.technical_results:
                    if (
                        durable.saved_augmentations.get(prior_result.source.id)
                        != prior_result.augmentation
                    ):
                        raise ValueError(
                            "augmentation ledger does not match completed evaluation evidence"
                        )
            else:
                with create_private_augmentation_ledger(
                    request.augmentations_output,
                    generation_context=evaluation.augmentation_generation_context,
                    selected_records=durable.all_records,
                ) as completed_ledger:
                    for prior_result in resume_evidence.technical_results:
                        completed_ledger.append(
                            source=prior_result.source, augmentation=prior_result.augmentation
                        )
        except (OSError, ValueError) as error:
            message = str(error) if isinstance(error, ValueError) else error.__class__.__name__
            raise typer.BadParameter(
                f"cannot safely persist augmentations ({message})",
                param_hint="--augmentations-output",
            ) from None
    assert evaluation.run_context is not None
    finding_output = output.with_name(f"{output.name}.findings.jsonl")
    try:
        reference_context = resolve_finding_reference_context(finding_output)
        write_finding_package_snapshot(
            finding_output,
            resume_evidence.technical_results,
            resume_evidence.invariant_evaluations,
            evaluation.invariant_suite.rules if evaluation.invariant_suite is not None else (),
            campaign_id=evaluation.run_context.context_sha256,
            reference_context=reference_context,
        )
    except (OSError, ValueError) as error:
        message = str(error) if isinstance(error, ValueError) else error.__class__.__name__
        raise typer.BadParameter(
            f"cannot safely reconcile finding packages ({message})", param_hint="--resume"
        ) from None
    console.print(
        f"Resume compatible: all {durable.skipped_count} selected interaction(s) are complete in "
        f"{output}. Nothing to do."
    )
    failure_count = len(resume_evidence.source_preparation_failures)
    if failure_count:
        console.print(
            f"Source preparation failures: {failure_count}; no target calls were made for those "
            "sources."
        )
    exit_code = dataset_invariant_exit_code(resume_evidence.invariant_evaluations)
    if exit_code:
        raise typer.Exit(code=exit_code)
    if resume_evidence.has_review_findings:
        raise typer.Exit(code=1)
    if resume_evidence.has_inconclusive_materiality or failure_count:
        raise typer.Exit(code=2)
    if dataset_results_exit_code(resume_evidence.technical_results):
        raise typer.Exit(code=2)
    raise typer.Exit(code=0)


def report_campaign_results(
    evaluation: PreparedDatasetEvaluation,
    durable: PreparedDurableRun,
    progress_runtime: CampaignProgressRuntime,
    results: tuple[DatasetEvaluationResult, ...],
    invariant_evaluations: tuple[DatasetInvariantEvaluation, ...],
    source_preparation_events: tuple[DatasetSourcePreparationFailureEvent, ...],
) -> None:
    output = evaluation.request.output
    assert output is not None
    if durable.skipped_count > 0:
        console.print(
            f"Resumed: {durable.skipped_count} interaction(s) skipped (already in evidence), "
            f"{len(results)} newly evaluated."
        )
    progress_runtime.tracker.emit(status="running", stage="report")
    prior = durable.resume_evidence
    all_failures = (prior.source_preparation_failures if prior is not None else ()) + tuple(
        event.evidence for event in source_preparation_events
    )
    try:
        print_dataset_results(
            results,
            output,
            augmentations_output=evaluation.request.augmentations_output,
            invariant_evaluations=invariant_evaluations,
            show_report_guidance=evaluation.request.requested.show_report_guidance,
            source_preparation_failure_count=len(all_failures),
        )
    except Exception:
        progress_runtime.tracker.emit(status="failed", stage="terminal")
        raise
    progress_runtime.tracker.emit(
        status="failed" if all_failures else "completed", stage="terminal"
    )
    prior_invariants = prior.invariant_evaluations if prior is not None else ()
    invariant_exit_code = dataset_invariant_exit_code((*prior_invariants, *invariant_evaluations))
    if invariant_exit_code:
        raise typer.Exit(code=invariant_exit_code)
    all_results = (*(prior.technical_results if prior is not None else ()), *results)
    semantic_exit_code = dataset_results_exit_code(all_results)
    if semantic_exit_code:
        raise typer.Exit(code=semantic_exit_code)
    if all_failures:
        raise typer.Exit(code=2)


def _fsync_directory(directory: Path) -> None:
    if sys.platform == "win32":
        return
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
