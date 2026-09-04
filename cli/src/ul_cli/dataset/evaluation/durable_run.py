from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from ul import DatasetAugmentationResult, InteractionRecord
from ul.dataset_invariants import DatasetInvariantSuite

from ul_cli.dataset.evidence.persistence import (
    create_durable_evidence_output,
    read_resume_evidence,
)
from ul_cli.dataset_augmentation_ledger import (
    DatasetAugmentationGenerationContext,
    read_augmentation_ledger,
)
from ul_cli.dataset_review import DatasetEvidenceRunContext, DatasetResumeEvidence
from ul_cli.dataset_trial_journal import (
    DatasetRunManifest,
    DatasetTrialJournal,
    DatasetTrialJournalSnapshot,
    create_dataset_trial_journal,
    create_quarantine_resolution,
    fsync_run_directory,
    journal_path,
    manifest_path,
    open_dataset_trial_journal,
    persist_dataset_run_manifest,
    persist_quarantine_resolution,
    quarantine_resolution_path,
    read_dataset_run_manifest,
    read_quarantine_resolution,
)

from .resume_compatibility import (
    effective_command_incompatibility_reason,
    manifest_incompatibility_reason,
)

_SOURCE_PREPARATION_REASON_CODES = {
    "source_semantic_preparation_failed",
    "source_outcome_projection_failed",
    "source_comparison_surface_incompatible",
}


class DurableRunPreparationError(ValueError):
    def __init__(
        self,
        phase: Literal["journal", "augmentation_input", "resume_evidence"],
        cause: OSError | ValueError,
    ) -> None:
        super().__init__(str(cause) if isinstance(cause, ValueError) else cause.__class__.__name__)
        self.phase = phase


@dataclass
class PreparedDurableRun:
    trial_journal: DatasetTrialJournal | None
    resume_evidence: DatasetResumeEvidence | None
    all_records: tuple[InteractionRecord, ...]
    remaining_records: tuple[InteractionRecord, ...]
    saved_augmentations: dict[str, DatasetAugmentationResult]
    augmentations_input_sha256: str | None
    skipped_count: int

    def close(self) -> None:
        if self.trial_journal is not None:
            self.trial_journal.close()


def prepare_durable_run(
    *,
    resume: Path | None,
    output: Path | None,
    recorded_manifest: DatasetRunManifest | None,
    run_context: DatasetEvidenceRunContext | None,
    selected_records: tuple[InteractionRecord, ...],
    augmentation_generation_context: DatasetAugmentationGenerationContext,
    augmentations_input: Path | None,
    augmentations_output: Path | None,
    invariant_suite: DatasetInvariantSuite | None,
) -> PreparedDurableRun:
    trial_journal: DatasetTrialJournal | None = None
    if recorded_manifest is not None:
        assert resume is not None
        try:
            trial_journal = open_dataset_trial_journal(journal_path(resume), recorded_manifest)
        except (OSError, ValueError) as error:
            raise DurableRunPreparationError("journal", error) from None

    try:
        saved_augmentations, augmentations_input_sha256 = _load_complete_augmentation_input(
            augmentations_input,
            expected_context=augmentation_generation_context,
            selected_records=selected_records,
            recorded_sha256=(
                recorded_manifest.effective_command.augmentations_input_sha256
                if recorded_manifest is not None
                else None
            ),
        )
    except (OSError, ValueError) as error:
        _close_trial_journal(trial_journal)
        raise DurableRunPreparationError("augmentation_input", error) from None

    resume_evidence: DatasetResumeEvidence | None = None
    remaining_records = selected_records
    skipped_count = 0
    if resume is not None:
        assert output is not None
        assert run_context is not None
        try:
            resume_evidence = read_resume_evidence(
                output,
                expected_context=run_context,
                selected_records=selected_records,
                invariant_suite=invariant_suite,
            )
            if trial_journal is not None:
                _reconcile_source_preparation_failures(trial_journal, resume_evidence)
            if augmentations_output is not None and augmentations_output.exists():
                augmentation_snapshot = read_augmentation_ledger(
                    augmentations_output,
                    expected_context=augmentation_generation_context,
                    selected_records=selected_records,
                )
                saved_augmentations = {
                    record.source.id: record.augmentation
                    for record in augmentation_snapshot.records
                }
                _validate_augmentation_evidence(saved_augmentations, resume_evidence)
        except (OSError, ValueError) as error:
            _close_trial_journal(trial_journal)
            raise DurableRunPreparationError("resume_evidence", error) from None
        remaining_records = tuple(
            record for record in selected_records if record.id not in resume_evidence.processed_ids
        )
        skipped_count = len(resume_evidence.processed_ids)

    return PreparedDurableRun(
        trial_journal=trial_journal,
        resume_evidence=resume_evidence,
        all_records=selected_records,
        remaining_records=remaining_records,
        saved_augmentations=saved_augmentations,
        augmentations_input_sha256=augmentations_input_sha256,
        skipped_count=skipped_count,
    )


def create_or_validate_durable_state(
    *,
    prepared: PreparedDurableRun,
    resume: Path | None,
    output: Path,
    expected_manifest: DatasetRunManifest,
    resolve_quarantine_after: Literal["environment-reset", "environment-replacement"] | None,
) -> None:
    run_manifest_path = manifest_path(output)
    if resume is None:
        persist_dataset_run_manifest(run_manifest_path, expected_manifest)
        prepared.trial_journal = create_dataset_trial_journal(
            journal_path(output), expected_manifest
        )
        create_durable_evidence_output(output, expected_manifest.manifest_sha256)
        fsync_run_directory(output)
        return
    if not run_manifest_path.exists():
        return

    recorded_manifest = read_dataset_run_manifest(run_manifest_path)
    incompatibility = manifest_incompatibility_reason(
        recorded_manifest.run_context, expected_manifest.run_context
    )
    if incompatibility is not None:
        raise ValueError(f"resume_incompatible:{incompatibility}")
    if recorded_manifest != expected_manifest:
        effective_incompatibility = effective_command_incompatibility_reason(
            recorded_manifest, expected_manifest
        )
        raise ValueError(f"resume_incompatible:{effective_incompatibility or 'effective_command'}")
    if prepared.trial_journal is None:
        raise ValueError("resume journal lock was not acquired")
    quarantined_unit_ids = prepared.trial_journal.snapshot.quarantined_unit_ids
    if quarantined_unit_ids:
        resolution_path = quarantine_resolution_path(output)
        if resolution_path.exists():
            resolution = read_quarantine_resolution(resolution_path)
        elif resolve_quarantine_after is not None:
            resolution = create_quarantine_resolution(
                recorded_manifest,
                quarantined_unit_ids,
                resolve_quarantine_after,
                datetime.now(UTC),
            )
            persist_quarantine_resolution(resolution_path, resolution)
        else:
            raise ValueError(
                "resume_quarantined:target_delivery_or_cleanup_uncertain; after an operator has "
                "reset or replaced the recorded test environment, attest with "
                "--resolve-quarantine-after environment-reset (or environment-replacement). UL "
                "records but cannot independently verify this cleanup"
            )
        if (
            resolution.manifest_sha256 != recorded_manifest.manifest_sha256
            or resolution.target_sha256 != recorded_manifest.run_context.target.sha256
            or frozenset(resolution.quarantined_unit_ids) != quarantined_unit_ids
        ):
            raise ValueError("resume_quarantined:cleanup_attestation_does_not_match_campaign")
    elif resolve_quarantine_after is not None:
        raise ValueError("resume_incompatible:no_quarantined_trials_to_resolve")
    validate_resume_augmentation_durability(
        recorded_manifest, prepared.trial_journal, prepared.resume_evidence
    )


def _load_complete_augmentation_input(
    path: Path | None,
    *,
    expected_context: DatasetAugmentationGenerationContext,
    selected_records: tuple[InteractionRecord, ...],
    recorded_sha256: str | None,
) -> tuple[dict[str, DatasetAugmentationResult], str | None]:
    if path is None:
        return {}, None
    snapshot = read_augmentation_ledger(
        path,
        expected_context=expected_context,
        selected_records=selected_records,
    )
    expected_source_ids = frozenset(record.id for record in selected_records)
    if snapshot.processed_source_ids != expected_source_ids:
        raise ValueError("augmentation input must contain every selected interaction exactly once")
    if recorded_sha256 is not None and recorded_sha256 != snapshot.raw_ledger_sha256:
        raise ValueError("augmentation input digest does not match the recorded campaign")
    return (
        {record.source.id: record.augmentation for record in snapshot.records},
        snapshot.raw_ledger_sha256,
    )


def restore_recorded_augmentation_input(
    requested: Path | None,
    recorded_manifest: DatasetRunManifest | None,
) -> Path | None:
    if requested is not None or recorded_manifest is None:
        return requested
    recorded_path = recorded_manifest.effective_command.augmentations_input_path
    return Path(recorded_path) if recorded_path is not None else None


def _close_trial_journal(trial_journal: DatasetTrialJournal | None) -> None:
    if trial_journal is not None:
        trial_journal.close()


def validate_resume_augmentation_durability(
    manifest: DatasetRunManifest,
    trial_journal: DatasetTrialJournal,
    resume_evidence: DatasetResumeEvidence | None,
) -> None:
    command = manifest.effective_command
    if command.save_augmentations or command.augmentations_input_path is not None:
        return
    source_failure_ids = {
        failure.interaction_id
        for failure in (
            resume_evidence.source_preparation_failures if resume_evidence is not None else ()
        )
    }
    terminal_interaction_ids = {
        unit.interaction_id
        for unit in manifest.work_plan
        if unit.id in trial_journal.snapshot.terminal_states
    }
    if terminal_interaction_ids - source_failure_ids:
        raise ValueError(
            "resume_incompatible:augmentation_not_durable; start a new output with "
            "augmentation retention enabled"
        )


def _reconcile_source_preparation_failures(
    trial_journal: DatasetTrialJournal,
    resume_evidence: DatasetResumeEvidence,
) -> None:
    snapshot = trial_journal.snapshot
    failures_by_interaction_id = {
        failure.interaction_id: failure for failure in resume_evidence.source_preparation_failures
    }
    for unit in trial_journal.manifest.work_plan:
        failure = failures_by_interaction_id.get(unit.interaction_id)
        if failure is None:
            continue
        terminal_state = snapshot.terminal_states.get(unit.id)
        if terminal_state is None:
            trial_journal.terminal(unit, "errored", failure.reason_code)
            continue
        reason_code = snapshot.terminal_reason_codes.get(unit.id)
        if terminal_state == "errored" and reason_code == failure.reason_code:
            continue
        if unit.arm == "probe" and terminal_state in {"inapplicable", "rejected"}:
            continue
        raise ValueError("source failure evidence conflicts with durable trial state")


def attempted_target_calls(snapshot: DatasetTrialJournalSnapshot) -> int:
    return sum(
        state in {"completed", "errored", "inconclusive", "quarantined"}
        and snapshot.terminal_reason_codes.get(unit_id) not in _SOURCE_PREPARATION_REASON_CODES
        for unit_id, state in snapshot.terminal_states.items()
    )


def _validate_augmentation_evidence(
    saved_augmentations: dict[str, DatasetAugmentationResult],
    resume_evidence: DatasetResumeEvidence,
) -> None:
    for prior_result in resume_evidence.technical_results:
        if saved_augmentations.get(prior_result.source.id) != prior_result.augmentation:
            raise ValueError("augmentation ledger does not match completed evaluation evidence")
