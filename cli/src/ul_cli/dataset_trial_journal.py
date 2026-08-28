from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from ul import (
    DatasetEvaluationTrial,
    DatasetInvariantSuite,
    DatasetTargetLifecycleFailure,
    DatasetTrialUnit,
    InteractionRecord,
    RedactionPolicy,
)
from ul.http_environment import JsonHttpTargetConfig

from ul_cli.dataset.storage.private_files import open_resume_descriptor
from ul_cli.dataset_review import DatasetEvidenceRunContext
from ul_cli.http_target_resolution import HttpTargetConfirmation

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_MAXIMUM_MANIFEST_BYTES = 32_000_000
_MAXIMUM_JOURNAL_BYTES = 256_000_000
_MAXIMUM_JOURNAL_RECORD_BYTES = 8_000_000
_MAXIMUM_REPETITIONS = 100

TrialState = Literal[
    "planned",
    "running",
    "completed",
    "skipped",
    "inapplicable",
    "rejected",
    "discarded",
    "errored",
    "inconclusive",
    "quarantined",
]
_TERMINAL_STATES = frozenset(
    {
        "completed",
        "skipped",
        "inapplicable",
        "rejected",
        "discarded",
        "errored",
        "inconclusive",
        "quarantined",
    }
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DatasetRunEffectiveCommand(_StrictModel):
    repetitions: int = Field(ge=1, le=_MAXIMUM_REPETITIONS)
    target_timeout_seconds: float = Field(default=30.0, gt=0, le=3_600)
    max_environment_api_calls: int = Field(ge=1)
    allow_environment_network: bool
    confirm_test_environment: bool
    allow_insecure_http: bool
    save_augmentations: bool
    semantic_provider_type: Literal["openrouter", "openai-compatible"] = "openrouter"
    semantic_base_url: str = Field(default="https://openrouter.ai/api/v1", min_length=1)
    semantic_live_calls: bool = False
    semantic_allow_external_data_processing: bool = False
    invariant_suite_snapshot: DatasetInvariantSuite | None = None
    invariant_suite_source: str | None = None
    redaction_policy_snapshot: RedactionPolicy | None = None
    redaction_policy_source: str | None = None
    redaction_state_path: str | None = None
    redaction_state_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    augmentations_output_path: str | None = None
    http_target_confirmation: HttpTargetConfirmation | None = None
    http_target_config: JsonHttpTargetConfig | None = None


class DatasetRunManifest(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    run_context: DatasetEvidenceRunContext
    selected_records: tuple[InteractionRecord, ...] = Field(min_length=1)
    selected_operator_ids: tuple[str, ...] = Field(min_length=1)
    effective_command: DatasetRunEffectiveCommand
    work_plan: tuple[DatasetTrialUnit, ...] = Field(min_length=1)
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_manifest(self) -> DatasetRunManifest:
        if len({record.id for record in self.selected_records}) != len(self.selected_records):
            raise ValueError("manifest contains duplicate interaction ids")
        if len(set(self.selected_operator_ids)) != len(self.selected_operator_ids):
            raise ValueError("manifest contains duplicate operators")
        recorded_operator_ids = tuple(operator.id for operator in self.run_context.operators)
        selected_operator_ids = tuple(
            operator_id.partition("@")[0] for operator_id in self.selected_operator_ids
        )
        if selected_operator_ids != recorded_operator_ids:
            raise ValueError("manifest operators do not match its run context")
        if self.effective_command.repetitions != self.run_context.repetitions:
            raise ValueError("manifest repetitions do not match its run context")
        unit_ids = tuple(unit.id for unit in self.work_plan)
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("manifest contains duplicate trial units")
        expected_units = _materialize_work_plan(
            self.selected_records,
            self.selected_operator_ids,
            self.effective_command.repetitions,
        )
        if self.work_plan != expected_units:
            raise ValueError("manifest work plan does not match its effective command")
        expected_digest = _canonical_sha256(
            self.model_dump(mode="json", exclude={"manifest_sha256"})
        )
        if self.manifest_sha256 != expected_digest:
            raise ValueError("manifest digest does not match its canonical content")
        return self


class DatasetTrialJournalRecord(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    sequence: int = Field(ge=1)
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    unit: DatasetTrialUnit
    state: TrialState
    trial: DatasetEvaluationTrial | None = None
    reason_code: str | None = Field(default=None, min_length=1, max_length=100)
    previous_record_sha256: str = Field(pattern=_SHA256_PATTERN)
    record_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_record(self) -> DatasetTrialJournalRecord:
        if self.state in {"planned", "running"} and self.trial is not None:
            raise ValueError("non-terminal journal records cannot include trial evidence")
        if self.state == "completed" and self.trial is None:
            raise ValueError("completed journal records require trial evidence")
        expected_digest = _canonical_sha256(self.model_dump(mode="json", exclude={"record_sha256"}))
        if self.record_sha256 != expected_digest:
            raise ValueError("journal record digest does not match its canonical content")
        return self


class DatasetTrialJournalAnchor(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    record_count: int = Field(ge=0)
    last_record_sha256: str = Field(pattern=_SHA256_PATTERN)
    anchor_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_anchor(self) -> DatasetTrialJournalAnchor:
        expected = _canonical_sha256(self.model_dump(mode="json", exclude={"anchor_sha256"}))
        if self.anchor_sha256 != expected:
            raise ValueError("trial journal anchor digest does not match")
        return self


class DatasetQuarantineResolution(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    target_sha256: str = Field(pattern=_SHA256_PATTERN)
    quarantined_unit_ids: tuple[str, ...] = Field(min_length=1)
    operator_attestation: Literal["environment-reset", "environment-replacement"]
    resolved_at: datetime
    independently_verified: Literal[False] = False
    resolution_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_resolution(self) -> DatasetQuarantineResolution:
        if tuple(sorted(set(self.quarantined_unit_ids))) != self.quarantined_unit_ids:
            raise ValueError("quarantine resolution unit ids must be unique and sorted")
        expected = _canonical_sha256(self.model_dump(mode="json", exclude={"resolution_sha256"}))
        if self.resolution_sha256 != expected:
            raise ValueError("quarantine resolution digest does not match")
        return self


@dataclass(frozen=True)
class DatasetTrialJournalSnapshot:
    recovered_trials: dict[str, DatasetEvaluationTrial]
    terminal_states: dict[str, TrialState]
    terminal_reason_codes: dict[str, str | None]
    quarantined_unit_ids: frozenset[str]


def manifest_path(evidence_output: Path) -> Path:
    return evidence_output.with_name(f"{evidence_output.name}.manifest.json")


def journal_path(evidence_output: Path) -> Path:
    return evidence_output.with_name(f"{evidence_output.name}.trials.jsonl")


def journal_anchor_path(evidence_output: Path) -> Path:
    journal = journal_path(evidence_output)
    return journal.with_name(f"{journal.name}.anchor.json")


def quarantine_resolution_path(evidence_output: Path) -> Path:
    return evidence_output.with_name(f"{evidence_output.name}.quarantine-resolution.json")


def create_dataset_run_manifest(
    *,
    run_context: DatasetEvidenceRunContext,
    selected_records: tuple[InteractionRecord, ...],
    selected_operator_ids: tuple[str, ...],
    repetitions: int,
    target_timeout_seconds: float = 30.0,
    max_environment_api_calls: int,
    allow_environment_network: bool,
    confirm_test_environment: bool,
    allow_insecure_http: bool,
    save_augmentations: bool,
    semantic_provider_type: Literal["openrouter", "openai-compatible"] = "openrouter",
    semantic_base_url: str = "https://openrouter.ai/api/v1",
    semantic_live_calls: bool = False,
    semantic_allow_external_data_processing: bool = False,
    invariant_suite_snapshot: DatasetInvariantSuite | None = None,
    invariant_suite_source: str | None = None,
    redaction_policy_snapshot: RedactionPolicy | None = None,
    redaction_policy_source: str | None = None,
    redaction_state_path: str | None = None,
    redaction_state_sha256: str | None = None,
    augmentations_output_path: str | None = None,
    http_target_confirmation: HttpTargetConfirmation | None = None,
    http_target_config: JsonHttpTargetConfig | None = None,
) -> DatasetRunManifest:
    if repetitions > _MAXIMUM_REPETITIONS:
        raise ValueError(f"repetitions cannot exceed {_MAXIMUM_REPETITIONS}")
    effective_command = DatasetRunEffectiveCommand(
        repetitions=repetitions,
        target_timeout_seconds=target_timeout_seconds,
        max_environment_api_calls=max_environment_api_calls,
        allow_environment_network=allow_environment_network,
        confirm_test_environment=confirm_test_environment,
        allow_insecure_http=allow_insecure_http,
        save_augmentations=save_augmentations,
        semantic_provider_type=semantic_provider_type,
        semantic_base_url=semantic_base_url,
        semantic_live_calls=semantic_live_calls,
        semantic_allow_external_data_processing=semantic_allow_external_data_processing,
        invariant_suite_snapshot=invariant_suite_snapshot,
        invariant_suite_source=invariant_suite_source,
        redaction_policy_snapshot=redaction_policy_snapshot,
        redaction_policy_source=redaction_policy_source,
        redaction_state_path=redaction_state_path,
        redaction_state_sha256=redaction_state_sha256,
        augmentations_output_path=augmentations_output_path,
        http_target_confirmation=http_target_confirmation,
        http_target_config=http_target_config,
    )
    content = {
        "schema_version": "1.0.0",
        "run_context": run_context.model_dump(mode="json"),
        "selected_records": [record.model_dump(mode="json") for record in selected_records],
        "selected_operator_ids": list(selected_operator_ids),
        "effective_command": effective_command.model_dump(mode="json"),
        "work_plan": [
            unit.model_dump(mode="json")
            for unit in _materialize_work_plan(selected_records, selected_operator_ids, repetitions)
        ],
    }
    return DatasetRunManifest(
        run_context=run_context,
        selected_records=selected_records,
        selected_operator_ids=selected_operator_ids,
        effective_command=effective_command,
        work_plan=tuple(DatasetTrialUnit.model_validate(unit) for unit in content["work_plan"]),
        manifest_sha256=_canonical_sha256(content),
    )


def persist_dataset_run_manifest(path: Path, manifest: DatasetRunManifest) -> None:
    encoded = _canonical_bytes(manifest.model_dump(mode="json")) + b"\n"
    if len(encoded) > _MAXIMUM_MANIFEST_BYTES:
        raise ValueError("run manifest exceeds its size limit")
    if path.exists():
        if read_dataset_run_manifest(path) != manifest:
            raise ValueError("existing run manifest does not match this campaign")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary_path, path)
        fsync_run_directory(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_dataset_run_manifest(path: Path) -> DatasetRunManifest:
    raw = _read_private_file(path, _MAXIMUM_MANIFEST_BYTES, "run manifest")
    try:
        return DatasetRunManifest.model_validate_json(raw)
    except ValidationError:
        raise ValueError("run manifest is invalid or corrupted") from None


def private_file_sha256(path: Path) -> str:
    descriptor = _open_hardened_descriptor(path, writable=False)
    try:
        raw = _read_descriptor(
            descriptor,
            32_000_000,
            "private run input",
            require_terminal_newline=False,
        )
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


class DatasetTrialJournal:
    def __init__(
        self,
        descriptor: int,
        manifest: DatasetRunManifest,
        anchor_path: Path,
        records: tuple[DatasetTrialJournalRecord, ...] = (),
    ) -> None:
        self._descriptor = descriptor
        self.manifest = manifest
        self._anchor_path = anchor_path
        self._records = list(records)
        self._unit_ids = {unit.id for unit in manifest.work_plan}
        self._states = _validate_journal_records(manifest, records)

    @property
    def snapshot(self) -> DatasetTrialJournalSnapshot:
        recovered_trials = {
            record.unit.id: (
                record.trial
                if record.trial is not None
                else DatasetEvaluationTrial(
                    repetition=record.unit.repetition,
                    inconclusive_reasons=(
                        "prior target delivery was uncertain; quarantined trial was not retried",
                    ),
                    lifecycle_failure=DatasetTargetLifecycleFailure(
                        failed_phase="interrupted_target_delivery",
                        cleanup_reset_failed=True,
                        environment_state_may_remain=True,
                    ),
                )
            )
            for record in self._records
            if record.state in _TERMINAL_STATES
            and (record.trial is not None or record.state == "quarantined")
        }
        terminal_states: dict[str, TrialState] = {
            record.unit.id: record.state
            for record in self._records
            if record.state in _TERMINAL_STATES
        }
        terminal_reason_codes = {
            record.unit.id: record.reason_code
            for record in self._records
            if record.state in _TERMINAL_STATES
        }
        quarantined = {
            unit_id
            for unit_id, state in self._states.items()
            if state in {"running", "quarantined"}
        }
        return DatasetTrialJournalSnapshot(
            recovered_trials=recovered_trials,
            terminal_states=terminal_states,
            terminal_reason_codes=terminal_reason_codes,
            quarantined_unit_ids=frozenset(quarantined),
        )

    def start(self, unit: DatasetTrialUnit) -> None:
        self._append(unit, "running")

    def is_terminal(self, unit: DatasetTrialUnit) -> bool:
        return self._states.get(unit.id) in _TERMINAL_STATES

    def finish(self, unit: DatasetTrialUnit, trial: DatasetEvaluationTrial) -> None:
        if (
            trial.lifecycle_failure is not None
            and trial.lifecycle_failure.environment_state_may_remain
        ):
            state: TrialState = "quarantined"
            reason = "target_state_uncertain"
        elif trial.inconclusive_reasons:
            state = (
                "skipped"
                if trial.execution_evidence is None
                and any("not executed" in reason for reason in trial.inconclusive_reasons)
                else "inconclusive"
            )
            reason = "trial_inconclusive"
        else:
            state = "completed"
            reason = None
        self._append(unit, state, trial=trial, reason_code=reason)

    def terminal(self, unit: DatasetTrialUnit, state: TrialState, reason_code: str) -> None:
        if state not in _TERMINAL_STATES:
            raise ValueError("terminal journal transition requires a terminal state")
        self._append(unit, state, reason_code=reason_code)

    def flush(self) -> None:
        self._require_open()
        os.fsync(self._descriptor)

    def close(self) -> None:
        if self._descriptor < 0:
            return
        descriptor = self._descriptor
        self._descriptor = -1
        os.close(descriptor)

    def __enter__(self) -> DatasetTrialJournal:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _append(
        self,
        unit: DatasetTrialUnit,
        state: TrialState,
        *,
        trial: DatasetEvaluationTrial | None = None,
        reason_code: str | None = None,
    ) -> None:
        self._require_open()
        if unit.id not in self._unit_ids:
            raise ValueError("trial unit is outside the immutable work plan")
        prior_state = self._states.get(unit.id, "planned")
        if prior_state in _TERMINAL_STATES:
            raise ValueError("trial unit already has a terminal outcome")
        if state == "running" and prior_state != "planned":
            raise ValueError("trial unit cannot start twice")
        if state in _TERMINAL_STATES and prior_state not in {"planned", "running"}:
            raise ValueError("invalid terminal trial transition")
        previous_digest = (
            self._records[-1].record_sha256 if self._records else hashlib.sha256(b"").hexdigest()
        )
        content = {
            "schema_version": "1.0.0",
            "sequence": len(self._records) + 1,
            "manifest_sha256": self.manifest.manifest_sha256,
            "unit": unit.model_dump(mode="json"),
            "state": state,
            "trial": trial.model_dump(mode="json") if trial is not None else None,
            "reason_code": reason_code,
            "previous_record_sha256": previous_digest,
        }
        record = DatasetTrialJournalRecord(
            sequence=len(self._records) + 1,
            manifest_sha256=self.manifest.manifest_sha256,
            unit=unit,
            state=state,
            trial=trial,
            reason_code=reason_code,
            previous_record_sha256=previous_digest,
            record_sha256=_canonical_sha256(content),
        )
        encoded = _canonical_bytes(record.model_dump(mode="json")) + b"\n"
        if len(encoded) > _MAXIMUM_JOURNAL_RECORD_BYTES:
            raise ValueError("trial journal record exceeds its size limit")
        if len(self._records) >= len(self.manifest.work_plan) * 2:
            raise ValueError("trial journal exceeds its transition count limit")
        if os.fstat(self._descriptor).st_size + len(encoded) > _MAXIMUM_JOURNAL_BYTES:
            raise ValueError("trial journal exceeds its size limit")
        _write_all(self._descriptor, encoded)
        os.fsync(self._descriptor)
        _persist_journal_anchor(
            self._anchor_path,
            _create_journal_anchor(self.manifest, (*self._records, record)),
        )
        self._records.append(record)
        self._states[unit.id] = state

    def _require_open(self) -> None:
        if self._descriptor < 0:
            raise ValueError("trial journal is closed")


def create_dataset_trial_journal(path: Path, manifest: DatasetRunManifest) -> DatasetTrialJournal:
    descriptor = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        _lock_new_descriptor(descriptor)
        anchor = _create_journal_anchor(manifest, ())
        _persist_journal_anchor(_anchor_path_for_journal(path), anchor, exclusive=True)
        fsync_run_directory(path)
    except BaseException:
        os.close(descriptor)
        raise
    return DatasetTrialJournal(descriptor, manifest, _anchor_path_for_journal(path))


def fsync_run_directory(path: Path) -> None:
    if sys.platform == "win32":
        return
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def open_dataset_trial_journal(path: Path, manifest: DatasetRunManifest) -> DatasetTrialJournal:
    descriptor = _open_hardened_descriptor(path, writable=True)
    try:
        raw = _read_descriptor(descriptor, _MAXIMUM_JOURNAL_BYTES, "trial journal")
        records: list[DatasetTrialJournalRecord] = []
        for line in raw.splitlines():
            if len(line) > _MAXIMUM_JOURNAL_RECORD_BYTES:
                raise ValueError("trial journal record exceeds its size limit")
            records.append(DatasetTrialJournalRecord.model_validate_json(line))
        if len(records) > len(manifest.work_plan) * 2:
            raise ValueError("trial journal exceeds its transition count limit")
    except ValidationError:
        os.close(descriptor)
        raise ValueError("trial journal is invalid or corrupted") from None
    except BaseException:
        os.close(descriptor)
        raise
    anchor_path = _anchor_path_for_journal(path)
    try:
        anchor = _read_journal_anchor(anchor_path)
        if anchor != _create_journal_anchor(manifest, tuple(records)):
            raise ValueError("trial journal does not match its durable anchor")
        journal = DatasetTrialJournal(descriptor, manifest, anchor_path, tuple(records))
    except BaseException:
        os.close(descriptor)
        raise
    running_units = {record.unit.id: record.unit for record in records if record.state == "running"}
    for record in records:
        if record.state in _TERMINAL_STATES:
            running_units.pop(record.unit.id, None)
    for unit in running_units.values():
        journal.terminal(unit, "quarantined", "interrupted_target_delivery_uncertain")
    return journal


def _materialize_work_plan(
    records: tuple[InteractionRecord, ...], operator_ids: tuple[str, ...], repetitions: int
) -> tuple[DatasetTrialUnit, ...]:
    return tuple(
        unit
        for record in records
        for repetition in range(1, repetitions + 1)
        for unit in (
            DatasetTrialUnit(
                interaction_id=record.id,
                operator_id="current_baseline",
                arm="original",
                repetition=repetition,
            ),
            *(
                DatasetTrialUnit(
                    interaction_id=record.id,
                    operator_id=operator_id.partition("@")[0],
                    arm="probe",
                    repetition=repetition,
                )
                for operator_id in operator_ids
            ),
        )
    )


def _validate_journal_records(
    manifest: DatasetRunManifest, records: tuple[DatasetTrialJournalRecord, ...]
) -> dict[str, TrialState]:
    states: dict[str, TrialState] = {}
    expected_previous = hashlib.sha256(b"").hexdigest()
    unit_ids = {unit.id for unit in manifest.work_plan}
    for sequence, record in enumerate(records, start=1):
        if record.sequence != sequence or record.previous_record_sha256 != expected_previous:
            raise ValueError("trial journal hash chain is invalid")
        if record.manifest_sha256 != manifest.manifest_sha256:
            raise ValueError("trial journal belongs to a different manifest")
        if record.unit.id not in unit_ids:
            raise ValueError("trial journal contains a unit outside its work plan")
        prior_state = states.get(record.unit.id, "planned")
        if prior_state in _TERMINAL_STATES:
            raise ValueError("trial journal contains a duplicate terminal outcome")
        if record.state == "running" and prior_state != "planned":
            raise ValueError("trial journal contains a duplicate start")
        if record.state in _TERMINAL_STATES and prior_state not in {"planned", "running"}:
            raise ValueError("trial journal contains an invalid transition")
        states[record.unit.id] = record.state
        expected_previous = record.record_sha256
    return states


def _read_private_file(path: Path, maximum_bytes: int, label: str) -> bytes:
    descriptor = _open_hardened_descriptor(path, writable=False)
    try:
        return _read_descriptor(descriptor, maximum_bytes, label)
    finally:
        os.close(descriptor)


def _read_descriptor(
    descriptor: int,
    maximum_bytes: int,
    label: str,
    *,
    require_terminal_newline: bool = True,
) -> bytes:
    status = os.fstat(descriptor)
    if status.st_size > maximum_bytes:
        raise ValueError(f"{label} exceeds its size limit")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = status.st_size
    while remaining:
        chunk = os.read(descriptor, min(65_536, remaining))
        if not chunk:
            raise ValueError(f"{label} changed while reading")
        chunks.append(chunk)
        remaining -= len(chunk)
    raw = b"".join(chunks)
    if require_terminal_newline and raw and not raw.endswith(b"\n"):
        raise ValueError(f"{label} is truncated or corrupted")
    return raw


def _open_hardened_descriptor(path: Path, *, writable: bool) -> int:
    try:
        descriptor = open_resume_descriptor(path, writable=writable)
    except OSError as error:
        raise ValueError(str(error)) from None
    status = os.fstat(descriptor)
    if status.st_nlink != 1:
        os.close(descriptor)
        raise ValueError("durable run file must have exactly one hard link")
    if sys.platform != "win32":
        if status.st_uid != os.getuid():
            os.close(descriptor)
            raise ValueError("durable run file is not owned by the current user")
        if stat.S_IMODE(status.st_mode) & 0o077:
            os.close(descriptor)
            raise ValueError("durable run file permissions are not private")
    return descriptor


def _lock_new_descriptor(descriptor: int) -> None:
    if sys.platform == "win32":
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        return
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _anchor_path_for_journal(path: Path) -> Path:
    return path.with_name(f"{path.name}.anchor.json")


def _create_journal_anchor(
    manifest: DatasetRunManifest,
    records: tuple[DatasetTrialJournalRecord, ...],
) -> DatasetTrialJournalAnchor:
    content = {
        "schema_version": "1.0.0",
        "manifest_sha256": manifest.manifest_sha256,
        "record_count": len(records),
        "last_record_sha256": (
            records[-1].record_sha256 if records else hashlib.sha256(b"").hexdigest()
        ),
    }
    return DatasetTrialJournalAnchor(
        manifest_sha256=manifest.manifest_sha256,
        record_count=len(records),
        last_record_sha256=(
            records[-1].record_sha256 if records else hashlib.sha256(b"").hexdigest()
        ),
        anchor_sha256=_canonical_sha256(content),
    )


def _read_journal_anchor(path: Path) -> DatasetTrialJournalAnchor:
    raw = _read_private_file(path, 10_000, "trial journal anchor")
    try:
        return DatasetTrialJournalAnchor.model_validate_json(raw)
    except ValidationError:
        raise ValueError("trial journal anchor is invalid or corrupted") from None


def create_quarantine_resolution(
    manifest: DatasetRunManifest,
    quarantined_unit_ids: frozenset[str],
    operator_attestation: Literal["environment-reset", "environment-replacement"],
    resolved_at: datetime,
) -> DatasetQuarantineResolution:
    content = {
        "schema_version": "1.0.0",
        "manifest_sha256": manifest.manifest_sha256,
        "target_sha256": manifest.run_context.target.sha256,
        "quarantined_unit_ids": sorted(quarantined_unit_ids),
        "operator_attestation": operator_attestation,
        "resolved_at": resolved_at.isoformat().replace("+00:00", "Z"),
        "independently_verified": False,
    }
    return DatasetQuarantineResolution(
        manifest_sha256=manifest.manifest_sha256,
        target_sha256=manifest.run_context.target.sha256,
        quarantined_unit_ids=tuple(sorted(quarantined_unit_ids)),
        operator_attestation=operator_attestation,
        resolved_at=resolved_at,
        resolution_sha256=_canonical_sha256(content),
    )


def persist_quarantine_resolution(path: Path, resolution: DatasetQuarantineResolution) -> None:
    encoded = _canonical_bytes(resolution.model_dump(mode="json")) + b"\n"
    if path.exists():
        if read_quarantine_resolution(path) != resolution:
            raise ValueError("existing quarantine resolution does not match")
        return
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary_path, path)
        fsync_run_directory(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_quarantine_resolution(path: Path) -> DatasetQuarantineResolution:
    raw = _read_private_file(path, 100_000, "quarantine resolution")
    try:
        return DatasetQuarantineResolution.model_validate_json(raw)
    except ValidationError:
        raise ValueError("quarantine resolution is invalid or corrupted") from None


def _persist_journal_anchor(
    path: Path,
    anchor: DatasetTrialJournalAnchor,
    *,
    exclusive: bool = False,
) -> None:
    encoded = _canonical_bytes(anchor.model_dump(mode="json")) + b"\n"
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        if exclusive:
            os.link(temporary_path, path)
        else:
            os.replace(temporary_path, path)
        fsync_run_directory(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("journal write made no progress")
        view = view[written:]
