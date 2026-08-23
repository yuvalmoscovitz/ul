from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from ul import DatasetEvaluationTrial, DatasetTrialUnit, InteractionRecord

from ul_cli.dataset_review import DatasetEvidenceRunContext

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_MAXIMUM_MANIFEST_BYTES = 32_000_000
_MAXIMUM_JOURNAL_BYTES = 256_000_000

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
    repetitions: int = Field(ge=1)
    max_environment_api_calls: int = Field(ge=1)
    allow_environment_network: bool
    confirm_test_environment: bool
    allow_insecure_http: bool
    save_augmentations: bool


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


@dataclass(frozen=True)
class DatasetTrialJournalSnapshot:
    recovered_trials: dict[str, DatasetEvaluationTrial]
    terminal_states: dict[str, TrialState]
    quarantined_unit_ids: frozenset[str]


def manifest_path(evidence_output: Path) -> Path:
    return evidence_output.with_name(f"{evidence_output.name}.manifest.json")


def journal_path(evidence_output: Path) -> Path:
    return evidence_output.with_name(f"{evidence_output.name}.trials.jsonl")


def create_dataset_run_manifest(
    *,
    run_context: DatasetEvidenceRunContext,
    selected_records: tuple[InteractionRecord, ...],
    selected_operator_ids: tuple[str, ...],
    repetitions: int,
    max_environment_api_calls: int,
    allow_environment_network: bool,
    confirm_test_environment: bool,
    allow_insecure_http: bool,
    save_augmentations: bool,
) -> DatasetRunManifest:
    effective_command = DatasetRunEffectiveCommand(
        repetitions=repetitions,
        max_environment_api_calls=max_environment_api_calls,
        allow_environment_network=allow_environment_network,
        confirm_test_environment=confirm_test_environment,
        allow_insecure_http=allow_insecure_http,
        save_augmentations=save_augmentations,
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
    finally:
        temporary_path.unlink(missing_ok=True)


def read_dataset_run_manifest(path: Path) -> DatasetRunManifest:
    raw = _read_private_file(path, _MAXIMUM_MANIFEST_BYTES, "run manifest")
    try:
        return DatasetRunManifest.model_validate_json(raw)
    except ValidationError:
        raise ValueError("run manifest is invalid or corrupted") from None


class DatasetTrialJournal:
    def __init__(
        self,
        descriptor: int,
        manifest: DatasetRunManifest,
        records: tuple[DatasetTrialJournalRecord, ...] = (),
    ) -> None:
        self._descriptor = descriptor
        self.manifest = manifest
        self._records = list(records)
        self._unit_ids = {unit.id for unit in manifest.work_plan}
        self._states = _validate_journal_records(manifest, records)

    @property
    def snapshot(self) -> DatasetTrialJournalSnapshot:
        recovered_trials = {
            record.unit.id: record.trial
            for record in self._records
            if record.state in {"completed", "inconclusive"} and record.trial is not None
        }
        terminal_states: dict[str, TrialState] = {
            record.unit.id: record.state
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
        if os.fstat(self._descriptor).st_size + len(encoded) > _MAXIMUM_JOURNAL_BYTES:
            raise ValueError("trial journal exceeds its size limit")
        _write_all(self._descriptor, encoded)
        os.fsync(self._descriptor)
        self._records.append(record)
        self._states[unit.id] = state

    def _require_open(self) -> None:
        if self._descriptor < 0:
            raise ValueError("trial journal is closed")


def create_dataset_trial_journal(path: Path, manifest: DatasetRunManifest) -> DatasetTrialJournal:
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL, 0o600)
    return DatasetTrialJournal(descriptor, manifest)


def open_dataset_trial_journal(path: Path, manifest: DatasetRunManifest) -> DatasetTrialJournal:
    raw = _read_private_file(path, _MAXIMUM_JOURNAL_BYTES, "trial journal")
    records: list[DatasetTrialJournalRecord] = []
    for line in raw.splitlines():
        try:
            records.append(DatasetTrialJournalRecord.model_validate_json(line))
        except ValidationError:
            raise ValueError("trial journal is invalid or corrupted") from None
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    journal = DatasetTrialJournal(descriptor, manifest, tuple(records))
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
    descriptor = os.open(path, os.O_RDONLY)
    try:
        status = os.fstat(descriptor)
        if sys.platform != "win32" and stat.S_IMODE(status.st_mode) & 0o077:
            raise ValueError(f"{label} permissions are not private")
        if status.st_size > maximum_bytes:
            raise ValueError(f"{label} exceeds its size limit")
        raw = os.read(descriptor, status.st_size)
    finally:
        os.close(descriptor)
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"{label} is truncated or corrupted")
    return raw


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
