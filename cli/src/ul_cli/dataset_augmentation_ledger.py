from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from ul import DatasetAugmentationResult, InteractionRecord, SemanticDeconstructorIdentity

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

_MAXIMUM_LEDGER_BYTES = 128_000_000
_MAXIMUM_LEDGER_RECORDS = 100
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_SEMANTIC_VERSION_PATTERN = r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DatasetAugmentationLedgerOperator(_StrictModel):
    id: str = Field(min_length=1, max_length=251)
    version: str = Field(pattern=_SEMANTIC_VERSION_PATTERN)


class DatasetAugmentationLedgerSemanticSettings(_StrictModel):
    provider: str = Field(min_length=1, max_length=100)
    endpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    model: str = Field(min_length=1, max_length=200)
    render_model: str = Field(min_length=1, max_length=200)
    equivalence_model: str = Field(min_length=1, max_length=200)
    deconstruct_reasoning: Literal["required", "omitted"] = "required"
    render_reasoning: Literal["required", "omitted"] = "required"
    equivalence_reasoning: Literal["required", "omitted"] = "required"
    max_input_chars: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    max_render_tokens: int = Field(ge=1)
    max_response_bytes: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    deconstructor_identity: SemanticDeconstructorIdentity | None = None


class DatasetAugmentationGenerationContext(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    pipeline_version: str = Field(pattern=_SEMANTIC_VERSION_PATTERN)
    selected_dataset_sha256: str = Field(pattern=_SHA256_PATTERN)
    operators: tuple[DatasetAugmentationLedgerOperator, ...] = Field(min_length=1)
    semantic_settings: DatasetAugmentationLedgerSemanticSettings
    redaction_policy_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    source_outcome_projection_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    context_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_context(self) -> Self:
        operator_keys = tuple((operator.id, operator.version) for operator in self.operators)
        if len(operator_keys) != len(set(operator_keys)):
            raise ValueError("generation context contains duplicate operators")
        context_content = self.model_dump(mode="json", exclude={"context_sha256"})
        if self.semantic_settings.deconstructor_identity is None:
            cast(dict[str, object], context_content["semantic_settings"]).pop(
                "deconstructor_identity"
            )
        if self.source_outcome_projection_sha256 is None:
            context_content.pop("source_outcome_projection_sha256")
        expected_digest = _canonical_json_sha256(context_content)
        if self.context_sha256 != expected_digest:
            raise ValueError("generation context digest must match its canonical content")
        return self


class DatasetAugmentationLedgerRecord(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    generation_context: DatasetAugmentationGenerationContext
    source: InteractionRecord
    augmentation: DatasetAugmentationResult
    record_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        if len(self.augmentation.source_frames) != 1:
            raise ValueError("ledger augmentation must contain exactly one source frame")
        if self.augmentation.source_records != (self.source,):
            raise ValueError("ledger augmentation source must match its source interaction")
        if self.augmentation.source_frames[0].interaction_id != self.source.id:
            raise ValueError("ledger source frame must match its source interaction")

        candidate_keys = tuple(
            (candidate.operator_id, candidate.operator_version)
            for candidate in self.augmentation.candidates
        )
        if len(candidate_keys) != len(set(candidate_keys)):
            raise ValueError("ledger augmentation contains duplicate operators")
        if any(
            candidate.source_interaction_id != self.source.id
            for candidate in self.augmentation.candidates
        ):
            raise ValueError("ledger candidate must match its source interaction")
        expected_operator_order = {
            (operator.id, operator.version): index
            for index, operator in enumerate(self.generation_context.operators)
        }
        stored_operator_references = tuple(
            (reference.id, reference.version) for reference in self.augmentation.operator_references
        )
        if stored_operator_references != tuple(expected_operator_order):
            raise ValueError("ledger augmentation does not match the generation operator plan")
        try:
            candidate_order = tuple(expected_operator_order[key] for key in candidate_keys)
        except KeyError:
            raise ValueError("ledger candidate is outside the generation context") from None
        if candidate_order != tuple(sorted(candidate_order)):
            raise ValueError("ledger candidates must follow generation context operator order")

        expected_digest = _canonical_json_sha256(
            self.model_dump(mode="json", exclude={"record_sha256"})
        )
        if self.record_sha256 != expected_digest:
            raise ValueError("ledger record digest must match its canonical content")
        return self


@dataclass(frozen=True)
class DatasetAugmentationLedgerSnapshot:
    records: tuple[DatasetAugmentationLedgerRecord, ...]
    processed_source_ids: frozenset[str]
    raw_ledger_sha256: str


def create_dataset_augmentation_generation_context(
    *,
    selected_records: tuple[InteractionRecord, ...],
    operators: tuple[tuple[str, str], ...],
    semantic_settings: DatasetAugmentationLedgerSemanticSettings,
    pipeline_version: str = "1.0.0",
    redaction_policy_sha256: str | None = None,
    source_outcome_projection_sha256: str | None = None,
) -> DatasetAugmentationGenerationContext:
    _validate_unique_source_ids(selected_records)
    operator_snapshots = tuple(
        DatasetAugmentationLedgerOperator(id=operator_id, version=operator_version)
        for operator_id, operator_version in operators
    )
    selected_dataset_sha256 = _selected_dataset_sha256(selected_records)
    content = {
        "schema_version": "1.0.0",
        "pipeline_version": pipeline_version,
        "selected_dataset_sha256": selected_dataset_sha256,
        "operators": [operator.model_dump(mode="json") for operator in operator_snapshots],
        "semantic_settings": semantic_settings.model_dump(
            mode="json",
            exclude={"deconstructor_identity"}
            if semantic_settings.deconstructor_identity is None
            else None,
        ),
        "redaction_policy_sha256": redaction_policy_sha256,
        **(
            {"source_outcome_projection_sha256": source_outcome_projection_sha256}
            if source_outcome_projection_sha256 is not None
            else {}
        ),
    }
    return DatasetAugmentationGenerationContext(
        pipeline_version=pipeline_version,
        selected_dataset_sha256=selected_dataset_sha256,
        operators=operator_snapshots,
        semantic_settings=semantic_settings,
        redaction_policy_sha256=redaction_policy_sha256,
        source_outcome_projection_sha256=source_outcome_projection_sha256,
        context_sha256=_canonical_json_sha256(content),
    )


def create_dataset_augmentation_ledger_record(
    *,
    generation_context: DatasetAugmentationGenerationContext,
    source: InteractionRecord,
    augmentation: DatasetAugmentationResult,
) -> DatasetAugmentationLedgerRecord:
    content = {
        "schema_version": "1.0.0",
        "generation_context": generation_context.model_dump(mode="json"),
        "source": source.model_dump(mode="json"),
        "augmentation": augmentation.model_dump(mode="json"),
    }
    return DatasetAugmentationLedgerRecord(
        generation_context=generation_context,
        source=source,
        augmentation=augmentation,
        record_sha256=_canonical_json_sha256(content),
    )


class DatasetAugmentationLedger:
    def __init__(
        self,
        descriptor: int,
        *,
        generation_context: DatasetAugmentationGenerationContext,
        selected_records: tuple[InteractionRecord, ...],
        existing_records: tuple[DatasetAugmentationLedgerRecord, ...] = (),
        raw_ledger_sha256: str | None = None,
    ) -> None:
        self._descriptor = descriptor
        self.generation_context = generation_context
        self._selected_records = {record.id: record for record in selected_records}
        self._records = list(existing_records)
        self._processed_source_ids = {record.source.id for record in existing_records}
        self._raw_ledger_sha256 = raw_ledger_sha256 or hashlib.sha256(b"").hexdigest()

    @property
    def snapshot(self) -> DatasetAugmentationLedgerSnapshot:
        return DatasetAugmentationLedgerSnapshot(
            records=tuple(self._records),
            processed_source_ids=frozenset(self._processed_source_ids),
            raw_ledger_sha256=self._raw_ledger_sha256,
        )

    def append(
        self,
        *,
        source: InteractionRecord,
        augmentation: DatasetAugmentationResult,
    ) -> DatasetAugmentationLedgerRecord:
        self._require_open()
        selected_source = self._selected_records.get(source.id)
        if selected_source is None or selected_source != source:
            raise ValueError("ledger source does not match the selected dataset")
        if source.id in self._processed_source_ids:
            raise ValueError("ledger already contains the source interaction")
        if len(self._records) == _MAXIMUM_LEDGER_RECORDS:
            raise ValueError("augmentation ledger exceeds the 100 record limit")
        record = create_dataset_augmentation_ledger_record(
            generation_context=self.generation_context,
            source=source,
            augmentation=augmentation,
        )
        encoded_record = _canonical_json_bytes(record.model_dump(mode="json")) + b"\n"
        current_size = os.fstat(self._descriptor).st_size
        if current_size + len(encoded_record) > _MAXIMUM_LEDGER_BYTES:
            raise ValueError("augmentation ledger exceeds the 128 MB limit")
        _write_all(self._descriptor, encoded_record)
        os.fsync(self._descriptor)
        self._records.append(record)
        self._processed_source_ids.add(source.id)
        self._raw_ledger_sha256 = _descriptor_sha256(self._descriptor)
        return record

    def close(self) -> None:
        if self._descriptor < 0:
            return
        descriptor = self._descriptor
        self._descriptor = -1
        try:
            _unlock_descriptor(descriptor)
        finally:
            os.close(descriptor)

    def discard_if_empty(self, path: Path) -> None:
        self._require_open()
        descriptor_status = os.fstat(self._descriptor)
        path_status = os.lstat(path)
        if descriptor_status.st_size == 0 and os.path.samestat(path_status, descriptor_status):
            os.unlink(path)

    def __enter__(self) -> DatasetAugmentationLedger:
        self._require_open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._descriptor < 0:
            raise ValueError("augmentation ledger is closed")


def create_private_augmentation_ledger(
    path: Path,
    *,
    generation_context: DatasetAugmentationGenerationContext,
    selected_records: tuple[InteractionRecord, ...],
) -> DatasetAugmentationLedger:
    _validate_selected_dataset(generation_context, selected_records)
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    binary_flag = os.O_BINARY if sys.platform == "win32" else 0
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL | no_follow_flag | binary_flag,
        0o600,
    )
    try:
        if sys.platform != "win32":
            os.fchmod(descriptor, 0o600)
        _validate_private_descriptor(descriptor)
        _lock_descriptor(descriptor, exclusive=True)
        return DatasetAugmentationLedger(
            descriptor,
            generation_context=generation_context,
            selected_records=selected_records,
        )
    except BaseException:
        os.close(descriptor)
        raise


def open_augmentation_ledger_for_resume(
    path: Path,
    *,
    expected_context: DatasetAugmentationGenerationContext,
    selected_records: tuple[InteractionRecord, ...],
) -> DatasetAugmentationLedger:
    _validate_selected_dataset(expected_context, selected_records)
    descriptor = _open_private_ledger(path, writable=True)
    try:
        _lock_descriptor(descriptor, exclusive=True)
        raw_ledger = _read_bounded_descriptor(descriptor)
        snapshot = _validate_raw_ledger(
            raw_ledger,
            expected_context=expected_context,
            selected_records=selected_records,
        )
        os.lseek(descriptor, 0, os.SEEK_END)
        return DatasetAugmentationLedger(
            descriptor,
            generation_context=expected_context,
            selected_records=selected_records,
            existing_records=snapshot.records,
            raw_ledger_sha256=snapshot.raw_ledger_sha256,
        )
    except BaseException:
        os.close(descriptor)
        raise


def read_augmentation_ledger(
    path: Path,
    *,
    expected_context: DatasetAugmentationGenerationContext,
    selected_records: tuple[InteractionRecord, ...],
) -> DatasetAugmentationLedgerSnapshot:
    _validate_selected_dataset(expected_context, selected_records)
    descriptor = _open_private_ledger(path, writable=False)
    try:
        _lock_descriptor(descriptor, exclusive=False)
        return _validate_raw_ledger(
            _read_bounded_descriptor(descriptor),
            expected_context=expected_context,
            selected_records=selected_records,
        )
    finally:
        os.close(descriptor)


def _validate_raw_ledger(
    raw_ledger: bytes,
    *,
    expected_context: DatasetAugmentationGenerationContext,
    selected_records: tuple[InteractionRecord, ...],
) -> DatasetAugmentationLedgerSnapshot:
    if raw_ledger and not raw_ledger.endswith(b"\n"):
        raise ValueError("augmentation ledger must end with a newline")
    raw_lines = raw_ledger.splitlines()
    if len(raw_lines) > _MAXIMUM_LEDGER_RECORDS:
        raise ValueError("augmentation ledger exceeds the 100 record limit")
    if any(not raw_line.strip() for raw_line in raw_lines):
        raise ValueError("augmentation ledger contains an empty JSONL record")

    selected_records_by_id = {record.id: record for record in selected_records}
    records: list[DatasetAugmentationLedgerRecord] = []
    processed_source_ids: set[str] = set()
    for raw_line in raw_lines:
        decoded_record = _decode_json_object(raw_line)
        try:
            record = DatasetAugmentationLedgerRecord.model_validate_json(raw_line)
        except (ValidationError, ValueError):
            raise ValueError("augmentation ledger is not valid UL JSONL") from None
        if _canonical_json_bytes(decoded_record) != _canonical_json_bytes(
            record.model_dump(mode="json")
        ):
            raise ValueError("augmentation ledger contains non-strict values")
        if record.generation_context != expected_context:
            raise ValueError("augmentation ledger has an incompatible generation context")
        if record.source.id in processed_source_ids:
            raise ValueError("augmentation ledger contains duplicate source interactions")
        selected_source = selected_records_by_id.get(record.source.id)
        if selected_source is None or selected_source != record.source:
            raise ValueError("augmentation ledger source does not match the selected dataset")
        records.append(record)
        processed_source_ids.add(record.source.id)
    return DatasetAugmentationLedgerSnapshot(
        records=tuple(records),
        processed_source_ids=frozenset(processed_source_ids),
        raw_ledger_sha256=hashlib.sha256(raw_ledger).hexdigest(),
    )


def _validate_selected_dataset(
    context: DatasetAugmentationGenerationContext,
    selected_records: tuple[InteractionRecord, ...],
) -> None:
    _validate_unique_source_ids(selected_records)
    if context.selected_dataset_sha256 != _selected_dataset_sha256(selected_records):
        raise ValueError("generation context does not match the selected dataset")


def _validate_unique_source_ids(selected_records: tuple[InteractionRecord, ...]) -> None:
    source_ids = tuple(record.id for record in selected_records)
    if not source_ids:
        raise ValueError("selected dataset must contain at least one interaction")
    if len(source_ids) > _MAXIMUM_LEDGER_RECORDS:
        raise ValueError("selected dataset exceeds the 100 record limit")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("selected dataset contains duplicate interaction IDs")


def _selected_dataset_sha256(selected_records: tuple[InteractionRecord, ...]) -> str:
    return _canonical_json_sha256([record.model_dump(mode="json") for record in selected_records])


def _open_private_ledger(path: Path, *, writable: bool) -> int:
    path_status = os.lstat(path)
    if not stat.S_ISREG(path_status.st_mode):
        raise OSError("augmentation ledger is not a regular file")
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    binary_flag = os.O_BINARY if sys.platform == "win32" else 0
    access_flags = os.O_RDWR | os.O_APPEND if writable else os.O_RDONLY
    descriptor = os.open(path, access_flags | no_follow_flag | binary_flag)
    try:
        descriptor_status = os.fstat(descriptor)
        if not os.path.samestat(path_status, descriptor_status):
            raise OSError("augmentation ledger changed while opening")
        _validate_private_descriptor(descriptor)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validate_private_descriptor(descriptor: int) -> None:
    descriptor_status = os.fstat(descriptor)
    if not stat.S_ISREG(descriptor_status.st_mode):
        raise OSError("augmentation ledger is not a regular file")
    if descriptor_status.st_nlink != 1:
        raise OSError("augmentation ledger must have exactly one hard link")
    if sys.platform != "win32":
        if descriptor_status.st_uid != os.geteuid():
            raise OSError("augmentation ledger must be owned by the current user")
        if stat.S_IMODE(descriptor_status.st_mode) != 0o600:
            raise OSError("augmentation ledger permissions must be 0600")


def _lock_descriptor(descriptor: int, *, exclusive: bool) -> None:
    if sys.platform == "win32":
        os.lseek(descriptor, 0, os.SEEK_SET)
        mode = msvcrt.LK_NBLCK if exclusive else msvcrt.LK_NBRLCK
        msvcrt.locking(descriptor, mode, 1)
        return
    mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    fcntl.flock(descriptor, mode | fcntl.LOCK_NB)


def _unlock_descriptor(descriptor: int) -> None:
    if sys.platform == "win32":
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return
    fcntl.flock(descriptor, fcntl.LOCK_UN)


def _read_bounded_descriptor(descriptor: int) -> bytes:
    size = os.fstat(descriptor).st_size
    if size > _MAXIMUM_LEDGER_BYTES:
        raise ValueError("augmentation ledger exceeds the 128 MB limit")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, min(65_536, remaining))
        if not chunk:
            raise OSError("augmentation ledger changed while reading")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.fstat(descriptor).st_size != size:
        raise OSError("augmentation ledger changed while reading")
    return b"".join(chunks)


def _descriptor_sha256(descriptor: int) -> str:
    return hashlib.sha256(_read_bounded_descriptor(descriptor)).hexdigest()


def _write_all(descriptor: int, content: bytes) -> None:
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written == 0:
            raise OSError("could not append augmentation ledger record")
        remaining = remaining[written:]


def _decode_json_object(raw_line: bytes) -> dict[str, object]:
    try:
        decoded: object = json.loads(
            raw_line,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (json.JSONDecodeError, RecursionError, UnicodeDecodeError, ValueError):
        raise ValueError("augmentation ledger is not valid JSON") from None
    if not isinstance(decoded, dict):
        raise ValueError("augmentation ledger record must be a JSON object")
    return cast(dict[str, object], decoded)


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
