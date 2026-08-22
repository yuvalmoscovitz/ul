from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import TextIO

from pydantic import ValidationError
from ul import (
    DatasetSemanticSettings,
    EvaluatorModelPreflight,
    InteractionRecord,
    ProviderDiagnosticError,
    validate_evaluator_preflight,
)
from ul.dataset_invariants import DatasetInvariantSuite

from ul_cli.dataset_review import (
    DatasetEvidenceRunContext,
    DatasetResumeEvidence,
    validate_dataset_resume_evidence,
)

from ..storage.private_files import create_private_output, open_resume_descriptor
from .customer import build_customer_evidence_record

_MAXIMUM_EVIDENCE_BYTES = 128_000_000
_MAXIMUM_PREFLIGHT_RECEIPT_BYTES = 100_000
_PROVIDER_DIAGNOSTIC_SCHEMA_VERSION = "1.0.0"


def default_augmentations_output(evidence_output: Path) -> Path:
    if evidence_output.suffix:
        return evidence_output.with_name(f"{evidence_output.stem}.augmentations.jsonl")
    return evidence_output.with_name(f"{evidence_output.name}.augmentations.jsonl")


def write_provider_diagnostic(output: Path, error: ProviderDiagnosticError) -> Path:
    payload = {
        "schema_version": _PROVIDER_DIAGNOSTIC_SCHEMA_VERSION,
        "record_type": "provider_diagnostic",
        "diagnostic": error.diagnostic.model_dump(mode="json"),
    }
    for sequence in range(1, 101):
        sequence_suffix = "" if sequence == 1 else f".{sequence}"
        diagnostic_output = output.with_name(f"{output.name}.debug{sequence_suffix}.json")
        try:
            output_stream = create_private_output(diagnostic_output)
        except FileExistsError:
            continue
        with output_stream:
            json.dump(payload, output_stream, ensure_ascii=False, sort_keys=True)
            output_stream.write("\n")
            output_stream.flush()
            os.fsync(output_stream.fileno())
        return diagnostic_output
    raise FileExistsError("provider diagnostic receipt slots are occupied")


async def load_evaluator_preflight(
    evidence_output: Path,
    settings: DatasetSemanticSettings,
) -> tuple[EvaluatorModelPreflight, Path]:
    receipt_path = _evaluator_preflight_output(evidence_output)
    try:
        result = _read_evaluator_preflight(receipt_path)
    except FileNotFoundError:
        raise ValueError(f"required receipt {receipt_path.name} is missing") from None
    except OSError as error:
        raise ValueError(
            f"required receipt {receipt_path.name} cannot be safely read "
            f"({error.__class__.__name__})"
        ) from None
    validate_evaluator_preflight(settings, result)
    return result, receipt_path


def _evaluator_preflight_output(evidence_output: Path) -> Path:
    return evidence_output.with_name(f"{evidence_output.name}.preflight.json")


def persist_evaluator_preflight(
    evidence_output: Path,
    result: EvaluatorModelPreflight,
) -> Path:
    receipt_path = _evaluator_preflight_output(evidence_output)
    if receipt_path.exists():
        existing = _read_evaluator_preflight(receipt_path)
        if existing != result:
            raise ValueError("existing evaluator preflight receipt does not match this run")
        return receipt_path
    with create_private_output(receipt_path) as output_stream:
        json.dump(result.model_dump(mode="json"), output_stream, ensure_ascii=False, sort_keys=True)
        output_stream.write("\n")
        output_stream.flush()
        os.fsync(output_stream.fileno())
    return receipt_path


def _read_evaluator_preflight(receipt_path: Path) -> EvaluatorModelPreflight:
    descriptor = open_resume_descriptor(receipt_path, writable=False)
    try:
        descriptor_status = os.fstat(descriptor)
        if sys.platform != "win32" and stat.S_IMODE(descriptor_status.st_mode) & 0o077:
            raise ValueError("evaluator preflight receipt permissions are not private")
        size = descriptor_status.st_size
        if size > _MAXIMUM_PREFLIGHT_RECEIPT_BYTES:
            raise ValueError("evaluator preflight receipt exceeds its size limit")
        try:
            return EvaluatorModelPreflight.model_validate_json(os.read(descriptor, size))
        except ValidationError:
            raise ValueError("evaluator preflight receipt is invalid") from None
    finally:
        os.close(descriptor)


def read_resume_evidence(
    path: Path,
    *,
    expected_context: DatasetEvidenceRunContext,
    selected_records: tuple[InteractionRecord, ...],
    invariant_suite: DatasetInvariantSuite | None,
) -> DatasetResumeEvidence:
    descriptor = open_resume_descriptor(path, writable=False)
    try:
        return _read_resume_descriptor(
            descriptor,
            expected_context=expected_context,
            selected_records=selected_records,
            invariant_suite=invariant_suite,
        )
    finally:
        os.close(descriptor)


def open_resume_output(
    path: Path,
    *,
    expected_context: DatasetEvidenceRunContext,
    selected_records: tuple[InteractionRecord, ...],
    invariant_suite: DatasetInvariantSuite | None,
) -> tuple[TextIO, DatasetResumeEvidence]:
    descriptor = open_resume_descriptor(path, writable=True)
    try:
        resume_evidence = _read_resume_descriptor(
            descriptor,
            expected_context=expected_context,
            selected_records=selected_records,
            invariant_suite=invariant_suite,
        )
        if sys.platform != "win32":
            os.fchmod(descriptor, 0o600)
        os.lseek(descriptor, 0, os.SEEK_END)
        return os.fdopen(descriptor, "a", encoding="utf-8"), resume_evidence
    except BaseException:
        os.close(descriptor)
        raise


def _read_resume_descriptor(
    descriptor: int,
    *,
    expected_context: DatasetEvidenceRunContext,
    selected_records: tuple[InteractionRecord, ...],
    invariant_suite: DatasetInvariantSuite | None,
) -> DatasetResumeEvidence:
    size = os.fstat(descriptor).st_size
    if size > _MAXIMUM_EVIDENCE_BYTES:
        raise ValueError("resume evidence exceeds the 128 MB limit")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, min(65_536, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    raw_evidence = b"".join(chunks)
    if raw_evidence and not raw_evidence.endswith(b"\n"):
        raise ValueError("resume evidence must end with a newline")
    return validate_dataset_resume_evidence(
        raw_evidence,
        expected_context=expected_context,
        selected_records=selected_records,
        invariant_suite=invariant_suite,
        evidence_projector=build_customer_evidence_record,
    )
