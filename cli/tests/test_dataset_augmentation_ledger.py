from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from ul import DatasetAugmentationResult, InteractionRecord, SemanticFrame
from ul.dataset_augmentation import DatasetAugmentationCandidate
from ul_cli import dataset_augmentation_ledger as ledger_module
from ul_cli.dataset_augmentation_ledger import (
    DatasetAugmentationLedgerSemanticSettings,
    create_dataset_augmentation_generation_context,
    create_dataset_augmentation_ledger_record,
    create_private_augmentation_ledger,
    open_augmentation_ledger_for_resume,
    read_augmentation_ledger,
)

_ENDPOINT_SHA256 = "1" * 64


def _source(identifier: str = "interaction-1") -> InteractionRecord:
    return InteractionRecord(
        id=identifier,
        raw_input="Transfer 100 to Alice.",
        raw_observed_output={"status": "approved"},
    )


def _augmentation(source: InteractionRecord) -> DatasetAugmentationResult:
    source_frame = SemanticFrame(interaction_id=source.id, extractor_version="test")
    candidate = DatasetAugmentationCandidate(
        source_interaction_id=source.id,
        operator_id="input.surface.rephrase",
        operator_version="1.0.0",
        augmented_input="Please transfer 100 to Alice.",
        expected_input_frame=source_frame,
        reparsed_input_frame=SemanticFrame(
            interaction_id=f"{source.id}:input.surface.rephrase",
            extractor_version="test",
        ),
        passed=True,
    )
    return DatasetAugmentationResult(
        operator_references=({"id": "input.surface.rephrase", "version": "1.0.0"},),
        source_records=(source,),
        source_frames=(source_frame,),
        candidates=(candidate,),
    )


def _context(
    selected_records: tuple[InteractionRecord, ...],
    *,
    model: str = "test/model",
):
    return create_dataset_augmentation_generation_context(
        selected_records=selected_records,
        operators=(("input.surface.rephrase", "1.0.0"),),
        semantic_settings=DatasetAugmentationLedgerSemanticSettings(
            provider="test-provider",
            endpoint_sha256=_ENDPOINT_SHA256,
            model=model,
            render_model="test/render-model",
            equivalence_model="test/equivalence-model",
            max_input_chars=50_000,
            max_output_tokens=4_096,
            max_render_tokens=512,
            max_response_bytes=1_000_000,
            timeout_seconds=60.0,
        ),
    )


def test_private_ledger_round_trip_and_resume_append(tmp_path: Path) -> None:
    first_source = _source("interaction-1")
    second_source = _source("interaction-2")
    selected_records = (first_source, second_source)
    context = _context(selected_records)
    path = tmp_path / "augmentations.jsonl"

    with create_private_augmentation_ledger(
        path,
        generation_context=context,
        selected_records=selected_records,
    ) as ledger:
        first_record = ledger.append(
            source=first_source,
            augmentation=_augmentation(first_source),
        )
        assert ledger.snapshot.processed_source_ids == frozenset({first_source.id})

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_bytes().endswith(b"\n")

    with open_augmentation_ledger_for_resume(
        path,
        expected_context=context,
        selected_records=selected_records,
    ) as ledger:
        assert ledger.snapshot.records == (first_record,)
        ledger.append(source=second_source, augmentation=_augmentation(second_source))

    snapshot = read_augmentation_ledger(
        path,
        expected_context=context,
        selected_records=selected_records,
    )
    assert snapshot.processed_source_ids == frozenset({first_source.id, second_source.id})
    assert tuple(record.source.id for record in snapshot.records) == (
        first_source.id,
        second_source.id,
    )
    assert len(snapshot.raw_ledger_sha256) == 64


def test_private_ledger_never_overwrites_existing_file(tmp_path: Path) -> None:
    source = _source()
    context = _context((source,))
    path = tmp_path / "augmentations.jsonl"
    path.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError):
        create_private_augmentation_ledger(
            path,
            generation_context=context,
            selected_records=(source,),
        )

    assert path.read_text(encoding="utf-8") == "keep me"


def test_resume_rejects_truncated_record(tmp_path: Path) -> None:
    source = _source()
    context = _context((source,))
    path = tmp_path / "augmentations.jsonl"
    with create_private_augmentation_ledger(
        path,
        generation_context=context,
        selected_records=(source,),
    ) as ledger:
        ledger.append(source=source, augmentation=_augmentation(source))
    path.write_bytes(path.read_bytes()[:-1])

    with pytest.raises(ValueError, match="must end with a newline"):
        open_augmentation_ledger_for_resume(
            path,
            expected_context=context,
            selected_records=(source,),
        )


def test_read_is_bounded_before_parsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source()
    context = _context((source,))
    path = tmp_path / "augmentations.jsonl"
    with create_private_augmentation_ledger(
        path,
        generation_context=context,
        selected_records=(source,),
    ) as ledger:
        ledger.append(source=source, augmentation=_augmentation(source))
    monkeypatch.setattr(ledger_module, "_MAXIMUM_LEDGER_BYTES", 1)

    with pytest.raises(ValueError, match="128 MB limit"):
        read_augmentation_ledger(
            path,
            expected_context=context,
            selected_records=(source,),
        )


def test_resume_rejects_duplicate_source_records(tmp_path: Path) -> None:
    source = _source()
    context = _context((source,))
    path = tmp_path / "augmentations.jsonl"
    with create_private_augmentation_ledger(
        path,
        generation_context=context,
        selected_records=(source,),
    ) as ledger:
        ledger.append(source=source, augmentation=_augmentation(source))
    original_record = path.read_bytes()
    with path.open("ab") as output_stream:
        output_stream.write(original_record)

    with pytest.raises(ValueError, match="duplicate source interactions"):
        read_augmentation_ledger(
            path,
            expected_context=context,
            selected_records=(source,),
        )


def test_resume_rejects_generation_context_and_source_mismatches(tmp_path: Path) -> None:
    source = _source()
    context = _context((source,))
    path = tmp_path / "augmentations.jsonl"
    with create_private_augmentation_ledger(
        path,
        generation_context=context,
        selected_records=(source,),
    ) as ledger:
        ledger.append(source=source, augmentation=_augmentation(source))

    with pytest.raises(ValueError, match="incompatible generation context"):
        read_augmentation_ledger(
            path,
            expected_context=_context((source,), model="changed/model"),
            selected_records=(source,),
        )

    changed_source = source.model_copy(update={"raw_input": "Changed input"})
    with pytest.raises(ValueError, match="generation context does not match"):
        read_augmentation_ledger(
            path,
            expected_context=context,
            selected_records=(changed_source,),
        )


def test_resume_rejects_tampered_record_and_duplicate_json_keys(tmp_path: Path) -> None:
    source = _source()
    context = _context((source,))
    path = tmp_path / "augmentations.jsonl"
    with create_private_augmentation_ledger(
        path,
        generation_context=context,
        selected_records=(source,),
    ) as ledger:
        ledger.append(source=source, augmentation=_augmentation(source))

    decoded_record = json.loads(path.read_text(encoding="utf-8"))
    decoded_record["source"]["raw_input"] = "Tampered input"
    path.write_text(json.dumps(decoded_record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid UL JSONL"):
        read_augmentation_ledger(
            path,
            expected_context=context,
            selected_records=(source,),
        )

    path.write_text('{"schema_version":"1.0.0","schema_version":"1.0.0"}\n')
    with pytest.raises(ValueError, match="not valid JSON"):
        read_augmentation_ledger(
            path,
            expected_context=context,
            selected_records=(source,),
        )


def test_resume_rejects_values_coerced_by_nested_sdk_models(tmp_path: Path) -> None:
    source = _source()
    context = _context((source,))
    path = tmp_path / "augmentations.jsonl"
    with create_private_augmentation_ledger(
        path,
        generation_context=context,
        selected_records=(source,),
    ) as ledger:
        ledger.append(source=source, augmentation=_augmentation(source))

    decoded_record = json.loads(path.read_text(encoding="utf-8"))
    decoded_record["augmentation"]["candidates"][0]["passed"] = 1
    path.write_text(json.dumps(decoded_record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-strict values"):
        read_augmentation_ledger(
            path,
            expected_context=context,
            selected_records=(source,),
        )


def test_record_rejects_candidate_outside_context() -> None:
    source = _source()
    context = _context((source,))
    augmentation = _augmentation(source)
    changed_candidate = augmentation.candidates[0].model_copy(
        update={"operator_id": "input.style.terse"}
    )

    with pytest.raises(ValidationError, match="outside the requested operator plan"):
        create_dataset_augmentation_ledger_record(
            generation_context=context,
            source=source,
            augmentation=augmentation.model_copy(update={"candidates": (changed_candidate,)}),
        )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_resume_rejects_nonprivate_permissions(tmp_path: Path) -> None:
    source = _source()
    context = _context((source,))
    path = tmp_path / "augmentations.jsonl"
    with create_private_augmentation_ledger(
        path,
        generation_context=context,
        selected_records=(source,),
    ):
        pass
    path.chmod(0o640)

    with pytest.raises(OSError, match="permissions must be 0600"):
        read_augmentation_ledger(
            path,
            expected_context=context,
            selected_records=(source,),
        )


def test_resume_rejects_symlink(tmp_path: Path) -> None:
    source = _source()
    context = _context((source,))
    path = tmp_path / "augmentations.jsonl"
    with create_private_augmentation_ledger(
        path,
        generation_context=context,
        selected_records=(source,),
    ):
        pass
    link = tmp_path / "augmentations-link.jsonl"
    link.symlink_to(path)

    with pytest.raises(OSError, match="not a regular file"):
        read_augmentation_ledger(
            link,
            expected_context=context,
            selected_records=(source,),
        )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX locking behavior")
def test_resume_holds_an_exclusive_lock(tmp_path: Path) -> None:
    source = _source()
    context = _context((source,))
    path = tmp_path / "augmentations.jsonl"
    with create_private_augmentation_ledger(
        path,
        generation_context=context,
        selected_records=(source,),
    ):
        pass

    with (
        open_augmentation_ledger_for_resume(
            path,
            expected_context=context,
            selected_records=(source,),
        ),
        pytest.raises(BlockingIOError),
    ):
        open_augmentation_ledger_for_resume(
            path,
            expected_context=context,
            selected_records=(source,),
        )


def test_append_rejects_duplicate_and_unselected_sources(tmp_path: Path) -> None:
    source = _source()
    other_source = _source("interaction-2")
    context = _context((source,))
    path = tmp_path / "augmentations.jsonl"
    with create_private_augmentation_ledger(
        path,
        generation_context=context,
        selected_records=(source,),
    ) as ledger:
        ledger.append(source=source, augmentation=_augmentation(source))
        with pytest.raises(ValueError, match="already contains"):
            ledger.append(source=source, augmentation=_augmentation(source))
        with pytest.raises(ValueError, match="does not match the selected dataset"):
            ledger.append(source=other_source, augmentation=_augmentation(other_source))

    assert os.stat(path).st_size > 0
