from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from pydantic import JsonValue, ValidationError
from ul import InteractionRecord

_MAXIMUM_DATASET_BYTES = 10_000_000
_MAXIMUM_DATASET_RECORDS = 100


class DatasetInputError(ValueError):
    pass


def load_interaction_records(path: Path) -> tuple[InteractionRecord, ...]:
    dataset_name = path.name
    try:
        with path.open("rb") as dataset_stream:
            encoded_dataset = dataset_stream.read(_MAXIMUM_DATASET_BYTES + 1)
        if len(encoded_dataset) > _MAXIMUM_DATASET_BYTES:
            raise DatasetInputError(
                f"{dataset_name}: dataset exceeds {_MAXIMUM_DATASET_BYTES} bytes"
            )
        lines = encoded_dataset.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        raise DatasetInputError(f"{dataset_name}: dataset must be UTF-8") from None
    except OSError as error:
        raise DatasetInputError(
            f"{dataset_name}: cannot read dataset ({error.__class__.__name__})"
        ) from None

    records: list[InteractionRecord] = []
    known_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise DatasetInputError(
                f"{dataset_name} line {line_number}: blank lines are not allowed"
            )
        if len(records) == _MAXIMUM_DATASET_RECORDS:
            raise DatasetInputError(
                f"{dataset_name} line {line_number}: dataset exceeds "
                f"{_MAXIMUM_DATASET_RECORDS} records"
            )
        try:
            untyped_payload = json.loads(
                line,
                object_pairs_hook=reject_duplicate_json_keys,
                parse_constant=reject_nonstandard_json_constant,
            )
        except (json.JSONDecodeError, RecursionError, ValueError):
            raise DatasetInputError(f"{dataset_name} line {line_number}: invalid JSON") from None
        if not isinstance(untyped_payload, dict):
            raise DatasetInputError(f"{dataset_name} line {line_number}: record must be an object")
        payload = cast(dict[str, JsonValue], untyped_payload)
        expected_fields = {"id", "input", "output"}
        payload_fields = set(payload)
        if payload_fields != expected_fields:
            missing_fields = sorted(expected_fields - payload_fields)
            unknown_fields = sorted(payload_fields - expected_fields)
            details: list[str] = []
            if missing_fields:
                details.append(f"missing {', '.join(missing_fields)}")
            if unknown_fields:
                details.append("unknown field(s)")
            raise DatasetInputError(
                f"{dataset_name} line {line_number}: expected exactly id, input, output "
                f"({'; '.join(details)})"
            )
        try:
            record = InteractionRecord.model_validate(
                {
                    "id": payload["id"],
                    "raw_input": payload["input"],
                    "raw_observed_output": payload["output"],
                }
            )
        except ValidationError as error:
            invalid_fields = sorted(
                {
                    {"raw_input": "input", "raw_observed_output": "output"}.get(
                        str(item["loc"][0]), str(item["loc"][0])
                    )
                    for item in error.errors(include_input=False)
                    if item["loc"]
                }
            )
            field_summary = ", ".join(invalid_fields) or "record"
            raise DatasetInputError(
                f"{dataset_name} line {line_number}: invalid {field_summary}"
            ) from None
        if record.id in known_ids:
            raise DatasetInputError(f"{dataset_name} line {line_number}: duplicate id")
        known_ids.add(record.id)
        records.append(record)
    if not records:
        raise DatasetInputError(f"{dataset_name}: dataset contains no records")
    return tuple(records)


def validate_interaction_dataset(path: Path) -> None:
    """Validate a dataset without executing or retaining its records."""
    load_interaction_records(path)


def reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")


def reject_duplicate_json_keys(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def validate_model_input_bounds(
    records: tuple[InteractionRecord, ...],
    maximum_characters: int,
) -> None:
    for case_number, record in enumerate(records, start=1):
        serialized_record = json.dumps(
            {
                "raw_input": record.raw_input,
                "raw_observed_output": record.raw_observed_output,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(serialized_record) > maximum_characters:
            raise DatasetInputError(
                f"selected interaction {case_number} exceeds the semantic model input limit"
            )
