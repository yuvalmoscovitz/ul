from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError
from ul import RichInteractionCase

_MAXIMUM_RECORDS = 100_000
_OPENAI_FUNCTION_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
_GORILLA_TO_OPENAI_TYPE = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
    "byte": "integer",
    "short": "integer",
    "long": "integer",
    "double": "number",
    "char": "string",
    "ArrayList": "array",
    "Array": "array",
    "HashMap": "object",
    "Hashtable": "object",
    "Queue": "array",
    "Stack": "array",
    "Any": "string",
    "String": "string",
    "Bigint": "integer",
}


class BfclInputError(ValueError):
    pass


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _BfclMessage(_StrictModel):
    role: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=1)


class _BfclRecord(_StrictModel):
    id: str = Field(min_length=1, max_length=500)


class _BfclQuestion(_BfclRecord):
    question: list[list[_BfclMessage]]
    function: list[dict[str, JsonValue]] = Field(min_length=1)


class _BfclAnswer(_BfclRecord):
    ground_truth: list[JsonValue]


@dataclass(frozen=True)
class BfclIngestResult:
    records: tuple[RichInteractionCase, ...]
    source_record_count: int
    question_sha256: str
    answer_sha256: str


def materialize_bfcl_cohort(
    question_bytes: bytes,
    answer_bytes: bytes,
    *,
    category: str,
    source_revision: str,
    seed: int,
    limit: int,
) -> BfclIngestResult:
    category = _bounded_text(category, field="category")
    source_revision = _bounded_text(source_revision, field="source revision")
    questions = _parse_jsonl(question_bytes, _BfclQuestion, label="question")
    answers = _parse_jsonl(answer_bytes, _BfclAnswer, label="possible-answer")
    answers_by_id = _index_by_id(answers, label="possible-answer")
    question_ids = {record.id for record in questions}
    answer_ids = set(answers_by_id)
    if question_ids != answer_ids:
        missing_answers = len(question_ids - answer_ids)
        missing_questions = len(answer_ids - question_ids)
        raise BfclInputError(
            "BFCL question and possible-answer IDs do not align "
            f"({missing_answers} missing answer(s), {missing_questions} missing question(s))"
        )
    if limit > len(questions):
        raise BfclInputError(
            f"requested {limit} records but the BFCL category contains {len(questions)}"
        )

    question_sha256 = hashlib.sha256(question_bytes).hexdigest()
    answer_sha256 = hashlib.sha256(answer_bytes).hexdigest()
    ranked_questions = sorted(
        questions,
        key=lambda record: (
            hashlib.sha256(f"{seed}\0{record.id}".encode()).digest(),
            record.id,
        ),
    )
    records = tuple(
        _materialize_case(
            question,
            answers_by_id[question.id],
            category=category,
            source_revision=source_revision,
            question_sha256=question_sha256,
            answer_sha256=answer_sha256,
            seed=seed,
            sample_rank=sample_rank,
        )
        for sample_rank, question in enumerate(ranked_questions[:limit], start=1)
    )
    return BfclIngestResult(
        records=records,
        source_record_count=len(questions),
        question_sha256=question_sha256,
        answer_sha256=answer_sha256,
    )


def _parse_jsonl[BfclRecordType: _BfclRecord](
    encoded: bytes,
    model: type[BfclRecordType],
    *,
    label: str,
) -> tuple[BfclRecordType, ...]:
    try:
        lines = encoded.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        raise BfclInputError(f"BFCL {label} file must be UTF-8") from None
    if not lines:
        raise BfclInputError(f"BFCL {label} file contains no records")
    records: list[BfclRecordType] = []
    known_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise BfclInputError(f"BFCL {label} line {line_number} is blank")
        if len(records) == _MAXIMUM_RECORDS:
            raise BfclInputError(f"BFCL {label} file exceeds {_MAXIMUM_RECORDS} records")
        try:
            payload = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonstandard_constant,
            )
            record = model.model_validate(payload)
        except (json.JSONDecodeError, RecursionError, ValidationError, ValueError):
            raise BfclInputError(f"BFCL {label} line {line_number} is invalid") from None
        if record.id in known_ids:
            raise BfclInputError(f"BFCL {label} line {line_number} has a duplicate ID")
        known_ids.add(record.id)
        records.append(record)
    return tuple(records)


def _index_by_id(
    records: tuple[_BfclAnswer, ...],
    *,
    label: str,
) -> dict[str, _BfclAnswer]:
    indexed = {record.id: record for record in records}
    if len(indexed) != len(records):
        raise BfclInputError(f"BFCL {label} file contains duplicate IDs")
    return indexed


def _materialize_case(
    question: _BfclQuestion,
    answer: _BfclAnswer,
    *,
    category: str,
    source_revision: str,
    question_sha256: str,
    answer_sha256: str,
    seed: int,
    sample_rank: int,
) -> RichInteractionCase:
    if len(question.question) != 1 or len(question.question[0]) != 1:
        raise BfclInputError(
            f"BFCL record {question.id!r} is not a supported single-turn, single-message case"
        )
    message = question.question[0][0]
    if message.role != "user":
        raise BfclInputError(f"BFCL record {question.id!r} does not contain one user message")
    openai_tools, tool_name_map = _openai_tools(question.function, record_id=question.id)
    try:
        return RichInteractionCase.model_validate(
            {
                "schema_version": "1.0.0",
                "id": question.id,
                "inputs": {
                    "messages": [{"role": "user", "content": message.content}],
                    "bfcl_functions": copy.deepcopy(question.function),
                    "openai_tools": openai_tools,
                    "openai_tool_name_map": tool_name_map,
                },
                "augmentation_targets": [
                    {
                        "id": "user-request",
                        "kind": "input_field",
                        "json_pointer": "/inputs/messages/0/content",
                    }
                ],
                "fixture": {"id": "bfcl-v4-stateless", "version": source_revision},
                "observed_output": {
                    "kind": "bfcl_reference",
                    "ground_truth": copy.deepcopy(answer.ground_truth),
                },
                "metadata": {
                    "source": {
                        "benchmark": "BFCL",
                        "version": "v4",
                        "category": category,
                        "revision": source_revision,
                        "question_sha256": question_sha256,
                        "possible_answer_sha256": answer_sha256,
                        "sampling_algorithm": "sha256-seeded-rank-v1",
                        "sampling_seed": seed,
                        "sample_rank": sample_rank,
                    }
                },
            }
        )
    except ValidationError:
        raise BfclInputError(f"BFCL record {question.id!r} cannot form a valid UL case") from None


def _openai_tools(
    functions: list[dict[str, JsonValue]],
    *,
    record_id: str,
) -> tuple[list[dict[str, JsonValue]], dict[str, JsonValue]]:
    tools: list[dict[str, JsonValue]] = []
    name_map: dict[str, JsonValue] = {}
    for function in functions:
        raw_name = function.get("name")
        description = function.get("description")
        parameters = function.get("parameters")
        if (
            not isinstance(raw_name, str)
            or not raw_name
            or not isinstance(description, str)
            or not isinstance(parameters, dict)
        ):
            raise BfclInputError(f"BFCL record {record_id!r} has an invalid function schema")
        normalized_name = raw_name.replace(".", "_")
        if _OPENAI_FUNCTION_NAME.fullmatch(normalized_name) is None:
            raise BfclInputError(
                f"BFCL record {record_id!r} has a function name unsupported by OpenAI tools"
            )
        if normalized_name in name_map:
            raise BfclInputError(
                f"BFCL record {record_id!r} has colliding normalized function names"
            )
        normalized_parameters = _normalize_openai_schema(parameters)
        normalized_parameters["type"] = "object"
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": normalized_name,
                    "description": description,
                    "parameters": normalized_parameters,
                },
            }
        )
        name_map[normalized_name] = raw_name
    return tools, name_map


def _normalize_openai_schema(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    normalized = copy.deepcopy(value)
    raw_type = normalized.get("type")
    if isinstance(raw_type, str):
        normalized["type"] = _GORILLA_TO_OPENAI_TYPE.get(raw_type, "string")
    properties = normalized.get("properties")
    if isinstance(properties, dict):
        normalized["properties"] = {
            key: _normalize_openai_schema(cast(dict[str, JsonValue], item))
            if isinstance(item, dict)
            else item
            for key, item in properties.items()
        }
    items = normalized.get("items")
    if isinstance(items, dict):
        normalized["items"] = _normalize_openai_schema(cast(dict[str, JsonValue], items))
    additional_properties = normalized.get("additionalProperties")
    if isinstance(additional_properties, dict):
        normalized["additionalProperties"] = _normalize_openai_schema(
            cast(dict[str, JsonValue], additional_properties)
        )
    return normalized


def _bounded_text(value: str, *, field: str) -> str:
    stripped = value.strip()
    if not stripped or len(stripped) > 200:
        raise BfclInputError(f"BFCL {field} must contain 1 to 200 non-whitespace characters")
    return stripped


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")
