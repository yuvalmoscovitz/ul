from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError
from ul import RichInteractionCase

_MAXIMUM_RECORDS = 100_000
_MAXIMUM_RECORD_BYTES = 2_000_000
_MAXIMUM_FUNCTIONS_PER_RECORD = 128
_MAXIMUM_SCHEMA_DEPTH = 100
_MAXIMUM_SCHEMA_NODES = 100_000
_OPENAI_FUNCTION_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
_SUPPORTED_CATEGORIES = (
    "live_parallel_multiple",
    "parallel_multiple",
    "simple_javascript",
    "simple_python",
    "simple_java",
    "live_parallel",
    "live_multiple",
    "live_simple",
    "parallel",
    "multiple",
)
_GORILLA_TO_OPENAI_TYPE = {
    "": "string",
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "Boolean": "boolean",
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
    def __init__(self, message: str, *, param_hint: str | None = None) -> None:
        super().__init__(message)
        self.param_hint = param_hint


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _BfclMessage(_StrictModel):
    role: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=1)


class _BfclRecord(_StrictModel):
    id: str = Field(min_length=1, max_length=500)


class _BfclQuestion(_BfclRecord):
    question: list[list[_BfclMessage]]
    function: list[dict[str, JsonValue]] = Field(
        min_length=1,
        max_length=_MAXIMUM_FUNCTIONS_PER_RECORD,
    )


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
    category = _bounded_text(category, field="category", param_hint="--category")
    source_revision = _bounded_text(
        source_revision,
        field="source revision",
        param_hint="--source-revision",
    )
    if category not in _SUPPORTED_CATEGORIES:
        raise BfclInputError(
            f"unsupported BFCL V4 category {category!r}",
            param_hint="--category",
        )
    questions = _parse_jsonl(
        question_bytes,
        _BfclQuestion,
        label="question",
        param_hint="QUESTIONS",
    )
    answers = _parse_jsonl(
        answer_bytes,
        _BfclAnswer,
        label="possible-answer",
        param_hint="POSSIBLE_ANSWERS",
    )
    mismatched_ids = [record.id for record in questions if _category_from_id(record.id) != category]
    if mismatched_ids:
        raise BfclInputError(
            f"BFCL question IDs do not match category {category!r}",
            param_hint="--category",
        )
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
    param_hint: str,
) -> tuple[BfclRecordType, ...]:
    try:
        lines = encoded.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        raise BfclInputError(
            f"BFCL {label} file must be UTF-8",
            param_hint=param_hint,
        ) from None
    if not lines:
        raise BfclInputError(
            f"BFCL {label} file contains no records",
            param_hint=param_hint,
        )
    records: list[BfclRecordType] = []
    known_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise BfclInputError(
                f"BFCL {label} line {line_number} is blank",
                param_hint=param_hint,
            )
        if len(line.encode("utf-8")) > _MAXIMUM_RECORD_BYTES:
            raise BfclInputError(
                f"BFCL {label} line {line_number} exceeds the 2 MB record limit",
                param_hint=param_hint,
            )
        if len(records) == _MAXIMUM_RECORDS:
            raise BfclInputError(
                f"BFCL {label} file exceeds {_MAXIMUM_RECORDS} records",
                param_hint=param_hint,
            )
        try:
            payload = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonstandard_constant,
            )
            record = model.model_validate(payload)
        except (json.JSONDecodeError, RecursionError, ValidationError, ValueError):
            raise BfclInputError(
                f"BFCL {label} line {line_number} is invalid",
                param_hint=param_hint,
            ) from None
        if record.id in known_ids:
            raise BfclInputError(
                f"BFCL {label} line {line_number} has a duplicate ID",
                param_hint=param_hint,
            )
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
    if len(question.question) != 1:
        raise BfclInputError(f"BFCL record {question.id!r} is not a supported single-turn case")
    messages = question.question[0]
    user_message_indexes = [
        index for index, message in enumerate(messages) if message.role == "user"
    ]
    if len(user_message_indexes) != 1 or any(
        message.role not in {"system", "user"} for message in messages
    ):
        raise BfclInputError(
            f"BFCL record {question.id!r} does not contain one supported user turn"
        )
    user_message_index = user_message_indexes[0]
    openai_tools, tool_name_map = _openai_tools(question.function, record_id=question.id)
    try:
        return RichInteractionCase.model_validate(
            {
                "schema_version": "1.0.0",
                "id": question.id,
                "inputs": {
                    "messages": [message.model_dump() for message in messages],
                    "bfcl_functions": question.function,
                    "openai_tools": openai_tools,
                    "openai_tool_name_map": tool_name_map,
                },
                "augmentation_targets": [
                    {
                        "id": "user-request",
                        "kind": "input_field",
                        "json_pointer": f"/inputs/messages/{user_message_index}/content",
                    }
                ],
                "fixture": {"id": "bfcl-v4-stateless", "version": source_revision},
                "observed_output": {
                    "kind": "bfcl_reference",
                    "ground_truth": answer.ground_truth,
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
        normalized_parameters = _normalize_openai_schema(parameters, record_id=record_id)
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


def _normalize_openai_schema(
    value: dict[str, JsonValue],
    *,
    record_id: str,
) -> dict[str, JsonValue]:
    visited_nodes = [0]
    normalized = _normalize_schema_value(
        value,
        record_id=record_id,
        depth=0,
        visited_nodes=visited_nodes,
    )
    return cast(dict[str, JsonValue], normalized)


def _normalize_schema_value(
    value: JsonValue,
    *,
    record_id: str,
    depth: int,
    visited_nodes: list[int],
) -> JsonValue:
    if depth > _MAXIMUM_SCHEMA_DEPTH:
        raise BfclInputError(f"BFCL schema is too deep in record {record_id!r}")
    visited_nodes[0] += 1
    if visited_nodes[0] > _MAXIMUM_SCHEMA_NODES:
        raise BfclInputError(f"BFCL record {record_id!r} has a function schema that is too large")
    if isinstance(value, dict):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if key == "type" and isinstance(item, str):
                normalized_type = _GORILLA_TO_OPENAI_TYPE.get(item)
                if normalized_type is None:
                    raise BfclInputError(
                        f"unsupported BFCL schema type {item!r} in record {record_id!r}"
                    )
                normalized[key] = normalized_type
            else:
                normalized[key] = _normalize_schema_value(
                    item,
                    record_id=record_id,
                    depth=depth + 1,
                    visited_nodes=visited_nodes,
                )
        return normalized
    if isinstance(value, list):
        return [
            _normalize_schema_value(
                item,
                record_id=record_id,
                depth=depth + 1,
                visited_nodes=visited_nodes,
            )
            for item in value
        ]
    return value


def _bounded_text(value: str, *, field: str, param_hint: str) -> str:
    stripped = value.strip()
    if not stripped or len(stripped) > 200:
        raise BfclInputError(
            f"BFCL {field} must contain 1 to 200 non-whitespace characters",
            param_hint=param_hint,
        )
    return stripped


def _category_from_id(record_id: str) -> str | None:
    for category in _SUPPORTED_CATEGORIES:
        if record_id.startswith(f"{category}_"):
            return category
    return None


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")
