from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

_JSON_POINTER_PATTERN = re.compile(r"(?:/(?:[^~/]|~[01])*)*")
_MAXIMUM_NORMALIZED_BYTES = 64_000
_MAXIMUM_TOOL_ARGUMENT_DEPTH = 100
_MAXIMUM_TOOL_ARGUMENT_NODES = 10_000
_MAXIMUM_TOOL_ARGUMENT_FIELDS = 2_000


class OutcomeProjectionError(ValueError):
    def __init__(self, field: str, selector: str, reason: str) -> None:
        super().__init__(f"outcome field {field!r} at selector {selector!r} {reason}")
        self.field = field
        self.selector = selector
        self.reason = reason


class ToolCallOutcomeProjection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    name: str = Field(max_length=1_000)
    arguments: str = Field(max_length=1_000)


class OutcomeProjection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    action: str | None = Field(default=None, max_length=1_000)
    status: str | None = Field(default=None, max_length=1_000)
    resource_id: str | None = Field(default=None, max_length=1_000)
    decision: str | None = Field(default=None, max_length=1_000)
    amount: str | None = Field(default=None, max_length=1_000)
    effects: str | None = Field(default=None, max_length=1_000)
    complete_result: str | None = Field(default=None, max_length=1_000)
    tool_call: ToolCallOutcomeProjection | None = None
    private_json_pointers: tuple[str, ...] = Field(default=(), max_length=100)

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        selectors = self.field_selectors
        selected_modes = sum(
            (bool(selectors), self.complete_result is not None, self.tool_call is not None)
        )
        if selected_modes != 1:
            raise ValueError(
                "outcome requires exactly one of field selectors, complete_result, or tool_call"
            )
        all_selectors = (
            tuple(selectors.values())
            + ((self.complete_result,) if self.complete_result is not None else ())
            + (
                (self.tool_call.name, self.tool_call.arguments)
                if self.tool_call is not None
                else ()
            )
        )
        if any(_JSON_POINTER_PATTERN.fullmatch(pointer) is None for pointer in all_selectors):
            raise ValueError("outcome selectors must follow RFC 6901 syntax")
        if len(all_selectors) != len(set(all_selectors)):
            raise ValueError("outcome selectors must be unique")
        if len(self.private_json_pointers) != len(set(self.private_json_pointers)):
            raise ValueError("private outcome pointers must be unique")
        if any(
            not pointer or _JSON_POINTER_PATTERN.fullmatch(pointer) is None
            for pointer in self.private_json_pointers
        ):
            raise ValueError("private outcome pointers must identify nested RFC 6901 values")
        private_tokens = tuple(_pointer_tokens(pointer) for pointer in self.private_json_pointers)
        if any(
            left == right[: len(left)] or right == left[: len(right)]
            for index, left in enumerate(private_tokens)
            for right in private_tokens[index + 1 :]
        ):
            raise ValueError("private outcome pointers must not overlap")
        return self

    @property
    def field_selectors(self) -> dict[str, str]:
        return {
            name: pointer
            for name in ("action", "status", "resource_id", "decision", "amount", "effects")
            if (pointer := cast(str | None, getattr(self, name))) is not None
        }

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def project(self, response: JsonValue) -> dict[str, JsonValue]:
        if self.complete_result is not None:
            result = _resolve(response, self.complete_result, field="complete_result")
            if not isinstance(result, dict):
                raise OutcomeProjectionError(
                    "complete_result", self.complete_result, "must resolve to a JSON object"
                )
            normalized = copy.deepcopy(result)
        elif self.tool_call is not None:
            normalized = _project_tool_call(response, self.tool_call)
        else:
            normalized = {
                field: copy.deepcopy(_resolve(response, pointer, field=field))
                for field, pointer in self.field_selectors.items()
            }
            _validate_common_roles(normalized, self.field_selectors)
        try:
            encoded = json.dumps(
                normalized,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError):
            raise OutcomeProjectionError(
                "complete_result", self.complete_result or "", "is not JSON"
            ) from None
        if len(encoded) > _MAXIMUM_NORMALIZED_BYTES:
            raise OutcomeProjectionError(
                "complete_result" if self.complete_result is not None else "outcome",
                self.complete_result or "",
                f"exceeds the {_MAXIMUM_NORMALIZED_BYTES}-byte normalized-result limit",
            )
        for pointer in self.private_json_pointers:
            _resolve(normalized, pointer, field="private_json_pointers")
        return cast(dict[str, JsonValue], normalized)

    def public_result(self, normalized: dict[str, JsonValue]) -> dict[str, JsonValue]:
        public = copy.deepcopy(normalized)
        for pointer in self.private_json_pointers:
            try:
                _resolve(public, pointer, field="private_json_pointers")
                _replace_with_private_marker(public, pointer)
            except OutcomeProjectionError:
                raise
            except (AssertionError, IndexError, KeyError, TypeError, ValueError):
                raise OutcomeProjectionError(
                    "private_json_pointers", pointer, "does not resolve"
                ) from None
        return public


def _resolve(value: JsonValue, pointer: str, *, field: str) -> JsonValue:
    current: JsonValue = value
    for encoded_token in pointer[1:].split("/") if pointer else ():
        token = encoded_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif (
            isinstance(current, list)
            and token.isascii()
            and token.isdecimal()
            and (token == "0" or not token.startswith("0"))
            and int(token) < len(current)
        ):
            current = current[int(token)]
        else:
            raise OutcomeProjectionError(field, pointer, "does not resolve")
    return current


def _pointer_tokens(pointer: str) -> tuple[str, ...]:
    return tuple(token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/"))


def _validate_common_roles(normalized: dict[str, JsonValue], selectors: dict[str, str]) -> None:
    for field in ("action", "status", "resource_id", "decision"):
        if field in normalized and (
            not isinstance(normalized[field], str) or not normalized[field]
        ):
            raise OutcomeProjectionError(
                field, selectors[field], "must resolve to a non-empty string"
            )
    if "amount" in normalized:
        amount = normalized["amount"]
        if (
            isinstance(amount, bool)
            or not isinstance(amount, int | float)
            or (isinstance(amount, float) and not math.isfinite(amount))
        ):
            raise OutcomeProjectionError(
                "amount", selectors["amount"], "must resolve to a finite number"
            )
    if "effects" in normalized and not isinstance(normalized["effects"], list | dict):
        raise OutcomeProjectionError(
            "effects", selectors["effects"], "must resolve to an object or array"
        )


def _project_tool_call(
    response: JsonValue,
    projection: ToolCallOutcomeProjection,
) -> dict[str, JsonValue]:
    name = _resolve(response, projection.name, field="tool_call.name")
    if not isinstance(name, str) or not name:
        raise OutcomeProjectionError(
            "tool_call.name",
            projection.name,
            "must resolve to a non-empty string",
        )
    encoded_arguments = _resolve(
        response,
        projection.arguments,
        field="tool_call.arguments",
    )
    if not isinstance(encoded_arguments, str):
        raise OutcomeProjectionError(
            "tool_call.arguments",
            projection.arguments,
            "must resolve to a JSON-encoded object string",
        )
    try:
        encoded_argument_size = len(encoded_arguments.encode("utf-8"))
    except UnicodeEncodeError:
        raise OutcomeProjectionError(
            "tool_call.arguments",
            projection.arguments,
            "must resolve to a valid JSON-encoded object string",
        ) from None
    if encoded_argument_size > _MAXIMUM_NORMALIZED_BYTES:
        raise OutcomeProjectionError(
            "tool_call.arguments",
            projection.arguments,
            f"exceeds the {_MAXIMUM_NORMALIZED_BYTES}-byte argument limit",
        )
    try:
        arguments = json.loads(
            encoded_arguments,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (RecursionError, TypeError, ValueError):
        raise OutcomeProjectionError(
            "tool_call.arguments",
            projection.arguments,
            "must resolve to a valid JSON-encoded object string",
        ) from None
    if not isinstance(arguments, dict):
        raise OutcomeProjectionError(
            "tool_call.arguments",
            projection.arguments,
            "must resolve to a JSON-encoded object string",
        )
    if "action" in arguments:
        raise OutcomeProjectionError(
            "tool_call.arguments",
            projection.arguments,
            "must not contain the reserved action field",
        )
    try:
        flattened_arguments = _flatten_argument_fields(cast(dict[str, JsonValue], arguments))
    except ValueError as error:
        raise OutcomeProjectionError(
            "tool_call.arguments",
            projection.arguments,
            str(error),
        ) from None
    return {"action": name, **flattened_arguments}


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(_: str) -> None:
    raise ValueError("non-finite JSON number")


def _flatten_argument_fields(arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
    flattened: dict[str, JsonValue] = {}
    pending: list[tuple[str, JsonValue, int]] = [
        (key, value, 1) for key, value in reversed(tuple(arguments.items()))
    ]
    node_count = 0
    while pending:
        prefix, value, depth = pending.pop()
        node_count += 1
        if depth > _MAXIMUM_TOOL_ARGUMENT_DEPTH or node_count > _MAXIMUM_TOOL_ARGUMENT_NODES:
            raise ValueError("exceeds canonical flattening limits")
        if isinstance(value, dict):
            nested_fields = tuple(value.items())
            if nested_fields:
                pending.extend(
                    (f"{prefix}.{key}", nested_value, depth + 1)
                    for key, nested_value in reversed(nested_fields)
                )
                continue
        elif isinstance(value, list) and value:
            pending.extend(
                (f"{prefix}[{index}]", nested_value, depth + 1)
                for index, nested_value in reversed(tuple(enumerate(value)))
            )
            continue
        if prefix in flattened:
            raise ValueError("contains ambiguous field names after canonical flattening")
        if len(flattened) >= _MAXIMUM_TOOL_ARGUMENT_FIELDS:
            raise ValueError("exceeds canonical flattening limits")
        flattened[prefix] = value
    return flattened


def _replace_with_private_marker(value: dict[str, JsonValue], pointer: str) -> None:
    tokens = [token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/")]
    parent: JsonValue = value
    for token in tokens[:-1]:
        if isinstance(parent, dict):
            parent = parent[token]
        elif isinstance(parent, list):
            parent = parent[int(token)]
        else:
            raise AssertionError("validated private pointer has a container parent")
    final = tokens[-1]
    if isinstance(parent, dict):
        parent[final] = "[PRIVATE]"
    elif isinstance(parent, list):
        parent[int(final)] = "[PRIVATE]"
    else:
        raise AssertionError("validated private pointer has a container parent")
