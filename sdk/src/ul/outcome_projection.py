from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_serializer, model_validator

_JSON_POINTER_PATTERN = re.compile(r"(?:/(?:[^~/]|~[01])*)*")
_MAXIMUM_NORMALIZED_BYTES = 64_000
_MAXIMUM_COMPOSE_DEPTH = 100
_MAXIMUM_COMPOSE_NODES = 10_000
_MAXIMUM_COMPOSE_FIELDS = 2_000
_MAXIMUM_COMPOSE_FIELD_NAME = 1_000


class OutcomeProjectionError(ValueError):
    def __init__(self, field: str, selector: str, reason: str) -> None:
        super().__init__(f"outcome field {field!r} at selector {selector!r} {reason}")
        self.field = field
        self.selector = selector
        self.reason = reason


class OutcomeSpreadProjection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    selector: str = Field(max_length=1_000)
    decode: Literal["none", "json_string"] = "none"
    flatten: bool = False


class ComposedOutcomeProjection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    fields: Mapping[str, str] = Field(default_factory=dict, max_length=100)
    spread: OutcomeSpreadProjection | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if not self.fields and self.spread is None:
            raise ValueError("compose requires fields or spread")
        if any(not name or len(name) > _MAXIMUM_COMPOSE_FIELD_NAME for name in self.fields):
            raise ValueError("compose field names must contain 1..1000 characters")
        if any(len(selector) > 1_000 for selector in self.fields.values()):
            raise ValueError("compose field selectors must contain at most 1000 characters")
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        return self

    @field_serializer("fields")
    def serialize_fields(self, fields: Mapping[str, str]) -> dict[str, str]:
        return dict(fields)


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
    compose: ComposedOutcomeProjection | None = None
    private_json_pointers: tuple[str, ...] = Field(default=(), max_length=100)

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        selectors = self.field_selectors
        selected_modes = sum(
            (bool(selectors), self.complete_result is not None, self.compose is not None)
        )
        if selected_modes != 1:
            raise ValueError(
                "outcome requires exactly one of field selectors, complete_result, or compose"
            )
        all_selectors = (
            tuple(selectors.values())
            + ((self.complete_result,) if self.complete_result is not None else ())
            + (
                tuple(self.compose.fields.values())
                + ((self.compose.spread.selector,) if self.compose.spread is not None else ())
                if self.compose is not None
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
        elif self.compose is not None:
            normalized, common_role_selectors = _compose_outcome(response, self.compose)
            _validate_common_roles(normalized, common_role_selectors)
        else:
            normalized = {
                field: copy.deepcopy(_resolve(response, pointer, field=field))
                for field, pointer in self.field_selectors.items()
            }
            _validate_common_roles(normalized, self.field_selectors)
        try:
            _validate_json_structure(normalized)
            encoded_size = _encoded_json_size(normalized)
        except RecursionError:
            raise OutcomeProjectionError("outcome", "", "exceeds structural limits") from None
        except OverflowError:
            raise OutcomeProjectionError(
                "outcome",
                "",
                f"exceeds the {_MAXIMUM_NORMALIZED_BYTES}-byte normalized-result limit",
            ) from None
        except UnicodeEncodeError:
            raise OutcomeProjectionError("outcome", "", "contains invalid Unicode") from None
        except (TypeError, ValueError):
            raise OutcomeProjectionError(
                "complete_result", self.complete_result or "", "is not JSON"
            ) from None
        if encoded_size > _MAXIMUM_NORMALIZED_BYTES:
            raise OutcomeProjectionError(
                "complete_result" if self.complete_result is not None else "outcome",
                self.complete_result or "",
                f"exceeds the {_MAXIMUM_NORMALIZED_BYTES}-byte normalized-result limit",
            )
        for pointer in self.private_json_pointers:
            _resolve(normalized, pointer, field="private_json_pointers")
        return copy.deepcopy(cast(dict[str, JsonValue], normalized))

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


def _compose_outcome(
    response: JsonValue,
    projection: ComposedOutcomeProjection,
) -> tuple[dict[str, JsonValue], dict[str, str]]:
    normalized = {
        field: _resolve(response, selector, field=f"compose.fields.{field}")
        for field, selector in projection.fields.items()
    }
    spread = projection.spread
    if spread is not None:
        spread_value = _resolve(response, spread.selector, field="compose.spread")
        spread_object = _decode_spread_object(spread_value, spread)
        if spread.flatten:
            try:
                spread_fields = _flatten_fields(spread_object)
            except ValueError as error:
                raise OutcomeProjectionError(
                    "compose.spread",
                    spread.selector,
                    str(error),
                ) from None
        else:
            try:
                _validate_json_structure(spread_object)
            except RecursionError:
                raise OutcomeProjectionError(
                    "compose.spread",
                    spread.selector,
                    "exceeds structural limits",
                ) from None
            except OverflowError:
                raise OutcomeProjectionError(
                    "compose.spread",
                    spread.selector,
                    f"exceeds the {_MAXIMUM_NORMALIZED_BYTES}-byte normalized-result limit",
                ) from None
            spread_fields = spread_object
        collisions = normalized.keys() & spread_fields.keys()
        if collisions:
            raise OutcomeProjectionError(
                "compose.spread",
                spread.selector,
                "collides with a selected composed field",
            )
        normalized.update(spread_fields)
    common_role_selectors = {
        field: projection.fields.get(field, spread.selector if spread is not None else "")
        for field in ("action", "status", "resource_id", "decision", "amount", "effects")
        if field in normalized
    }
    return normalized, common_role_selectors


def _decode_spread_object(
    value: JsonValue,
    spread: OutcomeSpreadProjection,
) -> dict[str, JsonValue]:
    if spread.decode == "none":
        if isinstance(value, dict):
            return value
        raise OutcomeProjectionError(
            "compose.spread",
            spread.selector,
            "must resolve to a JSON object",
        )
    if not isinstance(value, str):
        raise OutcomeProjectionError(
            "compose.spread",
            spread.selector,
            "must resolve to a JSON-encoded object string",
        )
    try:
        encoded_size = len(value.encode("utf-8"))
    except UnicodeEncodeError:
        raise OutcomeProjectionError(
            "compose.spread",
            spread.selector,
            "must resolve to a valid JSON-encoded object string",
        ) from None
    if encoded_size > _MAXIMUM_NORMALIZED_BYTES:
        raise OutcomeProjectionError(
            "compose.spread",
            spread.selector,
            f"exceeds the {_MAXIMUM_NORMALIZED_BYTES}-byte encoded-object limit",
        )
    try:
        decoded = json.loads(
            value,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (RecursionError, TypeError, ValueError):
        raise OutcomeProjectionError(
            "compose.spread",
            spread.selector,
            "must resolve to a valid JSON-encoded object string",
        ) from None
    if not isinstance(decoded, dict):
        raise OutcomeProjectionError(
            "compose.spread",
            spread.selector,
            "must resolve to a JSON-encoded object string",
        )
    return cast(dict[str, JsonValue], decoded)


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(_: str) -> None:
    raise ValueError("non-finite JSON number")


def _flatten_fields(arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
    flattened: dict[str, JsonValue] = {}
    pending: list[tuple[str, Iterator[tuple[str | int, JsonValue]], int, bool]] = [
        ("", cast(Iterator[tuple[str | int, JsonValue]], iter(arguments.items())), 0, False)
    ]
    node_count = 0
    while pending:
        parent_prefix, children, parent_depth, indexed = pending[-1]
        try:
            segment, value = next(children)
        except StopIteration:
            pending.pop()
            continue
        segment_text = f"[{segment}]" if indexed else cast(str, segment)
        separator = "" if not parent_prefix or indexed else "."
        if len(parent_prefix) + len(separator) + len(segment_text) > _MAXIMUM_COMPOSE_FIELD_NAME:
            raise ValueError("exceeds canonical flattening limits")
        prefix = f"{parent_prefix}{separator}{segment_text}"
        depth = parent_depth + 1
        node_count += 1
        if depth > _MAXIMUM_COMPOSE_DEPTH or node_count > _MAXIMUM_COMPOSE_NODES:
            raise ValueError("exceeds canonical flattening limits")
        if isinstance(value, dict) and value:
            pending.append(
                (
                    prefix,
                    cast(Iterator[tuple[str | int, JsonValue]], iter(value.items())),
                    depth,
                    False,
                )
            )
            continue
        elif isinstance(value, list) and value:
            pending.append(
                (
                    prefix,
                    cast(Iterator[tuple[str | int, JsonValue]], iter(enumerate(value))),
                    depth,
                    True,
                )
            )
            continue
        if prefix in flattened:
            raise ValueError("contains ambiguous field names after canonical flattening")
        if len(flattened) >= _MAXIMUM_COMPOSE_FIELDS:
            raise ValueError("exceeds canonical flattening limits")
        flattened[prefix] = value
    return flattened


def _validate_json_structure(value: JsonValue) -> None:
    pending: list[Iterator[tuple[JsonValue, int, int]]] = [iter(((value, 0, 0),))]
    node_count = 0
    character_count = 0
    while pending:
        try:
            current, depth, key_characters = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        node_count += 1
        if depth > _MAXIMUM_COMPOSE_DEPTH or node_count > _MAXIMUM_COMPOSE_NODES:
            raise RecursionError
        character_count += key_characters
        if isinstance(current, str):
            character_count += len(current)
        if character_count > _MAXIMUM_NORMALIZED_BYTES:
            raise OverflowError
        if isinstance(current, dict):
            pending.append(((nested, depth + 1, len(key)) for key, nested in current.items()))
        elif isinstance(current, list):
            pending.append((nested, depth + 1, 0) for nested in current)


def _encoded_json_size(value: JsonValue) -> int:
    encoded_size = 0
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    for chunk in encoder.iterencode(value):
        encoded_size += len(chunk.encode("utf-8"))
        if encoded_size > _MAXIMUM_NORMALIZED_BYTES:
            return encoded_size
    return encoded_size


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
