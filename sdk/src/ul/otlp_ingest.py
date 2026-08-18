from __future__ import annotations

import base64
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

_MAXIMUM_SPANS_PER_TRACE = 256
_MAXIMUM_MESSAGES_PER_TRACE = 512
_MAXIMUM_TOOL_CALLS_PER_TRACE = 512
_MAXIMUM_ERRORS_PER_TRACE = 512
_MAXIMUM_CONTENT_CHARACTERS = 100_000
_MAXIMUM_METADATA_CHARACTERS = 512


class OtlpAttributeMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: tuple[str, ...] = ("gen_ai.conversation.id", "session.id")
    agent_name: tuple[str, ...] = ("gen_ai.agent.name", "agent.name", "service.name")
    agent_version: tuple[str, ...] = ("gen_ai.agent.version", "service.version")
    source_reference: tuple[str, ...] = ("gen_ai.response.id", "prompt.url")
    input_messages: tuple[str, ...] = ("gen_ai.input.messages", "llm.input_messages")
    output_messages: tuple[str, ...] = ("gen_ai.output.messages", "llm.output_messages")
    prompt: tuple[str, ...] = ("gen_ai.prompt",)
    completion: tuple[str, ...] = ("gen_ai.completion",)
    tool_call_id: tuple[str, ...] = ("gen_ai.tool.call.id", "tool.id")
    tool_name: tuple[str, ...] = ("gen_ai.tool.name", "tool.name")
    tool_arguments: tuple[str, ...] = ("gen_ai.tool.call.arguments", "input.value")
    tool_result: tuple[str, ...] = ("gen_ai.tool.call.result", "output.value")
    retry_attempt: tuple[str, ...] = ("retry.attempt", "gen_ai.retry.attempt")
    state_snapshot: tuple[str, ...] = ("ul.state.snapshot",)
    state_delta: tuple[str, ...] = ("ul.state.delta",)
    error_type: tuple[str, ...] = ("error.type", "exception.type")
    error_message: tuple[str, ...] = ("exception.message",)


class OtlpMappingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    include_raw_content: bool = False
    maximum_content_characters: int = Field(default=16_000, ge=1, le=_MAXIMUM_CONTENT_CHARACTERS)
    attributes: OtlpAttributeMapping = Field(default_factory=OtlpAttributeMapping)


@dataclass(frozen=True)
class OtlpInteractionRecord:
    interaction_id: str
    input: str
    output: JsonValue


@dataclass(frozen=True)
class OtlpIngestResult:
    records: tuple[OtlpInteractionRecord, ...]
    skipped_no_gen_ai: int
    skipped_no_input: int
    skipped_no_output: int
    skipped_limit: int
    truncated: bool


@dataclass(frozen=True)
class _ExportedSpan:
    span: dict[str, Any]
    resource_attributes: dict[str, JsonValue]
    scope_name: str | None
    scope_version: str | None


def parse_otlp_traces(
    data: object,
    *,
    limit: int = 100,
    mapping: OtlpMappingConfig | None = None,
) -> OtlpIngestResult:
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    if not isinstance(data, dict):
        raise ValueError("OTLP export must be a JSON object")

    raw_resource_spans = cast(dict[str, Any], data).get("resourceSpans")
    if not isinstance(raw_resource_spans, list):
        raise ValueError("OTLP export must contain a resourceSpans array")

    spans_by_trace: dict[str, list[_ExportedSpan]] = {}
    for resource_span in cast(list[Any], raw_resource_spans):
        if not isinstance(resource_span, dict):
            continue
        typed_resource_span = cast(dict[str, Any], resource_span)
        resource = typed_resource_span.get("resource")
        resource_attributes = (
            _span_attributes(cast(dict[str, Any], resource)) if isinstance(resource, dict) else {}
        )
        for scope_span in cast(list[Any], typed_resource_span.get("scopeSpans") or []):
            if not isinstance(scope_span, dict):
                continue
            typed_scope_span = cast(dict[str, Any], scope_span)
            scope = typed_scope_span.get("scope")
            typed_scope = cast(dict[str, Any], scope) if isinstance(scope, dict) else {}
            scope_name = _nonempty_string(typed_scope.get("name"))
            scope_version = _nonempty_string(typed_scope.get("version"))
            for span in cast(list[Any], typed_scope_span.get("spans") or []):
                if not isinstance(span, dict):
                    continue
                typed_span = cast(dict[str, Any], span)
                trace_id = _normalize_id_field(typed_span.get("traceId"))
                if trace_id is None:
                    continue
                spans_by_trace.setdefault(trace_id, []).append(
                    _ExportedSpan(
                        span=typed_span,
                        resource_attributes=resource_attributes,
                        scope_name=scope_name,
                        scope_version=scope_version,
                    )
                )

    records: list[OtlpInteractionRecord] = []
    skipped_no_gen_ai = 0
    skipped_no_input = 0
    skipped_no_output = 0
    skipped_limit = 0
    truncated = False

    for trace_id in sorted(spans_by_trace):
        if len(records) >= limit:
            truncated = True
            break
        exported_spans = spans_by_trace[trace_id]
        gen_ai_spans = [item for item in exported_spans if _is_gen_ai_span(item.span)]
        if not gen_ai_spans:
            skipped_no_gen_ai += 1
            continue

        if mapping is None:
            target_span = _pick_root_span(gen_ai_spans)
            input_text = _extract_legacy_input(target_span.span)
            if input_text is None:
                skipped_no_input += 1
                continue
            output_text = _extract_legacy_output(target_span.span)
            if output_text is None:
                skipped_no_output += 1
                continue
            records.append(OtlpInteractionRecord(trace_id, input_text, output_text))
            continue

        if len(exported_spans) > _MAXIMUM_SPANS_PER_TRACE:
            skipped_limit += 1
            continue
        scenario = _build_trace_scenario(trace_id, exported_spans, mapping)
        if scenario is None:
            skipped_no_input += 1
            continue
        input_text, scenario_output = scenario
        records.append(OtlpInteractionRecord(trace_id, input_text, scenario_output))

    return OtlpIngestResult(
        records=tuple(records),
        skipped_no_gen_ai=skipped_no_gen_ai,
        skipped_no_input=skipped_no_input,
        skipped_no_output=skipped_no_output,
        skipped_limit=skipped_limit,
        truncated=truncated,
    )


def _build_trace_scenario(
    trace_id: str,
    exported_spans: list[_ExportedSpan],
    mapping: OtlpMappingConfig,
) -> tuple[str, dict[str, JsonValue]] | None:
    ordered_spans = sorted(
        exported_spans,
        key=lambda item: (
            _parse_nano(item.span.get("startTimeUnixNano")),
            _normalize_id_field(item.span.get("spanId")) or "",
        ),
    )
    canonical_spans: list[dict[str, JsonValue]] = []
    all_messages: list[dict[str, JsonValue]] = []
    resource_attributes: dict[str, JsonValue] = {}
    tool_call_count = 0
    error_count = 0

    for exported_span in ordered_spans:
        span = exported_span.span
        attributes = _span_attributes(span)
        if not resource_attributes:
            resource_attributes = exported_span.resource_attributes
        messages = _extract_span_messages(span, attributes, mapping)
        if len(all_messages) + len(messages) > _MAXIMUM_MESSAGES_PER_TRACE:
            return None
        all_messages.extend(messages)
        tool_calls = _extract_tool_calls(attributes, mapping)
        errors = _extract_errors(span, attributes, mapping)
        tool_call_count += len(tool_calls)
        error_count += len(errors)
        if (
            tool_call_count > _MAXIMUM_TOOL_CALLS_PER_TRACE
            or error_count > _MAXIMUM_ERRORS_PER_TRACE
        ):
            return None
        canonical_span: dict[str, JsonValue] = {
            "span_id": _normalize_id_field(span.get("spanId")) or "unknown",
            "parent_span_id": _normalize_id_field(span.get("parentSpanId")),
            "name": _bounded_text(_nonempty_string(span.get("name")) or "unnamed", 512),
            "start_time_unix_nano": _parse_nano(span.get("startTimeUnixNano")),
            "end_time_unix_nano": _parse_nano(span.get("endTimeUnixNano")),
            "operation": _mapped_metadata(attributes, ("gen_ai.operation.name",)),
            "openinference_kind": _mapped_metadata(attributes, ("openinference.span.kind",)),
            "scope": _compact_object(
                {"name": exported_span.scope_name, "version": exported_span.scope_version}
            ),
            "status": _extract_status(span),
            "messages": cast(JsonValue, messages),
            "tool_calls": cast(JsonValue, tool_calls),
            "errors": cast(JsonValue, errors),
            "retry_attempt": _mapped_scalar(attributes, mapping.attributes.retry_attempt),
        }
        if mapping.include_raw_content:
            canonical_span["state_snapshot"] = _mapped_span_json(
                span,
                attributes,
                mapping.attributes.state_snapshot,
                mapping.maximum_content_characters,
            )
            canonical_span["state_delta"] = _mapped_span_json(
                span,
                attributes,
                mapping.attributes.state_delta,
                mapping.maximum_content_characters,
            )
        canonical_spans.append(_compact_object(canonical_span))

    input_text = _first_message_text(all_messages, "user")
    if input_text is None:
        return None
    session_id = _bounded_metadata_value(
        _first_mapped_value(ordered_spans, mapping.attributes.session_id, include_resource=True)
    )
    agent_name = _bounded_metadata_value(
        _first_mapped_value(ordered_spans, mapping.attributes.agent_name, include_resource=True)
    )
    agent_version = _bounded_metadata_value(
        _first_mapped_value(ordered_spans, mapping.attributes.agent_version, include_resource=True)
    )
    source_reference = _bounded_metadata_value(
        _first_mapped_value(
            ordered_spans, mapping.attributes.source_reference, include_resource=True
        )
    )
    output: dict[str, JsonValue] = {
        "schema_version": "1.0.0",
        "kind": "ul.trace_scenario",
        "trace_id": trace_id,
        "session_id": session_id,
        "agent": _compact_object({"name": agent_name, "version": agent_version}),
        "source": _compact_object(
            {
                "reference": source_reference,
                "resource": _compact_object(
                    {
                        "service_name": resource_attributes.get("service.name"),
                        "service_version": resource_attributes.get("service.version"),
                    }
                ),
            }
        ),
        "messages": cast(JsonValue, all_messages),
        "spans": cast(JsonValue, canonical_spans),
    }
    return input_text, output


def _extract_span_messages(
    span: dict[str, Any],
    attributes: dict[str, JsonValue],
    mapping: OtlpMappingConfig,
) -> list[dict[str, JsonValue]]:
    messages = _extract_mapped_messages(attributes, mapping)
    for event in _ordered_span_events(span):
        messages.extend(_extract_mapped_messages(_span_attributes(event), mapping))
    unique_messages: list[dict[str, JsonValue]] = []
    known_messages: set[str] = set()
    for message in messages:
        identity = json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if identity not in known_messages:
            known_messages.add(identity)
            unique_messages.append(message)
    return unique_messages


def _extract_mapped_messages(
    attributes: dict[str, JsonValue], mapping: OtlpMappingConfig
) -> list[dict[str, JsonValue]]:
    messages: list[dict[str, JsonValue]] = []
    for direction, keys in (
        ("input", mapping.attributes.input_messages),
        ("output", mapping.attributes.output_messages),
    ):
        value = _mapped_json(attributes, keys, mapping.maximum_content_characters)
        if isinstance(value, list):
            for message in cast(list[JsonValue], value):
                normalized = _normalize_structured_message(message, direction, mapping)
                if normalized is not None:
                    messages.append(normalized)
        messages.extend(_extract_openinference_messages(attributes, keys, direction, mapping))

    if not messages:
        prompt = _mapped_scalar(attributes, mapping.attributes.prompt)
        completion = _mapped_scalar(attributes, mapping.attributes.completion)
        if mapping.include_raw_content and isinstance(prompt, str) and prompt.strip():
            messages.append({"role": "user", "direction": "input", "content": prompt})
        if mapping.include_raw_content and isinstance(completion, str) and completion.strip():
            messages.append({"role": "assistant", "direction": "output", "content": completion})
    return messages


def _normalize_structured_message(
    value: JsonValue,
    direction: str,
    mapping: OtlpMappingConfig,
) -> dict[str, JsonValue] | None:
    if not isinstance(value, dict):
        return None
    role = value.get("role")
    if not isinstance(role, str) or not role:
        return None
    result: dict[str, JsonValue] = {
        "role": _bounded_text(role, _MAXIMUM_METADATA_CHARACTERS),
        "direction": direction,
    }
    parts = value.get("parts")
    if isinstance(parts, list):
        normalized_parts: list[JsonValue] = []
        for part in cast(list[JsonValue], parts):
            if not isinstance(part, dict):
                continue
            normalized: dict[str, JsonValue] = {}
            for key in ("type", "id", "name"):
                item = part.get(key)
                if isinstance(item, str):
                    normalized[key] = _bounded_text(item, 512)
            if mapping.include_raw_content:
                for key in ("content", "arguments", "result", "response"):
                    if key in part:
                        normalized[key] = _bounded_json(
                            part[key], mapping.maximum_content_characters
                        )
            if normalized:
                normalized_parts.append(normalized)
        result["parts"] = normalized_parts
        text_parts = [
            part.get("content")
            for part in normalized_parts
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("content"), str)
        ]
        if text_parts:
            result["content"] = "\n".join(cast(list[str], text_parts))
    content = value.get("content")
    if mapping.include_raw_content and isinstance(content, str):
        result["content"] = _bounded_text(content, mapping.maximum_content_characters)
    return result


def _extract_openinference_messages(
    attributes: dict[str, JsonValue],
    configured_keys: tuple[str, ...],
    direction: str,
    mapping: OtlpMappingConfig,
) -> list[dict[str, JsonValue]]:
    prefixes = [key for key in configured_keys if key.startswith("llm.")]
    indexed: dict[int, dict[str, JsonValue]] = {}
    for key, value in attributes.items():
        for prefix in prefixes:
            match = re.fullmatch(rf"{re.escape(prefix)}\.(\d+)\.message\.(role|content)", key)
            if match is None:
                continue
            index = int(match.group(1))
            field = match.group(2)
            if field == "role" and isinstance(value, str):
                indexed.setdefault(index, {})[field] = _bounded_text(
                    value, _MAXIMUM_METADATA_CHARACTERS
                )
            elif field == "content" and mapping.include_raw_content and isinstance(value, str):
                indexed.setdefault(index, {})[field] = _bounded_text(
                    value, mapping.maximum_content_characters
                )
    return [
        {"direction": direction, **indexed[index]}
        for index in sorted(indexed)
        if isinstance(indexed[index].get("role"), str)
    ]


def _extract_tool_calls(
    attributes: dict[str, JsonValue], mapping: OtlpMappingConfig
) -> list[dict[str, JsonValue]]:
    calls: list[dict[str, JsonValue]] = []
    direct = _compact_object(
        {
            "id": _mapped_metadata(attributes, mapping.attributes.tool_call_id),
            "name": _mapped_metadata(attributes, mapping.attributes.tool_name),
        }
    )
    if mapping.include_raw_content:
        direct["arguments"] = _mapped_json(
            attributes, mapping.attributes.tool_arguments, mapping.maximum_content_characters
        )
        direct["result"] = _mapped_json(
            attributes, mapping.attributes.tool_result, mapping.maximum_content_characters
        )
        direct = _compact_object(direct)
    if direct:
        calls.append(direct)
    pattern = re.compile(
        r"llm\.output_messages\.(\d+)\.message\.tool_calls\.(\d+)\.tool_call\."
        r"(id|function\.name|function\.arguments)"
    )
    indexed: dict[tuple[int, int], dict[str, JsonValue]] = {}
    for key, value in attributes.items():
        match = pattern.fullmatch(key)
        if match is None:
            continue
        item = indexed.setdefault((int(match.group(1)), int(match.group(2))), {})
        field = match.group(3)
        if field == "id" and isinstance(value, str):
            item["id"] = _bounded_text(value, _MAXIMUM_METADATA_CHARACTERS)
        elif field == "function.name" and isinstance(value, str):
            item["name"] = _bounded_text(value, _MAXIMUM_METADATA_CHARACTERS)
        elif field == "function.arguments" and mapping.include_raw_content:
            item["arguments"] = _bounded_json(value, mapping.maximum_content_characters)
    calls.extend(indexed[index] for index in sorted(indexed))
    return calls


def _extract_errors(
    span: dict[str, Any],
    attributes: dict[str, JsonValue],
    mapping: OtlpMappingConfig,
) -> list[dict[str, JsonValue]]:
    errors: list[dict[str, JsonValue]] = []
    error_type = _mapped_metadata(attributes, mapping.attributes.error_type)
    error: dict[str, JsonValue] = {}
    if isinstance(error_type, str):
        error["type"] = error_type
    if mapping.include_raw_content:
        message = _mapped_scalar(attributes, mapping.attributes.error_message)
        if isinstance(message, str):
            error["message"] = _bounded_text(message, mapping.maximum_content_characters)
    if error:
        errors.append(error)
    for typed_event in _ordered_span_events(span):
        if typed_event.get("name") != "exception":
            continue
        event_attributes = _span_attributes(typed_event)
        item: dict[str, JsonValue] = {}
        event_type = _mapped_metadata(event_attributes, mapping.attributes.error_type)
        if isinstance(event_type, str):
            item["type"] = event_type
        if mapping.include_raw_content:
            message = _mapped_scalar(event_attributes, mapping.attributes.error_message)
            if isinstance(message, str):
                item["message"] = _bounded_text(message, mapping.maximum_content_characters)
        if item:
            errors.append(item)
    return errors


def _ordered_span_events(span: dict[str, Any]) -> list[dict[str, Any]]:
    events = [
        cast(dict[str, Any], event)
        for event in cast(list[Any], span.get("events") or [])
        if isinstance(event, dict)
    ]
    return sorted(
        events,
        key=lambda event: (
            _parse_nano(event.get("timeUnixNano")),
            _nonempty_string(event.get("name")) or "",
        ),
    )


def _extract_status(span: dict[str, Any]) -> dict[str, JsonValue]:
    status = span.get("status")
    if not isinstance(status, dict):
        return {}
    typed_status = cast(dict[str, Any], status)
    code = typed_status.get("code")
    result: dict[str, JsonValue] = {}
    if isinstance(code, int):
        result["code"] = code
    return result


def _first_message_text(messages: list[dict[str, JsonValue]], role: str) -> str | None:
    for message in messages:
        if message.get("role") == role and isinstance(message.get("content"), str):
            content = cast(str, message["content"])
            if content.strip():
                return content
    return None


def _first_mapped_value(
    spans: list[_ExportedSpan], keys: tuple[str, ...], *, include_resource: bool
) -> JsonValue:
    for item in spans:
        attributes = _span_attributes(item.span)
        if include_resource:
            attributes = {**item.resource_attributes, **attributes}
        value = _mapped_scalar(attributes, keys)
        if value is not None:
            return value
    return None


def _mapped_scalar(attributes: dict[str, JsonValue], keys: tuple[str, ...]) -> JsonValue:
    for key in keys:
        value = attributes.get(key)
        if isinstance(value, (str, int, float, bool)):
            return value
    return None


def _mapped_metadata(attributes: dict[str, JsonValue], keys: tuple[str, ...]) -> JsonValue:
    return _bounded_metadata_value(_mapped_scalar(attributes, keys))


def _bounded_metadata_value(value: JsonValue) -> JsonValue:
    if isinstance(value, str):
        return _bounded_text(value, _MAXIMUM_METADATA_CHARACTERS)
    return value


def _mapped_json(
    attributes: dict[str, JsonValue], keys: tuple[str, ...], maximum_characters: int
) -> JsonValue:
    for key in keys:
        if key not in attributes:
            continue
        value = attributes[key]
        if isinstance(value, str):
            if len(value) > maximum_characters:
                return _bounded_text(value, maximum_characters)
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, RecursionError):
                return value
            return _bounded_json(cast(JsonValue, parsed), maximum_characters)
        return _bounded_json(value, maximum_characters)
    return None


def _mapped_span_json(
    span: dict[str, Any],
    attributes: dict[str, JsonValue],
    keys: tuple[str, ...],
    maximum_characters: int,
) -> JsonValue:
    value = _mapped_json(attributes, keys, maximum_characters)
    if value is not None:
        return value
    for event in _ordered_span_events(span):
        value = _mapped_json(_span_attributes(event), keys, maximum_characters)
        if value is not None:
            return value
    return None


def _bounded_json(value: JsonValue, maximum_characters: int) -> JsonValue:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) <= maximum_characters:
        return value
    if isinstance(value, str):
        return _bounded_text(value, maximum_characters)
    return {"truncated": True, "original_characters": len(encoded)}


def _bounded_text(value: str, maximum_characters: int) -> str:
    if len(value) <= maximum_characters:
        return value
    return value[:maximum_characters] + "…"


def _compact_object(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: item for key, item in value.items() if item not in (None, {}, [])}


def _pick_root_span(gen_ai_spans: list[_ExportedSpan]) -> _ExportedSpan:
    span_ids = {_normalize_id_field(item.span.get("spanId")) for item in gen_ai_spans} - {None}
    roots = [
        item
        for item in gen_ai_spans
        if _normalize_id_field(item.span.get("parentSpanId")) not in span_ids
    ]
    candidates = roots if roots else gen_ai_spans
    return sorted(
        candidates,
        key=lambda item: (
            _parse_nano(item.span.get("startTimeUnixNano")),
            _normalize_id_field(item.span.get("spanId")) or "",
        ),
    )[0]


def _is_gen_ai_span(span: dict[str, Any]) -> bool:
    attributes = _span_attributes(span)
    return any(
        key in attributes
        for key in (
            "gen_ai.operation.name",
            "gen_ai.system",
            "gen_ai.provider.name",
            "gen_ai.prompt",
            "openinference.span.kind",
        )
    )


def _extract_legacy_input(span: dict[str, Any]) -> str | None:
    return _extract_legacy_content(span, "gen_ai.content.prompt", "gen_ai.prompt")


def _extract_legacy_output(span: dict[str, Any]) -> str | None:
    return _extract_legacy_content(span, "gen_ai.content.completion", "gen_ai.completion")


def _extract_legacy_content(span: dict[str, Any], event_name: str, key: str) -> str | None:
    for event in cast(list[Any], span.get("events") or []):
        if not isinstance(event, dict):
            continue
        typed_event = cast(dict[str, Any], event)
        if typed_event.get("name") == event_name:
            value = _span_attributes(typed_event).get(key)
            if isinstance(value, str) and value.strip():
                return value
    value = _span_attributes(span).get(key)
    return value if isinstance(value, str) and value.strip() else None


def _span_attributes(span: dict[str, Any]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for attr in cast(list[Any], span.get("attributes") or []):
        if not isinstance(attr, dict):
            continue
        typed_attr = cast(dict[str, Any], attr)
        key = typed_attr.get("key")
        if not isinstance(key, str):
            continue
        value = _unwrap_otlp_value(typed_attr.get("value"))
        if value is not None:
            result[key] = value
    return result


def _unwrap_otlp_value(value: object, *, depth: int = 0) -> JsonValue:
    if depth > 20 or not isinstance(value, dict):
        return None
    typed = cast(dict[str, Any], value)
    if "stringValue" in typed and isinstance(typed["stringValue"], str):
        return typed["stringValue"]
    if "intValue" in typed:
        raw = typed["intValue"]
        if isinstance(raw, str):
            try:
                return int(raw)
            except ValueError:
                return None
        return raw if isinstance(raw, int) and not isinstance(raw, bool) else None
    if "doubleValue" in typed:
        raw = typed["doubleValue"]
        return raw if isinstance(raw, (int, float)) and math.isfinite(float(raw)) else None
    if "boolValue" in typed and isinstance(typed["boolValue"], bool):
        return typed["boolValue"]
    if "arrayValue" in typed and isinstance(typed["arrayValue"], dict):
        values = cast(dict[str, Any], typed["arrayValue"]).get("values")
        if isinstance(values, list):
            result_values: list[JsonValue] = []
            for item in cast(list[Any], values):
                result_values.append(_unwrap_otlp_value(item, depth=depth + 1))
            return result_values
    if "kvlistValue" in typed and isinstance(typed["kvlistValue"], dict):
        values = cast(dict[str, Any], typed["kvlistValue"]).get("values")
        if isinstance(values, list):
            result: dict[str, JsonValue] = {}
            for item in cast(list[Any], values):
                if not isinstance(item, dict):
                    continue
                typed_item = cast(dict[str, Any], item)
                if not isinstance(typed_item.get("key"), str):
                    continue
                result[cast(str, typed_item["key"])] = _unwrap_otlp_value(
                    typed_item.get("value"), depth=depth + 1
                )
            return result
    return None


def _normalize_id_field(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if all(character in "0123456789abcdefABCDEF" for character in value):
        normalized = value.lower()
        return None if all(character == "0" for character in normalized) else normalized
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, TypeError):
        return None
    normalized = decoded.hex()
    if not normalized or all(character == "0" for character in normalized):
        return None
    return normalized


def _parse_nano(value: object) -> int:
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0


def _nonempty_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
