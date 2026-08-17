from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True)
class OtlpInteractionRecord:
    interaction_id: str
    input: str
    output: str


@dataclass(frozen=True)
class OtlpIngestResult:
    records: tuple[OtlpInteractionRecord, ...]
    skipped_no_gen_ai: int
    skipped_no_input: int
    skipped_no_output: int
    truncated: bool


def parse_otlp_traces(data: object, *, limit: int = 100) -> OtlpIngestResult:
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    if not isinstance(data, dict):
        raise ValueError("OTLP export must be a JSON object")

    typed_data = cast(dict[str, Any], data)
    raw_resource_spans = typed_data.get("resourceSpans")
    if not isinstance(raw_resource_spans, list):
        raise ValueError("OTLP export must contain a resourceSpans array")

    spans_by_trace: dict[str, list[dict[str, Any]]] = {}
    for resource_span in cast(list[Any], raw_resource_spans):
        if not isinstance(resource_span, dict):
            continue
        typed_resource_span = cast(dict[str, Any], resource_span)
        for scope_span in cast(list[Any], typed_resource_span.get("scopeSpans") or []):
            if not isinstance(scope_span, dict):
                continue
            typed_scope_span = cast(dict[str, Any], scope_span)
            for span in cast(list[Any], typed_scope_span.get("spans") or []):
                if not isinstance(span, dict):
                    continue
                typed_span = cast(dict[str, Any], span)
                trace_id = _normalize_id_field(typed_span.get("traceId"))
                if trace_id is None:
                    continue
                if trace_id not in spans_by_trace:
                    spans_by_trace[trace_id] = []
                spans_by_trace[trace_id].append(typed_span)

    records: list[OtlpInteractionRecord] = []
    skipped_no_gen_ai = 0
    skipped_no_input = 0
    skipped_no_output = 0
    truncated = False

    for trace_id, spans in spans_by_trace.items():
        if len(records) >= limit:
            truncated = True
            break

        gen_ai_spans = [span for span in spans if _is_gen_ai_span(span)]
        if not gen_ai_spans:
            skipped_no_gen_ai += 1
            continue

        target_span = _pick_root_span(gen_ai_spans)
        input_text = _extract_input(target_span)
        if input_text is None:
            skipped_no_input += 1
            continue

        output_text = _extract_output(target_span)
        if output_text is None:
            skipped_no_output += 1
            continue

        records.append(
            OtlpInteractionRecord(
                interaction_id=trace_id,
                input=input_text,
                output=output_text,
            )
        )

    return OtlpIngestResult(
        records=tuple(records),
        skipped_no_gen_ai=skipped_no_gen_ai,
        skipped_no_input=skipped_no_input,
        skipped_no_output=skipped_no_output,
        truncated=truncated,
    )


def _pick_root_span(gen_ai_spans: list[dict[str, Any]]) -> dict[str, Any]:
    gen_ai_span_ids = {_normalize_id_field(span.get("spanId")) for span in gen_ai_spans} - {None}

    root_spans = [
        span
        for span in gen_ai_spans
        if _normalize_id_field(span.get("parentSpanId")) not in gen_ai_span_ids
    ]

    candidates = root_spans if root_spans else gen_ai_spans
    candidates.sort(key=lambda s: _parse_nano(s.get("startTimeUnixNano")))
    return candidates[0]


def _is_gen_ai_span(span: dict[str, Any]) -> bool:
    attributes = _span_attributes(span)
    return (
        "gen_ai.operation.name" in attributes
        or "gen_ai.system" in attributes
        or "gen_ai.prompt" in attributes
    )


def _extract_input(span: dict[str, Any]) -> str | None:
    for event in cast(list[Any], span.get("events") or []):
        if not isinstance(event, dict):
            continue
        typed_event = cast(dict[str, Any], event)
        if typed_event.get("name") == "gen_ai.content.prompt":
            value = _span_attributes(typed_event).get("gen_ai.prompt")
            if isinstance(value, str) and value.strip():
                return value

    value = _span_attributes(span).get("gen_ai.prompt")
    if isinstance(value, str) and value.strip():
        return value

    return None


def _extract_output(span: dict[str, Any]) -> str | None:
    for event in cast(list[Any], span.get("events") or []):
        if not isinstance(event, dict):
            continue
        typed_event = cast(dict[str, Any], event)
        if typed_event.get("name") == "gen_ai.content.completion":
            value = _span_attributes(typed_event).get("gen_ai.completion")
            if isinstance(value, str) and value.strip():
                return value

    value = _span_attributes(span).get("gen_ai.completion")
    if isinstance(value, str) and value.strip():
        return value

    return None


def _span_attributes(span: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
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


def _unwrap_otlp_value(value: object) -> object:
    if not isinstance(value, dict):
        return None
    typed = cast(dict[str, Any], value)
    if "stringValue" in typed:
        return typed["stringValue"]
    if "intValue" in typed:
        raw = typed["intValue"]
        return int(raw) if isinstance(raw, str) else raw
    if "doubleValue" in typed:
        raw = typed["doubleValue"]
        return raw if isinstance(raw, (int, float)) and math.isfinite(float(raw)) else None
    if "boolValue" in typed:
        return typed["boolValue"]
    return None


def _normalize_id_field(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if all(c in "0123456789abcdefABCDEF" for c in value):
        normalized = value.lower()
        if all(c == "0" for c in normalized):
            return None
        return normalized
    try:
        decoded = base64.b64decode(value)
        normalized = decoded.hex()
        if all(c == "0" for c in normalized):
            return None
        return normalized
    except Exception:
        return None


def _parse_nano(value: object) -> int:
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    if isinstance(value, int):
        return value
    return 0
