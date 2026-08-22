from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import math
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from ul_core.evaluation import (
    ObservationRequest,
    ObservationSourceCapabilities,
    ProbeObservation,
)

_TRANSLATOR_VERSION = "1.0.0"
_HEX_IDENTIFIER = re.compile(r"^[0-9a-fA-F]+$")


class OtlpObservationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source_id: str = Field(default="ul-otlp", min_length=1, max_length=500)
    settle_window_seconds: float = Field(default=0.1, gt=0, le=30)
    observation_timeout_seconds: float = Field(default=1.0, gt=0, le=120)
    poll_interval_seconds: float = Field(default=0.01, gt=0, le=1)
    maximum_spans: int = Field(default=10_000, ge=1, le=1_000_000)
    maximum_spans_per_observation: int = Field(default=1_000, ge=1, le=10_000)
    maximum_payload_bytes: int = Field(default=10_000_000, ge=1, le=100_000_000)
    maximum_buffer_bytes: int = Field(default=50_000_000, ge=1, le=500_000_000)
    maximum_observation_bytes: int = Field(default=10_000_000, ge=10_000, le=100_000_000)
    retention_seconds: float = Field(default=300, gt=0, le=86_400)
    retain_raw_spans: bool = False
    redacted_attribute_terms: tuple[str, ...] = (
        "authorization",
        "api_key",
        "apikey",
        "password",
        "secret",
        "access_token",
        "refresh_token",
        "auth.token",
    )

    @model_validator(mode="after")
    def validate_windows(self) -> OtlpObservationConfig:
        finite_values = (
            self.settle_window_seconds,
            self.observation_timeout_seconds,
            self.poll_interval_seconds,
            self.retention_seconds,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("OTLP observation durations must be finite")
        if self.settle_window_seconds > self.observation_timeout_seconds:
            raise ValueError("settle window cannot exceed the observation timeout")
        if any(not term.strip() for term in self.redacted_attribute_terms):
            raise ValueError("redacted attribute terms must not be empty")
        return self


class OtlpExportReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    accepted_spans: int = Field(ge=0)
    rejected_spans: int = Field(ge=0)
    partial_success: bool


@dataclass(frozen=True)
class _BufferedSpan:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    correlation_id: str | None
    received_at: float
    size_bytes: int
    raw: dict[str, JsonValue]


class OtlpObservationSource:
    def __init__(self, config: OtlpObservationConfig | None = None) -> None:
        self.config = config or OtlpObservationConfig()
        self.capabilities = ObservationSourceCapabilities(
            source_id=self.config.source_id,
            authority="independent_observer",
            observation_size_limit_bytes=self.config.maximum_observation_bytes,
            supports_traces=True,
            supports_tool_calls=True,
            supports_handoffs=True,
            supports_errors=True,
            supports_usage=True,
            supports_metadata=True,
            counts_toward_environment_api_calls=False,
        )
        self._lock = threading.Lock()
        self._spans: list[_BufferedSpan] = []
        self._rejected_correlations: dict[str, float] = {}
        self._rejected_trace_ids: dict[str, float] = {}

    def export(self, payload: object) -> OtlpExportReceipt:
        try:
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError, RecursionError):
            return OtlpExportReceipt(
                accepted_spans=0,
                rejected_spans=1,
                partial_success=True,
            )
        if len(encoded) > self.config.maximum_payload_bytes:
            return OtlpExportReceipt(
                accepted_spans=0,
                rejected_spans=1,
                partial_success=True,
            )
        exported_spans = _exported_spans(payload)
        now = time.monotonic()
        accepted = 0
        rejected = 0
        with self._lock:
            self._prune(now)
            buffered_bytes = sum(span.size_bytes for span in self._spans)
            for raw_span in exported_spans:
                span_identity = _span_identity(raw_span)
                existing_index = (
                    next(
                        (
                            index
                            for index, existing_span in enumerate(self._spans)
                            if (existing_span.trace_id, existing_span.span_id) == span_identity[:2]
                        ),
                        None,
                    )
                    if span_identity is not None
                    else None
                )
                if existing_index is None and len(self._spans) >= self.config.maximum_spans:
                    rejected += 1
                    if span_identity is not None:
                        self._record_rejection(span_identity[0], span_identity[2], now)
                    continue
                protected_span = cast(
                    dict[str, JsonValue],
                    _redact(raw_span, self.config.redacted_attribute_terms),
                )
                buffered_span = _buffered_span(protected_span, now)
                if buffered_span is None:
                    rejected += 1
                    continue
                existing_size = (
                    self._spans[existing_index].size_bytes if existing_index is not None else 0
                )
                if (
                    buffered_bytes - existing_size + buffered_span.size_bytes
                    > self.config.maximum_buffer_bytes
                    or buffered_span.size_bytes * 3 > self.config.maximum_observation_bytes
                ):
                    rejected += 1
                    self._record_rejection(
                        buffered_span.trace_id, buffered_span.correlation_id, now
                    )
                    continue
                if existing_index is None:
                    self._spans.append(buffered_span)
                else:
                    self._spans[existing_index] = buffered_span
                buffered_bytes += buffered_span.size_bytes - existing_size
                accepted += 1
        return OtlpExportReceipt(
            accepted_spans=accepted,
            rejected_spans=rejected,
            partial_success=rejected > 0,
        )

    async def observe(self, request: ObservationRequest) -> ProbeObservation:
        deadline = time.monotonic() + self.config.observation_timeout_seconds
        matched: tuple[_BufferedSpan, ...] = ()
        while True:
            now = time.monotonic()
            with self._lock:
                self._prune(now)
                matched = self._matching_spans(request)
            if (
                matched
                and now - max(span.received_at for span in matched)
                >= self.config.settle_window_seconds
            ):
                break
            if now >= deadline:
                break
            await asyncio.sleep(min(self.config.poll_interval_seconds, deadline - now))
        if not matched:
            return ProbeObservation(
                id=f"otlp:{request.correlation_id}:missing",
                source_id=self.config.source_id,
                correlation_id=request.correlation_id,
                authority="independent_observer",
                status="missing",
                limitation="no correlated OTLP spans arrived before the observation deadline",
            )
        selected = _bounded_selection(matched, self.config)
        truncated = len(selected) < len(matched)
        trajectory = _normalize_trajectory(request, selected, self.config)
        raw_expected_parent_id = request.context.get("span_id")
        expected_parent_id = (
            raw_expected_parent_id if isinstance(raw_expected_parent_id, str) else None
        )
        root_present = any(span.parent_span_id in {None, expected_parent_id} for span in selected)
        ended = all(_span_ended(span.raw) for span in selected)
        with self._lock:
            rejected = request.correlation_id in self._rejected_correlations or any(
                span.trace_id in self._rejected_trace_ids for span in selected
            )
            self._spans = [span for span in self._spans if span not in matched]
            self._rejected_correlations.pop(request.correlation_id, None)
            for span in selected:
                self._rejected_trace_ids.pop(span.trace_id, None)
        complete = root_present and ended and not truncated and not rejected
        limitation_parts: list[str] = []
        if not root_present:
            limitation_parts.append("root span was not observed")
        if not ended:
            limitation_parts.append("one or more spans were unfinished")
        if truncated or rejected:
            limitation_parts.append("the bounded receiver dropped spans")
        normalized = trajectory["normalized"]
        if not isinstance(normalized, dict) or not isinstance(normalized.get("spans"), list):
            raise AssertionError("normalized OTLP trajectory must contain spans")
        normalized_spans = cast(list[dict[str, JsonValue]], normalized["spans"])
        tool_calls = tuple(span for span in normalized_spans if span["kind"] == "tool")
        handoffs = tuple(span for span in normalized_spans if span["kind"] == "agent_handoff")
        errors = tuple(span for span in normalized_spans if span["status"] == "error")
        return ProbeObservation(
            id=f"otlp:{request.correlation_id}",
            source_id=self.config.source_id,
            correlation_id=request.correlation_id,
            authority="independent_observer",
            status="complete" if complete else "incomplete",
            limitation=None if complete else "; ".join(limitation_parts),
            traces=(trajectory,),
            tool_calls=tool_calls,
            handoffs=handoffs,
            errors=errors,
            usage=_usage(normalized_spans),
            metadata={
                "translator": "ul.otlp-openinference",
                "translator_version": _TRANSLATOR_VERSION,
                "raw_spans_retained": self.config.retain_raw_spans,
                "redaction_policy_sha256": _redaction_digest(self.config),
                "span_count": len(selected),
                **{key: value for key, value in request.context.items() if key.startswith("ul.")},
            },
        )

    def _matching_spans(self, request: ObservationRequest) -> tuple[_BufferedSpan, ...]:
        trace_id = request.context.get("trace_id")
        if isinstance(trace_id, str):
            return tuple(span for span in self._spans if span.trace_id == trace_id)
        return tuple(span for span in self._spans if span.correlation_id == request.correlation_id)

    def _record_rejection(
        self, trace_id: str, correlation_id: str | None, received_at: float
    ) -> None:
        self._remember_rejection(self._rejected_trace_ids, trace_id, received_at)
        if correlation_id is not None:
            self._remember_rejection(self._rejected_correlations, correlation_id, received_at)

    def _remember_rejection(
        self, markers: dict[str, float], identifier: str, received_at: float
    ) -> None:
        markers[identifier] = received_at
        while len(markers) > self.config.maximum_spans:
            oldest_identifier = min(markers, key=lambda key: (markers[key], key))
            del markers[oldest_identifier]

    def _prune(self, now: float) -> None:
        oldest = now - self.config.retention_seconds
        self._spans = [span for span in self._spans if span.received_at >= oldest]
        self._rejected_correlations = {
            correlation_id: rejected_at
            for correlation_id, rejected_at in self._rejected_correlations.items()
            if rejected_at >= oldest
        }
        self._rejected_trace_ids = {
            trace_id: rejected_at
            for trace_id, rejected_at in self._rejected_trace_ids.items()
            if rejected_at >= oldest
        }


def _exported_spans(payload: object) -> Iterator[dict[str, JsonValue]]:
    if not isinstance(payload, dict):
        return
    payload_object = cast(dict[str, object], payload)
    resource_spans = payload_object.get("resourceSpans")
    if not isinstance(resource_spans, list):
        return
    for raw_resource_entry in cast(list[object], resource_spans):
        resource_entry = (
            cast(dict[str, object], raw_resource_entry)
            if isinstance(raw_resource_entry, dict)
            else None
        )
        if resource_entry is None:
            continue
        resource = resource_entry.get("resource")
        resource_value = cast(dict[str, JsonValue], resource) if isinstance(resource, dict) else {}
        scope_spans = resource_entry.get("scopeSpans")
        if not isinstance(scope_spans, list):
            scope_spans = resource_entry.get("instrumentationLibrarySpans")
        if not isinstance(scope_spans, list):
            continue
        for raw_scope_entry in cast(list[object], scope_spans):
            if not isinstance(raw_scope_entry, dict):
                continue
            scope_entry = cast(dict[str, object], raw_scope_entry)
            scope = scope_entry.get("scope") or scope_entry.get("instrumentationLibrary")
            scope_value = cast(dict[str, JsonValue], scope) if isinstance(scope, dict) else {}
            spans = scope_entry.get("spans")
            if not isinstance(spans, list):
                continue
            for raw_span in cast(list[object], spans):
                if isinstance(raw_span, dict):
                    span = cast(dict[str, JsonValue], raw_span)
                    yield {
                        "resource": resource_value,
                        "scope": scope_value,
                        "span": span,
                    }


def _span_identity(raw: dict[str, JsonValue]) -> tuple[str, str, str | None] | None:
    span = raw.get("span")
    if not isinstance(span, dict):
        return None
    trace_id = _identifier(span.get("traceId"), 16)
    span_id = _identifier(span.get("spanId"), 8)
    if trace_id is None or span_id is None:
        return None
    attributes = _attributes(span.get("attributes"))
    resource = raw.get("resource")
    if isinstance(resource, dict):
        attributes = {**_attributes(resource.get("attributes")), **attributes}
    correlation_id = attributes.get("ul.correlation.id")
    return trace_id, span_id, correlation_id if isinstance(correlation_id, str) else None


def _buffered_span(raw: dict[str, JsonValue], received_at: float) -> _BufferedSpan | None:
    span = raw.get("span")
    if not isinstance(span, dict):
        return None
    trace_id = _identifier(span.get("traceId"), 16)
    span_id = _identifier(span.get("spanId"), 8)
    if trace_id is None or span_id is None:
        return None
    parent_span_id = _identifier(span.get("parentSpanId"), 8)
    attributes = _attributes(span.get("attributes"))
    resource = raw.get("resource")
    if isinstance(resource, dict):
        attributes = {**_attributes(resource.get("attributes")), **attributes}
    correlation_id = attributes.get("ul.correlation.id")
    return _BufferedSpan(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        correlation_id=correlation_id if isinstance(correlation_id, str) else None,
        received_at=received_at,
        size_bytes=len(json.dumps(raw, ensure_ascii=False, separators=(",", ":")).encode("utf-8")),
        raw=raw,
    )


def _bounded_selection(
    spans: tuple[_BufferedSpan, ...],
    config: OtlpObservationConfig,
) -> tuple[_BufferedSpan, ...]:
    selected: list[_BufferedSpan] = []
    raw_bytes = 0
    raw_budget = config.maximum_observation_bytes // 3
    for span in spans[: config.maximum_spans_per_observation]:
        if raw_bytes + span.size_bytes > raw_budget:
            break
        selected.append(span)
        raw_bytes += span.size_bytes
    return tuple(selected)


def _normalize_trajectory(
    request: ObservationRequest,
    spans: tuple[_BufferedSpan, ...],
    config: OtlpObservationConfig,
) -> dict[str, JsonValue]:
    normalized_spans = [_normalize_span(span) for span in spans]
    normalized_spans.sort(key=_span_start_sort_key)
    raw_expected_parent_id = request.context.get("span_id")
    expected_parent_id = raw_expected_parent_id if isinstance(raw_expected_parent_id, str) else None
    root = next(
        (span for span in normalized_spans if span["parent_span_id"] in {None, expected_parent_id}),
        None,
    )
    standards = sorted({str(span["standard"]) for span in normalized_spans})
    return cast(
        dict[str, JsonValue],
        {
            "schema_version": "1.0.0",
            "trace_id": spans[0].trace_id,
            "root_span_id": root["span_id"] if root is not None else None,
            "provenance": {
                "source_format": "otlp-json",
                "semantic_conventions": standards,
                "translator": "ul.otlp-openinference",
                "translator_version": _TRANSLATOR_VERSION,
            },
            "normalized": {
                "schema_version": "1.0.0",
                "spans": normalized_spans,
            },
            "raw_spans": [span.raw for span in spans] if config.retain_raw_spans else [],
        },
    )


def _normalize_span(buffered: _BufferedSpan) -> dict[str, JsonValue]:
    raw_span = cast(dict[str, JsonValue], buffered.raw["span"])
    resource = cast(dict[str, JsonValue], buffered.raw["resource"])
    attributes = _attributes(resource.get("attributes"))
    attributes.update(_attributes(raw_span.get("attributes")))
    openinference_kind = attributes.get("openinference.span.kind")
    operation = attributes.get("gen_ai.operation.name")
    kind, standard = _span_kind(openinference_kind, operation)
    status_value = raw_span.get("status")
    status_code = status_value.get("code") if isinstance(status_value, dict) else None
    error = status_code in {2, "STATUS_CODE_ERROR", "ERROR"} or _has_exception_event(
        raw_span.get("events")
    )
    return {
        "trace_id": buffered.trace_id,
        "span_id": buffered.span_id,
        "parent_span_id": buffered.parent_span_id,
        "name": str(raw_span.get("name", "unnamed")),
        "kind": kind,
        "standard": standard,
        "status": "error" if error else "ok",
        "start_time_unix_nano": raw_span.get("startTimeUnixNano"),
        "end_time_unix_nano": raw_span.get("endTimeUnixNano"),
        "attributes": attributes,
        "events": raw_span.get("events", []),
    }


def _span_start_sort_key(span: dict[str, JsonValue]) -> tuple[int, int, str, str]:
    value = span.get("start_time_unix_nano")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return 0, value, "", str(span["span_id"])
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return 0, int(value), "", str(span["span_id"])
    return 1, 0, json.dumps(value, ensure_ascii=True, sort_keys=True), str(span["span_id"])


def _span_kind(
    openinference_kind: JsonValue | None,
    operation: JsonValue | None,
) -> tuple[str, str]:
    if isinstance(openinference_kind, str):
        normalized = openinference_kind.casefold()
        if normalized in {"agent", "chain"}:
            return "agent", "openinference"
        if normalized == "tool":
            return "tool", "openinference"
        if normalized == "llm":
            return "llm", "openinference"
        if normalized == "guardrail":
            return "guardrail", "openinference"
        if normalized == "evaluator":
            return "evaluator", "openinference"
        if normalized == "retriever":
            return "retriever", "openinference"
        if normalized == "embedding":
            return "embedding", "openinference"
        if normalized == "reranker":
            return "reranker", "openinference"
        return "span", "openinference"
    if isinstance(operation, str):
        normalized = operation.casefold()
        if normalized in {"invoke_agent", "create_agent"}:
            return "agent", "opentelemetry-genai"
        if normalized in {"execute_tool", "call_tool"}:
            return "tool", "opentelemetry-genai"
        if normalized in {"chat", "text_completion", "generate_content"}:
            return "llm", "opentelemetry-genai"
    return "span", "opentelemetry"


def _attributes(value: object) -> dict[str, JsonValue]:
    if not isinstance(value, list):
        return {}
    result: dict[str, JsonValue] = {}
    for raw_attribute in cast(list[object], value):
        attribute = (
            cast(dict[str, object], raw_attribute) if isinstance(raw_attribute, dict) else None
        )
        if attribute is None:
            continue
        if not isinstance(attribute.get("key"), str):
            continue
        result[cast(str, attribute["key"])] = _attribute_value(attribute.get("value"))
    return result


def _attribute_value(value: object) -> JsonValue:
    if not isinstance(value, dict):
        return cast(JsonValue, value)
    value_object = cast(dict[str, object], value)
    for key in ("stringValue", "boolValue", "doubleValue", "bytesValue"):
        if key in value_object:
            return cast(JsonValue, value_object[key])
    if "intValue" in value_object:
        int_value = value_object["intValue"]
        if not isinstance(int_value, str | int | float):
            return str(int_value)
        try:
            return int(int_value)
        except (TypeError, ValueError):
            return str(int_value)
    array_value = value_object.get("arrayValue")
    array_object = cast(dict[str, object], array_value) if isinstance(array_value, dict) else None
    if array_object is not None and isinstance(array_object.get("values"), list):
        return [_attribute_value(item) for item in cast(list[object], array_object["values"])]
    kvlist_value = value_object.get("kvlistValue")
    if isinstance(kvlist_value, dict):
        return _attributes(cast(dict[str, object], kvlist_value).get("values"))
    return None


def _has_exception_event(value: object) -> bool:
    if not isinstance(value, list):
        return False
    return any(
        isinstance(event, dict) and cast(dict[str, object], event).get("name") == "exception"
        for event in cast(list[object], value)
    )


def _identifier(value: object, expected_bytes: int) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if len(value) == expected_bytes * 2 and _HEX_IDENTIFIER.fullmatch(value):
        normalized = value.casefold()
        return None if set(normalized) == {"0"} else normalized
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        return None
    return decoded.hex() if len(decoded) == expected_bytes and any(decoded) else None


def _span_ended(raw: dict[str, JsonValue]) -> bool:
    span = raw.get("span")
    return isinstance(span, dict) and span.get("endTimeUnixNano") not in {None, "", "0", 0}


def _usage(spans: list[dict[str, JsonValue]]) -> JsonValue | None:
    totals: dict[str, int] = {}
    names = {
        "gen_ai.usage.input_tokens": "input_tokens",
        "gen_ai.usage.output_tokens": "output_tokens",
        "llm.token_count.prompt": "input_tokens",
        "llm.token_count.completion": "output_tokens",
    }
    for span in spans:
        attributes = span.get("attributes")
        if not isinstance(attributes, dict):
            continue
        for source_name, target_name in names.items():
            value = attributes.get(source_name)
            if isinstance(value, int) and not isinstance(value, bool):
                totals[target_name] = totals.get(target_name, 0) + value
    return cast(JsonValue, totals) if totals else None


def _redact(value: JsonValue, terms: tuple[str, ...], key: str = "") -> JsonValue:
    if any(term.casefold() in key.casefold() for term in terms):
        return "[REDACTED]"
    if isinstance(value, list):
        redacted: list[JsonValue] = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("key"), str):
                attribute_key = cast(str, item["key"])
                if any(term.casefold() in attribute_key.casefold() for term in terms):
                    redacted.append({**item, "value": {"stringValue": "[REDACTED]"}})
                    continue
            redacted.append(_redact(item, terms))
        return redacted
    if isinstance(value, dict):
        return {item_key: _redact(item, terms, item_key) for item_key, item in value.items()}
    return value


def _redaction_digest(config: OtlpObservationConfig) -> str:
    encoded = json.dumps(
        {
            "terms": config.redacted_attribute_terms,
            "retain_raw_spans": config.retain_raw_spans,
            "retention_seconds": config.retention_seconds,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
