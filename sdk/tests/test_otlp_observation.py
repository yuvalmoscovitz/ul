from __future__ import annotations

import asyncio
import socket
import threading
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import cast
from urllib.parse import urlsplit

import httpx
import pytest
from pydantic import JsonValue
from ul import otlp_observation as observation_module
from ul.environment import evaluation_case_from_inputs
from ul.otlp_observation import (
    OtlpJsonHttpReceiver,
    OtlpObservationConfig,
    OtlpObservationSource,
)
from ul.probe_execution import ComposedEnvironmentExecutor
from ul_core.evaluation import (
    ObservationRequest,
    ProbeInvokerCapabilities,
    ProbeRequest,
    ProbeResult,
    ProbeTurn,
)

_CONFIG_SHA256 = "a" * 64
pytestmark = pytest.mark.asyncio


def _attribute(key: str, value: str | int) -> dict[str, JsonValue]:
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    return {"key": key, "value": {"stringValue": value}}


def _payload(
    request: ProbeRequest,
    *,
    include_root: bool = True,
    include_child: bool = True,
    root_ended: bool = True,
) -> dict[str, JsonValue]:
    trace_id = request.context["trace_id"]
    parent_id = request.context["span_id"]
    correlation_id = request.correlation_id
    root: dict[str, JsonValue] = {
        "traceId": trace_id,
        "spanId": "1" * 16,
        "parentSpanId": parent_id,
        "name": "support-agent",
        "startTimeUnixNano": "100",
        "attributes": [
            _attribute("ul.correlation.id", correlation_id),
            _attribute("gen_ai.operation.name", "invoke_agent"),
            _attribute("gen_ai.usage.input_tokens", 4),
            _attribute("authorization", "Bearer private-value"),
        ],
    }
    if root_ended:
        root["endTimeUnixNano"] = "200"
    spans: list[JsonValue] = [root] if include_root else []
    if include_child:
        spans.append(
            cast(
                dict[str, JsonValue],
                {
                    "traceId": trace_id,
                    "spanId": "2" * 16,
                    "parentSpanId": "1" * 16,
                    "name": "lookup-account",
                    "startTimeUnixNano": "120",
                    "endTimeUnixNano": "180",
                    "attributes": [
                        _attribute("openinference.span.kind", "TOOL"),
                        _attribute("tool.name", "lookup_account"),
                    ],
                },
            )
        )
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": [_attribute("service.name", "customer-agent")]},
                "scopeSpans": [
                    {
                        "scope": {"name": "customer.instrumentation", "version": "2.0"},
                        "spans": spans,
                    }
                ],
            }
        ]
    }


def _openinference_family_payload(request: ProbeRequest) -> dict[str, JsonValue]:
    trace_id = request.context["trace_id"]
    parent_id = request.context["span_id"]
    spans: list[JsonValue] = []
    parent_span_id = parent_id
    for index, kind in enumerate(
        (
            "AGENT",
            "CHAIN",
            "LLM",
            "TOOL",
            "GUARDRAIL",
            "EVALUATOR",
            "RETRIEVER",
            "EMBEDDING",
            "RERANKER",
            "UNKNOWN_FUTURE_KIND",
        ),
        start=1,
    ):
        span_id = f"{index:x}" * 16
        attributes = [_attribute("openinference.span.kind", kind)]
        if index == 1:
            attributes.append(_attribute("ul.correlation.id", request.correlation_id))
        spans.append(
            cast(
                dict[str, JsonValue],
                {
                    "traceId": trace_id,
                    "spanId": span_id,
                    "parentSpanId": parent_span_id,
                    "name": kind.casefold(),
                    "startTimeUnixNano": str(index * 100),
                    "endTimeUnixNano": str(index * 100 + 50),
                    "attributes": attributes,
                },
            )
        )
        parent_span_id = span_id
    return {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "scope": {"name": "openinference.instrumentation"},
                        "spans": spans,
                    }
                ]
            }
        ]
    }


@dataclass
class _ExportingInvoker:
    observer: OtlpObservationSource
    capabilities: ProbeInvokerCapabilities = field(
        default_factory=lambda: ProbeInvokerCapabilities(
            invoker_id="local-agent",
            response_size_limit_bytes=1_000,
            supports_conversations=True,
        )
    )
    requests: list[ProbeRequest] = field(default_factory=lambda: [])
    late_tasks: set[asyncio.Task[None]] = field(default_factory=lambda: set())

    async def invoke(self, request: ProbeRequest) -> ProbeResult:
        self.requests.append(request)
        self.observer.export(_payload(request, include_child=False))

        async def export_late_child() -> None:
            await asyncio.sleep(0.02)
            self.observer.export(_payload(request, include_root=False))

        late_task = asyncio.create_task(export_late_child())
        self.late_tasks.add(late_task)
        late_task.add_done_callback(self.late_tasks.discard)
        return ProbeResult(
            id="result-1",
            correlation_id=request.correlation_id,
            response="done",
        )


async def test_active_probe_joins_late_otlp_spans_with_redacted_raw_provenance() -> None:
    observer = OtlpObservationSource(
        OtlpObservationConfig(
            retain_raw_spans=True,
            settle_window_seconds=0.03,
            observation_timeout_seconds=0.2,
            poll_interval_seconds=0.005,
        )
    )
    invoker = _ExportingInvoker(observer)
    executor = ComposedEnvironmentExecutor(
        invoker,
        config_sha256=_CONFIG_SHA256,
        observation_source=observer,
        campaign_id="campaign-1",
        variation_id="punctuation-noise",
        repetition=2,
        observation_timeout_seconds=0.3,
    )
    case = evaluation_case_from_inputs(
        case_id="case-1",
        raw_inputs=("help me",),
        max_environment_api_calls=2,
        timeout_seconds=1,
    )

    evidence = await executor.execute(case)

    request = invoker.requests[0]
    assert executor.api_calls_for_case(case) == 1
    assert request.context["ul.campaign.id"] == "campaign-1"
    assert request.context["ul.case.id"] == "case-1"
    assert request.context["ul.variation.id"] == "punctuation-noise"
    assert request.context["ul.repetition"] == 2
    assert str(request.context["traceparent"]).startswith(f"00-{request.context['trace_id']}-")
    assert "ul.correlation.id=" in str(request.context["baggage"])
    observation = evidence.observations[0]
    identity = evidence.probe_identity
    assert identity is not None
    assert identity.campaign_id == "campaign-1"
    assert identity.case_id == "case-1"
    assert identity.probe_id == request.context["ul.probe.id"]
    assert identity.attempt_id == request.context["ul.attempt.id"]
    assert identity.session_id == request.context["ul.session.id"]
    assert identity.turn_ids == (request.turn.id,)
    assert identity.variation_id == "punctuation-noise"
    assert identity.repetition == 2
    assert observation.status == "complete"
    assert observation.metadata["translator_version"] == "1.0.0"
    assert observation.metadata["span_count"] == 2
    assert len(observation.tool_calls) == 1
    assert observation.usage == {"input_tokens": 4}
    assert evidence.evidence_scope == "response_only"
    assert evidence.final_state is None
    encoded = observation.model_dump_json()
    assert "private-value" not in encoded
    assert "[REDACTED]" in encoded
    trajectory = observation.traces[0]
    assert isinstance(trajectory, dict)
    provenance = trajectory["provenance"]
    assert isinstance(provenance, dict)
    assert provenance["semantic_conventions"] == [
        "openinference",
        "opentelemetry-genai",
    ]
    raw_spans = trajectory["raw_spans"]
    assert isinstance(raw_spans, list)
    assert len(raw_spans) == 2


async def test_otlp_observation_reports_missing_and_incomplete_evidence() -> None:
    observer = OtlpObservationSource(
        OtlpObservationConfig(
            settle_window_seconds=0.01,
            observation_timeout_seconds=0.02,
            poll_interval_seconds=0.002,
        )
    )
    context: dict[str, JsonValue] = {
        "trace_id": "a" * 32,
        "span_id": "b" * 16,
        "ul.campaign.id": "campaign-1",
    }
    request = ObservationRequest(
        case_id="case-1",
        session_id="session-1",
        correlation_id="correlation-1",
        context=context,
    )

    missing = await observer.observe(request)
    assert missing.status == "missing"
    probe_request = ProbeRequest(
        case_id=request.case_id,
        session_id=request.session_id,
        correlation_id=request.correlation_id,
        turn=ProbeTurn(id="turn-1", input="hello"),
        context=context,
    )
    observer.export(_payload(probe_request, include_child=False, root_ended=False))
    incomplete = await observer.observe(request)

    assert incomplete.status == "incomplete"
    assert incomplete.limitation == "one or more spans were unfinished"


async def test_late_span_update_resets_settle_window_without_duplicate_evidence() -> None:
    observer = OtlpObservationSource(
        OtlpObservationConfig(
            settle_window_seconds=0.02,
            observation_timeout_seconds=0.1,
            poll_interval_seconds=0.002,
        )
    )
    request = ProbeRequest(
        case_id="case-1",
        session_id="session-1",
        correlation_id="correlation-1",
        turn=ProbeTurn(id="turn-1", input="hello"),
        context={"trace_id": "a" * 32, "span_id": "b" * 16},
    )
    observer.export(_payload(request, include_child=False, root_ended=False))

    async def finish_root_span() -> None:
        await asyncio.sleep(0.01)
        observer.export(_payload(request, include_child=False))

    finish_task = asyncio.create_task(finish_root_span())
    observation = await observer.observe(
        ObservationRequest(
            case_id=request.case_id,
            session_id=request.session_id,
            correlation_id=request.correlation_id,
            context=request.context,
        )
    )
    await finish_task

    assert observation.status == "complete"
    assert observation.metadata["span_count"] == 1


async def test_same_otlp_export_can_feed_ul_and_an_independent_receiver() -> None:
    first = OtlpObservationSource()
    second = OtlpObservationSource()
    request = ProbeRequest(
        case_id="case-1",
        session_id="session-1",
        correlation_id="correlation-1",
        turn=ProbeTurn(id="turn-1", input="hello"),
        context={"trace_id": "a" * 32, "span_id": "b" * 16},
    )
    payload = _payload(request)

    first_receipt = first.export(payload)
    second_receipt = second.export(payload)

    assert first_receipt == second_receipt
    assert first_receipt.accepted_spans == 2
    observation_request = ObservationRequest(
        case_id=request.case_id,
        session_id=request.session_id,
        correlation_id=request.correlation_id,
        context=request.context,
    )
    first_observation, second_observation = await asyncio.gather(
        first.observe(observation_request),
        second.observe(observation_request),
    )
    assert first_observation.traces == second_observation.traces


async def test_standard_otlp_http_json_stream_reaches_ul_and_parallel_backend() -> None:
    config = OtlpObservationConfig(
        settle_window_seconds=0.01,
        observation_timeout_seconds=0.1,
    )
    first = OtlpObservationSource(config)
    second = OtlpObservationSource(config)
    request = ProbeRequest(
        case_id="case-1",
        session_id="session-1",
        correlation_id="correlation-1",
        turn=ProbeTurn(id="turn-1", input="hello"),
        context={"trace_id": "a" * 32, "span_id": "b" * 16},
    )
    payload = _payload(request)

    with (
        OtlpJsonHttpReceiver(first) as first_receiver,
        OtlpJsonHttpReceiver(second) as second_receiver,
    ):
        async with httpx.AsyncClient() as client:
            first_response, second_response = await asyncio.gather(
                client.post(first_receiver.endpoint, json=payload),
                client.post(second_receiver.endpoint, json=payload),
            )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json()["partialSuccess"]["rejectedSpans"] == 0
    observation_request = ObservationRequest(
        case_id=request.case_id,
        session_id=request.session_id,
        correlation_id=request.correlation_id,
        context=request.context,
    )
    first_observation, second_observation = await asyncio.gather(
        first.observe(observation_request),
        second.observe(observation_request),
    )
    assert first_observation.status == "complete"
    assert first_observation.traces == second_observation.traces


async def test_otlp_http_json_receiver_rejects_oversized_body_before_export() -> None:
    observer = OtlpObservationSource(OtlpObservationConfig(maximum_payload_bytes=100))

    with OtlpJsonHttpReceiver(observer) as receiver:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                receiver.endpoint,
                content=b"x" * 101,
                headers={"Content-Type": "application/json"},
            )

    assert response.status_code == 413
    assert observer._spans == []

    with pytest.raises(ValueError, match="loopback"):
        OtlpJsonHttpReceiver(observer, host="0.0.0.0")


async def test_otlp_http_json_receiver_bounds_concurrent_handlers() -> None:
    observer = OtlpObservationSource()
    entered = threading.Event()
    release = threading.Event()
    active_exports = 0
    maximum_active_exports = 0
    export_lock = threading.Lock()
    original_export = observer.export

    def blocking_export(payload: object) -> object:
        nonlocal active_exports, maximum_active_exports
        with export_lock:
            active_exports += 1
            maximum_active_exports = max(maximum_active_exports, active_exports)
        entered.set()
        release.wait(timeout=2)
        try:
            return original_export(payload)
        finally:
            with export_lock:
                active_exports -= 1

    observer.export = blocking_export  # type: ignore[method-assign]
    payload = _payload(
        ProbeRequest(
            case_id="case-1",
            session_id="session-1",
            correlation_id="correlation-1",
            turn=ProbeTurn(id="turn-1", input="hello"),
            context={"trace_id": "a" * 32, "span_id": "b" * 16},
        )
    )

    with OtlpJsonHttpReceiver(observer, maximum_concurrent_requests=1) as receiver:
        async with httpx.AsyncClient() as client:
            first_request = asyncio.create_task(client.post(receiver.endpoint, json=payload))
            try:
                assert await asyncio.to_thread(entered.wait, 1)
                with pytest.raises(httpx.TransportError):
                    await client.post(receiver.endpoint, json=payload)
                server = receiver._server
                assert server is not None
                assert server.active_request_count == 1
                assert server.rejected_request_count == 1
                assert maximum_active_exports == 1
            finally:
                release.set()
            assert (await first_request).status_code == 200


async def test_otlp_http_json_receiver_times_out_before_headers_arrive() -> None:
    observer = OtlpObservationSource()

    def withhold_headers(endpoint: str) -> bytes:
        parsed = urlsplit(endpoint)
        assert parsed.hostname is not None
        assert parsed.port is not None
        with socket.create_connection((parsed.hostname, parsed.port), timeout=1) as connection:
            connection.settimeout(1)
            return connection.recv(1)

    with OtlpJsonHttpReceiver(observer, request_timeout_seconds=0.05) as receiver:
        assert await asyncio.to_thread(withhold_headers, receiver.endpoint) == b""


class _CountingList[T](list[T]):
    def __init__(self, values: list[T]) -> None:
        super().__init__(values)
        self.iteration_count = 0

    def __iter__(self) -> Iterator[T]:
        self.iteration_count += 1
        return super().__iter__()


class _CountingOrderedDict(OrderedDict[str, float]):
    def __init__(self) -> None:
        super().__init__()
        self.iteration_count = 0

    def __iter__(self) -> Iterator[str]:
        self.iteration_count += 1
        return super().__iter__()


async def test_raw_span_retention_is_disabled_by_default() -> None:
    observer = OtlpObservationSource(
        OtlpObservationConfig(
            settle_window_seconds=0.01,
            observation_timeout_seconds=0.1,
        )
    )
    request = ProbeRequest(
        case_id="case-1",
        session_id="session-1",
        correlation_id="correlation-1",
        turn=ProbeTurn(id="turn-1", input="hello"),
        context={"trace_id": "a" * 32, "span_id": "b" * 16},
    )
    observer.export(_payload(request))

    observation = await observer.observe(
        ObservationRequest(
            case_id=request.case_id,
            session_id=request.session_id,
            correlation_id=request.correlation_id,
            context=request.context,
        )
    )

    trajectory = observation.traces[0]
    assert isinstance(trajectory, dict)
    assert trajectory["raw_spans"] == []
    assert observation.metadata["raw_spans_retained"] is False


async def test_expected_trace_id_cannot_be_overridden_by_matching_correlation() -> None:
    observer = OtlpObservationSource(
        OtlpObservationConfig(
            settle_window_seconds=0.01,
            observation_timeout_seconds=0.02,
            poll_interval_seconds=0.002,
        )
    )
    exported_request = ProbeRequest(
        case_id="case-1",
        session_id="session-1",
        correlation_id="shared-correlation",
        turn=ProbeTurn(id="turn-1", input="hello"),
        context={"trace_id": "b" * 32, "span_id": "c" * 16},
    )
    observer.export(_payload(exported_request, include_child=False))

    mismatched = await observer.observe(
        ObservationRequest(
            case_id="case-1",
            session_id="session-1",
            correlation_id="shared-correlation",
            context={"trace_id": "a" * 32, "span_id": "c" * 16},
        )
    )
    fallback = await observer.observe(
        ObservationRequest(
            case_id="case-1",
            session_id="session-1",
            correlation_id="shared-correlation",
            context={"span_id": "c" * 16},
        )
    )

    assert mismatched.status == "missing"
    assert fallback.status == "complete"


async def test_rejection_markers_are_bounded_by_span_retention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    monkeypatch.setattr(observation_module.time, "monotonic", lambda: now[0])
    observer = OtlpObservationSource(
        OtlpObservationConfig(
            maximum_spans=1,
            retention_seconds=1,
            settle_window_seconds=0.01,
            observation_timeout_seconds=0.1,
        )
    )
    request = ProbeRequest(
        case_id="case-1",
        session_id="session-1",
        correlation_id="correlation-1",
        turn=ProbeTurn(id="turn-1", input="hello"),
        context={"trace_id": "a" * 32, "span_id": "b" * 16},
    )

    first_receipt = observer.export(_payload(request))
    assert first_receipt.rejected_spans == 1
    assert observer._rejected_trace_ids == {"a" * 32: 100.0}

    now[0] = 102.0
    observer.export(_payload(request, include_child=False))
    assert observer._rejected_trace_ids == {}
    assert observer._rejected_correlations == {}
    now[0] = 102.02
    observation = await observer.observe(
        ObservationRequest(
            case_id=request.case_id,
            session_id=request.session_id,
            correlation_id=request.correlation_id,
            context=request.context,
        )
    )

    assert observation.status == "complete"


async def test_rejection_markers_stay_bounded_under_distinct_hostile_traces() -> None:
    observer = OtlpObservationSource(OtlpObservationConfig(maximum_spans=1))
    retained_request = ProbeRequest(
        case_id="case-1",
        session_id="session-1",
        correlation_id="retained",
        turn=ProbeTurn(id="turn-1", input="hello"),
        context={"trace_id": "a" * 32, "span_id": "b" * 16},
    )
    observer.export(_payload(retained_request, include_child=False))

    for index in range(1, 21):
        hostile_request = ProbeRequest(
            case_id="case-1",
            session_id="session-1",
            correlation_id=f"hostile-{index}",
            turn=ProbeTurn(id="turn-1", input="hello"),
            context={"trace_id": f"{index:032x}", "span_id": "b" * 16},
        )
        receipt = observer.export(_payload(hostile_request, include_child=False))
        assert receipt.rejected_spans == 1

    assert len(observer._rejected_trace_ids) == 1
    assert len(observer._rejected_correlations) == 1
    assert set(observer._rejected_trace_ids) == {f"{20:032x}"}
    assert set(observer._rejected_correlations) == {"hostile-20"}


async def test_rejection_marker_eviction_does_not_scan_the_bounded_collection() -> None:
    observer = OtlpObservationSource(OtlpObservationConfig(maximum_spans=500))
    markers = _CountingOrderedDict()
    for index in range(500):
        markers[f"old-{index}"] = float(index)
    observer._rejected_trace_ids = markers

    for index in range(500):
        observer._remember_rejection(
            observer._rejected_trace_ids,
            f"new-{index}",
            float(500 + index),
        )

    assert len(markers) == 500
    assert markers.iteration_count == 0
    assert next(iter(markers.items())) == ("new-0", 500.0)
    assert next(reversed(markers.items())) == ("new-499", 999.0)


async def test_existing_span_updates_use_the_identity_index_without_repeated_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = OtlpObservationSource(OtlpObservationConfig(maximum_spans=500))
    request = ProbeRequest(
        case_id="case-1",
        session_id="session-1",
        correlation_id="correlation-1",
        turn=ProbeTurn(id="turn-1", input="hello"),
        context={"trace_id": "a" * 32, "span_id": "b" * 16},
    )
    payload = _payload(request, include_child=False)
    scope_spans = payload["resourceSpans"][0]["scopeSpans"]
    assert isinstance(scope_spans, list)
    spans = scope_spans[0]["spans"]
    assert isinstance(spans, list)
    root = spans[0]
    assert isinstance(root, dict)
    spans.extend({**root, "spanId": f"{index:016x}", "attributes": []} for index in range(2, 501))
    assert observer.export(payload).accepted_spans == 500
    counted_spans = _CountingList(observer._spans)
    observer._spans = counted_spans
    monkeypatch.setattr(observer, "_prune", lambda now: None)

    receipt = observer.export(payload)

    assert receipt.accepted_spans == 500
    assert receipt.rejected_spans == 0
    assert counted_spans.iteration_count == 1
    assert len(observer._span_indexes) == 500


async def test_span_limit_is_enforced_before_shared_otlp_envelopes_are_copied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = OtlpObservationSource(OtlpObservationConfig(maximum_spans=1))
    request = ProbeRequest(
        case_id="case-1",
        session_id="session-1",
        correlation_id="correlation-1",
        turn=ProbeTurn(id="turn-1", input="hello"),
        context={"trace_id": "a" * 32, "span_id": "b" * 16},
    )
    payload = _payload(request, include_child=False)
    scope_spans = payload["resourceSpans"][0]["scopeSpans"]
    assert isinstance(scope_spans, list)
    spans = scope_spans[0]["spans"]
    assert isinstance(spans, list)
    root = spans[0]
    assert isinstance(root, dict)
    spans.extend(
        {
            **root,
            "spanId": f"{index:016x}",
            "attributes": [],
        }
        for index in range(2, 102)
    )
    redact_calls = 0
    original_redact = observation_module._redact

    def counted_redact(value: JsonValue, terms: tuple[str, ...], key: str = "") -> JsonValue:
        nonlocal redact_calls
        if isinstance(value, dict) and set(value) == {"resource", "scope", "span"}:
            redact_calls += 1
        return original_redact(value, terms, key)

    monkeypatch.setattr(observation_module, "_redact", counted_redact)

    receipt = observer.export(payload)

    assert receipt.accepted_spans == 1
    assert receipt.rejected_spans == 100
    assert redact_calls == 1


async def test_bounded_receiver_reports_dropped_spans_as_incomplete() -> None:
    observer = OtlpObservationSource(
        OtlpObservationConfig(
            maximum_spans=1,
            settle_window_seconds=0.01,
            observation_timeout_seconds=0.1,
        )
    )
    request = ProbeRequest(
        case_id="case-1",
        session_id="session-1",
        correlation_id="correlation-1",
        turn=ProbeTurn(id="turn-1", input="hello"),
        context={"trace_id": "a" * 32, "span_id": "b" * 16},
    )

    receipt = observer.export(_payload(request))
    observation = await observer.observe(
        ObservationRequest(
            case_id=request.case_id,
            session_id=request.session_id,
            correlation_id=request.correlation_id,
            context=request.context,
        )
    )

    assert receipt.accepted_spans == 1
    assert receipt.rejected_spans == 1
    assert receipt.partial_success is True
    assert observation.status == "incomplete"
    assert observation.limitation == "the bounded receiver dropped spans"


async def test_receiver_rejects_non_json_and_recursive_payloads_without_retaining_data() -> None:
    observer = OtlpObservationSource()
    recursive: list[object] = []
    recursive.append(recursive)

    non_json_receipt = observer.export(object())
    recursive_receipt = observer.export(recursive)

    assert non_json_receipt.rejected_spans == 1
    assert recursive_receipt.rejected_spans == 1
    assert non_json_receipt.partial_success is True
    assert recursive_receipt.partial_success is True


async def test_full_openinference_span_kind_family_preserves_provenance() -> None:
    observer = OtlpObservationSource(
        OtlpObservationConfig(
            settle_window_seconds=0.01,
            observation_timeout_seconds=0.1,
        )
    )
    request = ProbeRequest(
        case_id="case-1",
        session_id="session-1",
        correlation_id="correlation-1",
        turn=ProbeTurn(id="turn-1", input="hello"),
        context={"trace_id": "a" * 32, "span_id": "b" * 16},
    )
    observer.export(_openinference_family_payload(request))

    observation = await observer.observe(
        ObservationRequest(
            case_id=request.case_id,
            session_id=request.session_id,
            correlation_id=request.correlation_id,
            context=request.context,
        )
    )

    assert observation.status == "complete"
    trajectory = observation.traces[0]
    assert isinstance(trajectory, dict)
    normalized = trajectory["normalized"]
    assert isinstance(normalized, dict)
    spans = normalized["spans"]
    assert isinstance(spans, list)
    assert [span["kind"] for span in spans if isinstance(span, dict)] == [
        "agent",
        "agent",
        "llm",
        "tool",
        "guardrail",
        "evaluator",
        "retriever",
        "embedding",
        "reranker",
        "span",
    ]
    assert all(span["standard"] == "openinference" for span in spans if isinstance(span, dict))


async def test_span_order_uses_numeric_timestamps_with_malformed_values_last() -> None:
    observer = OtlpObservationSource(
        OtlpObservationConfig(
            settle_window_seconds=0.01,
            observation_timeout_seconds=0.1,
        )
    )
    request = ProbeRequest(
        case_id="case-1",
        session_id="session-1",
        correlation_id="correlation-1",
        turn=ProbeTurn(id="turn-1", input="hello"),
        context={"trace_id": "a" * 32, "span_id": "b" * 16},
    )
    payload = _payload(request, include_child=False)
    scope_spans = payload["resourceSpans"][0]["scopeSpans"]
    assert isinstance(scope_spans, list)
    spans = scope_spans[0]["spans"]
    assert isinstance(spans, list)
    root = spans[0]
    assert isinstance(root, dict)
    root["startTimeUnixNano"] = "10"
    spans.extend(
        [
            {**root, "spanId": "2" * 16, "parentSpanId": "1" * 16, "startTimeUnixNano": "9"},
            {
                **root,
                "spanId": "3" * 16,
                "parentSpanId": "1" * 16,
                "startTimeUnixNano": "bad",
            },
            {
                key: value
                for key, value in {
                    **root,
                    "spanId": "4" * 16,
                    "parentSpanId": "1" * 16,
                }.items()
                if key != "startTimeUnixNano"
            },
        ]
    )
    observer.export(payload)

    observation = await observer.observe(
        ObservationRequest(
            case_id=request.case_id,
            session_id=request.session_id,
            correlation_id=request.correlation_id,
            context=request.context,
        )
    )

    trajectory = observation.traces[0]
    assert isinstance(trajectory, dict)
    normalized = trajectory["normalized"]
    assert isinstance(normalized, dict)
    normalized_spans = normalized["spans"]
    assert isinstance(normalized_spans, list)
    assert [span["span_id"] for span in normalized_spans if isinstance(span, dict)] == [
        "2" * 16,
        "1" * 16,
        "3" * 16,
        "4" * 16,
    ]


@dataclass
class _FlushExporter:
    observer: OtlpObservationSource
    requests: list[ProbeRequest] = field(default_factory=lambda: [])

    def flush(self, request: ProbeRequest) -> None:
        self.requests.append(request)
        self.observer.export(_payload(request, include_child=False))


@dataclass
class _PassiveInvoker:
    capabilities: ProbeInvokerCapabilities = field(
        default_factory=lambda: ProbeInvokerCapabilities(
            invoker_id="local-agent",
            response_size_limit_bytes=1_000,
            supports_conversations=True,
        )
    )
    requests: list[ProbeRequest] = field(default_factory=lambda: [])

    async def invoke(self, request: ProbeRequest) -> ProbeResult:
        self.requests.append(request)
        return ProbeResult(
            id="result-1",
            correlation_id=request.correlation_id,
            response="done",
        )


async def test_worker_flush_hook_runs_before_observation() -> None:
    observer = OtlpObservationSource(
        OtlpObservationConfig(
            settle_window_seconds=0.01,
            observation_timeout_seconds=0.1,
        )
    )
    invoker = _PassiveInvoker()
    flusher = _FlushExporter(observer)
    executor = ComposedEnvironmentExecutor(
        invoker,
        config_sha256=_CONFIG_SHA256,
        observation_source=observer,
        worker_trace_flusher=flusher,
        observation_timeout_seconds=0.2,
    )
    case = evaluation_case_from_inputs(
        case_id="case-1",
        raw_inputs=("help me",),
        max_environment_api_calls=2,
        timeout_seconds=1,
    )

    evidence = await executor.execute(case)

    assert flusher.requests == invoker.requests
    assert evidence.observations[0].status == "complete"


async def test_case_context_supplies_variation_and_repetition_identity() -> None:
    observer = OtlpObservationSource(
        OtlpObservationConfig(
            settle_window_seconds=0.01,
            observation_timeout_seconds=0.1,
        )
    )
    invoker = _PassiveInvoker()
    executor = ComposedEnvironmentExecutor(
        invoker,
        config_sha256=_CONFIG_SHA256,
        observation_source=observer,
        worker_trace_flusher=_FlushExporter(observer),
        observation_timeout_seconds=0.2,
    )
    case = evaluation_case_from_inputs(
        case_id="dataset-record-1",
        raw_inputs=("help me",),
        max_environment_api_calls=2,
        timeout_seconds=1,
    ).model_copy(
        update={
            "probe_context": {
                "ul.variation.id": "input.surface.rephrase",
                "ul.repetition": 3,
            }
        }
    )

    evidence = await executor.execute(case)

    assert invoker.requests[0].context["ul.case.id"] == "dataset-record-1"
    assert invoker.requests[0].context["ul.variation.id"] == "input.surface.rephrase"
    assert invoker.requests[0].context["ul.repetition"] == 3
    assert evidence.probe_identity is not None
    assert evidence.probe_identity.variation_id == "input.surface.rephrase"
    assert evidence.probe_identity.repetition == 3


async def test_multi_turn_probe_uses_distinct_trace_ids_and_correlates_each_turn() -> None:
    observer = OtlpObservationSource(
        OtlpObservationConfig(
            settle_window_seconds=0.01,
            observation_timeout_seconds=0.1,
        )
    )
    invoker = _PassiveInvoker()
    flusher = _FlushExporter(observer)
    executor = ComposedEnvironmentExecutor(
        invoker,
        config_sha256=_CONFIG_SHA256,
        observation_source=observer,
        worker_trace_flusher=flusher,
        observation_timeout_seconds=0.2,
    )
    case = evaluation_case_from_inputs(
        case_id="case-1",
        raw_inputs=("first", "second"),
        max_environment_api_calls=2,
        timeout_seconds=1,
    )

    evidence = await executor.execute(case)

    assert len(evidence.observations) == 2
    assert all(observation.status == "complete" for observation in evidence.observations)
    assert invoker.requests[0].context["trace_id"] != invoker.requests[1].context["trace_id"]
    assert evidence.observations[0].correlation_id != evidence.observations[1].correlation_id
