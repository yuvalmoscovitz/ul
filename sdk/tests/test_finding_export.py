from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Literal, cast

import httpx
import pytest
from pydantic import JsonValue, ValidationError
from ul.finding_export import (
    FindingExportInputError,
    append_finding_annotations,
    create_finding_artifact_reference,
    create_finding_bundle,
    create_finding_evidence_reference,
    create_finding_record,
    finding_otlp_events,
    finding_otlp_json,
    parse_finding_annotations_jsonl,
    parse_private_finding_bundle_json,
    parse_private_finding_bundle_jsonl,
    parse_safe_finding_bundle_json,
    parse_safe_finding_bundle_jsonl,
    private_finding_bundle_json,
    private_finding_bundle_jsonl,
    safe_finding_bundle,
    safe_finding_bundle_json,
    safe_finding_bundle_jsonl,
)
from ul.otlp_observation import OtlpJsonHttpReceiver, OtlpObservationSource
from ul_core.finding_export import (
    FindingAnnotationInput,
    FindingEvidenceLevel,
    FindingProvenance,
    FindingRecord,
    W3CTraceReference,
)

pytestmark = pytest.mark.asyncio
_CANARY = "private-customer-content-secret-9f3b"
_RECORDED_AT = datetime(2026, 8, 23, 7, 0, tzinfo=UTC)


def _bundle():
    evidence_reference = create_finding_evidence_reference(
        kind="trajectory",
        source_id=f"observer-{_CANARY}",
        authority="independent_observer",
        sha256="a" * 64,
        locator=f"s3://private/{_CANARY}/raw-trace.json",
    )
    artifact_reference = create_finding_artifact_reference(
        kind="run_receipt",
        media_type="application/json",
        sha256="b" * 64,
        locator=f"file:///private/{_CANARY}/receipt.json",
    )
    provenance = FindingProvenance(
        producer_name="underlayer",
        producer_version="0.1.0",
        config_sha256="c" * 64,
        source_finding_id=f"source-{_CANARY}",
        campaign_id=f"campaign-{_CANARY}",
        case_id=f"case-{_CANARY}",
        source_interaction_id=f"interaction-{_CANARY}",
        probe_id="probe-1",
        attempt_id="attempt-1",
        session_id="session-1",
        turn_ids=("turn-1",),
        variation_id="input.surface.rephrase",
        repetition=1,
        fixture_id="accounts",
        fixture_version="2",
        metadata={"authorization": f"Bearer {_CANARY}", "prompt": _CANARY},
    )
    evidence_level = FindingEvidenceLevel(
        facts=("response_observed", "trajectory_observed"),
        sources={
            "response_observed": "local-agent",
            "trajectory_observed": f"observer-{_CANARY}",
        },
        authorities={
            "response_observed": "invoker_self_reported",
            "trajectory_observed": "independent_observer",
        },
        limitations=("committed state was not verified",),
    )
    finding = create_finding_record(
        conclusion="observed_variance",
        category="changed_grounded_effect_argument",
        review_status="needs_review",
        severity="unrated",
        evidence_level=evidence_level,
        target_trace=W3CTraceReference(trace_id="1" * 32, span_id="2" * 16),
        evidence_references=(evidence_reference,),
        artifact_references=(artifact_reference,),
        recorded_at=_RECORDED_AT,
        provenance=provenance,
        private_payload={
            "content": _CANARY,
            "secret": _CANARY,
            "raw_trace": {"input": _CANARY, "output": _CANARY},
            "state": {"before": _CANARY, "after": _CANARY},
            "prompt": _CANARY,
        },
    )
    return create_finding_bundle((finding,), created_at=_RECORDED_AT)


async def test_safe_and_private_json_and_jsonl_round_trip() -> None:
    bundle = _bundle()

    safe_json = safe_finding_bundle_json(bundle)
    safe_jsonl = safe_finding_bundle_jsonl(bundle)
    private_json = private_finding_bundle_json(bundle, private_export_confirmed=True)
    private_jsonl = private_finding_bundle_jsonl(bundle, private_export_confirmed=True)

    assert _CANARY not in safe_json
    assert _CANARY not in safe_jsonl
    assert _CANARY in private_json
    assert _CANARY in private_jsonl
    assert parse_safe_finding_bundle_json(safe_json) == safe_finding_bundle(bundle)
    assert parse_safe_finding_bundle_jsonl(safe_jsonl) == safe_finding_bundle(bundle)
    assert parse_private_finding_bundle_json(private_json) == bundle
    assert parse_private_finding_bundle_jsonl(private_jsonl) == bundle
    with pytest.raises(FindingExportInputError, match="explicit confirmation"):
        private_finding_bundle_json(bundle, private_export_confirmed=cast(Literal[True], False))


async def test_deterministic_ids_and_strict_w3c_and_evidence_validation() -> None:
    first = _bundle()
    second = _bundle()

    assert first == second
    assert first.bundle_id == second.bundle_id
    assert first.findings[0].finding_id == second.findings[0].finding_id
    with pytest.raises(ValidationError, match="W3C trace"):
        W3CTraceReference(trace_id="0" * 32, span_id="2" * 16)
    with pytest.raises(ValidationError, match="sorted and unique"):
        FindingEvidenceLevel(
            facts=("trajectory_observed", "response_observed"),
            sources={"trajectory_observed": "a", "response_observed": "b"},
            authorities={
                "trajectory_observed": "independent_observer",
                "response_observed": "invoker_self_reported",
            },
        )
    invalid = first.findings[0].model_dump(mode="python")
    invalid["finding_id"] = f"ulf_export_v1_{'0' * 64}"
    with pytest.raises(ValidationError, match="not deterministic"):
        FindingRecord.model_validate(invalid)


async def test_neutral_annotation_appends_review_without_rewriting_evidence() -> None:
    bundle = _bundle()
    finding = bundle.findings[0]
    annotation_jsonl = json.dumps(
        {
            "schema_version": "1.0.0",
            "finding_id": finding.finding_id,
            "status": "confirmed",
            "severity": "high",
            "annotator_kind": "HUMAN",
            "reviewer": f"reviewer-{_CANARY}",
            "reason": f"confirmed from private evidence {_CANARY}",
            "reviewed_at": (_RECORDED_AT + timedelta(minutes=1)).isoformat(),
            "supersedes_annotation_id": None,
        },
        separators=(",", ":"),
    )

    inputs = parse_finding_annotations_jsonl(f"{annotation_jsonl}\n")
    reviewed = append_finding_annotations(bundle, inputs)
    safe = safe_finding_bundle(reviewed)

    assert reviewed.findings == bundle.findings
    assert reviewed.bundle_id == bundle.bundle_id
    assert len(reviewed.annotations) == 1
    assert safe.findings[0].conclusion == "confirmed_correctness_failure"
    assert safe.findings[0].review_status == "confirmed"
    assert safe.findings[0].severity == "high"
    assert _CANARY not in safe_finding_bundle_json(reviewed)
    assert _CANARY in private_finding_bundle_json(reviewed, private_export_confirmed=True)
    with pytest.raises(FindingExportInputError, match="supersede"):
        append_finding_annotations(bundle, inputs + inputs)


async def test_otlp_carrier_uses_generic_link_and_safe_evaluation_conventions() -> None:
    bundle = _bundle()

    first = finding_otlp_json(bundle)
    second = finding_otlp_json(bundle)
    events = finding_otlp_events(bundle)
    encoded = json.dumps(first, sort_keys=True)
    span = first["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert isinstance(span, dict)

    assert first == second
    assert events[0].target_trace == bundle.findings[0].target_trace
    assert span["kind"] == 1
    assert span["links"] == [{"traceId": "1" * 32, "spanId": "2" * 16}]
    assert span["events"][0]["name"] == "gen_ai.evaluation.result"
    attributes = {
        item["key"]: item["value"] for item in span["attributes"] if isinstance(item, dict)
    }
    assert attributes["openinference.span.kind"] == {"stringValue": "EVALUATOR"}
    assert attributes["evaluations.0.evaluation.label"] == {"stringValue": "observed_variance"}
    assert attributes["underlayer.finding.evidence.facts"]["arrayValue"]["values"] == [
        {"stringValue": "response_observed"},
        {"stringValue": "trajectory_observed"},
    ]
    assert _CANARY not in encoded
    for forbidden in ("content", "secret", "raw_trace", "state", "prompt", "authorization"):
        assert forbidden not in encoded.casefold()


async def test_otlp_payload_reaches_bounded_ul_receiver() -> None:
    observer = OtlpObservationSource()
    payload = finding_otlp_json(_bundle())

    with OtlpJsonHttpReceiver(observer) as receiver:
        async with httpx.AsyncClient() as client:
            response = await client.post(receiver.endpoint, json=payload)

    assert response.status_code == 200
    assert response.json()["partialSuccess"]["rejectedSpans"] == 0


async def test_otlp_payload_reaches_independent_generic_http_receiver() -> None:
    payloads: list[dict[str, JsonValue]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/v1/traces" or self.headers.get("Content-Type") != "application/json":
                self.send_response(400)
                self.end_headers()
                return
            length = int(self.headers["Content-Length"])
            parsed = json.loads(self.rfile.read(length))
            payloads.append(cast(dict[str, JsonValue], parsed))
            response = b'{"partialSuccess":{"rejectedSpans":0,"errorMessage":""}}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{server.server_port}/v1/traces",
                json=finding_otlp_json(_bundle()),
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    assert response.status_code == 200
    assert len(payloads) == 1
    spans = payloads[0]["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert isinstance(spans, list)
    assert len(spans) == 1


async def test_parsers_reject_duplicate_keys_nonstandard_numbers_and_unknown_records() -> None:
    with pytest.raises(FindingExportInputError, match="invalid"):
        parse_private_finding_bundle_json('{"schema_version":"1.0.0","schema_version":"1.0.0"}')
    with pytest.raises(FindingExportInputError, match="invalid"):
        parse_private_finding_bundle_json('{"value":NaN}')
    with pytest.raises(FindingExportInputError, match="unknown record"):
        parse_private_finding_bundle_jsonl(
            '{"record_type":"private_bundle"}\n{"record_type":"unknown"}\n'
        )


async def test_annotation_input_requires_utc_and_confirmed_severity() -> None:
    finding_id = _bundle().findings[0].finding_id
    with pytest.raises(ValidationError, match="UTC"):
        FindingAnnotationInput(
            finding_id=finding_id,
            status="confirmed",
            severity="high",
            reviewer="reviewer",
            reason="confirmed",
            reviewed_at=datetime(2026, 8, 23, 7, 0),
        )
    with pytest.raises(ValidationError, match="only confirmed"):
        FindingAnnotationInput(
            finding_id=finding_id,
            status="expected",
            severity="high",
            reviewer="reviewer",
            reason="expected",
            reviewed_at=_RECORDED_AT,
        )
