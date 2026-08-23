from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal, cast

from pydantic import JsonValue, ValidationError
from ul_core.evaluation import EvidenceAuthority
from ul_core.finding_export import (
    FindingAnnotation,
    FindingAnnotationInput,
    FindingArtifactKind,
    FindingArtifactReference,
    FindingBundle,
    FindingConclusion,
    FindingEvidenceLevel,
    FindingEvidenceReference,
    FindingExportSeverity,
    FindingOtlpEvent,
    FindingProvenance,
    FindingRecord,
    FindingReferenceKind,
    FindingReviewStatus,
    SafeFindingAnnotation,
    SafeFindingBundle,
    SafeFindingRecord,
    W3CTraceReference,
    finding_annotation_id,
    finding_artifact_reference_id,
    finding_bundle_id,
    finding_evidence_reference_id,
    finding_record_id,
    finding_record_sha256,
    safe_finding_bundle_id,
)

_MAXIMUM_BUNDLE_BYTES = 128_000_000
_MAXIMUM_JSON_DEPTH = 64
_MAXIMUM_JSONL_RECORDS = 20_001
_FINDING_SCHEMA_VERSION = "1.0.0"


class FindingExportInputError(ValueError):
    pass


def create_finding_evidence_reference(
    *,
    kind: FindingReferenceKind,
    source_id: str,
    authority: EvidenceAuthority,
    sha256: str,
    locator: str | None = None,
) -> FindingEvidenceReference:
    return FindingEvidenceReference(
        reference_id=finding_evidence_reference_id(
            kind=kind,
            source_id=source_id,
            authority=authority,
            sha256=sha256,
            locator=locator,
        ),
        kind=kind,
        source_id=source_id,
        authority=authority,
        sha256=sha256,
        locator=locator,
    )


def create_finding_artifact_reference(
    *,
    kind: FindingArtifactKind,
    media_type: str,
    sha256: str,
    locator: str | None = None,
) -> FindingArtifactReference:
    return FindingArtifactReference(
        reference_id=finding_artifact_reference_id(
            kind=kind,
            media_type=media_type,
            sha256=sha256,
            locator=locator,
        ),
        kind=kind,
        media_type=media_type,
        sha256=sha256,
        locator=locator,
    )


def create_finding_record(
    *,
    conclusion: FindingConclusion,
    category: str,
    review_status: FindingReviewStatus,
    severity: FindingExportSeverity,
    evidence_level: FindingEvidenceLevel,
    target_trace: W3CTraceReference,
    evidence_references: tuple[FindingEvidenceReference, ...],
    recorded_at: datetime,
    provenance: FindingProvenance,
    artifact_references: tuple[FindingArtifactReference, ...] = (),
    private_payload: JsonValue = None,
) -> FindingRecord:
    ordered_evidence = tuple(sorted(evidence_references, key=lambda item: item.reference_id))
    ordered_artifacts = tuple(sorted(artifact_references, key=lambda item: item.reference_id))
    finding_id = finding_record_id(
        category=category,
        target_trace=target_trace,
        provenance=provenance,
        evidence_reference_ids=tuple(item.reference_id for item in ordered_evidence),
    )
    return FindingRecord(
        finding_id=finding_id,
        conclusion=conclusion,
        category=category,
        review_status=review_status,
        severity=severity,
        evidence_level=evidence_level,
        target_trace=target_trace,
        evidence_references=ordered_evidence,
        artifact_references=ordered_artifacts,
        recorded_at=recorded_at,
        provenance=provenance,
        private_payload=private_payload,
    )


def create_finding_bundle(
    findings: tuple[FindingRecord, ...], *, created_at: datetime
) -> FindingBundle:
    ordered = tuple(sorted(findings, key=lambda item: item.finding_id))
    return FindingBundle(
        bundle_id=finding_bundle_id(tuple(item.finding_id for item in ordered)),
        created_at=created_at,
        findings=ordered,
    )


def append_finding_annotations(
    bundle: FindingBundle, annotations: tuple[FindingAnnotationInput, ...]
) -> FindingBundle:
    findings = {finding.finding_id: finding for finding in bundle.findings}
    appended = list(bundle.annotations)
    active: dict[str, FindingAnnotation] = {}
    for existing in bundle.annotations:
        active[existing.finding_id] = existing
    for annotation_input in annotations:
        finding = findings.get(annotation_input.finding_id)
        if finding is None:
            raise FindingExportInputError("annotation references a finding outside the bundle")
        current = active.get(annotation_input.finding_id)
        expected_supersedes = current.annotation_id if current is not None else None
        if annotation_input.supersedes_annotation_id != expected_supersedes:
            raise FindingExportInputError("annotation must supersede the active review")
        if current is not None and annotation_input.reviewed_at <= current.reviewed_at:
            raise FindingExportInputError("superseding annotation time must increase")
        record_digest = finding_record_sha256(finding)
        annotation = FindingAnnotation(
            **annotation_input.model_dump(mode="python"),
            annotation_id=finding_annotation_id(annotation_input, record_digest),
            finding_record_sha256=record_digest,
        )
        appended.append(annotation)
        active[annotation.finding_id] = annotation
    bundle_data = bundle.model_dump(mode="python")
    bundle_data["annotations"] = tuple(
        sorted(appended, key=lambda item: (item.reviewed_at, item.annotation_id))
    )
    return FindingBundle.model_validate(bundle_data)


def safe_finding_bundle(bundle: FindingBundle) -> SafeFindingBundle:
    active_annotations: dict[str, FindingAnnotation] = {}
    for annotation in bundle.annotations:
        active_annotations[annotation.finding_id] = annotation
    findings = tuple(
        _safe_finding(finding, active_annotations.get(finding.finding_id))
        for finding in bundle.findings
    )
    annotations = tuple(
        SafeFindingAnnotation(
            annotation_id=annotation.annotation_id,
            finding_id=annotation.finding_id,
            status=annotation.status,
            severity=annotation.severity,
            annotator_kind=annotation.annotator_kind,
            reviewed_at=annotation.reviewed_at,
        )
        for annotation in bundle.annotations
    )
    return SafeFindingBundle(
        bundle_id=safe_finding_bundle_id(findings, annotations),
        source_bundle_id=bundle.bundle_id,
        created_at=bundle.created_at,
        findings=findings,
        annotations=annotations,
    )


def safe_finding_bundle_json(bundle: FindingBundle) -> str:
    return _serialize_json(safe_finding_bundle(bundle))


def private_finding_bundle_json(
    bundle: FindingBundle, *, private_export_confirmed: Literal[True]
) -> str:
    if private_export_confirmed is not True:
        raise FindingExportInputError("private finding export requires explicit confirmation")
    return _serialize_json(bundle)


def safe_finding_bundle_jsonl(bundle: FindingBundle) -> str:
    safe_bundle = safe_finding_bundle(bundle)
    return _serialize_jsonl(
        header={
            "record_type": "safe_bundle",
            "schema_version": safe_bundle.schema_version,
            "bundle_id": safe_bundle.bundle_id,
            "source_bundle_id": safe_bundle.source_bundle_id,
            "created_at": safe_bundle.created_at.isoformat(),
        },
        findings=tuple(
            {"record_type": "finding", "finding": finding.model_dump(mode="json")}
            for finding in safe_bundle.findings
        ),
        annotations=tuple(
            {"record_type": "annotation", "annotation": annotation.model_dump(mode="json")}
            for annotation in safe_bundle.annotations
        ),
    )


def private_finding_bundle_jsonl(
    bundle: FindingBundle, *, private_export_confirmed: Literal[True]
) -> str:
    if private_export_confirmed is not True:
        raise FindingExportInputError("private finding export requires explicit confirmation")
    return _serialize_jsonl(
        header={
            "record_type": "private_bundle",
            "schema_version": bundle.schema_version,
            "bundle_id": bundle.bundle_id,
            "created_at": bundle.created_at.isoformat(),
        },
        findings=tuple(
            {"record_type": "finding", "finding": finding.model_dump(mode="json")}
            for finding in bundle.findings
        ),
        annotations=tuple(
            {"record_type": "annotation", "annotation": annotation.model_dump(mode="json")}
            for annotation in bundle.annotations
        ),
    )


def parse_safe_finding_bundle_json(value: str | bytes) -> SafeFindingBundle:
    return _parse_model_json(value, SafeFindingBundle)


def parse_private_finding_bundle_json(value: str | bytes) -> FindingBundle:
    return _parse_model_json(value, FindingBundle)


def parse_safe_finding_bundle_jsonl(value: str | bytes) -> SafeFindingBundle:
    header, findings, annotations = _parse_jsonl(value, expected_header="safe_bundle")
    return SafeFindingBundle.model_validate(
        {
            "schema_version": header.get("schema_version"),
            "bundle_id": header.get("bundle_id"),
            "source_bundle_id": header.get("source_bundle_id"),
            "created_at": header.get("created_at"),
            "findings": findings,
            "annotations": annotations,
        },
        strict=False,
    )


def parse_private_finding_bundle_jsonl(value: str | bytes) -> FindingBundle:
    header, findings, annotations = _parse_jsonl(value, expected_header="private_bundle")
    return FindingBundle.model_validate(
        {
            "schema_version": header.get("schema_version"),
            "bundle_id": header.get("bundle_id"),
            "created_at": header.get("created_at"),
            "findings": findings,
            "annotations": annotations,
        },
        strict=False,
    )


def parse_finding_annotations_jsonl(value: str | bytes) -> tuple[FindingAnnotationInput, ...]:
    records = _parse_json_lines(value)
    try:
        return tuple(
            FindingAnnotationInput.model_validate(record, strict=False) for record in records
        )
    except ValidationError:
        raise FindingExportInputError("finding annotation JSONL is invalid") from None


def finding_otlp_events(bundle: FindingBundle) -> tuple[FindingOtlpEvent, ...]:
    safe_bundle = safe_finding_bundle(bundle)
    return tuple(_finding_otlp_event(finding) for finding in safe_bundle.findings)


def finding_otlp_json(bundle: FindingBundle) -> dict[str, JsonValue]:
    events = finding_otlp_events(bundle)
    spans = [_otlp_span(event) for event in events]
    return cast(
        dict[str, JsonValue],
        {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            _otlp_attribute("service.name", "underlayer.finding-export"),
                            _otlp_attribute("service.version", _FINDING_SCHEMA_VERSION),
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {
                                "name": "underlayer.finding_export",
                                "version": _FINDING_SCHEMA_VERSION,
                            },
                            "spans": spans,
                        }
                    ],
                }
            ]
        },
    )


def _safe_finding(
    finding: FindingRecord, annotation: FindingAnnotation | None
) -> SafeFindingRecord:
    if annotation is None:
        conclusion = finding.conclusion
        review_status = finding.review_status
        severity = finding.severity
    else:
        conclusion = (
            "confirmed_correctness_failure"
            if annotation.status == "confirmed"
            else "observed_variance"
        )
        review_status = annotation.status
        severity = annotation.severity
    authority_order = cast(
        tuple[EvidenceAuthority, ...],
        tuple(finding.evidence_level.authorities[fact] for fact in finding.evidence_level.facts),
    )
    return SafeFindingRecord(
        finding_id=finding.finding_id,
        conclusion=conclusion,
        category=finding.category,
        review_status=review_status,
        severity=severity,
        evidence_facts=finding.evidence_level.facts,
        evidence_authorities=authority_order,
        target_trace=finding.target_trace,
        evidence_reference_ids=tuple(item.reference_id for item in finding.evidence_references),
        artifact_reference_ids=tuple(item.reference_id for item in finding.artifact_references),
        recorded_at=finding.recorded_at,
    )


def _finding_otlp_event(finding: SafeFindingRecord) -> FindingOtlpEvent:
    trace_id = hashlib.sha256(f"ul-finding-trace:{finding.finding_id}".encode()).hexdigest()[:32]
    span_id = hashlib.sha256(f"ul-finding-span:{finding.finding_id}".encode()).hexdigest()[:16]
    if set(trace_id) == {"0"}:
        trace_id = f"1{trace_id[1:]}"
    if set(span_id) == {"0"}:
        span_id = f"1{span_id[1:]}"
    seconds = int(finding.recorded_at.timestamp())
    time_unix_nano = seconds * 1_000_000_000 + finding.recorded_at.microsecond * 1_000
    return FindingOtlpEvent(
        finding_id=finding.finding_id,
        carrier_trace=W3CTraceReference(trace_id=trace_id, span_id=span_id),
        target_trace=finding.target_trace,
        time_unix_nano=time_unix_nano,
        conclusion=finding.conclusion,
        category=finding.category,
        review_status=finding.review_status,
        severity=finding.severity,
        evidence_facts=finding.evidence_facts,
        evidence_authorities=finding.evidence_authorities,
        evidence_reference_ids=finding.evidence_reference_ids,
        artifact_reference_ids=finding.artifact_reference_ids,
    )


def _otlp_span(event: FindingOtlpEvent) -> dict[str, JsonValue]:
    attributes = [
        _otlp_attribute("openinference.span.kind", "EVALUATOR"),
        _otlp_attribute("evaluations.0.evaluation.name", "ul.finding"),
        _otlp_attribute("evaluations.0.evaluation.label", event.conclusion),
        _otlp_attribute("evaluations.0.evaluation.annotator_kind", "CODE"),
        _otlp_attribute("evaluations.0.evaluation.identifier", event.finding_id),
        _otlp_attribute("underlayer.finding.schema_version", event.schema_version),
        _otlp_attribute("underlayer.finding.id", event.finding_id),
        _otlp_attribute("underlayer.finding.conclusion", event.conclusion),
        _otlp_attribute("underlayer.finding.category", event.category),
        _otlp_attribute("underlayer.finding.review.status", event.review_status),
        _otlp_attribute("underlayer.finding.severity", event.severity),
        _otlp_attribute("underlayer.finding.evidence.facts", event.evidence_facts),
        _otlp_attribute("underlayer.finding.evidence.authorities", event.evidence_authorities),
        _otlp_attribute("underlayer.finding.evidence.reference_ids", event.evidence_reference_ids),
        _otlp_attribute("underlayer.finding.artifact.reference_ids", event.artifact_reference_ids),
    ]
    generic_event_attributes = [
        _otlp_attribute("gen_ai.evaluation.name", "ul.finding"),
        _otlp_attribute("gen_ai.evaluation.score.label", event.conclusion),
        _otlp_attribute("underlayer.finding.id", event.finding_id),
        _otlp_attribute("underlayer.finding.category", event.category),
        _otlp_attribute("underlayer.finding.review.status", event.review_status),
        _otlp_attribute("underlayer.finding.evidence.facts", event.evidence_facts),
    ]
    return cast(
        dict[str, JsonValue],
        {
            "traceId": event.carrier_trace.trace_id,
            "spanId": event.carrier_trace.span_id,
            "name": "underlayer finding evaluation",
            "kind": 1,
            "startTimeUnixNano": str(event.time_unix_nano),
            "endTimeUnixNano": str(event.time_unix_nano),
            "attributes": attributes,
            "events": [
                {
                    "timeUnixNano": str(event.time_unix_nano),
                    "name": "gen_ai.evaluation.result",
                    "attributes": generic_event_attributes,
                }
            ],
            "links": [
                {
                    "traceId": event.target_trace.trace_id,
                    "spanId": event.target_trace.span_id,
                }
            ],
            "status": {"code": 1},
        },
    )


def _otlp_attribute(key: str, value: object) -> dict[str, JsonValue]:
    if isinstance(value, tuple):
        items = cast(tuple[object, ...], value)
        return {
            "key": key,
            "value": {"arrayValue": {"values": [{"stringValue": str(item)} for item in items]}},
        }
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": str(value)}}


def _serialize_json(model: FindingBundle | SafeFindingBundle) -> str:
    encoded = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    if len(encoded.encode("utf-8")) > _MAXIMUM_BUNDLE_BYTES:
        raise FindingExportInputError("finding bundle exceeds the 128 MB limit")
    return encoded


def _serialize_jsonl(
    *,
    header: dict[str, JsonValue],
    findings: tuple[dict[str, JsonValue], ...],
    annotations: tuple[dict[str, JsonValue], ...],
) -> str:
    records = (header, *findings, *annotations)
    encoded = "\n".join(
        json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        for record in records
    )
    if len(encoded.encode("utf-8")) > _MAXIMUM_BUNDLE_BYTES:
        raise FindingExportInputError("finding bundle exceeds the 128 MB limit")
    return f"{encoded}\n"


def _parse_model_json[ModelT: FindingBundle | SafeFindingBundle](
    value: str | bytes, model: type[ModelT]
) -> ModelT:
    raw = _parse_json(value)
    try:
        return model.model_validate(raw, strict=False)
    except ValidationError:
        raise FindingExportInputError("finding bundle JSON is invalid") from None


def _parse_jsonl(
    value: str | bytes, *, expected_header: str
) -> tuple[dict[str, object], tuple[object, ...], tuple[object, ...]]:
    records = _parse_json_lines(value)
    if not records or records[0].get("record_type") != expected_header:
        raise FindingExportInputError("finding bundle JSONL header is invalid")
    header = records[0]
    findings: list[object] = []
    annotations: list[object] = []
    for record in records[1:]:
        if record.get("record_type") == "finding" and "finding" in record:
            findings.append(record["finding"])
        elif record.get("record_type") == "annotation" and "annotation" in record:
            annotations.append(record["annotation"])
        else:
            raise FindingExportInputError("finding bundle JSONL contains an unknown record")
    return header, tuple(findings), tuple(annotations)


def _parse_json_lines(value: str | bytes) -> tuple[dict[str, object], ...]:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    if len(encoded) > _MAXIMUM_BUNDLE_BYTES:
        raise FindingExportInputError("finding bundle exceeds the 128 MB limit")
    try:
        decoded = encoded.decode("utf-8")
    except UnicodeDecodeError:
        raise FindingExportInputError("finding JSONL must be UTF-8") from None
    lines = decoded.splitlines()
    if not lines or len(lines) > _MAXIMUM_JSONL_RECORDS or any(not line for line in lines):
        raise FindingExportInputError("finding JSONL record count is invalid")
    records: list[dict[str, object]] = []
    for line in lines:
        parsed = _parse_json(line)
        if not isinstance(parsed, dict):
            raise FindingExportInputError("finding JSONL records must be objects")
        records.append(cast(dict[str, object], parsed))
    return tuple(records)


def _parse_json(value: str | bytes) -> object:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    if len(encoded) > _MAXIMUM_BUNDLE_BYTES:
        raise FindingExportInputError("finding bundle exceeds the 128 MB limit")
    try:
        parsed = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise FindingExportInputError("finding JSON is invalid") from None
    _reject_deep_json(parsed)
    return parsed


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_nonstandard_number(value: str) -> None:
    del value
    raise ValueError("non-standard JSON number")


def _reject_deep_json(value: object) -> None:
    pending: list[tuple[object, int]] = [(value, 1)]
    while pending:
        current, depth = pending.pop()
        if depth > _MAXIMUM_JSON_DEPTH:
            raise FindingExportInputError("finding JSON nesting exceeds the limit")
        if isinstance(current, dict):
            current_object = cast(dict[str, object], current)
            pending.extend((item, depth + 1) for item in current_object.values())
        elif isinstance(current, list):
            current_list = cast(list[object], current)
            pending.extend((item, depth + 1) for item in current_list)
