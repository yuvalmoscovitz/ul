from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Literal, Self, cast

from pydantic import ConfigDict, Field, JsonValue, model_validator

from ul_core.evaluation import EvidenceAuthority, EvidenceFact
from ul_core.models import ULModel

FindingConclusion = Literal["observed_variance", "confirmed_correctness_failure"]
FindingReviewStatus = Literal[
    "needs_review",
    "confirmed",
    "expected",
    "unsupported",
    "inconclusive",
]
FindingExportSeverity = Literal["unrated", "low", "medium", "high", "critical"]
FindingAnnotationStatus = Literal["confirmed", "expected", "unsupported", "inconclusive"]
FindingAnnotatorKind = Literal["HUMAN", "LLM", "CODE"]
FindingReferenceKind = Literal["response", "trajectory", "state", "replay", "artifact"]
FindingArtifactKind = Literal["evidence", "report", "run_receipt", "trace", "other"]

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_FINDING_ID_PATTERN = r"^ulf_export_v1_[0-9a-f]{64}$"
_EVIDENCE_REFERENCE_ID_PATTERN = r"^ule_v1_[0-9a-f]{64}$"
_ARTIFACT_REFERENCE_ID_PATTERN = r"^ular_v1_[0-9a-f]{64}$"
_ANNOTATION_ID_PATTERN = r"^ulann_v1_[0-9a-f]{64}$"
_BUNDLE_ID_PATTERN = r"^ulfb_v1_[0-9a-f]{64}$"
_SAFE_BUNDLE_ID_PATTERN = r"^ulfs_v1_[0-9a-f]{64}$"
_CATEGORY_PATTERN = r"^[a-z][a-z0-9._-]{0,99}$"
_MEDIA_TYPE_PATTERN = r"^[a-z0-9][a-z0-9.+-]{0,99}/[a-z0-9][a-z0-9.+-]{0,99}$"
_TRACE_ID = re.compile(r"^[0-9a-f]{32}$")
_SPAN_ID = re.compile(r"^[0-9a-f]{16}$")
_MAXIMUM_PRIVATE_RECORD_BYTES = 10_000_000


class _StrictModel(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(prefix: str, value: object) -> str:
    return f"{prefix}{hashlib.sha256(_canonical_json(value)).hexdigest()}"


def _validate_utc(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(None):
        raise ValueError(f"{name} must use UTC")


class W3CTraceReference(_StrictModel):
    trace_id: str
    span_id: str

    @model_validator(mode="after")
    def validate_identifiers(self) -> Self:
        if _TRACE_ID.fullmatch(self.trace_id) is None or set(self.trace_id) == {"0"}:
            raise ValueError("trace_id must be a non-zero lowercase W3C trace identifier")
        if _SPAN_ID.fullmatch(self.span_id) is None or set(self.span_id) == {"0"}:
            raise ValueError("span_id must be a non-zero lowercase W3C span identifier")
        return self

    @property
    def traceparent(self) -> str:
        return f"00-{self.trace_id}-{self.span_id}-01"


class FindingEvidenceLevel(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    facts: tuple[EvidenceFact, ...] = Field(min_length=1)
    sources: dict[EvidenceFact, str]
    authorities: dict[EvidenceFact, EvidenceAuthority]
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_level(self) -> Self:
        if self.facts != tuple(sorted(set(self.facts))):
            raise ValueError("finding evidence facts must be sorted and unique")
        if set(self.sources) != set(self.facts) or set(self.authorities) != set(self.facts):
            raise ValueError("every finding evidence fact requires source and authority")
        if any(not 1 <= len(source) <= 500 for source in self.sources.values()):
            raise ValueError("finding evidence source IDs must contain 1 to 500 characters")
        if any(
            not limitation.strip() or len(limitation) > 1_000 for limitation in self.limitations
        ):
            raise ValueError("finding evidence limitations must contain 1 to 1000 characters")
        return self


class FindingEvidenceReference(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    reference_id: str = Field(pattern=_EVIDENCE_REFERENCE_ID_PATTERN)
    kind: FindingReferenceKind
    source_id: str = Field(min_length=1, max_length=500)
    authority: EvidenceAuthority
    sha256: str = Field(pattern=_SHA256_PATTERN)
    locator: str | None = Field(default=None, min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_reference_id(self) -> Self:
        if self.reference_id != finding_evidence_reference_id(
            kind=self.kind,
            source_id=self.source_id,
            authority=self.authority,
            sha256=self.sha256,
            locator=self.locator,
        ):
            raise ValueError("finding evidence reference ID is not deterministic")
        return self


class FindingArtifactReference(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    reference_id: str = Field(pattern=_ARTIFACT_REFERENCE_ID_PATTERN)
    kind: FindingArtifactKind
    media_type: str = Field(pattern=_MEDIA_TYPE_PATTERN)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    locator: str | None = Field(default=None, min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_reference_id(self) -> Self:
        if self.reference_id != finding_artifact_reference_id(
            kind=self.kind,
            media_type=self.media_type,
            sha256=self.sha256,
            locator=self.locator,
        ):
            raise ValueError("finding artifact reference ID is not deterministic")
        return self


class FindingProvenance(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    producer_name: str = Field(min_length=1, max_length=200)
    producer_version: str = Field(min_length=1, max_length=100)
    config_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_finding_id: str = Field(min_length=1, max_length=500)
    campaign_id: str = Field(min_length=1, max_length=500)
    case_id: str = Field(min_length=1, max_length=500)
    source_interaction_id: str | None = Field(default=None, min_length=1, max_length=500)
    probe_id: str | None = Field(default=None, min_length=1, max_length=500)
    attempt_id: str | None = Field(default=None, min_length=1, max_length=500)
    session_id: str | None = Field(default=None, min_length=1, max_length=500)
    turn_ids: tuple[str, ...] = ()
    variation_id: str | None = Field(default=None, min_length=1, max_length=500)
    repetition: int | None = Field(default=None, ge=1)
    fixture_id: str | None = Field(default=None, min_length=1, max_length=500)
    fixture_version: str | None = Field(default=None, min_length=1, max_length=500)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_provenance(self) -> Self:
        if self.turn_ids != tuple(dict.fromkeys(self.turn_ids)):
            raise ValueError("finding provenance turn IDs must be unique and ordered")
        if any(not 1 <= len(turn_id) <= 500 for turn_id in self.turn_ids):
            raise ValueError("finding provenance turn IDs must contain 1 to 500 characters")
        if (self.fixture_id is None) != (self.fixture_version is None):
            raise ValueError("finding fixture ID and version must be provided together")
        return self


class FindingRecord(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    finding_id: str = Field(pattern=_FINDING_ID_PATTERN)
    conclusion: FindingConclusion
    category: str = Field(pattern=_CATEGORY_PATTERN)
    review_status: FindingReviewStatus
    severity: FindingExportSeverity = "unrated"
    evidence_level: FindingEvidenceLevel
    target_trace: W3CTraceReference
    evidence_references: tuple[FindingEvidenceReference, ...] = Field(min_length=1)
    artifact_references: tuple[FindingArtifactReference, ...] = ()
    recorded_at: datetime
    provenance: FindingProvenance
    private_payload: JsonValue = None

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        _validate_utc(self.recorded_at, "recorded_at")
        if self.conclusion == "confirmed_correctness_failure":
            if self.review_status != "confirmed":
                raise ValueError("confirmed correctness failures require confirmed review")
        elif self.review_status == "confirmed":
            raise ValueError("confirmed review requires a confirmed correctness failure")
        if self.review_status != "confirmed" and self.severity != "unrated":
            raise ValueError("only confirmed findings can have rated severity")
        evidence_ids = tuple(reference.reference_id for reference in self.evidence_references)
        artifact_ids = tuple(reference.reference_id for reference in self.artifact_references)
        if evidence_ids != tuple(sorted(set(evidence_ids))):
            raise ValueError("finding evidence references must be sorted and unique")
        if artifact_ids != tuple(sorted(set(artifact_ids))):
            raise ValueError("finding artifact references must be sorted and unique")
        if self.finding_id != finding_record_id(
            category=self.category,
            target_trace=self.target_trace,
            provenance=self.provenance,
            evidence_reference_ids=evidence_ids,
        ):
            raise ValueError("finding ID is not deterministic")
        if len(_canonical_json(self.model_dump(mode="json"))) > _MAXIMUM_PRIVATE_RECORD_BYTES:
            raise ValueError("private finding record exceeds the 10 MB limit")
        return self


class FindingAnnotationInput(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    finding_id: str = Field(pattern=_FINDING_ID_PATTERN)
    status: FindingAnnotationStatus
    severity: FindingExportSeverity = "unrated"
    annotator_kind: FindingAnnotatorKind = "HUMAN"
    reviewer: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=4_000)
    reviewed_at: datetime
    supersedes_annotation_id: str | None = Field(default=None, pattern=_ANNOTATION_ID_PATTERN)

    @model_validator(mode="after")
    def validate_annotation(self) -> Self:
        _validate_utc(self.reviewed_at, "reviewed_at")
        if not self.reviewer.strip() or not self.reason.strip():
            raise ValueError("annotation reviewer and reason must contain non-whitespace text")
        if self.status != "confirmed" and self.severity != "unrated":
            raise ValueError("only confirmed annotations can have rated severity")
        return self


class FindingAnnotation(FindingAnnotationInput):
    annotation_id: str = Field(pattern=_ANNOTATION_ID_PATTERN)
    finding_record_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_annotation_id(self) -> Self:
        if self.annotation_id != finding_annotation_id(
            cast(FindingAnnotationInput, self), self.finding_record_sha256
        ):
            raise ValueError("finding annotation ID is not deterministic")
        return self


class FindingBundle(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    bundle_id: str = Field(pattern=_BUNDLE_ID_PATTERN)
    created_at: datetime
    findings: tuple[FindingRecord, ...] = Field(min_length=1, max_length=10_000)
    annotations: tuple[FindingAnnotation, ...] = Field(default=(), max_length=10_000)

    @model_validator(mode="after")
    def validate_bundle(self) -> Self:
        _validate_utc(self.created_at, "created_at")
        finding_ids = tuple(finding.finding_id for finding in self.findings)
        if finding_ids != tuple(sorted(set(finding_ids))):
            raise ValueError("bundle finding IDs must be sorted and unique")
        if self.bundle_id != finding_bundle_id(finding_ids):
            raise ValueError("finding bundle ID is not deterministic")
        findings = {finding.finding_id: finding for finding in self.findings}
        active_annotations: dict[str, FindingAnnotation] = {}
        annotations_by_id: dict[str, FindingAnnotation] = {}
        annotation_order = tuple(
            (annotation.reviewed_at, annotation.annotation_id) for annotation in self.annotations
        )
        if annotation_order != tuple(sorted(annotation_order)):
            raise ValueError("finding annotations must be ordered by time and ID")
        for annotation in self.annotations:
            finding = findings.get(annotation.finding_id)
            if finding is None:
                raise ValueError("finding annotation references an unknown finding")
            if annotation.finding_record_sha256 != finding_record_sha256(finding):
                raise ValueError("finding annotation evidence digest does not match")
            if annotation.annotation_id in annotations_by_id:
                raise ValueError("finding bundle contains a duplicate annotation ID")
            active = active_annotations.get(annotation.finding_id)
            if annotation.supersedes_annotation_id is None:
                if active is not None:
                    raise ValueError("finding has multiple active annotations")
            elif active is None or annotation.supersedes_annotation_id != active.annotation_id:
                raise ValueError("finding annotation supersession target is not active")
            annotations_by_id[annotation.annotation_id] = annotation
            active_annotations[annotation.finding_id] = annotation
        return self


class SafeFindingRecord(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    finding_id: str = Field(pattern=_FINDING_ID_PATTERN)
    conclusion: FindingConclusion
    category: str = Field(pattern=_CATEGORY_PATTERN)
    review_status: FindingReviewStatus
    severity: FindingExportSeverity
    evidence_facts: tuple[EvidenceFact, ...] = Field(min_length=1)
    evidence_authorities: tuple[EvidenceAuthority, ...] = Field(min_length=1)
    target_trace: W3CTraceReference
    evidence_reference_ids: tuple[str, ...] = Field(min_length=1)
    artifact_reference_ids: tuple[str, ...] = ()
    recorded_at: datetime


class SafeFindingAnnotation(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    annotation_id: str = Field(pattern=_ANNOTATION_ID_PATTERN)
    finding_id: str = Field(pattern=_FINDING_ID_PATTERN)
    status: FindingAnnotationStatus
    severity: FindingExportSeverity
    annotator_kind: FindingAnnotatorKind
    reviewed_at: datetime


class SafeFindingBundle(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    bundle_id: str = Field(pattern=_SAFE_BUNDLE_ID_PATTERN)
    source_bundle_id: str = Field(pattern=_BUNDLE_ID_PATTERN)
    created_at: datetime
    findings: tuple[SafeFindingRecord, ...] = Field(min_length=1, max_length=10_000)
    annotations: tuple[SafeFindingAnnotation, ...] = Field(default=(), max_length=10_000)

    @model_validator(mode="after")
    def validate_bundle_id(self) -> Self:
        expected = safe_finding_bundle_id(self.findings, self.annotations)
        if self.bundle_id != expected:
            raise ValueError("safe finding bundle ID is not deterministic")
        return self


class FindingOtlpEvent(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    finding_id: str = Field(pattern=_FINDING_ID_PATTERN)
    carrier_trace: W3CTraceReference
    target_trace: W3CTraceReference
    time_unix_nano: int = Field(ge=0)
    conclusion: FindingConclusion
    category: str = Field(pattern=_CATEGORY_PATTERN)
    review_status: FindingReviewStatus
    severity: FindingExportSeverity
    evidence_facts: tuple[EvidenceFact, ...] = Field(min_length=1)
    evidence_authorities: tuple[EvidenceAuthority, ...] = Field(min_length=1)
    evidence_reference_ids: tuple[str, ...] = Field(min_length=1)
    artifact_reference_ids: tuple[str, ...] = ()


def finding_evidence_reference_id(
    *,
    kind: FindingReferenceKind,
    source_id: str,
    authority: EvidenceAuthority,
    sha256: str,
    locator: str | None,
) -> str:
    return _digest(
        "ule_v1_",
        {
            "kind": kind,
            "source_id": source_id,
            "authority": authority,
            "sha256": sha256,
            "locator": locator,
        },
    )


def finding_artifact_reference_id(
    *, kind: FindingArtifactKind, media_type: str, sha256: str, locator: str | None
) -> str:
    return _digest(
        "ular_v1_",
        {"kind": kind, "media_type": media_type, "sha256": sha256, "locator": locator},
    )


def finding_record_id(
    *,
    category: str,
    target_trace: W3CTraceReference,
    provenance: FindingProvenance,
    evidence_reference_ids: tuple[str, ...],
) -> str:
    return _digest(
        "ulf_export_v1_",
        {
            "category": category,
            "target_trace": target_trace.model_dump(mode="json"),
            "source_finding_id": provenance.source_finding_id,
            "campaign_id": provenance.campaign_id,
            "case_id": provenance.case_id,
            "source_interaction_id": provenance.source_interaction_id,
            "probe_id": provenance.probe_id,
            "attempt_id": provenance.attempt_id,
            "variation_id": provenance.variation_id,
            "repetition": provenance.repetition,
            "evidence_reference_ids": evidence_reference_ids,
        },
    )


def finding_record_sha256(finding: FindingRecord) -> str:
    return hashlib.sha256(_canonical_json(finding.model_dump(mode="json"))).hexdigest()


def finding_annotation_id(annotation: FindingAnnotationInput, finding_record_digest: str) -> str:
    payload = annotation.model_dump(mode="json", exclude={"annotation_id", "finding_record_sha256"})
    return _digest(
        "ulann_v1_",
        {"annotation": payload, "finding_record_sha256": finding_record_digest},
    )


def finding_bundle_id(finding_ids: tuple[str, ...]) -> str:
    return _digest("ulfb_v1_", {"finding_ids": finding_ids})


def safe_finding_bundle_id(
    findings: tuple[SafeFindingRecord, ...], annotations: tuple[SafeFindingAnnotation, ...]
) -> str:
    return _digest(
        "ulfs_v1_",
        {
            "findings": [finding.model_dump(mode="json") for finding in findings],
            "annotations": [annotation.model_dump(mode="json") for annotation in annotations],
        },
    )
