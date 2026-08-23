# Finding export

UL findings have two neutral, versioned export forms:

- `finding_otlp_json(bundle)` and `safe_finding_bundle_json/jsonl(bundle)` are safe by
  default. They contain the allowlisted campaign, case, probe, attempt, session, turn,
  variation, repetition, and fixture identifiers; category; review state; severity; exact
  evidence facts and authorities; hashes; and W3C trace/span linkage. They omit finding
  content, prompts, secrets, raw traces, state, private metadata, source IDs, locators, and
  review explanations. Operational ID fields must contain IDs, never private content.
- `private_finding_bundle_json/jsonl(bundle, private_export_confirmed=True)` preserves the
  complete validated record. Treat this output as sensitive evidence and store it only in an
  access-controlled location. The explicit confirmation prevents selecting it accidentally.

`observed_variance` means UL observed a material difference that still needs review. It does
not claim the agent is wrong. `confirmed_correctness_failure` is only valid with a confirmed
review. `FindingEvidenceLevel` records non-ordinal evidence facts such as
`response_observed`, `trajectory_observed`, `committed_state_verified`, and
`deterministic_replay_verified`, with the source and authority for every fact. Consumers
should not convert those facts into an invented confidence score.

## OTLP mapping

Each safe finding becomes an OTLP/HTTP JSON evaluator carrier span:

- The carrier has a deterministic W3C trace ID and span ID. An appended review emits a new
  carrier identity at the review time; it never changes the identity or timestamp of an
  already-exported carrier.
- Its single span link points to the agent span identified by `target_trace`.
- `openinference.span.kind=EVALUATOR` and flattened
  `evaluations.0.evaluation.*` attributes make the post-hoc evaluation discoverable by
  OpenInference-compatible tooling. The annotator kind is `HUMAN`, `LLM`, or `CODE` from the
  effective review state.
- The `gen_ai.evaluation.result` event and `underlayer.finding.*` attributes provide a
  generic OpenTelemetry mapping without using vendor fields.

The returned object is the OTLP `ExportTraceServiceRequest` JSON body. Post it to any
OTLP/HTTP JSON endpoint, normally `/v1/traces`:

```python
import httpx
from ul import finding_otlp_json

payload = finding_otlp_json(bundle)
response = httpx.post(
    "http://127.0.0.1:4318/v1/traces",
    headers={"Content-Type": "application/json"},
    json=payload,
    timeout=10,
)
response.raise_for_status()
```

The mapping is tested against UL's bounded OTLP receiver and an independent standard-library
OTLP/HTTP receiver. Collector or backend configuration remains outside this API; no vendor
client is required.

## Review annotation import

Import neutral annotation JSONL with `parse_finding_annotations_jsonl`, then call
`append_finding_annotations`. Each line is one versioned annotation object:

```json
{"schema_version":"1.0.0","finding_id":"ulf_export_v1_…","status":"confirmed","severity":"high","annotator_kind":"HUMAN","reviewer":"reviewer-id","reason":"Evidence reviewed","reviewed_at":"2026-08-23T07:01:00Z","supersedes_annotation_id":null}
```

Annotations bind to the SHA-256 digest of the full finding record. They append review history
without modifying evidence or changing the private bundle's stable ID. A later annotation
must name the currently active annotation in `supersedes_annotation_id` and have a later UTC
timestamp. Safe exports include the effective review decision but omit reviewer identity and
reason.

## Generic source mapping

To map an existing evaluator result, construct:

1. a `FindingEvidenceReference` for every immutable evidence object, using its content hash;
2. a `FindingEvidenceLevel` with only facts actually established and their real authorities;
3. a `FindingProvenance` containing stable campaign, case, probe, attempt, session, turn,
   variation, repetition, and fixture identities when available;
4. a `FindingRecord` linked to the evaluated agent span through `W3CTraceReference`.

Keep messages, expected/observed outputs, explanations, and raw evaluator payloads in
`private_payload` or separately hashed artifacts. Do not copy them into category, IDs, or
OTLP attributes. Existing evidence JSONL and report pipelines can map into this API later;
finding export does not change or depend on those artifact schemas.

This minimal example is directly constructible:

```python
from datetime import UTC, datetime

from ul import (
    FindingEvidenceLevel,
    FindingProvenance,
    W3CTraceReference,
    create_finding_bundle,
    create_finding_evidence_reference,
    create_finding_record,
    safe_finding_bundle_json,
)

recorded_at = datetime(2026, 8, 23, 7, 0, tzinfo=UTC)
evidence = create_finding_evidence_reference(
    kind="response",
    source_id="probe-invoker-1",
    authority="invoker_self_reported",
    sha256="a" * 64,
)
finding = create_finding_record(
    conclusion="observed_variance",
    category="changed_grounded_effect_argument",
    review_status="needs_review",
    severity="unrated",
    evidence_level=FindingEvidenceLevel(
        facts=("response_observed",),
        sources={"response_observed": "probe-invoker-1"},
        authorities={"response_observed": "invoker_self_reported"},
    ),
    target_trace=W3CTraceReference(trace_id="1" * 32, span_id="2" * 16),
    evidence_references=(evidence,),
    recorded_at=recorded_at,
    provenance=FindingProvenance(
        producer_name="my-exporter",
        producer_version="1.0.0",
        config_sha256="b" * 64,
        source_finding_id="finding-17",
        campaign_id="campaign-4",
        case_id="case-9",
        probe_id="probe-2",
        attempt_id="attempt-1",
        session_id="session-3",
        turn_ids=("turn-1",),
    ),
)
bundle = create_finding_bundle((finding,), created_at=recorded_at)
safe_json = safe_finding_bundle_json(bundle)
```

The wire choices follow the [OTLP specification](https://opentelemetry.io/docs/specs/otlp/),
[W3C Trace Context](https://www.w3.org/TR/trace-context/), the
[OpenTelemetry trace API](https://opentelemetry.io/docs/specs/otel/trace/api/), and the
[OpenInference annotation specification](https://github.com/Arize-ai/openinference/blob/main/spec/annotations.md).
