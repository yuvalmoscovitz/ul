# Finding export

UL findings have two neutral, versioned export forms:

- `finding_otlp_json(bundle)` and `safe_finding_bundle_json/jsonl(bundle)` are safe by
  default. They contain operational identifiers, category, review state, severity, exact
  evidence facts and authorities, hashes, and W3C trace/span linkage. They omit finding
  content, prompts, secrets, raw traces, state, private metadata, source IDs, locators, and
  review explanations.
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

- The carrier has a deterministic W3C trace ID and span ID.
- Its single span link points to the agent span identified by `target_trace`.
- `openinference.span.kind=EVALUATOR` and flattened
  `evaluations.0.evaluation.*` attributes make the post-hoc evaluation discoverable by
  OpenInference-compatible tooling.
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

The wire choices follow the [OTLP specification](https://opentelemetry.io/docs/specs/otlp/),
[W3C Trace Context](https://www.w3.org/TR/trace-context/), the
[OpenTelemetry trace API](https://opentelemetry.io/docs/specs/otel/trace/api/), and the
[OpenInference annotation specification](https://github.com/Arize-ai/openinference/blob/main/spec/annotations.md).
