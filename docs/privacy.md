# Redaction and pseudonymization

UL can keep explicitly selected sensitive values outside the semantic model provider while
preserving the identity relationships needed to generate executable stress-test variations.

The boundary is opt-in and intentionally explicit. It does not attempt automatic PII detection.
Text rules use bounded literal matching, structured rules use RFC 6901 JSON pointers, and
unsupported selectors fail validation. Executable input supports only reversible pseudonymization. Provider-only
output and transformation-instruction context can also be removed or replaced.

```python
from pathlib import Path

from pydantic import SecretStr
from ul import (
    LocalPseudonymStore,
    RedactedSemanticPipeline,
    RedactionEngine,
    RedactionPolicy,
    RedactionRule,
)

policy = RedactionPolicy(
    rules=(
        RedactionRule(
            name="account",
            locations=("input", "output", "context"),
            literal="ACCT-12345678",
        ),
        RedactionRule(
            name="access_token",
            locations=("output",),
            selector="/authentication/token",
        ),
        RedactionRule(
            name="internal_context",
            locations=("output",),
            selector="/debug/internal_context",
            action="remove",
        ),
    )
)
store = LocalPseudonymStore(
    Path(".ul-private/pseudonyms.json"),
    SecretStr("load-at-least-32-bytes-from-your-secret-manager"),
)
boundary = RedactedSemanticPipeline(
    semantic_pipeline,
    RedactionEngine(policy, store),
)

coverage = boundary.dry_run(source_record)
protected_source = boundary.protect_record(source_record)
protected_sandbox = boundary.wrap_sandbox(sandbox_connection)
```

Pass `boundary` as the deconstructor, renderer, and equivalence verifier, pass
`protected_sandbox` to `DatasetEvaluationRunner`, and run the protected source record. All semantic
calls use placeholders, including repeated calls. The customer-managed sandbox API receives the
rehydrated input just before execution. Persisted results contain placeholders and a policy digest,
never the mapping.

The state directory must be private (`0700`) and the mapping file is written as `0600`. The file is
integrity-protected with the supplied key and updated under a process lock with atomic replacement.
UL fails closed for a missing or wrong key, permissive permissions, corrupted state, unknown
placeholders, overlapping matches, or provider output that mutates a protected placeholder. Keep
the key in a secret manager and the state file in governed local storage. The state file contains
reversible mappings and is sensitive by itself; the key protects deterministic identity and file
integrity rather than encrypting those mappings. Losing either prevents rehydration.

Dry-run coverage contains only the policy SHA-256, match counts, rule names, and JSON paths. It does
not write mappings or include selected values.

The CLI accepts the same policy as strict JSON:

```json
{
  "version": 1,
  "rules": [
    {
      "name": "account",
      "locations": ["input", "output", "context"],
      "selector": "$text",
      "literal": "ACCT-12345678",
      "action": "pseudonymize"
    },
    {
      "name": "debug_context",
      "locations": ["output"],
      "selector": "/debug/internal_context",
      "action": "remove"
    }
  ]
}
```

Set the key through the environment, never a command argument:

```console
export UL_DATASET_REDACTION_KEY="$(your-secret-manager read ul-redaction-key)"
ul dataset evaluate interactions.jsonl \
  --redaction-policy redaction.json \
  --redaction-state .ul-private/pseudonyms.json \
  --sandbox-config sandbox.json \
  --allow-sandbox-network-egress --confirm-isolated-sandbox \
  --output evidence.jsonl
```

`--dry-run` needs only `--redaction-policy`; it reports value-free coverage without reading a key
or writing mapping state. Execution and resume also require `--redaction-state` and the
`UL_DATASET_REDACTION_KEY` environment variable. The evidence run context contains the policy
digest and aggregate input/output coverage, never the key, selected values, state path, or mapping.

Dataset execution saves generated augmentations by default beside the evidence file as
`NAME.augmentations.jsonl`. This private ledger is written before sandbox execution and reused on
resume. It contains the effective raw or pseudonymized interaction plus derived semantic data; it
does not serialize configured semantic/sandbox authentication values or reversible pseudonym
mappings. Raw interaction text may itself contain secrets when redaction is absent or incomplete.
Mode `0600` limits filesystem access on Unix but does not encrypt or redact the contents. Use `--augmentations-output` to place
it in an approved retention location, or `--no-save-augmentations` when policy prohibits local
retention and repeated generation is acceptable.
