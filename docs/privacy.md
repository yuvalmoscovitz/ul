# Redaction and pseudonymization

UL can keep explicitly selected sensitive values outside the semantic model provider while
preserving the identity relationships needed to generate executable stress-test variations.

The boundary is opt-in and intentionally explicit. It does not attempt automatic PII detection.
Text rules use regular expressions, structured rules use RFC 6901 JSON pointers, and unsupported
selectors fail validation. Executable input supports only reversible pseudonymization. Provider-only
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
            pattern=r"ACCT-[0-9]{8}",
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
protected_target = boundary.wrap_target(dataset_target)
```

Pass `boundary` as the deconstructor, renderer, and equivalence verifier, pass
`protected_target` to `DatasetEvaluationRunner`, and run the protected source record. All semantic
calls use placeholders, including repeated calls. The target receives the rehydrated input just
before execution. Persisted results contain placeholders and a policy digest, never the mapping.

The state directory must be private (`0700`) and the mapping file is written as `0600`. The file is
integrity-protected with the supplied key and updated under a process lock with atomic replacement.
UL fails closed for a missing or wrong key, permissive permissions, corrupted state, unknown
placeholders, overlapping matches, or provider output that mutates a protected placeholder. Keep
the key in a secret manager and the state file in governed local storage. The state file contains
reversible mappings and is sensitive by itself; the key protects deterministic identity and file
integrity rather than encrypting those mappings. Losing either prevents rehydration.

Dry-run coverage contains only the policy SHA-256, match counts, rule names, and JSON paths. It does
not write mappings or include selected values.
