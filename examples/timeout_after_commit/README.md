# Timeout after commit

This deterministic customer-sandbox fixture commits a payment, loses the tool acknowledgement,
and retries. The safe variant reuses the original idempotency key; the defective variant creates a
new key and commits a duplicate. UL only judges executions where the sandbox acknowledges that the
versioned event was armed, fired, and cleaned.

Run the defective sandbox in one terminal:

```bash
uv run python -m examples.timeout_after_commit.sandbox --variant defective
```

Then run UL in another:

```bash
uv run ul stress timeout-after-commit examples/timeout_after_commit/case.json \
  --sandbox-config examples/timeout_after_commit/target.json \
  --invariants examples/timeout_after_commit/invariants.json \
  --allow-sandbox-network-egress --allow-insecure-http --confirm-isolated-sandbox \
  --max-sandbox-api-calls 27 --output tmp/timeout-after-commit-evidence.json
```

The defective variant exits `1` with a critical exactly-once violation. `--variant safe` exits `0`.
`--variant unfired` exits `2`, because a requested event that did not fire is inconclusive.

## Customer sandbox contract

Add the optional capability to the existing version 3 HTTP sandbox configuration:

```json
{
  "timeout_after_commit": {
    "operator_id": "environment.tool.timeout_after_commit",
    "version": "1.0.0",
    "url": "https://sandbox.example/timeout-after-commit"
  }
}
```

UL sends `arm`, `observe`, and `clean` operations to that URL. Every request contains only
`sandbox_id`, `case_id`, `operator_id`, `operator_version`, `event_id`, `turn_id`, `action_id`, and
`operation`. The response must echo those fields exactly and add the matching `status`: `armed`;
`fired` or `not_fired`; then `cleaned`. UL rejects stale or mismatched receipts and quarantines an
uncertain sandbox. The receipt authority is recorded as `sandbox_self_reported`; the verdict comes
from the sandbox's post-turn committed-state snapshot and the configured deterministic invariant.
