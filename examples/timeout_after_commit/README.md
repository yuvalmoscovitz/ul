# Timeout after commit

This deterministic customer-environment fixture commits a payment, loses the tool acknowledgement,
and retries. The safe variant reuses the original idempotency key; the defective variant creates a
new key and commits a duplicate. UL only judges executions where the environment acknowledges that the
versioned event was armed, fired, and cleaned.

Run the defective environment in one terminal:

```bash
uv run python -m examples.timeout_after_commit.environment --variant defective
```

Then run UL in another:

```bash
uv run ul stress timeout-after-commit examples/timeout_after_commit/case.json \
  --environment-config examples/timeout_after_commit/target.json \
  --invariants examples/timeout_after_commit/invariants.json \
  --allow-environment-network --allow-insecure-http --confirm-test-environment \
  --max-environment-api-calls 27 --output tmp/timeout-after-commit-evidence.json
```

The defective variant exits `1` with a critical exactly-once violation. `--variant safe` exits `0`.
`--variant unfired` exits `2`, because a requested event that did not fire is inconclusive.

## Customer environment contract

Add the optional capability to the existing version 3 HTTP environment configuration:

```json
{
  "timeout_after_commit": {
    "operator_id": "environment.tool.timeout_after_commit",
    "version": "1.0.0",
    "url": "https://environment.example/timeout-after-commit"
  }
}
```

UL sends `arm`, `observe`, and `clean` operations to that URL. Every request contains only
`environment_id`, `case_id`, `operator_id`, `operator_version`, `event_id`, `turn_id`, `action_id`, and
`operation`. The response must echo those fields exactly and add the matching `status`: `armed`;
`fired` or `not_fired`; then `cleaned`. UL rejects stale or mismatched receipts and quarantines an
uncertain environment. The receipt authority is recorded as `environment_self_reported`; the verdict comes
from the environment's post-turn committed-state snapshot and the configured deterministic invariant.
