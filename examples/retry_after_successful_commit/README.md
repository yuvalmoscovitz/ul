# Retry after a successful commit

This deterministic, model-free example finds an exactly-once failure in a local isolated sandbox.
The agent commits a payment, receives an explicit retry after reporting success, then incorrectly
creates a second payment with a new idempotency key.

From the repository root:

```bash
uv run python -m examples.retry_after_successful_commit.run
```

The runner starts the defective sandbox on a free loopback port, runs
`conversation.retry_after_successful_commit@1.0.0` three times, and cleans up the server and temporary
connection configuration. A successful demonstration exits `0` after UL confirms that:

- the one-turn baseline has exactly one unique committed payment;
- the two-turn variation also has exactly one payment after its first successful checkpoint; and
- the explicit retry leaves two payments for the same invoice in all three repetitions.

The complete evidence is retained at the private path printed by the command. It contains raw
sandbox responses and committed-state snapshots. The sandbox is self-reported and synthetic; UL
does not claim this proves causality or a production failure rate.

To inspect the underlying CLI plan without making sandbox calls:

```bash
uv run ul stress retry-after-successful-commit \
  examples/retry_after_successful_commit/case.json \
  --sandbox-config examples/retry_after_successful_commit/target.json \
  --invariants examples/retry_after_successful_commit/invariants.json \
  --allow-insecure-http --dry-run
```
