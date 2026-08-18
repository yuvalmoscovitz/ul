# Multi-turn correction failure

This deterministic sandbox accepts an invoice payment on the first turn, then acknowledges but
ignores a corrected invoice on the second turn. UL preserves both responses and committed-state
snapshots and reports the critical invariant violation.

In one terminal:

```bash
uv run python -m examples.multiturn_correction.defective_agent
```

In another:

```bash
uv run ul stress correction examples/multiturn_correction/case.json \
  --target-config examples/multiturn_correction/target.json \
  --invariants examples/multiturn_correction/invariants.json \
  --allow-target-network --allow-insecure-http --confirm-isolated-sandbox \
  --max-target-calls 36 --output tmp/multiturn-correction-evidence.json
```

The expected exit code is `1`: all three repetitions preserve the correction turn and show
`committed_invoice=AC-100` while `requested_invoice=AC-101`.
