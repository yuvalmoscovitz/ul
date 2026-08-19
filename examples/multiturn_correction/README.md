# Multi-turn correction failure

This deterministic sandbox accepts an invoice payment on the first turn, then acknowledges but
ignores a corrected invoice on the second turn. UL preserves both responses and committed-state
snapshots and reports the critical invariant violation.

From the repository root, run:

```bash
uv run python -m examples.multiturn_correction.run
```

The runner starts an ephemeral loopback-only sandbox, executes the real `ul stress correction`
path three times, retains private evidence under `tmp/`, and stops the sandbox. It requires no API
key and, after dependencies are installed, makes no semantic-provider or non-loopback sandbox
calls.

The wrapper exits `0` only when UL successfully demonstrates the seeded critical failure. The
underlying `ul stress correction` command retains its normal exit contract: `1` means it found a
rule violation. All three repetitions preserve the correction turn and show
`committed_invoice=AC-100` while `requested_invoice=AC-101`.
