# Quickstart example

This package is a complete but synthetic black-box evaluation:

- `dataset.jsonl` contains one historical accounts-payable interaction.
- `target.json` maps UL's input into a nested vendor-style request and extracts `/result` from
  the nested response.
- `defective_agent.py` is a resettable local HTTP sandbox with one seeded parsing defect.
- `run.py` starts that sandbox and runs UL.

Run from the repository root after installing dependencies and setting the three environment
variables shown in the [main README](../../README.md):

```bash
uv run python -m examples.quickstart.run
```

The runner chooses a free localhost port and changes only the URL from the checked-in
`target.json`. The checked-in mapping describes this exchange:

```json
{
  "request": {"message": "Pay AC-100."},
  "settings": {"mode": "sandbox"}
}
```

```json
{
  "result": {
    "action": "payment_committed",
    "invoice_reference": "AC-100"
  }
}
```

UL applies `surface.disfluency_repeat`, replays the original and accepted variation three
times each, and compares their observed actions. The seeded agent mishandles a repeated word,
changing the committed invoice from `AC-100` to `AC-101`, so a typical model-generated variation
produces a stable 3/3 `changed action value` finding for review. This is deliberately a
behavioral finding, not a claim that UL established which response was correct.

## Inspect before calling anything

```bash
uv run python -m examples.quickstart.run --dry-run
```

This sends zero requests. Internally, the live runner starts the server and invokes this CLI
shape with its private ephemeral target configuration:

```bash
uv run ul dataset evaluate examples/quickstart/dataset.jsonl \
  --target-config tmp/quickstart-.../target.json \
  --operator surface.disfluency_repeat \
  --limit 1 \
  --repetitions 3 \
  --max-target-calls 6 \
  --allow-target-network \
  --confirm-isolated-sandbox \
  --confirm-fresh-state \
  --allow-insecure-http \
  --output tmp/quickstart-.../evidence.jsonl
```

The runner handles the server lifecycle and uses a new private output path, so it is the
recommended route. UL never overwrites an existing evidence file.

## Limitations

The example is intentionally small and deterministic on the target side. Variation generation,
validation, and behavioral comparison explicitly request `x-ai/grok-4.6` for
deconstruction, rendering, and equivalence checking. This may cost more and can improve
consistency, but OpenRouter and its underlying provider may still vary; the finding is not
guaranteed. The result is evidence for human review, not an invariant check, correctness
label, causal proof, or production-rate estimate. Do not point the command at a production
system or any endpoint
that can cause business side effects.
