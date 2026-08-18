# Quickstart example

This package is a complete but synthetic black-box evaluation:

- `dataset.jsonl` contains one historical accounts-payable interaction.
- `target.json` maps UL's reset, setup, execution, and committed-state snapshot calls to the local
  sandbox.
- `invariants.json` declares that the committed invoice must match the requested invoice in the
  structured target result.
- `defective_agent.py` is a resettable local HTTP sandbox with one seeded parsing defect.
- `run.py` starts that sandbox and runs UL.

Run from the repository root after installing dependencies and configuring either OpenRouter or
an OpenAI-compatible provider as shown in the [main README](../../README.md):

```bash
uv run python -m examples.quickstart.run
```

The runner chooses a free localhost port and changes only the four lifecycle URLs from the
checked-in `target.json`. The execution mapping describes this exchange:

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
    "invoice_reference": "AC-100",
    "requested_invoice_reference": "AC-100"
  }
}
```

UL applies `surface.disfluency_repeat`, replays the original and accepted variation three
times each, and compares their observed actions. The seeded agent mishandles a repeated word,
changing the committed invoice from `AC-100` to `AC-101`, so a typical model-generated variation
produces a stable 3/3 `changed action value` finding for review. This is deliberately a
behavioral finding, not a claim that UL established which response was correct.
Separately, the customer rule is satisfied by all three original trials and violated by all
three defective variation trials. That rule is a deterministic comparison of two declared JSON
fields and requires no additional model or target calls.

## Inspect before calling anything

```bash
uv run python -m examples.quickstart.run --dry-run
```

This sends zero requests. Internally, the live runner starts the server and invokes this CLI
shape with its private ephemeral target configuration:

```bash
uv run ul dataset evaluate examples/quickstart/dataset.jsonl \
  --target-config tmp/quickstart-.../target.json \
  --invariants examples/quickstart/invariants.json \
  --operator surface.disfluency_repeat \
  --limit 1 \
  --repetitions 3 \
  --max-target-calls 30 \
  --allow-target-network \
  --confirm-isolated-sandbox \
  --allow-insecure-http \
  --output tmp/quickstart-.../evidence.jsonl
```

The runner handles the server lifecycle and uses a new private output path, so it is the
recommended route. UL never overwrites an existing evidence file.

The successful runner prints the evidence path. Use it to inspect and record a human judgment:

```bash
uv run ul dataset report PATH_TO_EVIDENCE.jsonl
uv run ul dataset review PATH_TO_EVIDENCE.jsonl FINDING_ID \
  --status confirmed \
  --severity high \
  --reviewer "quickstart-reviewer" \
  --reason "The observed payment used a different invoice reference."
uv run ul dataset report PATH_TO_EVIDENCE.jsonl
```

The review command appends to a separate sidecar and leaves the original evidence byte-for-byte
unchanged. UL creates it with mode `0600` on Unix; on Windows it inherits the parent directory's
access controls, so use a directory restricted to the review team. The report keeps UL's machine
observation separate from the reviewer's contextual judgment.

If the original satisfies a declared invariant and the variation violates it, UL gives that rule
transition its own reviewable finding ID even when no semantic difference was found.

Invariant values remain hidden from terminal output by default. If they are needed to make the
review decision, use `ul dataset report PATH_TO_EVIDENCE.jsonl --show-sensitive-values --finding
FINDING_ID`. The all-or-none bounded output is limited to that finding, may contain secrets or PII,
and may be retained in terminal scrollback, CI output, or logs.

The bundled runner stops its ephemeral server and deletes its temporary target configuration
when it exits, so its evidence is intentionally not replayable afterward. In a customer workflow,
keep a persistent sandbox configuration and use it to preserve a confirmed finding as an exact
regression case:

```bash
uv run ul regression save PATH_TO_EVIDENCE.jsonl FINDING_ID \
  --rule committed-invoice-matches-request \
  --target-config PATH_TO_TARGET.json \
  --output regressions/quickstart-wrong-invoice.json \
  --confirm-versioned-input

uv run ul regression replay regressions/quickstart-wrong-invoice.json \
  --target-config PATH_TO_TARGET.json \
  --allow-target-network \
  --confirm-isolated-sandbox \
  --allow-insecure-http \
  --max-target-calls 15 \
  --output tmp/quickstart-replay.json
```

When saving the invariant-violation finding itself, omit `--rule`; UL selects the rule identified
by that finding automatically. Semantic findings continue to require an explicit `--rule`.

Saving requires the explicit confirmation because the case contains the exact raw variation and
literal target-template values, which may be sensitive. UL does not redact them automatically
because that could alter the reproduced behavior. The embedded target configuration is declared
by the customer when the case is created, not verified as the discovery target, and may contain
sensitive literal template values.
It is never executed; replay requires a separately trusted configuration with the same digest.
Complete replay evidence contains raw target outputs and may also be sensitive.

Replay performs the saved three trials directly against the target with no generation,
equivalence, or other semantic-model calls. The seeded defective target violates the rule and
therefore exits `1`. A target that returns matching requested and committed invoice references in
all three trials exits `0`. That outcome shows only that this saved customer rule held for this
case; it is not proof that the implementation is correct or that every related failure is fixed.

## Limitations

The example is intentionally small and deterministic on the target side. With OpenRouter,
variation generation, validation, and behavioral comparison explicitly request `x-ai/grok-4.6`;
an OpenAI-compatible provider uses its configured models. Model behavior may still vary, so the
finding is not guaranteed. The invariant result applies only to configured fields returned by the
sandbox's separate committed-state snapshot endpoint. UL validates the reset acknowledgement
contract but cannot independently prove that the sandbox erased every state store. Satisfying the
rule does not establish overall correctness or safety.
The behavioral result is evidence for human review, not a causal proof or production-rate
estimate. Do not point the command at a production system or any endpoint that can cause business
side effects.
