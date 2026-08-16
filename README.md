# UL

UL actively tests black-box AI agents for behavioral differences that could matter in
high-risk workflows. It starts with a real interaction, makes a realistic variation, and
replays both against the same isolated agent several times.

UL reports observed differences for human review. It does not decide which behavior is
correct, prove causality, or estimate a production failure rate.

## Quickstart

You need Python 3.12+, [`uv`](https://docs.astral.sh/uv/), and an OpenRouter API key.

```bash
git clone https://github.com/yuvalmoscovitz/ul.git
cd ul
uv sync
# Provide OPEN_ROUTER_API_KEY through your environment or secret manager.
export UL_DATASET_LIVE_CALLS=true
export UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING=true
uv run python -m examples.quickstart.run
```

Never put the key in a committed file or paste it into the command. The two `UL_DATASET_*`
variables are separate opt-ins to billed model calls and external data processing.

The command starts an intentionally defective agent on localhost, gives every request clean
synthetic state, and evaluates one synthetic accounts-payable interaction. A typical run finds
a `REPEATABLE DIFFERENCE — REVIEW`: the agent pays invoice `AC-100` for the original request but
pays `AC-101` for a naturally repeated-word variation. Generation and checking use models, so
the exact variation and result can differ between runs.

The quickstart also applies one deterministic customer rule to the raw structured target output:
the committed invoice reference must equal the requested invoice reference. The original trials
satisfy that rule and the defective variation violates it. The rule uses no model calls and stays
separate from UL's behavioral-difference finding.

The quickstart permits up to 6 calls to the local target and up to 10 semantic-model calls. The
quickstart explicitly requests `x-ai/grok-4.6` for semantic deconstruction, rendering, and
equivalence checking to improve consistency. This model may cost more, and OpenRouter or the
underlying provider can still behave differently between runs. The semantic provider receives
the synthetic historical input and output, generated variation, and replayed target responses.
Model usage may cost money. The local target performs no real payment or network action.

```bash
uv run python -m examples.quickstart.run --dry-run
```

Dry-run makes no target or model calls. It prints the dataset plan, destination, external-data
notice, and maximum call counts first.

The full evidence is written locally as JSONL. The quickstart exits `0` when it confirms both its
expected stable 3/3 `changed action value` finding and the customer-rule violation; any
unconfirmed or interrupted run exits nonzero.
For the underlying `ul dataset evaluate` command, exit `0` means no review finding or declared-rule
violation, `1` means an observed difference needs review or a declared rule was violated, and `2`
means the evaluation could not finish or a declared rule was not evaluable. Exit `1` is not a
general correctness judgment.

Review findings without making more model or target calls:

```bash
uv run ul dataset report PATH_TO_EVIDENCE.jsonl
uv run ul dataset review PATH_TO_EVIDENCE.jsonl FINDING_ID \
  --status confirmed \
  --severity high \
  --reviewer "payments-risk" \
  --reason "The variation committed payment for the wrong invoice."
uv run ul dataset report PATH_TO_EVIDENCE.jsonl
```

Reviews are appended to a separate `PATH_TO_EVIDENCE.reviews.jsonl` audit file; the evaluation
evidence is never rewritten. UL creates the sidecar with mode `0600` on Unix. On Windows it
inherits the parent directory's access controls, so store it in a directory restricted to the
review team. Available judgments are `confirmed` (the reviewer sees a problem in this context),
`expected` (a supported but acceptable difference), `unsupported` (the machine finding is not
supported), and `inconclusive` (the reviewer needs more context). These are human judgments, not
UL correctness labels. Correcting a judgment requires `--supersedes REVIEW_ID`, preserving the
earlier decision.

After a finding has an active `confirmed` review, save its exact variation and one or more
violated customer rules as a replayable regression case:

```bash
uv run ul regression save PATH_TO_EVIDENCE.jsonl FINDING_ID \
  --rule committed-invoice-matches-request \
  --target-config target.json \
  --output regressions/wrong-invoice.json \
  --confirm-versioned-input
```

`--confirm-versioned-input` is required because the case copies the exact raw input and literal
target-template values, which can contain sensitive data. UL does not automatically redact them:
changing the input or template could change the behavior being reproduced. Treat the case as
sensitive, inspect it before committing, and apply your own data-governance policy.

Replay the saved input and deterministic rules against a sandbox:

```bash
uv run ul regression replay regressions/wrong-invoice.json \
  --target-config target.json \
  --allow-target-network \
  --confirm-isolated-sandbox \
  --confirm-fresh-state \
  --max-target-calls 3 \
  --output tmp/wrong-invoice-replay.json
```

For a local HTTP sandbox, also pass `--allow-insecure-http`. Replay makes exactly the saved number
of target calls, refuses to start when that exceeds `--max-target-calls`, and makes no
semantic-model calls. It exits `0` when every selected rule is satisfied,
`1` when any selected rule is violated, and `2` when a rule or target call is inconclusive. A
passing replay means only that the saved customer expectations held in those trials; it does not
prove the agent is correct or that a failure was fixed.

The case contains a target configuration declared by the customer when the case is created. UL
cannot verify that it was the discovery target; its digest only binds future replay to that
declaration. UL never executes the embedded configuration. You must separately provide a trusted
target configuration, and its digest must match before any request is sent. Literal request-template
values and replay evidence can contain sensitive data. Keep cases and complete replay evidence as
versioned artifacts only when that matches your access-control and retention policy.

See [the quickstart details](examples/quickstart/README.md) for the expanded command and file
layout.

## Connect your own agent

Create a target description and adapt its nested request and response paths:

```bash
uv run ul dataset init target.json --url https://your-sandbox.example/execute
uv run ul dataset evaluate your-data.jsonl --target-config target.json --dry-run
```

The target must be an isolated sandbox that starts every request from the same clean state.
Use `headers_from_env` in the target file for credentials so secret values remain outside the
configuration. Dry-run validates the dataset and target mapping without making external calls.

To add customer-defined deterministic checks, provide a strict invariant file:

```json
{
  "schema_version": "1.0.0",
  "observation_source": "target_output",
  "observation_authority": "committed_state_snapshot",
  "rules": [
    {
      "type": "json_values_equal",
      "id": "committed-amount-matches-corrected-amount",
      "version": "1.0.0",
      "description": "The committed amount must equal the corrected amount.",
      "severity": "high",
      "left_pointer": "/committed_amount",
      "right_pointer": "/corrected_amount"
    }
  ]
}
```

Pass it with `--invariants invariants.json`. UL evaluates the rule locally for every executed
original and variation trial. Results are `satisfied`, `violated`, or `not_evaluable`; missing or
non-scalar values never silently satisfy a rule. A satisfied declared rule does not establish
that the agent is correct or safe beyond that rule. Non-integer JSON numbers and selected values
larger than 4 KiB are `not_evaluable` in this first rule type. Represent exact decimal values as
strings or integer minor units rather than binary JSON floats. `observation_authority` is the
customer's statement about what the target output represents; UL does not independently verify it.
