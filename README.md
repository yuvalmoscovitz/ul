# UL

[![CI](https://github.com/yuvalmoscovitz/ul/actions/workflows/ci.yml/badge.svg)](https://github.com/yuvalmoscovitz/ul/actions/workflows/ci.yml)

UL is early-stage; APIs and evidence schemas may change. [Contributing](CONTRIBUTING.md) ·
[Security](SECURITY.md) · [MIT License](LICENSE)

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
export UL_LIVE=true
uv run python -m examples.quickstart.run
```

Never put the key in a committed file or paste it into the command. `UL_LIVE=true` is the
convenience setting for a local run: it enables both billed semantic-model calls and external
processing of the selected data. For separate policy control, use
`UL_DATASET_LIVE_CALLS=true` and `UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING=true` instead.
Either granular variable takes precedence when explicitly set, including `false`.

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

Resume an interrupted evaluation by passing its existing evidence file with the identical
dataset selection, operators, repetitions, target mapping, invariant suite, and semantic-model
configuration:

```bash
uv run ul dataset evaluate interactions.jsonl \
  --target-config target.json \
  --resume results.jsonl \
  --dry-run
```

The preflight validates the saved run context and reports completed and remaining interactions.
Execution appends only after the compatibility check succeeds. Older evidence without run-context
metadata, changed inputs, or changed evaluation semantics are rejected; call-budget and credential
changes remain allowed because they authorize execution rather than change its meaning.

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

`--confirm-versioned-input` is required because the case copies the exact raw input, literal
target-template values, and selected customer-rule definitions. Rule literals and allowed sets
can contain sensitive data. UL does not automatically redact these values: changing them could
change the behavior being reproduced. Treat the case as sensitive, inspect it before committing,
and apply your own data-governance policy.

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

Replay a directory of saved cases against the current black-box target:

```bash
uv run ul regression run regressions/ \
  --target-config target.json \
  --allow-target-network \
  --confirm-isolated-sandbox \
  --confirm-fresh-state \
  --max-target-calls 100 \
  --output tmp/regression-run.json
```

The runner reads every immediate `.json` file in the directory in filename order, validates the
complete suite and its total call budget before resolving target credentials, and then executes
all cases sequentially. It exits `0` when every case passes, `1` when any known failure is
observed, and `2` when no case fails but at least one is inconclusive. A failed case takes
precedence over an inconclusive case so a known recurrence is never hidden; the JSON artifact
retains every case result and its source filename. The artifact includes UTC start and completion
times and the trusted target-configuration digest. It does not claim that a behavioral change was
caused by a vendor deployment or model update.

Use the same command for manual, CI, and scheduled monitoring. For example, a cron job can run it
daily with a unique immutable output such as
`tmp/regression-run-$(date -u +\%Y\%m\%dT\%H\%M\%SZ).json`. A GitHub Actions workflow can use both
`workflow_dispatch` and `schedule` triggers:

```yaml
name: Scheduled UL regressions

on:
  workflow_dispatch:
  schedule:
    - cron: "17 3 * * *"

permissions:
  contents: read

concurrency:
  group: ul-regressions-production-sandbox
  cancel-in-progress: false

jobs:
  monitor:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
      - uses: astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d # v10.0.1
        with:
          version: "0.12.4"
          python-version: "3.12"
      - run: uv sync --locked
      - name: Run UL regressions
        env:
          TARGET_TOKEN: ${{ secrets.TARGET_TOKEN }}
        run: uv run --frozen ul regression run regressions/ --target-config target.json --allow-target-network --confirm-isolated-sandbox --confirm-fresh-state --max-target-calls 100 --output tmp/regression-run-${{ github.run_id }}.json
      - name: Upload UL evidence
        if: ${{ !cancelled() && vars.UL_UPLOAD_RAW_EVIDENCE == 'true' }}
        uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4.6.2
        with:
          name: ul-regression-${{ github.run_id }}
          path: tmp/regression-run-${{ github.run_id }}.json
          if-no-files-found: ignore
          retention-days: 1
```

Review and update the pinned action commits under your dependency policy. Raw-evidence upload is
disabled unless the repository variable `UL_UPLOAD_RAW_EVIDENCE` is set to `true`; enable it only
when GitHub Actions artifact access and retention meet your data policy. Otherwise send the safe
terminal summary to your alerting system and move raw evidence directly to approved encrypted
storage. Scheduled monitoring still requires a target that is isolated, has no real business
effects, and starts every request from equivalent fresh state. Regression run evidence contains
raw target outputs and may be sensitive; store and retain it according to the same policy as
single-case replay evidence.

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

Invariant schema `1.1.0` also supports literal values, allowed sets, and array uniqueness by a
customer-declared composite key:

```json
{
  "schema_version": "1.1.0",
  "observation_source": "target_output",
  "observation_authority": "committed_state_snapshot",
  "rules": [
    {
      "type": "json_value_equals_literal",
      "id": "approval-is-current",
      "version": "1.0.0",
      "description": "The approval version must be current.",
      "severity": "critical",
      "value_pointer": "/approval/version",
      "literal": 7
    },
    {
      "type": "json_value_in_allowed_set",
      "id": "action-is-allowed",
      "version": "1.0.0",
      "description": "The committed action must be explicitly allowed.",
      "severity": "critical",
      "value_pointer": "/action",
      "allowed_values": ["approved", "rejected", "needs_human_review"]
    },
    {
      "type": "json_array_items_unique_by",
      "id": "never-pay-twice",
      "version": "1.0.0",
      "description": "An invoice must not be paid twice from one account.",
      "severity": "critical",
      "array_pointer": "/payments",
      "key_pointers": ["/invoice_reference", "/source_bank_account_id"]
    }
  ]
}
```

Configured and observed rule values are limited to JSON strings, integers, booleans, or null.
Comparison is exact and type-sensitive: `true`, `1`, and `"1"` are different values. The
uniqueness rule checks 1–10 relative JSON pointers for each array item and reports only item counts,
pointer locations, and duplicate indices to the terminal; selected values remain in private evidence.
An empty or single-item array satisfies uniqueness but does not prove an action occurred. The rule
applies only within one declared target-output snapshot, not across independent customer requests.
UL also applies a fixed work budget across array rules and trials for each interaction. If their
combined item and pointer-processing work exceeds that budget, affected trials are `not_evaluable`
rather than silently passing or consuming unbounded CPU time.
