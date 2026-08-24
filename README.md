# UL

See the [augmentation library index](core/src/ul_core/augmentations/README.md) for every built-in
augmentation, its surface, controlled change, expected relation, and code location.

[![CI](https://github.com/yuvalmoscovitz/ul/actions/workflows/ci.yml/badge.svg)](https://github.com/yuvalmoscovitz/ul/actions/workflows/ci.yml)

UL finds consequential failures in high-risk AI agents. It takes recorded interactions, creates
realistic variations, runs them against a customer-provided environment, and produces evidence
for human review.

UL does not test production systems, decide which behavior is correct, prove causality, or estimate
a production failure rate.

> UL is early-stage. APIs and evidence schemas may change.

## Install

You need Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv tool install git+https://github.com/yuvalmoscovitz/ul.git
```

If `uv` reports that its tool directory is not on `PATH`, run `uv tool update-shell`.

## Try the demo

Run the first no-key product experience:

```bash
ul demo
```

The demo evaluates two fake customer requests with three common input augmentations: typing errors,
a frustrated tone, and a short message. It shows three repeatable behavior changes in the same
evidence format used by dataset evaluations. It makes no model or network calls, so no API key or
connected agent is required.

## Quickstart

Create a JSONL dataset with one interaction per line:

```json
{"id":"case-1","input":"Pay AC-100.","output":{"action":"payment_committed","invoice":"AC-100"}}
```

Configure UL once:

```bash
ul init interactions.jsonl \
  --environment-url https://your-environment.example \
  --fixture-id standard-account \
  --fixture-version v1 \
  --allow-environment-network \
  --confirm-test-environment
```

The generated `.ul/environment.json` uses UL's full `stateful-lifecycle` adapter. Your environment
implements reset, execute-turn, and snapshot requests. Reset asks separately for a fresh agent
session and a clean external environment; both are required by default. If you already have a custom
mapping, use `--environment-config environment.json` instead of `--environment-url`.
The fixture identity names the resettable business state used by the run. Change its version whenever
that state or setup logic changes. See [Design valid test cases](docs/test-cases.md).
For objective assertions, model-judged rubrics, pairwise preference, and human review in custom SDK
workflows, see [Customer-defined evaluators](docs/evaluators.md).
To probe a Python callable or an explicit local command without hosting an HTTP server, see
[Local process targets](docs/local-targets.md).
To add reset and authoritative committed-state inspection with ordinary Python callbacks, including
beside an existing response-only HTTP agent, see [Composable state hooks](docs/state-hooks.md).
For the shortest smoke-first journey from grounded examples to a bounded active probe, see
[Guided active-probe quickstart](docs/probe.md).

To connect an existing response-only JSON endpoint that starts every request from isolated state:

```bash
export UL_ENVIRONMENT_AGENT_TOKEN='Bearer replace-me'

ul init interactions.jsonl \
  --environment-url https://your-environment.example/v1/chat/completions \
  --adapter-tier isolated-response \
  --isolated-preset openai-chat \
  --agent-model your-test-model \
  --header-from-env Authorization=UL_ENVIRONMENT_AGENT_TOKEN \
  --allow-environment-network \
  --confirm-test-environment \
  --confirm-request-isolation \
  --confirm-safe-test-target
```

This translates the OpenAI-style request and response shape into UL's internal contract; no
UL-specific endpoint is required. `generic-json` is the default preset (`{"input":"{{input}}"}`
and `/response`). For another JSON shape, use `--request-json-template` and
`--response-json-pointer`. Header values come only from the named `UL_ENVIRONMENT_*` variables and
are never written to the config. The separate confirmations attest that this is a test target,
every request starts fresh and is isolated from every other request, and requests cannot cause
real-world effects. Without a separately composed state observer, UL records response evidence only
at this tier and rejects committed-state invariants, conversations, timeout-after-commit checks, and
other state-dependent stress tests. Use [composable state hooks](docs/state-hooks.md) to inspect a
local test fixture, or move to `stateful-lifecycle` when the HTTP environment owns state control.

The endpoint URL cannot contain credentials, a query string, or a fragment. Put credentials in
`--header-from-env`; use a custom adapter when the endpoint requires query parameters.

Verify the adapter before spending money on model calls:

```bash
ul environment check .ul/environment.json \
  --probe "Return environment health only; do not take action." \
  --allow-environment-network \
  --confirm-test-environment \
  --confirm-harmless-probe

ul run --dry-run
```

Then run and report:

```bash
export OPEN_ROUTER_API_KEY=YOUR_SECRET_FROM_A_SECRET_MANAGER
export UL_LIVE=true

ul run
ul report
```

`ul init` stores private project settings under `.ul/`. `ul run` discovers that directory from the
current directory or a parent and writes a new evidence file. `ul report` opens the latest
reportable run. Use `ul run --resume` after an interruption.

One-run overrides do not change the saved project:

```bash
ul run --limit 3 --repetitions 3 --operator input.surface.rephrase
```

Inspect which built-in augmentations the current project can run:

```bash
ul augmentations enabled
ul augmentations list --mode dataset_variation
ul augmentations plan
ul augmentations plan input.surface.typing_noise
ul augmentations plan --json
```

The planner classifies every augmentation as ready, blocked, or manual and explains why. It checks
local configuration only; it makes no model, environment, or network calls and does not prove that
the configured services are reachable.

Configure the augmentations used by future `ul run` commands:

```bash
ul augmentations enable input.surface.typing_noise  # `add` is an alias
ul augmentations disable input.surface.rephrase     # `remove` is an alias
ul augmentations reset                              # restore recommended defaults
ul run --dry-run                                    # verify the saved selection and call budget
ul run --dry-run --json                             # machine-readable campaign plan
ul run --dry-run --show-sensitive-values            # include private saved candidate inputs
```

These commands update only the current project's private `.ul/config.json`. Enabling a currently
blocked augmentation still saves the selection and prints its required data and configuration steps.
Catalog entries without a dataset CLI binding cannot be enabled for `ul run`; use their focused
`ul augmentations plan ID` output to find the supported command or SDK path.

The run dry-run classifies every operator for each selected interaction, explains conditional or
ineligible cases, and separates baseline, variation, repetition, retry, evaluator, token, and
environment-call budgets. Planning makes no model or environment requests. When a resumable
augmentation ledger already contains a deterministic candidate, the candidate input is included for
inspection only with the explicit `--show-sensitive-values` opt-in; monetary estimates stay
unavailable unless trusted model pricing is configured.

During a run, successful identical semantic requests reuse a private in-memory cache bounded to 256
entries and 16 MiB of serialized responses, then cleared when the evaluator closes. Complete
evidence and terminal output report actual semantic provider calls separately from private cache
hits.

## How it works

```text
recorded interaction
  → realistic input variation
  → fresh environment runs for original and variation
  → comparison of responses and committed state
  → evidence for human review
  → confirmed regression case
```

With the default `stateful-lifecycle` tier, every repetition uses this lifecycle:

```text
reset → optional setup → initial snapshot → execute turn → snapshot → cleanup reset
```

UL only calls the configured customer-owned environment. The environment is responsible for isolation,
deterministic reset, and preventing real business effects.

## Review and regressions

`ul report` is offline. It makes no model or environment calls.

```bash
ul report
ul report PRIVATE_EVIDENCE.json --json

ul dataset review .ul/runs/EVIDENCE.jsonl FINDING_ID \
  --status confirmed \
  --severity high \
  --reviewer payments-risk \
  --reason "The variation committed payment for the wrong invoice."

# Preview an exact pattern snapshot without writing a decision.
ul dataset review-pattern .ul/runs/EVIDENCE.jsonl PATTERN_SNAPSHOT_ID

# Apply one decision to only that snapshot's still-unreviewed occurrences.
ul dataset review-pattern .ul/runs/EVIDENCE.jsonl PATTERN_SNAPSHOT_ID \
  --status confirmed \
  --severity high \
  --reviewer payments-risk \
  --reason "The reviewed occurrences show the same consequential failure."
```

With an explicit evidence path, `ul report` auto-detects dataset evaluation, correction,
retry-after-successful-commit, and timeout-after-commit evidence. Its default human summary and
versioned JSON omit inputs, responses, state, customer descriptions, and arbitrary evidence text.
For reviewable dataset findings, the report groups matching evidence into deterministic finding
patterns. Each pattern shows how many test questions are affected, which augmentations it was
observed under, its review queue, occurrence-level exceptions, and the exact underlying finding IDs.
Pattern decisions are append-only and bind the complete membership and evidence snapshot. Existing
occurrence decisions are unchanged exceptions, and later matching occurrences remain unreviewed;
an earlier pattern decision is context only. Patterns are
evidence-navigation aids, not correctness, causation, or root-cause claims. Inspect-only findings,
such as unstable behavior without a reviewable semantic difference, remain listed separately.
Use `ul dataset report EVIDENCE.jsonl` when you need the detailed private dataset review surface.
The private `.ul/review-history.key` authenticates pattern decisions independently of the rotatable
pattern identity key. Back it up with the project: pattern review history cannot be verified without
it.
Trace replay bundles are not supported by `ul report`.

Reviews are appended to a separate audit file. Evidence is never rewritten. The human report and
versioned JSON expose review workflow status (`review_status` in report schema `1.7.0`). Exit codes
map to that review status:

- `0` (`resolved`): no actionable finding remains; `expected` and `unsupported` reviews resolve a
  finding.
- `1` (`action_required`): a finding needs review, is confirmed, or an unreviewed declared rule was
  violated.
- `2` (`inconclusive`): the evaluation or a finding review is inconclusive and no actionable
  finding remains.

Review status is workflow state, not an agent correctness verdict.

Save selected confirmed findings as regressions one occurrence at a time. A pattern decision never
promotes every member automatically:

```bash
ul regression save EVIDENCE.jsonl FINDING_ID \
  --rule RULE_ID \
  --environment-config .ul/environment.json \
  --output regressions/finding.json \
  --confirm-versioned-input

ul regression run regressions \
  --environment-config .ul/environment.json \
  --allow-environment-network \
  --confirm-test-environment \
  --max-environment-api-calls 100 \
  --output regression-results.jsonl
```

## Configuration

Project defaults are saved by `ul init`. Common options include:

- `--invariants invariants.json` for deterministic customer rules.
- `--redaction-policy redaction.json --redaction-state STATE` for provider-boundary redaction.
- `--no-save-augmentations` when local augmentation retention is prohibited.
- `--allow-insecure-http` for an exact local HTTP environment.
- `--limit`, `--repetitions`, `--operator`, and `--max-environment-api-calls` for run scope.

### State-transition rules

Invariant schema `1.2.0` can check what changed between the snapshots immediately before and after
each agent turn:

```json
{
  "schema_version": "1.2.0",
  "observation_source": "target_output",
  "observation_authority": "committed_state_snapshot",
  "rules": [{
    "type": "exactly_one_new_effect",
    "id": "one-payment-per-turn",
    "version": "1.0.0",
    "description": "Each turn must append exactly one payment.",
    "severity": "critical",
    "before_checkpoint": "before_turn",
    "after_checkpoint": "after_turn",
    "observation_pointer": "/committed_effects"
  }]
}
```

Use `no_new_effect` when a turn must not append anything, or
`unchanged_between_checkpoints` when a value must stay unchanged. Effect arrays must be append-only;
rewrites, removals, and reordering are reported as not evaluable. See the runnable
[retry-after-commit rules](examples/retry_after_successful_commit/invariants.json).

See [Privacy and redaction](docs/privacy.md) for the policy schema and data-flow details.

OpenRouter is the default semantic-model provider. For a customer-controlled OpenAI-compatible
endpoint:

```bash
export UL_DATASET_SEMANTIC_PROVIDER=openai-compatible
export UL_DATASET_OPENAI_BASE_URL=https://models.example.com/v1
export UL_DATASET_OPENAI_API_KEY=YOUR_SECRET_FROM_A_SECRET_MANAGER
export UL_DATASET_MODEL=your-semantic-model
export UL_LIVE=true
```

Secrets belong in environment variables or a secret manager, never in datasets or configuration
files. Environment header mappings may reference only dedicated `UL_ENVIRONMENT_*` environment variables.

## Traces and stateful stress

Import OTLP traces without sending data to a model:

```bash
ul dataset ingest otlp traces.json --output interactions.jsonl
```

Create an approved replay bundle and stress a selected conversation:

```bash
ul dataset ingest otlp traces.json \
  --mapping examples/otlp_mapping.json \
  --replay-output trace-replay.json

ul stress trace trace-replay.json \
  --environment-config .ul/environment.json \
  --case-id CASE_ID \
  --dry-run
```

Run the highest-priority replay cases as one bounded campaign:

```bash
ul stress trace-replay-campaign trace-replay.json \
  --environment-config .ul/environment.json \
  --limit 10 \
  --max-environment-api-calls 100 \
  --dry-run
```

The dry run does not require environment credential values. It explains priority signals, shows
per-case and cumulative call budgets, and prints a copy-ready execution command without revealing
recorded content or making external calls. The private campaign result contains every case replay
plus deterministic groups for drift and inconclusive outcomes. This is a ranked reproducibility
check; it does not apply the suggested stress focuses or input augmentations. A completed campaign
exits `0` even when drift needs review, while an inconclusive campaign exits `2`.

Other stateful operators cover correction, retry-after-commit, and timeout-after-commit scenarios.
Run `ul stress --help` for the available commands.

## Local examples

From a source checkout:

```bash
uv sync

# No model key required.
uv run python -m examples.retry_after_successful_commit.run

# Full synthetic-agent example.
export OPEN_ROUTER_API_KEY=YOUR_SECRET_FROM_A_SECRET_MANAGER
export UL_LIVE=true
uv run python -m examples.quickstart.run
```

Useful examples:

- [Quickstart agent](examples/quickstart/README.md)
- [Retry after successful commit](examples/retry_after_successful_commit/README.md)
- [Timeout after commit](examples/timeout_after_commit/README.md)
- [Multi-turn correction](examples/multiturn_correction/README.md)
- [Accounts payable](examples/accounts_payable/README.md)

## Development

```bash
uv sync --locked
uv run --frozen ruff format --check .
uv run --frozen ruff check .
uv run --frozen pyright
uv run --frozen pytest -q
```

See [VISION.md](VISION.md), [Contributing](CONTRIBUTING.md), [Security](SECURITY.md), and the
[MIT License](LICENSE).
