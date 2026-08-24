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

Start with observed interactions and the test entry point your agent already exposes. Create
`interactions.jsonl` with one interaction per line:

```json
{"id":"case-1","input":"Return the status for ticket 42.","output":{"status":"open"}}
```

`output` is the response that was observed historically. UL uses it as reference evidence; it is
not assumed to be correct.

### Probe a Python callable

If your existing `agent.py` contains:

```python
def invoke(value):
    return {"response": value}
```

point UL at it directly:

```bash
ul probe interactions.jsonl --target agent:invoke
```

UL validates the dataset and target first. It then asks you to confirm the exact safe test target
before making one original smoke call. No semantic-model call happens before that smoke succeeds.

### Probe an authenticated JSON endpoint

For a synchronous endpoint that accepts `{"input": ...}` and returns `{"response": ...}`:

```bash
export UL_ENVIRONMENT_AGENT_TOKEN='Bearer secret-from-your-secret-manager'

ul probe interactions.jsonl \
  --target https://agent.test/invoke \
  --header-from-env Authorization=UL_ENVIRONMENT_AGENT_TOKEN
```

UL reads the secret from the named `UL_ENVIRONMENT_*` variable. It does not place the value in the
target configuration, evidence, diagnostics, or confirmation text. Never put credentials in the
URL. Direct endpoints must start each request from isolated test state and must not cause real-world
effects. For OpenAI-compatible chat endpoints and custom JSON shapes, see the
[guided probe reference](docs/probe.md).

### Augment and compare

After the smoke result, UL shows the target-call, semantic-call, token, time, repetition, and known
cost bounds for a small campaign. It runs only after a second confirmation. Configure the campaign
without changing integration paths:

```bash
export OPEN_ROUTER_API_KEY=YOUR_SECRET_FROM_A_SECRET_MANAGER
export UL_LIVE=true

ul probe interactions.jsonl \
  --target agent:invoke \
  --operator input.surface.typing_noise \
  --limit 10 \
  --repetitions 1 \
  --output .ul/runs/probe-evidence.jsonl
```

UL replays the original input against the current agent, invokes controlled variations, and
cross-examines the historical reference, fresh baseline, and variation. A single repetition is
screening evidence. The report keeps response, trajectory, and committed-state conclusions
separate. A response-only target does not verify trajectory or committed state; unavailable evidence
is never shown as passed. Run `ul report .ul/runs/probe-evidence.jsonl` for the offline report.

See [Guided active-probe quickstart](docs/probe.md) for structured inputs, target presets, reusable
secret-free target configurations, stronger repetitions, resumability, and full evidence details.

## Advanced: stateful evidence projects

Use `ul init` and `ul run` after response-level value is visible when UL must reset a disposable
fixture and independently inspect committed state:

```bash
ul init interactions.jsonl \
  --environment-url https://your-test-environment.example \
  --fixture-id standard-account \
  --fixture-version v1 \
  --allow-environment-network \
  --confirm-test-environment

ul run --dry-run
ul run
ul report
```

This environment implements reset, execute-turn, and snapshot requests. It is an optional stronger
evidence tier, not a prerequisite for probing an agent. See [Design valid test cases](docs/test-cases.md),
[Composable state hooks](docs/state-hooks.md), [Local process targets](docs/local-targets.md), and
[Customer-defined evaluators](docs/evaluators.md) for advanced configuration.

Inspect available augmentations at any time:

```bash
ul augmentations list --mode dataset_variation
ul augmentations plan input.surface.typing_noise
```

## How it works

```text
recorded interaction
  → realistic input variation
  → fresh target calls for original and variation
  → comparison of responses and available evidence
  → evidence for human review
  → confirmed regression case
```

With the optional `stateful-lifecycle` tier, every repetition uses this lifecycle:

```text
reset → optional setup → initial snapshot → execute turn → snapshot → cleanup reset
```

UL only calls the configured customer-owned environment. The environment is responsible for isolation,
deterministic reset, and preventing real business effects.

## Review and regressions

`ul report` is offline. It makes no model or environment calls.
For canonical `.findings.jsonl` packages, its default human and JSON forms expose only bounded,
privacy-safe explanations and opaque evidence pointers. Raw inputs, responses, state, tool data,
and provenance remain inside private normalized receipts and require an explicit per-finding
disclosure command. That command also resolves opaque case, operator, and customer-rule references
to the private configured identities needed for investigation. The disclosure is capped before any
private output is printed.
Dataset finding sidecars also keep an adjacent private reference key so resumed runs retain one
privacy-safe campaign identity; preserve that key with the evidence bundle.

```bash
ul report
ul report PRIVATE_EVIDENCE.json --json

# Safe decision-ready explanations over canonical dataset or stateful finding packages.
ul report PRIVATE_EVIDENCE.json.findings.jsonl

# Explicitly disclose one finding's bounded private normalized receipts.
ul report PRIVATE_EVIDENCE.json.findings.jsonl \
  --show-sensitive-values \
  --finding FINDING_OCCURRENCE_ID

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
