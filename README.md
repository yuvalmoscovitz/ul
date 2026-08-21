# UL

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
  --allow-environment-network \
  --confirm-test-environment
```

The generated `.ul/environment.json` defines the adapter contract. Your environment must implement
its reset, execute-turn, and snapshot requests. Reset asks separately for a fresh agent session and
a clean external environment; both are required by default. If you already have a custom mapping, use
`--environment-config environment.json` instead of `--environment-url`.

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

## How it works

```text
recorded interaction
  → realistic input variation
  → fresh environment runs for original and variation
  → comparison of responses and committed state
  → evidence for human review
  → confirmed regression case
```

Every repetition uses this lifecycle:

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
```

With an explicit evidence path, `ul report` auto-detects dataset evaluation, correction,
retry-after-successful-commit, and timeout-after-commit evidence. Its default human summary and
versioned JSON omit inputs, responses, state, customer descriptions, and arbitrary evidence text.
Use `ul dataset report EVIDENCE.jsonl` when you need the detailed private dataset review surface.
Trace replay bundles are not supported by `ul report`.

Reviews are appended to a separate audit file. Evidence is never rewritten. The human report and
versioned JSON expose review workflow status (`review_status` in report schema `1.2.0`). Exit codes
map to that review status:

- `0` (`resolved`): no actionable finding remains; `expected` and `unsupported` reviews resolve a
  finding.
- `1` (`action_required`): a finding needs review, is confirmed, or an unreviewed declared rule was
  violated.
- `2` (`inconclusive`): the evaluation or a finding review is inconclusive and no actionable
  finding remains.

Review status is workflow state, not an agent correctness verdict.

Save a confirmed finding as a regression:

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
