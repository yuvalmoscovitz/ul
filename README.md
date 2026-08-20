# UL

[![CI](https://github.com/yuvalmoscovitz/ul/actions/workflows/ci.yml/badge.svg)](https://github.com/yuvalmoscovitz/ul/actions/workflows/ci.yml)

UL finds consequential failures in high-risk AI agents. It takes recorded interactions, creates
realistic variations, runs them against an isolated customer-owned sandbox, and produces evidence
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

## Quickstart

Create a JSONL dataset with one interaction per line:

```json
{"id":"case-1","input":"Pay AC-100.","output":{"action":"payment_committed","invoice":"AC-100"}}
```

Configure UL once:

```bash
ul init interactions.jsonl \
  --sandbox-url https://your-sandbox.example \
  --allow-sandbox-network-egress \
  --confirm-isolated-sandbox
```

The generated `.ul/sandbox.json` defines the adapter contract. Your isolated sandbox must implement
its reset, setup, execute-turn, and snapshot requests. If you already have a custom mapping, use
`--sandbox-config sandbox.json` instead of `--sandbox-url`.

Verify the adapter before spending money on model calls:

```bash
ul sandbox check .ul/sandbox.json \
  --probe "Return sandbox health only; do not take action." \
  --allow-sandbox-network-egress \
  --confirm-isolated-sandbox \
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
  → fresh sandbox runs for original and variation
  → comparison of responses and committed state
  → evidence for human review
  → confirmed regression case
```

Every repetition uses this lifecycle:

```text
reset → optional setup → initial snapshot → execute turn → snapshot → cleanup reset
```

UL only calls the configured customer-owned sandbox. The sandbox is responsible for isolation,
deterministic reset, and preventing real business effects.

## Review and regressions

`ul report` is offline. It makes no model or sandbox calls.

```bash
ul report

ul dataset review .ul/runs/EVIDENCE.jsonl FINDING_ID \
  --status confirmed \
  --severity high \
  --reviewer payments-risk \
  --reason "The variation committed payment for the wrong invoice."
```

Reviews are appended to a separate audit file. Evidence is never rewritten. Exit codes are:

- `0`: no review finding or declared-rule violation.
- `1`: a difference needs review or a declared rule was violated.
- `2`: the evaluation was incomplete or not evaluable.

Exit `1` is not a general correctness verdict.

Save a confirmed finding as a regression:

```bash
ul regression save EVIDENCE.jsonl FINDING_ID \
  --rule RULE_ID \
  --sandbox-config .ul/sandbox.json \
  --output regressions/finding.json \
  --confirm-versioned-input

ul regression run regressions \
  --sandbox-config .ul/sandbox.json \
  --allow-sandbox-network-egress \
  --confirm-isolated-sandbox \
  --max-sandbox-api-calls 100 \
  --output regression-results.jsonl
```

## Configuration

Project defaults are saved by `ul init`. Common options include:

- `--invariants invariants.json` for deterministic customer rules.
- `--redaction-policy redaction.json --redaction-state STATE` for provider-boundary redaction.
- `--no-save-augmentations` when local augmentation retention is prohibited.
- `--allow-insecure-http` for an exact local HTTP sandbox.
- `--limit`, `--repetitions`, `--operator`, and `--max-sandbox-api-calls` for run scope.

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
files. Sandbox header mappings may reference only dedicated `UL_SANDBOX_*` environment variables.

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
  --sandbox-config .ul/sandbox.json \
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
