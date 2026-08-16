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

The full evidence is written locally as JSONL. The quickstart exits `0` when it confirms its
expected stable 3/3 `changed action value` finding; any unconfirmed or interrupted run exits
nonzero.
For the underlying `ul dataset evaluate` command, exit `0` means no review finding, `1` means an
observed difference needs review, and `2` means the evaluation could not finish. Exit `1` is not
a correctness judgment.

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
