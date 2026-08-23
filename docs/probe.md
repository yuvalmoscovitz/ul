# Guided active-probe quickstart

`ul probe` calls one original interaction against a real test target before it prepares or runs any
semantic evaluation. You can stop after that smoke result with no evidence run in progress and no
semantic-model calls.

Never point UL at production or at an agent that can cause real-world effects.

## 1. Provide grounded examples

Create `interactions.jsonl` with one to ten observed interactions:

```json
{"id":"case-1","input":"Return the status for ticket 42.","output":{"status":"open"}}
```

Historical output grounds the semantic comparison; it is not treated as the expected answer.

## 2. Point UL at a callable

If `agent.py` contains this existing agent entry point:

```python
def run(value):
    return {"answer": value}
```

run:

```bash
ul probe interactions.jsonl --target agent:run
```

UL first validates the dataset, interpreter, working directory, and target reference without
importing the callable or calling a semantic model. It displays a SHA-256 digest and asks you to
confirm that exact dedicated test target. The callable is then imported only in an isolated child
process for one original smoke call.

You can instead pass an existing local callable/command configuration from
[Local process targets](local-targets.md), or an existing UL HTTP environment configuration:

```bash
ul probe interactions.jsonl --target local-target.json
ul probe interactions.jsonl --target http-target.json
```

UL binds the resolved executable, the direct Python module and UL worker, allowlisted environment
value digests, and command arguments that resolve to files. Repeat `--target-artifact PATH` for
every transitive Python helper, command script, bundle, or other executable dependency that is not
resolved directly from the target declaration. UL validates every declared artifact immediately
before launch; it does not claim undeclared transitive dependencies are bound.

Plain local HTTP also requires `--allow-insecure-http`. HTTP is never the implicit default.

## 3. Inspect the smoke proof

The first result includes a bounded structural summary and digest of the live normalized target
response, response-only or response-and-state evidence level, available trajectory observations,
and state-summary availability. Use `--show-smoke-response` only when private response content is
safe to print. Case, turn, and canonical request identities are printed without request content.
Only after a
successful smoke does UL save private target/dataset bindings in `.ul/probe.json`.

UL then selects at most ten examples, recommends the low-risk
`input.surface.typing_noise` operator, and shows exact original/probe target calls, environment API
requests, maximum semantic calls, completion-token bound, monetary-estimate availability, one
repetition, and maximum active wall time. Declining the second confirmation stops with zero semantic
calls.

The campaign receipt binds the semantic provider and endpoint, model, render model, equivalence
model, input/output/token/response/time bounds, data policy, target receipt, and command-wide call,
wall-time, and cost status. Normal evidence records the same semantic settings plus dataset,
operator, and target-receipt provenance in `run_context`.

When no trusted provider pricing is configured, the confirmation says monetary cost is unknown and
unbounded. Confirming that receipt accepts this uncertainty; it is not a cost guarantee.

## 4. Run the bounded pilot

Configure the existing UL semantic provider only when you are ready for the displayed budget:

```bash
export OPEN_ROUTER_API_KEY=YOUR_SECRET_FROM_A_SECRET_MANAGER
export UL_LIVE=true

ul probe interactions.jsonl --target agent:run
```

Confirm the paid/network campaign prompt. The pilot runs one original and one accepted probe for
each of at most ten examples, with one repetition. It writes the normal private UL evidence JSONL
and prints the normal unified report. A hosted UL account is not required.

For automation, bind both confirmations to the exact digests printed by a prior dry review:

```bash
ul probe interactions.jsonl \
  --target agent:run \
  --confirm-target TARGET_CONFIRMATION_SHA256 \
  --confirm-paid-execution CAMPAIGN_CONFIRMATION_SHA256
```

Changing executable/module bytes, target configuration, semantic provider endpoint, data policy,
or the bounded campaign invalidates the corresponding digest.

After reviewing the pilot, use the copy-ready command printed by UL with `--confirmation-run` to
repeat every original/probe arm three times under a newly displayed budget and a new evidence path.

Failures always name their stage, stable reason code, safe explanation, one remediation, and whether
the target is safe to reuse. Terminal failures omit customer exceptions and private target output.
Use `--diagnostic-artifact PATH` only when you explicitly want a private local diagnostic file.
