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

### Optional deterministic outcome projection

When the target already returns a structured business result, add a versioned `outcome` contract
to its local or HTTP target configuration. Each selector is a required RFC 6901 JSON Pointer into
the target response:

For a runnable callable example, make `agent.py` return a structured result:

```python
def run(value):
    return {
        "result": {
            "action": "lookup_ticket",
            "status": "completed",
            "ticket_id": "42",
            "customer": {"email": "private@example.test"},
        }
    }
```

Then save and validate a portable, absolute-path configuration with `configure_target.py`:

```python
import json
import sys
from pathlib import Path

from ul import create_local_target_dry_run_plan, load_local_target_config

root = Path.cwd().resolve()
configuration = {
    "version": 1,
    "kind": "python_callable",
    "target_id": "ticket-agent-local",
    "working_directory": str(root),
    "interpreter": str(Path(sys.executable).resolve()),
    "target": "agent:run",
    "outcome": {
        "schema_version": "1.0.0",
        "complete_result": "/result",
        "private_json_pointers": ["/customer/email"],
    },
}
path = root / "projected-target.json"
path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
validated = load_local_target_config(path)
print(create_local_target_dry_run_plan(validated).model_dump_json(indent=2))
```

Run `uv run python configure_target.py`, inspect the validation plan, then use the discovered
configuration directly:

```bash
uv run ul probe interactions.jsonl --target projected-target.json
```

UL asks for target confirmation before importing or invoking the callable. This configuration path
works the same way on Windows, macOS, and Linux because Python resolves the platform-specific
interpreter and working-directory paths.

For named roles, selectors such as `action: "/result/action"` address the raw target response. The
normalized object is then `{"action": "lookup_ticket", ...}`, so private pointers address that
normalized object (for example `/resource_id`, not `/result/resource_id`). With
`complete_result: "/result"`, the selected `result` object becomes the normalized root, so
`/customer/email` addresses `result.customer.email`.

```json
{
  "outcome": {
    "schema_version": "1.0.0",
    "action": "/result/action",
    "status": "/result/status",
    "resource_id": "/result/order_id",
    "decision": "/result/decision",
    "amount": "/result/amount"
  }
}
```

The common roles are `action`, `status`, `resource_id`, `decision`, `amount`, and `effects`.
String roles must select non-empty strings, `amount` must select a finite number, and `effects`
must select an object or array. For a domain-specific object, use one named selector instead:

```json
{
  "outcome": {
    "schema_version": "1.0.0",
    "complete_result": "/result",
    "private_json_pointers": ["/customer/email", "/internal_note"]
  }
}
```

`complete_result` must select an object and cannot be combined with role selectors. Private
pointers address the normalized object and are replaced with `[PRIVATE]` in the public smoke
preview. The full normalized result remains private evidence. Projections are bounded to 64 KB and
do not run code or expressions.

## 3. Inspect the smoke proof

The first result includes a bounded structural summary and digest of the live raw target response,
response-only or response-and-state evidence level, available trajectory observations, and
state-summary availability. Without a state observer, UL labels committed state unverified. When
configured, UL validates the outcome projection here and prints its independently filtered,
target-reported normalized preview before loading semantic-provider settings. A missing
or type-invalid selector fails with `PROBE_OUTCOME_PROJECTION_INVALID`, the exact field and pointer,
and zero semantic-model calls. A successful smoke proves only that the selector matched that one
live response; UL applies it again to every later response and does not assume future response
shapes. Use `--show-smoke-response` only when private raw and normalized
response content is safe to print. Case, turn, and canonical request identities are printed without
request content.
Only after a
successful smoke does UL save private target/dataset bindings in `.ul/probe.json`.

UL then selects at most ten examples, recommends the low-risk
`input.surface.typing_noise` operator, and shows exact original/probe target calls, environment API
requests, maximum semantic calls, completion-token bound, monetary-estimate availability, one
repetition, and maximum active wall time. Declining the second confirmation stops with zero semantic
calls.

Raw target response, normalized result, trajectory/tool observations, and independently observed
state remain four separate evidence channels. A target-declared outcome is authoritative only as a
reported business result; it is never committed-state proof. Semantic deconstruction uses the
public-safe normalized view when present and retains its existing fallback for targets without a
projection. The private normalized evidence remains independently inspectable and is not used to
reintroduce prohibited fields into derived public summaries.

The campaign receipt binds the semantic provider and endpoint, model, render model, equivalence
model, input/output/token/response/time bounds, data policy, target receipt, and command-wide call,
wall-time, and cost status. Normal evidence records the same semantic settings plus dataset,
operator, and target-receipt provenance in `run_context`. The target receipt includes the canonical
projection definition and SHA-256 digest; `.ul/probe.json` also binds that digest, so changing the
projection is incompatible with the saved run configuration.

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
