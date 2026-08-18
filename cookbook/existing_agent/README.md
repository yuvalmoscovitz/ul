# Connect an existing Python agent

UL wraps the callable your test harness already exposes. It does not require a new agent class,
request format, or return format. Save this as `ul_adapter.py` next to your agent:

```python
from ul import SafetyEnvelope, callable_target_factory

from my_product import agent

create_target = callable_target_factory(
    agent.invoke,
    reset=agent.reset_test_state,
    snapshot=lambda result: agent.snapshot(),
    safety_envelope=SafetyEnvelope(
        description="Disposable agent test environment",
        isolated=True,
        allows_network_egress=False,
        allows_business_side_effects=False,
    ),
    fresh_state_per_execution=True,
)
```

Those are the only integration lines. `agent.invoke` may be synchronous or asynchronous; its
existing non-null JSON-compatible return becomes UL's `raw_output` unchanged. `reset` runs before
every evaluation. The optional `snapshot` receives the original result and supplies committed state
for invariant checks. UL serializes reset, invoke, and snapshot so concurrent callers cannot overlap
one target's evaluation cycles. An optional `cleanup` hook can release resources when UL closes the
target.

The safety envelope and `fresh_state_per_execution=True` are explicit assertions about your test
harness. UL does not infer isolation. If the reset does not actually create fresh, side-effect-free
state, do not make those declarations and do not run the evaluation.

```bash
uv run ul dataset evaluate interactions.jsonl \
  --target-factory ul_adapter:create_target \
  --confirm-isolated-sandbox \
  --output results.jsonl
```

Add `--allow-target-network` only if the declared test harness needs network access. UL validates
and bounds the result and snapshot before recording them; hook failures are reported without their
exception details so secrets do not enter evidence.
