# Composable state hooks

UL can probe a response-only agent without state hooks. Add a `CallbackStateEnvironment` when a
campaign must prove what the agent committed to a test fixture. The callback adapter keeps UL's
lifecycle identities, reset receipts, generation counter, ordering, evidence labels, and quarantine
behavior out of customer code.

Use only disposable test fixtures. Reset and cleanup callbacks are allowed to mutate that fixture;
snapshot callbacks should only read it.

```python
from ul import CallbackStateEnvironment, StateCallbackContext


def reset_orders(context: StateCallbackContext) -> None:
    fixtures.reset(context.fixture_id)


async def snapshot_orders(context: StateCallbackContext):
    return await orders.read_fixture(context.fixture_id)


state = CallbackStateEnvironment(
    environment_id="orders-sandbox-observer",
    reset=reset_orders,
    snapshot=snapshot_orders,
    authority="independent_observer",
)
```

Callbacks may be synchronous or asynchronous. Cleanup defaults to the reset callback; pass a
separate `cleanup` callback when the fixture requires it. Optional setup runs after reset and before
the initial snapshot. Lifecycle callbacks return `None`; the snapshot callback returns JSON. UL
constructs the operation receipts and snapshot envelope.

Every callback receives a frozen `StateCallbackContext` with:

- `phase`: reset, setup, snapshot, or cleanup;
- `fixture_id`, `case_id`, `session_id`, `turn_id`, and `correlation_id`;
- UL's monotonically increasing local reset `generation`;
- the case's unchanged generic `probe_context` as `case_context`.

The conservative default authority is `environment_self_reported`. Set
`authority="independent_observer"` when the callback directly inspects state outside the agent; the
environment ID is then the default observer ID.

## Normalize and compare snapshots

Declare volatile fields and unordered arrays by JSON Pointer. This is useful for timestamps,
generated IDs, and database reads whose ordering is not meaningful.

```python
from ul import (
    JsonStateNormalization,
    diff_json_states,
    json_state_digest,
)

normalization = JsonStateNormalization(
    volatile_json_pointers=("/updated_at", "/request_id"),
    unordered_json_pointers=("/orders",),
)

state = CallbackStateEnvironment(
    environment_id="orders-sandbox-observer",
    reset=reset_orders,
    snapshot=snapshot_orders,
    authority="independent_observer",
    normalization=normalization,
)

digest = json_state_digest(snapshot, normalization)
differences = diff_json_states(before, after, normalization)
```

Normalized snapshots are stored in execution evidence. Each dataset output also receives
`committed_state_before_turn`, `committed_state_snapshot`, and a bounded deterministic
`committed_state_diff`, so wrong-record, duplicate-write, missing-write, and collateral-change
invariants can inspect the exact affected JSON paths.

Run the double-reset conformance check before composing the state adapter into a campaign:

```python
from ul import StateFixtureRequest, check_deterministic_reset

report = await check_deterministic_reset(
    state,
    StateFixtureRequest(
        fixture_id="orders-v1",
        case_id="conformance",
        session_id="conformance-session",
        correlation_id="conformance-reset",
    ),
)
assert report.deterministic
```

The adapter advertises deterministic replay only after this check produces identical normalized
clean-state digests.

## Compose a local observer with an HTTP agent

An isolated-response HTTP target can use the local callback adapter for state observation without
adding UL lifecycle endpoints to the agent server:

```python
from ul import JsonHttpEnvironmentConnection

environment = JsonHttpEnvironmentConnection.from_config(
    http_config,
    test_environment_confirmed=True,
    state_environment=state,
    state_fixture_id="orders-v1",
)
```

UL resets the local fixture, records the initial snapshot, invokes the HTTP agent, records a
correlated snapshot after the turn, and cleans the fixture. An exception, timeout, or uncertain
cleanup quarantines the composed environment, preventing another case from running against state
that may be dirty.

Without `state_environment`, the same HTTP adapter remains response-only. Its evidence profile does
not claim committed-state verification, and state-dependent cases are rejected without blocking
response-only cases.
