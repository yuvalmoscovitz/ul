# Clean-room probe qualification

These disposable fixtures reproduce the two response-only onboarding journeys used for UL-55.
They use one genuine observed interaction, make no production calls, and write private runtime
state only beneath `tmp/`. The documented HTTP bearer value is public fixture data, not a secret or
production credential; the automated test generates a fresh value for every run.

Install the locked project in a disposable environment:

```bash
mkdir -p tmp/ul-55
UV_PROJECT_ENVIRONMENT=tmp/ul-55/venv uv sync --frozen
export UL_QUALIFICATION_ROOT="$PWD"
mkdir -p tmp/ul-55/callable-run tmp/ul-55/http-run
```

For the remaining human timing gate, start a timer before the first command and record elapsed time
after UL prints the smoke result. The qualification report currently records the command count and
post-install automated runtime, not an unmeasured human setup time.

## Callable smoke

From the repository root, enter the disposable run directory and run:

```bash
cd "$UL_QUALIFICATION_ROOT/tmp/ul-55/callable-run"
export UL_QUALIFICATION_RECEIPT="$PWD/target-calls.jsonl"
"$UL_QUALIFICATION_ROOT/tmp/ul-55/venv/bin/ul" probe \
  "$UL_QUALIFICATION_ROOT/examples/probe_qualification/interactions.jsonl" \
  --target examples.probe_qualification.callable_agent:invoke \
  --target-working-directory "$UL_QUALIFICATION_ROOT" \
  --target-environment-variable UL_QUALIFICATION_RECEIPT \
  --limit 1
```

Confirm the dedicated target, inspect the smoke receipt and bounded campaign, then decline the
paid/network campaign. The receipt must contain exactly one line and UL must say that the smoke
succeeded with one target call and zero semantic-model calls.

## Authenticated synchronous HTTP smoke

Start the disposable endpoint in one terminal:

```bash
cd "$UL_QUALIFICATION_ROOT/tmp/ul-55/http-run"
export UL_ENVIRONMENT_AGENT_TOKEN='Bearer test'
export UL_QUALIFICATION_RECEIPT="$PWD/target-calls.jsonl"
"$UL_QUALIFICATION_ROOT/tmp/ul-55/venv/bin/python" \
  -m examples.probe_qualification.authenticated_agent --port 8765
```

In another terminal, export the repository root and the same public fixture value, then run:

```bash
export UL_QUALIFICATION_ROOT=/absolute/path/to/ul
export UL_ENVIRONMENT_AGENT_TOKEN='Bearer test'
cd "$UL_QUALIFICATION_ROOT/tmp/ul-55/http-run"
"$UL_QUALIFICATION_ROOT/tmp/ul-55/venv/bin/ul" probe \
  "$UL_QUALIFICATION_ROOT/examples/probe_qualification/interactions.jsonl" \
  --target http://127.0.0.1:8765/invoke \
  --header-from-env Authorization=UL_ENVIRONMENT_AGENT_TOKEN \
  --allow-insecure-http \
  --limit 1
```

Again, confirm the target and decline the campaign. The HTTP receipt must contain exactly one line.
The terminal output may contain the environment-variable name and a value digest, but never the
bearer value.

The fixture is intentionally response-only. Its report must show trajectory evidence and committed
state verification as unavailable rather than passed. See
[`docs/qualification/ul-55.md`](../../docs/qualification/ul-55.md) for the measured result and the
credential-dependent finding gate.
