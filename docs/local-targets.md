# Local process targets

The SDK can actively probe an existing Python callable or an arbitrary executable without an HTTP
server. UL starts the target as a child process, keeps it alive across cases, and communicates over
a versioned JSON Lines protocol. The target runs outside the UL process, but it still runs with your
user's operating-system permissions. Only enable a target you trust.

## Python callable

Create a configuration with absolute paths:

```json
{
  "version": 1,
  "kind": "python_callable",
  "target_id": "invoice-agent-local",
  "working_directory": "/absolute/path/to/customer/project",
  "interpreter": "/absolute/path/to/customer/project/.venv/bin/python",
  "target": "invoice_agent:handle",
  "input_mode": "value",
  "environment_allowlist": [],
  "limits": {
    "startup_timeout_seconds": 10,
    "turn_timeout_seconds": 20,
    "total_execution_timeout_seconds": 300,
    "shutdown_timeout_seconds": 2,
    "max_executions": 100,
    "max_input_bytes": 1000000,
    "max_output_bytes": 1000000,
    "max_stderr_bytes": 64000
  }
}
```

The reference uses `module.path:callable.path`. The selected interpreter imports it only in the
child process. Sync and async callables are supported. With `input_mode: "value"`, the callable
receives the turn input. With `input_mode: "request"`, it receives:

```json
{
  "schema_version": "1.0.0",
  "case_id": "case-id",
  "session_id": "session-id",
  "probe_id": "probe-id",
  "turn": {
    "schema_version": "1.0.0",
    "id": "turn-id",
    "input": "candidate input",
    "metadata": {}
  },
  "context": {}
}
```

The case, session, probe, and turn identities are explicit transport fields. The context is forwarded
unchanged, including rich-case or trace context supplied by the campaign. Turn metadata remains
inside the bounded turn object.
A callable may return any JSON value as the response, or return a response plus self-reported
execution events:

```python
async def handle(value):
    return {
        "response": {"answer": value},
        "execution_events": [{"id": "tool-1", "kind": "agent.tool", "payload": {"name": "lookup"}}],
    }
```

## Validate, plan, and execute

Dry-run planning validates paths, executability, environment-variable availability, configuration,
and all resource bounds without importing the callable or starting the command:

```python
import asyncio

from ul import LocalTargetConnection, create_local_target_dry_run_plan, load_local_target_config
from ul.environment import evaluation_case_from_inputs

config = load_local_target_config("local-target.json")
print(create_local_target_dry_run_plan(config).model_dump_json(indent=2))


async def run():
    case = evaluation_case_from_inputs(
        case_id="local-case",
        raw_inputs=("Pay invoice AC-100.",),
        max_environment_api_calls=1,
        timeout_seconds=30,
    )
    async with LocalTargetConnection(
        config,
        customer_code_execution_confirmed=True,
    ) as target:
        evidence = await target.execute(case)
        print(evidence.model_dump_json(indent=2))


asyncio.run(run())
```

`customer_code_execution_confirmed=True` is deliberately required. Planning does not require the
confirmation because it never launches customer code.

Only variable names listed in `environment_allowlist` are copied from the parent environment. The
configuration stores names, not values. Missing allowlisted variables fail validation. No other
environment variables are inherited.

## Command worker protocol

A `command` target names an explicit argv array whose first item is an absolute executable path:

```json
{
  "version": 1,
  "kind": "command",
  "target_id": "rust-agent-local",
  "working_directory": "/absolute/path/to/agent",
  "argv": ["/absolute/path/to/agent/bin/worker", "--local-probe"],
  "environment_allowlist": []
}
```

UL never invokes a shell or interprets command text. The executable reads one UTF-8 JSON object per
line from stdin and writes exactly one response object per line to stdout. Logs belong on stderr.
Every message has `protocol_version: "1.0.0"`, `type`, and `request_id`.

The session is:

1. UL sends `start`; the worker replies with `ready` and a runtime object containing non-empty
   `name` and `version` strings.
2. UL sends `session_start` with `session_id`; the worker replies with `session_ready` and the same
   identifiers.
3. UL sends one or more `invoke` messages containing `session_id`, `turn`, and `context`. The worker
   replies with `result`, the same `request_id`, a JSON `response`, and an `execution_events` array.
4. UL sends `shutdown`; the worker replies with `shutdown_complete` and exits.

A worker may instead reply with `type: "error"`, the same `request_id`, and a stable `code`. UL
records a sanitized lifecycle failure; it does not expose worker stderr or private exception text.

Startup, each turn, aggregate active execution time, stdout, stderr, input, call count, and shutdown
are independently bounded. Timeout, cancellation, crash, malformed output, oversized output, or
stderr overflow terminates the entire child process tree before another worker is started. UL uses a
dedicated process group on POSIX and a kill-on-close Job Object on Windows. Evidence records the
target, configuration, executable and callable hashes, runtime, worker attempt, execution attempt,
response, and delivery certainty.
