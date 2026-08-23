# ruff: noqa: E501

from __future__ import annotations

import asyncio
import json
import os
import signal
import stat
import sys
import time
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue
from ul.environment import evaluation_case_from_inputs
from ul.local_target import (
    CommandTargetConfig,
    LocalTargetConnection,
    LocalTargetLimits,
    PythonCallableTargetConfig,
    _LocalTargetInvoker,  # pyright: ignore[reportPrivateUsage]
    _open_executable_identity,  # pyright: ignore[reportPrivateUsage]
    _supervised_target_command,  # pyright: ignore[reportPrivateUsage]
    create_local_target_dry_run_plan,
    load_local_target_config,
)
from ul.state_hooks import CallbackStateEnvironment, StateAdapterIdentity, StateCallbackContext
from ul_core.evaluation import ProbeRequest, ProbeTurn


def _case(case_id: str, *inputs: str, timeout_seconds: float = 2):
    return evaluation_case_from_inputs(
        case_id=case_id,
        raw_inputs=inputs,
        max_environment_api_calls=20,
        timeout_seconds=timeout_seconds,
    )


def _runtime_payload(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    return cast(dict[str, JsonValue], value)


def _string_payload(payload: dict[str, JsonValue], key: str) -> str:
    value = payload[key]
    assert isinstance(value, str)
    return value


def _integer_payload(payload: dict[str, JsonValue], key: str) -> int:
    value = payload[key]
    assert isinstance(value, int)
    return value


def _python_config(
    working_directory: Path,
    target: str,
    *,
    input_mode: str = "value",
    limits: LocalTargetLimits | None = None,
    environment_allowlist: tuple[str, ...] = (),
) -> PythonCallableTargetConfig:
    return PythonCallableTargetConfig.model_validate(
        {
            "target_id": "local-python",
            "working_directory": working_directory,
            "interpreter": Path(sys.executable),
            "target": target,
            "input_mode": input_mode,
            "environment_allowlist": environment_allowlist,
            "limits": (limits or LocalTargetLimits()).model_dump(),
        }
    )


def _write_python_target(tmp_path: Path) -> None:
    (tmp_path / "customer_agent.py").write_text(
        """
import asyncio
import os
import sys

invocations = 0

def sync_agent(value):
    global invocations
    invocations += 1
    return {"response": {"value": value, "invocations": invocations}, "execution_events": [
        {"id": f"customer-{invocations}", "kind": "customer.event", "payload": {"ok": True}}
    ]}

async def async_agent(value):
    await asyncio.sleep(0)
    return {"async": value}

def request_agent(request):
    return request

def environment_agent(value):
    return {"allowed": os.environ.get("YUV20_ALLOWED"), "secret": os.environ.get("YUV20_SECRET")}

async def slow_agent(value):
    await asyncio.sleep(10)
    return value

async def delayed_agent(value):
    await asyncio.sleep(0.08)
    return value

def noisy_agent(value):
    sys.stderr.write("x" * 10000)
    sys.stderr.flush()
    return value

def pid_agent(value):
    with open(value, "w", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    return slow_agent(value)
""".lstrip(),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_python_worker_runs_sync_async_and_reuses_process(tmp_path: Path) -> None:
    _write_python_target(tmp_path)
    sync_config = _python_config(tmp_path, "customer_agent:sync_agent")

    async with LocalTargetConnection(
        sync_config, customer_code_execution_confirmed=True
    ) as connection:
        first = await connection.execute(_case("case-1", "one"))
        second = await connection.execute(_case("case-2", "two"))

    assert first.final_response == {"value": "one", "invocations": 1}
    assert second.final_response == {"value": "two", "invocations": 2}
    first_runtime = _runtime_payload(first.execution_events[0].payload)
    second_runtime = _runtime_payload(second.execution_events[0].payload)
    assert _integer_payload(first_runtime, "worker_attempt") == 1
    assert _integer_payload(second_runtime, "worker_attempt") == 1
    assert _integer_payload(first_runtime, "execution_attempt") == 1
    assert _integer_payload(second_runtime, "execution_attempt") == 2
    assert len(_string_payload(first_runtime, "target_sha256")) == 64
    assert len(_string_payload(first_runtime, "config_sha256")) == 64
    assert len(_string_payload(first_runtime, "executable_sha256")) == 64
    assert _string_payload(first_runtime, "runtime_name")
    assert _string_payload(first_runtime, "runtime_version")
    assert first.execution_events[1].kind == "customer.event"

    async_config = _python_config(tmp_path, "customer_agent:async_agent")
    async with LocalTargetConnection(
        async_config, customer_code_execution_confirmed=True
    ) as connection:
        evidence = await connection.execute(_case("case-async", "hello"))

    assert evidence.final_response == {"async": "hello"}


@pytest.mark.asyncio
async def test_local_process_target_composes_with_callback_state_without_http(
    tmp_path: Path,
) -> None:
    _write_python_target(tmp_path)
    state: dict[str, JsonValue] = {}

    def reset(context: StateCallbackContext) -> None:
        state.clear()
        state["fixture_id"] = context.fixture_id
        state["status"] = "clean"

    state_environment = CallbackStateEnvironment(
        environment_id="local-state-observer",
        identity=StateAdapterIdentity(
            adapter_id="local-state-adapter",
            adapter_version="1.0.0",
            fixture_id="fixture-local",
            fixture_version="1",
        ),
        reset=reset,
        snapshot=lambda context: dict(state),
        authority="independent_observer",
    )
    case = _case("case-state", "hello").model_copy(
        update={
            "required_state_observation_authority": "independent_observer",
            "required_state_observer_id": "local-state-observer",
        }
    )

    async with LocalTargetConnection(
        _python_config(tmp_path, "customer_agent:sync_agent"),
        customer_code_execution_confirmed=True,
        state_environment=state_environment,
    ) as connection:
        evidence = await connection.execute(case)

    assert evidence.final_response == {"value": "hello", "invocations": 1}
    assert evidence.evidence_scope == "response_and_state"
    assert evidence.initial_state is not None
    assert evidence.initial_state.value == {"fixture_id": "fixture-local", "status": "clean"}
    assert evidence.final_state is not None
    assert evidence.final_state.authority == "independent_observer"


@pytest.mark.asyncio
async def test_ten_case_original_variation_campaign_uses_one_persistent_worker(
    tmp_path: Path,
) -> None:
    _write_python_target(tmp_path)
    config = _python_config(tmp_path, "customer_agent:sync_agent")
    connection = LocalTargetConnection(config, customer_code_execution_confirmed=True)
    inputs = tuple(f"{arm}-{index}" for index in range(1, 6) for arm in ("original", "variation"))

    try:
        evidence = tuple(
            [
                await connection.execute(_case(f"campaign-{index}", value))
                for index, value in enumerate(inputs, start=1)
            ]
        )
    finally:
        await connection.aclose()

    assert len(evidence) == 10
    assert all(item.lifecycle.terminal_status == "succeeded" for item in evidence)
    assert all(item.lifecycle.delivery == "certain" for item in evidence)
    assert [item.final_response for item in evidence] == [
        {"value": value, "invocations": index} for index, value in enumerate(inputs, start=1)
    ]
    runtime_payloads = [_runtime_payload(item.execution_events[0].payload) for item in evidence]
    assert {_string_payload(payload, "target_id") for payload in runtime_payloads} == {
        "local-python"
    }
    assert {_string_payload(payload, "target_kind") for payload in runtime_payloads} == {
        "python_callable"
    }
    assert {_integer_payload(payload, "worker_attempt") for payload in runtime_payloads} == {1}
    assert [_integer_payload(payload, "execution_attempt") for payload in runtime_payloads] == list(
        range(1, 11)
    )
    assert {_string_payload(payload, "config_sha256") for payload in runtime_payloads} == {
        connection.config_sha256
    }
    assert len({_string_payload(payload, "target_sha256") for payload in runtime_payloads}) == 1
    assert len({_string_payload(payload, "executable_sha256") for payload in runtime_payloads}) == 1
    assert (
        len(
            {
                (
                    _string_payload(payload, "runtime_name"),
                    _string_payload(payload, "runtime_version"),
                )
                for payload in runtime_payloads
            }
        )
        == 1
    )


@pytest.mark.asyncio
async def test_probe_request_context_crosses_process_boundary_unchanged(tmp_path: Path) -> None:
    _write_python_target(tmp_path)
    config = _python_config(tmp_path, "customer_agent:request_agent", input_mode="request")
    invoker = _LocalTargetInvoker(config)
    context: dict[str, JsonValue] = {
        "schema_version": "1.0.0",
        "source_interaction_id": "interaction-1",
        "inputs": {"message": "hello"},
        "context": [{"id": "prior-1", "role": "user", "content": "prior"}],
        "ul.campaign.id": "campaign-1",
        "traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
    }

    try:
        result = await invoker.invoke(
            ProbeRequest(
                case_id="case-context",
                session_id="session-context",
                correlation_id="correlation-context",
                turn=ProbeTurn(id="turn-context", input="hello", metadata={"source": "test"}),
                context=context,
            )
        )
    finally:
        await invoker.aclose()

    assert result.response == {
        "schema_version": "1.0.0",
        "case_id": "case-context",
        "session_id": "session-context",
        "probe_id": "correlation-context",
        "turn": {
            "schema_version": "1.0.0",
            "id": "turn-context",
            "input": "hello",
            "metadata": {"source": "test"},
        },
        "context": context,
    }


@pytest.mark.asyncio
async def test_request_mode_preserves_turn_and_generic_context_envelope(tmp_path: Path) -> None:
    _write_python_target(tmp_path)
    config = _python_config(
        tmp_path,
        "customer_agent:request_agent",
        input_mode="request",
    )

    case = _case("case-context", "hello")
    rich_context: dict[str, JsonValue] = {
        "schema_version": "1.0.0",
        "source_interaction_id": "interaction-1",
        "inputs": {"message": "hello", "customer": {"id": "cus-7"}},
        "context": [{"id": "prior-1", "role": "user", "content": "prior"}],
        "fixture": {"id": "accounts", "version": "2"},
    }
    case = case.model_copy(
        update={
            "turns": (case.turns[0].model_copy(update={"metadata": {"target": "message"}}),),
            "probe_context": rich_context,
        }
    )

    async with LocalTargetConnection(config, customer_code_execution_confirmed=True) as connection:
        evidence = await connection.execute(case)

    response = _runtime_payload(evidence.final_response)
    assert response["schema_version"] == "1.0.0"
    assert response["case_id"] == "case-context"
    assert _string_payload(response, "session_id").startswith("ul-session-")
    assert _string_payload(response, "probe_id").startswith("ul-probe-")
    assert response["turn"] == {
        "schema_version": "1.0.0",
        "id": "case-context:turn-1",
        "input": "hello",
        "metadata": {"target": "message"},
    }
    request_context = _runtime_payload(response["context"])
    assert {key: request_context[key] for key in rich_context} == rich_context
    assert request_context["ul.case.id"] == "case-context"
    assert request_context["ul.turn.id"] == "case-context:turn-1"
    assert request_context["ul.correlation.id"] == response["probe_id"]
    assert _string_payload(request_context, "traceparent").startswith("00-")
    assert "ul.campaign.id=" in _string_payload(request_context, "baggage")
    assert evidence.probe_identity is not None
    assert evidence.probe_identity.session_id == response["session_id"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable-script test")
@pytest.mark.asyncio
async def test_explicit_non_python_command_uses_same_persistent_protocol(tmp_path: Path) -> None:
    command = tmp_path / "command-worker"
    command.write_text(
        """#!/bin/sh
read -r line
printf '%s\n' '{"protocol_version":"1.0.0","type":"ready","request_id":"startup","runtime":{"name":"sh","version":"1"}}'
while read -r line; do
  request_id=$(printf '%s' "$line" | /usr/bin/sed -n 's/.*"request_id":"\\([^"]*\\)".*/\\1/p')
  case "$line" in
    *'"type":"session_start"'*)
      session_id=$(printf '%s' "$line" | /usr/bin/sed -n 's/.*"session_id":"\\([^"]*\\)".*/\\1/p')
      printf '{"protocol_version":"1.0.0","type":"session_ready","request_id":"%s","session_id":"%s"}\n' "$request_id" "$session_id"
      ;;
    *'"type":"invoke"'*)
      printf '{"protocol_version":"1.0.0","type":"result","request_id":"%s","response":{"transport":"command"},"execution_events":[]}\n' "$request_id"
      ;;
    *'"type":"shutdown"'*)
      printf '%s\n' '{"protocol_version":"1.0.0","type":"shutdown_complete","request_id":"shutdown"}'
      exit 0
      ;;
  esac
done
""",
        encoding="utf-8",
    )
    command.chmod(command.stat().st_mode | stat.S_IXUSR)
    config = CommandTargetConfig(
        target_id="local-command",
        working_directory=tmp_path,
        argv=(str(command),),
    )

    async with LocalTargetConnection(config, customer_code_execution_confirmed=True) as connection:
        evidence = await connection.execute(_case("case-command", "hello", "again"))

    assert evidence.lifecycle.terminal_status == "succeeded"
    assert evidence.final_response == {"transport": "command"}
    assert _runtime_payload(evidence.execution_events[0].payload)["target_kind"] == "command"


def test_dry_run_validates_and_never_imports_target(tmp_path: Path) -> None:
    marker = tmp_path / "imported"
    (tmp_path / "dangerous.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\ndef run(value): return value\n",
        encoding="utf-8",
    )
    config = _python_config(tmp_path, "dangerous:run")

    plan = create_local_target_dry_run_plan(config)

    assert plan.target_id == "local-python"
    assert plan.target_kind == "python_callable"
    assert plan.worker_command[0] == sys.executable
    assert plan.maximum_executions == 100
    assert plan.maximum_active_wall_seconds == 302
    assert not marker.exists()
    with pytest.raises(ValueError, match="trust confirmation"):
        LocalTargetConnection(config, customer_code_execution_confirmed=False)


def test_config_loader_rejects_duplicates_and_shell_like_executable(tmp_path: Path) -> None:
    config_path = tmp_path / "target.json"
    config_path.write_text(
        json.dumps(
            {
                "kind": "command",
                "target_id": "command",
                "working_directory": str(tmp_path),
                "argv": ["echo hello"],
            }
        ),
        encoding="utf-8",
    )
    config = load_local_target_config(config_path)
    with pytest.raises(ValueError, match="absolute file"):
        create_local_target_dry_run_plan(config)

    config_path.write_text('{"kind":"command","kind":"command"}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        load_local_target_config(config_path)


def test_config_loader_rejects_oversized_file_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "oversized.json"
    with config_path.open("wb") as stream:
        stream.truncate(1_000_001)

    def unexpected_read(descriptor: int, maximum_bytes: int) -> bytes:
        del descriptor, maximum_bytes
        raise AssertionError("oversized config must be rejected from descriptor metadata")

    monkeypatch.setattr(os, "read", unexpected_read)

    with pytest.raises(ValueError, match="exceeds the 1 MB limit"):
        load_local_target_config(config_path)


def test_config_loader_rejects_final_component_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    symlink = tmp_path / "config.json"
    try:
        symlink.symlink_to(target)
    except OSError:
        pytest.skip("test environment cannot create symbolic links")

    with pytest.raises(RuntimeError, match="could not be read"):
        load_local_target_config(symlink)


@pytest.mark.asyncio
async def test_allowlisted_environment_is_the_only_inherited_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_python_target(tmp_path)
    monkeypatch.setenv("YUV20_ALLOWED", "visible")
    monkeypatch.setenv("YUV20_SECRET", "hidden")
    config = _python_config(
        tmp_path,
        "customer_agent:environment_agent",
        environment_allowlist=("YUV20_ALLOWED",),
    )

    async with LocalTargetConnection(config, customer_code_execution_confirmed=True) as connection:
        evidence = await connection.execute(_case("case-env", "hello"))

    assert evidence.final_response == {"allowed": "visible", "secret": None}


@pytest.mark.parametrize(
    ("target", "limits", "expected_code"),
    [
        (
            "customer_agent:slow_agent",
            LocalTargetLimits(turn_timeout_seconds=0.05, shutdown_timeout_seconds=0.1),
            "response_timeout",
        ),
        (
            "customer_agent:noisy_agent",
            LocalTargetLimits(max_stderr_bytes=100, shutdown_timeout_seconds=0.1),
            "response_too_large",
        ),
    ],
)
@pytest.mark.asyncio
async def test_timeout_and_stderr_overflow_become_safe_failure_evidence(
    tmp_path: Path,
    target: str,
    limits: LocalTargetLimits,
    expected_code: str,
) -> None:
    _write_python_target(tmp_path)
    config = _python_config(tmp_path, target, limits=limits)

    async with LocalTargetConnection(config, customer_code_execution_confirmed=True) as connection:
        evidence = await connection.execute(_case("case-failure", "hello"))

    assert evidence.lifecycle.terminal_status == "failed"
    assert evidence.lifecycle.failure_code == expected_code
    assert "x" * 20 not in (evidence.lifecycle.failure_reason or "")


@pytest.mark.asyncio
async def test_call_budget_fails_without_starting_another_worker(tmp_path: Path) -> None:
    _write_python_target(tmp_path)
    config = _python_config(
        tmp_path,
        "customer_agent:sync_agent",
        limits=LocalTargetLimits(max_executions=1),
    )

    async with LocalTargetConnection(config, customer_code_execution_confirmed=True) as connection:
        first = await connection.execute(_case("case-first", "one"))
        second = await connection.execute(_case("case-second", "two"))

    assert first.lifecycle.terminal_status == "succeeded"
    assert second.lifecycle.failure_code == "call_budget"


@pytest.mark.asyncio
async def test_failed_invocation_still_consumes_call_budget(tmp_path: Path) -> None:
    _write_python_target(tmp_path)
    config = _python_config(
        tmp_path,
        "customer_agent:slow_agent",
        limits=LocalTargetLimits(
            max_executions=1,
            turn_timeout_seconds=0.05,
            shutdown_timeout_seconds=0.1,
        ),
    )

    async with LocalTargetConnection(config, customer_code_execution_confirmed=True) as connection:
        first = await connection.execute(_case("case-failed-first", "one"))
        second = await connection.execute(_case("case-budget-second", "two"))

    assert first.lifecycle.failure_code == "response_timeout"
    assert second.lifecycle.failure_code == "call_budget"


@pytest.mark.asyncio
async def test_total_active_execution_bound_spans_persistent_turns(tmp_path: Path) -> None:
    _write_python_target(tmp_path)
    config = _python_config(
        tmp_path,
        "customer_agent:delayed_agent",
        limits=LocalTargetLimits(
            turn_timeout_seconds=1,
            total_execution_timeout_seconds=0.3,
            shutdown_timeout_seconds=0.1,
        ),
    )

    async with LocalTargetConnection(config, customer_code_execution_confirmed=True) as connection:
        first = await connection.execute(_case("case-total-first", "one"))
        second = await connection.execute(_case("case-total-second", "two", "three", "four"))

    assert first.lifecycle.terminal_status == "succeeded"
    assert first.final_response == "one"
    assert second.lifecycle.terminal_status == "failed"
    assert second.lifecycle.failure_code == "response_timeout"


def _write_failure_worker(tmp_path: Path, behavior: str) -> Path:
    command = tmp_path / f"worker-{behavior}"
    invoke_behavior = {
        "crash": "exit 9",
        "malformed": "printf '%s\\n' 'not-json'",
        "oversized": "/usr/bin/printf '%010000d\\n' 0",
        "shutdown_hang": (
            'printf \'{"protocol_version":"1.0.0","type":"result",'
            '"request_id":"%s","response":"ok","execution_events":[]}\\n\' '
            '"$request_id"'
        ),
    }[behavior]
    shutdown_behavior = (
        "trap '' TERM; /bin/sleep 10"
        if behavior == "shutdown_hang"
        else 'printf \'%s\\n\' \'{"protocol_version":"1.0.0","type":"shutdown_complete","request_id":"shutdown"}\'; exit 0'
    )
    command.write_text(
        f"""#!/bin/sh
read -r line
printf '%s\n' '{{"protocol_version":"1.0.0","type":"ready","request_id":"startup","runtime":{{"name":"sh","version":"1"}}}}'
while read -r line; do
  request_id=$(printf '%s' "$line" | /usr/bin/sed -n 's/.*"request_id":"\\([^"]*\\)".*/\\1/p')
  case "$line" in
    *'"type":"session_start"'*)
      session_id=$(printf '%s' "$line" | /usr/bin/sed -n 's/.*"session_id":"\\([^"]*\\)".*/\\1/p')
      printf '{{"protocol_version":"1.0.0","type":"session_ready","request_id":"%s","session_id":"%s"}}\n' "$request_id" "$session_id"
      ;;
    *'"type":"invoke"'*) {invoke_behavior} ;;
    *'"type":"shutdown"'*) {shutdown_behavior} ;;
  esac
done
""",
        encoding="utf-8",
    )
    command.chmod(command.stat().st_mode | stat.S_IXUSR)
    return command


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable replacement")
@pytest.mark.asyncio
async def test_executable_identity_change_is_rejected_before_spawn(tmp_path: Path) -> None:
    command = _write_failure_worker(tmp_path, "shutdown_hang")
    config = CommandTargetConfig(
        target_id="identity-command",
        working_directory=tmp_path,
        argv=(str(command),),
    )
    connection = LocalTargetConnection(config, customer_code_execution_confirmed=True)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(command.read_bytes())
    replacement.chmod(command.stat().st_mode)
    os.replace(replacement, command)

    try:
        evidence = await connection.execute(_case("case-identity", "hello"))
    finally:
        await connection.aclose()

    assert evidence.lifecycle.terminal_status == "failed"
    assert evidence.lifecycle.failure_code == "environment_identity"
    assert evidence.lifecycle.delivery == "certain"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable mutation")
@pytest.mark.asyncio
async def test_executable_digest_is_revalidated_after_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = _write_failure_worker(tmp_path, "shutdown_hang")
    config = CommandTargetConfig(
        target_id="digest-command",
        working_directory=tmp_path,
        argv=(str(command),),
    )
    connection = LocalTargetConnection(config, customer_code_execution_confirmed=True)
    original_stat = command.stat()

    class _FakeProcess:
        returncode: int | None = None
        pid = 999_999

        async def wait(self) -> int:
            self.returncode = 1
            return 1

    async def mutate_during_spawn(*arguments: object, **keywords: object) -> _FakeProcess:
        del arguments, keywords
        with command.open("r+b") as stream:
            stream.write(b"X")
        os.utime(
            command,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        return _FakeProcess()

    def record_termination(process: object, *, force: bool) -> None:
        del process
        assert force is True

    monkeypatch.setattr(asyncio, "create_subprocess_exec", mutate_during_spawn)
    monkeypatch.setattr("ul.local_target._signal_process", record_termination)

    try:
        evidence = await connection.execute(_case("case-digest", "hello"))
    finally:
        await connection.aclose()

    assert evidence.lifecycle.failure_code == "environment_identity"
    assert evidence.lifecycle.delivery == "certain"


def test_windows_branch_wraps_target_in_kill_on_close_job_supervisor(tmp_path: Path) -> None:
    command = _write_failure_worker(tmp_path, "shutdown_hang")
    config = CommandTargetConfig(
        target_id="windows-command",
        working_directory=tmp_path,
        argv=(str(command), "argument"),
    )
    with _open_executable_identity(command) as (_, identity):
        supervised = _supervised_target_command(config, identity, platform="win32")

    separator = supervised.index("--")
    assert supervised[:3] == (
        sys.executable,
        "-u",
        str(Path(__file__).parents[1] / "src" / "ul" / "_windows_job_worker.py"),
    )
    assert supervised[3:separator] == (
        "--expected-executable-sha256",
        identity.sha256,
    )
    assert supervised[separator + 1 :] == (str(command.resolve()), "argument")
    supervisor_source = Path(supervised[2]).read_text(encoding="utf-8")
    assert "_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE" in supervisor_source
    assert "AssignProcessToJobObject" in supervisor_source
    assert "subprocess.Popen(command, shell=False" in supervisor_source


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable-script test")
@pytest.mark.parametrize(
    ("behavior", "expected_code"),
    [
        ("crash", "transport_failed"),
        ("malformed", "invalid_json"),
        ("oversized", "response_too_large"),
    ],
)
@pytest.mark.asyncio
async def test_command_failures_become_deterministic_safe_evidence(
    tmp_path: Path, behavior: str, expected_code: str
) -> None:
    command = _write_failure_worker(tmp_path, behavior)
    config = CommandTargetConfig(
        target_id="failing-command",
        working_directory=tmp_path,
        argv=(str(command),),
        limits=LocalTargetLimits(max_output_bytes=500, shutdown_timeout_seconds=0.1),
    )

    async with LocalTargetConnection(config, customer_code_execution_confirmed=True) as connection:
        evidence = await connection.execute(_case(f"case-{behavior}", "hello"))

    assert evidence.lifecycle.terminal_status == "failed"
    assert evidence.lifecycle.failure_code == expected_code
    assert evidence.lifecycle.failure_reason == "probe invocation failed"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable-script test")
@pytest.mark.asyncio
async def test_startup_and_shutdown_bounds_are_enforced(tmp_path: Path) -> None:
    slow_start = tmp_path / "slow-start"
    slow_start.write_text(
        "#!/bin/sh\n/bin/sleep 10\n",
        encoding="utf-8",
    )
    slow_start.chmod(slow_start.stat().st_mode | stat.S_IXUSR)
    startup_config = CommandTargetConfig(
        target_id="slow-start",
        working_directory=tmp_path,
        argv=(str(slow_start),),
        limits=LocalTargetLimits(startup_timeout_seconds=0.05, shutdown_timeout_seconds=0.1),
    )
    async with LocalTargetConnection(
        startup_config, customer_code_execution_confirmed=True
    ) as connection:
        startup_evidence = await connection.execute(_case("case-startup", "hello"))
    assert startup_evidence.lifecycle.failure_code == "response_timeout"

    shutdown_worker = _write_failure_worker(tmp_path, "shutdown_hang")
    shutdown_config = CommandTargetConfig(
        target_id="shutdown-hang",
        working_directory=tmp_path,
        argv=(str(shutdown_worker),),
        limits=LocalTargetLimits(shutdown_timeout_seconds=0.1),
    )
    connection = LocalTargetConnection(shutdown_config, customer_code_execution_confirmed=True)
    evidence = await connection.execute(_case("case-shutdown", "hello"))
    started_at = time.monotonic()
    await connection.aclose()

    assert evidence.lifecycle.terminal_status == "succeeded"
    assert time.monotonic() - started_at < 0.5


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process liveness assertion")
@pytest.mark.asyncio
async def test_cancellation_terminates_the_customer_process(tmp_path: Path) -> None:
    _write_python_target(tmp_path)
    pid_file = tmp_path / "worker.pid"
    config = _python_config(
        tmp_path,
        "customer_agent:pid_agent",
        limits=LocalTargetLimits(turn_timeout_seconds=20, shutdown_timeout_seconds=0.2),
    )
    connection = LocalTargetConnection(config, customer_code_execution_confirmed=True)
    task = asyncio.create_task(connection.execute(_case("case-cancel", str(pid_file))))
    async with asyncio.timeout(2):
        while not pid_file.exists():
            await asyncio.sleep(0.01)
    pid = int(pid_file.read_text(encoding="utf-8"))

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await connection.aclose()

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.01)
    else:
        os.kill(pid, signal.SIGKILL)
        pytest.fail("cancelled local worker remained alive")
