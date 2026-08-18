from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import AsyncIterator, Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
import ul.http_target as http_target_module
from pydantic import JsonValue, ValidationError
from ul.http_target import (
    JsonHttpDatasetTarget,
    JsonHttpDatasetTargetConfig,
    json_http_target_calls_per_execution,
    load_json_http_dataset_target_config,
    validate_json_http_dataset_target_configuration,
)
from ul_core.contracts import DatasetTargetLifecycleError

pytestmark = pytest.mark.asyncio


class _StaticResponseStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._body


def _raw_response(
    body: bytes,
    *,
    content_type: str = "application/json",
    status_code: int = 200,
) -> httpx.Response:
    return httpx.Response(
        status_code,
        stream=_StaticResponseStream(body),
        headers={"content-type": content_type},
    )


def _lifecycle_config(
    base_url: str = "https://sandbox.example.test",
    *,
    headers_from_env: dict[str, str] | None = None,
    clean_state_value: JsonValue = True,
    setup: bool = True,
) -> JsonHttpDatasetTargetConfig:
    config: dict[str, Any] = {
        "version": 2,
        "headers_from_env": headers_from_env or {},
        "reset": {
            "url": f"{base_url}/reset",
            "generation_json_pointer": "/generation",
            "clean_state_json_pointer": "/clean",
            "clean_state_value": clean_state_value,
        },
        "execute_turn": {
            "url": f"{base_url}/execute",
            "request_json_template": {"input": "{{input}}"},
            "response_json_pointer": "/agent_response",
        },
        "snapshot": {
            "url": f"{base_url}/snapshot",
            "response_json_pointer": "/state",
        },
    }
    if setup:
        config["setup"] = {
            "url": f"{base_url}/setup",
            "request_json": {"starting_amount": 100},
        }
    return JsonHttpDatasetTargetConfig.model_validate(config)


class _LifecycleHandler(BaseHTTPRequestHandler):
    events: ClassVar[list[str]] = []
    amount = 0
    generation = 0
    clean_state: JsonValue = True

    def do_POST(self) -> None:
        content_length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(content_length))
        type(self).events.append(self.path)
        if self.path == "/reset":
            type(self).amount = 0
            type(self).generation += 1
            response = {
                "generation": type(self).generation,
                "clean": type(self).clean_state,
            }
        elif self.path == "/setup":
            type(self).amount = request["starting_amount"]
            response = {}
        elif self.path == "/execute":
            type(self).amount += 50
            response = {"agent_response": {"message": "completed", "input": request["input"]}}
        elif self.path == "/snapshot":
            response = {"state": {"committed_amount": type(self).amount}}
        else:
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


@contextmanager
def _loopback_server() -> Generator[str]:
    _LifecycleHandler.events = []
    _LifecycleHandler.amount = 0
    _LifecycleHandler.generation = 0
    _LifecycleHandler.clean_state = True
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LifecycleHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()


async def test_runs_real_lifecycle_and_keeps_snapshot_separate() -> None:
    with _loopback_server() as base_url:
        config = _lifecycle_config(base_url)
        async with JsonHttpDatasetTarget.from_config(
            config,
            sandbox_confirmed=True,
            allow_insecure_http=True,
            max_target_calls=5,
        ) as target:
            output = await target.execute("increase the amount")

    assert _LifecycleHandler.events == [
        "/reset",
        "/setup",
        "/execute",
        "/snapshot",
        "/reset",
    ]
    assert _LifecycleHandler.amount == 0
    assert output.raw_output == {"message": "completed", "input": "increase the amount"}
    assert output.metadata["committed_state_snapshot"] == {"committed_amount": 150}
    assert output.metadata["target_protocol_version"] == 2
    assert target.fresh_state_per_execution is True
    assert json_http_target_calls_per_execution(config) == 5


async def test_json_null_is_a_valid_configured_clean_state() -> None:
    with _loopback_server() as base_url:
        _LifecycleHandler.clean_state = None
        async with JsonHttpDatasetTarget.from_config(
            _lifecycle_config(base_url, clean_state_value=None),
            sandbox_confirmed=True,
            allow_insecure_http=True,
            max_target_calls=5,
        ) as target:
            output = await target.execute("increase the amount")

    assert output.metadata["committed_state_snapshot"] == {"committed_amount": 150}
    assert _LifecycleHandler.events[-1] == "/reset"


def _successful_handler() -> tuple[Any, list[str]]:
    observed_paths: list[str] = []
    reset_generation = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal reset_generation
        observed_paths.append(request.url.path)
        if request.url.path == "/reset":
            reset_generation += 1
            return _raw_response(
                json.dumps({"generation": reset_generation, "clean": True}).encode()
            )
        if request.url.path == "/setup":
            return _raw_response(b"")
        if request.url.path == "/execute":
            return _raw_response(b'{"agent_response":{"ok":true}}')
        return _raw_response(b'{"state":{"committed":true}}')

    return handler, observed_paths


async def test_cleanup_failure_fails_closed() -> None:
    reset_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal reset_calls
        if request.url.path == "/reset":
            reset_calls += 1
            if reset_calls == 2:
                return httpx.Response(500)
            return _raw_response(b'{"generation":1,"clean":true}')
        if request.url.path == "/setup":
            return _raw_response(b"")
        if request.url.path == "/execute":
            return _raw_response(b'{"agent_response":{"ok":true}}')
        return _raw_response(b'{"state":{"committed":true}}')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        target = JsonHttpDatasetTarget.from_config(
            _lifecycle_config(), sandbox_confirmed=True, max_target_calls=5, client=client
        )
        with pytest.raises(DatasetTargetLifecycleError) as captured:
            await target.execute("work")

    assert captured.value.failed_phase == "cleanup_reset"
    assert captured.value.cleanup_reset_failed is True
    assert captured.value.target_state_uncertain is True


async def test_late_commit_transport_failure_permanently_blocks_target() -> None:
    observed_paths: list[str] = []
    reset_generation = 0
    committed_once = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal committed_once, reset_generation
        observed_paths.append(request.url.path)
        if request.url.path == "/reset":
            reset_generation += 1
            return _raw_response(
                json.dumps({"generation": reset_generation, "clean": True}).encode()
            )
        if request.url.path == "/setup":
            return _raw_response(b"")
        if request.url.path == "/execute":
            committed_once = True
            raise httpx.ReadTimeout("response lost after commit", request=request)
        raise AssertionError("snapshot must not follow uncertain execution")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        target = JsonHttpDatasetTarget.from_config(
            _lifecycle_config(), sandbox_confirmed=True, max_target_calls=10, client=client
        )
        with pytest.raises(DatasetTargetLifecycleError) as captured:
            await target.execute("work")
        with pytest.raises(DatasetTargetLifecycleError) as blocked:
            await target.execute("more work")

    assert committed_once is True
    assert captured.value.failed_phase == "execute_turn"
    assert captured.value.cleanup_reset_failed is False
    assert captured.value.target_state_uncertain is True
    assert blocked.value.failed_phase == "blocked_state_uncertain"
    assert observed_paths == ["/reset", "/setup", "/execute", "/reset"]


async def test_cancellation_during_cleanup_blocks_later_execution() -> None:
    cleanup_started = asyncio.Event()
    never_complete = asyncio.Event()
    observed_paths: list[str] = []
    reset_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal reset_calls
        observed_paths.append(request.url.path)
        if request.url.path == "/reset":
            reset_calls += 1
            if reset_calls == 2:
                cleanup_started.set()
                await never_complete.wait()
            return _raw_response(json.dumps({"generation": reset_calls, "clean": True}).encode())
        if request.url.path == "/setup":
            return _raw_response(b"")
        if request.url.path == "/execute":
            return _raw_response(b'{"agent_response":{"ok":true}}')
        return _raw_response(b'{"state":{"committed":true}}')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        target = JsonHttpDatasetTarget.from_config(
            _lifecycle_config(), sandbox_confirmed=True, max_target_calls=10, client=client
        )
        execution = asyncio.create_task(target.execute("work"))
        await cleanup_started.wait()
        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await execution
        with pytest.raises(DatasetTargetLifecycleError) as blocked:
            await target.execute("more work")

    assert blocked.value.failed_phase == "blocked_state_uncertain"
    assert observed_paths == ["/reset", "/setup", "/execute", "/snapshot", "/reset"]


async def test_records_failed_reset_generation_before_clean_assertion() -> None:
    reset_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal reset_calls
        reset_calls += 1
        return _raw_response(json.dumps({"generation": 1, "clean": reset_calls > 1}).encode())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        target = JsonHttpDatasetTarget.from_config(
            _lifecycle_config(), sandbox_confirmed=True, max_target_calls=5, client=client
        )
        with pytest.raises(DatasetTargetLifecycleError) as captured:
            await target.execute("work")

    assert captured.value.failed_phase == "reset"
    assert captured.value.cleanup_reset_failed is True
    assert reset_calls == 2


async def test_serializes_complete_lifecycles() -> None:
    handler, observed_paths = _successful_handler()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        target = JsonHttpDatasetTarget.from_config(
            _lifecycle_config(), sandbox_confirmed=True, max_target_calls=10, client=client
        )
        await asyncio.gather(target.execute("first"), target.execute("second"))

    lifecycle = ["/reset", "/setup", "/execute", "/snapshot", "/reset"]
    assert observed_paths == [*lifecycle, *lifecycle]


async def test_reserves_complete_budget_before_first_request() -> None:
    handler, observed_paths = _successful_handler()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        target = JsonHttpDatasetTarget.from_config(
            _lifecycle_config(), sandbox_confirmed=True, max_target_calls=4, client=client
        )
        with pytest.raises(RuntimeError, match="call budget exhausted"):
            await target.execute("work")
    assert observed_paths == []


async def test_loads_lifecycle_config_and_sends_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_AGENT_TOKEN", "Bearer token")
    config_path = tmp_path / "target.json"
    config_path.write_text(
        json.dumps(
            _lifecycle_config(headers_from_env={"Authorization": "TEST_AGENT_TOKEN"}).model_dump(
                mode="json"
            )
        ),
        encoding="utf-8",
    )
    observed_authorizations: list[str | None] = []
    handler, _ = _successful_handler()

    def recording_handler(request: httpx.Request) -> httpx.Response:
        observed_authorizations.append(request.headers.get("authorization"))
        return handler(request)

    config = load_json_http_dataset_target_config(config_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(recording_handler)) as client:
        target = JsonHttpDatasetTarget.from_config(
            config, sandbox_confirmed=True, max_target_calls=5, client=client
        )
        output = await target.execute("hello")

    assert output.raw_output == {"ok": True}
    assert observed_authorizations == ["Bearer token"] * 5


async def test_rejects_cross_origin_lifecycle() -> None:
    config = _lifecycle_config().model_dump(mode="json")
    config["snapshot"]["url"] = "https://other.example.test/snapshot"
    with pytest.raises(ValidationError, match="same origin"):
        JsonHttpDatasetTargetConfig.model_validate(config)


@pytest.mark.parametrize(
    "encoded_config",
    (
        b'{"version":2,"version":2}',
        b'{"version":NaN}',
        b"[" * 200 + b"0" + b"]" * 200,
    ),
)
async def test_loader_rejects_adversarial_json(tmp_path: Path, encoded_config: bytes) -> None:
    config_path = tmp_path / "target.json"
    config_path.write_bytes(encoded_config)
    with pytest.raises(ValueError, match="invalid JSON"):
        load_json_http_dataset_target_config(config_path)


async def test_loader_rejects_symlink_and_fifo(tmp_path: Path) -> None:
    real_path = tmp_path / "real.json"
    real_path.write_text("{}", encoding="utf-8")
    symlink_path = tmp_path / "link.json"
    symlink_path.symlink_to(real_path)
    with pytest.raises(RuntimeError, match="could not be read"):
        load_json_http_dataset_target_config(symlink_path)

    fifo_path = tmp_path / "config.fifo"
    os.mkfifo(fifo_path)
    with pytest.raises(RuntimeError, match="could not be read"):
        load_json_http_dataset_target_config(fifo_path)


@pytest.mark.parametrize("unsafe_value", ("value\x00suffix", "value\x01suffix", "value\x7fsuffix"))
async def test_rejects_header_control_bytes_before_network(
    monkeypatch: pytest.MonkeyPatch, unsafe_value: str
) -> None:
    monkeypatch.setattr(http_target_module.os, "environ", {"TEST_UNSAFE_HEADER": unsafe_value})
    with pytest.raises(RuntimeError, match="environment variable is invalid"):
        JsonHttpDatasetTarget.from_config(
            _lifecycle_config(headers_from_env={"Authorization": "TEST_UNSAFE_HEADER"}),
            sandbox_confirmed=True,
        )


async def test_requires_explicit_sandbox_confirmation() -> None:
    with pytest.raises(ValueError, match="sandbox confirmation"):
        JsonHttpDatasetTarget.from_config(_lifecycle_config(), sandbox_confirmed=False)


async def test_rejects_insecure_transport_without_opt_in() -> None:
    with pytest.raises(ValueError, match="insecure transport opt-in"):
        JsonHttpDatasetTarget.from_config(
            _lifecycle_config("http://sandbox.example.test"), sandbox_confirmed=True
        )


async def test_public_validation_resolves_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_AGENT_TOKEN", "private-token")
    assert validate_json_http_dataset_target_configuration(
        _lifecycle_config(headers_from_env={"Authorization": "TEST_AGENT_TOKEN"}),
        sandbox_confirmed=True,
    ) == {"Authorization": "private-token"}


async def test_redirect_is_not_followed_and_cleanup_failure_is_preserved() -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(307, headers={"location": "https://other.example.test"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        target = JsonHttpDatasetTarget.from_config(
            _lifecycle_config(), sandbox_confirmed=True, max_target_calls=5, client=client
        )
        with pytest.raises(DatasetTargetLifecycleError) as captured:
            await target.execute("work")
    assert requests == 2
    assert captured.value.failed_phase == "reset"
    assert captured.value.cleanup_reset_failed is True


async def test_timeout_during_execute_poison_target_even_after_cleanup() -> None:
    reset_generation = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal reset_generation
        if request.url.path == "/reset":
            reset_generation += 1
            return _raw_response(
                json.dumps({"generation": reset_generation, "clean": True}).encode()
            )
        if request.url.path == "/setup":
            return _raw_response(b"")
        if request.url.path == "/execute":
            await asyncio.sleep(1)
        return _raw_response(b'{"state":{}}')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        target = JsonHttpDatasetTarget.from_config(
            _lifecycle_config(),
            sandbox_confirmed=True,
            timeout_seconds=0.01,
            max_target_calls=10,
            client=client,
        )
        with pytest.raises(DatasetTargetLifecycleError) as captured:
            await target.execute("work")
        with pytest.raises(DatasetTargetLifecycleError) as blocked:
            await target.execute("more")
    assert captured.value.target_state_uncertain is True
    assert blocked.value.failed_phase == "blocked_state_uncertain"
