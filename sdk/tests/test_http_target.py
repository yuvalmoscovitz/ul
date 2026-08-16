from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest
from ul.http_target import (
    JsonHttpDatasetTarget,
    validate_json_http_dataset_target_configuration,
)

pytestmark = pytest.mark.asyncio


class _StaticResponseStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._body


def _raw_response(body: bytes, *, content_type: str) -> httpx.Response:
    return httpx.Response(
        200,
        stream=_StaticResponseStream(body),
        headers={"content-type": content_type},
    )


class _JsonTargetHandler(BaseHTTPRequestHandler):
    request_body: dict[str, Any] | None = None
    authorization: str | None = None
    accept_encoding: str | None = None

    def do_POST(self) -> None:
        content_length = int(self.headers["content-length"])
        type(self).request_body = json.loads(self.rfile.read(content_length))
        type(self).authorization = self.headers.get("authorization")
        type(self).accept_encoding = self.headers.get("accept-encoding")
        response_body = json.dumps(
            {
                "actions": [
                    {
                        "action": "transfer",
                        "amount": 100,
                        "recipient": "Alice",
                    }
                ]
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format: str, *args: object) -> None:
        pass


@contextmanager
def _loopback_server() -> Generator[str]:
    _JsonTargetHandler.request_body = None
    _JsonTargetHandler.authorization = None
    _JsonTargetHandler.accept_encoding = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), _JsonTargetHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/execute"
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()


async def test_posts_input_to_real_json_endpoint_and_preserves_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_AGENT_AUTHORIZATION", "Bearer private-token")
    with _loopback_server() as endpoint:
        async with JsonHttpDatasetTarget(
            endpoint,
            sandbox_confirmed=True,
            request_field="query",
            header_environment_variables={
                "Authorization": "TEST_AGENT_AUTHORIZATION",
            },
            allow_insecure_http=True,
        ) as target:
            output = await target.execute("transfer 100 to Alice")

    assert _JsonTargetHandler.request_body == {"query": "transfer 100 to Alice"}
    assert _JsonTargetHandler.authorization == "Bearer private-token"
    assert _JsonTargetHandler.accept_encoding == "identity"
    assert output.raw_output == {
        "actions": [{"action": "transfer", "amount": 100, "recipient": "Alice"}]
    }
    assert output.metadata == {}
    assert target.safety_envelope.isolated is True
    assert target.safety_envelope.allows_network_egress is True
    assert target.safety_envelope.allows_business_side_effects is False


@pytest.mark.parametrize(
    ("endpoint", "allow_insecure_http", "message"),
    [
        ("https://user:password@example.com/run", False, "credentials"),
        ("https://example.com/run?token=secret", False, "query or fragment"),
        ("https://example.com/run#result", False, "query or fragment"),
        ("file:///tmp/agent", False, "valid HTTP"),
        ("http://example.com/run", False, "insecure transport opt-in"),
    ],
)
async def test_rejects_unsafe_endpoints(
    endpoint: str,
    allow_insecure_http: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        JsonHttpDatasetTarget(
            endpoint,
            sandbox_confirmed=True,
            allow_insecure_http=allow_insecure_http,
        )


async def test_requires_explicit_sandbox_confirmation() -> None:
    with pytest.raises(ValueError, match="explicit sandbox confirmation"):
        JsonHttpDatasetTarget("https://example.com/run", sandbox_confirmed=False)

    invalid_confirmation: Any = "true"
    with pytest.raises(ValueError, match="explicit sandbox confirmation"):
        validate_json_http_dataset_target_configuration(
            "https://example.com/run",
            sandbox_confirmed=invalid_confirmation,
        )


async def test_public_configuration_validation_resolves_headers_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_AGENT_TOKEN", "private-token")

    headers = validate_json_http_dataset_target_configuration(
        "https://example.com/run",
        sandbox_confirmed=True,
        request_field="query",
        header_environment_variables={"Authorization": "TEST_AGENT_TOKEN"},
    )

    assert headers == {"Authorization": "private-token"}


async def test_target_stores_headers_resolved_during_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_AGENT_TOKEN", "original-token")
    observed_authorization: str | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal observed_authorization
        observed_authorization = request.headers.get("authorization")
        return _raw_response(b'{"answer":"ok"}', content_type="application/json")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        target = JsonHttpDatasetTarget(
            "https://example.com/run",
            sandbox_confirmed=True,
            header_environment_variables={"Authorization": "TEST_AGENT_TOKEN"},
            client=client,
        )
        monkeypatch.setenv("TEST_AGENT_TOKEN", "changed-token")
        await target.execute("hello")

    assert observed_authorization == "original-token"


@pytest.mark.parametrize(
    "request_field",
    ("", "nested.input", "input value", "input\nInjected"),
)
async def test_rejects_non_simple_request_fields(request_field: str) -> None:
    with pytest.raises(ValueError, match="simple JSON field"):
        JsonHttpDatasetTarget(
            "https://example.com/run",
            sandbox_confirmed=True,
            request_field=request_field,
        )


@pytest.mark.parametrize(
    "headers",
    (
        {"Host": "AGENT_HOST"},
        {"Content-Length": "AGENT_LENGTH"},
        {"Accept-Encoding": "AGENT_ENCODING"},
        {"Bad Header": "AGENT_TOKEN"},
        {"Authorization": "NOT-VALID"},
        {"X-Test": "FIRST", "x-test": "SECOND"},
    ),
)
async def test_rejects_unsafe_header_configuration(headers: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="header"):
        JsonHttpDatasetTarget(
            "https://example.com/run",
            sandbox_confirmed=True,
            header_environment_variables=headers,
        )


@pytest.mark.parametrize("value", (None, "", "  ", "safe\r\nInjected: value"))
async def test_invalid_header_environment_variable_fails_during_configuration(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
) -> None:
    if value is None:
        monkeypatch.delenv("TEST_AGENT_TOKEN", raising=False)
    else:
        monkeypatch.setenv("TEST_AGENT_TOKEN", value)
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, json={"answer": "unexpected"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="environment variable"):
            JsonHttpDatasetTarget(
                "https://example.com/run",
                sandbox_confirmed=True,
                header_environment_variables={"Authorization": "TEST_AGENT_TOKEN"},
                client=client,
            )
        assert not client.is_closed

    assert request_count == 0


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(302, headers={"location": "https://elsewhere.test"}), "non-success"),
        (httpx.Response(500, text="private server detail"), "non-success"),
        (httpx.Response(200, text='{"answer":"ok"}'), "must be JSON"),
        (
            _raw_response(b"not-json", content_type="application/json"),
            "invalid JSON",
        ),
        (
            _raw_response(b"null", content_type="application/json"),
            "null JSON",
        ),
        (
            _raw_response(b"NaN", content_type="application/json"),
            "invalid JSON",
        ),
        (
            _raw_response(b"1e10000", content_type="application/json"),
            "invalid JSON",
        ),
    ],
)
async def test_rejects_unsafe_or_invalid_responses(
    response: httpx.Response,
    message: str,
) -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response)) as client:
        target = JsonHttpDatasetTarget(
            "https://example.com/run",
            sandbox_confirmed=True,
            client=client,
        )
        with pytest.raises(RuntimeError, match=message) as error:
            await target.execute("private input")

    assert "private" not in str(error.value)


async def test_enforces_streamed_response_size_limit() -> None:
    response = _raw_response(b'{"answer":"too large"}', content_type="application/json")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response)) as client:
        target = JsonHttpDatasetTarget(
            "https://example.com/run",
            sandbox_confirmed=True,
            max_response_bytes=10,
            client=client,
        )
        with pytest.raises(RuntimeError, match="size limit"):
            await target.execute("hello")


async def test_rejects_content_encoding_before_reading_response() -> None:
    request_accept_encoding: str | None = None

    class UnreadableStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            raise AssertionError("encoded response body was read")
            yield b""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_accept_encoding
        request_accept_encoding = request.headers.get("accept-encoding")
        return httpx.Response(
            200,
            stream=UnreadableStream(),
            headers={
                "content-type": "application/json",
                "content-encoding": "gzip",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        target = JsonHttpDatasetTarget(
            "https://example.com/run",
            sandbox_confirmed=True,
            client=client,
        )
        with pytest.raises(RuntimeError, match="content encoding"):
            await target.execute("hello")

    assert request_accept_encoding == "identity"


@pytest.mark.parametrize(
    "response_body",
    (
        b'{"answer":"first","answer":"second"}',
        b'{"answer":{"value":1,"value":2}}',
    ),
)
async def test_rejects_duplicate_json_keys_at_every_depth(response_body: bytes) -> None:
    response = _raw_response(response_body, content_type="application/json")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response)) as client:
        target = JsonHttpDatasetTarget(
            "https://example.com/run",
            sandbox_confirmed=True,
            client=client,
        )
        with pytest.raises(RuntimeError, match="invalid JSON") as error:
            await target.execute("private input")

    assert "first" not in str(error.value)
    assert "second" not in str(error.value)


async def test_deeply_nested_json_fails_with_a_sanitized_error() -> None:
    response_body = b"[" * 50_000 + b"0" + b"]" * 50_000
    response = _raw_response(response_body, content_type="application/json")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response)) as client:
        target = JsonHttpDatasetTarget(
            "https://example.com/run",
            sandbox_confirmed=True,
            client=client,
        )
        with pytest.raises(RuntimeError, match="invalid JSON"):
            await target.execute("private input")


async def test_total_timeout_stops_a_slow_drip_response() -> None:
    class SlowResponseStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            for chunk in (b'{"answer":', b'"ok"}'):
                await asyncio.sleep(0.03)
                yield chunk

    response = httpx.Response(
        200,
        stream=SlowResponseStream(),
        headers={"content-type": "application/json"},
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response)) as client:
        target = JsonHttpDatasetTarget(
            "https://example.com/run",
            sandbox_confirmed=True,
            timeout_seconds=0.05,
            client=client,
        )
        with pytest.raises(RuntimeError, match="timed out"):
            await target.execute("hello")


async def test_caller_cancellation_is_not_converted_to_a_target_error() -> None:
    response_started = asyncio.Event()

    class WaitingResponseStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            response_started.set()
            await asyncio.Event().wait()
            yield b""

    response = httpx.Response(
        200,
        stream=WaitingResponseStream(),
        headers={"content-type": "application/json"},
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response)) as client:
        target = JsonHttpDatasetTarget(
            "https://example.com/run",
            sandbox_confirmed=True,
            timeout_seconds=10,
            client=client,
        )
        execution = asyncio.create_task(target.execute("hello"))
        await response_started.wait()
        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await execution


async def test_transport_errors_do_not_expose_endpoint_or_input() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private transport detail", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        target = JsonHttpDatasetTarget(
            "https://example.com/private-path",
            sandbox_confirmed=True,
            client=client,
        )
        with pytest.raises(RuntimeError, match="request failed") as error:
            await target.execute("private input")

    assert "private" not in str(error.value)


async def test_does_not_retry_failed_action_request() -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(503, text="try again")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        target = JsonHttpDatasetTarget(
            "https://example.com/run",
            sandbox_confirmed=True,
            client=client,
        )
        with pytest.raises(RuntimeError, match="non-success"):
            await target.execute("perform one action")

    assert request_count == 1
