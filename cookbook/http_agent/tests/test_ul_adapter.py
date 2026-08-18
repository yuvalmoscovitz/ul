from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest

from cookbook.http_agent import ul_adapter


class _AsyncBytes(httpx.AsyncByteStream):
    def __init__(self, content: bytes) -> None:
        self._content = content

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._content


def _install_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    async_client = httpx.AsyncClient

    def create_client(**kwargs: Any) -> httpx.AsyncClient:
        return async_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(ul_adapter.httpx, "AsyncClient", create_client)


@pytest.mark.asyncio
async def test_adapter_posts_input_and_preserves_result_and_committed_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == "https://agent.example.test/run"
        assert request.headers["Authorization"] == "Bearer test-secret"
        assert request.headers["Accept-Encoding"] == "identity"
        assert request.read() == b'{"input":"approve invoice 123"}'
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            stream=_AsyncBytes(
                b'{"result":{"message":"approved"},'
                b'"committed_state_snapshot":{"invoice_123":"approved"}}'
            ),
        )

    _install_transport(monkeypatch, handler)
    monkeypatch.setenv("UL_HTTP_AGENT_CONFIRMED_ISOLATED", "true")
    monkeypatch.setenv("UL_HTTP_AGENT_ENDPOINT", "https://agent.example.test/run")
    monkeypatch.setenv("UL_HTTP_AGENT_BEARER_TOKEN", "test-secret")

    target = ul_adapter.create_target()
    try:
        output = await target.execute("approve invoice 123")
    finally:
        await target.aclose()

    assert output.raw_output == {"message": "approved"}
    assert output.metadata == {
        "adapter": "http_agent",
        "committed_state_snapshot": {"invoice_123": "approved"},
    }
    assert target.safety_envelope.allows_network_egress is True
    assert target.safety_envelope.allows_business_side_effects is False
    assert target.fresh_state_per_execution is True


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://agent.example.test/run",
        "https://user:secret@agent.example.test/run",
        "https://agent.example.test/run?token=secret",
        "https://agent.example.test/run#secret",
    ],
)
def test_factory_rejects_unsafe_endpoint(endpoint: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UL_HTTP_AGENT_CONFIRMED_ISOLATED", "true")
    monkeypatch.setenv("UL_HTTP_AGENT_ENDPOINT", endpoint)

    with pytest.raises(ValueError) as error:
        ul_adapter.create_target()

    assert "secret" not in str(error.value)


def test_factory_requires_explicit_isolation_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UL_HTTP_AGENT_ENDPOINT", "https://agent.example.test/run")

    with pytest.raises(RuntimeError, match="verifying test isolation"):
        ul_adapter.create_target()


@pytest.mark.asyncio
async def test_adapter_rejects_redirects_without_leaking_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(307, headers={"Location": "https://attacker.example/run"})

    _install_transport(monkeypatch, handler)
    target = ul_adapter.ExistingHttpAgentTarget("https://agent.example.test/run", "do-not-leak")
    try:
        with pytest.raises(RuntimeError) as error:
            await target.execute("hello")
    finally:
        await target.aclose()

    assert str(error.value) == "Agent API request failed or returned an invalid response"
    assert "do-not-leak" not in str(error.value)


@pytest.mark.asyncio
async def test_adapter_bounds_response_size(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_AsyncBytes(b"x" * 1_000_001))

    _install_transport(monkeypatch, handler)
    target = ul_adapter.ExistingHttpAgentTarget("https://agent.example.test/run", None)
    try:
        with pytest.raises(RuntimeError, match="exceeded 1 MB"):
            await target.execute("hello")
    finally:
        await target.aclose()


@pytest.mark.asyncio
async def test_adapter_rejects_encoded_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Encoding": "gzip"},
            stream=_AsyncBytes(b"not inspected or decoded"),
        )

    _install_transport(monkeypatch, handler)
    target = ul_adapter.ExistingHttpAgentTarget("https://agent.example.test/run", None)
    try:
        with pytest.raises(RuntimeError, match="encoded content"):
            await target.execute("hello")
    finally:
        await target.aclose()


@pytest.mark.asyncio
async def test_adapter_does_not_reuse_response_cookies(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        assert "Cookie" not in request.headers
        return httpx.Response(
            200,
            headers={"Set-Cookie": "session=previous-trial; Secure"},
            stream=_AsyncBytes(b'{"result":"ok"}'),
        )

    _install_transport(monkeypatch, handler)
    target = ul_adapter.ExistingHttpAgentTarget("https://agent.example.test/run", None)
    try:
        await target.execute("first")
        await target.execute("second")
    finally:
        await target.aclose()

    assert requests == 2
