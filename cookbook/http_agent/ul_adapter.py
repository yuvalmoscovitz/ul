from __future__ import annotations

import json
import os
from ipaddress import ip_address
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, JsonValue
from ul import ObservedAgentOutput, SafetyEnvelope

_MAX_RESPONSE_BYTES = 1_000_000


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    result: JsonValue
    committed_state_snapshot: JsonValue | None = None


class ExistingHttpAgentTarget:
    safety_envelope = SafetyEnvelope(
        description="Customer-confirmed isolated HTTP agent test environment.",
        isolated=True,
        allows_network_egress=True,
        allows_business_side_effects=False,
    )
    fresh_state_per_execution = True

    def __init__(self, endpoint: str, bearer_token: str | None) -> None:
        self._endpoint = endpoint
        headers = {"Accept-Encoding": "identity"}
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(10, connect=3),
            follow_redirects=False,
            trust_env=False,
        )

    async def execute(self, raw_input: str) -> ObservedAgentOutput:
        try:
            async with self._client.stream(
                "POST", self._endpoint, json={"input": raw_input}
            ) as response:
                response.raise_for_status()
                if (
                    response.headers.get("Content-Encoding", "identity").strip().lower()
                    != "identity"
                ):
                    raise RuntimeError("Agent API returned encoded content")
                body = bytearray()
                async for chunk in response.aiter_raw(chunk_size=65_536):
                    body.extend(chunk)
                    if len(body) > _MAX_RESPONSE_BYTES:
                        raise RuntimeError("Agent API response exceeded 1 MB")
            parsed = AgentResponse.model_validate(json.loads(body))
            if parsed.result is None:
                raise ValueError("result must not be null")
        except (httpx.HTTPError, json.JSONDecodeError, ValueError):
            raise RuntimeError("Agent API request failed or returned an invalid response") from None

        metadata: dict[str, JsonValue] = {"adapter": "http_agent"}
        if parsed.committed_state_snapshot is not None:
            metadata["committed_state_snapshot"] = parsed.committed_state_snapshot
        return ObservedAgentOutput(raw_output=parsed.result, metadata=metadata)

    async def aclose(self) -> None:
        await self._client.aclose()


def create_target() -> ExistingHttpAgentTarget:
    if os.environ.get("UL_HTTP_AGENT_CONFIRMED_ISOLATED") != "true":
        raise RuntimeError(
            "Set UL_HTTP_AGENT_CONFIRMED_ISOLATED=true after verifying test isolation"
        )
    endpoint = os.environ.get("UL_HTTP_AGENT_ENDPOINT", "")
    _validate_endpoint(endpoint)
    return ExistingHttpAgentTarget(endpoint, os.environ.get("UL_HTTP_AGENT_BEARER_TOKEN"))


def _validate_endpoint(endpoint: str) -> None:
    parsed = urlsplit(endpoint)
    if (
        not endpoint
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.hostname
    ):
        raise ValueError(
            "UL_HTTP_AGENT_ENDPOINT must be a URL without credentials, query, or fragment"
        )
    try:
        loopback = ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = parsed.hostname == "localhost"
    if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
        raise ValueError("UL_HTTP_AGENT_ENDPOINT must use HTTPS or loopback HTTP")
