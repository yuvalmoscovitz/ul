from __future__ import annotations

import json
from typing import cast

import httpx
import pytest
from pydantic import SecretStr, ValidationError
from ul.llm import LLMClient, LLMClientConfig, LLMRoleConfig

pytestmark = pytest.mark.asyncio


def _config() -> LLMClientConfig:
    return LLMClientConfig(
        provider_id="openrouter",
        provider_type="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key=SecretStr("private-api-key"),
        api_key_environment_variable="OPEN_ROUTER_API_KEY",
        api_key_required=True,
        live_calls=True,
        allow_external_data_processing=True,
        upstream_provider="pinned-provider",
        roles=(
            LLMRoleConfig(role="deconstruct", model="test/deconstruct", max_output_tokens=100),
            LLMRoleConfig(role="render", model="test/render", max_output_tokens=101),
            LLMRoleConfig(role="equivalence", model="test/equivalence", max_output_tokens=102),
            LLMRoleConfig(role="materiality", model="test/materiality", max_output_tokens=103),
        ),
        timeout_seconds=60.0,
        max_response_bytes=1_000_000,
    )


async def test_one_client_applies_the_same_deterministic_route_to_every_semantic_role() -> None:
    requests: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = cast(dict[str, object], json.loads(request.content))
        requests.append(body)
        return httpx.Response(
            200,
            json={
                "id": "generation-1",
                "model": body["model"],
                "provider": "Pinned Provider",
                "choices": [{"message": {"content": "{}"}}],
            },
        )

    config = _config()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        client = LLMClient(config, client=transport)
        for role in ("deconstruct", "render", "equivalence", "materiality"):
            await client.complete(
                role=role,
                reasoning={"effort": "low"},
                seed=0,
                top_p=None,
                schema_name="test_response",
                schema={"type": "object"},
                strict_schema=True,
                system_prompt="Return JSON.",
                user_payload="{}",
            )

    assert [request["model"] for request in requests] == [
        "test/deconstruct",
        "test/render",
        "test/equivalence",
        "test/materiality",
    ]
    for request in requests:
        assert request["temperature"] == 0
        assert request["provider"] == {
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
            "only": ["pinned-provider"],
            "allow_fallbacks": False,
        }


async def test_llm_configuration_is_frozen_and_secret_free_in_evidence() -> None:
    config = _config()

    with pytest.raises(ValidationError):
        config.__setattr__("temperature", 1)

    assert "private-api-key" not in json.dumps(config.evidence_identity())
