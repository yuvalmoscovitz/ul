from __future__ import annotations

import json
from typing import cast

import httpx
import pytest
from pydantic import SecretStr, ValidationError
from ul.deconstruction import OpenAICompatibleDatasetSettings
from ul.llm import (
    LLMClient,
    LLMClientConfig,
    LLMProviderMismatchError,
    LLMRoleConfig,
    llm_client_config_from_dataset_settings,
)

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
            LLMRoleConfig(
                role="deconstruct",
                model="test/deconstruct",
                max_output_tokens=100,
                reasoning_mode="required",
                reasoning_effort="minimal",
            ),
            LLMRoleConfig(
                role="render",
                model="test/render",
                max_output_tokens=101,
                reasoning_mode="required",
                reasoning_effort="none",
            ),
            LLMRoleConfig(
                role="equivalence",
                model="test/equivalence",
                max_output_tokens=102,
                reasoning_mode="required",
                reasoning_effort="low",
            ),
            LLMRoleConfig(
                role="materiality",
                model="test/materiality",
                max_output_tokens=103,
                reasoning_mode="omitted",
            ),
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
    assert [request.get("reasoning") for request in requests] == [
        {"effort": "minimal"},
        {"effort": "none"},
        {"effort": "low"},
        None,
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


async def test_google_vertex_pin_accepts_openrouter_google_response_identity() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        body = cast(dict[str, object], json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "generation-1",
                "model": body["model"],
                "provider": "Google",
                "choices": [{"message": {"content": "{}"}}],
            },
        )

    config = _config().model_copy(update={"upstream_provider": "google-vertex"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        client = LLMClient(config, client=transport)
        completion = await client.complete(
            role="materiality",
            seed=0,
            top_p=None,
            schema_name="test_response",
            schema={"type": "object"},
            strict_schema=True,
            system_prompt="Return JSON.",
            user_payload="{}",
        )

    assert completion.provider == "Google"


async def test_regional_google_vertex_pin_rejects_generic_google_response_identity() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        body = cast(dict[str, object], json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "generation-1",
                "model": body["model"],
                "provider": "Google",
                "choices": [{"message": {"content": "{}"}}],
            },
        )

    config = _config().model_copy(update={"upstream_provider": "google-vertex/us-east5"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        client = LLMClient(config, client=transport)
        with pytest.raises(LLMProviderMismatchError):
            await client.complete(
                role="materiality",
                seed=0,
                top_p=None,
                schema_name="test_response",
                schema={"type": "object"},
                strict_schema=True,
                system_prompt="Return JSON.",
                user_payload="{}",
            )


async def test_llm_configuration_is_frozen_and_secret_free_in_evidence() -> None:
    config = _config()

    with pytest.raises(ValidationError):
        config.__setattr__("temperature", 1)

    assert "private-api-key" not in config.evidence_identity().model_dump_json()


async def test_openai_compatible_identity_records_reasoning_as_omitted() -> None:
    config = llm_client_config_from_dataset_settings(
        OpenAICompatibleDatasetSettings(
            live_calls=True,
            allow_external_data_processing=True,
            api_key=SecretStr("customer-key"),
            provider_id="customer-gateway",
            base_url="https://models.example.test/v1",
            model="customer/model",
            deconstruct_reasoning="required",
            render_reasoning="required",
            equivalence_reasoning="required",
        )
    )

    for role in ("deconstruct", "render", "equivalence"):
        assert config.role_config(role).reasoning_metadata() == {
            "mode": "omitted",
            "effort": None,
        }
        assert "reasoning" not in config.request_options(role=role, seed=0, top_p=None)


@pytest.mark.parametrize(
    ("config_update", "expected_error"),
    [
        ({"live_calls": False}, "require UL_LIVE=true"),
        ({"allow_external_data_processing": False}, "process raw inputs and outputs"),
        ({"api_key": None}, "require OPEN_ROUTER_API_KEY"),
    ],
)
async def test_llm_access_controls_fail_before_any_request(
    config_update: dict[str, object],
    expected_error: str,
) -> None:
    request_count = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(500)

    config = _config().model_copy(update=config_update)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        client = LLMClient(config, client=transport)
        with pytest.raises(RuntimeError, match=expected_error):
            await client.complete(
                role="materiality",
                seed=0,
                top_p=None,
                schema_name="test_response",
                schema={"type": "object"},
                strict_schema=True,
                system_prompt="Return JSON.",
                user_payload="{}",
            )

    assert request_count == 0
