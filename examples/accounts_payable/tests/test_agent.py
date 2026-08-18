import json
from typing import cast

import httpx
import pytest
from pydantic import SecretStr

from examples.accounts_payable.agent import (
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterSettings,
    run_openrouter_agent,
)
from examples.accounts_payable.environment import AccountsPayableEnvironment
from examples.accounts_payable.scenarios import get_seed_scenario


@pytest.mark.asyncio
async def test_live_calls_are_disabled_by_default() -> None:
    scenario = get_seed_scenario("single-approved-invoice")
    settings = OpenRouterSettings(api_key=SecretStr("not-used"), live_calls_enabled=False)
    environment = AccountsPayableEnvironment(scenario.state)

    with pytest.raises(RuntimeError, match="disabled"):
        await run_openrouter_agent(scenario, environment, settings)


@pytest.mark.asyncio
async def test_single_request_uses_system_prompt_tools_and_safe_parameters() -> None:
    requests: list[httpx.Request] = []
    response_body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-one",
                            "type": "function",
                            "function": {
                                "name": "get_invoice",
                                "arguments": '{"invoice_id":"inv-acme-100"}',
                            },
                        }
                    ],
                }
            }
        ]
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=response_body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = OpenRouterSettings(
        api_key=SecretStr("test-secret"),
        live_calls_enabled=True,
    )
    scenario = get_seed_scenario("single-approved-invoice")
    environment = AccountsPayableEnvironment(scenario.state)

    result = await run_openrouter_agent(scenario, environment, settings, client)
    await client.aclose()

    request_body = json.loads(requests[0].content)
    assert request_body["model"] == DEFAULT_OPENROUTER_MODEL
    assert request_body["reasoning"] == {"effort": "none", "exclude": True}
    assert request_body["parallel_tool_calls"] is False
    assert len(requests) == 1
    assert len(result.tool_steps) == 1
    assert "test-secret" not in requests[0].content.decode()


@pytest.mark.parametrize(
    "response_body",
    cast(
        list[dict[str, object]],
        [
            {"choices": []},
            {"choices": [{"message": None}]},
            {"choices": [{"message": {"role": "assistant", "tool_calls": {}}}]},
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{"id": "call-one", "function": {"name": 7}}],
                        }
                    }
                ]
            },
        ],
    ),
)
@pytest.mark.asyncio
async def test_invalid_openrouter_responses_fail_validation(
    response_body: dict[str, object],
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = OpenRouterSettings(api_key=SecretStr("test-secret"), live_calls_enabled=True)
    scenario = get_seed_scenario("single-approved-invoice")
    environment = AccountsPayableEnvironment(scenario.state)

    with pytest.raises(ValueError):
        await run_openrouter_agent(scenario, environment, settings, client)
    await client.aclose()


@pytest.mark.asyncio
async def test_all_tool_calls_are_validated_before_any_are_dispatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "valid-call",
                            "function": {
                                "name": "get_invoice",
                                "arguments": '{"invoice_id":"inv-acme-100"}',
                            },
                        },
                        {"id": "invalid-call", "function": {"name": 7}},
                    ],
                }
            }
        ]
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_body)

    scenario = get_seed_scenario("single-approved-invoice")
    environment = AccountsPayableEnvironment(scenario.state)
    dispatched_tools: list[str] = []

    def record_dispatch(tool_name: str, arguments_json: str) -> str:
        dispatched_tools.append(tool_name)
        return "{}"

    monkeypatch.setattr(environment, "dispatch_json", record_dispatch)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = OpenRouterSettings(api_key=SecretStr("test-secret"), live_calls_enabled=True)

    with pytest.raises(ValueError):
        await run_openrouter_agent(scenario, environment, settings, client)
    await client.aclose()

    assert dispatched_tools == []
