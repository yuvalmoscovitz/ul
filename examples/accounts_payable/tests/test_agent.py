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
async def test_tool_results_are_sent_back_to_the_model() -> None:
    requests: list[httpx.Request] = []
    responses = iter(
        [
            {
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
                ],
                "usage": {"cost": 0.001},
            },
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The invoice is approved for payment.",
                        }
                    }
                ],
                "usage": {"cost": 0.002},
            },
        ]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=next(responses))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = OpenRouterSettings(
        api_key=SecretStr("test-secret"),
        live_calls_enabled=True,
    )
    scenario = get_seed_scenario("single-approved-invoice")
    environment = AccountsPayableEnvironment(scenario.state)

    result = await run_openrouter_agent(scenario, environment, settings, client)
    await client.aclose()

    first_request = json.loads(requests[0].content)
    second_request = json.loads(requests[1].content)
    assert first_request["model"] == DEFAULT_OPENROUTER_MODEL
    assert first_request["reasoning"] == {"effort": "none", "exclude": True}
    assert first_request["parallel_tool_calls"] is False
    assert second_request["messages"][-1]["role"] == "tool"
    tool_result = json.loads(second_request["messages"][-1]["content"])
    assert tool_result["result"]["id"] == "inv-acme-100"
    assert len(requests) == 2
    assert len(result.tool_steps) == 1
    assert result.final_answer == "The invoice is approved for payment."
    assert abs(result.cost_usd - 0.003) < 1e-12
    assert result.error is None
    assert "test-secret" not in requests[0].content.decode()


@pytest.mark.asyncio
async def test_model_call_loop_is_bounded() -> None:
    request_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call-{request_count}",
                                    "function": {
                                        "name": "get_invoice",
                                        "arguments": '{"invoice_id":"inv-acme-100"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = OpenRouterSettings(api_key=SecretStr("test-secret"), live_calls_enabled=True)
    scenario = get_seed_scenario("single-approved-invoice")
    environment = AccountsPayableEnvironment(scenario.state)

    result = await run_openrouter_agent(scenario, environment, settings, client)
    await client.aclose()

    assert request_count == 10
    assert result.error == "OpenRouter agent reached the model call limit."


@pytest.mark.parametrize(
    "response_body",
    cast(
        list[dict[str, object]],
        [
            {"choices": []},
            {"choices": [{"message": None}]},
            {"choices": [{"message": {"role": "assistant", "tool_calls": {}}}]},
            {
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"cost": float("nan")},
            },
            {
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"cost": float("inf")},
            },
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


@pytest.mark.asyncio
async def test_accumulated_cost_must_remain_finite() -> None:
    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-one",
                                    "function": {
                                        "name": "get_invoice",
                                        "arguments": '{"invoice_id":"inv-acme-100"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"cost": 1e308},
            },
            {
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"cost": 1e308},
            },
        ]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(responses))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = OpenRouterSettings(api_key=SecretStr("test-secret"), live_calls_enabled=True)
    scenario = get_seed_scenario("single-approved-invoice")
    environment = AccountsPayableEnvironment(scenario.state)

    with pytest.raises(ValueError, match="accumulated cost"):
        await run_openrouter_agent(scenario, environment, settings, client)
    await client.aclose()
