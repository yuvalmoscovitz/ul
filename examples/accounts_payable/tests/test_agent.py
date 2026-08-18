import json

import httpx
import pytest
from pydantic import SecretStr

from examples.accounts_payable.agent import (
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterAccountsPayableAgent,
    OpenRouterSettings,
)
from examples.accounts_payable.environment import AccountsPayableEnvironment
from examples.accounts_payable.scenarios import get_seed_scenario


@pytest.mark.asyncio
async def test_live_calls_are_disabled_by_default() -> None:
    scenario = get_seed_scenario("single-approved-invoice")
    agent = OpenRouterAccountsPayableAgent(
        OpenRouterSettings(api_key=SecretStr("not-used"), live_calls_enabled=False)
    )

    with pytest.raises(RuntimeError, match="disabled"):
        await agent.run(scenario)


@pytest.mark.asyncio
async def test_tool_loop_is_bounded_and_uses_safe_request_parameters() -> None:
    requests: list[httpx.Request] = []
    responses = iter(
        [
            {
                "id": "generation-one",
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
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                    "cost": 0.001,
                },
            },
            {
                "id": "generation-two",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I need to verify approval before paying.",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 5,
                    "total_tokens": 20,
                    "cost": 0.002,
                },
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
        max_steps=3,
    )
    scenario = get_seed_scenario("single-approved-invoice")
    environment = AccountsPayableEnvironment(scenario.state)

    result = await OpenRouterAccountsPayableAgent(settings, client).run(scenario, environment)
    await client.aclose()

    request_body = json.loads(requests[0].content)
    assert request_body["model"] == DEFAULT_OPENROUTER_MODEL
    assert request_body["reasoning"] == {"effort": "none", "exclude": True}
    assert request_body["parallel_tool_calls"] is False
    assert len(result.tool_steps) == 1
    assert result.usage["total_tokens"] == 32
    assert abs(result.cost_usd - 0.003) < 1e-12
    assert result.stop_reason == "completed"
    assert [prompt["name"] for prompt in result.prompts if isinstance(prompt, dict)] == [
        "examples.accounts_payable.system",
        "examples.accounts_payable.tools.search_invoices",
        "examples.accounts_payable.tools.get_invoice",
        "examples.accounts_payable.tools.get_vendor",
        "examples.accounts_payable.tools.get_approval_status",
        "examples.accounts_payable.tools.get_payment_account",
        "examples.accounts_payable.tools.list_payments",
        "examples.accounts_payable.tools.execute_payment",
        "examples.accounts_payable.tools.get_payment_status",
    ]
    assert all(
        isinstance(prompt, dict) and len(str(prompt.get("version", ""))) == 64
        for prompt in result.prompts
    )
    assert "test-secret" not in requests[0].content.decode()
