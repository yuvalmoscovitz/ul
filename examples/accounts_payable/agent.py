from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import httpx
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from ul_core.prompts import PromptManager

from examples.accounts_payable.environment import AccountsPayableEnvironment
from examples.accounts_payable.models import AccountsPayableScenario
from examples.accounts_payable.tool_schemas import openrouter_tool_definitions

DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4-flash-0731"
_PROMPTS = PromptManager.instance()
_MAXIMUM_MODEL_CALLS = 10


class OpenRouterSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    api_key: SecretStr | None = Field(default=None, validation_alias="OPEN_ROUTER_API_KEY")
    model: str = Field(default=DEFAULT_OPENROUTER_MODEL, validation_alias="UL_AP_MODEL")
    live_calls_enabled: bool = Field(default=False, validation_alias="UL_AP_LIVE_CALLS")
    max_output_tokens: int = Field(
        default=800, ge=64, le=4000, validation_alias="UL_AP_MAX_OUTPUT_TOKENS"
    )
    request_timeout_seconds: float = Field(
        default=30.0, gt=0, le=120, validation_alias="UL_AP_REQUEST_TIMEOUT_SECONDS"
    )


@dataclass(frozen=True)
class AgentToolStep:
    tool_name: str
    arguments_json: str
    result_json: str


@dataclass(frozen=True)
class AgentRunResult:
    final_answer: str
    tool_steps: tuple[AgentToolStep, ...]
    cost_usd: float
    error: str | None = None


async def run_openrouter_agent(
    scenario: AccountsPayableScenario,
    environment: AccountsPayableEnvironment,
    settings: OpenRouterSettings,
    client: httpx.AsyncClient | None = None,
) -> AgentRunResult:
    if not settings.live_calls_enabled:
        raise RuntimeError(
            "Live OpenRouter calls are disabled. Set UL_AP_LIVE_CALLS=true to opt in."
        )
    if settings.api_key is None:
        raise RuntimeError("OPEN_ROUTER_API_KEY is required for live OpenRouter calls.")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _PROMPTS.get_prompt("examples.accounts_payable.system")},
        *[{"role": "user", "content": message} for message in scenario.user_messages],
    ]
    owns_client = client is None
    active_client = client or httpx.AsyncClient(timeout=settings.request_timeout_seconds)
    tool_steps: list[AgentToolStep] = []
    cost_usd = 0.0

    try:
        for _ in range(_MAXIMUM_MODEL_CALLS):
            response = await active_client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.api_key.get_secret_value()}"},
                json={
                    "model": settings.model,
                    "messages": messages,
                    "tools": openrouter_tool_definitions(),
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                    "reasoning": {"effort": "none", "exclude": True},
                    "temperature": 0,
                    "max_completion_tokens": settings.max_output_tokens,
                },
            )
            response.raise_for_status()
            assistant_message, tool_calls, response_cost = _parse_response(response)
            accumulated_cost = cost_usd + response_cost
            if not math.isfinite(accumulated_cost):
                raise ValueError("OpenRouter returned an invalid accumulated cost")
            cost_usd = accumulated_cost
            messages.append(assistant_message)
            if not tool_calls:
                content = assistant_message.get("content")
                return AgentRunResult(
                    content if isinstance(content, str) else "",
                    tuple(tool_steps),
                    cost_usd,
                )

            for tool_call_id, tool_name, arguments_json in tool_calls:
                result_json = environment.dispatch_json(tool_name, arguments_json)
                tool_steps.append(AgentToolStep(tool_name, arguments_json, result_json))
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call_id, "content": result_json}
                )
    finally:
        if owns_client:
            await active_client.aclose()

    return AgentRunResult(
        "The model call limit was reached before the task completed.",
        tuple(tool_steps),
        cost_usd,
        "OpenRouter agent reached the model call limit.",
    )


def _parse_response(
    response: httpx.Response,
) -> tuple[dict[str, Any], list[tuple[str, str, str]], float]:
    raw_response_data = response.json()
    if not isinstance(raw_response_data, dict):
        raise ValueError("OpenRouter returned an invalid assistant message")
    response_data = cast(dict[str, Any], raw_response_data)
    try:
        raw_assistant_message = response_data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("OpenRouter returned an invalid assistant message") from error
    if not isinstance(raw_assistant_message, dict):
        raise ValueError("OpenRouter returned an invalid assistant message")
    assistant_message = cast(dict[str, Any], raw_assistant_message)
    raw_tool_calls: object = assistant_message.get("tool_calls")
    if raw_tool_calls is None:
        raw_tool_calls = []
    if not isinstance(raw_tool_calls, list):
        raise ValueError("OpenRouter returned invalid tool calls")

    parsed_tool_calls: list[tuple[str, str, str]] = []
    for raw_tool_call in cast(list[object], raw_tool_calls):
        if not isinstance(raw_tool_call, dict):
            raise ValueError("OpenRouter returned an invalid tool call")
        tool_call = cast(dict[str, Any], raw_tool_call)
        try:
            tool_call_id = tool_call["id"]
            function = tool_call["function"]
            tool_name = function["name"]
            arguments_json = function["arguments"]
        except (KeyError, TypeError) as error:
            raise ValueError("OpenRouter returned an invalid tool call") from error
        if not all(isinstance(value, str) for value in (tool_call_id, tool_name, arguments_json)):
            raise ValueError("OpenRouter returned an invalid tool call")
        parsed_tool_calls.append((tool_call_id, tool_name, arguments_json))

    raw_usage: object = response_data.get("usage", {})
    raw_cost: object = (
        cast(dict[str, object], raw_usage).get("cost", 0) if isinstance(raw_usage, dict) else 0
    )
    if (
        isinstance(raw_cost, bool)
        or not isinstance(raw_cost, (int, float))
        or not math.isfinite(raw_cost)
        or raw_cost < 0
    ):
        raise ValueError("OpenRouter returned an invalid cost")
    return assistant_message, parsed_tool_calls, float(raw_cost)
