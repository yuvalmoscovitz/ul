from __future__ import annotations

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

    try:
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
    finally:
        if owns_client:
            await active_client.aclose()

    try:
        raw_assistant_message = response.json()["choices"][0]["message"]
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

    parsed_tool_calls: list[tuple[str, str]] = []
    for raw_tool_call in cast(list[object], raw_tool_calls):
        if not isinstance(raw_tool_call, dict):
            raise ValueError("OpenRouter returned an invalid tool call")
        tool_call = cast(dict[str, Any], raw_tool_call)
        try:
            function = tool_call["function"]
            tool_name = function["name"]
            arguments_json = function["arguments"]
        except (KeyError, TypeError) as error:
            raise ValueError("OpenRouter returned an invalid tool call") from error
        if not isinstance(tool_name, str) or not isinstance(arguments_json, str):
            raise ValueError("OpenRouter returned an invalid tool call")
        parsed_tool_calls.append((tool_name, arguments_json))

    tool_steps: list[AgentToolStep] = []
    for tool_name, arguments_json in parsed_tool_calls:
        result_json = environment.dispatch_json(tool_name, arguments_json)
        tool_steps.append(AgentToolStep(tool_name, arguments_json, result_json))

    content = assistant_message.get("content")
    return AgentRunResult(content if isinstance(content, str) else "", tuple(tool_steps))
