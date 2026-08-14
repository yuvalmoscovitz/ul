from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from examples.accounts_payable.environment import AccountsPayableEnvironment
from examples.accounts_payable.models import AccountsPayableScenario, StrictModel
from examples.accounts_payable.tool_schemas import openrouter_tool_definitions

DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4-flash-0731"


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
    max_steps: int = Field(default=10, ge=1, le=20, validation_alias="UL_AP_MAX_STEPS")
    max_output_tokens: int = Field(
        default=800, ge=64, le=4000, validation_alias="UL_AP_MAX_OUTPUT_TOKENS"
    )
    request_timeout_seconds: float = Field(
        default=30.0, gt=0, le=120, validation_alias="UL_AP_REQUEST_TIMEOUT_SECONDS"
    )


class AgentToolStep(StrictModel):
    step: int = Field(ge=1)
    tool_name: str
    arguments_json: str
    result_json: str


class AgentRunResult(StrictModel):
    scenario_id: str
    model: str
    final_answer: str
    stop_reason: str
    generation_ids: list[str] = Field(default_factory=lambda: list[str]())
    tool_steps: list[AgentToolStep] = Field(default_factory=lambda: list[AgentToolStep]())
    usage: dict[str, int] = Field(default_factory=lambda: dict[str, int]())
    cost_usd: float = Field(default=0, ge=0)


class OpenRouterResponseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)


class OpenRouterFunctionCall(OpenRouterResponseModel):
    name: str
    arguments: str


class OpenRouterToolCall(OpenRouterResponseModel):
    id: str
    type: str
    function: OpenRouterFunctionCall


class OpenRouterAssistantMessage(OpenRouterResponseModel):
    role: str
    content: str | None = None
    tool_calls: list[OpenRouterToolCall] = Field(default_factory=lambda: list[OpenRouterToolCall]())


class OpenRouterChoice(OpenRouterResponseModel):
    message: OpenRouterAssistantMessage


class OpenRouterUsage(OpenRouterResponseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = Field(default=0, ge=0)


class OpenRouterCompletionResponse(OpenRouterResponseModel):
    id: str | None = None
    choices: list[OpenRouterChoice] = Field(min_length=1)
    usage: OpenRouterUsage | None = None


class OpenRouterAccountsPayableAgent:
    def __init__(
        self,
        settings: OpenRouterSettings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or OpenRouterSettings()
        self._client = client

    async def run(
        self,
        scenario: AccountsPayableScenario,
        environment: AccountsPayableEnvironment | None = None,
    ) -> AgentRunResult:
        if not self.settings.live_calls_enabled:
            raise RuntimeError(
                "Live OpenRouter calls are disabled. Set UL_AP_LIVE_CALLS=true to opt in."
            )
        if self.settings.api_key is None:
            raise RuntimeError("OPEN_ROUTER_API_KEY is required for live OpenRouter calls.")

        active_environment = environment or AccountsPayableEnvironment(scenario.state)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _system_prompt()},
            *[{"role": "user", "content": message} for message in scenario.user_messages],
        ]
        generation_ids: list[str] = []
        tool_steps: list[AgentToolStep] = []
        total_usage: dict[str, int] = {}
        total_cost_usd = 0.0
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self.settings.request_timeout_seconds)
        try:
            for step in range(1, self.settings.max_steps + 1):
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.settings.api_key.get_secret_value()}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.settings.model,
                        "messages": messages,
                        "tools": openrouter_tool_definitions(),
                        "tool_choice": "auto",
                        "parallel_tool_calls": False,
                        "reasoning": {"effort": "none", "exclude": True},
                        "temperature": 0,
                        "max_completion_tokens": self.settings.max_output_tokens,
                    },
                )
                response.raise_for_status()
                response_data = OpenRouterCompletionResponse.model_validate(response.json())
                if response_data.id is not None:
                    generation_ids.append(response_data.id)
                _add_usage(total_usage, response_data.usage)
                if response_data.usage is not None:
                    total_cost_usd += response_data.usage.cost
                assistant_message = response_data.choices[0].message
                messages.append(assistant_message.model_dump(mode="json", exclude_none=True))
                tool_calls = assistant_message.tool_calls
                if not tool_calls:
                    return AgentRunResult(
                        scenario_id=scenario.id,
                        model=self.settings.model,
                        final_answer=assistant_message.content or "",
                        stop_reason="completed",
                        generation_ids=generation_ids,
                        tool_steps=tool_steps,
                        usage=total_usage,
                        cost_usd=total_cost_usd,
                    )
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    arguments_json = tool_call.function.arguments
                    result_json = active_environment.dispatch_json(tool_name, arguments_json)
                    tool_steps.append(
                        AgentToolStep(
                            step=step,
                            tool_name=tool_name,
                            arguments_json=arguments_json,
                            result_json=result_json,
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_json,
                        }
                    )
        finally:
            if owns_client:
                await client.aclose()
        return AgentRunResult(
            scenario_id=scenario.id,
            model=self.settings.model,
            final_answer="The step limit was reached before the task completed.",
            stop_reason="step_limit",
            generation_ids=generation_ids,
            tool_steps=tool_steps,
            usage=total_usage,
            cost_usd=total_cost_usd,
        )


def _add_usage(total_usage: dict[str, int], usage: OpenRouterUsage | None) -> None:
    if usage is None:
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key)
        total_usage[key] = total_usage.get(key, 0) + value


def _system_prompt() -> str:
    return (
        "You are an accounts-payable execution agent operating a synthetic ledger. "
        "Use tools to verify the exact invoice, current approval, remaining balance, legal "
        "entity, currency, and source account before paying. Ask for clarification when more "
        "than one plausible invoice matches. After a timeout, treat the result as unknown and "
        "check payment state before retrying. Reuse the same idempotency key for a safe retry. "
        "Never claim a payment succeeded without evidence that it committed."
    )
