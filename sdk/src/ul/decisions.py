from __future__ import annotations

import asyncio
import hashlib
import json
import math
from types import TracebackType
from typing import Annotated, Literal, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
type Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
type DecisionText = Annotated[str, Field(min_length=1, max_length=32_000)]
type DecisionKey = Annotated[str, Field(min_length=1, max_length=200)]


class _DecisionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, hide_input_in_errors=True)


class ChoiceQuestion(_DecisionModel):
    type: Literal["choice"] = "choice"
    instructions: DecisionText
    criteria: dict[DecisionKey, DecisionText] = Field(min_length=2, max_length=100)


class NoulQuestion(_DecisionModel):
    type: Literal["noul"] = "noul"
    instructions: DecisionText
    criteria: dict[Literal["true", "false"], DecisionText] | None = None


class ScoreQuestion(_DecisionModel):
    type: Literal["score"] = "score"
    instructions: DecisionText
    criteria: tuple[DecisionText, ...] = Field(min_length=2, max_length=100)


type DecisionQuestion = Annotated[
    ChoiceQuestion | NoulQuestion | ScoreQuestion, Field(discriminator="type")
]


class ChoiceAnswer(_DecisionModel):
    type: Literal["choice"]
    choice: DecisionKey
    probabilities: dict[DecisionKey, Probability] | None = None
    confidence: Probability | None = None


class NoulAnswer(_DecisionModel):
    type: Literal["noul"]
    noul: Probability


class ScoreAnswer(_DecisionModel):
    type: Literal["score"]
    score: float = Field(ge=0, allow_inf_nan=False)
    legend: dict[DecisionKey, DecisionText] | None = None
    probabilities: dict[DecisionKey, Probability] | None = None
    confidence: Probability | None = None


type DecisionAnswer = Annotated[
    ChoiceAnswer | NoulAnswer | ScoreAnswer, Field(discriminator="type")
]


class DecisionUsage(_DecisionModel):
    input_tokens: int = Field(ge=0, strict=True)
    output_tokens: int = Field(ge=0, strict=True)
    cost: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class DecisionResponse(_DecisionModel):
    id: str = Field(min_length=1, max_length=500)
    model: str = Field(min_length=1, max_length=200)
    provider: str = Field(min_length=1, max_length=100)
    answers: dict[DecisionKey, DecisionAnswer]
    usage: DecisionUsage


class _DecisionRequest(_DecisionModel):
    state: str | dict[str, JsonValue] | list[JsonValue]
    questions: dict[DecisionKey, DecisionQuestion] = Field(min_length=1, max_length=100)


class OpenRouterDecisionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="UL_DECISION_",
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
        populate_by_name=True,
    )
    api_key: SecretStr | None = Field(
        default=None, validation_alias="OPEN_ROUTER_API_KEY", repr=False, exclude=True
    )
    model: str = Field(default="typesafe/jev-1.13", pattern=r"^typesafe/jev-[0-9][a-zA-Z0-9.-]*$")
    live_calls: bool = False
    allow_external_data_processing: bool = False
    timeout_seconds: float = Field(default=45, gt=0, le=300, allow_inf_nan=False)
    max_request_bytes: int = Field(default=64_000, ge=1_024, le=1_000_000)
    max_response_bytes: int = Field(default=256_000, ge=1_024, le=5_000_000)

    @property
    def version(self) -> str:
        value = {"endpoint": _ENDPOINT, "settings": self.model_dump(mode="json")}
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class DecisionError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OpenRouterDecisionClient:
    def __init__(
        self, settings: OpenRouterDecisionSettings, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self.settings = settings
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient(trust_env=False)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def decide(
        self,
        state: str | dict[str, JsonValue] | list[JsonValue],
        questions: dict[str, DecisionQuestion],
    ) -> DecisionResponse:
        if not self.settings.live_calls or not self.settings.allow_external_data_processing:
            raise DecisionError("Decision calls require live and external-processing opt-ins")
        credential = self.settings.api_key
        if credential is None or not credential.get_secret_value().strip():
            raise DecisionError("Decision calls require OPEN_ROUTER_API_KEY")
        key = credential.get_secret_value().strip()
        request = _DecisionRequest(state=state, questions=questions)
        for question in request.questions.values():
            if (
                isinstance(question, NoulQuestion)
                and question.criteria is not None
                and set(question.criteria) != {"true", "false"}
            ):
                raise DecisionError("Noul criteria must describe both true and false")
        body = json.dumps(
            {
                "model": self.settings.model,
                **request.model_dump(mode="json", exclude_none=True),
                "provider": {
                    "only": ["typesafe"],
                    "allow_fallbacks": False,
                    "data_collection": "deny",
                    "zdr": True,
                },
            },
            allow_nan=False,
        ).encode()
        if len(body) > self.settings.max_request_bytes:
            raise DecisionError("Decision request exceeds max_request_bytes")
        try:
            async with asyncio.timeout(self.settings.timeout_seconds):
                async with self._client.stream(
                    "POST",
                    _ENDPOINT,
                    content=body,
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                        "Accept-Encoding": "identity",
                    },
                    timeout=self.settings.timeout_seconds,
                    follow_redirects=False,
                ) as response:
                    if response.status_code != 200:
                        raise DecisionError(
                            "Decision provider returned an unsuccessful status",
                            status_code=response.status_code,
                        )
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise DecisionError("Decision response encoding is not supported")
                    content = bytearray()
                    if response.is_stream_consumed:
                        content.extend(response.content)
                    else:
                        async for chunk in response.aiter_raw():
                            content.extend(chunk)
                            if len(content) > self.settings.max_response_bytes:
                                raise DecisionError("Decision response exceeds max_response_bytes")
                    if len(content) > self.settings.max_response_bytes:
                        raise DecisionError("Decision response exceeds max_response_bytes")
            result = DecisionResponse.model_validate_json(content)
        except (httpx.HTTPError, TimeoutError):
            raise DecisionError("Decision request failed or timed out") from None
        except ValidationError:
            raise DecisionError("Decision provider returned an invalid response") from None
        if key in result.model_dump_json():
            raise DecisionError("Decision response contains the configured credential")
        if result.provider.lower() != "typesafe":
            raise DecisionError("Decision response did not honor the pinned provider")
        if result.model != self.settings.model and not result.model.startswith(
            self.settings.model + "-"
        ):
            raise DecisionError("Decision response did not honor the requested model")
        if set(result.answers) != set(request.questions):
            raise DecisionError("Decision response question IDs do not match the request")
        for name, question in request.questions.items():
            _validate_answer(question, result.answers[name])
        return result


def _validate_answer(question: DecisionQuestion, answer: DecisionAnswer) -> None:
    if question.type != answer.type:
        raise DecisionError("Decision answer type does not match the question")
    if isinstance(question, ChoiceQuestion) and isinstance(answer, ChoiceAnswer):
        if answer.choice not in question.criteria:
            raise DecisionError("Decision choice is outside the supplied criteria")
        if answer.probabilities is not None:
            _validate_probabilities(answer.probabilities, set(question.criteria))
            if answer.probabilities[answer.choice] < max(answer.probabilities.values()):
                raise DecisionError("Decision choice is not a highest-probability option")
    if isinstance(question, ScoreQuestion) and isinstance(answer, ScoreAnswer):
        if answer.score > len(question.criteria) - 1:
            raise DecisionError("Decision score is outside the supplied rubric")
        expected = {str(index): description for index, description in enumerate(question.criteria)}
        if answer.legend is not None and answer.legend != expected:
            raise DecisionError("Decision score legend does not match the supplied rubric")
        if answer.probabilities is not None:
            _validate_probabilities(answer.probabilities, set(expected))


def _validate_probabilities(probabilities: dict[str, float], expected: set[str]) -> None:
    if set(probabilities) != expected:
        raise DecisionError("Decision probabilities do not cover the supplied criteria")
    if not math.isclose(sum(probabilities.values()), 1, abs_tol=0.005 * len(expected) + 1e-9):
        raise DecisionError("Decision probabilities do not sum to one within rounding tolerance")
