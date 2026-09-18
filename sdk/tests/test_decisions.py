from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from pydantic import SecretStr, ValidationError
from ul import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionError,
    NoulAnswer,
    NoulQuestion,
    OpenRouterDecisionClient,
    OpenRouterDecisionSettings,
    ScoreAnswer,
    ScoreQuestion,
)


def settings(**changes):
    return OpenRouterDecisionSettings(
        **{
            "api_key": SecretStr("private-test-credential"),
            "live_calls": True,
            "allow_external_data_processing": True,
            **changes,
        }
    )


def questions():
    return {
        "route": ChoiceQuestion(
            instructions="Select a team.", criteria={"billing": "Payments", "technical": "Bugs"}
        ),
        "refund": NoulQuestion(instructions="Is a refund requested?"),
        "severity": ScoreQuestion(instructions="Rate severity.", criteria=("Low", "High")),
    }


def response_body():
    return {
        "id": "decision-1",
        "model": "typesafe/jev-1.13-20260917",
        "provider": "TypeSafe",
        "answers": {
            "route": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.95, "technical": 0.05},
                "confidence": 0.9,
            },
            "refund": {"type": "noul", "noul": 0.97},
            "severity": {
                "type": "score",
                "score": 0.3,
                "probabilities": {"0": 0.7, "1": 0.3},
                "legend": {"0": "Low", "1": "High"},
                "confidence": 0.4,
            },
        },
        "usage": {"input_tokens": 120, "output_tokens": 30, "cost": 0.00000504},
    }


@pytest.mark.asyncio
async def test_public_sdk_batches_all_primitives_with_pinned_private_routing():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=response_body())

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        async with OpenRouterDecisionClient(settings(), client=transport) as client:
            result = await client.decide({"text": "Refund my charge."}, questions())
        assert not transport.is_closed
    assert isinstance(result.answers["route"], ChoiceAnswer)
    assert isinstance(result.answers["refund"], NoulAnswer)
    assert isinstance(result.answers["severity"], ScoreAnswer)
    assert result.usage.cost == 0.00000504
    request = requests[0]
    assert str(request.url) == "https://openrouter.ai/api/alpha/decisions"
    assert request.headers["Authorization"] == "Bearer private-test-credential"
    body = json.loads(request.content)
    assert body["state"] == {"text": "Refund my charge."}
    assert set(body["questions"]) == {"route", "refund", "severity"}
    assert body["provider"] == {
        "only": ["typesafe"],
        "allow_fallbacks": False,
        "data_collection": "deny",
        "zdr": True,
    }
    assert "criteria" not in body["questions"]["refund"]
    assert body["questions"]["severity"]["criteria"] == ["Low", "High"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"live_calls": False},
        {"allow_external_data_processing": False},
        {"api_key": None},
        {"api_key": SecretStr(" ")},
    ],
)
async def test_opt_ins_and_credentials_fail_before_network(changes):
    def reject(request):
        pytest.fail("Network was reached")

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(reject)) as transport,
        OpenRouterDecisionClient(settings(**changes), client=transport) as client,
    ):
        with pytest.raises(DecisionError):
            await client.decide("text", questions())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "provider",
        "model",
        "missing_answer",
        "extra_answer",
        "wrong_type",
        "unknown_choice",
        "missing_probability",
        "bad_probability",
        "nan",
        "wrong_winner",
        "high_score",
        "wrong_legend",
        "bad_cost",
        "credential_echo",
        "malformed_json",
    ],
)
async def test_invalid_provider_responses_fail_closed_without_payload_disclosure(change):
    body = response_body()
    answer = body["answers"]["route"]
    if change == "provider":
        body["provider"] = "another-provider"
    elif change == "model":
        body["model"] = "typesafe/jev-9.99"
    elif change == "missing_answer":
        del body["answers"]["refund"]
    elif change == "extra_answer":
        body["answers"]["extra"] = body["answers"]["refund"]
    elif change == "wrong_type":
        body["answers"]["route"] = body["answers"]["refund"]
    elif change == "unknown_choice":
        answer["choice"] = "unknown"
    elif change == "missing_probability":
        del answer["probabilities"]["technical"]
    elif change == "bad_probability":
        answer["probabilities"]["technical"] = 0.9
    elif change == "nan":
        answer["confidence"] = "NaN"
    elif change == "wrong_winner":
        answer["choice"] = "technical"
    elif change == "high_score":
        body["answers"]["severity"]["score"] = 2
    elif change == "wrong_legend":
        body["answers"]["severity"]["legend"]["0"] = "High"
    elif change == "bad_cost":
        body["usage"]["cost"] = -1
    elif change == "credential_echo":
        body["id"] = "private-test-credential"
    encoded = b"private payload" if change == "malformed_json" else json.dumps(body).encode()
    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=encoded))
        ) as transport,
        OpenRouterDecisionClient(settings(), client=transport) as client,
    ):
        with pytest.raises(DecisionError) as error:
            await client.decide("private payload", questions())
    assert "private payload" not in str(error.value)
    assert "private-test-credential" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [302, 401, 429, 500])
async def test_http_errors_are_safe_and_never_retried_or_redirected(status):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(
            status,
            headers={"Location": "https://other.example"},
            text="private payload private-test-credential",
        )

    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(respond), follow_redirects=True
        ) as transport,
        OpenRouterDecisionClient(settings(), client=transport) as client,
    ):
        with pytest.raises(DecisionError) as error:
            await client.decide("text", questions())
    assert error.value.status_code == status
    assert "private" not in str(error.value)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_request_and_streamed_response_limits():
    class Chunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"x" * 600
            yield b"x" * 600
            pytest.fail("Read past the response limit")

    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, stream=Chunks())

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport,
        OpenRouterDecisionClient(
            settings(max_request_bytes=1024, max_response_bytes=1024), client=transport
        ) as client,
    ):
        with pytest.raises(DecisionError, match="max_request_bytes"):
            await client.decide("x" * 2000, questions())
        assert calls == []
        with pytest.raises(DecisionError, match="max_response_bytes"):
            await client.decide("text", questions())


@pytest.mark.asyncio
async def test_total_timeout_is_enforced():
    async def respond(request):
        await asyncio.sleep(1)
        return httpx.Response(200, json=response_body())

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport,
        OpenRouterDecisionClient(settings(timeout_seconds=0.01), client=transport) as client,
    ):
        with pytest.raises(DecisionError, match="timed out"):
            await client.decide("text", questions())


@pytest.mark.asyncio
async def test_absent_optional_usage_and_confidence_remain_unknown():
    body = response_body()
    del body["usage"]["cost"]
    for key in ("confidence", "probabilities"):
        del body["answers"]["route"][key]
    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
        ) as transport,
        OpenRouterDecisionClient(settings(), client=transport) as client,
    ):
        result = await client.decide("text", questions())
    assert result.usage.cost is None
    assert result.answers["route"].confidence is None
    assert result.answers["route"].probabilities is None


def test_settings_fingerprint_excludes_credentials_and_changes_with_model():
    assert settings().version == settings(api_key=SecretStr("another-key")).version
    assert settings().version != settings(model="typesafe/jev-1.14").version
    assert "private-test-credential" not in repr(settings())
    assert "private-test-credential" not in settings().model_dump_json()
    with pytest.raises(ValidationError):
        settings(model="typesafe/jev-latest")
