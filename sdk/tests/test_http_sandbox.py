from __future__ import annotations

import asyncio
import json
import os
import socket
import ssl
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import JsonValue, ValidationError
from ul.http_sandbox import (
    JsonHttpSandboxConfig,
    JsonHttpSandboxConnection,
    json_http_sandbox_calls_per_conversation,
    load_json_http_sandbox_config,
    validate_json_http_sandbox_configuration,
)
from ul_core.dataset import SandboxSetupFixture
from ul_core.evaluation import EvaluationCase, TimeoutAfterCommitEventRequest
from ul_core.models import ConversationRole, ConversationTurn

pytestmark = pytest.mark.asyncio


class _StaticResponseStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._body


def _raw_response(body: bytes, *, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        stream=_StaticResponseStream(body),
        headers={"content-type": "application/json"},
    )


def _config(
    base_url: str = "https://sandbox.example.test",
    *,
    headers_from_env: dict[str, str] | None = None,
    clean_state_value: JsonValue = True,
    setup: bool = True,
    timeout_after_commit: bool = False,
) -> JsonHttpSandboxConfig:
    raw: dict[str, Any] = {
        "version": 3,
        "sandbox_id": "payments-test",
        "headers_from_env": headers_from_env or {},
        "reset": {
            "url": f"{base_url}/reset",
            "request_json_template": {"case_id": "{{case_id}}"},
            "case_id_json_pointer": "/case_id",
            "sandbox_id_json_pointer": "/sandbox_id",
            "generation_json_pointer": "/generation",
            "clean_state_json_pointer": "/clean",
            "clean_state_value": clean_state_value,
        },
        "execute_turn": {
            "url": f"{base_url}/execute",
            "request_json_template": {
                "case_id": "{{case_id}}",
                "turn_id": "{{turn_id}}",
                "input": "{{input}}",
            },
            "response_json_pointer": "/agent_response",
            "case_id_json_pointer": "/case_id",
            "turn_id_json_pointer": "/turn_id",
            "sandbox_id_json_pointer": "/sandbox_id",
        },
        "snapshot": {
            "url": f"{base_url}/snapshot",
            "request_json_template": {
                "case_id": "{{case_id}}",
                "turn_id": "{{turn_id}}",
            },
            "response_json_pointer": "/state",
            "case_id_json_pointer": "/case_id",
            "turn_id_json_pointer": "/turn_id",
            "sandbox_id_json_pointer": "/sandbox_id",
        },
    }
    if setup:
        raw["setup"] = {
            "url": f"{base_url}/setup",
            "request_json_template": {
                "case_id": "{{case_id}}",
                "starting_amount": 100,
            },
            "case_id_json_pointer": "/case_id",
            "sandbox_id_json_pointer": "/sandbox_id",
        }
    if timeout_after_commit:
        raw["timeout_after_commit"] = {
            "operator_id": "environment.tool.timeout_after_commit",
            "version": "1.0.0",
            "url": f"{base_url}/timeout-after-commit",
        }
    return JsonHttpSandboxConfig.model_validate(raw)


def _case(*inputs: str, max_calls: int = 20) -> EvaluationCase:
    return EvaluationCase(
        id="case-1",
        turns=tuple(
            ConversationTurn(
                id=f"turn-{index}",
                role=ConversationRole.USER,
                content=content,
            )
            for index, content in enumerate(inputs, start=1)
        ),
        max_sandbox_api_calls=max_calls,
        timeout_seconds=30,
    )


def _timeout_after_commit_case(*, max_calls: int = 9) -> EvaluationCase:
    return _case("pay once", max_calls=max_calls).model_copy(
        update={
            "timeout_after_commit_event": TimeoutAfterCommitEventRequest(
                event_id="lost-payment-ack",
                turn_id="turn-1",
                action_id="execute-payment",
            )
        }
    )


def _successful_handler() -> tuple[Any, list[tuple[str, dict[str, JsonValue]]]]:
    requests: list[tuple[str, dict[str, JsonValue]]] = []
    generation = 0
    amount = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal amount, generation
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path == "/reset":
            generation += 1
            amount = 0
            return _raw_response(
                json.dumps(
                    {
                        "sandbox_id": "payments-test",
                        "case_id": body["case_id"],
                        "generation": generation,
                        "clean": True,
                    }
                ).encode()
            )
        if request.url.path == "/setup":
            amount = 100
            return _raw_response(
                json.dumps({"sandbox_id": "payments-test", "case_id": body["case_id"]}).encode()
            )
        if request.url.path == "/execute":
            amount += 50
            return _raw_response(
                json.dumps(
                    {
                        "sandbox_id": "payments-test",
                        "case_id": body["case_id"],
                        "turn_id": body["turn_id"],
                        "agent_response": {"input": body["input"]},
                    }
                ).encode()
            )
        return _raw_response(
            json.dumps(
                {
                    "sandbox_id": "payments-test",
                    "case_id": body["case_id"],
                    "turn_id": body["turn_id"],
                    "state": {"committed_amount": amount},
                }
            ).encode()
        )

    return handler, requests


async def test_executes_remote_case_and_returns_explicit_evidence() -> None:
    handler, requests = _successful_handler()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            _config(), sandbox_confirmed=True, max_sandbox_api_calls=6, client=client
        )
        evidence = await sandbox.execute(_case("increase the amount", max_calls=6))

    assert [path for path, _ in requests] == [
        "/reset",
        "/setup",
        "/snapshot",
        "/execute",
        "/snapshot",
        "/reset",
    ]
    assert {body["case_id"] for _, body in requests} == {"case-1"}
    assert requests[2][1]["turn_id"] == "__ul_initial_state__"
    assert requests[3][1]["turn_id"] == "turn-1"
    assert requests[4][1]["turn_id"] == "turn-1"
    assert evidence.lifecycle.terminal_status == "succeeded"
    assert evidence.sandbox_id == "payments-test"
    assert evidence.initial_state is not None
    assert evidence.initial_state.value == {"committed_amount": 100}
    assert evidence.turns[0].response == {"input": "increase the amount"}
    assert evidence.turns[0].state_snapshot == {"committed_amount": 150}
    assert evidence.turns[0].state_observation_authority == "sandbox_self_reported"


async def test_substitutes_per_record_setup_and_binds_its_digest_to_evidence() -> None:
    fixture = SandboxSetupFixture.from_payload(
        {"account": {"id": "AC-100", "available": 150}, "approval": True}
    )
    config = _config()
    assert config.setup is not None
    config = config.model_copy(
        update={
            "setup": config.setup.model_copy(
                update={
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "fixture": "{{sandbox_setup}}",
                    }
                }
            )
        }
    )
    handler, requests = _successful_handler()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            config, sandbox_confirmed=True, max_sandbox_api_calls=6, client=client
        )
        evidence = await sandbox.execute(
            _case("increase the amount", max_calls=6).model_copy(update={"sandbox_setup": fixture})
        )

    assert requests[1][0] == "/setup"
    assert requests[1][1]["fixture"] == fixture.payload
    assert evidence.sandbox_setup_sha256 == fixture.sha256


async def test_per_record_setup_requires_an_exact_template_case_match() -> None:
    fixture = SandboxSetupFixture.from_payload({"account": "AC-100"})
    static_sandbox = JsonHttpSandboxConnection.from_config(
        _config(), sandbox_confirmed=True, client=httpx.AsyncClient()
    )
    per_record_config = _config()
    assert per_record_config.setup is not None
    per_record_config = per_record_config.model_copy(
        update={
            "setup": per_record_config.setup.model_copy(
                update={
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "fixture": "{{sandbox_setup}}",
                    }
                }
            )
        }
    )
    per_record_sandbox = JsonHttpSandboxConnection.from_config(
        per_record_config, sandbox_confirmed=True, client=httpx.AsyncClient()
    )
    try:
        with pytest.raises(ValueError, match="does not contain"):
            static_sandbox.api_calls_for_case(
                _case("pay").model_copy(update={"sandbox_setup": fixture})
            )
        with pytest.raises(ValueError, match="requires a sandbox setup fixture"):
            per_record_sandbox.api_calls_for_case(_case("pay"))
    finally:
        await static_sandbox.aclose()
        await per_record_sandbox.aclose()


async def test_per_record_setup_rejects_payload_mutation_after_hashing() -> None:
    fixture = SandboxSetupFixture.from_payload({"account": "AC-100"})
    fixture.payload["account"] = "mutated"
    config = _config()
    assert config.setup is not None
    config = config.model_copy(
        update={
            "setup": config.setup.model_copy(
                update={
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "fixture": "{{sandbox_setup}}",
                    }
                }
            )
        }
    )
    sandbox = JsonHttpSandboxConnection.from_config(
        config, sandbox_confirmed=True, client=httpx.AsyncClient()
    )
    try:
        with pytest.raises(ValueError, match="digest must match"):
            sandbox.api_calls_for_case(_case("pay").model_copy(update={"sandbox_setup": fixture}))
    finally:
        await sandbox.aclose()


async def test_timeout_after_commit_receipts_are_correlated_and_budgeted() -> None:
    handler, _ = _successful_handler()

    def event_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/timeout-after-commit":
            return handler(request)
        body = json.loads(request.content)
        status = {"arm": "armed", "observe": "fired", "clean": "cleaned"}[body["operation"]]
        return _raw_response(json.dumps({**body, "status": status}).encode())

    async with httpx.AsyncClient(transport=httpx.MockTransport(event_handler)) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            _config(timeout_after_commit=True),
            sandbox_confirmed=True,
            max_sandbox_api_calls=9,
            client=client,
        )
        evidence = await sandbox.execute(_timeout_after_commit_case())

    assert sandbox.api_calls_for_case(_timeout_after_commit_case()) == 9
    assert evidence.lifecycle.terminal_status == "succeeded"
    event = evidence.timeout_after_commit_event
    assert event is not None
    assert event.requested is True
    assert event.armed is True
    assert event.trigger_status == "fired"
    assert event.cleaned is True


async def test_timeout_after_commit_reserves_control_calls_before_network() -> None:
    handler, requests = _successful_handler()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            _config(timeout_after_commit=True),
            sandbox_confirmed=True,
            max_sandbox_api_calls=8,
            client=client,
        )
        with pytest.raises(RuntimeError, match="API call budget exhausted"):
            await sandbox.execute(_timeout_after_commit_case())

    assert requests == []


async def test_stale_timeout_after_commit_receipt_fails_and_cleans_by_event_identity() -> None:
    handler, _ = _successful_handler()
    operations: list[str] = []

    def stale_event_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/timeout-after-commit":
            return handler(request)
        body = json.loads(request.content)
        operation = body["operation"]
        operations.append(operation)
        response = {
            **body,
            "status": {"arm": "armed", "observe": "fired", "clean": "cleaned"}[operation],
        }
        if operation == "observe":
            response["event_id"] = "stale-event"
        return _raw_response(json.dumps(response).encode())

    async with httpx.AsyncClient(transport=httpx.MockTransport(stale_event_handler)) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            _config(timeout_after_commit=True),
            sandbox_confirmed=True,
            max_sandbox_api_calls=9,
            client=client,
        )
        evidence = await sandbox.execute(_timeout_after_commit_case())

    assert operations == ["arm", "observe", "clean"]
    assert evidence.lifecycle.failed_phase == "observe_timeout_after_commit"
    assert evidence.lifecycle.delivery == "uncertain"
    assert evidence.lifecycle.sandbox_state_uncertain is True
    event = evidence.timeout_after_commit_event
    assert event is not None
    assert event.armed is True
    assert event.trigger_status == "unknown"
    assert event.cleaned is True


async def test_timeout_event_cleanup_cancellation_still_attempts_reset_and_quarantines() -> None:
    handler, requests = _successful_handler()
    operations: list[str] = []

    def cancelled_cleanup_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/timeout-after-commit":
            return handler(request)
        body = json.loads(request.content)
        operation = body["operation"]
        operations.append(operation)
        if operation == "clean":
            raise asyncio.CancelledError
        status = "armed" if operation == "arm" else "fired"
        return _raw_response(json.dumps({**body, "status": status}).encode())

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(cancelled_cleanup_handler)
    ) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            _config(timeout_after_commit=True),
            sandbox_confirmed=True,
            max_sandbox_api_calls=9,
            client=client,
        )
        with pytest.raises(asyncio.CancelledError):
            await sandbox.execute(_timeout_after_commit_case())
        request_count_after_cancellation = len(requests)
        blocked = await sandbox.execute(_timeout_after_commit_case())

    assert operations == ["arm", "observe", "clean"]
    assert [path for path, _ in requests][-1] == "/reset"
    assert len(requests) == request_count_after_cancellation
    assert blocked.lifecycle.failed_phase == "blocked_state_uncertain"
    assert blocked.lifecycle.sandbox_state_uncertain is True


async def test_preserves_state_within_case_and_resets_between_cases() -> None:
    handler, requests = _successful_handler()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            _config(), sandbox_confirmed=True, max_sandbox_api_calls=14, client=client
        )
        first = await sandbox.execute(_case("first", "correction", max_calls=8))
        second = await sandbox.execute(
            EvaluationCase(
                id="case-2",
                turns=(
                    ConversationTurn(id="turn-1", role=ConversationRole.USER, content="new case"),
                ),
                max_sandbox_api_calls=6,
                timeout_seconds=30,
            )
        )

    assert [turn.state_snapshot for turn in first.turns] == [
        {"committed_amount": 150},
        {"committed_amount": 200},
    ]
    assert second.turns[0].state_snapshot == {"committed_amount": 150}
    assert json_http_sandbox_calls_per_conversation(_config(), 2) == 8
    assert [body["case_id"] for _, body in requests].count("case-2") == 6


async def test_cleanup_failure_is_evidence_and_quarantines_connection() -> None:
    resets = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal resets
        body = json.loads(request.content)
        if request.url.path == "/reset":
            resets += 1
            if resets == 2:
                return httpx.Response(500)
            return _raw_response(
                json.dumps(
                    {
                        "sandbox_id": "payments-test",
                        "case_id": body["case_id"],
                        "generation": 1,
                        "clean": True,
                    }
                ).encode()
            )
        if request.url.path == "/setup":
            return _raw_response(
                json.dumps({"sandbox_id": "payments-test", "case_id": body["case_id"]}).encode()
            )
        if request.url.path == "/execute":
            return _raw_response(
                json.dumps(
                    {
                        "sandbox_id": "payments-test",
                        "case_id": body["case_id"],
                        "turn_id": body["turn_id"],
                        "agent_response": {"ok": True},
                    }
                ).encode()
            )
        return _raw_response(
            json.dumps(
                {
                    "sandbox_id": "payments-test",
                    "case_id": body["case_id"],
                    "turn_id": body["turn_id"],
                    "state": {},
                }
            ).encode()
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            _config(), sandbox_confirmed=True, max_sandbox_api_calls=10, client=client
        )
        failed = await sandbox.execute(_case("work", max_calls=6))
        blocked = await sandbox.execute(_case("again", max_calls=6))

    assert failed.lifecycle.cleanup == "failed"
    assert failed.lifecycle.sandbox_state_uncertain is True
    assert blocked.lifecycle.failed_phase == "blocked_state_uncertain"


async def test_ambiguous_execute_delivery_stays_quarantined_after_cleanup() -> None:
    requests: list[str] = []
    generation = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal generation
        requests.append(request.url.path)
        body = json.loads(request.content)
        if request.url.path == "/reset":
            generation += 1
            return _raw_response(
                json.dumps(
                    {
                        "sandbox_id": "payments-test",
                        "case_id": body["case_id"],
                        "generation": generation,
                        "clean": True,
                    }
                ).encode()
            )
        if request.url.path == "/setup":
            return _raw_response(
                json.dumps({"sandbox_id": "payments-test", "case_id": body["case_id"]}).encode()
            )
        if request.url.path == "/execute":
            raise httpx.ReadTimeout("response lost after delivery", request=request)
        if request.url.path == "/snapshot" and body["turn_id"] == "__ul_initial_state__":
            return _raw_response(
                json.dumps(
                    {
                        "sandbox_id": "payments-test",
                        "case_id": body["case_id"],
                        "turn_id": body["turn_id"],
                        "state": {},
                    }
                ).encode()
            )
        raise AssertionError("snapshot must not follow uncertain execution")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            _config(), sandbox_confirmed=True, max_sandbox_api_calls=10, client=client
        )
        failed = await sandbox.execute(_case("work", max_calls=6))
        blocked = await sandbox.execute(_case("again", max_calls=6))

    assert failed.lifecycle.delivery == "uncertain"
    assert failed.lifecycle.failure_code == "response_timeout"
    assert failed.lifecycle.failure_reason == "sandbox API response timed out"
    assert failed.lifecycle.sandbox_state_uncertain is True
    assert blocked.lifecycle.failed_phase == "blocked_state_uncertain"
    assert requests == ["/reset", "/setup", "/snapshot", "/execute", "/reset"]


@pytest.mark.parametrize(
    ("cause", "expected_code", "expected_reason"),
    (
        (socket.gaierror(), "dns_resolution", "sandbox API DNS resolution failed"),
        (ssl.SSLError(), "tls_connection", "sandbox API TLS connection failed"),
    ),
)
async def test_connect_failures_retain_safe_category(
    cause: BaseException, expected_code: str, expected_reason: str
) -> None:
    successful_handler, _ = _successful_handler()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/execute":
            raise httpx.ConnectError("private detail", request=request) from cause
        return successful_handler(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            _config(), sandbox_confirmed=True, max_sandbox_api_calls=6, client=client
        )
        evidence = await sandbox.execute(_case("work", max_calls=6))

    assert evidence.lifecycle.failure_code == expected_code
    assert evidence.lifecycle.failure_reason == expected_reason
    assert evidence.lifecycle.delivery == "certain"
    assert evidence.lifecycle.sandbox_state_uncertain is False
    assert "private detail" not in evidence.model_dump_json()


@pytest.mark.parametrize(
    ("error_kind", "expected_code"),
    (
        ("connect", "connect_timeout"),
        ("pool", "pool_timeout"),
        ("dns", "dns_resolution"),
    ),
)
async def test_pre_delivery_initial_reset_failure_does_not_quarantine_or_cleanup(
    error_kind: str, expected_code: str
) -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if error_kind == "connect":
            raise httpx.ConnectTimeout("private detail", request=request)
        if error_kind == "pool":
            raise httpx.PoolTimeout("private detail", request=request)
        raise httpx.ConnectError("private detail", request=request) from socket.gaierror()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            _config(), sandbox_confirmed=True, max_sandbox_api_calls=6, client=client
        )
        evidence = await sandbox.execute(_case("work", max_calls=6))

    assert evidence.lifecycle.failed_phase == "reset"
    assert evidence.lifecycle.failure_code == expected_code
    assert evidence.lifecycle.delivery == "certain"
    assert evidence.lifecycle.cleanup == "not_attempted"
    assert evidence.lifecycle.sandbox_state_uncertain is False
    assert requests == ["/reset"]


async def test_ambiguous_initial_reset_is_not_retried_as_cleanup() -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        raise httpx.ReadTimeout("private detail", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            _config(), sandbox_confirmed=True, max_sandbox_api_calls=6, client=client
        )
        evidence = await sandbox.execute(_case("work", max_calls=6))

    assert evidence.lifecycle.failed_phase == "reset"
    assert evidence.lifecycle.failure_code == "response_timeout"
    assert evidence.lifecycle.delivery == "uncertain"
    assert evidence.lifecycle.cleanup == "not_attempted"
    assert evidence.lifecycle.sandbox_state_uncertain is True
    assert requests == ["/reset"]


async def test_unexpected_initial_reset_error_is_redacted_and_not_retried() -> None:
    secret = "private-value-error-detail"
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        raise ValueError(secret)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            _config(), sandbox_confirmed=True, max_sandbox_api_calls=6, client=client
        )
        evidence = await sandbox.execute(_case("work", max_calls=6))
        blocked = await sandbox.execute(_case("again", max_calls=6))

    assert evidence.lifecycle.failed_phase == "reset"
    assert evidence.lifecycle.failure_code == "sandbox_lifecycle_error"
    assert evidence.lifecycle.delivery == "uncertain"
    assert evidence.lifecycle.cleanup == "not_attempted"
    assert evidence.lifecycle.sandbox_state_uncertain is True
    assert secret not in evidence.model_dump_json()
    assert blocked.lifecycle.failed_phase == "blocked_state_uncertain"
    assert requests == ["/reset"]


async def test_oversized_initial_reset_request_is_known_not_delivered() -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        raise AssertionError("oversized request must not reach the transport")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            _config(),
            sandbox_confirmed=True,
            max_request_bytes=1,
            max_sandbox_api_calls=12,
            client=client,
        )
        first = await sandbox.execute(_case("work", max_calls=6))
        second = await sandbox.execute(_case("again", max_calls=6))

    assert first.lifecycle.failure_code == "request_too_large"
    assert first.lifecycle.delivery == "certain"
    assert first.lifecycle.cleanup == "not_attempted"
    assert first.lifecycle.sandbox_state_uncertain is False
    assert second.lifecycle.failed_phase == "reset"
    assert requests == []


async def test_null_snapshot_is_a_safe_protocol_failure() -> None:
    successful_handler, requests = _successful_handler()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/snapshot" and body["turn_id"] != "__ul_initial_state__":
            requests.append((request.url.path, body))
            return _raw_response(
                json.dumps(
                    {
                        "sandbox_id": "payments-test",
                        "case_id": body["case_id"],
                        "turn_id": body["turn_id"],
                        "state": None,
                    }
                ).encode()
            )
        return successful_handler(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            _config(), sandbox_confirmed=True, max_sandbox_api_calls=6, client=client
        )
        evidence = await sandbox.execute(_case("work", max_calls=6))

    assert evidence.lifecycle.failed_phase == "snapshot"
    assert evidence.lifecycle.failure_code == "response_mapping"
    assert evidence.lifecycle.failure_reason == ("sandbox API response JSON pointer selected null")
    assert evidence.lifecycle.cleanup == "succeeded"
    assert evidence.final_response is None
    assert [path for path, _ in requests][-1] == "/reset"


async def test_ambiguous_cleanup_reset_marks_delivery_uncertain() -> None:
    successful_handler, _ = _successful_handler()
    reset_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal reset_calls
        if request.url.path == "/reset":
            reset_calls += 1
            if reset_calls == 2:
                raise httpx.ReadTimeout("private detail", request=request)
        return successful_handler(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            _config(), sandbox_confirmed=True, max_sandbox_api_calls=6, client=client
        )
        evidence = await sandbox.execute(_case("work", max_calls=6))

    assert evidence.lifecycle.failed_phase == "cleanup_reset"
    assert evidence.lifecycle.delivery == "uncertain"
    assert evidence.lifecycle.cleanup_failure_reason == "sandbox API response timed out"
    assert evidence.lifecycle.cleanup_failure_code == "response_timeout"
    assert evidence.lifecycle.sandbox_state_uncertain is True


async def test_unexpected_runtime_error_detail_is_not_persisted() -> None:
    secret = "private-transport-detail"
    successful_handler, _ = _successful_handler()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/execute":
            raise RuntimeError(secret)
        return successful_handler(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            _config(), sandbox_confirmed=True, max_sandbox_api_calls=6, client=client
        )
        evidence = await sandbox.execute(_case("work", max_calls=6))
        blocked = await sandbox.execute(_case("again", max_calls=6))

    assert evidence.lifecycle.failed_phase == "execute_turn"
    assert evidence.lifecycle.failure_code == "sandbox_lifecycle_error"
    assert evidence.lifecycle.failure_reason == "sandbox lifecycle failed"
    assert evidence.lifecycle.delivery == "uncertain"
    assert evidence.lifecycle.sandbox_state_uncertain is True
    assert blocked.lifecycle.failed_phase == "blocked_state_uncertain"
    assert secret not in evidence.model_dump_json()


async def test_sandbox_identity_mismatch_stops_before_execute() -> None:
    requests: list[str] = []
    reset_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal reset_calls
        requests.append(request.url.path)
        reset_calls += 1
        return _raw_response(
            json.dumps(
                {"sandbox_id": "production", "generation": reset_calls, "clean": True}
            ).encode()
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            _config(), sandbox_confirmed=True, max_sandbox_api_calls=6, client=client
        )
        evidence = await sandbox.execute(_case("work", max_calls=6))

    assert evidence.lifecycle.failed_phase == "reset"
    assert evidence.lifecycle.failure_reason == (
        "HTTP sandbox identity did not match its configuration"
    )
    assert evidence.lifecycle.sandbox_state_uncertain is True
    assert evidence.lifecycle.cleanup == "not_attempted"
    assert requests == ["/reset"]


@pytest.mark.parametrize(
    ("wrong_field", "failed_phase"), (("case_id", "execute_turn"), ("turn_id", "execute_turn"))
)
async def test_stale_execute_response_is_rejected_and_quarantined(
    wrong_field: str, failed_phase: str
) -> None:
    generation = 0

    def stale_handler(request: httpx.Request) -> httpx.Response:
        nonlocal generation
        request_body = json.loads(request.content)
        response_body: dict[str, JsonValue] = {
            "sandbox_id": "payments-test",
            "case_id": request_body["case_id"],
        }
        if request.url.path == "/reset":
            generation += 1
            response_body.update({"generation": generation, "clean": True})
        elif request.url.path == "/execute":
            response_body.update(
                {"turn_id": request_body["turn_id"], "agent_response": {"ok": True}}
            )
            response_body[wrong_field] = "stale-identity"
        elif request.url.path == "/snapshot":
            response_body.update({"turn_id": request_body["turn_id"], "state": {}})
        return _raw_response(json.dumps(response_body).encode())

    async with httpx.AsyncClient(transport=httpx.MockTransport(stale_handler)) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            _config(), sandbox_confirmed=True, max_sandbox_api_calls=10, client=client
        )
        failed = await sandbox.execute(_case("work", max_calls=6))
        blocked = await sandbox.execute(_case("again", max_calls=6))

    assert failed.lifecycle.failed_phase == failed_phase
    assert failed.lifecycle.delivery == "uncertain"
    assert failed.lifecycle.sandbox_state_uncertain is True
    assert blocked.lifecycle.failed_phase == "blocked_state_uncertain"


async def test_reserves_complete_budget_before_network() -> None:
    handler, requests = _successful_handler()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            _config(), sandbox_confirmed=True, max_sandbox_api_calls=5, client=client
        )
        with pytest.raises(RuntimeError, match="API call budget exhausted"):
            await sandbox.execute(_case("work", max_calls=6))
    assert requests == []


async def test_loader_and_headers_do_not_persist_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UL_SANDBOX_TEST_AGENT_TOKEN", "Bearer private-token")
    config_path = tmp_path / "sandbox.json"
    config_path.write_text(
        _config(
            headers_from_env={"Authorization": "UL_SANDBOX_TEST_AGENT_TOKEN"}
        ).model_dump_json(),
        encoding="utf-8",
    )
    authorizations: list[str | None] = []
    handler, _ = _successful_handler()

    def recording_handler(request: httpx.Request) -> httpx.Response:
        authorizations.append(request.headers.get("authorization"))
        return handler(request)

    config = load_json_http_sandbox_config(config_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(recording_handler)) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            config, sandbox_confirmed=True, max_sandbox_api_calls=6, client=client
        )
        evidence = await sandbox.execute(_case("hello", max_calls=6))

    assert authorizations == ["Bearer private-token"] * 6
    assert "private-token" not in evidence.model_dump_json()
    assert "private-token" not in config.model_dump_json()


async def test_rejects_non_ascii_header_value_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "Bearer s\N{LATIN SMALL LETTER E WITH ACUTE}cret"
    monkeypatch.setenv("UL_SANDBOX_TEST_AGENT_TOKEN", secret)

    with pytest.raises(
        RuntimeError, match="sandbox API header environment variable is invalid"
    ) as error:
        JsonHttpSandboxConnection.from_config(
            _config(headers_from_env={"Authorization": "UL_SANDBOX_TEST_AGENT_TOKEN"}),
            sandbox_confirmed=True,
        )

    assert secret not in str(error.value)


async def test_rejects_header_credentials_outside_sandbox_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "ambient-cloud-secret"
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret)

    with pytest.raises(ValidationError, match="UL_SANDBOX_ namespace") as error:
        _config(headers_from_env={"Authorization": "AWS_SECRET_ACCESS_KEY"})

    assert secret not in str(error.value)


async def test_rejects_cross_origin_lifecycle() -> None:
    raw = _config().model_dump(mode="json")
    raw["snapshot"]["url"] = "https://other.example.test/snapshot"
    with pytest.raises(ValidationError, match="same origin"):
        JsonHttpSandboxConfig.model_validate(raw)


async def test_rejects_endpoint_http_client_cannot_parse() -> None:
    raw = _config().model_dump(mode="json")
    for lifecycle_name in ("reset", "setup", "execute_turn", "snapshot"):
        raw[lifecycle_name]["url"] = "https://é_foo.example/reset"
    with pytest.raises(ValidationError, match="valid HTTP"):
        JsonHttpSandboxConfig.model_validate(raw)


@pytest.mark.parametrize(
    "encoded_config",
    (
        b'{"version":3,"version":3}',
        b'{"version":NaN}',
        b"[" * 200 + b"0" + b"]" * 200,
    ),
)
async def test_loader_rejects_adversarial_json(tmp_path: Path, encoded_config: bytes) -> None:
    config_path = tmp_path / "sandbox.json"
    config_path.write_bytes(encoded_config)
    with pytest.raises(ValueError, match="invalid JSON"):
        load_json_http_sandbox_config(config_path)


async def test_loader_rejects_symlink_and_fifo(tmp_path: Path) -> None:
    real_path = tmp_path / "real.json"
    real_path.write_text("{}", encoding="utf-8")
    symlink_path = tmp_path / "link.json"
    symlink_path.symlink_to(real_path)
    with pytest.raises(RuntimeError, match="could not be read"):
        load_json_http_sandbox_config(symlink_path)

    fifo_path = tmp_path / "config.fifo"
    os.mkfifo(fifo_path)
    with pytest.raises(RuntimeError, match="could not be read"):
        load_json_http_sandbox_config(fifo_path)


async def test_requires_confirmation_and_explicit_insecure_transport_opt_in() -> None:
    with pytest.raises(ValueError, match="isolation attestation"):
        JsonHttpSandboxConnection.from_config(_config(), sandbox_confirmed=False)
    with pytest.raises(ValueError, match="insecure transport opt-in"):
        JsonHttpSandboxConnection.from_config(
            _config("http://sandbox.example.test"), sandbox_confirmed=True
        )


async def test_public_validation_resolves_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UL_SANDBOX_TEST_AGENT_TOKEN", "private-token")
    assert validate_json_http_sandbox_configuration(
        _config(headers_from_env={"Authorization": "UL_SANDBOX_TEST_AGENT_TOKEN"}),
        sandbox_confirmed=True,
    ) == {"Authorization": "private-token"}


async def test_cancellation_during_execution_quarantines_connection() -> None:
    execute_started = asyncio.Event()
    never_complete = asyncio.Event()
    generation = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal generation
        body = json.loads(request.content)
        if request.url.path == "/reset":
            generation += 1
            return _raw_response(
                json.dumps(
                    {
                        "sandbox_id": "payments-test",
                        "case_id": body["case_id"],
                        "generation": generation,
                        "clean": True,
                    }
                ).encode()
            )
        if request.url.path == "/setup":
            return _raw_response(
                json.dumps({"sandbox_id": "payments-test", "case_id": body["case_id"]}).encode()
            )
        if request.url.path == "/execute":
            execute_started.set()
            await never_complete.wait()
        return _raw_response(
            json.dumps(
                {
                    "sandbox_id": "payments-test",
                    "case_id": body["case_id"],
                    "turn_id": body["turn_id"],
                    "state": {},
                }
            ).encode()
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            _config(), sandbox_confirmed=True, max_sandbox_api_calls=10, client=client
        )
        task = asyncio.create_task(sandbox.execute(_case("work", max_calls=6)))
        await execute_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        blocked = await sandbox.execute(_case("again", max_calls=6))

    assert blocked.lifecycle.failed_phase == "blocked_state_uncertain"


async def test_case_deadline_bounds_the_complete_lifecycle() -> None:
    execute_started = asyncio.Event()
    never_complete = asyncio.Event()
    generation = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal generation
        body = json.loads(request.content)
        if request.url.path == "/reset":
            generation += 1
            return _raw_response(
                json.dumps(
                    {
                        "sandbox_id": "payments-test",
                        "case_id": body["case_id"],
                        "generation": generation,
                        "clean": True,
                    }
                ).encode()
            )
        if request.url.path == "/setup":
            return _raw_response(
                json.dumps({"sandbox_id": "payments-test", "case_id": body["case_id"]}).encode()
            )
        if request.url.path == "/execute":
            execute_started.set()
            await never_complete.wait()
        return _raw_response(
            json.dumps(
                {
                    "sandbox_id": "payments-test",
                    "case_id": body["case_id"],
                    "turn_id": body["turn_id"],
                    "state": {},
                }
            ).encode()
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sandbox = JsonHttpSandboxConnection.from_config(
            _config(), sandbox_confirmed=True, max_sandbox_api_calls=10, client=client
        )
        short_case = _case("work", max_calls=6).model_copy(update={"timeout_seconds": 0.01})
        with pytest.raises(TimeoutError):
            await sandbox.execute(short_case)
        assert execute_started.is_set()
        blocked = await sandbox.execute(_case("again", max_calls=6))

    assert blocked.lifecycle.failed_phase == "blocked_state_uncertain"
