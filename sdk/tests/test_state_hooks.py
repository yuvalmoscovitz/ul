from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import cast

import httpx
import pytest
from pydantic import JsonValue
from ul.environment import evaluation_case_from_inputs, validate_execution_evidence
from ul.http_environment import (
    JsonHttpEnvironmentConnection,
    JsonHttpIsolatedResponseConfig,
)
from ul.probe_execution import ComposedEnvironmentExecutor
from ul.state_hooks import (
    CallbackStateEnvironment,
    JsonStateNormalization,
    StateCallbackContext,
    check_deterministic_reset,
    diff_json_states,
    json_state_digest,
    normalize_json_state,
)
from ul_core.evaluation import (
    EvaluationCase,
    ProbeInvokerCapabilities,
    ProbeRequest,
    ProbeResult,
    StateFixtureRequest,
)

_CONFIG_SHA256 = "a" * 64


class _StaticResponseStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._body


@dataclass
class _MutableStateInvoker:
    state: dict[str, JsonValue]
    capabilities: ProbeInvokerCapabilities = field(
        default_factory=lambda: ProbeInvokerCapabilities(
            invoker_id="mutable-agent",
            response_size_limit_bytes=10_000,
            supports_conversations=True,
        )
    )

    def invoke(self, request: ProbeRequest) -> ProbeResult:
        orders = cast(dict[str, JsonValue], self.state["orders"])
        audit = cast(list[JsonValue], self.state["audit"])
        if request.turn.input == "wrong-record":
            orders["order-2"] = {"status": "cancelled"}
        elif request.turn.input == "duplicate-write":
            audit.extend([{"order_id": "order-1"}, {"order_id": "order-1"}])
        elif request.turn.input == "collateral-change":
            cast(dict[str, JsonValue], orders["order-1"])["status"] = "cancelled"
            cast(dict[str, JsonValue], self.state["account"])["tier"] = "basic"
        return ProbeResult(
            id=f"{request.correlation_id}:result",
            correlation_id=request.correlation_id,
            response={"handled": request.turn.input},
        )


def _case(action: str) -> EvaluationCase:
    return evaluation_case_from_inputs(
        case_id=f"case-{action}",
        raw_inputs=(action,),
        max_environment_api_calls=10,
        timeout_seconds=2,
        required_state_observation_authority="independent_observer",
        required_state_observer_id="orders-observer",
    ).model_copy(
        update={
            "probe_context": {
                "fixture": {"id": "orders", "version": "1"},
                "inputs": {"action": action},
            }
        }
    )


def _clean_state() -> dict[str, JsonValue]:
    return {
        "orders": {"order-1": {"status": "pending"}},
        "account": {"tier": "gold"},
        "audit": [],
    }


def _replace_state(state: dict[str, JsonValue]) -> None:
    state.clear()
    state.update(_clean_state())


def _snapshot_state(state: dict[str, JsonValue]) -> JsonValue:
    return json.loads(json.dumps(state))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "expected_paths"),
    (
        ("wrong-record", {"/orders/order-2"}),
        ("duplicate-write", {"/audit/0", "/audit/1"}),
        ("missing-write", set[str]()),
        ("collateral-change", {"/account/tier", "/orders/order-1/status"}),
    ),
)
async def test_callback_state_evidence_exposes_domain_failures(
    action: str,
    expected_paths: set[str],
) -> None:
    state = _clean_state()
    contexts: list[StateCallbackContext] = []

    def reset(context: StateCallbackContext) -> None:
        contexts.append(context)
        _replace_state(state)

    async def snapshot(context: StateCallbackContext) -> JsonValue:
        contexts.append(context)
        await asyncio.sleep(0)
        return _snapshot_state(state)

    state_environment = CallbackStateEnvironment(
        environment_id="orders-observer",
        reset=reset,
        snapshot=snapshot,
        authority="independent_observer",
    )
    executor = ComposedEnvironmentExecutor(
        _MutableStateInvoker(state),
        config_sha256=_CONFIG_SHA256,
        state_environment=state_environment,
        fixture_id="orders-v1",
    )

    evidence = await executor.execute(_case(action))

    assert evidence.lifecycle.terminal_status == "succeeded"
    assert evidence.evidence_scope == "response_and_state"
    assert evidence.initial_state is not None
    assert evidence.final_state is not None
    assert evidence.final_state.authority == "independent_observer"
    assert evidence.final_state.observer_id == "orders-observer"
    differences = diff_json_states(evidence.initial_state.value, evidence.final_state.value)
    assert {difference.path for difference in differences} == expected_paths
    assert [context.phase for context in contexts] == [
        "reset",
        "snapshot",
        "snapshot",
        "cleanup",
    ]
    assert [context.generation for context in contexts] == [1, 1, 1, 2]
    assert all(context.fixture_id == "orders-v1" for context in contexts)
    assert all(context.case_id == f"case-{action}" for context in contexts)
    assert all(context.session_id.startswith("ul-session-") for context in contexts)
    assert contexts[0].turn_id is None
    assert contexts[1].turn_id == "__ul_initial_state__"
    assert contexts[2].turn_id == f"case-{action}:turn-1"
    assert contexts[0].correlation_id == contexts[-1].correlation_id
    assert all(context.case_context == _case(action).probe_context for context in contexts)


def test_normalization_digest_and_diff_are_deterministic() -> None:
    normalization = JsonStateNormalization(
        volatile_json_pointers=("/generated_id", "/updated_at"),
        unordered_json_pointers=("/records",),
    )
    first: JsonValue = {
        "generated_id": "id-1",
        "updated_at": "2026-01-01T00:00:00Z",
        "records": [{"id": 2}, {"id": 1}],
    }
    second: JsonValue = {
        "records": [{"id": 1}, {"id": 2}],
        "updated_at": "2026-08-23T00:00:00Z",
        "generated_id": "id-2",
    }

    assert normalize_json_state(first, normalization) == normalize_json_state(second, normalization)
    assert json_state_digest(first, normalization) == json_state_digest(second, normalization)
    assert diff_json_states(first, second, normalization) == ()


def test_callback_state_identity_includes_normalization_and_defaults_conservatively() -> None:
    default_environment = CallbackStateEnvironment(
        environment_id="state-source",
        reset=lambda context: None,
        snapshot=lambda context: {},
    )
    normalized_environment = CallbackStateEnvironment(
        environment_id="state-source",
        reset=lambda context: None,
        snapshot=lambda context: {},
        normalization=JsonStateNormalization(volatile_json_pointers=("/updated_at",)),
    )

    assert default_environment.capabilities.state_observation_authority == (
        "environment_self_reported"
    )
    assert default_environment.config_sha256 != normalized_environment.config_sha256


@pytest.mark.asyncio
async def test_double_reset_conformance_uses_normalized_clean_state_digest() -> None:
    state: dict[str, JsonValue] = {}
    reset_count = 0

    def reset(context: StateCallbackContext) -> None:
        nonlocal reset_count
        reset_count += 1
        state.clear()
        state.update(
            {
                "generated_id": f"generated-{reset_count}",
                "updated_at": f"time-{reset_count}",
                "records": [{"id": 2}, {"id": 1}],
            }
        )

    environment = CallbackStateEnvironment(
        environment_id="conformance-observer",
        reset=reset,
        snapshot=lambda context: _snapshot_state(state),
        authority="independent_observer",
        normalization=JsonStateNormalization(
            volatile_json_pointers=("/generated_id", "/updated_at"),
            unordered_json_pointers=("/records",),
        ),
    )
    request = StateFixtureRequest(
        fixture_id="fixture-1",
        case_id="conformance",
        session_id="session-1",
        correlation_id="correlation-1",
    )

    report = await check_deterministic_reset(environment, request)

    assert report.deterministic is True
    assert report.first_digest == report.second_digest
    assert report.differences == ()
    assert reset_count == 2
    assert environment.capabilities.supports_deterministic_replay is True


@pytest.mark.asyncio
async def test_cleanup_failure_quarantines_callback_environment() -> None:
    state = _clean_state()

    def reset(context: StateCallbackContext) -> None:
        _replace_state(state)

    def cleanup(context: StateCallbackContext) -> None:
        raise RuntimeError("private cleanup detail")

    environment = CallbackStateEnvironment(
        environment_id="orders-observer",
        reset=reset,
        snapshot=lambda context: _snapshot_state(state),
        cleanup=cleanup,
        authority="independent_observer",
    )
    executor = ComposedEnvironmentExecutor(
        _MutableStateInvoker(state),
        config_sha256=_CONFIG_SHA256,
        state_environment=environment,
    )

    failed = await executor.execute(_case("missing-write"))
    blocked = await executor.execute(_case("missing-write"))

    assert failed.lifecycle.cleanup == "failed"
    assert failed.lifecycle.environment_state_uncertain is True
    assert "private cleanup detail" not in failed.model_dump_json()
    assert blocked.lifecycle.failed_phase == "blocked_state_uncertain"
    assert blocked.lifecycle.environment_state_uncertain is True


@pytest.mark.asyncio
async def test_optional_callback_setup_runs_before_initial_snapshot() -> None:
    state = _clean_state()
    phases: list[str] = []

    def reset(context: StateCallbackContext) -> None:
        phases.append(context.phase)
        _replace_state(state)

    async def setup(context: StateCallbackContext) -> None:
        phases.append(context.phase)
        state["fixture_ready"] = True

    def snapshot(context: StateCallbackContext) -> JsonValue:
        phases.append(context.phase)
        return _snapshot_state(state)

    environment = CallbackStateEnvironment(
        environment_id="orders-observer",
        reset=reset,
        setup=setup,
        snapshot=snapshot,
        authority="independent_observer",
    )
    executor = ComposedEnvironmentExecutor(
        _MutableStateInvoker(state),
        config_sha256=_CONFIG_SHA256,
        state_environment=environment,
    )

    evidence = await executor.execute(_case("missing-write"))

    assert evidence.lifecycle.terminal_status == "succeeded"
    assert evidence.initial_state is not None
    assert cast(dict[str, JsonValue], evidence.initial_state.value)["fixture_ready"] is True
    assert phases == ["reset", "setup", "snapshot", "snapshot", "cleanup"]


@pytest.mark.asyncio
async def test_isolated_http_agent_composes_with_local_state_observer() -> None:
    state = _clean_state()
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        orders = cast(dict[str, JsonValue], state["orders"])
        cast(dict[str, JsonValue], orders["order-1"])["status"] = "cancelled"
        return httpx.Response(
            200,
            stream=_StaticResponseStream(b'{"response":{"status":"cancelled"}}'),
            headers={"content-type": "application/json"},
        )

    def reset(context: StateCallbackContext) -> None:
        _replace_state(state)

    observer = CallbackStateEnvironment(
        environment_id="orders-observer",
        reset=reset,
        snapshot=lambda context: _snapshot_state(state),
        authority="independent_observer",
    )
    config = JsonHttpIsolatedResponseConfig.model_validate(
        {
            "version": 1,
            "adapter_tier": "isolated_response",
            "environment_id": "orders-agent",
            "request_isolation_attested": True,
            "safe_test_target_attested": True,
            "execute": {
                "url": "https://agent.example.test/execute",
                "request_json_template": {"input": "{{input}}"},
                "response_json_pointer": "/response",
            },
        }
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        environment = JsonHttpEnvironmentConnection.from_config(
            config,
            test_environment_confirmed=True,
            state_environment=observer,
            state_fixture_id="orders-v1",
            client=client,
        )
        evidence = await environment.execute(_case("cancel-order"))

    assert requests == [{"input": "cancel-order"}]
    assert evidence.lifecycle.terminal_status == "succeeded", evidence.lifecycle.model_dump()
    assert evidence.final_response == {"status": "cancelled"}
    assert evidence.initial_state is not None
    assert evidence.final_state is not None
    assert [
        difference.path
        for difference in diff_json_states(evidence.initial_state.value, evidence.final_state.value)
    ] == ["/orders/order-1/status"]
    assert environment.capabilities.supports_state_observation is True
    assert environment.capabilities.state_observation_authority == "independent_observer"
    assert environment.evidence_profile.available_facts == frozenset(
        {"response_observed", "committed_state_verified"}
    )
    validate_execution_evidence(_case("cancel-order"), environment, evidence)


@pytest.mark.asyncio
async def test_http_target_without_state_hooks_remains_explicitly_response_only() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=_StaticResponseStream(b'{"response":"ok"}'),
            headers={"content-type": "application/json"},
        )

    config = JsonHttpIsolatedResponseConfig.model_validate(
        {
            "version": 1,
            "adapter_tier": "isolated_response",
            "environment_id": "response-only-agent",
            "request_isolation_attested": True,
            "safe_test_target_attested": True,
            "execute": {
                "url": "https://agent.example.test/execute",
                "request_json_template": {"input": "{{input}}"},
                "response_json_pointer": "/response",
            },
        }
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        environment = JsonHttpEnvironmentConnection.from_config(
            config,
            test_environment_confirmed=True,
            client=client,
        )
        evidence = await environment.execute(
            evaluation_case_from_inputs(
                case_id="response-only",
                raw_inputs=("hello",),
                max_environment_api_calls=1,
                timeout_seconds=2,
            )
        )

    assert evidence.evidence_scope == "response_only"
    assert evidence.initial_state is None
    assert evidence.final_state is None
    assert environment.evidence_profile.available_facts == frozenset({"response_observed"})
