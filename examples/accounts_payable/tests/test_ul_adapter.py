import asyncio
import json

import httpx
import pytest
from pydantic import SecretStr
from ul import CampaignRunner
from ul_core.contracts import OracleEvaluator, ScenarioMaterializer, TargetExecutor
from ul_core.models import ExecutionMode, ExecutionStatus
from ul_core.operators import (
    AmbiguityAugmentation,
    ExistingPartialOperationAugmentation,
    LaterCorrectionAugmentation,
    TimeoutAfterCommitAugmentation,
)

from examples.accounts_payable.agent import OpenRouterSettings
from examples.accounts_payable.environment import AccountsPayableEnvironment
from examples.accounts_payable.models import EnvironmentState, ExpectedOutcome
from examples.accounts_payable.scenarios import get_seed_scenario
from examples.accounts_payable.ul_adapter import (
    SUPPORTED_AUGMENTATION_IDS,
    AccountsPayableOpenRouterTarget,
    AccountsPayableOracleEvaluator,
    AccountsPayableScenarioMaterializer,
    AccountsPayableScriptedTarget,
    accounts_payable_augmentation_registry,
    to_ul_scenario,
    ul_seed_scenarios,
)


def test_every_accounts_payable_seed_maps_to_generic_scenario() -> None:
    scenarios = ul_seed_scenarios()

    assert len(scenarios) == 12
    assert all(scenario.metadata["domain_pack"] == "accounts-payable" for scenario in scenarios)
    assert all(scenario.actions[0].kind == "execute_payment" for scenario in scenarios)
    assert any(scenario.environment_events for scenario in scenarios)


def test_domain_registry_contains_only_materializable_augmentations() -> None:
    registered_ids = {
        augmentation.metadata.id for augmentation in accounts_payable_augmentation_registry().list()
    }

    assert registered_ids == SUPPORTED_AUGMENTATION_IDS


def test_adapter_implements_generic_contracts_end_to_end() -> None:
    scenario = ul_seed_scenarios()[0]
    materializer = AccountsPayableScenarioMaterializer()
    target = AccountsPayableScriptedTarget()
    oracle = AccountsPayableOracleEvaluator()

    assert isinstance(materializer, ScenarioMaterializer)
    assert isinstance(target, TargetExecutor)
    assert isinstance(oracle, OracleEvaluator)

    materialized = materializer.materialize(scenario)
    assert materialized.execution_mode == ExecutionMode.SANDBOX
    assert materialized.safety_envelope.isolated
    assert not materialized.safety_envelope.allows_network_egress
    assert not materialized.safety_envelope.allows_business_side_effects
    execution = asyncio.run(target.execute(materialized))
    findings = asyncio.run(oracle.evaluate(scenario, materialized, execution))

    assert execution.scenario_id == scenario.id
    assert findings[0].passed


def test_later_correction_changes_materialized_amount_and_execution() -> None:
    source = to_ul_scenario(get_seed_scenario("single-approved-invoice"))
    corrected = LaterCorrectionAugmentation().apply(source)[0].scenario
    materialized = AccountsPayableScenarioMaterializer().materialize(corrected)
    state = EnvironmentState.model_validate_json(json.dumps(materialized.environment))
    expectations = ExpectedOutcome.model_validate_json(json.dumps(materialized.expectations))

    assert isinstance(materialized.target_input, dict)
    materialized_messages = materialized.target_input["user_messages"]
    assert isinstance(materialized_messages, list)
    assert len(materialized_messages) == 2
    assert (
        state.invoices[0].amount
        == get_seed_scenario("single-approved-invoice").state.invoices[0].amount
    )
    assert expectations.requires_clarification
    assert expectations.payments == []

    execution = asyncio.run(AccountsPayableScriptedTarget().execute(materialized))
    final_state = EnvironmentState.model_validate_json(json.dumps(execution.state_after))
    assert final_state.payments == []


def test_timeout_after_commit_changes_faults_and_executes_once() -> None:
    source = to_ul_scenario(get_seed_scenario("single-approved-invoice"))
    augmented = TimeoutAfterCommitAugmentation().apply(source)[0].scenario
    materialized = AccountsPayableScenarioMaterializer().materialize(augmented)
    state = EnvironmentState.model_validate_json(json.dumps(materialized.environment))

    assert [fault.value for fault in state.tool_faults["execute_payment"]] == [
        "timeout_after_commit"
    ]

    execution = asyncio.run(AccountsPayableScriptedTarget().execute(materialized))
    final_state = EnvironmentState.model_validate_json(json.dumps(execution.state_after))
    assert [call.name for call in execution.tool_calls] == [
        "execute_payment",
        "list_payments",
    ]
    assert len(final_state.payments) == 1


def test_ambiguity_adds_an_invoice_and_prevents_scripted_payment() -> None:
    source = to_ul_scenario(get_seed_scenario("single-approved-invoice"))
    augmented = AmbiguityAugmentation().apply(source)[0].scenario
    materialized = AccountsPayableScenarioMaterializer().materialize(augmented)
    state = EnvironmentState.model_validate_json(json.dumps(materialized.environment))
    expectations = ExpectedOutcome.model_validate_json(json.dumps(materialized.expectations))

    assert len(state.invoices) == 2
    assert expectations.requires_clarification
    assert expectations.payments == []

    execution = asyncio.run(AccountsPayableScriptedTarget().execute(materialized))
    final_state = EnvironmentState.model_validate_json(json.dumps(execution.state_after))
    assert execution.tool_calls == ()
    assert final_state.payments == []


def test_materializer_rejects_unsupported_augmentation() -> None:
    source = to_ul_scenario(get_seed_scenario("single-approved-invoice"))
    unsupported = ExistingPartialOperationAugmentation().apply(source)[0].scenario

    try:
        AccountsPayableScenarioMaterializer().materialize(unsupported)
    except ValueError as error:
        assert "state.existing_partial_operation" in str(error)
    else:
        raise AssertionError("Expected unsupported augmentation to be rejected")


@pytest.mark.asyncio
async def test_openrouter_target_receives_augmented_materialized_conversation() -> None:
    captured_bodies: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "augmented-generation",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I need to validate the correction.",
                        }
                    }
                ],
            },
        )

    source = to_ul_scenario(get_seed_scenario("single-approved-invoice"))
    corrected = LaterCorrectionAugmentation().apply(source)[0].scenario
    materialized = AccountsPayableScenarioMaterializer(ExecutionMode.LIVE).materialize(corrected)
    settings = OpenRouterSettings(api_key=SecretStr("test-key"), live_calls_enabled=True)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = AccountsPayableOpenRouterTarget(settings, client)

    execution = await target.execute(materialized)
    await client.aclose()

    messages = captured_bodies[0]["messages"]
    assert isinstance(messages, list)
    assert '"content": "Correction:' in json.dumps(messages[-1])
    assert execution.final_output == "I need to validate the correction."
    assert execution.metadata["requested_model"] == settings.model
    prompts = execution.metadata["prompts"]
    assert isinstance(prompts, list)
    assert len(prompts) == 9
    assert all(
        isinstance(prompt, dict) and len(str(prompt.get("version", ""))) == 64 for prompt in prompts
    )
    assert execution.cost_usd == 0
    assert materialized.safety_envelope.allows_network_egress
    assert not materialized.safety_envelope.allows_business_side_effects


@pytest.mark.asyncio
async def test_campaign_deadline_cancels_openrouter_before_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    tool_activity: list[str] = []
    blocked_request_cancelled = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            blocked_request_cancelled.set()
            raise
        raise AssertionError("The blocked request should have been cancelled")

    original_dispatch_json = AccountsPayableEnvironment.dispatch_json

    def record_tool_activity(
        environment: AccountsPayableEnvironment,
        tool_name: str,
        arguments_json: str,
    ) -> str:
        tool_activity.append(tool_name)
        return original_dispatch_json(environment, tool_name, arguments_json)

    monkeypatch.setattr(AccountsPayableEnvironment, "dispatch_json", record_tool_activity)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    runner = CampaignRunner(
        AccountsPayableScenarioMaterializer(ExecutionMode.LIVE),
        AccountsPayableOpenRouterTarget(
            OpenRouterSettings(
                api_key=SecretStr("test-key"),
                live_calls_enabled=True,
            ),
            client,
        ),
        AccountsPayableOracleEvaluator(),
        allow_network_egress=True,
        case_timeout_seconds=0.05,
    )

    result = await runner.run(
        "cancellation-regression",
        to_ul_scenario(get_seed_scenario("single-approved-invoice")),
        max_cases=1,
    )
    await asyncio.wait_for(blocked_request_cancelled.wait(), timeout=0.1)
    request_count_at_deadline = len(requests)
    tool_count_at_deadline = len(tool_activity)
    await asyncio.sleep(0.05)
    await client.aclose()

    assert result.cases[0].execution.status == ExecutionStatus.TIMED_OUT
    assert request_count_at_deadline == len(requests) == 1
    assert tool_count_at_deadline == len(tool_activity) == 0


def test_every_scripted_campaign_case_passes_the_financial_oracle() -> None:
    async def run_campaigns() -> None:
        for scenario in ul_seed_scenarios():
            runner = CampaignRunner(
                AccountsPayableScenarioMaterializer(),
                AccountsPayableScriptedTarget(),
                AccountsPayableOracleEvaluator(),
                registry=accounts_payable_augmentation_registry(),
            )
            result = await runner.run(
                f"scripted-{scenario.id}",
                scenario,
                max_cases=100,
            )
            assert all(finding.passed for case in result.cases for finding in case.findings), (
                result.model_dump(mode="json")
            )

    asyncio.run(run_campaigns())
