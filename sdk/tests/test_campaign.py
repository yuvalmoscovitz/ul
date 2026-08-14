import asyncio

import pytest
from ul import (
    Action,
    ActionEffect,
    CampaignRunner,
    CoverageArchive,
    ExecutionMode,
    ExecutionResult,
    ExecutionStatus,
    FindingSeverity,
    MaterializedScenario,
    OracleFinding,
    SafetyEnvelope,
    Scenario,
    ScenarioProvenance,
)

pytestmark = pytest.mark.asyncio


class FakeMaterializer:
    def materialize(self, scenario: Scenario) -> MaterializedScenario:
        return MaterializedScenario(
            scenario_id=scenario.id,
            target_input={"objective": scenario.objective},
            environment={
                "events": [event.model_dump(mode="json") for event in scenario.environment_events]
            },
            safety_envelope=SafetyEnvelope(
                description="In-memory synthetic ledger",
                isolated=True,
                allows_network_egress=False,
                allows_business_side_effects=False,
            ),
        )


class FakeExecutor:
    async def execute(self, scenario: MaterializedScenario) -> ExecutionResult:
        return ExecutionResult(
            scenario_id=scenario.scenario_id,
            status=ExecutionStatus.SUCCEEDED,
            state_before={"commits": 0},
            state_after={"commits": 1},
        )


class FakeOracle:
    async def evaluate(
        self,
        scenario: Scenario,
        materialized_scenario: MaterializedScenario,
        execution: ExecutionResult,
    ) -> tuple[OracleFinding, ...]:
        has_timeout_after_commit = any(
            event.payload.get("commit_state") == "committed"
            for event in scenario.environment_events
        )
        return (
            OracleFinding(
                oracle_id="exactly-once",
                passed=not has_timeout_after_commit,
                category="duplicate_effect",
                message="The committed effect must not be repeated.",
                severity=FindingSeverity.HIGH,
                evidence={"scenario_id": materialized_scenario.scenario_id},
            ),
        )


def seed_scenario() -> Scenario:
    return Scenario(
        id="seed",
        title="Consequential operation",
        objective="Execute the operation exactly once.",
        actions=(Action(id="write", kind="execute", effect=ActionEffect.WRITE),),
        provenance=ScenarioProvenance(source="production_trace"),
    )


async def test_campaign_runs_generic_augmentation_execution_and_oracle_loop() -> None:
    runner = CampaignRunner(FakeMaterializer(), FakeExecutor(), FakeOracle())

    result = await runner.run("campaign-1", seed_scenario())

    assert len(result.cases) == 4
    assert result.failed_case_count == 1
    assert result.cases[0].augmentation_ids == ()
    assert result.cases[0].scenario == seed_scenario()
    assert {case.augmentation_ids[0] for case in result.cases[1:]} == {
        "state.existing_partial_operation",
        "tool.timeout_after_commit",
        "tool.timeout_before_commit",
    }
    assert len(runner.archive) == 4
    assert all(case.scenario_id == case.scenario.id for case in result.cases)
    assert result.cases[1].augmentation_applications
    assert result.cases[1].oracle_relations


async def test_campaign_has_a_hard_case_limit() -> None:
    archive = CoverageArchive()
    runner = CampaignRunner(FakeMaterializer(), FakeExecutor(), FakeOracle(), archive=archive)

    result = await runner.run("campaign-1", seed_scenario(), max_cases=1)

    assert len(result.cases) == 1
    assert result.cases[0].augmentation_ids == ()
    assert runner.archive is archive


async def test_campaign_rejects_negative_augmentation_limit_before_execution() -> None:
    execution_count = 0

    class CountingExecutor:
        async def execute(self, scenario: MaterializedScenario) -> ExecutionResult:
            nonlocal execution_count
            execution_count += 1
            return ExecutionResult(
                scenario_id=scenario.scenario_id,
                status=ExecutionStatus.SUCCEEDED,
            )

    runner = CampaignRunner(FakeMaterializer(), CountingExecutor(), FakeOracle())

    with pytest.raises(ValueError, match="augmentation_limit must not be negative"):
        await runner.run("campaign-1", seed_scenario(), augmentation_limit=-1)

    assert execution_count == 0


async def test_campaign_records_mismatched_executor_results() -> None:
    class WrongExecutor:
        async def execute(self, scenario: MaterializedScenario) -> ExecutionResult:
            return ExecutionResult(scenario_id="wrong", status=ExecutionStatus.SUCCEEDED)

    runner = CampaignRunner(FakeMaterializer(), WrongExecutor(), FakeOracle())

    result = await runner.run("campaign-1", seed_scenario(), max_cases=1)

    assert result.cases[0].execution.status == ExecutionStatus.ERROR
    assert result.cases[0].findings[0].category == "case_execution_error"


async def test_campaign_records_timeout_without_aborting() -> None:
    class SlowExecutor:
        async def execute(self, scenario: MaterializedScenario) -> ExecutionResult:
            await asyncio.sleep(1)
            return ExecutionResult(
                scenario_id=scenario.scenario_id, status=ExecutionStatus.SUCCEEDED
            )

    runner = CampaignRunner(
        FakeMaterializer(),
        SlowExecutor(),
        FakeOracle(),
        case_timeout_seconds=0.001,
    )

    result = await runner.run("campaign-1", seed_scenario(), max_cases=1)

    assert result.cases[0].execution.status == ExecutionStatus.TIMED_OUT
    assert result.cases[0].findings[0].category == "case_timeout"


async def test_target_domain_timeout_is_not_misclassified_as_deadline() -> None:
    class DomainTimeoutExecutor:
        async def execute(self, scenario: MaterializedScenario) -> ExecutionResult:
            raise TimeoutError("synthetic downstream operation timed out")

    runner = CampaignRunner(FakeMaterializer(), DomainTimeoutExecutor(), FakeOracle())

    result = await runner.run("campaign-1", seed_scenario(), max_cases=1)

    assert result.cases[0].execution.status == ExecutionStatus.ERROR
    assert result.cases[0].execution.error == "TimeoutError"
    assert result.cases[0].findings[0].category == "case_execution_error"


async def test_non_successful_target_status_fails_the_case() -> None:
    class FailedExecutor:
        async def execute(self, scenario: MaterializedScenario) -> ExecutionResult:
            return ExecutionResult(
                scenario_id=scenario.scenario_id,
                status=ExecutionStatus.FAILED,
            )

    runner = CampaignRunner(FakeMaterializer(), FailedExecutor(), FakeOracle())

    result = await runner.run("campaign-1", seed_scenario(), max_cases=1)

    assert result.failed_case_count == 1
    assert result.cases[0].findings[-1].category == "target_execution_failed"


async def test_campaign_blocks_business_side_effects_without_opt_in() -> None:
    class BusinessSideEffectMaterializer:
        def materialize(self, scenario: Scenario) -> MaterializedScenario:
            return MaterializedScenario(
                scenario_id=scenario.id,
                target_input={},
                environment={},
                execution_mode=ExecutionMode.LIVE,
                safety_envelope=SafetyEnvelope(
                    description="Customer production target",
                    isolated=False,
                    allows_network_egress=False,
                    allows_business_side_effects=True,
                ),
            )

    blocked_runner = CampaignRunner(BusinessSideEffectMaterializer(), FakeExecutor(), FakeOracle())
    allowed_runner = CampaignRunner(
        BusinessSideEffectMaterializer(),
        FakeExecutor(),
        FakeOracle(),
        allow_business_side_effects=True,
    )

    blocked = await blocked_runner.run("blocked", seed_scenario(), max_cases=1)
    allowed = await allowed_runner.run("allowed", seed_scenario(), max_cases=1)

    assert blocked.cases[0].execution.status == ExecutionStatus.ERROR
    assert blocked.cases[0].findings[0].category == "business_side_effects_blocked"
    assert allowed.cases[0].execution.status == ExecutionStatus.SUCCEEDED


async def test_network_egress_has_an_independent_opt_in() -> None:
    class NetworkedSandboxMaterializer:
        def materialize(self, scenario: Scenario) -> MaterializedScenario:
            return MaterializedScenario(
                scenario_id=scenario.id,
                target_input={},
                environment={},
                execution_mode=ExecutionMode.LIVE,
                safety_envelope=SafetyEnvelope(
                    description="Isolated ledger with a billed model API call",
                    isolated=True,
                    allows_network_egress=True,
                    allows_business_side_effects=False,
                ),
            )

    blocked_runner = CampaignRunner(NetworkedSandboxMaterializer(), FakeExecutor(), FakeOracle())
    allowed_runner = CampaignRunner(
        NetworkedSandboxMaterializer(),
        FakeExecutor(),
        FakeOracle(),
        allow_network_egress=True,
    )

    blocked = await blocked_runner.run("blocked", seed_scenario(), max_cases=1)
    allowed = await allowed_runner.run("allowed", seed_scenario(), max_cases=1)

    assert blocked.cases[0].findings[0].category == "network_egress_blocked"
    assert allowed.cases[0].execution.status == ExecutionStatus.SUCCEEDED
