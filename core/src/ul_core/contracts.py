from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol, runtime_checkable

from ul_core.models import ExecutionResult, MaterializedScenario, OracleFinding, Scenario


@runtime_checkable
class ScenarioMaterializer(Protocol):
    def materialize(self, scenario: Scenario) -> MaterializedScenario: ...


@runtime_checkable
class TargetExecutor(Protocol):
    def execute(self, scenario: MaterializedScenario) -> Awaitable[ExecutionResult]: ...


@runtime_checkable
class OracleEvaluator(Protocol):
    def evaluate(
        self,
        scenario: Scenario,
        materialized_scenario: MaterializedScenario,
        execution: ExecutionResult,
    ) -> Awaitable[tuple[OracleFinding, ...]]: ...
