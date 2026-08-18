from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol, runtime_checkable

from ul_core.dataset import (
    InteractionRecord,
    ObservedAgentOutput,
    RenderedUserInput,
    SemanticEquivalenceAssessment,
    SemanticFrame,
    UserInputRecord,
)
from ul_core.models import (
    ExecutionResult,
    MaterializedScenario,
    OracleFinding,
    SafetyEnvelope,
    Scenario,
)


class DatasetTargetLifecycleError(RuntimeError):
    def __init__(
        self,
        *,
        failed_phase: str,
        completed_phases: tuple[str, ...],
        cleanup_reset_failed: bool,
        target_state_uncertain: bool,
    ) -> None:
        super().__init__("dataset target lifecycle failed")
        self.failed_phase = failed_phase
        self.completed_phases = completed_phases
        self.cleanup_reset_failed = cleanup_reset_failed
        self.target_state_uncertain = target_state_uncertain


@runtime_checkable
class SemanticDeconstructor(Protocol):
    def deconstruct(
        self,
        record: InteractionRecord | UserInputRecord,
        reference_frame: SemanticFrame | None = None,
    ) -> Awaitable[SemanticFrame]: ...


@runtime_checkable
class SemanticRenderer(Protocol):
    def render(
        self,
        raw_input: str,
        instruction: str,
        *,
        allow_temporary_value: bool = False,
    ) -> Awaitable[RenderedUserInput]: ...


@runtime_checkable
class SemanticEquivalenceVerifier(Protocol):
    def verify(
        self,
        source_input: str,
        candidate_input: str,
    ) -> Awaitable[SemanticEquivalenceAssessment]: ...


@runtime_checkable
class DatasetTargetExecutor(Protocol):
    @property
    def safety_envelope(self) -> SafetyEnvelope: ...

    @property
    def fresh_state_per_execution(self) -> bool: ...

    def execute(self, raw_input: str) -> Awaitable[ObservedAgentOutput]: ...


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
