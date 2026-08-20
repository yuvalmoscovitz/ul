from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol, runtime_checkable

from ul_core.dataset import (
    InteractionRecord,
    RenderedUserInput,
    SemanticEquivalenceAssessment,
    SemanticFrame,
    UserInputRecord,
)
from ul_core.evaluation import (
    EnvironmentCapabilities,
    EvaluationCase,
    ExecutionEvidence,
    ProductionSourcePage,
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
class ProductionSource(Protocol):
    def read(
        self,
        checkpoint: str | None,
        *,
        limit: int,
    ) -> Awaitable[ProductionSourcePage]: ...


@runtime_checkable
class EnvironmentExecutor(Protocol):
    @property
    def environment_id(self) -> str: ...

    @property
    def config_sha256(self) -> str: ...

    @property
    def capabilities(self) -> EnvironmentCapabilities: ...

    def api_calls_for_case(self, case: EvaluationCase) -> int: ...

    def execute(self, case: EvaluationCase) -> Awaitable[ExecutionEvidence]: ...
