"""Core primitives for UL."""

from ul_core.augmentation import (
    AugmentationRegistry,
    builtin_augmentation_registry,
)
from ul_core.contracts import ProductionSource, SemanticEquivalenceVerifier
from ul_core.coverage import (
    CoverageArchive,
    DeterministicAugmentationSelector,
    extract_semantic_coverage,
)
from ul_core.dataset import (
    ObservedAgentOutput,
    SemanticDelta,
    SemanticEquivalenceAssessment,
)
from ul_core.evaluation import (
    EvaluationCase,
    ExecutionEvidence,
    ProductionObservation,
    ProductionSourcePage,
    SandboxCapabilities,
    SandboxLifecycleEvidence,
    SandboxStateEvidence,
    SandboxTurnEvidence,
    StateObservationAuthority,
    TimeoutAfterCommitEventEvidence,
    TimeoutAfterCommitEventRequest,
    TimeoutAfterCommitTriggerStatus,
)
from ul_core.models import (
    AugmentationApplication,
    CampaignCaseResult,
    CampaignResult,
    ExecutionResult,
    ExecutionStatus,
    FindingSeverity,
    OracleFinding,
    OracleRelation,
    Scenario,
)
from ul_core.prompts import PromptManager, PromptTemplateInfo

__all__ = [
    "AugmentationApplication",
    "AugmentationRegistry",
    "CampaignCaseResult",
    "CampaignResult",
    "CoverageArchive",
    "DeterministicAugmentationSelector",
    "EvaluationCase",
    "ExecutionEvidence",
    "ExecutionResult",
    "ExecutionStatus",
    "FindingSeverity",
    "ObservedAgentOutput",
    "OracleFinding",
    "OracleRelation",
    "ProductionObservation",
    "ProductionSource",
    "ProductionSourcePage",
    "PromptManager",
    "PromptTemplateInfo",
    "SandboxCapabilities",
    "SandboxLifecycleEvidence",
    "SandboxStateEvidence",
    "SandboxTurnEvidence",
    "Scenario",
    "SemanticDelta",
    "SemanticEquivalenceAssessment",
    "SemanticEquivalenceVerifier",
    "StateObservationAuthority",
    "TimeoutAfterCommitEventEvidence",
    "TimeoutAfterCommitEventRequest",
    "TimeoutAfterCommitTriggerStatus",
    "builtin_augmentation_registry",
    "extract_semantic_coverage",
]
