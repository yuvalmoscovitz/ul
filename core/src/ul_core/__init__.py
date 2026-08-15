"""Core primitives for UL."""

from ul_core.augmentation import (
    AugmentationRegistry,
    builtin_augmentation_registry,
)
from ul_core.contracts import (
    DatasetTargetExecutor,
    OracleEvaluator,
    ScenarioMaterializer,
    TargetExecutor,
)
from ul_core.coverage import (
    CoverageArchive,
    DeterministicAugmentationSelector,
    extract_semantic_coverage,
)
from ul_core.dataset import ObservedAgentOutput
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

__all__ = [
    "AugmentationApplication",
    "AugmentationRegistry",
    "CampaignCaseResult",
    "CampaignResult",
    "CoverageArchive",
    "DatasetTargetExecutor",
    "DeterministicAugmentationSelector",
    "ExecutionResult",
    "ExecutionStatus",
    "FindingSeverity",
    "ObservedAgentOutput",
    "OracleEvaluator",
    "OracleFinding",
    "OracleRelation",
    "Scenario",
    "ScenarioMaterializer",
    "TargetExecutor",
    "builtin_augmentation_registry",
    "extract_semantic_coverage",
]
