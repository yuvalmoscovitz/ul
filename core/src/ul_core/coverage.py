from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable

from ul_core.augmentations.registry import Augmentation, AugmentationRegistry
from ul_core.models import (
    EnvironmentEvent,
    ExecutionResult,
    OracleFinding,
    Scenario,
    SemanticCoverageFeatures,
)


def extract_semantic_coverage(
    scenario: Scenario,
    execution: ExecutionResult | None = None,
    findings: Iterable[OracleFinding] = (),
) -> SemanticCoverageFeatures:
    metadata_tags = scenario.metadata.get("semantic_tags", [])
    semantic_tags = (
        tuple(sorted(str(tag) for tag in metadata_tags)) if isinstance(metadata_tags, list) else ()
    )
    return SemanticCoverageFeatures(
        action_kinds=tuple(sorted({action.kind for action in scenario.actions})),
        action_statuses=tuple(sorted({action.status.value for action in scenario.actions})),
        policy_states=tuple(sorted({policy.state for policy in scenario.policies})),
        environment_event_kinds=tuple(
            sorted({event.kind for event in scenario.environment_events})
        ),
        environment_event_semantics=tuple(
            sorted(_event_semantic_feature(event) for event in scenario.environment_events)
        ),
        tool_sequence=()
        if execution is None
        else tuple(tool_call.name for tool_call in execution.tool_calls),
        execution_outcome=None if execution is None else execution.status.value,
        oracle_categories=tuple(sorted({finding.category for finding in findings})),
        semantic_tags=semantic_tags,
    )


def coverage_signature(features: SemanticCoverageFeatures) -> str:
    serialized = json.dumps(features.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


class CoverageArchive:
    def __init__(self) -> None:
        self._signatures: Counter[str] = Counter()
        self._augmentation_uses: Counter[str] = Counter()

    def record(
        self,
        features: SemanticCoverageFeatures,
        augmentation_ids: Iterable[str] = (),
    ) -> bool:
        signature = coverage_signature(features)
        is_new = self._signatures[signature] == 0
        self._signatures[signature] += 1
        self._augmentation_uses.update(augmentation_ids)
        return is_new

    def coverage_count(self, features: SemanticCoverageFeatures) -> int:
        return self._signatures[coverage_signature(features)]

    def augmentation_count(self, augmentation_id: str) -> int:
        return self._augmentation_uses[augmentation_id]

    def __len__(self) -> int:
        return len(self._signatures)


class DeterministicAugmentationSelector:
    def __init__(self, registry: AugmentationRegistry, archive: CoverageArchive) -> None:
        self._registry = registry
        self._archive = archive

    def select(self, scenario: Scenario, *, limit: int | None = None) -> tuple[Augmentation, ...]:
        applicable = self._registry.applicable(scenario)
        ordered = sorted(
            applicable,
            key=lambda augmentation: (
                self._archive.augmentation_count(augmentation.metadata.id),
                _stable_tiebreaker(scenario.id, augmentation.metadata.id),
                augmentation.metadata.id,
            ),
        )
        if limit is None:
            return tuple(ordered)
        if limit < 0:
            raise ValueError("limit must be non-negative")
        return tuple(ordered[:limit])


def _stable_tiebreaker(scenario_id: str, augmentation_id: str) -> str:
    return hashlib.sha256(f"{scenario_id}\0{augmentation_id}".encode()).hexdigest()


def _event_semantic_feature(event: EnvironmentEvent) -> str:
    commit_state = event.payload.get("commit_state")
    suffix = "" if commit_state is None else f":{commit_state}"
    return f"{event.kind}:{event.timing.value}{suffix}"
