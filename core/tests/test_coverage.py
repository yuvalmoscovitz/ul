from ul_core.augmentations.registry import builtin_augmentation_registry
from ul_core.coverage import (
    CoverageArchive,
    DeterministicAugmentationSelector,
    extract_semantic_coverage,
)
from ul_core.models import (
    Action,
    ActionEffect,
    ExecutionResult,
    ExecutionStatus,
    Scenario,
    ScenarioProvenance,
    ToolCall,
)


def test_coverage_is_semantic_and_selection_is_deterministic() -> None:
    scenario = Scenario(
        id="operation",
        title="Read and write",
        objective="Perform the requested operation.",
        actions=(
            Action(id="read", kind="inspect", effect=ActionEffect.READ),
            Action(id="write", kind="execute", effect=ActionEffect.WRITE),
        ),
        provenance=ScenarioProvenance(source="test"),
        metadata={"semantic_tags": ["high_risk", "approved"]},
    )
    execution = ExecutionResult(
        scenario_id=scenario.id,
        status=ExecutionStatus.SUCCEEDED,
        tool_calls=(ToolCall(name="inspect"), ToolCall(name="execute")),
    )
    features = extract_semantic_coverage(scenario, execution)
    archive = CoverageArchive()
    selector = DeterministicAugmentationSelector(builtin_augmentation_registry(), archive)

    first = selector.select(scenario, limit=3)
    second = selector.select(scenario, limit=3)

    assert [item.metadata.id for item in first] == [item.metadata.id for item in second]
    assert features.tool_sequence == ("inspect", "execute")
    assert features.semantic_tags == ("approved", "high_risk")
    assert archive.record(features)
    assert not archive.record(features)
    assert len(archive) == 1


def test_selector_prioritizes_less_used_augmentations() -> None:
    scenario = Scenario(
        id="write",
        title="Write",
        objective="Execute once.",
        actions=(Action(id="write", kind="execute", effect=ActionEffect.WRITE),),
        provenance=ScenarioProvenance(source="test"),
    )
    archive = CoverageArchive()
    selector = DeterministicAugmentationSelector(builtin_augmentation_registry(), archive)
    first = selector.select(scenario, limit=1)[0]
    archive.record(extract_semantic_coverage(scenario), (first.metadata.id,))

    assert selector.select(scenario, limit=1)[0].metadata.id != first.metadata.id
