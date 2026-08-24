import copy

import pytest
from pydantic import ValidationError
from ul_core.augmentations.projections import AugmentationProjection, ProjectionTarget


def _source_case() -> dict[str, object]:
    return {
        "inputs": {"request": {"message": "Cancel order 7"}, "tenant": "test"},
        "conversation": [{"id": "turn-1", "role": "user", "content": "Cancel it"}],
        "state": {"order-7": {"status": "pending", "owner": "customer-1"}},
        "tool_results": {"lookup-order": {"status": "pending", "version": 4}},
        "policies": {"cancellation": {"role": "member", "confirmation": True}},
        "environment_events": [{"id": "event-1", "kind": "clock", "payload": {"hour": 10}}],
    }


def _target(identifier: str, surface: str, path: str, event_id: str | None = None):
    return ProjectionTarget.model_validate(
        {"id": identifier, "surface": surface, "path": path, "event_id": event_id}
    )


def test_projection_reads_and_writes_all_six_surfaces() -> None:
    source = _source_case()
    locations = (
        ("structured_input", "/inputs/request/message", None),
        ("conversation", "/conversation/0/content", None),
        ("state", "/state/order-7/status", None),
        ("tool", "/tool_results/lookup-order/status", None),
        ("policy", "/policies/cancellation/role", None),
        ("environment", "/environment_events/0/payload/hour", "event-1"),
    )
    projection = AugmentationProjection(
        reads=tuple(
            _target(f"read-{index}", surface, path, event_id)
            for index, (surface, path, event_id) in enumerate(locations)
        ),
        writes=tuple(
            _target(f"write-{index}", surface, path, event_id)
            for index, (surface, path, event_id) in enumerate(locations)
        ),
    )

    assert projection.read(source) == {
        "read-0": "Cancel order 7",
        "read-1": "Cancel it",
        "read-2": "pending",
        "read-3": "pending",
        "read-4": "member",
        "read-5": 10,
    }

    candidate = copy.deepcopy(source)
    candidate["inputs"]["request"]["message"] = "Cancel order 8"  # type: ignore[index]
    candidate["conversation"][0]["content"] = "Cancel order 8"  # type: ignore[index]
    candidate["state"]["order-7"]["status"] = "cancelled"  # type: ignore[index]
    candidate["tool_results"]["lookup-order"]["status"] = "cancelled"  # type: ignore[index]
    candidate["policies"]["cancellation"]["role"] = "admin"  # type: ignore[index]
    candidate["environment_events"][0]["payload"]["hour"] = 11  # type: ignore[index]

    changes = projection.validate_candidate(source, candidate)

    assert set(changes.changed_paths) == {path for _, path, _ in locations}
    assert changes.changed_events == ("event-1",)
    assert candidate["inputs"]["tenant"] == "test"  # type: ignore[index]
    assert candidate["state"]["order-7"]["owner"] == "customer-1"  # type: ignore[index]


def test_projection_rejects_invalid_missing_conflicting_and_untargeted_changes() -> None:
    with pytest.raises(ValidationError, match="RFC 6901"):
        _target("invalid", "state", "/state/bad~path")

    source = _source_case()
    missing = AugmentationProjection(
        reads=(_target("read", "state", "/state/missing"),),
        writes=(_target("write", "state", "/state/order-7/status"),),
    )
    with pytest.raises(ValueError, match="does not resolve before execution"):
        missing.validate_source(source)

    with pytest.raises(ValidationError, match="write projection targets conflict"):
        AugmentationProjection(
            reads=(_target("read", "state", "/state"),),
            writes=(
                _target("write-state", "state", "/state"),
                _target("write-status", "state", "/state/order-7/status"),
            ),
        )

    projection = AugmentationProjection(
        reads=(_target("read", "state", "/state/order-7/status"),),
        writes=(_target("write", "state", "/state/order-7/status"),),
    )
    candidate = copy.deepcopy(source)
    candidate["inputs"]["tenant"] = "other"  # type: ignore[index]
    with pytest.raises(ValueError, match="untargeted paths"):
        projection.validate_candidate(source, candidate)
