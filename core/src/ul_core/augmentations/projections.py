"""Authoritative projection contracts and materialized augmentation targets."""

from __future__ import annotations

import copy
from typing import Literal, Self, cast

from pydantic import ConfigDict, Field, JsonValue, model_validator

from ul_core.dataset import resolve_json_pointer
from ul_core.models import ULModel

AugmentationTargetSurface = Literal[
    "structured_input",
    "conversation",
    "state",
    "tool",
    "policy",
    "environment",
]
ProjectionTargetOperation = Literal["existing", "create"]

_SURFACE_ROOTS: dict[AugmentationTargetSurface, tuple[str, ...]] = {
    "structured_input": ("/inputs", "/raw_input", "/actions", "/metadata"),
    "conversation": ("/context", "/conversation"),
    "state": ("/state", "/artifacts", "/resources", "/actions"),
    "tool": ("/tool_results", "/actions"),
    "policy": ("/policies", "/actions"),
    "environment": ("/environment_events",),
}
_MAXIMUM_DIFF_NODES = 100_000


class _ProjectionModel(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProjectionContract(_ProjectionModel):
    reads: tuple[AugmentationTargetSurface, ...] = Field(min_length=1)
    writes: tuple[AugmentationTargetSurface, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_surfaces(self) -> Self:
        if len(self.reads) != len(set(self.reads)) or len(self.writes) != len(set(self.writes)):
            raise ValueError("projection contract surfaces must be unique")
        return self

    def validate_projection(self, projection: AugmentationProjection) -> None:
        invalid_reads = tuple(
            target.id for target in projection.reads if target.surface not in self.reads
        )
        invalid_writes = tuple(
            target.id for target in projection.writes if target.surface not in self.writes
        )
        if invalid_reads or invalid_writes:
            raise ValueError("materialized projection exceeds its authoritative binding contract")


class ProjectionTarget(_ProjectionModel):
    id: str = Field(min_length=1, max_length=500)
    surface: AugmentationTargetSurface
    path: str = Field(min_length=1, max_length=1_000)
    event_id: str | None = Field(default=None, min_length=1, max_length=500)
    operation: ProjectionTargetOperation = "existing"

    @model_validator(mode="after")
    def validate_selector(self) -> Self:
        try:
            resolve_json_pointer({}, self.path)
        except ValueError as error:
            if str(error) == "json pointer must follow RFC 6901 syntax":
                raise ValueError("projection target path must follow RFC 6901 syntax") from None
        if not any(_path_contains(root, self.path) for root in _SURFACE_ROOTS[self.surface]):
            raise ValueError("projection target path does not belong to its surface")
        if (self.surface == "environment") != (self.event_id is not None):
            raise ValueError("environment targets require exactly one event identifier")
        if self.surface == "environment" and _environment_event_index(self.path) is None:
            raise ValueError("environment target path must select one event")
        if self.operation == "create" and self.surface != "environment":
            raise ValueError("only environment event targets may create values")
        return self


class AugmentationChangeSet(_ProjectionModel):
    changed_paths: tuple[str, ...]
    changed_events: tuple[str, ...] = ()


class AugmentationProjection(_ProjectionModel):
    reads: tuple[ProjectionTarget, ...] = Field(min_length=1, max_length=100)
    writes: tuple[ProjectionTarget, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_targets(self) -> Self:
        identifiers = tuple(target.id for target in (*self.reads, *self.writes))
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("projection target identifiers must be unique")
        if any(target.operation == "create" for target in self.reads):
            raise ValueError("read projection targets cannot create values")
        for index, target in enumerate(self.writes):
            for other in self.writes[index + 1 :]:
                if _paths_overlap(target.path, other.path):
                    raise ValueError("write projection targets conflict")
        return self

    def read(self, source: JsonValue) -> dict[str, JsonValue]:
        self.validate_source(source)
        return {
            target.id: copy.deepcopy(resolve_json_pointer(source, target.path))
            for target in self.reads
        }

    def validate_source(self, source: JsonValue) -> None:
        source_events = _validated_events_by_id(source)
        for target in (*self.reads, *self.writes):
            if target.operation == "create":
                parent_path = target.path.rsplit("/", 1)[0]
                parent = resolve_json_pointer(source, parent_path)
                if not isinstance(parent, list) or _resolves(source, target.path):
                    raise ValueError(f"projection target {target.id!r} is not a new event location")
                if target.event_id in source_events:
                    raise ValueError(
                        f"projection target {target.id!r} reuses an environment event identifier"
                    )
                continue
            try:
                resolve_json_pointer(source, target.path)
            except ValueError:
                raise ValueError(
                    f"projection target {target.id!r} does not resolve before execution"
                ) from None
            if target.event_id is not None:
                _validate_event_target(source, target)

    def validate_candidate(self, source: JsonValue, candidate: JsonValue) -> AugmentationChangeSet:
        self.validate_source(source)
        _validated_events_by_id(candidate)
        for target in self.writes:
            if target.event_id is not None:
                _validate_event_target(candidate, target)
        changed_paths = tuple(_changed_paths(source, candidate))
        if not changed_paths:
            raise ValueError("augmentation candidate does not change its source")
        uncontrolled = tuple(
            path
            for path in changed_paths
            if not any(_path_contains(target.path, path) for target in self.writes)
        )
        if uncontrolled:
            raise ValueError(
                "augmentation candidate changed untargeted paths: " + ", ".join(uncontrolled)
            )
        unused_writes = tuple(
            target.id
            for target in self.writes
            if not any(_path_contains(target.path, path) for path in changed_paths)
        )
        if unused_writes:
            raise ValueError(
                "augmentation candidate did not change declared targets: "
                + ", ".join(unused_writes)
            )
        changed_events = _changed_event_ids(source, candidate)
        declared_events = tuple(
            target.event_id for target in self.writes if target.event_id is not None
        )
        if set(changed_events) != set(declared_events):
            raise ValueError("augmentation candidate changed undeclared environment events")
        return AugmentationChangeSet(
            changed_paths=changed_paths,
            changed_events=changed_events,
        )


def _paths_overlap(left: str, right: str) -> bool:
    return _path_contains(left, right) or _path_contains(right, left)


def _path_contains(container: str, path: str) -> bool:
    return path == container or path.startswith(f"{container}/")


def _changed_paths(source: JsonValue, candidate: JsonValue) -> list[str]:
    changed: list[str] = []
    pending: list[tuple[JsonValue, JsonValue, str]] = [(source, candidate, "")]
    visited = 0
    while pending:
        source_value, candidate_value, path = pending.pop()
        visited += 1
        if visited > _MAXIMUM_DIFF_NODES:
            raise ValueError("augmentation candidate diff exceeds the 100000-node limit")
        if type(source_value) is not type(candidate_value):
            changed.append(path)
        elif isinstance(source_value, dict) and isinstance(candidate_value, dict):
            for key in reversed(sorted(set(source_value) | set(candidate_value))):
                child_path = f"{path}/{_encode_token(key)}"
                if key not in source_value or key not in candidate_value:
                    changed.append(child_path)
                else:
                    pending.append((source_value[key], candidate_value[key], child_path))
        elif isinstance(source_value, list) and isinstance(candidate_value, list):
            shared_length = min(len(source_value), len(candidate_value))
            changed.extend(
                f"{path}/{index}"
                for index in range(shared_length, max(len(source_value), len(candidate_value)))
            )
            for index in reversed(range(shared_length)):
                pending.append((source_value[index], candidate_value[index], f"{path}/{index}"))
        elif source_value != candidate_value:
            changed.append(path)
    return sorted(changed)


def _changed_event_ids(source: JsonValue, candidate: JsonValue) -> tuple[str, ...]:
    source_events = _validated_events_by_id(source)
    candidate_events = _validated_events_by_id(candidate)
    return tuple(
        event_id
        for event_id in sorted(set(source_events) | set(candidate_events))
        if source_events.get(event_id) != candidate_events.get(event_id)
    )


def _validated_events_by_id(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        return {}
    raw_events = value.get("environment_events")
    if raw_events is None:
        return {}
    if not isinstance(raw_events, list):
        raise ValueError("environment events must be an array")
    events: dict[str, JsonValue] = {}
    for raw_event in raw_events:
        if not isinstance(raw_event, dict) or not isinstance(raw_event.get("id"), str):
            raise ValueError("environment events require string identifiers")
        event_id = cast(str, raw_event["id"])
        if event_id in events:
            raise ValueError("environment event identifiers must be unique")
        events[event_id] = cast(JsonValue, raw_event)
    return events


def _validate_event_target(value: JsonValue, target: ProjectionTarget) -> None:
    event_index = _environment_event_index(target.path)
    if event_index is None or not isinstance(value, dict):
        raise ValueError(f"projection target {target.id!r} does not select its environment event")
    raw_events = value.get("environment_events")
    if not isinstance(raw_events, list) or event_index >= len(raw_events):
        raise ValueError(f"projection target {target.id!r} does not select its environment event")
    raw_event = raw_events[event_index]
    if not isinstance(raw_event, dict) or raw_event.get("id") != target.event_id:
        raise ValueError(f"projection target {target.id!r} does not select its environment event")


def _environment_event_index(path: str) -> int | None:
    tokens = path.split("/")
    if len(tokens) < 3 or tokens[1] != "environment_events":
        return None
    token = tokens[2]
    if not token.isascii() or not token.isdecimal() or (token != "0" and token.startswith("0")):
        return None
    return int(token)


def _resolves(value: JsonValue, path: str) -> bool:
    try:
        resolve_json_pointer(value, path)
    except ValueError:
        return False
    return True


def _encode_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")
