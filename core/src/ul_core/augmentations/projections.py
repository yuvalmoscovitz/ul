"""Typed read and write projections for controlled augmentation changes."""

from __future__ import annotations

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


class _ProjectionModel(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProjectionTarget(_ProjectionModel):
    id: str = Field(min_length=1, max_length=500)
    surface: AugmentationTargetSurface
    path: str = Field(max_length=1_000)
    event_id: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_event_selector(self) -> Self:
        try:
            resolve_json_pointer({}, self.path)
        except ValueError as error:
            if str(error) == "json pointer must follow RFC 6901 syntax":
                raise ValueError("projection target path must follow RFC 6901 syntax") from None
        if (self.surface == "environment") != (self.event_id is not None):
            raise ValueError("environment targets require exactly one event identifier")
        return self


class AugmentationChangeSet(_ProjectionModel):
    changed_paths: tuple[str, ...] = Field(min_length=1)
    changed_events: tuple[str, ...] = ()


class AugmentationProjection(_ProjectionModel):
    reads: tuple[ProjectionTarget, ...] = Field(min_length=1, max_length=100)
    writes: tuple[ProjectionTarget, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_targets(self) -> Self:
        identifiers = tuple(target.id for target in (*self.reads, *self.writes))
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("projection target identifiers must be unique")
        for index, target in enumerate(self.writes):
            for other in self.writes[index + 1 :]:
                if _paths_overlap(target.path, other.path):
                    raise ValueError("write projection targets conflict")
        return self

    def read(self, source: JsonValue) -> dict[str, JsonValue]:
        self.validate_source(source)
        return {target.id: resolve_json_pointer(source, target.path) for target in self.reads}

    def validate_source(self, source: JsonValue) -> None:
        for target in (*self.reads, *self.writes):
            try:
                resolve_json_pointer(source, target.path)
            except ValueError:
                raise ValueError(
                    f"projection target {target.id!r} does not resolve before execution"
                ) from None
        for target in self.reads:
            if target.event_id is not None and target.event_id not in _events_by_id(source):
                raise ValueError(
                    f"projection target {target.id!r} references a missing environment event"
                )

    def validate_candidate(self, source: JsonValue, candidate: JsonValue) -> AugmentationChangeSet:
        self.validate_source(source)
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
    return container == "" or path == container or path.startswith(f"{container}/")


def _changed_paths(source: JsonValue, candidate: JsonValue, path: str = "") -> list[str]:
    if type(source) is not type(candidate):
        return [path]
    if isinstance(source, dict) and isinstance(candidate, dict):
        changed: list[str] = []
        for key in sorted(set(source) | set(candidate)):
            child_path = f"{path}/{_encode_token(key)}"
            if key not in source or key not in candidate:
                changed.append(child_path)
            else:
                changed.extend(_changed_paths(source[key], candidate[key], child_path))
        return changed
    if isinstance(source, list) and isinstance(candidate, list):
        changed = []
        shared_length = min(len(source), len(candidate))
        for index in range(shared_length):
            changed.extend(_changed_paths(source[index], candidate[index], f"{path}/{index}"))
        changed.extend(
            f"{path}/{index}" for index in range(shared_length, max(len(source), len(candidate)))
        )
        return changed
    return [] if source == candidate else [path]


def _changed_event_ids(source: JsonValue, candidate: JsonValue) -> tuple[str, ...]:
    source_events = _events_by_id(source)
    candidate_events = _events_by_id(candidate)
    return tuple(
        event_id
        for event_id in sorted(set(source_events) | set(candidate_events))
        if source_events.get(event_id) != candidate_events.get(event_id)
    )


def _events_by_id(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        return {}
    raw_events = value.get("environment_events")
    if not isinstance(raw_events, list):
        return {}
    events: dict[str, JsonValue] = {}
    for raw_event in raw_events:
        if isinstance(raw_event, dict) and isinstance(raw_event.get("id"), str):
            events[cast(str, raw_event["id"])] = cast(JsonValue, raw_event)
    return events


def _encode_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")
