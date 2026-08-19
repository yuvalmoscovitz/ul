from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Literal, Self, cast

from pydantic import ConfigDict, Field, JsonValue, model_validator

from ul_core.models import ULModel


class _StrictULModel(ULModel):
    model_config = ConfigDict(strict=True)


_MAXIMUM_SANDBOX_SETUP_BYTES = 65_536
_MAXIMUM_SANDBOX_SETUP_DEPTH = 20
_MAXIMUM_SANDBOX_SETUP_VALUES = 1_000


class SandboxSetupFixture(_StrictULModel):
    payload: dict[str, JsonValue]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def from_payload(cls, payload: object) -> Self:
        validated_payload = _validate_sandbox_setup_payload(payload)
        copied_payload = cast(
            dict[str, JsonValue],
            json.loads(_canonical_sandbox_setup(validated_payload)),
        )
        return cls(
            payload=copied_payload,
            sha256=_sandbox_setup_sha256(copied_payload),
        )

    @model_validator(mode="after")
    def validate_digest(self) -> Self:
        self.verify_digest()
        return self

    def verify_digest(self) -> None:
        _validate_sandbox_setup_payload(self.payload)
        if self.sha256 != _sandbox_setup_sha256(self.payload):
            raise ValueError("sandbox setup fixture digest must match its payload")


class UserInputRecord(_StrictULModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    id: str = Field(min_length=1)
    raw_input: str = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class InteractionRecord(UserInputRecord):
    raw_observed_output: JsonValue
    sandbox_setup: SandboxSetupFixture | None = None

    @model_validator(mode="after")
    def validate_observed_output(self) -> Self:
        if self.raw_observed_output is None:
            raise ValueError("interaction records require an observed output")
        return self


def _validate_sandbox_setup_payload(payload: object) -> dict[str, JsonValue]:
    if not isinstance(payload, dict):
        raise ValueError("sandbox setup fixture must be a JSON object")
    values_to_visit: list[tuple[object, int]] = [(payload, 0)]
    value_count = 0
    while values_to_visit:
        value, depth = values_to_visit.pop()
        if depth > _MAXIMUM_SANDBOX_SETUP_DEPTH:
            raise ValueError("sandbox setup fixture exceeds the nesting limit")
        value_count += 1
        if value_count > _MAXIMUM_SANDBOX_SETUP_VALUES:
            raise ValueError("sandbox setup fixture contains too many values")
        if value is None or isinstance(value, bool | int | str):
            continue
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("sandbox setup fixture must contain standard JSON values")
            continue
        if isinstance(value, list):
            values_to_visit.extend((item, depth + 1) for item in cast(list[object], value))
            continue
        if isinstance(value, dict):
            object_value = cast(dict[object, object], value)
            if not all(isinstance(key, str) for key in object_value):
                raise ValueError("sandbox setup fixture object keys must be strings")
            values_to_visit.extend((item, depth + 1) for item in object_value.values())
            continue
        raise ValueError("sandbox setup fixture must contain JSON values")
    validated_payload = cast(dict[str, JsonValue], payload)
    try:
        encoded_payload = _canonical_sandbox_setup(validated_payload)
    except (TypeError, ValueError):
        raise ValueError("sandbox setup fixture must contain standard JSON values") from None
    if len(encoded_payload) > _MAXIMUM_SANDBOX_SETUP_BYTES:
        raise ValueError(
            f"sandbox setup fixture exceeds {_MAXIMUM_SANDBOX_SETUP_BYTES} UTF-8 bytes"
        )
    return validated_payload


def _canonical_sandbox_setup(payload: dict[str, JsonValue]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sandbox_setup_sha256(payload: dict[str, JsonValue]) -> str:
    return hashlib.sha256(_canonical_sandbox_setup(payload)).hexdigest()


class RenderedUserInput(_StrictULModel):
    text: str = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ObservedAgentOutput(_StrictULModel):
    raw_output: JsonValue
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_output(self) -> Self:
        if self.raw_output is None:
            raise ValueError("agent observations require an output")
        return self


class EvidenceReference(_StrictULModel):
    source: Literal["input", "output"]
    json_pointer: str
    text_quote: str | None = Field(min_length=1)

    @model_validator(mode="after")
    def validate_location(self) -> Self:
        if not re.fullmatch(r"(?:/(?:[^~/]|~[01])*)*", self.json_pointer):
            raise ValueError("json_pointer must follow RFC 6901 syntax")
        return self


class _SemanticElement(_StrictULModel):
    id: str = Field(min_length=1)
    evidence: tuple[EvidenceReference, ...] = ()
    confidence: float = Field(ge=0, le=1)
    status: str = Field(min_length=1)


class RequestUnit(_SemanticElement):
    mode: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    factor_ids: tuple[str, ...] = ()


class SemanticFactor(_SemanticElement):
    kind: str = Field(min_length=1)
    role: str = Field(min_length=1)
    value: JsonValue


class SemanticRelation(_SemanticElement):
    kind: str = Field(min_length=1)
    source_ids: tuple[str, ...] = Field(min_length=1)
    target_ids: tuple[str, ...] = Field(min_length=1)


class CommunicationAct(_SemanticElement):
    kind: str = Field(min_length=1)
    factor_ids: tuple[str, ...] = ()
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class ObservedOutcome(_SemanticElement):
    request_unit_ids: tuple[str, ...] = ()
    position: int = Field(ge=0)
    kind: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    fields: dict[str, JsonValue] = Field(default_factory=dict)
    propositions: tuple[str, ...] = ()


class SemanticFrame(_StrictULModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    interaction_id: str = Field(min_length=1)
    request_units: tuple[RequestUnit, ...] = ()
    factors: tuple[SemanticFactor, ...] = ()
    relations: tuple[SemanticRelation, ...] = ()
    communication_acts: tuple[CommunicationAct, ...] = ()
    outcomes: tuple[ObservedOutcome, ...] = ()
    extractor_version: str = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        elements = (
            *self.request_units,
            *self.factors,
            *self.relations,
            *self.communication_acts,
            *self.outcomes,
        )
        element_ids = tuple(element.id for element in elements)
        if len(element_ids) != len(set(element_ids)):
            raise ValueError("semantic element identifiers must be globally unique")

        factor_ids = {factor.id for factor in self.factors}
        request_unit_ids = {request_unit.id for request_unit in self.request_units}
        relation_endpoint_ids = {
            element.id
            for element in (
                *self.request_units,
                *self.factors,
                *self.communication_acts,
                *self.outcomes,
            )
        }

        for request_unit in self.request_units:
            self._validate_references(request_unit.factor_ids, factor_ids, request_unit.id)
        for communication_act in self.communication_acts:
            self._validate_references(
                communication_act.factor_ids, factor_ids, communication_act.id
            )
        for relation in self.relations:
            self._validate_references(relation.source_ids, relation_endpoint_ids, relation.id)
            self._validate_references(relation.target_ids, relation_endpoint_ids, relation.id)
        for outcome in self.outcomes:
            self._validate_references(outcome.request_unit_ids, request_unit_ids, outcome.id)

        positions = tuple(outcome.position for outcome in self.outcomes)
        if len(positions) != len(set(positions)):
            raise ValueError("outcome positions must be unique")
        return self

    @staticmethod
    def _validate_references(
        references: tuple[str, ...], known_ids: set[str], owner_id: str
    ) -> None:
        if len(references) != len(set(references)):
            raise ValueError(f"duplicate reference on semantic element {owner_id}")
        unknown_ids = set(references) - known_ids
        if unknown_ids:
            raise ValueError(
                f"unknown reference on semantic element {owner_id}: {sorted(unknown_ids)}"
            )


class SemanticDelta(_StrictULModel):
    category: Literal[
        "request",
        "entity",
        "value",
        "constraint",
        "negation",
        "request_order",
        "relationship",
        "cardinality",
        "communication",
    ]
    operation: Literal["added", "removed", "changed", "reordered"]
    source_quote: str | None = Field(default=None, min_length=1)
    candidate_quote: str | None = Field(default=None, min_length=1)
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.source_quote is None and self.candidate_quote is None:
            raise ValueError("semantic deltas require source or candidate evidence")
        return self


class SemanticEquivalenceAssessment(_StrictULModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    verdict: Literal["equivalent", "different", "uncertain"]
    explanation: str = Field(min_length=1)
    deltas: tuple[SemanticDelta, ...] = ()
    verifier_version: str = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_verdict(self) -> Self:
        if self.verdict == "equivalent" and self.deltas:
            raise ValueError("equivalent assessments cannot contain deltas")
        if self.verdict == "different" and not self.deltas:
            raise ValueError("different assessments require at least one delta")
        return self
