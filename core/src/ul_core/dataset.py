from __future__ import annotations

import re
from typing import Literal, Self

from pydantic import ConfigDict, Field, JsonValue, model_validator

from ul_core.models import ULModel


class _StrictULModel(ULModel):
    model_config = ConfigDict(strict=True)


class UserInputRecord(_StrictULModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    id: str = Field(min_length=1)
    raw_input: str = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class InteractionRecord(UserInputRecord):
    raw_observed_output: JsonValue

    @model_validator(mode="after")
    def validate_observed_output(self) -> Self:
        if self.raw_observed_output is None:
            raise ValueError("interaction records require an observed output")
        return self


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
