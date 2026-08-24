from __future__ import annotations

import copy
import json
import re
from typing import Any, Literal, Self, cast

from pydantic import ConfigDict, Field, JsonValue, field_validator, model_validator

from ul_core.models import ULModel


class _StrictULModel(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


_MAXIMUM_CASE_JSON_BYTES = 1_000_000
_JSON_POINTER_PATTERN = re.compile(r"(?:/(?:[^~/]|~[01])*)*")


class VisibleContextTurn(_StrictULModel):
    id: str = Field(min_length=1, max_length=500)
    role: Literal["user", "assistant", "tool"]
    content: JsonValue
    name: str | None = Field(default=None, min_length=1, max_length=200)


class AugmentationTarget(_StrictULModel):
    id: str = Field(min_length=1, max_length=500)
    kind: Literal["input_field", "conversation_turn"]
    json_pointer: str | None = Field(default=None, max_length=1_000)
    turn_id: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_location(self) -> Self:
        if self.kind == "input_field":
            if (
                self.json_pointer is None
                or self.turn_id is not None
                or not self.json_pointer.startswith("/inputs/")
                or _JSON_POINTER_PATTERN.fullmatch(self.json_pointer) is None
            ):
                raise ValueError("input-field targets require an RFC 6901 /inputs pointer")
        elif self.turn_id is None or self.json_pointer is not None:
            raise ValueError("conversation-turn targets require exactly one turn identifier")
        return self


class CaseFixtureReference(_StrictULModel):
    id: str = Field(min_length=1, max_length=500)
    version: str = Field(min_length=1, max_length=200)


class ObservedEvidenceReference(_StrictULModel):
    kind: Literal["trace", "state"]
    source_id: str = Field(min_length=1, max_length=500)
    reference: str = Field(min_length=1, max_length=1_000)


class CaseEvaluatorReference(_StrictULModel):
    id: str = Field(min_length=1, max_length=500)
    version: str | None = Field(default=None, min_length=1, max_length=200)


class RichInteractionCase(_StrictULModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    id: str = Field(min_length=1, max_length=500)
    inputs: JsonValue
    context: tuple[VisibleContextTurn, ...] = Field(default=(), max_length=100)
    augmentation_targets: tuple[AugmentationTarget, ...] = Field(min_length=1, max_length=100)
    fixture: CaseFixtureReference
    observed_output: JsonValue
    observed_evidence: tuple[ObservedEvidenceReference, ...] = Field(default=(), max_length=100)
    evaluator: CaseEvaluatorReference | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("context", "augmentation_targets", "observed_evidence", mode="before")
    @classmethod
    def accept_json_arrays(cls, value: Any) -> Any:
        return tuple(cast(list[Any], value)) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        if self.observed_output is None:
            raise ValueError("rich interaction cases require an observed output")
        turn_ids = tuple(turn.id for turn in self.context)
        if len(turn_ids) != len(set(turn_ids)):
            raise ValueError("visible context turn identifiers must be unique")
        target_ids = tuple(target.id for target in self.augmentation_targets)
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("augmentation target identifiers must be unique")
        target_locations: set[tuple[str, str]] = set()
        turns_by_id = {turn.id: turn for turn in self.context}
        canonical_case = self.model_dump(mode="json", exclude={"augmentation_targets"})
        for target in self.augmentation_targets:
            location = (
                target.kind,
                target.json_pointer if target.json_pointer is not None else target.turn_id or "",
            )
            if location in target_locations:
                raise ValueError("augmentation targets must identify distinct text values")
            target_locations.add(location)
            if target.kind == "input_field":
                value = resolve_json_pointer(canonical_case, target.json_pointer or "")
            else:
                turn = turns_by_id.get(target.turn_id or "")
                if turn is None:
                    raise ValueError("augmentation target references an unknown context turn")
                if turn.role != "user":
                    raise ValueError("only user context turns may be augmented")
                value = turn.content
            if not isinstance(value, str) or not value:
                raise ValueError("augmentation targets must resolve to non-empty text")
        encoded_case = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded_case) > _MAXIMUM_CASE_JSON_BYTES:
            raise ValueError("rich interaction case exceeds the 1 MB JSON limit")
        return self


def resolve_json_pointer(value: JsonValue, pointer: str) -> JsonValue:
    current: JsonValue = value
    if _JSON_POINTER_PATTERN.fullmatch(pointer) is None:
        raise ValueError("json pointer must follow RFC 6901 syntax")
    for encoded_token in pointer[1:].split("/") if pointer else ():
        token = encoded_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif (
            isinstance(current, list)
            and token.isascii()
            and token.isdecimal()
            and (token == "0" or not token.startswith("0"))
            and int(token) < len(current)
        ):
            current = current[int(token)]
        else:
            raise ValueError("json pointer does not resolve")
    return current


def _replace_json_pointer(value: JsonValue, pointer: str, replacement: str) -> JsonValue:
    copied_value = copy.deepcopy(value)
    tokens = tuple(token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/"))
    parent_pointer = "/".join(pointer.split("/")[:-1])
    parent = resolve_json_pointer(copied_value, parent_pointer)
    final_token = tokens[-1]
    if isinstance(parent, dict):
        parent[final_token] = replacement
    elif isinstance(parent, list):
        parent[int(final_token)] = replacement
    else:
        raise ValueError("augmentation target parent must be an object or array")
    return copied_value


class UserInputRecord(_StrictULModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    id: str = Field(min_length=1, max_length=500)
    raw_input: str = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict, max_length=100)


class InteractionRecord(UserInputRecord):
    raw_observed_output: JsonValue
    structured_input: JsonValue | None = None
    structured_input_target: str | None = Field(default=None, max_length=1_000)
    source_case: RichInteractionCase | None = None
    augmentation_target: AugmentationTarget | None = None

    @model_validator(mode="after")
    def validate_observed_output(self) -> Self:
        if self.raw_observed_output is None:
            raise ValueError("interaction records require an observed output")
        if (self.structured_input is None) != (self.structured_input_target is None):
            raise ValueError("structured input and its augmentation target must be paired")
        if self.structured_input_target is not None:
            if not self.structured_input_target.startswith("/"):
                raise ValueError("structured input target must be an RFC 6901 JSON pointer")
            selected_input = resolve_json_pointer(
                self.structured_input, self.structured_input_target
            )
            if selected_input != self.raw_input:
                raise ValueError("structured input target must select the raw input text")
        if (self.source_case is None) != (self.augmentation_target is None):
            raise ValueError("rich interaction source and augmentation target must be paired")
        if self.source_case is not None and self.augmentation_target not in (
            self.source_case.augmentation_targets
        ):
            raise ValueError("interaction augmentation target must belong to its source case")
        return self

    @property
    def source_interaction_id(self) -> str:
        return self.source_case.id if self.source_case is not None else self.id

    def target_input(self, selected_text: str | None = None) -> JsonValue:
        selected_value = selected_text if selected_text is not None else self.raw_input
        if self.structured_input is None or self.structured_input_target is None:
            return selected_value
        return _replace_json_pointer(
            self.structured_input, self.structured_input_target, selected_value
        )

    @property
    def input_value(self) -> JsonValue:
        return self.structured_input if self.structured_input is not None else self.raw_input

    @property
    def augmentation_path(self) -> str:
        if self.source_case is None or self.augmentation_target is None:
            return self.structured_input_target or "/raw_input"
        if self.augmentation_target.kind == "input_field":
            return self.augmentation_target.json_pointer or ""
        turn_index = next(
            index
            for index, turn in enumerate(self.source_case.context)
            if turn.id == self.augmentation_target.turn_id
        )
        return f"/context/{turn_index}/content"

    def with_input_value(self, value: JsonValue) -> InteractionRecord:
        if self.structured_input_target is None:
            if not isinstance(value, str):
                raise ValueError("executable input must remain text")
            return self.model_copy(update={"raw_input": value})
        selected_input = resolve_json_pointer(value, self.structured_input_target)
        if not isinstance(selected_input, str) or not selected_input:
            raise ValueError("structured input target must remain non-empty text")
        return self.model_copy(update={"raw_input": selected_input, "structured_input": value})

    def probe_context(self, selected_text: str | None = None) -> dict[str, JsonValue]:
        if self.structured_input is not None:
            return {"ul.target.input": self.target_input(selected_text)}
        if self.source_case is None or self.augmentation_target is None:
            return {}
        selected_value = selected_text if selected_text is not None else self.raw_input
        source_case = self.source_case
        target = self.augmentation_target
        inputs = copy.deepcopy(source_case.inputs)
        context = [cast(JsonValue, turn.model_dump(mode="json")) for turn in source_case.context]
        if target.kind == "input_field":
            replaced_case = _replace_json_pointer(
                {"inputs": inputs}, target.json_pointer or "", selected_value
            )
            if not isinstance(replaced_case, dict):
                raise AssertionError("input targets require an object case envelope")
            inputs = replaced_case["inputs"]
        else:
            for turn in context:
                if isinstance(turn, dict) and turn.get("id") == target.turn_id:
                    turn["content"] = selected_value
                    break
        return cast(
            dict[str, JsonValue],
            {
                "schema_version": source_case.schema_version,
                "source_interaction_id": source_case.id,
                "inputs": inputs,
                "context": context,
                "augmentation_target": cast(JsonValue, target.model_dump(mode="json")),
                "fixture": cast(JsonValue, source_case.fixture.model_dump(mode="json")),
                "observed_evidence": [
                    cast(JsonValue, reference.model_dump(mode="json"))
                    for reference in source_case.observed_evidence
                ],
                "evaluator": (
                    cast(JsonValue, source_case.evaluator.model_dump(mode="json"))
                    if source_case.evaluator is not None
                    else None
                ),
                "metadata": copy.deepcopy(source_case.metadata),
            },
        )


def project_rich_interaction_case(
    source_case: RichInteractionCase,
) -> tuple[InteractionRecord, ...]:
    projected: list[InteractionRecord] = []
    canonical_case = source_case.model_dump(mode="json", exclude={"augmentation_targets"})
    context_by_id = {turn.id: turn for turn in source_case.context}
    for target in source_case.augmentation_targets:
        if target.kind == "input_field":
            raw_input = resolve_json_pointer(canonical_case, target.json_pointer or "")
        else:
            raw_input = context_by_id[target.turn_id or ""].content
        if not isinstance(raw_input, str):
            raise AssertionError("validated augmentation targets resolve to text")
        projected.append(
            InteractionRecord(
                id=f"{source_case.id}::{target.id}",
                raw_input=raw_input,
                raw_observed_output=source_case.observed_output,
                metadata=source_case.metadata,
                source_case=source_case,
                augmentation_target=target,
            )
        )
    return tuple(projected)


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
