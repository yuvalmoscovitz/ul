from __future__ import annotations

import json
from collections.abc import Iterable
from itertools import islice
from typing import Any, Literal

from pydantic import Field
from ul_core.contracts import SemanticDeconstructor, SemanticRenderer
from ul_core.dataset import InteractionRecord, SemanticFrame, UserInputRecord
from ul_core.models import ULModel


class DatasetAugmentationCandidate(ULModel):
    source_interaction_id: str = Field(min_length=1)
    operator_id: Literal["surface.rephrase"] = "surface.rephrase"
    operator_version: Literal["1.0.0"] = "1.0.0"
    augmented_input: str = Field(min_length=1)
    expected_input_frame: SemanticFrame
    reparsed_input_frame: SemanticFrame
    passed: bool
    failure_reasons: tuple[str, ...] = ()


class DatasetAugmentationResult(ULModel):
    source_frames: tuple[SemanticFrame, ...]
    candidates: tuple[DatasetAugmentationCandidate, ...]


class DatasetAugmentationEngine:
    maximum_records = 100

    def __init__(
        self,
        deconstructor: SemanticDeconstructor,
        renderer: SemanticRenderer,
    ) -> None:
        self._deconstructor = deconstructor
        self._renderer = renderer

    async def augment(
        self,
        records: Iterable[InteractionRecord],
        *,
        max_records: int = 25,
    ) -> DatasetAugmentationResult:
        if not 1 <= max_records <= self.maximum_records:
            raise ValueError(f"max_records must be between 1 and {self.maximum_records}")
        source_records = tuple(islice(records, max_records + 1))
        if len(source_records) > max_records:
            raise ValueError("record count exceeds max_records")
        record_ids = tuple(record.id for record in source_records)
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("interaction record identifiers must be unique")
        source_frames: list[SemanticFrame] = []
        candidates: list[DatasetAugmentationCandidate] = []
        for record in source_records:
            source_frame = await self._deconstructor.deconstruct(record)
            if source_frame.interaction_id != record.id:
                raise ValueError("deconstructed frame must reference its source interaction")
            source_frames.append(source_frame)
            expected_input_frame = _input_only_frame(source_frame)
            if (
                not expected_input_frame.request_units
                or not source_frame.outcomes
                or _has_unresolved_nodes(source_frame)
                or _has_ambiguous_nodes(expected_input_frame)
            ):
                continue
            augmented_input = await self._renderer.render(
                record.raw_input,
                "Rephrase the user input naturally without changing any meaning, request order, "
                "or communication behavior.",
            )
            candidate_record = UserInputRecord(
                id=f"{record.id}:surface.rephrase",
                raw_input=augmented_input,
            )
            reparsed_frame = await self._deconstructor.deconstruct(
                candidate_record, expected_input_frame
            )
            failure_reasons = list(
                _semantic_difference_reasons(expected_input_frame, reparsed_frame)
            )
            if _has_unresolved_nodes(reparsed_frame):
                failure_reasons.append("reparsed frame contains unresolved semantic elements")
            if augmented_input == record.raw_input:
                failure_reasons.append("renderer did not change the source input")
            candidates.append(
                DatasetAugmentationCandidate(
                    source_interaction_id=record.id,
                    augmented_input=augmented_input,
                    expected_input_frame=expected_input_frame,
                    reparsed_input_frame=reparsed_frame,
                    passed=not failure_reasons,
                    failure_reasons=tuple(failure_reasons),
                )
            )
        return DatasetAugmentationResult(
            source_frames=tuple(source_frames), candidates=tuple(candidates)
        )


def _input_only_frame(frame: SemanticFrame) -> SemanticFrame:
    factors = tuple(
        factor.model_copy(update={"evidence": _input_evidence(factor)})
        for factor in frame.factors
        if _has_input_evidence(factor)
    )
    factor_ids = {factor.id for factor in factors}
    request_units = tuple(
        request_unit.model_copy(update={"evidence": _input_evidence(request_unit)})
        for request_unit in frame.request_units
        if _has_input_evidence(request_unit) and set(request_unit.factor_ids) <= factor_ids
    )
    communication_acts = tuple(
        communication_act.model_copy(update={"evidence": _input_evidence(communication_act)})
        for communication_act in frame.communication_acts
        if _has_input_evidence(communication_act)
        and set(communication_act.factor_ids) <= factor_ids
    )
    retained_ids = {
        *factor_ids,
        *(request_unit.id for request_unit in request_units),
        *(communication_act.id for communication_act in communication_acts),
    }
    relations = tuple(
        relation.model_copy(update={"evidence": _input_evidence(relation)})
        for relation in frame.relations
        if _has_input_evidence(relation)
        and set((*relation.source_ids, *relation.target_ids)) <= retained_ids
    )
    return SemanticFrame.model_validate(
        {
            **frame.model_dump(mode="python"),
            "request_units": request_units,
            "factors": factors,
            "relations": relations,
            "communication_acts": communication_acts,
            "outcomes": (),
            "metadata": {},
        }
    )


def _has_input_evidence(element: Any) -> bool:
    return any(evidence.source == "input" for evidence in element.evidence)


def _input_evidence(element: Any) -> tuple[Any, ...]:
    return tuple(evidence for evidence in element.evidence if evidence.source == "input")


def _semantic_difference_reasons(
    expected: SemanticFrame, reparsed: SemanticFrame
) -> tuple[str, ...]:
    expected_semantics = _canonical_semantics(expected)
    reparsed_semantics = _canonical_semantics(reparsed)
    labels = ("factors", "request units", "relations", "communication acts")
    return tuple(
        f"{label} differ from the expected frame"
        for label, expected_part, reparsed_part in zip(
            labels, expected_semantics, reparsed_semantics, strict=True
        )
        if expected_part != reparsed_part
    )


def _has_ambiguous_nodes(frame: SemanticFrame) -> bool:
    factors_by_id = {
        factor.id: (factor.kind, factor.role, _json_key(factor.value)) for factor in frame.factors
    }
    factor_semantics = list(factors_by_id.values())
    request_semantics = [
        (
            request.mode,
            request.predicate,
            tuple(sorted(factors_by_id[factor_id] for factor_id in request.factor_ids)),
        )
        for request in frame.request_units
    ]
    communication_semantics = [
        (
            act.kind,
            tuple(sorted(factors_by_id[factor_id] for factor_id in act.factor_ids)),
            _json_key(act.attributes),
        )
        for act in frame.communication_acts
    ]
    return any(
        len(semantics) != len(set(semantics))
        for semantics in (factor_semantics, request_semantics, communication_semantics)
    )


def _has_unresolved_nodes(frame: SemanticFrame) -> bool:
    return any(
        element.status.casefold().startswith("unresolved")
        for element in (
            *frame.request_units,
            *frame.factors,
            *frame.relations,
            *frame.communication_acts,
            *frame.outcomes,
        )
    )


def _canonical_semantics(frame: SemanticFrame) -> tuple[object, ...]:
    factor_semantics = {
        factor.id: (factor.kind, factor.role, _json_key(factor.value)) for factor in frame.factors
    }
    request_semantics = {
        request.id: (
            request.mode,
            request.predicate,
            tuple(sorted(factor_semantics[factor_id] for factor_id in request.factor_ids)),
        )
        for request in frame.request_units
    }
    communication_semantics = {
        act.id: (
            act.kind,
            tuple(sorted(factor_semantics[factor_id] for factor_id in act.factor_ids)),
            _json_key(act.attributes),
        )
        for act in frame.communication_acts
    }
    endpoint_semantics = {
        **{identifier: ("factor", value) for identifier, value in factor_semantics.items()},
        **{identifier: ("request", value) for identifier, value in request_semantics.items()},
        **{
            identifier: ("communication", value)
            for identifier, value in communication_semantics.items()
        },
    }
    relation_semantics = tuple(
        sorted(
            (
                relation.kind,
                tuple(sorted(endpoint_semantics[item] for item in relation.source_ids)),
                tuple(sorted(endpoint_semantics[item] for item in relation.target_ids)),
            )
            for relation in frame.relations
        )
    )
    return (
        tuple(sorted(factor_semantics.values())),
        tuple(request_semantics[request.id] for request in frame.request_units),
        relation_semantics,
        tuple(communication_semantics[act.id] for act in frame.communication_acts),
    )


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
