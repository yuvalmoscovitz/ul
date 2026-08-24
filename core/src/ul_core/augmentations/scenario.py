"""Deterministic scenario augmentation runtime implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, cast

from pydantic import JsonValue

from ul_core.augmentations.definitions import builtin_augmentation_catalog
from ul_core.augmentations.projections import (
    AugmentationProjection,
    AugmentationTargetSurface,
    ProjectionTarget,
)
from ul_core.augmentations.registry import (
    Applicability,
    AugmentationMetadata,
    AugmentationResult,
    ValidationResult,
)
from ul_core.models import (
    Action,
    ActionEffect,
    Artifact,
    AugmentationApplication,
    AugmentationHints,
    ConversationRole,
    ConversationTurn,
    EnvironmentEvent,
    EventTiming,
    ItemValidity,
    OracleRelation,
    Scenario,
    ShrinkMetadata,
)


class BuiltinAugmentation(ABC):
    metadata: ClassVar[AugmentationMetadata]

    @abstractmethod
    def applicability(self, scenario: Scenario) -> Applicability: ...

    @abstractmethod
    def apply(self, scenario: Scenario) -> tuple[AugmentationResult, ...]: ...

    def validate(self, source: Scenario, candidate: Scenario) -> ValidationResult:
        issues: list[str] = []
        if candidate.provenance.parent_scenario_id != source.id:
            issues.append("candidate provenance does not reference the source scenario")
        if not candidate.provenance.lineage:
            issues.append("candidate is missing augmentation lineage")
        elif candidate.provenance.lineage[-1].augmentation_id != self.metadata.id:
            issues.append("candidate lineage does not identify this augmentation")
        else:
            application = candidate.provenance.lineage[-1]
            if application.version != self.metadata.version:
                issues.append("candidate lineage has the wrong augmentation version")
            if application.changed_paths == ():
                issues.append("candidate lineage has no changed paths")
            if candidate.provenance.lineage[:-1] != source.provenance.lineage:
                issues.append("candidate does not preserve source lineage")
        source_semantics = source.model_dump(exclude={"id", "provenance"})
        candidate_semantics = candidate.model_dump(exclude={"id", "provenance"})
        if candidate_semantics == source_semantics:
            issues.append("augmentation did not change scenario semantics")
        return ValidationResult(valid=not issues, issues=tuple(issues))

    def build_result(
        self,
        source: Scenario,
        *,
        variant: str,
        projection: AugmentationProjection,
        **updates: object,
    ) -> AugmentationResult:
        source_semantics = source.model_dump(mode="json", exclude={"id", "provenance"})
        projection.read(source_semantics)
        candidate_without_lineage = source.model_copy(deep=True, update=updates)
        candidate_semantics = candidate_without_lineage.model_dump(
            mode="json", exclude={"id", "provenance"}
        )
        changes = projection.validate_candidate(source_semantics, candidate_semantics)
        application = AugmentationApplication(
            augmentation_id=self.metadata.id,
            version=self.metadata.version,
            changed_paths=changes.changed_paths,
            changed_events=changes.changed_events,
        )
        provenance = source.provenance.model_copy(
            update={
                "parent_scenario_id": source.id,
                "lineage": (*source.provenance.lineage, application),
            }
        )
        candidate = source.model_copy(
            deep=True,
            update={
                "id": f"{source.id}::{self.metadata.id}@{self.metadata.version}:{variant}",
                "provenance": provenance,
                **updates,
            },
        )
        candidate = Scenario.model_validate(candidate.model_dump(mode="python"))
        validation = self.validate(source, candidate)
        if not validation.valid:
            raise ValueError("; ".join(validation.issues))
        return AugmentationResult(
            scenario=candidate,
            augmentation_id=self.metadata.id,
            augmentation_version=self.metadata.version,
            variant=variant,
            projection=projection,
            changed_paths=changes.changed_paths,
            changed_events=changes.changed_events,
            oracle_relation=self.metadata.oracle_relation,
            shrink=self.metadata.shrink,
        )


def _metadata(
    augmentation_id: str,
    oracle_kind: str,
    *,
    invariants: tuple[str, ...] = (),
    permitted_changes: tuple[str, ...] = (),
    shrink: ShrinkMetadata | None = None,
) -> AugmentationMetadata:
    definition = builtin_augmentation_catalog().get(augmentation_id)
    binding = next(item for item in definition.bindings if item.mode == "scenario_materialization")
    return AugmentationMetadata(
        id=definition.ref.id,
        version=definition.ref.version,
        category=definition.surface,
        summary=definition.summary,
        required_features=binding.requirements.required_source_features,
        oracle_relation=OracleRelation(
            kind=oracle_kind,
            description=definition.expected_relation,
            invariants=invariants,
            permitted_changes=permitted_changes,
        ),
        shrink=shrink or ShrinkMetadata(),
    )


class AmbiguityAugmentation(BuiltinAugmentation):
    metadata = _metadata(
        "conversation.ambiguity",
        "requires_disambiguation",
        invariants=("no irreversible action before disambiguation",),
        permitted_changes=("clarification dialogue",),
        shrink=ShrinkMetadata(removable_paths=("artifacts",)),
    )

    def applicability(self, scenario: Scenario) -> Applicability:
        if not scenario.artifacts:
            return Applicability(applicable=False, reasons=("scenario has no artifact",))
        if not any(turn.role == ConversationRole.USER for turn in scenario.conversation):
            return Applicability(applicable=False, reasons=("scenario has no user turn",))
        return Applicability(applicable=True)

    def apply(self, scenario: Scenario) -> tuple[AugmentationResult, ...]:
        if not self.applicability(scenario).applicable:
            return ()
        hint = _augmentation_hints(scenario).ambiguity
        source_artifact = (
            scenario.artifacts[0]
            if hint is None
            else next(
                artifact
                for artifact in scenario.artifacts
                if artifact.id == hint.source_artifact_id
            )
        )
        artifact_id = hint.alternative_id if hint is not None else None
        if artifact_id is None:
            artifact_id = _unique_id(
                f"{source_artifact.id}-alternative",
                {artifact.id for artifact in scenario.artifacts},
            )
        alternative = Artifact(
            id=artifact_id,
            kind=source_artifact.kind,
            label=_hint_value(hint.label if hint else None, source_artifact.label),
            state=_hint_value(hint.state if hint else None, source_artifact.state),
            version=_hint_value(hint.version if hint else None, source_artifact.version),
            attributes={
                **source_artifact.attributes,
                **({} if hint is None else hint.attributes),
                "augmentation_role": "plausible_alternative",
            },
        )
        conversation = tuple(
            _make_reference_ambiguous(
                turn,
                source_artifact,
                alternative,
                None if hint is None else hint.replacement_text,
            )
            if turn.role == ConversationRole.USER
            else turn
            for turn in scenario.conversation
        )
        return (
            self.build_result(
                scenario,
                variant="similar-artifact",
                projection=AugmentationProjection(
                    reads=(
                        _target("source-artifact", "state", "/artifacts/0"),
                        _target("source-conversation", "conversation", "/conversation"),
                    ),
                    writes=(
                        _target("candidate-artifacts", "state", "/artifacts"),
                        _target("candidate-conversation", "conversation", "/conversation"),
                        *(
                            (
                                _target(
                                    "consumed-ambiguity-hint",
                                    "structured_input",
                                    "/metadata/augmentation_hints",
                                ),
                            )
                            if hint is not None
                            else ()
                        ),
                    ),
                ),
                artifacts=(*scenario.artifacts, alternative),
                conversation=conversation,
                **(
                    {"metadata": _metadata_without_hint(scenario, "ambiguity")}
                    if hint is not None
                    else {}
                ),
            ),
        )


class LaterCorrectionAugmentation(BuiltinAugmentation):
    metadata = _metadata(
        "conversation.correction_after_first_response",
        "latest_intent_wins",
        invariants=("uncorrected parameters remain unchanged",),
        permitted_changes=("corrected parameter", "conversation length"),
        shrink=ShrinkMetadata(
            removable_paths=("conversation",), simplifications=("retain only the corrected field",)
        ),
    )

    def applicability(self, scenario: Scenario) -> Applicability:
        if not any(turn.role == ConversationRole.USER for turn in scenario.conversation):
            return Applicability(applicable=False, reasons=("scenario has no user turn",))
        if _correction_target(scenario) is None:
            return Applicability(
                applicable=False, reasons=("scenario has no correctable parameter",)
            )
        return Applicability(applicable=True)

    def apply(self, scenario: Scenario) -> tuple[AugmentationResult, ...]:
        correction = _correction_target(scenario)
        if correction is None or not self.applicability(scenario).applicable:
            return ()
        correction_hint = _augmentation_hints(scenario).later_correction
        action_index, parameter_name, corrected_value, message = correction
        actions = list(scenario.actions)
        action = actions[action_index]
        actions[action_index] = action.model_copy(
            update={"parameters": {**action.parameters, parameter_name: corrected_value}}
        )
        turn_id = _unique_id("later-correction", {turn.id for turn in scenario.conversation})
        correction = ConversationTurn(
            id=turn_id,
            role=ConversationRole.USER,
            actor_id=next(
                turn.actor_id
                for turn in reversed(scenario.conversation)
                if turn.role == ConversationRole.USER
            ),
            content=message,
            metadata={
                "corrects_action_id": action.id,
                "parameter": parameter_name,
                "value": corrected_value,
            },
        )
        return (
            self.build_result(
                scenario,
                variant=parameter_name,
                projection=AugmentationProjection(
                    reads=(
                        _target(
                            "source-parameter",
                            "structured_input",
                            f"/actions/{action_index}/parameters/{_pointer_token(parameter_name)}",
                        ),
                        _target("source-conversation", "conversation", "/conversation"),
                    ),
                    writes=(
                        _target(
                            "corrected-parameter",
                            "structured_input",
                            f"/actions/{action_index}/parameters/{_pointer_token(parameter_name)}",
                        ),
                        _target("correction-turn", "conversation", "/conversation"),
                        *(
                            (
                                _target(
                                    "consumed-correction-hint",
                                    "structured_input",
                                    "/metadata/augmentation_hints",
                                ),
                            )
                            if correction_hint is not None
                            else ()
                        ),
                    ),
                ),
                actions=tuple(actions),
                conversation=(*scenario.conversation, correction),
                **(
                    {"metadata": _metadata_without_hint(scenario, "later_correction")}
                    if correction_hint is not None
                    else {}
                ),
            ),
        )


class BoundaryShiftAugmentation(BuiltinAugmentation):
    metadata = _metadata(
        "input.policy.boundary_shift",
        "boundary_sensitive",
        invariants=("policy interpretation", "all non-target parameters"),
        permitted_changes=("boundary-dependent decision",),
        shrink=ShrinkMetadata(simplifications=("use the nearest representable boundary value",)),
    )

    def applicability(self, scenario: Scenario) -> Applicability:
        action_by_id = {action.id: action for action in scenario.actions}
        for policy in scenario.policies:
            for boundary in policy.boundaries:
                action = action_by_id.get(boundary.action_id)
                if action is not None and boundary.parameter in action.parameters:
                    return Applicability(applicable=True)
        return Applicability(applicable=False, reasons=("no materializable policy boundary",))

    def apply(self, scenario: Scenario) -> tuple[AugmentationResult, ...]:
        boundary_match = _first_boundary(scenario)
        if boundary_match is None:
            return ()
        action_index, parameter_name, threshold, increment = boundary_match
        results: list[AugmentationResult] = []
        for variant, value in (
            ("below", threshold - increment),
            ("equal", threshold),
            ("above", threshold + increment),
        ):
            if scenario.actions[action_index].parameters[parameter_name] == value:
                continue
            actions = list(scenario.actions)
            action = actions[action_index]
            actions[action_index] = action.model_copy(
                update={"parameters": {**action.parameters, parameter_name: value}}
            )
            results.append(
                self.build_result(
                    scenario,
                    variant=variant,
                    projection=AugmentationProjection(
                        reads=(
                            _target("policy-boundary", "policy", "/policies"),
                            _target(
                                "source-parameter",
                                "structured_input",
                                f"/actions/{action_index}/parameters/{_pointer_token(parameter_name)}",
                            ),
                        ),
                        writes=(
                            _target(
                                "boundary-parameter",
                                "policy",
                                f"/actions/{action_index}/parameters/{_pointer_token(parameter_name)}",
                            ),
                        ),
                    ),
                    actions=tuple(actions),
                )
            )
        return tuple(results)


class ExistingPartialOperationAugmentation(BuiltinAugmentation):
    metadata = _metadata(
        "environment.state.existing_partial_operation",
        "remaining_work_only",
        invariants=("already committed effects are not repeated",),
        permitted_changes=("remaining work",),
        shrink=ShrinkMetadata(removable_paths=("environment_events",)),
    )

    def applicability(self, scenario: Scenario) -> Applicability:
        action = _first_action(scenario, ActionEffect.WRITE)
        if action is None:
            return _write_applicability(scenario)
        if _has_action_event(
            scenario, action, "existing_partial_operation", EventTiming.BEFORE_ACTION
        ):
            return Applicability(
                applicable=False, reasons=("partial-operation event already exists",)
            )
        return Applicability(applicable=True)

    def apply(self, scenario: Scenario) -> tuple[AugmentationResult, ...]:
        action = _first_action(scenario, ActionEffect.WRITE)
        if action is None or not self.applicability(scenario).applicable:
            return ()
        event = _action_event(
            scenario,
            action,
            kind="existing_partial_operation",
            timing=EventTiming.BEFORE_ACTION,
            payload={"completed_fraction": 0.5},
        )
        return (_event_result(self, scenario, event),)


class StateChangeBetweenReadWriteAugmentation(BuiltinAugmentation):
    metadata = _metadata(
        "environment.state.change_between_read_write",
        "revalidate_before_write",
        invariants=("writes use current authoritative state",),
        permitted_changes=("decision after revalidation",),
        shrink=ShrinkMetadata(removable_paths=("environment_events",)),
    )

    def applicability(self, scenario: Scenario) -> Applicability:
        pair = _read_write_pair(scenario)
        if pair is None:
            return Applicability(applicable=False, reasons=("no ordered read/write pair",))
        read_action, write_action = pair
        if any(
            event.kind == "state_change"
            and event.timing == EventTiming.BETWEEN_ACTIONS
            and event.after_action_id == read_action.id
            and event.before_action_id == write_action.id
            for event in scenario.environment_events
        ):
            return Applicability(applicable=False, reasons=("state-change event already exists",))
        return Applicability(applicable=True)

    def apply(self, scenario: Scenario) -> tuple[AugmentationResult, ...]:
        pair = _read_write_pair(scenario)
        if pair is None or not self.applicability(scenario).applicable:
            return ()
        read_action, write_action = pair
        event = EnvironmentEvent(
            id=_unique_id("state-change", {item.id for item in scenario.environment_events}),
            kind="state_change",
            timing=EventTiming.BETWEEN_ACTIONS,
            after_action_id=read_action.id,
            before_action_id=write_action.id,
            target_ids=tuple(
                dict.fromkeys(
                    (
                        *read_action.artifact_ids,
                        *read_action.resource_ids,
                        *write_action.artifact_ids,
                        *write_action.resource_ids,
                    )
                )
            ),
            payload={"new_state": "changed"},
        )
        return (_event_result(self, scenario, event),)


class StaleObservationAugmentation(BuiltinAugmentation):
    metadata = _metadata(
        "environment.tool.stale_observation",
        "authoritative_state_required",
        invariants=("underlying authoritative state",),
        permitted_changes=("observed state", "revalidation behavior"),
        shrink=ShrinkMetadata(removable_paths=("environment_events",)),
    )

    def applicability(self, scenario: Scenario) -> Applicability:
        action = _first_action(scenario, ActionEffect.READ)
        if action is None:
            return Applicability(applicable=False, reasons=("scenario has no read action",))
        if _has_action_event(scenario, action, "stale_observation", EventTiming.ON_OBSERVATION):
            return Applicability(applicable=False, reasons=("stale observation already exists",))
        return Applicability(applicable=True)

    def apply(self, scenario: Scenario) -> tuple[AugmentationResult, ...]:
        action = _first_action(scenario, ActionEffect.READ)
        if action is None or not self.applicability(scenario).applicable:
            return ()
        event = _action_event(
            scenario,
            action,
            kind="stale_observation",
            timing=EventTiming.ON_OBSERVATION,
            payload={"stale": True},
        )
        return (_event_result(self, scenario, event),)


class TimeoutBeforeCommitAugmentation(BuiltinAugmentation):
    metadata = _metadata(
        "environment.tool.timeout_before_commit",
        "known_not_committed",
        invariants=("no effect exists before retry",),
        permitted_changes=("retry behavior",),
        shrink=ShrinkMetadata(removable_paths=("environment_events",)),
    )

    def applicability(self, scenario: Scenario) -> Applicability:
        return _timeout_applicability(scenario, EventTiming.BEFORE_ACTION, "not_committed")

    def apply(self, scenario: Scenario) -> tuple[AugmentationResult, ...]:
        action = _first_action(scenario, ActionEffect.WRITE)
        if action is None or not self.applicability(scenario).applicable:
            return ()
        event = _action_event(
            scenario,
            action,
            kind="timeout",
            timing=EventTiming.BEFORE_ACTION,
            payload={"commit_state": "not_committed"},
        )
        return (_event_result(self, scenario, event),)


class TimeoutAfterCommitAugmentation(BuiltinAugmentation):
    metadata = _metadata(
        "environment.tool.timeout_after_commit",
        "uncertain_outcome",
        invariants=("committed effect is not duplicated",),
        permitted_changes=("status-check behavior",),
        shrink=ShrinkMetadata(removable_paths=("environment_events",)),
    )

    def applicability(self, scenario: Scenario) -> Applicability:
        return _timeout_applicability(scenario, EventTiming.AFTER_ACTION, "committed")

    def apply(self, scenario: Scenario) -> tuple[AugmentationResult, ...]:
        action = _first_action(scenario, ActionEffect.WRITE)
        if action is None or not self.applicability(scenario).applicable:
            return ()
        event = _action_event(
            scenario,
            action,
            kind="timeout",
            timing=EventTiming.AFTER_ACTION,
            payload={"commit_state": "committed", "acknowledgement": "lost"},
        )
        return (_event_result(self, scenario, event),)


class MixedValidityBatchAugmentation(BuiltinAugmentation):
    metadata = _metadata(
        "input.batch.mixed_validity",
        "itemwise_validity",
        invariants=("invalid item has no prohibited effect",),
        permitted_changes=("valid item handling", "batch rejection"),
        shrink=ShrinkMetadata(simplifications=("retain one valid and one invalid item",)),
    )

    def applicability(self, scenario: Scenario) -> Applicability:
        if not any(
            len(action.batch_items) >= 2
            and all(item.validity == ItemValidity.VALID for item in action.batch_items)
            for action in scenario.actions
        ):
            return Applicability(applicable=False, reasons=("no all-valid multi-item action",))
        return Applicability(applicable=True)

    def apply(self, scenario: Scenario) -> tuple[AugmentationResult, ...]:
        for action_index, action in enumerate(scenario.actions):
            if len(action.batch_items) < 2 or not all(
                item.validity == ItemValidity.VALID for item in action.batch_items
            ):
                continue
            items = list(action.batch_items)
            items[-1] = items[-1].model_copy(update={"validity": ItemValidity.INVALID})
            actions = list(scenario.actions)
            actions[action_index] = action.model_copy(update={"batch_items": tuple(items)})
            return (
                self.build_result(
                    scenario,
                    variant="one-invalid-item",
                    projection=AugmentationProjection(
                        reads=(
                            _target(
                                "source-batch-item",
                                "structured_input",
                                f"/actions/{action_index}/batch_items/{len(items) - 1}/validity",
                            ),
                        ),
                        writes=(
                            _target(
                                "invalid-batch-item",
                                "structured_input",
                                f"/actions/{action_index}/batch_items/{len(items) - 1}/validity",
                            ),
                        ),
                    ),
                    actions=tuple(actions),
                ),
            )
        return ()


def _write_applicability(scenario: Scenario) -> Applicability:
    if _first_action(scenario, ActionEffect.WRITE) is None:
        return Applicability(applicable=False, reasons=("scenario has no write action",))
    return Applicability(applicable=True)


def _timeout_applicability(
    scenario: Scenario, timing: EventTiming, commit_state: str
) -> Applicability:
    action = _first_action(scenario, ActionEffect.WRITE)
    if action is None:
        return _write_applicability(scenario)
    if _has_action_event(
        scenario,
        action,
        "timeout",
        timing,
        expected_payload={"commit_state": commit_state},
    ):
        return Applicability(
            applicable=False,
            reasons=(f"equivalent {commit_state} timeout already exists",),
        )
    return Applicability(applicable=True)


def _has_action_event(
    scenario: Scenario,
    action: Action,
    kind: str,
    timing: EventTiming,
    *,
    expected_payload: dict[str, JsonValue] | None = None,
) -> bool:
    expected_payload = expected_payload or {}
    return any(
        event.kind == kind
        and event.timing == timing
        and event.action_id == action.id
        and all(event.payload.get(key) == value for key, value in expected_payload.items())
        for event in scenario.environment_events
    )


def _first_action(scenario: Scenario, effect: ActionEffect) -> Action | None:
    return next((action for action in scenario.actions if action.effect == effect), None)


def _read_write_pair(scenario: Scenario) -> tuple[Action, Action] | None:
    for read_index, read_action in enumerate(scenario.actions):
        if read_action.effect != ActionEffect.READ:
            continue
        for write_action in scenario.actions[read_index + 1 :]:
            if write_action.effect == ActionEffect.WRITE:
                return read_action, write_action
    return None


def _first_mutable_parameter(scenario: Scenario) -> tuple[int, str, str | int | float] | None:
    for action_index, action in enumerate(scenario.actions):
        for name, value in action.parameters.items():
            is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
            if isinstance(value, str) or is_number:
                return action_index, name, cast(str | int | float, value)
    return None


def _correction_target(
    scenario: Scenario,
) -> tuple[int, str, JsonValue, str] | None:
    hint = _augmentation_hints(scenario).later_correction
    if hint is not None:
        action_index = next(
            index for index, action in enumerate(scenario.actions) if action.id == hint.action_id
        )
        corrected_value_text = str(hint.corrected_value)
        message = hint.message
        if corrected_value_text not in message:
            message = f"{message} Updated {hint.parameter}: {corrected_value_text}."
        return action_index, hint.parameter, hint.corrected_value, message
    fallback = _first_mutable_parameter(scenario)
    if fallback is None:
        return None
    action_index, parameter_name, original_value = fallback
    return (
        action_index,
        parameter_name,
        _corrected_value(original_value),
        f"Correction: use {_corrected_value(original_value)} for {parameter_name}.",
    )


def _corrected_value(value: str | int | float) -> str | int | float:
    if isinstance(value, str):
        return f"{value} (corrected)"
    return value + 1


def _make_reference_ambiguous(
    turn: ConversationTurn,
    source: Artifact,
    alternative: Artifact,
    replacement_text: str | None,
) -> ConversationTurn:
    replacement = replacement_text or source.label or f"the {source.kind}"
    return turn.model_copy(
        update={
            "content": turn.content.replace(source.id, replacement),
            "metadata": {
                **turn.metadata,
                "ambiguous_reference_ids": [source.id, alternative.id],
            },
        }
    )


def _augmentation_hints(scenario: Scenario) -> AugmentationHints:
    raw_hints = scenario.metadata.get("augmentation_hints")
    if raw_hints is None:
        return AugmentationHints()
    return AugmentationHints.model_validate(raw_hints)


def _metadata_without_hint(scenario: Scenario, hint_name: str) -> dict[str, JsonValue]:
    metadata = dict(scenario.metadata)
    remaining_hints = _augmentation_hints(scenario).model_dump(mode="json", exclude_none=True)
    remaining_hints.pop(hint_name, None)
    if remaining_hints:
        metadata["augmentation_hints"] = remaining_hints
    else:
        metadata.pop("augmentation_hints", None)
    return metadata


def _hint_value[T](hint_value: T | None, fallback: T | None) -> T | None:
    return fallback if hint_value is None else hint_value


def _first_boundary(scenario: Scenario) -> tuple[int, str, int | float, int | float] | None:
    action_indices = {action.id: index for index, action in enumerate(scenario.actions)}
    for policy in scenario.policies:
        for boundary in policy.boundaries:
            action_index = action_indices.get(boundary.action_id)
            if (
                action_index is not None
                and boundary.parameter in scenario.actions[action_index].parameters
            ):
                return action_index, boundary.parameter, boundary.threshold, boundary.increment
    return None


def _action_event(
    scenario: Scenario,
    action: Action,
    *,
    kind: str,
    timing: EventTiming,
    payload: dict[str, JsonValue],
) -> EnvironmentEvent:
    return EnvironmentEvent(
        id=_unique_id(kind, {event.id for event in scenario.environment_events}),
        kind=kind,
        timing=timing,
        action_id=action.id,
        target_ids=(*action.artifact_ids, *action.resource_ids),
        payload=payload,
    )


def _event_result(
    augmentation: BuiltinAugmentation, scenario: Scenario, event: EnvironmentEvent
) -> AugmentationResult:
    return augmentation.build_result(
        scenario,
        variant=event.kind,
        projection=AugmentationProjection(
            reads=(
                _target(
                    "source-action",
                    "tool" if augmentation.metadata.id.startswith("environment.tool.") else "state",
                    "/actions",
                ),
            ),
            writes=(
                _target(
                    "scheduled-event",
                    "environment",
                    "/environment_events",
                    event_id=event.id,
                ),
            ),
        ),
        environment_events=(*scenario.environment_events, event),
    )


def _target(
    identifier: str,
    surface: AugmentationTargetSurface,
    path: str,
    *,
    event_id: str | None = None,
) -> ProjectionTarget:
    return ProjectionTarget.model_validate(
        {"id": identifier, "surface": surface, "path": path, "event_id": event_id}
    )


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _unique_id(preferred: str, existing: set[str]) -> str:
    if preferred not in existing:
        return preferred
    counter = 2
    while f"{preferred}-{counter}" in existing:
        counter += 1
    return f"{preferred}-{counter}"


BUILTIN_AUGMENTATIONS = (
    AmbiguityAugmentation(),
    LaterCorrectionAugmentation(),
    BoundaryShiftAugmentation(),
    ExistingPartialOperationAugmentation(),
    StateChangeBetweenReadWriteAugmentation(),
    StaleObservationAugmentation(),
    TimeoutBeforeCommitAugmentation(),
    TimeoutAfterCommitAugmentation(),
    MixedValidityBatchAugmentation(),
)
