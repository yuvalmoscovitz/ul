from __future__ import annotations

from collections.abc import Iterable

from ul_core.contracts import EnvironmentExecutor
from ul_core.dataset import ObservedAgentOutput
from ul_core.evaluation import EvaluationCase, ExecutionEvidence, StateObservationAuthority
from ul_core.models import ConversationRole, ConversationTurn


def evaluation_case_from_inputs(
    *,
    case_id: str,
    raw_inputs: Iterable[str],
    max_environment_api_calls: int,
    timeout_seconds: float,
    required_state_observation_authority: StateObservationAuthority | None = None,
    required_state_observer_id: str | None = None,
) -> EvaluationCase:
    return EvaluationCase(
        id=case_id,
        turns=tuple(
            ConversationTurn(
                id=f"{case_id}:turn-{index}",
                role=ConversationRole.USER,
                content=raw_input,
            )
            for index, raw_input in enumerate(raw_inputs, start=1)
        ),
        max_environment_api_calls=max_environment_api_calls,
        timeout_seconds=timeout_seconds,
        required_state_observation_authority=required_state_observation_authority,
        required_state_observer_id=required_state_observer_id,
    )


def validate_execution_evidence(
    case: EvaluationCase,
    environment: EnvironmentExecutor,
    evidence: ExecutionEvidence,
) -> None:
    if evidence.case_id != case.id:
        raise ValueError("environment evidence does not match the requested case")
    if evidence.environment_id != environment.environment_id:
        raise ValueError("environment evidence identity does not match the connection")
    if evidence.environment_config_sha256 != environment.config_sha256:
        raise ValueError("environment evidence config does not match the connection")
    requested_event = case.timeout_after_commit_event
    event_evidence = evidence.timeout_after_commit_event
    if (requested_event is None) != (event_evidence is None):
        raise ValueError("environment event evidence does not match the requested case")
    if requested_event is not None and event_evidence is not None:
        if (
            event_evidence.operator_id,
            event_evidence.operator_version,
            event_evidence.event_id,
            event_evidence.turn_id,
            event_evidence.action_id,
        ) != (
            requested_event.operator_id,
            requested_event.operator_version,
            requested_event.event_id,
            requested_event.turn_id,
            requested_event.action_id,
        ):
            raise ValueError("environment event evidence does not match the requested event")
        if (
            environment.capabilities.timeout_after_commit_version
            != requested_event.operator_version
        ):
            raise ValueError("environment event evidence does not match its advertised capability")
        if evidence.lifecycle.terminal_status == "succeeded" and (
            not event_evidence.armed
            or event_evidence.trigger_status == "unknown"
            or not event_evidence.cleaned
        ):
            raise ValueError(
                "successful environment evidence contains an incomplete event lifecycle"
            )
    expected_turn_ids = tuple(turn.id for turn in case.turns)
    evidence_turn_ids = tuple(turn.turn_id for turn in evidence.turns)
    if evidence.lifecycle.terminal_status == "succeeded" and evidence_turn_ids != expected_turn_ids:
        raise ValueError("environment evidence turns do not match the requested case")
    if any(turn_id not in expected_turn_ids for turn_id in evidence_turn_ids):
        raise ValueError("environment evidence contains a turn outside the requested case")
    if case.required_state_observation_authority is not None and (
        evidence.lifecycle.terminal_status == "succeeded"
        and any(turn.state_snapshot is None for turn in evidence.turns)
    ):
        raise ValueError("environment evidence omitted a required state observation")
    if (
        case.required_state_observation_authority is not None
        and environment.capabilities.state_observation_authority
        != case.required_state_observation_authority
    ):
        raise ValueError("environment state authority does not match the evaluation case")
    if (
        case.required_state_observer_id is not None
        and environment.capabilities.state_observer_id != case.required_state_observer_id
    ):
        raise ValueError("environment state observer does not match the evaluation case")
    for state_evidence in (evidence.initial_state, evidence.final_state):
        if state_evidence is not None and (
            state_evidence.authority != environment.capabilities.state_observation_authority
            or state_evidence.observer_id != environment.capabilities.state_observer_id
        ):
            raise ValueError("environment state evidence authority does not match its capabilities")
    for turn in evidence.turns:
        if turn.state_snapshot is not None and (
            turn.state_observation_authority != environment.capabilities.state_observation_authority
            or turn.state_observer_id != environment.capabilities.state_observer_id
        ):
            raise ValueError("environment state evidence authority does not match its capabilities")


def observed_outputs_from_evidence(
    evidence: ExecutionEvidence,
) -> tuple[ObservedAgentOutput, ...]:
    outputs: list[ObservedAgentOutput] = []
    before_turn_state_present = evidence.initial_state is not None
    before_turn_state = evidence.initial_state.value if evidence.initial_state is not None else None
    for turn in evidence.turns:
        outputs.append(
            ObservedAgentOutput(
                raw_output=turn.response,
                metadata={
                    **(
                        {"committed_state_snapshot": turn.state_snapshot}
                        if turn.state_snapshot is not None
                        else {}
                    ),
                    **(
                        {"state_observation_authority": turn.state_observation_authority}
                        if turn.state_observation_authority is not None
                        else {}
                    ),
                    **(
                        {"committed_state_before_turn": before_turn_state}
                        if before_turn_state_present
                        else {}
                    ),
                },
            )
        )
        before_turn_state_present = turn.state_snapshot is not None
        before_turn_state = turn.state_snapshot
    return tuple(outputs)


def execution_evidence_requires_quarantine(evidence: ExecutionEvidence) -> bool:
    lifecycle = evidence.lifecycle
    return (
        lifecycle.delivery == "uncertain"
        or lifecycle.cleanup == "failed"
        or lifecycle.environment_state_uncertain
        or (
            evidence.timeout_after_commit_event is not None
            and evidence.timeout_after_commit_event.armed
            and not evidence.timeout_after_commit_event.cleaned
        )
    )
