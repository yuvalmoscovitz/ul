from __future__ import annotations

from typing import Literal

import pytest
from ul.sandbox import (
    execution_evidence_requires_quarantine,
    observed_outputs_from_evidence,
    validate_execution_evidence,
)
from ul_core.evaluation import (
    EvaluationCase,
    ExecutionEvidence,
    SandboxCapabilities,
    SandboxLifecycleEvidence,
    SandboxStateEvidence,
    SandboxTurnEvidence,
    TimeoutAfterCommitEventEvidence,
    TimeoutAfterCommitEventRequest,
)
from ul_core.models import ConversationRole, ConversationTurn


class _SandboxIdentity:
    sandbox_id = "invoice-sandbox"
    config_sha256 = "a" * 64
    capabilities = SandboxCapabilities(
        supports_conversations=True,
        supports_state_observation=True,
        state_observation_authority="sandbox_self_reported",
        cancellation_guarantee="best_effort",
    )

    def api_calls_for_case(self, case: EvaluationCase) -> int:
        return len(case.turns) + 4

    async def execute(self, case: EvaluationCase) -> ExecutionEvidence:
        raise AssertionError("identity-only test sandbox must not execute")


def _case() -> EvaluationCase:
    return EvaluationCase(
        id="case-1",
        turns=(
            ConversationTurn(id="turn-1", role=ConversationRole.USER, content="Pay invoice 42"),
        ),
        max_sandbox_api_calls=5,
        timeout_seconds=30,
    )


def _evidence(
    *,
    case_id: str = "case-1",
    sandbox_id: str = "invoice-sandbox",
    config_sha256: str = "a" * 64,
    turn_id: str = "turn-1",
    authority: Literal["sandbox_self_reported", "independent_observer"] = ("sandbox_self_reported"),
) -> ExecutionEvidence:
    return ExecutionEvidence(
        case_id=case_id,
        sandbox_id=sandbox_id,
        sandbox_config_sha256=config_sha256,
        initial_state=SandboxStateEvidence(
            value={"clean": True}, authority="sandbox_self_reported"
        ),
        turns=(
            SandboxTurnEvidence(
                turn_id=turn_id,
                response={"status": "ok"},
                state_snapshot={"payments": []},
                state_observation_authority=authority,
                state_observer_id=("observer-1" if authority == "independent_observer" else None),
            ),
        ),
        final_response={"status": "ok"},
        final_state=SandboxStateEvidence(
            value={"payments": []},
            authority=authority,
            observer_id=("observer-1" if authority == "independent_observer" else None),
        ),
        lifecycle=SandboxLifecycleEvidence(
            terminal_status="succeeded",
            completed_phases=("reset", "execute_turn", "snapshot", "cleanup_reset"),
            delivery="certain",
            cleanup="succeeded",
            sandbox_state_uncertain=False,
        ),
    )


@pytest.mark.parametrize(
    ("evidence", "message"),
    (
        (_evidence(case_id="other-case"), "requested case"),
        (_evidence(sandbox_id="other-sandbox"), "evidence identity"),
        (_evidence(config_sha256="b" * 64), "evidence config"),
        (_evidence(turn_id="other-turn"), "evidence turns"),
        (_evidence(authority="independent_observer"), "authority"),
    ),
)
def test_execution_evidence_is_bound_to_case_and_sandbox(
    evidence: ExecutionEvidence, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_execution_evidence(_case(), _SandboxIdentity(), evidence)


def test_required_state_observation_is_enforced() -> None:
    case = _case().model_copy(
        update={"required_state_observation_authority": "sandbox_self_reported"}
    )
    evidence = _evidence().model_copy(
        update={"turns": (SandboxTurnEvidence(turn_id="turn-1", response={"status": "ok"}),)}
    )

    with pytest.raises(ValueError, match="required state observation"):
        validate_execution_evidence(case, _SandboxIdentity(), evidence)


def test_observed_outputs_preserve_each_turns_before_and_after_state() -> None:
    first_turn = _evidence().turns[0]
    second_turn = SandboxTurnEvidence(
        turn_id="turn-2",
        response={"status": "retried"},
        state_snapshot={"payments": ["payment-1", "payment-2"]},
        state_observation_authority="sandbox_self_reported",
    )
    evidence = _evidence().model_copy(
        update={
            "turns": (first_turn, second_turn),
            "final_response": second_turn.response,
            "final_state": SandboxStateEvidence(
                value=second_turn.state_snapshot,
                authority="sandbox_self_reported",
            ),
        }
    )

    first_output, second_output = observed_outputs_from_evidence(evidence)

    assert first_output.metadata["committed_state_before_turn"] == {"clean": True}
    assert first_output.metadata["committed_state_snapshot"] == {"payments": []}
    assert second_output.metadata["committed_state_before_turn"] == {"payments": []}
    assert second_output.metadata["committed_state_snapshot"] == {
        "payments": ["payment-1", "payment-2"]
    }


def test_incomplete_timeout_event_is_rejected_and_quarantined() -> None:
    case = _case().model_copy(
        update={
            "timeout_after_commit_event": TimeoutAfterCommitEventRequest(
                event_id="lost-ack",
                turn_id="turn-1",
                action_id="execute-payment",
            )
        }
    )
    sandbox = _SandboxIdentity()
    sandbox.capabilities = sandbox.capabilities.model_copy(
        update={"timeout_after_commit_version": "1.0.0"}
    )
    incomplete_event = TimeoutAfterCommitEventEvidence.model_construct(
        event_id="lost-ack",
        turn_id="turn-1",
        action_id="execute-payment",
        armed=True,
        trigger_status="fired",
        cleaned=False,
    )
    evidence = _evidence().model_copy(update={"timeout_after_commit_event": incomplete_event})

    with pytest.raises(ValueError, match="incomplete event lifecycle"):
        validate_execution_evidence(case, sandbox, evidence)
    assert execution_evidence_requires_quarantine(evidence) is True


def test_every_independent_state_observation_is_bound_to_the_declared_observer() -> None:
    sandbox = _SandboxIdentity()
    sandbox.capabilities = SandboxCapabilities(
        supports_conversations=True,
        supports_state_observation=True,
        state_observation_authority="independent_observer",
        state_observer_id="observer-1",
        cancellation_guarantee="best_effort",
    )
    case = EvaluationCase(
        id="case-1",
        turns=(
            ConversationTurn(id="turn-1", role=ConversationRole.USER, content="Start"),
            ConversationTurn(id="turn-2", role=ConversationRole.USER, content="Correct it"),
        ),
        max_sandbox_api_calls=7,
        timeout_seconds=30,
        required_state_observation_authority="independent_observer",
        required_state_observer_id="observer-1",
    )
    evidence = ExecutionEvidence(
        case_id="case-1",
        sandbox_id="invoice-sandbox",
        sandbox_config_sha256="a" * 64,
        initial_state=SandboxStateEvidence(
            value={"payments": []},
            authority="independent_observer",
            observer_id="observer-1",
        ),
        turns=(
            SandboxTurnEvidence(
                turn_id="turn-1",
                response={"status": "started"},
                state_snapshot={"payments": []},
                state_observation_authority="independent_observer",
                state_observer_id="observer-2",
            ),
            SandboxTurnEvidence(
                turn_id="turn-2",
                response={"status": "corrected"},
                state_snapshot={"payments": ["42"]},
                state_observation_authority="independent_observer",
                state_observer_id="observer-1",
            ),
        ),
        final_response={"status": "corrected"},
        final_state=SandboxStateEvidence(
            value={"payments": ["42"]},
            authority="independent_observer",
            observer_id="observer-1",
        ),
        lifecycle=SandboxLifecycleEvidence(
            terminal_status="succeeded",
            completed_phases=(
                "reset",
                "execute_turn:1",
                "snapshot:1",
                "execute_turn:2",
                "snapshot:2",
                "cleanup_reset",
            ),
            delivery="certain",
            cleanup="succeeded",
            sandbox_state_uncertain=False,
        ),
    )

    with pytest.raises(ValueError, match="authority"):
        validate_execution_evidence(case, sandbox, evidence)
