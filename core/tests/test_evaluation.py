from __future__ import annotations

import pytest
from pydantic import ValidationError
from ul_core.evaluation import (
    EvaluationCase,
    ExecutionEvidence,
    ProductionObservation,
    SandboxCapabilities,
    SandboxLifecycleEvidence,
    SandboxStateEvidence,
    SandboxTurnEvidence,
    TimeoutAfterCommitEventEvidence,
    TimeoutAfterCommitEventRequest,
)
from ul_core.models import ConversationRole, ConversationTurn


def _turn() -> ConversationTurn:
    return ConversationTurn(id="turn-1", role=ConversationRole.USER, content="Pay invoice 42")


def test_production_observation_rejects_executable_configuration() -> None:
    with pytest.raises(ValidationError):
        ProductionObservation.model_validate(
            {
                "id": "observation-1",
                "source_kind": "otlp",
                "source_reference": "trace-1",
                "observed_input": "Pay invoice 42",
                "endpoint": "https://production.example/execute",
            }
        )


def test_evaluation_case_rejects_duplicate_turn_identifiers() -> None:
    with pytest.raises(ValidationError):
        EvaluationCase(
            id="case-1",
            turns=(_turn(), _turn()),
            max_sandbox_api_calls=10,
            timeout_seconds=30,
        )


def test_evaluation_case_rejects_production_routing_metadata() -> None:
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(
            {
                "id": "case-1",
                "turns": [_turn().model_dump(mode="json")],
                "max_sandbox_api_calls": 5,
                "timeout_seconds": 30,
                "production_endpoint": "https://production.example/execute",
                "production_headers": {"Authorization": "secret"},
            }
        )


def test_timeout_after_commit_event_must_reference_a_case_turn() -> None:
    with pytest.raises(ValidationError, match="must reference a case turn"):
        EvaluationCase(
            id="case-1",
            turns=(_turn(),),
            max_sandbox_api_calls=10,
            timeout_seconds=30,
            timeout_after_commit_event=TimeoutAfterCommitEventRequest(
                event_id="lost-ack",
                turn_id="other-turn",
                action_id="execute-payment",
            ),
        )


def test_unarmed_timeout_after_commit_event_cannot_claim_it_fired() -> None:
    with pytest.raises(ValidationError, match="unarmed"):
        TimeoutAfterCommitEventEvidence(
            event_id="lost-ack",
            turn_id="turn-1",
            action_id="execute-payment",
            armed=False,
            trigger_status="fired",
            cleaned=True,
        )


def test_successful_execution_requires_timeout_event_cleanup() -> None:
    with pytest.raises(ValidationError, match="completed event"):
        ExecutionEvidence(
            case_id="case-1",
            sandbox_id="invoice-sandbox",
            sandbox_config_sha256="a" * 64,
            initial_state=SandboxStateEvidence(
                value={"payments": []}, authority="sandbox_self_reported"
            ),
            turns=(
                SandboxTurnEvidence(
                    turn_id="turn-1",
                    response={"status": "ok"},
                    state_snapshot={"payments": ["payment-1"]},
                    state_observation_authority="sandbox_self_reported",
                ),
            ),
            final_response={"status": "ok"},
            final_state=SandboxStateEvidence(
                value={"payments": ["payment-1"]}, authority="sandbox_self_reported"
            ),
            timeout_after_commit_event=TimeoutAfterCommitEventEvidence(
                event_id="lost-ack",
                turn_id="turn-1",
                action_id="execute-payment",
                armed=True,
                trigger_status="fired",
                cleaned=False,
            ),
            lifecycle=SandboxLifecycleEvidence(
                terminal_status="succeeded",
                delivery="certain",
                cleanup="succeeded",
                sandbox_state_uncertain=False,
            ),
        )


def test_state_evidence_requires_explicit_authority() -> None:
    with pytest.raises(ValidationError):
        SandboxTurnEvidence(
            turn_id="turn-1",
            response={"status": "ok"},
            state_snapshot={"payments": []},
        )


def test_independent_state_evidence_requires_observer_identity() -> None:
    with pytest.raises(ValidationError):
        SandboxTurnEvidence(
            turn_id="turn-1",
            response={"status": "ok"},
            state_snapshot={"payments": []},
            state_observation_authority="independent_observer",
        )


def test_uncertain_delivery_requires_uncertain_sandbox_state() -> None:
    with pytest.raises(ValidationError):
        SandboxLifecycleEvidence(
            terminal_status="failed",
            failed_phase="execute_turn",
            failure_code="response_timeout",
            failure_reason="sandbox API response timed out",
            delivery="uncertain",
            cleanup="succeeded",
            sandbox_state_uncertain=False,
        )


def test_cleanup_failure_requires_safe_detail() -> None:
    with pytest.raises(ValidationError, match="cleanup failure detail"):
        SandboxLifecycleEvidence(
            terminal_status="failed",
            failed_phase="cleanup_reset",
            failure_code="reset_not_clean",
            failure_reason="sandbox API reset did not report clean state",
            delivery="certain",
            cleanup="failed",
            sandbox_state_uncertain=True,
        )


def test_failure_reason_requires_stable_code() -> None:
    with pytest.raises(ValidationError, match="failure code and reason"):
        SandboxLifecycleEvidence(
            terminal_status="failed",
            failed_phase="execute_turn",
            failure_reason="sandbox lifecycle failed",
            delivery="certain",
            cleanup="succeeded",
            sandbox_state_uncertain=False,
        )


def test_cleanup_failure_reason_requires_stable_code() -> None:
    with pytest.raises(ValidationError, match="cleanup failure code and reason"):
        SandboxLifecycleEvidence(
            terminal_status="failed",
            failed_phase="cleanup_reset",
            failure_code="reset_not_clean",
            failure_reason="sandbox API reset did not report clean state",
            delivery="certain",
            cleanup="failed",
            cleanup_failure_reason="sandbox API reset did not report clean state",
            sandbox_state_uncertain=True,
        )


def test_successful_execution_evidence_is_explicit() -> None:
    evidence = ExecutionEvidence(
        case_id="case-1",
        sandbox_id="invoice-sandbox",
        sandbox_config_sha256="a" * 64,
        initial_state=SandboxStateEvidence(
            value={"clean": True}, authority="sandbox_self_reported"
        ),
        turns=(
            SandboxTurnEvidence(
                turn_id="turn-1",
                response={"status": "ok"},
                state_snapshot={"payments": []},
                state_observation_authority="sandbox_self_reported",
            ),
        ),
        final_response={"status": "ok"},
        final_state=SandboxStateEvidence(value={"payments": []}, authority="sandbox_self_reported"),
        lifecycle=SandboxLifecycleEvidence(
            terminal_status="succeeded",
            completed_phases=("reset", "execute_turn", "snapshot", "cleanup_reset"),
            delivery="certain",
            cleanup="succeeded",
            sandbox_state_uncertain=False,
        ),
    )

    assert evidence.turns[0].state_observation_authority == "sandbox_self_reported"


def test_sandbox_capabilities_bind_state_authority() -> None:
    with pytest.raises(ValidationError):
        SandboxCapabilities(
            supports_conversations=True,
            supports_state_observation=True,
            cancellation_guarantee="best_effort",
        )
