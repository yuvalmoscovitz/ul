from __future__ import annotations

import pytest
from pydantic import ValidationError
from ul_core.evaluation import (
    EnvironmentCapabilities,
    EnvironmentLifecycleEvidence,
    EnvironmentResetEvidence,
    EnvironmentStateEvidence,
    EnvironmentTurnEvidence,
    EvaluationCase,
    ExecutionEvidence,
    ProductionObservation,
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
            max_environment_api_calls=10,
            timeout_seconds=30,
        )


def test_evaluation_case_rejects_production_routing_metadata() -> None:
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(
            {
                "id": "case-1",
                "turns": [_turn().model_dump(mode="json")],
                "max_environment_api_calls": 5,
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
            max_environment_api_calls=10,
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
            environment_id="invoice-environment",
            environment_config_sha256="a" * 64,
            initial_state=EnvironmentStateEvidence(
                value={"payments": []}, authority="environment_self_reported"
            ),
            turns=(
                EnvironmentTurnEvidence(
                    turn_id="turn-1",
                    response={"status": "ok"},
                    state_snapshot={"payments": ["payment-1"]},
                    state_observation_authority="environment_self_reported",
                ),
            ),
            final_response={"status": "ok"},
            final_state=EnvironmentStateEvidence(
                value={"payments": ["payment-1"]}, authority="environment_self_reported"
            ),
            timeout_after_commit_event=TimeoutAfterCommitEventEvidence(
                event_id="lost-ack",
                turn_id="turn-1",
                action_id="execute-payment",
                armed=True,
                trigger_status="fired",
                cleaned=False,
            ),
            lifecycle=EnvironmentLifecycleEvidence(
                initial_reset=EnvironmentResetEvidence(
                    reset_session_requested=True,
                    reset_session_acknowledged=True,
                    reset_env_requested=True,
                    reset_env_acknowledged=True,
                ),
                cleanup_reset=EnvironmentResetEvidence(
                    reset_session_requested=True,
                    reset_session_acknowledged=True,
                    reset_env_requested=True,
                    reset_env_acknowledged=True,
                ),
                terminal_status="succeeded",
                delivery="certain",
                cleanup="succeeded",
                environment_state_uncertain=False,
            ),
        )


def test_state_evidence_requires_explicit_authority() -> None:
    with pytest.raises(ValidationError):
        EnvironmentTurnEvidence(
            turn_id="turn-1",
            response={"status": "ok"},
            state_snapshot={"payments": []},
        )


def test_independent_state_evidence_requires_observer_identity() -> None:
    with pytest.raises(ValidationError):
        EnvironmentTurnEvidence(
            turn_id="turn-1",
            response={"status": "ok"},
            state_snapshot={"payments": []},
            state_observation_authority="independent_observer",
        )


def test_reset_receipt_requires_explicit_factual_acknowledgements() -> None:
    with pytest.raises(ValidationError):
        EnvironmentResetEvidence.model_validate(
            {"reset_session_requested": True, "reset_env_requested": True}
        )


def test_uncertain_delivery_requires_uncertain_environment_state() -> None:
    with pytest.raises(ValidationError):
        EnvironmentLifecycleEvidence(
            initial_reset=EnvironmentResetEvidence(
                reset_session_requested=True,
                reset_session_acknowledged=True,
                reset_env_requested=True,
                reset_env_acknowledged=True,
            ),
            cleanup_reset=EnvironmentResetEvidence(
                reset_session_requested=True,
                reset_session_acknowledged=True,
                reset_env_requested=True,
                reset_env_acknowledged=True,
            ),
            terminal_status="failed",
            failed_phase="execute_turn",
            failure_code="response_timeout",
            failure_reason="environment API response timed out",
            delivery="uncertain",
            cleanup="succeeded",
            environment_state_uncertain=False,
        )


def test_cleanup_failure_requires_safe_detail() -> None:
    with pytest.raises(ValidationError, match="cleanup failure detail"):
        EnvironmentLifecycleEvidence(
            initial_reset=EnvironmentResetEvidence(
                reset_session_requested=True,
                reset_session_acknowledged=True,
                reset_env_requested=True,
                reset_env_acknowledged=True,
            ),
            cleanup_reset=EnvironmentResetEvidence(
                reset_session_requested=True,
                reset_session_acknowledged=True,
                reset_env_requested=True,
                reset_env_acknowledged=True,
            ),
            terminal_status="failed",
            failed_phase="cleanup_reset",
            failure_code="reset_not_clean",
            failure_reason="environment API reset did not report clean state",
            delivery="certain",
            cleanup="failed",
            environment_state_uncertain=True,
        )


def test_failure_reason_requires_stable_code() -> None:
    with pytest.raises(ValidationError, match="failure code and reason"):
        EnvironmentLifecycleEvidence(
            initial_reset=EnvironmentResetEvidence(
                reset_session_requested=True,
                reset_session_acknowledged=True,
                reset_env_requested=True,
                reset_env_acknowledged=True,
            ),
            cleanup_reset=EnvironmentResetEvidence(
                reset_session_requested=True,
                reset_session_acknowledged=True,
                reset_env_requested=True,
                reset_env_acknowledged=True,
            ),
            terminal_status="failed",
            failed_phase="execute_turn",
            failure_reason="environment lifecycle failed",
            delivery="certain",
            cleanup="succeeded",
            environment_state_uncertain=False,
        )


def test_cleanup_failure_reason_requires_stable_code() -> None:
    with pytest.raises(ValidationError, match="cleanup failure code and reason"):
        EnvironmentLifecycleEvidence(
            initial_reset=EnvironmentResetEvidence(
                reset_session_requested=True,
                reset_session_acknowledged=True,
                reset_env_requested=True,
                reset_env_acknowledged=True,
            ),
            cleanup_reset=EnvironmentResetEvidence(
                reset_session_requested=True,
                reset_session_acknowledged=True,
                reset_env_requested=True,
                reset_env_acknowledged=True,
            ),
            terminal_status="failed",
            failed_phase="cleanup_reset",
            failure_code="reset_not_clean",
            failure_reason="environment API reset did not report clean state",
            delivery="certain",
            cleanup="failed",
            cleanup_failure_reason="environment API reset did not report clean state",
            environment_state_uncertain=True,
        )


def test_successful_execution_evidence_is_explicit() -> None:
    evidence = ExecutionEvidence(
        case_id="case-1",
        environment_id="invoice-environment",
        environment_config_sha256="a" * 64,
        initial_state=EnvironmentStateEvidence(
            value={"clean": True}, authority="environment_self_reported"
        ),
        turns=(
            EnvironmentTurnEvidence(
                turn_id="turn-1",
                response={"status": "ok"},
                state_snapshot={"payments": []},
                state_observation_authority="environment_self_reported",
            ),
        ),
        final_response={"status": "ok"},
        final_state=EnvironmentStateEvidence(
            value={"payments": []}, authority="environment_self_reported"
        ),
        lifecycle=EnvironmentLifecycleEvidence(
            initial_reset=EnvironmentResetEvidence(
                reset_session_requested=True,
                reset_session_acknowledged=True,
                reset_env_requested=True,
                reset_env_acknowledged=True,
            ),
            cleanup_reset=EnvironmentResetEvidence(
                reset_session_requested=True,
                reset_session_acknowledged=True,
                reset_env_requested=True,
                reset_env_acknowledged=True,
            ),
            terminal_status="succeeded",
            completed_phases=("reset", "execute_turn", "snapshot", "cleanup_reset"),
            delivery="certain",
            cleanup="succeeded",
            environment_state_uncertain=False,
        ),
    )

    assert evidence.turns[0].state_observation_authority == "environment_self_reported"


def test_successful_response_only_evidence_has_no_reset_or_cleanup_claims() -> None:
    evidence = ExecutionEvidence(
        evidence_scope="response_only",
        case_id="case-1",
        environment_id="response-only-environment",
        environment_config_sha256="a" * 64,
        turns=(EnvironmentTurnEvidence(turn_id="turn-1", response={"status": "ok"}),),
        final_response={"status": "ok"},
        lifecycle=EnvironmentLifecycleEvidence(
            terminal_status="succeeded",
            completed_phases=("execute_turn",),
            delivery="certain",
            cleanup="not_attempted",
            environment_state_uncertain=False,
        ),
    )

    assert evidence.lifecycle.initial_reset is None
    assert evidence.lifecycle.cleanup_reset is None


def test_successful_response_and_state_evidence_still_requires_cleanup() -> None:
    with pytest.raises(ValidationError, match="requires cleanup"):
        ExecutionEvidence(
            case_id="case-1",
            environment_id="stateful-environment",
            environment_config_sha256="a" * 64,
            initial_state=EnvironmentStateEvidence(
                value={"clean": True}, authority="environment_self_reported"
            ),
            turns=(
                EnvironmentTurnEvidence(
                    turn_id="turn-1",
                    response={"status": "ok"},
                    state_snapshot={"payments": []},
                    state_observation_authority="environment_self_reported",
                ),
            ),
            final_response={"status": "ok"},
            final_state=EnvironmentStateEvidence(
                value={"payments": []}, authority="environment_self_reported"
            ),
            lifecycle=EnvironmentLifecycleEvidence(
                initial_reset=EnvironmentResetEvidence(
                    reset_session_requested=True,
                    reset_session_acknowledged=True,
                    reset_env_requested=True,
                    reset_env_acknowledged=True,
                ),
                terminal_status="succeeded",
                completed_phases=("reset", "execute_turn", "snapshot"),
                delivery="certain",
                cleanup="not_attempted",
                environment_state_uncertain=False,
            ),
        )


def test_environment_capabilities_bind_state_authority() -> None:
    with pytest.raises(ValidationError):
        EnvironmentCapabilities(
            supports_conversations=True,
            supports_state_observation=True,
            cancellation_guarantee="best_effort",
        )
