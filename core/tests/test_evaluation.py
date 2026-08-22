from __future__ import annotations

import pytest
import ul_core
from pydantic import ValidationError
from ul_core.evaluation import (
    EnvironmentCapabilities,
    EnvironmentLifecycleEvidence,
    EnvironmentResetEvidence,
    EnvironmentStateEvidence,
    EnvironmentTurnEvidence,
    EvaluationCase,
    ExecutionEvidence,
    ObservationSourceCapabilities,
    ProbeCapabilities,
    ProbeExecutionEvent,
    ProbeInvokerCapabilities,
    ProbeObservation,
    ProbeResult,
    ProductionObservation,
    StateEnvironmentCapabilities,
    StateOperationResult,
    TimeoutAfterCommitEventEvidence,
    TimeoutAfterCommitEventRequest,
    evidence_profile_from_capabilities,
)
from ul_core.evaluators import (
    HttpResultEvaluator,
    JsonPropertyEvaluator,
    RubricEvaluator,
    StateChangeEvaluator,
    ToolCallEvaluator,
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


def test_probe_capabilities_default_to_bounded_evidence_payloads() -> None:
    invoker = ProbeInvokerCapabilities(
        invoker_id="invoker-1",
        response_size_limit_bytes=1_000,
    )
    state_environment = StateEnvironmentCapabilities(environment_id="environment-1")

    assert invoker.execution_events_size_limit_bytes == 1_000_000
    assert state_environment.snapshot_size_limit_bytes == 1_000_000


def test_probe_capabilities_reject_non_positive_evidence_payload_limits() -> None:
    with pytest.raises(ValidationError):
        ProbeInvokerCapabilities(
            invoker_id="invoker-1",
            response_size_limit_bytes=1_000,
            execution_events_size_limit_bytes=0,
        )
    with pytest.raises(ValidationError):
        StateEnvironmentCapabilities(
            environment_id="environment-1",
            snapshot_size_limit_bytes=0,
        )


def test_evaluation_case_accepts_multiple_customer_evaluators() -> None:
    evaluation_case = EvaluationCase(
        id="case-1",
        turns=(_turn(),),
        max_environment_api_calls=10,
        timeout_seconds=30,
        evaluators=(
            ToolCallEvaluator(
                id="payment-call",
                tool_name="execute_payment",
                arguments={"invoice_id": "INV-42"},
            ),
            RubricEvaluator(
                id="answer-quality", rubric="The answer must clearly describe the outcome."
            ),
        ),
    )

    assert EvaluationCase.model_validate_json(evaluation_case.model_dump_json()) == evaluation_case


def test_evaluation_case_rejects_duplicate_evaluator_identifiers() -> None:
    with pytest.raises(ValidationError, match="evaluator identifiers must be unique"):
        EvaluationCase(
            id="case-1",
            turns=(_turn(),),
            max_environment_api_calls=10,
            timeout_seconds=30,
            evaluators=(
                ToolCallEvaluator(id="duplicate", tool_name="first"),
                ToolCallEvaluator(id="duplicate", tool_name="second"),
            ),
        )


def test_evaluators_reject_parameters_the_selected_operator_would_ignore() -> None:
    with pytest.raises(ValidationError, match="existence checks"):
        JsonPropertyEvaluator(
            id="exists",
            source="answer",
            json_pointer="/value",
            operator="exists",
            expected=True,
        )
    with pytest.raises(ValidationError, match="existence checks"):
        JsonPropertyEvaluator(
            id="not-exists",
            source="answer",
            json_pointer="/value",
            operator="not_exists",
            expected=False,
        )
    with pytest.raises(ValidationError, match="not valid for type checks"):
        JsonPropertyEvaluator(
            id="type",
            source="answer",
            json_pointer="/value",
            operator="type",
            expected=True,
            expected_type="boolean",
        )
    with pytest.raises(ValidationError, match="require expected"):
        JsonPropertyEvaluator(
            id="equals",
            source="answer",
            json_pointer="/value",
            operator="equals",
        )
    with pytest.raises(ValidationError, match="do not accept expected"):
        StateChangeEvaluator(id="changed", operator="changed", expected="ignored")
    with pytest.raises(ValidationError, match="do not accept expected"):
        StateChangeEvaluator(id="unchanged", operator="unchanged", expected="ignored")
    with pytest.raises(ValidationError, match="require expected"):
        StateChangeEvaluator(id="equals", operator="equals")
    with pytest.raises(ValidationError, match="provided together"):
        HttpResultEvaluator(id="http", status_code=200, expected_body_value=True)
    with pytest.raises(ValidationError, match="provided together"):
        HttpResultEvaluator(id="http", body_json_pointer="/ok")


def test_operator_specific_evaluators_round_trip_without_ignored_default_fields() -> None:
    evaluators = (
        JsonPropertyEvaluator(
            id="exists", source="answer", json_pointer="/value", operator="exists"
        ),
        StateChangeEvaluator(id="changed", operator="changed"),
        HttpResultEvaluator(id="http", status_code=202),
    )

    for evaluator in evaluators:
        assert type(evaluator).model_validate_json(evaluator.model_dump_json()) == evaluator


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


@pytest.mark.parametrize(
    ("evidence_updates", "lifecycle_updates", "message"),
    (
        (
            {
                "initial_state": EnvironmentStateEvidence(
                    value={}, authority="environment_self_reported"
                )
            },
            {},
            "state observations",
        ),
        (
            {},
            {
                "initial_reset": EnvironmentResetEvidence(
                    reset_session_requested=True,
                    reset_session_acknowledged=True,
                    reset_env_requested=True,
                    reset_env_acknowledged=True,
                )
            },
            "reset receipts",
        ),
        ({}, {"completed_phases": ("reset",)}, "stateful phases"),
        ({}, {"failed_phase": "snapshot"}, "stateful phase"),
    ),
)
def test_failed_response_only_evidence_rejects_stateful_claims(
    evidence_updates: dict[str, object],
    lifecycle_updates: dict[str, object],
    message: str,
) -> None:
    lifecycle = EnvironmentLifecycleEvidence(
        terminal_status="failed",
        failed_phase="execute_turn",
        failure_code="response_timeout",
        failure_reason="environment API response timed out",
        delivery="uncertain",
        cleanup="not_attempted",
        environment_state_uncertain=True,
    ).model_copy(update=lifecycle_updates)

    with pytest.raises(ValidationError, match=message):
        ExecutionEvidence.model_validate(
            {
                "evidence_scope": "response_only",
                "case_id": "case-1",
                "environment_id": "response-only-environment",
                "environment_config_sha256": "a" * 64,
                "lifecycle": lifecycle,
                **evidence_updates,
            }
        )


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


def test_probe_result_rejects_mismatched_event_correlation() -> None:
    with pytest.raises(ValidationError, match="correlation"):
        ProbeResult(
            id="result-1",
            correlation_id="correlation-1",
            response={"answer": "done"},
            response_size_bytes=17,
            execution_events=(
                ProbeExecutionEvent(
                    id="event-1",
                    correlation_id="different-correlation",
                    kind="tool_call",
                    payload={"tool": "pay_invoice"},
                ),
            ),
        )


def test_probe_result_does_not_require_adapter_to_reproduce_json_byte_encoding() -> None:
    result = ProbeResult(
        id="result-1",
        correlation_id="correlation-1",
        response={"message": "héllo"},
    )

    assert result.response_size_bytes is None


def test_unavailable_observation_requires_a_safe_limitation() -> None:
    with pytest.raises(ValidationError, match="require a limitation"):
        ProbeObservation(
            id="observation-1",
            source_id="otel-collector",
            correlation_id="correlation-1",
            authority="independent_observer",
            status="missing",
        )


def test_missing_observation_rejects_claimed_evidence() -> None:
    with pytest.raises(ValidationError, match="cannot contain observed evidence"):
        ProbeObservation(
            id="observation-1",
            source_id="otel-collector",
            correlation_id="correlation-1",
            authority="independent_observer",
            status="missing",
            limitation="trace did not arrive",
            traces=({"span": "stale"},),
        )


def test_state_capabilities_reject_incomplete_deterministic_replay() -> None:
    with pytest.raises(ValidationError, match="deterministic replay requires"):
        StateEnvironmentCapabilities(
            environment_id="invoice-sandbox",
            supports_reset=True,
            supports_deterministic_replay=True,
        )


def test_state_operation_preserves_reset_acknowledgements_and_uncertainty() -> None:
    result = StateOperationResult(
        id="cleanup-1",
        fixture_id="fixture-1",
        correlation_id="correlation-1",
        operation="cleanup",
        succeeded=False,
        reset_session_requested=True,
        reset_session_acknowledged=True,
        reset_environment_requested=True,
        reset_environment_acknowledged=False,
        state_uncertain=True,
        failure_code="reset_not_clean",
        failure_reason="environment reset was not acknowledged",
    )

    assert result.reset_session_acknowledged is True
    assert result.reset_environment_acknowledged is False
    assert result.state_uncertain is True


def test_evidence_profile_is_non_ordinal_and_preserves_provenance() -> None:
    profile = evidence_profile_from_capabilities(
        ProbeCapabilities(
            invoker=ProbeInvokerCapabilities(
                invoker_id="local-python",
                response_size_limit_bytes=10_000,
                supports_structured_execution_events=True,
            ),
            observation_source=ObservationSourceCapabilities(
                source_id="otel-collector",
                authority="independent_observer",
                supports_traces=True,
                supports_tool_calls=True,
            ),
            state_environment=StateEnvironmentCapabilities(
                environment_id="invoice-sandbox",
                supports_reset=True,
                supports_snapshot=True,
                supports_cleanup=True,
                state_observation_authority="environment_self_reported",
                supports_deterministic_replay=True,
            ),
        )
    )

    assert profile.basis == "declared_capabilities"
    assert profile.available_facts == frozenset(
        {
            "response_observed",
            "trajectory_observed",
            "committed_state_verified",
            "deterministic_replay_verified",
        }
    )
    assert profile.sources["trajectory_observed"] == "otel-collector"
    assert profile.authorities["trajectory_observed"] == "independent_observer"
    assert profile.sources["committed_state_verified"] == "invoice-sandbox"
    assert "level" not in type(profile).model_fields


def test_deterministic_replay_preserves_independent_state_authority() -> None:
    profile = evidence_profile_from_capabilities(
        ProbeCapabilities(
            invoker=ProbeInvokerCapabilities(
                invoker_id="local-python",
                response_size_limit_bytes=10_000,
            ),
            state_environment=StateEnvironmentCapabilities(
                environment_id="database-observer",
                supports_reset=True,
                supports_snapshot=True,
                supports_cleanup=True,
                state_observation_authority="independent_observer",
                state_observer_id="database-audit-log",
                supports_deterministic_replay=True,
            ),
        )
    )

    assert profile.authorities["deterministic_replay_verified"] == "independent_observer"


def test_probe_contracts_are_public_core_api() -> None:
    assert ul_core.ProbeInvoker is not None
    assert ul_core.ObservationSource is not None
    assert ul_core.StateEnvironment is not None
    assert ul_core.ProbeRequest is not None
    assert ul_core.ProbeResult is not None
    assert ul_core.EvidenceProfile is not None
