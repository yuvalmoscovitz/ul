from typing import Literal

import pytest
from pydantic import ValidationError
from ul.augmentations.environment_fault import (
    TimeoutAfterCommitCase,
    TimeoutAfterCommitStressResult,
    TimeoutAfterCommitStressTrial,
)
from ul.dataset_invariants import (
    DatasetInvariantRuleResult,
    JsonValueEqualsLiteralInvariant,
    evaluate_dataset_invariant_rules,
)
from ul_core.dataset import ObservedAgentOutput
from ul_core.evaluation import (
    EnvironmentLifecycleEvidence,
    EnvironmentResetEvidence,
    EnvironmentStateEvidence,
    EnvironmentTurnEvidence,
    ExecutionEvidence,
    TimeoutAfterCommitEventEvidence,
)
from ul_core.models import ConversationRole, ConversationTurn


def _committed_count(count: int) -> ObservedAgentOutput:
    return ObservedAgentOutput(
        raw_output={},
        metadata={
            "committed_state_snapshot": {"matching_payment_count": count},
            "state_observation_authority": "environment_self_reported",
        },
    )


def _exactly_once_rule() -> JsonValueEqualsLiteralInvariant:
    return JsonValueEqualsLiteralInvariant(
        type="json_value_equals_literal",
        id="payment-committed-exactly-once",
        version="1.0.0",
        description="A payment must be committed exactly once.",
        severity="critical",
        value_pointer="/matching_payment_count",
        literal=1,
    )


def _result(
    *,
    status: Literal["passed", "failed", "inconclusive"],
    invariant_rules: tuple[DatasetInvariantRuleResult, ...],
    trials: tuple[TimeoutAfterCommitStressTrial, ...],
) -> TimeoutAfterCommitStressResult:
    return TimeoutAfterCommitStressResult(
        case=TimeoutAfterCommitCase(
            id="payment-timeout",
            event_id="lost-acknowledgement",
            action_id="execute-payment",
            turn=ConversationTurn(
                id="submit-payment",
                role=ConversationRole.USER,
                content="Submit the payment.",
            ),
        ),
        requested_repetitions=len(trials),
        target_calls_per_repetition=9,
        required_target_calls=9 * len(trials),
        status=status,
        trials=trials,
        invariant_rules=invariant_rules,
    )


def _successful_trial(repetition: int) -> TimeoutAfterCommitStressTrial:
    final_state = {"matching_payment_count": 2}
    return TimeoutAfterCommitStressTrial(
        repetition=repetition,
        execution_evidence=ExecutionEvidence(
            case_id=f"case-{repetition}",
            environment_id="payment-environment",
            environment_config_sha256="a" * 64,
            initial_state=EnvironmentStateEvidence(
                value={"matching_payment_count": 0}, authority="environment_self_reported"
            ),
            turns=(
                EnvironmentTurnEvidence(
                    turn_id="submit-payment",
                    response="Payment workflow completed.",
                    state_snapshot=final_state,
                    state_observation_authority="environment_self_reported",
                ),
            ),
            final_response="Payment workflow completed.",
            final_state=EnvironmentStateEvidence(
                value=final_state, authority="environment_self_reported"
            ),
            timeout_after_commit_event=TimeoutAfterCommitEventEvidence(
                event_id="lost-acknowledgement",
                turn_id="submit-payment",
                action_id="execute-payment",
                armed=True,
                trigger_status="fired",
                cleaned=True,
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
        ),
    )


def test_mixed_violation_is_inconclusive() -> None:
    invariant_rules = evaluate_dataset_invariant_rules(
        (_exactly_once_rule(),),
        (_committed_count(2), _committed_count(1)),
        observation_authority="committed_state_snapshot",
    )
    trials = tuple(_successful_trial(repetition) for repetition in (1, 2))

    assert invariant_rules[0].status == "violated"
    assert (
        _result(status="inconclusive", invariant_rules=invariant_rules, trials=trials).status
        == "inconclusive"
    )
    with pytest.raises(ValidationError, match="status must match"):
        _result(status="failed", invariant_rules=invariant_rules, trials=trials)


def test_lifecycle_ambiguity_takes_precedence_over_a_violation() -> None:
    invariant_rules = evaluate_dataset_invariant_rules(
        (_exactly_once_rule(),),
        (_committed_count(2), None),
        observation_authority="committed_state_snapshot",
    )
    trials = (
        _successful_trial(1),
        TimeoutAfterCommitStressTrial(
            repetition=2,
            inconclusive_reason="environment lifecycle failed",
        ),
    )

    assert invariant_rules[0].status == "violated"
    assert (
        _result(status="inconclusive", invariant_rules=invariant_rules, trials=trials).status
        == "inconclusive"
    )
    with pytest.raises(ValidationError, match="status must match"):
        _result(status="failed", invariant_rules=invariant_rules, trials=trials)


def test_result_rejects_truncated_invariant_repetition_evidence() -> None:
    invariant_rules = evaluate_dataset_invariant_rules(
        (_exactly_once_rule(),),
        (_committed_count(2), _committed_count(2)),
        observation_authority="committed_state_snapshot",
    )
    truncated_rule = invariant_rules[0].model_copy(update={"trials": invariant_rules[0].trials[:1]})

    with pytest.raises(ValidationError, match="cover every repetition"):
        _result(
            status="failed",
            invariant_rules=(truncated_rule,),
            trials=tuple(_successful_trial(repetition) for repetition in (1, 2)),
        )
