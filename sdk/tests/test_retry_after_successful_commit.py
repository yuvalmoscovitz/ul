from __future__ import annotations

from pathlib import Path

import pytest
from ul import (
    RetryAfterSuccessfulCommitCase,
    load_retry_after_successful_commit_case,
    plan_retry_after_successful_commit_stress_test,
    run_retry_after_successful_commit_stress_test,
)
from ul.dataset_invariants import (
    JsonArrayItemsUniqueByInvariant,
    JsonValueEqualsLiteralInvariant,
)
from ul.http_sandbox import JsonHttpSandboxConfig
from ul_core.dataset import ObservedAgentOutput
from ul_core.evaluation import (
    EvaluationCase,
    ExecutionEvidence,
    SandboxCapabilities,
    SandboxLifecycleEvidence,
    SandboxStateEvidence,
    SandboxTurnEvidence,
)
from ul_core.models import ConversationRole, ConversationTurn


def _case() -> RetryAfterSuccessfulCommitCase:
    return RetryAfterSuccessfulCommitCase(
        id="invoice-retry-after-commit",
        conversation=(
            ConversationTurn(
                id="initial-payment",
                role=ConversationRole.USER,
                content="Pay invoice AC-100.",
            ),
            ConversationTurn(
                id="explicit-retry",
                role=ConversationRole.USER,
                content="Retry the same payment for invoice AC-100.",
            ),
        ),
    )


def _config() -> JsonHttpSandboxConfig:
    return JsonHttpSandboxConfig.model_validate(
        {
            "version": 3,
            "sandbox_id": "retry-test-sandbox",
            "reset": {
                "url": "https://sandbox.example.test/reset",
                "generation_json_pointer": "/generation",
                "clean_state_json_pointer": "/clean",
                "clean_state_value": True,
            },
            "setup": {"url": "https://sandbox.example.test/setup"},
            "execute_turn": {
                "url": "https://sandbox.example.test/execute",
                "request_json_template": {
                    "case_id": "{{case_id}}",
                    "turn_id": "{{turn_id}}",
                    "input": "{{input}}",
                },
            },
            "snapshot": {
                "url": "https://sandbox.example.test/snapshot",
                "request_json_template": {
                    "case_id": "{{case_id}}",
                    "turn_id": "{{turn_id}}",
                },
            },
        }
    )


def _invariants() -> tuple[
    JsonValueEqualsLiteralInvariant,
    JsonArrayItemsUniqueByInvariant,
]:
    return (
        JsonValueEqualsLiteralInvariant(
            type="json_value_equals_literal",
            id="exactly-one-committed-payment",
            version="1.0.0",
            description="The invoice has exactly one committed payment.",
            severity="critical",
            value_pointer="/committed_effect_count",
            literal=1,
        ),
        JsonArrayItemsUniqueByInvariant(
            type="json_array_items_unique_by",
            id="committed-payments-are-unique",
            version="1.0.0",
            description="Committed payments remain unique by invoice.",
            severity="critical",
            array_pointer="/committed_effects",
            key_pointers=("/invoice_reference",),
        ),
    )


class _DuplicateOnRetrySandbox:
    sandbox_id = "retry-test-sandbox"
    config_sha256 = "0" * 64
    capabilities = SandboxCapabilities(
        supports_conversations=True,
        supports_state_observation=True,
        state_observation_authority="sandbox_self_reported",
        cancellation_guarantee="guaranteed",
    )

    def __init__(
        self,
        *,
        omit_first_variation_commit: bool = False,
        duplicate_on_retry: bool = True,
    ) -> None:
        self.omit_first_variation_commit = omit_first_variation_commit
        self.duplicate_on_retry = duplicate_on_retry
        self.conversations: list[tuple[str, ...]] = []

    def api_calls_for_case(self, case: EvaluationCase) -> int:
        return 3 + (2 * len(case.turns))

    async def execute(self, case: EvaluationCase) -> ExecutionEvidence:
        raw_inputs = tuple(turn.content for turn in case.turns)
        self.conversations.append(raw_inputs)
        outputs = self._outputs(raw_inputs)
        final_output = outputs[-1]
        return ExecutionEvidence(
            case_id=case.id,
            sandbox_id=self.sandbox_id,
            sandbox_config_sha256=self.config_sha256,
            initial_state=SandboxStateEvidence(
                value={"committed_effect_count": 0, "committed_effects": []},
                authority="sandbox_self_reported",
            ),
            turns=tuple(
                SandboxTurnEvidence(
                    turn_id=turn.id,
                    response=output.raw_output,
                    state_snapshot=output.metadata["committed_state_snapshot"],
                    state_observation_authority="sandbox_self_reported",
                )
                for turn, output in zip(case.turns, outputs, strict=True)
            ),
            final_response=final_output.raw_output,
            final_state=SandboxStateEvidence(
                value=final_output.metadata["committed_state_snapshot"],
                authority="sandbox_self_reported",
            ),
            lifecycle=SandboxLifecycleEvidence(
                terminal_status="succeeded",
                completed_phases=("execute", "cleanup"),
                delivery="certain",
                cleanup="succeeded",
                sandbox_state_uncertain=False,
            ),
        )

    def _outputs(self, raw_inputs: tuple[str, ...]) -> tuple[ObservedAgentOutput, ...]:
        committed_effects: list[dict[str, str]] = []
        outputs: list[ObservedAgentOutput] = []
        for turn_index, _raw_input in enumerate(raw_inputs):
            omit_commit = (
                self.omit_first_variation_commit and len(raw_inputs) == 2 and turn_index == 0
            )
            if self.omit_first_variation_commit and len(raw_inputs) == 2 and turn_index == 1:
                effects_to_commit = 2
            elif not self.duplicate_on_retry and len(raw_inputs) == 2 and turn_index == 1:
                effects_to_commit = 0
            else:
                effects_to_commit = 1
            if not omit_commit:
                first_effect_number = len(committed_effects) + 1
                for effect_number in range(
                    first_effect_number, first_effect_number + effects_to_commit
                ):
                    committed_effects.append(
                        {
                            "payment_id": f"payment-{effect_number}",
                            "invoice_reference": "AC-100",
                            "idempotency_key": f"attempt-{effect_number}",
                        }
                    )
            outputs.append(
                ObservedAgentOutput(
                    raw_output={
                        "message": "Payment committed."
                        if turn_index == 0
                        else "Payment retried and committed again."
                    },
                    metadata={
                        "committed_state_snapshot": {
                            "committed_effect_count": len(committed_effects),
                            "committed_effects": [dict(effect) for effect in committed_effects],
                        }
                    },
                )
            )
        return tuple(outputs)


class _UnstableRetrySandbox(_DuplicateOnRetrySandbox):
    def __init__(self) -> None:
        super().__init__()
        self.variation_execution_count = 0

    def _outputs(self, raw_inputs: tuple[str, ...]) -> tuple[ObservedAgentOutput, ...]:
        if len(raw_inputs) == 2:
            self.variation_execution_count += 1
            self.duplicate_on_retry = self.variation_execution_count == 1
        return super()._outputs(raw_inputs)


@pytest.mark.asyncio
async def test_finds_repeatable_retry_after_successful_commit_failure() -> None:
    sandbox = _DuplicateOnRetrySandbox()

    result = await run_retry_after_successful_commit_stress_test(
        _case(),
        sandbox,
        invariant_rules=_invariants(),
        repetitions=3,
        max_sandbox_api_calls=36,
        allow_network_egress=True,
    )

    assert result.case.operator_id == "conversation.retry_after_successful_commit"
    assert result.case.operator_version == "1.0.0"
    assert result.status == "failed"
    assert result.required_target_calls == 36
    assert result.baseline_drift_observed is False
    assert {rule.status for rule in result.baseline_invariant_rules} == {"satisfied"}
    assert {rule.status for rule in result.successful_commit_invariant_rules} == {"satisfied"}
    assert {rule.status for rule in result.retried_invariant_rules} == {"violated"}
    first_checkpoint = result.trials[0].variation[0].committed_state_snapshot
    retried_checkpoint = result.trials[0].variation[1].committed_state_snapshot
    assert isinstance(first_checkpoint, dict)
    assert isinstance(retried_checkpoint, dict)
    assert first_checkpoint["committed_effect_count"] == 1
    assert retried_checkpoint["committed_effect_count"] == 2
    assert (
        sandbox.conversations
        == [
            ("Pay invoice AC-100.",),
            ("Pay invoice AC-100.", "Retry the same payment for invoice AC-100."),
        ]
        * 3
    )


@pytest.mark.asyncio
async def test_does_not_attribute_failure_without_successful_first_checkpoint() -> None:
    result = await run_retry_after_successful_commit_stress_test(
        _case(),
        _DuplicateOnRetrySandbox(omit_first_variation_commit=True),
        invariant_rules=_invariants(),
        repetitions=1,
        max_sandbox_api_calls=12,
        allow_network_egress=True,
    )

    assert result.status == "inconclusive"
    assert {rule.status for rule in result.successful_commit_invariant_rules} == {
        "satisfied",
        "violated",
    }
    assert {rule.status for rule in result.retried_invariant_rules} == {"violated"}


@pytest.mark.asyncio
async def test_passes_when_retry_reuses_the_successful_committed_effect() -> None:
    result = await run_retry_after_successful_commit_stress_test(
        _case(),
        _DuplicateOnRetrySandbox(duplicate_on_retry=False),
        invariant_rules=_invariants(),
        repetitions=2,
        max_sandbox_api_calls=24,
        allow_network_egress=True,
    )

    assert result.status == "passed"
    assert {rule.status for rule in result.baseline_invariant_rules} == {"satisfied"}
    assert {rule.status for rule in result.successful_commit_invariant_rules} == {"satisfied"}
    assert {rule.status for rule in result.retried_invariant_rules} == {"satisfied"}


@pytest.mark.asyncio
async def test_mixed_retry_outcomes_are_inconclusive_not_a_finding() -> None:
    result = await run_retry_after_successful_commit_stress_test(
        _case(),
        _UnstableRetrySandbox(),
        invariant_rules=_invariants(),
        repetitions=3,
        max_sandbox_api_calls=36,
        allow_network_egress=True,
    )

    assert result.status == "inconclusive"
    assert {rule.status for rule in result.retried_invariant_rules} == {"violated"}
    assert [trial.status for trial in result.retried_invariant_rules[0].trials] == [
        "violated",
        "satisfied",
        "satisfied",
    ]


@pytest.mark.asyncio
async def test_requires_committed_state_invariant_observation() -> None:
    sandbox = _DuplicateOnRetrySandbox()

    with pytest.raises(ValueError, match="committed-state"):
        await run_retry_after_successful_commit_stress_test(
            _case(),
            sandbox,
            invariant_rules=_invariants(),
            observation_authority="agent_response",
            allow_network_egress=True,
        )

    assert sandbox.conversations == []


def test_retry_plan_and_case_loader_preserve_version_and_budget(tmp_path: Path) -> None:
    plan = plan_retry_after_successful_commit_stress_test(
        _case(), _config(), repetitions=2, max_sandbox_api_calls=28
    )

    assert plan.operator_id == "conversation.retry_after_successful_commit"
    assert plan.operator_version == "1.0.0"
    assert plan.target_calls_per_pair == 14
    assert plan.required_target_calls == 28
    with pytest.raises(ValueError, match="authorized target call budget"):
        plan_retry_after_successful_commit_stress_test(
            _case(), _config(), repetitions=2, max_sandbox_api_calls=27
        )

    case_path = tmp_path / "retry-case.json"
    case_path.write_text(_case().model_dump_json(), encoding="utf-8")
    assert load_retry_after_successful_commit_case(case_path) == _case()
