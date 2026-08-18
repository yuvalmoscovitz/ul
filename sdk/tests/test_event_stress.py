from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import typer
from ul.dataset_invariants import JsonValuesEqualInvariant
from ul.event_stress import (
    CorrectionAfterFirstResponseCase,
    create_multi_turn_regression_case,
    load_multi_turn_regression_case,
    plan_correction_stress_test,
    replay_multi_turn_regression,
    run_correction_stress_test,
)
from ul.http_target import JsonHttpDatasetTargetConfig
from ul_cli.event_stress import _print_result
from ul_core.dataset import ObservedAgentOutput
from ul_core.models import ConversationRole, ConversationTurn, SafetyEnvelope


def _case() -> CorrectionAfterFirstResponseCase:
    return CorrectionAfterFirstResponseCase(
        id="invoice-correction",
        conversation=(
            ConversationTurn(
                id="initial-request",
                role=ConversationRole.USER,
                content="Pay invoice AC-100.",
            ),
            ConversationTurn(
                id="corrected-request",
                role=ConversationRole.USER,
                content="Correction: pay AC-101 instead.",
            ),
        ),
    )


def _config() -> JsonHttpDatasetTargetConfig:
    return JsonHttpDatasetTargetConfig.model_validate(
        {
            "version": 2,
            "reset": {
                "url": "https://sandbox.example.test/reset",
                "generation_json_pointer": "/generation",
                "clean_state_json_pointer": "/clean",
                "clean_state_value": True,
            },
            "setup": {"url": "https://sandbox.example.test/setup"},
            "execute_turn": {
                "url": "https://sandbox.example.test/execute",
                "request_json_template": {"input": "{{input}}"},
            },
            "snapshot": {"url": "https://sandbox.example.test/snapshot"},
        }
    )


def _invariant() -> JsonValuesEqualInvariant:
    return JsonValuesEqualInvariant(
        type="json_values_equal",
        id="committed-invoice-follows-latest-request",
        version="1.0.0",
        description="The committed invoice must follow the latest request.",
        severity="critical",
        left_pointer="/committed_invoice",
        right_pointer="/requested_invoice",
    )


class _DefectiveCorrectionTarget:
    def __init__(self) -> None:
        self.conversations: list[tuple[str, ...]] = []
        self.safety_envelope = SafetyEnvelope(
            description="isolated defective target",
            isolated=True,
            allows_network_egress=False,
            allows_business_side_effects=False,
        )

    @property
    def fresh_state_per_execution(self) -> bool:
        return True

    def target_calls_for_conversation(self, turn_count: int) -> int:
        return 3 + (2 * turn_count)

    async def execute(self, raw_input: str) -> ObservedAgentOutput:
        return (await self.execute_conversation((raw_input,)))[0]

    async def execute_conversation(
        self, raw_inputs: tuple[str, ...]
    ) -> tuple[ObservedAgentOutput, ...]:
        self.conversations.append(raw_inputs)
        outputs: list[ObservedAgentOutput] = []
        for turn_index, _ in enumerate(raw_inputs):
            requested_invoice = "AC-101" if turn_index == 1 else "AC-100"
            snapshot = {
                "committed_invoice": "AC-100",
                "requested_invoice": requested_invoice,
            }
            outputs.append(
                ObservedAgentOutput(
                    raw_output={
                        "message": "paid AC-100" if turn_index == 0 else "correction ignored"
                    },
                    metadata={"committed_state_snapshot": snapshot},
                )
            )
        return tuple(outputs)


class _NondeterministicCorrectionTarget(_DefectiveCorrectionTarget):
    def __init__(self) -> None:
        super().__init__()
        self.variation_repetition = 0

    async def execute_conversation(
        self, raw_inputs: tuple[str, ...]
    ) -> tuple[ObservedAgentOutput, ...]:
        outputs = await super().execute_conversation(raw_inputs)
        if len(raw_inputs) != 2:
            return outputs
        self.variation_repetition += 1
        if self.variation_repetition != 2:
            return outputs
        drifted_outputs: list[ObservedAgentOutput] = []
        for output in outputs:
            snapshot = dict(output.metadata["committed_state_snapshot"])
            snapshot["baseline_drift"] = True
            drifted_outputs.append(
                ObservedAgentOutput(
                    raw_output={"message": f"drifted: {output.raw_output['message']}"},
                    metadata={"committed_state_snapshot": snapshot},
                )
            )
        return tuple(drifted_outputs)


@pytest.mark.asyncio
async def test_finds_repeatable_correction_failure_and_preserves_ordered_evidence() -> None:
    target = _DefectiveCorrectionTarget()

    result = await run_correction_stress_test(
        _case(),
        target,
        invariant_rules=(_invariant(),),
        repetitions=3,
        max_target_calls=36,
    )

    assert result.status == "failed"
    assert result.required_target_calls == 36
    assert result.first_response_divergence_turn_id == "corrected-request"
    assert result.first_committed_state_divergence_turn_id == "corrected-request"
    assert result.baseline_invariant_rules[0].status == "satisfied"
    assert result.corrected_invariant_rules[0].status == "violated"
    assert [observation.turn.id for observation in result.trials[0].variation] == [
        "initial-request",
        "corrected-request",
    ]
    assert result.trials[0].variation[1].committed_state_snapshot == {
        "committed_invoice": "AC-100",
        "requested_invoice": "AC-101",
    }
    assert (
        target.conversations
        == [
            ("Pay invoice AC-100.",),
            ("Pay invoice AC-100.", "Correction: pay AC-101 instead."),
        ]
        * 3
    )


@pytest.mark.asyncio
async def test_first_divergence_uses_conversation_order_and_flags_nondeterminism(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = await run_correction_stress_test(
        _case(),
        _NondeterministicCorrectionTarget(),
        invariant_rules=(_invariant(),),
        repetitions=2,
        max_target_calls=24,
    )

    assert [
        [
            divergence.variation_turn_id
            for divergence in trial.divergences
            if divergence.response_diverged
        ]
        for trial in result.trials
    ] == [["corrected-request"], ["initial-request", "corrected-request"]]
    assert result.first_response_divergence_turn_id == "initial-request"
    assert result.response_divergence_counts == {
        "initial-request": 1,
        "corrected-request": 2,
    }
    assert result.response_divergence_stability == "unstable"
    assert result.first_committed_state_divergence_turn_id == "initial-request"
    assert result.committed_state_divergence_stability == "unstable"
    assert result.baseline_drift_observed is True

    with pytest.raises(typer.Exit):
        _print_result(result, Path("private-evidence.json"))
    report = capsys.readouterr().out
    assert (
        "Response divergence stability: unstable; "
        "counts=initial-request=1, corrected-request=2" in report
    )
    assert "do not attribute the corrected-arm failure to the correction alone" in report


def test_dry_run_plan_enforces_complete_pair_budget_without_target_calls() -> None:
    plan = plan_correction_stress_test(_case(), _config(), repetitions=2, max_target_calls=24)

    assert plan.target_calls_per_pair == 12
    assert plan.required_target_calls == 24
    with pytest.raises(ValueError, match="authorized target call budget"):
        plan_correction_stress_test(_case(), _config(), repetitions=2, max_target_calls=23)


def test_saved_multi_turn_regression_round_trips_and_replays(tmp_path: Path) -> None:
    regression = create_multi_turn_regression_case(
        stress_case=_case(),
        target_config=_config(),
        source_suite_sha256="1" * 64,
        observation_authority="committed_state_snapshot",
        invariant_rules=(_invariant(),),
        repetitions=2,
    )
    path = tmp_path / "correction-regression.json"
    path.write_text(regression.model_dump_json(), encoding="utf-8")

    loaded = load_multi_turn_regression_case(path)
    result = asyncio.run(
        replay_multi_turn_regression(
            loaded,
            _DefectiveCorrectionTarget(),
            max_target_calls=24,
        )
    )

    assert loaded == regression
    assert loaded.case_id.startswith("ulmc_v1_")
    assert result.status == "failed"
