from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner
from ul import (
    DatasetEvaluationResult,
    ObservedAgentOutput,
)
from ul.dataset_invariants import (
    DatasetInvariantEvaluation,
    DatasetInvariantSuite,
    JsonValueEqualsLiteralInvariant,
    NoNewEffectInvariant,
)
from ul_cli import dataset_review
from ul_cli.dataset.evaluation import command as command_module
from ul_cli.dataset.evaluation import runner as runner_module
from ul_cli.dataset.evidence import customer as customer_module
from ul_cli.dataset.presentation import evaluation as presentation_module
from ul_cli.main import app as root_app
from ul_core.evaluation import (
    EnvironmentLifecycleEvidence,
    EnvironmentResetEvidence,
    EnvironmentStateEvidence,
    EnvironmentTurnEvidence,
    ExecutionEvidence,
)

from ._factories import (
    _evaluation_result,
    _evaluator_preflight,
    _invariant_evaluation,
    _run_context,
)
from ._files import (
    _record,
    _write_dataset,
    _write_invariant_suite,
)

runner = CliRunner()
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def test_invariant_dry_run_reports_rules_authority_and_no_extra_calls(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    invariant_suite = tmp_path / "invariants.json"
    _write_dataset(dataset, [_record()])
    _write_invariant_suite(invariant_suite)

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--invariants",
            str(invariant_suite),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Customer invariants: 1 rule(s)" in result.output
    assert "Declared observation authority: committed_state_snapshot" in result.output
    assert "Additional model calls for customer invariants: 0" in result.output
    assert "Additional environment API calls for customer invariants: 0" in result.output
    assert "Potential semantic model calls: up to 13" in result.output
    assert "Potential environment API calls: up to 6" in result.output


def test_isolated_response_dry_run_rejects_committed_state_invariants(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    invariant_suite = tmp_path / "invariants.json"
    target_config = tmp_path / "isolated-response.json"
    _write_dataset(dataset, [_record()])
    _write_invariant_suite(invariant_suite)
    target_config.write_text(
        json.dumps(
            {
                "version": 1,
                "adapter_tier": "isolated_response",
                "environment_id": "response-only-test",
                "request_isolation_attested": True,
                "safe_test_target_attested": True,
                "execute": {
                    "url": "https://environment.example.test/execute",
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "turn_id": "{{turn_id}}",
                        "input": "{{input}}",
                    },
                    "response_json_pointer": "/response",
                },
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--invariants",
            str(invariant_suite),
            "--dry-run",
        ],
        terminal_width=180,
    )

    assert result.exit_code != 0
    assert "isolated-response targets provide response evidence only" in " ".join(
        result.output.split()
    )


def test_extended_invariants_use_new_evidence_schema_and_hide_values_from_terminal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite = DatasetInvariantSuite(
        schema_version="1.1.0",
        observation_source="target_output",
        observation_authority="committed_state_snapshot",
        rules=(
            JsonValueEqualsLiteralInvariant(
                type="json_value_equals_literal",
                id="approval-is-current",
                version="1.0.0",
                description="The approval must be current.",
                severity="critical",
                value_pointer="/approval",
                literal="private-current-version",
            ),
        ),
    )
    evaluation_result = _evaluation_result("interaction-1")
    baseline_trial = evaluation_result.baseline.trial_set.trials[0].model_copy(
        update={
            "target_output": ObservedAgentOutput(raw_output={"approval": "private-stale-version"})
        }
    )
    evaluation_result = evaluation_result.model_copy(
        update={
            "baseline": evaluation_result.baseline.model_copy(
                update={
                    "trial_set": evaluation_result.baseline.trial_set.model_copy(
                        update={"trials": (baseline_trial,)}
                    )
                }
            )
        }
    )
    invariant_evaluation = runner_module.evaluate_dataset_invariants(evaluation_result, suite)
    run_context = _run_context((evaluation_result.source,), invariant_suite=suite)

    record = customer_module.build_customer_evidence_record(
        evaluation_result,
        repetitions=1,
        max_environment_api_calls=2,
        planned_target_calls=2,
        run_context=cast(Any, run_context),
        invariant_evaluation=invariant_evaluation,
    )
    parsed = dataset_review._EvidenceRecord.model_validate_json(json.dumps(record))
    presentation_module._print_invariant_results((invariant_evaluation,))
    terminal_output = capsys.readouterr().out

    assert parsed.schema_version == "1.12.0"
    assert parsed.evaluation_mode == "variance"
    assert parsed.run_context is not None
    assert parsed.run_context.evaluation_mode == "variance"
    missing_technical_mode = json.loads(json.dumps(record))
    missing_technical_mode["technical_details"].pop("evaluation_mode")
    with pytest.raises(ValidationError, match="evaluation mode must match technical details"):
        dataset_review._EvidenceRecord.model_validate_json(json.dumps(missing_technical_mode))
    assert "value=/approval" in terminal_output
    assert "private-current-version" not in terminal_output
    assert "private-stale-version" not in terminal_output


def test_transition_invariant_round_trips_evidence_and_reports_only_pointer_and_count(
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "private-payment-reference"
    suite = DatasetInvariantSuite(
        schema_version="1.2.0",
        observation_source="target_output",
        observation_authority="committed_state_snapshot",
        rules=(
            NoNewEffectInvariant(
                type="no_new_effect",
                id="no-payment",
                version="1.0.0",
                description="No payment may be committed.",
                severity="critical",
                before_checkpoint="before_turn",
                after_checkpoint="after_turn",
                observation_pointer="/payments",
            ),
        ),
    )
    evaluation_result = _evaluation_result("interaction-1")
    baseline_trial = evaluation_result.baseline.trial_set.trials[0].model_copy(
        update={
            "execution_evidence": ExecutionEvidence(
                case_id="interaction-1",
                environment_id="environment",
                environment_config_sha256="0" * 64,
                initial_state=EnvironmentStateEvidence(
                    value={"payments": []}, authority="environment_self_reported"
                ),
                turns=(
                    EnvironmentTurnEvidence(
                        turn_id="turn-1",
                        response={"ignored": True},
                        state_snapshot={"payments": [{"id": secret}]},
                        state_observation_authority="environment_self_reported",
                    ),
                ),
                final_response={"ignored": True},
                final_state=EnvironmentStateEvidence(
                    value={"payments": [{"id": secret}]},
                    authority="environment_self_reported",
                ),
                lifecycle=EnvironmentLifecycleEvidence(
                    terminal_status="succeeded",
                    delivery="certain",
                    cleanup="succeeded",
                    environment_state_uncertain=False,
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
                ),
            ),
            "target_output": ObservedAgentOutput(
                raw_output={"ignored": True},
                metadata={
                    "committed_state_before_turn": {"payments": []},
                    "committed_state_snapshot": {"payments": [{"id": secret}]},
                    "state_observation_authority": "environment_self_reported",
                },
            ),
        }
    )
    evaluation_result = evaluation_result.model_copy(
        update={
            "baseline": evaluation_result.baseline.model_copy(
                update={
                    "trial_set": evaluation_result.baseline.trial_set.model_copy(
                        update={"trials": (baseline_trial,)}
                    )
                }
            )
        }
    )
    invariant_evaluation = runner_module.evaluate_dataset_invariants(evaluation_result, suite)
    record = customer_module.build_customer_evidence_record(
        evaluation_result,
        repetitions=1,
        max_environment_api_calls=2,
        planned_target_calls=2,
        run_context=cast(Any, _run_context((evaluation_result.source,), invariant_suite=suite)),
        invariant_evaluation=invariant_evaluation,
    )

    parsed = dataset_review._EvidenceRecord.model_validate_json(json.dumps(record))
    presentation_module._print_invariant_results((invariant_evaluation,))
    terminal_output = capsys.readouterr().out

    assert parsed.invariant_evaluation is not None
    assert parsed.invariant_evaluation.baseline.rules[0].status == "violated"
    assert "before=before_turn; after=after_turn; value=/payments; new_effects=1" in (
        terminal_output
    )
    assert secret not in json.dumps(parsed.invariant_evaluation.model_dump(mode="json"))
    assert secret not in terminal_output


def test_invalid_invariant_config_stops_before_settings_network_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    invariant_suite = tmp_path / "invariants.json"
    output = tmp_path / "evidence.jsonl"
    _write_dataset(dataset, [_record()])
    invariant_suite.write_text(
        '{"schema_version":"1.0.0","observation_source":"target_output",'
        '"observation_authority":"agent_response","rules":[]}',
        encoding="utf-8",
    )

    def unexpected_settings() -> None:
        raise AssertionError("invalid invariants reached settings")

    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", unexpected_settings)
    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--invariants",
            str(invariant_suite),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert "invariant suite is invalid" in result.output
    assert not output.exists()


def test_invariant_evaluation_reuses_results_without_extra_runner_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invariant_path = tmp_path / "invariants.json"
    _write_invariant_suite(invariant_path)
    suite = command_module.load_dataset_invariant_suite(invariant_path)
    runner_calls = 0
    invariant_calls = 0
    stored_evaluations: list[DatasetInvariantEvaluation] = []

    class AsyncContext:
        async def __aenter__(self) -> object:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        def reuse_preflight(self, result: object) -> None:
            assert result == _evaluator_preflight()

    class FakeRunner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def run(self, *args: object, **kwargs: object) -> DatasetEvaluationResult:
            nonlocal runner_calls
            runner_calls += 1
            return cast(DatasetEvaluationResult, SimpleNamespace())

    evaluation = _invariant_evaluation("satisfied", "violated")

    def evaluate_once(*args: object, **kwargs: object) -> DatasetInvariantEvaluation:
        nonlocal invariant_calls
        invariant_calls += 1
        return evaluation

    monkeypatch.setattr(
        runner_module,
        "create_semantic_model_deconstructor",
        lambda settings: AsyncContext(),
    )
    monkeypatch.setattr(runner_module, "DatasetAugmentationEngine", lambda *args: object())
    monkeypatch.setattr(runner_module, "DatasetEvaluationRunner", FakeRunner)
    monkeypatch.setattr(runner_module, "evaluate_dataset_invariants", evaluate_once)
    monkeypatch.setattr(
        runner_module,
        "build_customer_evidence_record",
        lambda result, **options: {
            "schema_version": "1.4.0",
            "invariant_evaluation": options["invariant_evaluation"].model_dump(mode="json"),
        },
    )
    output = tmp_path / "evidence.jsonl"
    with output.open("w", encoding="utf-8") as output_stream:
        results = asyncio.run(
            runner_module.evaluate_interaction_records(
                (cast(Any, SimpleNamespace()),),
                ("input.surface.rephrase",),
                cast(Any, SimpleNamespace()),
                cast(Any, AsyncContext()),
                output_stream,
                repetitions=1,
                max_environment_api_calls=2,
                planned_target_calls=2,
                invariant_suite=suite,
                invariant_evaluations=stored_evaluations,
                evaluator_preflight=_evaluator_preflight(),
            )
        )

    assert len(results) == 1
    assert runner_calls == 1
    assert invariant_calls == 1
    assert stored_evaluations == [evaluation]
    assert json.loads(output.read_text(encoding="utf-8"))["invariant_evaluation"] == (
        evaluation.model_dump(mode="json")
    )
