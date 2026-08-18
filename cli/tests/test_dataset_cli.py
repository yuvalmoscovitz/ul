from __future__ import annotations

import asyncio
import json
import re
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import SecretStr
from typer.testing import CliRunner
from ul import (
    DatasetAugmentationResult,
    DatasetEvaluationBaseline,
    DatasetEvaluationCase,
    DatasetEvaluationFinding,
    DatasetEvaluationOutcomeGroup,
    DatasetEvaluationResult,
    DatasetEvaluationTrial,
    DatasetEvaluationTrialSet,
    InteractionRecord,
    JsonHttpDatasetTargetConfig,
    ObservedAgentOutput,
    SemanticFrame,
)
from ul.dataset_augmentation import DatasetAugmentationCandidate
from ul.dataset_invariants import (
    DatasetInvariantArmEvaluation,
    DatasetInvariantEvaluation,
    DatasetInvariantRuleEvaluation,
    DatasetInvariantSuite,
    JsonValueEqualsLiteralInvariant,
)
from ul_cli import dataset as main
from ul_cli import dataset_review
from ul_cli.main import app as root_app

runner = CliRunner()
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _write_dataset(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _record(identifier: str = "interaction-1") -> dict[str, Any]:
    return {
        "id": identifier,
        "input": "Transfer 100 to Alice.",
        "output": {"actions": [{"action": "transfer", "amount": 100, "recipient": "Alice"}]},
    }


def _settings(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "live_calls": True,
        "allow_external_data_processing": True,
        "api_key": SecretStr("test-key"),
        "model": "test/deconstructor",
        "render_model": "test/renderer",
        "equivalence_model": "test/equivalence",
        "max_input_chars": 50_000,
        "max_output_tokens": 4_096,
        "max_render_tokens": 512,
        "max_response_bytes": 1_000_000,
        "timeout_seconds": 60.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _evaluation_result(
    identifier: str,
    *,
    has_review_finding: bool = False,
) -> DatasetEvaluationResult:
    source = InteractionRecord(
        id=identifier,
        raw_input="Transfer 100 to Alice.",
        raw_observed_output={
            "actions": [{"action": "transfer", "amount": 100, "recipient": "Alice"}]
        },
    )
    source_frame = SemanticFrame(interaction_id=identifier, extractor_version="test")
    baseline_frame = SemanticFrame(
        interaction_id=f"{identifier}:current_baseline:round-1",
        extractor_version="test",
    )
    trial_set = DatasetEvaluationTrialSet(
        requested_repetitions=1,
        stability="stable",
        trials=(
            DatasetEvaluationTrial(
                repetition=1,
                target_output=ObservedAgentOutput(raw_output={"status": "ok"}),
                observed_frame=baseline_frame,
            ),
        ),
        outcome_groups=(
            DatasetEvaluationOutcomeGroup(repetitions=(1,), representative_effects=()),
        ),
    )
    candidate = DatasetAugmentationCandidate(
        source_interaction_id=identifier,
        operator_id="surface.rephrase",
        augmented_input="Please transfer 100 to Alice.",
        expected_input_frame=source_frame,
        reparsed_input_frame=source_frame if has_review_finding else None,
        passed=has_review_finding,
        failure_reasons=() if has_review_finding else ("test rejection",),
    )
    if has_review_finding:
        variation_trial_set = DatasetEvaluationTrialSet(
            requested_repetitions=1,
            stability="stable",
            trials=(
                DatasetEvaluationTrial(
                    repetition=1,
                    target_output=ObservedAgentOutput(raw_output={"status": "changed"}),
                    observed_frame=SemanticFrame(
                        interaction_id=f"{identifier}:surface.rephrase:round-1",
                        extractor_version="test",
                    ),
                ),
            ),
            outcome_groups=(
                DatasetEvaluationOutcomeGroup(repetitions=(1,), representative_effects=()),
            ),
        )
        evaluation_case = DatasetEvaluationCase(
            candidate=candidate,
            verdict="divergence_needs_review",
            trial_set=variation_trial_set,
            findings=(
                DatasetEvaluationFinding(
                    category="unexpected_effect",
                    message="The variation changed observable behavior.",
                ),
            ),
        )
    else:
        evaluation_case = DatasetEvaluationCase(
            candidate=candidate,
            verdict="augmentation_rejected",
        )
    return DatasetEvaluationResult(
        source=source,
        augmentation=DatasetAugmentationResult(
            source_frames=(source_frame,),
            candidates=(candidate,),
        ),
        baseline=DatasetEvaluationBaseline(verdict="no_divergence", trial_set=trial_set),
        cases=(evaluation_case,),
    )


def _run_context(
    records: tuple[InteractionRecord, ...],
    *,
    invariant_suite: object | None = None,
) -> object:
    return main._dataset_evidence_run_context(
        selected_records=records,
        selected_operator_ids=("surface.rephrase",),
        repetitions=1,
        invariant_suite=cast(Any, invariant_suite),
        target_config=JsonHttpDatasetTargetConfig(
            version=1,
            url="https://sandbox.example.test/execute",
            headers_from_env={},
            request_json_template={"input": "{{input}}"},
            response_json_pointer="",
        ),
        settings=cast(Any, _settings()),
    )


def _write_target_config(
    path: Path,
    *,
    url: str = "https://sandbox.example.test/execute",
    headers_from_env: dict[str, str] | None = None,
    request_json_template: object | None = None,
    response_json_pointer: str = "",
) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "url": url,
                "headers_from_env": headers_from_env or {},
                "request_json_template": request_json_template
                if request_json_template is not None
                else {"input": "{{input}}"},
                "response_json_pointer": response_json_pointer,
            }
        ),
        encoding="utf-8",
    )


def _write_invariant_suite(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "observation_source": "target_output",
                "observation_authority": "committed_state_snapshot",
                "rules": [
                    {
                        "type": "json_values_equal",
                        "id": "amount-matches-corrected",
                        "version": "1.0.0",
                        "description": "Final amount equals the corrected amount.",
                        "severity": "high",
                        "left_pointer": "/final_amount",
                        "right_pointer": "/corrected_amount",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _invariant_evaluation(
    baseline_status: str = "satisfied",
    variation_status: str | None = None,
    *,
    interaction_id: str = "case-1",
    suite_sha256: str = "a" * 64,
) -> DatasetInvariantEvaluation:
    def arm_rule(status: str) -> DatasetInvariantRuleEvaluation:
        if status == "satisfied":
            trial_reason = "values_equal"
            aggregate_reason = "all_trials_satisfied"
            resolved_values = {"left": 100, "right": 100}
        elif status == "violated":
            trial_reason = "values_differ"
            aggregate_reason = "one_or_more_trials_violated"
            resolved_values = {"left": 200, "right": 100}
        else:
            trial_reason = "left_pointer_missing"
            aggregate_reason = "one_or_more_trials_not_evaluable"
            resolved_values = {}
        return DatasetInvariantRuleEvaluation.model_validate(
            {
                "rule_type": "json_values_equal",
                "rule_id": "amount-matches-corrected",
                "rule_version": "1.0.0",
                "description": "Final amount equals the corrected amount.",
                "severity": "high",
                "status": status,
                "reason_code": aggregate_reason,
                "trials": (
                    {
                        "repetition": 1,
                        "status": status,
                        "reason_code": trial_reason,
                        "left_pointer": "/final_amount",
                        "right_pointer": "/corrected_amount",
                        "resolved_values": resolved_values,
                    },
                ),
            }
        )

    variations = (
        ()
        if variation_status is None
        else (
            DatasetInvariantArmEvaluation(
                arm="variation",
                operator_id="surface.rephrase",
                rules=(arm_rule(variation_status),),
            ),
        )
    )
    return DatasetInvariantEvaluation(
        interaction_id=interaction_id,
        suite_sha256=suite_sha256,
        observation_authority="committed_state_snapshot",
        baseline=DatasetInvariantArmEvaluation(
            arm="baseline",
            rules=(arm_rule(baseline_status),),
        ),
        variations=variations,
    )


def test_init_creates_private_strict_starter_config(tmp_path: Path) -> None:
    target_config = tmp_path / "target.json"

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "init",
            str(target_config),
            "--url",
            "https://sandbox.example.test/execute",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(target_config.read_text(encoding="utf-8")) == {
        "version": 1,
        "url": "https://sandbox.example.test/execute",
        "headers_from_env": {},
        "request_json_template": {"input": "{{input}}"},
        "response_json_pointer": "",
    }
    assert stat.S_IMODE(target_config.stat().st_mode) == 0o600
    assert "request_json_template" in result.output
    assert "response_json_pointer" in result.output
    assert "headers_from_env" in result.output
    assert "--dry-run" in result.output


def test_init_refuses_invalid_url_and_existing_file(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid.json"
    invalid_url = runner.invoke(
        root_app,
        ["dataset", "init", str(invalid_config), "--url", "file:///etc/passwd"],
    )

    assert invalid_url.exit_code != 0
    assert not invalid_config.exists()

    existing_config = tmp_path / "target.json"
    existing_config.write_text("keep me", encoding="utf-8")
    collision = runner.invoke(
        root_app,
        [
            "dataset",
            "init",
            str(existing_config),
            "--url",
            "https://sandbox.example.test/execute",
        ],
    )

    assert collision.exit_code != 0
    normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", collision.output).split())
    assert "will not" in normalized_output
    assert "overwrite it" in normalized_output
    assert existing_config.read_text(encoding="utf-8") == "keep me"


def _trial_set(
    *,
    requested_repetitions: int = 3,
    stability: str = "stable",
    outcome_group_repetitions: tuple[tuple[int, ...], ...] | None = None,
    representative_effect: object | None = None,
) -> SimpleNamespace:
    if outcome_group_repetitions is None:
        outcome_group_repetitions = (tuple(range(1, requested_repetitions + 1)),)
    grouped_repetitions = {
        repetition for group in outcome_group_repetitions for repetition in group
    }
    trials = tuple(
        SimpleNamespace(
            repetition=repetition,
            inconclusive_reasons=(
                () if repetition in grouped_repetitions else ("target execution failed",)
            ),
        )
        for repetition in range(1, requested_repetitions + 1)
    )
    effects = () if representative_effect is None else (representative_effect,)
    outcome_groups = tuple(
        SimpleNamespace(repetitions=repetitions, representative_effects=effects)
        for repetitions in outcome_group_repetitions
    )
    return SimpleNamespace(
        requested_repetitions=requested_repetitions,
        stability=stability,
        trials=trials,
        outcome_groups=outcome_groups,
    )


def test_dry_run_validates_and_makes_no_external_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record(), _record("interaction-2")])

    def unexpected_deconstructor(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run constructed a semantic model client")

    def unexpected_target(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run constructed a target client")

    monkeypatch.setattr(main, "OpenRouterSemanticDeconstructor", unexpected_deconstructor)
    monkeypatch.setattr(main.JsonHttpDatasetTarget, "from_config", unexpected_target)
    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--operator",
            "surface.disfluency_repeat",
            "--limit",
            "1",
            "--target-url",
            "https://sandbox.example.test/execute",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Dataset valid: 2 interaction(s)" in result.output
    assert "Selected interactions: 1" in result.output
    assert "Repetitions: 3 per original and accepted variation" in result.output
    assert "Potential semantic model calls: up to 10" in result.output
    assert "Potential target calls: up to 6" in result.output
    assert "authorized maximum: 100" in result.output
    assert "Semantic models receive historical inputs and outputs" in result.output
    assert "generated variations" in result.output
    assert "live control responses" in result.output
    assert "Every target request must start from the same clean state" in result.output
    assert "each accepted variation for every repetition" in " ".join(result.output.split())
    assert "do not determine correctness" in result.output
    assert "identify causality" in result.output
    assert "estimate a production failure rate" in result.output
    assert "No model or target requests sent." in result.output
    assert "Transfer 100" not in result.output


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
    assert "Additional target calls for customer invariants: 0" in result.output
    assert "Potential semantic model calls: up to 10" in result.output
    assert "Potential target calls: up to 6" in result.output


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
    invariant_evaluation = main.evaluate_dataset_invariants(evaluation_result, suite)
    run_context = _run_context((evaluation_result.source,), invariant_suite=suite)

    record = main._customer_evidence_record(
        evaluation_result,
        repetitions=1,
        max_target_calls=2,
        planned_target_calls=2,
        run_context=cast(Any, run_context),
        invariant_evaluation=invariant_evaluation,
    )
    parsed = dataset_review._EvidenceRecord.model_validate_json(json.dumps(record))
    main._print_invariant_results((invariant_evaluation,))
    terminal_output = capsys.readouterr().out

    assert parsed.schema_version == "1.6.0"
    assert "value=/approval" in terminal_output
    assert "private-current-version" not in terminal_output
    assert "private-stale-version" not in terminal_output


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

    monkeypatch.setattr(main, "OpenRouterDatasetSettings", unexpected_settings)
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
    suite = main.load_dataset_invariant_suite(invariant_path)
    runner_calls = 0
    invariant_calls = 0
    stored_evaluations: list[DatasetInvariantEvaluation] = []

    class AsyncContext:
        async def __aenter__(self) -> object:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

    class FakeRunner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def run(self, *args: object, **kwargs: object) -> DatasetEvaluationResult:
            nonlocal runner_calls
            runner_calls += 1
            return cast(DatasetEvaluationResult, SimpleNamespace())

    evaluation = _invariant_evaluation("satisfied", "violated")

    def evaluate_once(*args: object) -> DatasetInvariantEvaluation:
        nonlocal invariant_calls
        invariant_calls += 1
        return evaluation

    monkeypatch.setattr(main, "OpenRouterSemanticDeconstructor", lambda settings: AsyncContext())
    monkeypatch.setattr(main, "DatasetAugmentationEngine", lambda *args: object())
    monkeypatch.setattr(main, "DatasetEvaluationRunner", FakeRunner)
    monkeypatch.setattr(main, "evaluate_dataset_invariants", evaluate_once)
    monkeypatch.setattr(
        main,
        "_customer_evidence_record",
        lambda result, **options: {
            "schema_version": "1.4.0",
            "invariant_evaluation": options["invariant_evaluation"].model_dump(mode="json"),
        },
    )
    output = tmp_path / "evidence.jsonl"
    with output.open("w", encoding="utf-8") as output_stream:
        results = asyncio.run(
            main._evaluate_interaction_records(
                (cast(Any, SimpleNamespace()),),
                ("surface.rephrase",),
                cast(Any, SimpleNamespace()),
                cast(Any, AsyncContext()),
                output_stream,
                repetitions=1,
                max_target_calls=2,
                planned_target_calls=2,
                invariant_suite=suite,
                invariant_evaluations=stored_evaluations,
            )
        )

    assert len(results) == 1
    assert runner_calls == 1
    assert invariant_calls == 1
    assert stored_evaluations == [evaluation]
    assert json.loads(output.read_text(encoding="utf-8"))["invariant_evaluation"] == (
        evaluation.model_dump(mode="json")
    )


def test_target_config_dry_run_validates_environment_and_makes_no_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(
        target_config,
        headers_from_env={"Authorization": "SANDBOX_TOKEN"},
        request_json_template={"request": {"message": "{{input}}"}},
        response_json_pointer="/result",
    )
    monkeypatch.setenv("SANDBOX_TOKEN", "Bearer test-token")

    def unexpected_deconstructor(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run constructed a semantic model client")

    monkeypatch.setattr(main, "OpenRouterSemanticDeconstructor", unexpected_deconstructor)
    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target-config",
            str(target_config),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Target: configured" in result.output
    assert "https://sandbox.example.test/execute" in result.output
    assert "Authorization=SANDBOX_TOKEN" in result.output
    assert "Bearer test-token" not in result.output
    assert "No model or target requests sent" in result.output

    monkeypatch.delenv("SANDBOX_TOKEN")
    missing_environment = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target-config",
            str(target_config),
            "--dry-run",
        ],
    )

    assert missing_environment.exit_code != 0
    assert "environment variable is not set" in missing_environment.output
    assert "No model or target requests sent" not in missing_environment.output


@pytest.mark.parametrize(
    "conflicting_options",
    [
        ["--target-url", "https://sandbox.example.test/execute"],
        ["--request-field", "message"],
        ["--header-env", "Authorization=SANDBOX_TOKEN"],
    ],
)
def test_target_config_rejects_legacy_target_options(
    tmp_path: Path,
    conflicting_options: list[str],
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config)

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target-config",
            str(target_config),
            *conflicting_options,
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", result.output).split())
    assert "--target-config cannot be combined" in normalized_output


@pytest.mark.parametrize(
    "payload",
    [
        {
            "version": 1,
            "url": "https://sandbox.example.test/execute",
            "headers_from_env": {},
            "request_json_template": {"input": "{{input}}"},
            "response_json_pointer": "",
            "unknown": True,
        },
        {
            "version": 1,
            "url": "https://sandbox.example.test/execute",
            "headers_from_env": {},
            "request_json_template": {"input": "missing marker"},
            "response_json_pointer": "",
        },
        {
            "version": 1,
            "url": "https://sandbox.example.test/execute",
            "headers_from_env": {},
            "request_json_template": {"input": "{{input}}"},
            "response_json_pointer": "not-a-pointer",
        },
    ],
)
def test_dry_run_rejects_invalid_target_config(tmp_path: Path, payload: dict[str, Any]) -> None:
    dataset = tmp_path / "interactions.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    target_config.write_text(json.dumps(payload), encoding="utf-8")

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target-config",
            str(target_config),
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "No model or target requests sent" not in result.output


@pytest.mark.parametrize(
    "options",
    [
        ["--target-url", "file:///etc/passwd"],
        ["--target-url", "https://sandbox.test", "--request-field", "bad.field"],
        [
            "--target-url",
            "https://sandbox.test",
            "--header-env",
            "Host=PATH",
        ],
        [
            "--target-url",
            "https://sandbox.test",
            "--header-env",
            "Authorization=MISSING_SANDBOX_TOKEN",
        ],
    ],
)
def test_dry_run_rejects_invalid_target_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: list[str],
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record()])
    monkeypatch.delenv("MISSING_SANDBOX_TOKEN", raising=False)

    result = runner.invoke(
        root_app,
        ["dataset", "evaluate", str(dataset), *options, "--dry-run"],
    )

    assert result.exit_code != 0
    assert "No model or target requests sent" not in result.output


@pytest.mark.parametrize(
    ("invalid_line", "expected_error"),
    [
        ("not json\n", "line 2: invalid JSON"),
        (
            '{"id":"bad","input":"message","output":NaN}\n',
            "line 2: invalid JSON",
        ),
        (
            '{"id":"first","id":"second","input":"message","output":{}}\n',
            "line 2: invalid JSON",
        ),
        (
            '{"id":"bad","input":"message","output":{"action":"first","action":"second"}}\n',
            "line 2: invalid JSON",
        ),
        (json.dumps({"id": "bad", "input": "message"}) + "\n", "missing output"),
        (
            json.dumps({"id": "bad", "input": "message", "output": {}, "extra": True}) + "\n",
            "unknown field(s)",
        ),
        ("\n", "line 2: blank lines are not allowed"),
    ],
)
def test_preflight_reports_safe_line_numbered_errors_without_external_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_line: str,
    expected_error: str,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    dataset.write_text(json.dumps(_record()) + "\n" + invalid_line, encoding="utf-8")

    def unexpected_settings() -> None:
        raise AssertionError("invalid data reached model setup")

    monkeypatch.setattr(main, "OpenRouterDatasetSettings", unexpected_settings)
    result = runner.invoke(root_app, ["dataset", "evaluate", str(dataset)])

    assert result.exit_code != 0
    normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", result.output).split())
    assert expected_error in normalized_output
    assert "Transfer 100" not in result.output


def test_preflight_rejects_duplicate_ids_before_external_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record(), _record()])

    def unexpected_settings() -> None:
        raise AssertionError("duplicate data reached model setup")

    monkeypatch.setattr(main, "OpenRouterDatasetSettings", unexpected_settings)
    result = runner.invoke(root_app, ["dataset", "evaluate", str(dataset)])

    assert result.exit_code != 0
    assert "line 2: duplicate id" in result.output


def test_preflight_rejects_deeply_nested_json(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    nested_output = "[" * 1_100 + "0" + "]" * 1_100
    dataset.write_text(
        f'{{"id":"deep","input":"message","output":{nested_output}}}\n',
        encoding="utf-8",
    )

    result = runner.invoke(root_app, ["dataset", "evaluate", str(dataset), "--dry-run"])

    assert result.exit_code != 0
    assert "line 1: invalid output" in result.output


def test_preflight_rejects_selected_model_input_over_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record(), _record("interaction-2")])
    monkeypatch.setattr(
        main,
        "OpenRouterDatasetSettings",
        lambda: _settings(max_input_chars=50),
    )

    result = runner.invoke(root_app, ["dataset", "evaluate", str(dataset), "--dry-run"])

    assert result.exit_code != 0
    assert "selected interaction 1 exceeds the semantic model input limit" in result.output


def test_preflight_enforces_record_and_target_call_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record(f"interaction-{index}") for index in range(101)])

    def unexpected_settings() -> None:
        raise AssertionError("oversized data reached model setup")

    monkeypatch.setattr(main, "OpenRouterDatasetSettings", unexpected_settings)
    too_many_records = runner.invoke(
        root_app,
        ["dataset", "evaluate", str(dataset), "--dry-run"],
    )

    assert too_many_records.exit_code != 0
    assert "line 101: dataset exceeds 100 records" in too_many_records.output

    monkeypatch.setattr(
        main,
        "OpenRouterDatasetSettings",
        lambda: SimpleNamespace(max_input_chars=50_000),
    )
    _write_dataset(dataset, [_record(f"interaction-{index}") for index in range(17)])
    maximum_calls = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--limit",
            "16",
            "--operator",
            "surface.rephrase",
            "--dry-run",
        ],
    )

    assert maximum_calls.exit_code == 0, maximum_calls.output
    assert "Potential target calls: up to 96" in maximum_calls.output

    too_many_calls = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--limit",
            "17",
            "--operator",
            "surface.rephrase",
            "--dry-run",
        ],
    )

    assert too_many_calls.exit_code != 0
    normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", too_many_calls.output).split())
    assert "would make up to 102 target calls" in normalized_output
    assert "--max-target-calls 100" in normalized_output


def test_repetition_budget_is_explicit_and_checked_before_external_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record()])

    def unexpected_settings() -> None:
        raise AssertionError("over-budget repetition plan reached model setup")

    monkeypatch.setattr(main, "OpenRouterDatasetSettings", unexpected_settings)
    huge_plan = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--repetitions",
            "1000000000",
            "--dry-run",
        ],
    )

    assert huge_plan.exit_code != 0
    normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", huge_plan.output).split())
    assert "would make up to" in normalized_output
    assert "--max-target-calls" in normalized_output
    assert "call budget" in normalized_output

    monkeypatch.setattr(
        main,
        "OpenRouterDatasetSettings",
        lambda: SimpleNamespace(max_input_chars=50_000),
    )
    exact_budget = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--repetitions",
            "51",
            "--max-target-calls",
            "102",
            "--dry-run",
        ],
    )

    assert exact_budget.exit_code == 0, exact_budget.output
    assert "Potential target calls: up to 102 (authorized maximum: 102)" in " ".join(
        exact_budget.output.split()
    )


@pytest.mark.parametrize(
    "options",
    (
        ("--repetitions", "0"),
        ("--repetitions", "-1"),
        ("--max-target-calls", "0"),
        ("--max-target-calls", "-1"),
    ),
)
def test_repetition_and_call_budget_must_be_positive(
    tmp_path: Path,
    options: tuple[str, str],
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record()])

    result = runner.invoke(
        root_app,
        ["dataset", "evaluate", str(dataset), *options, "--dry-run"],
    )

    assert result.exit_code != 0


def test_default_limit_and_repetitions_fit_the_default_call_budget(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record(f"interaction-{index}") for index in range(11)])

    result = runner.invoke(root_app, ["dataset", "evaluate", str(dataset), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Selected interactions: 10" in result.output
    assert "Potential target calls: up to 60" in result.output


@pytest.mark.parametrize(
    ("options", "expected_error"),
    [
        ([], "execution requires --target-url"),
        (["--target-url", "https://sandbox.example.test"], "--allow-target-network"),
        (
            [
                "--target-url",
                "https://sandbox.example.test",
                "--allow-target-network",
            ],
            "--confirm-isolated-sandbox",
        ),
        (
            [
                "--target-url",
                "https://sandbox.example.test",
                "--allow-target-network",
                "--confirm-isolated-sandbox",
            ],
            "--confirm-fresh-state",
        ),
        (
            [
                "--target-url",
                "https://sandbox.example.test",
                "--allow-target-network",
                "--confirm-isolated-sandbox",
                "--confirm-fresh-state",
            ],
            "execution requires --output",
        ),
    ],
)
def test_execution_requires_explicit_target_confirmations_and_output(
    tmp_path: Path,
    options: list[str],
    expected_error: str,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record()])

    result = runner.invoke(root_app, ["dataset", "evaluate", str(dataset), *options])

    assert result.exit_code != 0
    normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", result.output).split())
    assert expected_error in normalized_output


def test_execution_refuses_to_overwrite_output_before_model_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    _write_dataset(dataset, [_record()])
    output.write_text("keep me", encoding="utf-8")

    def unexpected_settings() -> None:
        raise AssertionError("output collision reached model setup")

    monkeypatch.setattr(main, "OpenRouterDatasetSettings", unexpected_settings)
    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target-url",
            "https://sandbox.example.test",
            "--allow-target-network",
            "--confirm-isolated-sandbox",
            "--confirm-fresh-state",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert "will not overwrite" in result.output
    assert output.read_text(encoding="utf-8") == "keep me"


def test_execution_rejects_missing_header_secret_before_model_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    _write_dataset(dataset, [_record()])
    monkeypatch.delenv("MISSING_SANDBOX_TOKEN", raising=False)
    monkeypatch.setattr(
        main,
        "OpenRouterDatasetSettings",
        _settings,
    )

    def unexpected_deconstructor(*args: object, **kwargs: object) -> None:
        raise AssertionError("missing target auth reached semantic model setup")

    monkeypatch.setattr(main, "OpenRouterSemanticDeconstructor", unexpected_deconstructor)
    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target-url",
            "https://sandbox.example.test",
            "--header-env",
            "Authorization=MISSING_SANDBOX_TOKEN",
            "--allow-target-network",
            "--confirm-isolated-sandbox",
            "--confirm-fresh-state",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert not output.exists()


def test_execution_creates_private_explicit_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    _write_dataset(dataset, [_record()])
    captured_records: list[str] = []

    class FakeTarget:
        def __init__(self, endpoint: str, **options: object) -> None:
            assert endpoint == "http://127.0.0.1:8765/execute"
            assert options["sandbox_confirmed"] is True
            assert options["fresh_state_confirmed"] is True
            assert options["request_field"] == "query"

    async def fake_evaluate(
        records: tuple[Any, ...],
        operator_ids: tuple[str, ...],
        settings: object,
        target: object,
        output_stream: Any,
        *,
        repetitions: int,
        max_target_calls: int,
        planned_target_calls: int,
        run_context: object,
    ) -> tuple[object, ...]:
        del settings, target, run_context
        captured_records.extend(record.id for record in records)
        assert operator_ids == ("surface.disfluency_repeat",)
        assert repetitions == 3
        assert max_target_calls == 100
        assert planned_target_calls == 6
        output_stream.write('{"saved":true}\n')
        output_stream.flush()
        return ()

    monkeypatch.setattr(
        main,
        "OpenRouterDatasetSettings",
        _settings,
    )
    monkeypatch.setattr(main, "JsonHttpDatasetTarget", FakeTarget)
    monkeypatch.setattr(main, "_evaluate_interaction_records", fake_evaluate)
    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--operator",
            "surface.disfluency_repeat",
            "--target-url",
            "http://127.0.0.1:8765/execute",
            "--request-field",
            "query",
            "--allow-insecure-http",
            "--allow-target-network",
            "--confirm-isolated-sandbox",
            "--confirm-fresh-state",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured_records == ["interaction-1"]
    assert output.read_text(encoding="utf-8") == '{"saved":true}\n'
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert "Complete evidence" in result.output
    assert "Next: ul dataset report" in result.output
    assert "Transfer 100" not in result.output


def test_target_config_runs_nested_request_and_response_against_loopback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received_requests: list[object] = []

    class TargetHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers["Content-Length"])
            received_requests.append(json.loads(self.rfile.read(content_length)))
            response = json.dumps(
                {
                    "envelope": {
                        "agent": {
                            "actions": [{"action": "transfer", "amount": 100, "recipient": "Alice"}]
                        }
                    }
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            pass

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
    except PermissionError:
        pytest.skip("the test sandbox does not allow binding a loopback server")
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    try:
        dataset = tmp_path / "interactions.jsonl"
        target_config = tmp_path / "target.json"
        output = tmp_path / "results.jsonl"
        _write_dataset(dataset, [_record()])
        _write_target_config(
            target_config,
            url=f"http://127.0.0.1:{server.server_port}/execute",
            request_json_template={
                "payload": {
                    "messages": [{"role": "user", "content": "{{input}}"}],
                }
            },
            response_json_pointer="/envelope/agent",
        )
        observed_outputs: list[object] = []

        async def evaluate_once(
            records: tuple[Any, ...],
            operator_ids: tuple[str, ...],
            settings: object,
            target: Any,
            output_stream: Any,
            *,
            repetitions: int,
            max_target_calls: int,
            planned_target_calls: int,
            run_context: object,
        ) -> tuple[object, ...]:
            del (
                operator_ids,
                settings,
                repetitions,
                max_target_calls,
                planned_target_calls,
                run_context,
            )
            async with target:
                observed_outputs.append((await target.execute(records[0].raw_input)).raw_output)
            output_stream.write('{"saved":true}\n')
            return ()

        monkeypatch.setattr(
            main,
            "OpenRouterDatasetSettings",
            _settings,
        )
        monkeypatch.setattr(main, "_evaluate_interaction_records", evaluate_once)

        result = runner.invoke(
            root_app,
            [
                "dataset",
                "evaluate",
                str(dataset),
                "--target-config",
                str(target_config),
                "--allow-insecure-http",
                "--allow-target-network",
                "--confirm-isolated-sandbox",
                "--confirm-fresh-state",
                "--output",
                str(output),
            ],
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()

    assert result.exit_code == 0, result.output
    assert received_requests == [
        {
            "payload": {
                "messages": [{"role": "user", "content": "Transfer 100 to Alice."}],
            }
        }
    ]
    assert observed_outputs == [
        {"actions": [{"action": "transfer", "amount": 100, "recipient": "Alice"}]}
    ]
    assert output.read_text(encoding="utf-8") == '{"saved":true}\n'


def test_help_explains_dataset_target_and_operator_contract() -> None:
    result = runner.invoke(root_app, ["dataset", "evaluate", "--help"])

    assert result.exit_code == 0, result.output
    assert '"id"' in result.output
    assert '"input"' in result.output
    assert '"output"' in result.output
    normalized_help = " ".join(_ANSI_ESCAPE_PATTERN.sub("", result.output).split())
    assert "Simple sandbox" in normalized_help
    assert "POST" in result.output
    assert "non-null JSON" in normalized_help
    assert "UL_LIVE" in result.output
    assert "same clean state" in normalized_help
    assert "Fresh-state" in normalized_help
    assert "target executions" in normalized_help
    assert "Maximum target" in normalized_help
    assert "requests" in normalized_help
    assert "operator. Run 'ul" in normalized_help
    assert "operators' for" in normalized_help
    assert "--target-config" in normalized_help
    assert "configuration" in normalized_help
    help_text = " ".join(normalized_help.replace("│", "").split())
    assert "created by 'ul dataset init'" in help_text

    init_help = runner.invoke(root_app, ["dataset", "init", "--help"])
    assert init_help.exit_code == 0, init_help.output
    normalized_init_help = " ".join(_ANSI_ESCAPE_PATTERN.sub("", init_help.output).split())
    assert "target_config" in normalized_init_help
    assert "--url" in normalized_init_help
    assert "private starter" in normalized_init_help
    assert "{{input}}" in normalized_init_help
    assert "/choices/0/message/content" in normalized_init_help

    operators = runner.invoke(root_app, ["dataset", "operators"])
    assert operators.exit_code == 0, operators.output
    assert "surface.disfluency_repeat" in operators.output
    assert "tone.frustrated" in operators.output
    assert "intent.self_correction" in operators.output


def test_operator_list_is_fixed_and_self_correction_keeps_existing_call_accounting(
    tmp_path: Path,
) -> None:
    operators = runner.invoke(root_app, ["dataset", "operators"])

    assert operators.exit_code == 0, operators.output
    listed_operator_ids = tuple(
        line.removeprefix("- ").split(" ", 1)[0]
        for line in operators.output.splitlines()
        if line.startswith("- ")
    )
    assert listed_operator_ids == (
        "surface.rephrase",
        "surface.typing_noise",
        "surface.fragmented_syntax",
        "surface.disfluency_repeat",
        "style.terse",
        "style.verbose",
        "tone.frustrated",
        "intent.self_correction",
    )

    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record()])
    dry_run = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--operator",
            "intent.self_correction",
            "--dry-run",
        ],
    )

    assert dry_run.exit_code == 0, dry_run.output
    assert "Operators: intent.self_correction" in dry_run.output
    assert "Potential semantic model calls: up to 10" in dry_run.output
    assert "Potential target calls: up to 6" in dry_run.output


def test_resume_skips_already_processed_interaction_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    _write_dataset(dataset, [_record("interaction-1"), _record("interaction-2")])
    evaluation_results = (
        _evaluation_result("interaction-1"),
        _evaluation_result("interaction-2"),
    )
    selected_records = tuple(result.source for result in evaluation_results)
    run_context = _run_context(selected_records)
    evidence.write_text(
        json.dumps(
            main._customer_evidence_record(
                evaluation_results[0],
                repetitions=1,
                max_target_calls=4,
                planned_target_calls=4,
                run_context=cast(Any, run_context),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    evaluated_ids: list[str] = []

    class FakeTarget:
        def __init__(self, endpoint: str, **options: object) -> None:
            pass

    async def fake_evaluate(
        records: tuple[Any, ...],
        operator_ids: tuple[str, ...],
        settings: object,
        target: object,
        output_stream: Any,
        *,
        repetitions: int,
        max_target_calls: int,
        planned_target_calls: int,
        run_context: object,
    ) -> tuple[object, ...]:
        del operator_ids, settings, target, max_target_calls, planned_target_calls
        assert repetitions == 1
        for record in records:
            evaluated_ids.append(record.id)
        output_stream.write(
            json.dumps(
                main._customer_evidence_record(
                    evaluation_results[1],
                    repetitions=1,
                    max_target_calls=4,
                    planned_target_calls=4,
                    run_context=cast(Any, run_context),
                )
            )
            + "\n"
        )
        output_stream.flush()
        return ()

    monkeypatch.setattr(
        main,
        "OpenRouterDatasetSettings",
        _settings,
    )
    monkeypatch.setattr(main, "JsonHttpDatasetTarget", FakeTarget)
    monkeypatch.setattr(main, "_evaluate_interaction_records", fake_evaluate)
    command = [
        "dataset",
        "evaluate",
        str(dataset),
        "--target-url",
        "https://sandbox.example.test/execute",
        "--allow-target-network",
        "--confirm-isolated-sandbox",
        "--confirm-fresh-state",
        "--repetitions",
        "1",
        "--resume",
        str(evidence),
    ]
    dry_run = runner.invoke(root_app, [*command, "--dry-run"])

    assert dry_run.exit_code == 0, dry_run.output
    assert "Resume compatible: 1 complete interaction(s) skipped; 1 remaining" in dry_run.output
    assert "Evidence destination:" in dry_run.output
    assert evidence.name in dry_run.output

    result = runner.invoke(root_app, command)

    assert result.exit_code == 0, result.output
    assert evaluated_ids == ["interaction-2"]
    lines = [line for line in evidence.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["interaction_id"] == "interaction-1"
    assert json.loads(lines[1])["interaction_id"] == "interaction-2"
    assert "skipped" in result.output
    if sys.platform != "win32":
        assert stat.S_IMODE(evidence.stat().st_mode) == 0o600


def test_resume_exits_early_when_all_records_already_processed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    _write_dataset(dataset, [_record("interaction-1")])
    evaluation_result = _evaluation_result("interaction-1")
    run_context = _run_context((evaluation_result.source,))
    evidence.write_text(
        json.dumps(
            main._customer_evidence_record(
                evaluation_result,
                repetitions=1,
                max_target_calls=2,
                planned_target_calls=2,
                run_context=cast(Any, run_context),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "OpenRouterDatasetSettings", _settings)

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target-url",
            "https://sandbox.example.test/execute",
            "--allow-target-network",
            "--confirm-isolated-sandbox",
            "--confirm-fresh-state",
            "--repetitions",
            "1",
            "--resume",
            str(evidence),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Nothing to do" in result.output


def test_all_complete_resume_preserves_prior_review_finding_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    _write_dataset(dataset, [_record()])
    evaluation_result = _evaluation_result("interaction-1", has_review_finding=True)
    run_context = _run_context((evaluation_result.source,))
    evidence.write_text(
        json.dumps(
            main._customer_evidence_record(
                evaluation_result,
                repetitions=1,
                max_target_calls=2,
                planned_target_calls=2,
                run_context=cast(Any, run_context),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "OpenRouterDatasetSettings", _settings)

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target-url",
            "https://sandbox.example.test/execute",
            "--repetitions",
            "1",
            "--resume",
            str(evidence),
        ],
    )

    assert result.exit_code == 1, result.output
    assert "Nothing to do" in result.output


@pytest.mark.parametrize(
    ("invariant_status", "expected_exit_code"),
    [("violated", 1), ("not_evaluable", 2)],
)
def test_all_complete_resume_preserves_prior_invariant_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invariant_status: str,
    expected_exit_code: int,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    invariant_path = tmp_path / "invariants.json"
    _write_dataset(dataset, [_record()])
    _write_invariant_suite(invariant_path)
    invariant_suite = main.load_dataset_invariant_suite(invariant_path)
    evaluation_result = _evaluation_result("interaction-1")
    baseline_output = (
        {"final_amount": 200, "corrected_amount": 100}
        if invariant_status == "violated"
        else {"status": "ok"}
    )
    baseline_trial = evaluation_result.baseline.trial_set.trials[0].model_copy(
        update={"target_output": ObservedAgentOutput(raw_output=baseline_output)}
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
    invariant_evaluation = main.evaluate_dataset_invariants(evaluation_result, invariant_suite)
    assert invariant_evaluation.baseline.rules[0].status == invariant_status
    run_context = _run_context(
        (evaluation_result.source,),
        invariant_suite=invariant_suite,
    )
    evidence.write_text(
        json.dumps(
            main._customer_evidence_record(
                evaluation_result,
                repetitions=1,
                max_target_calls=2,
                planned_target_calls=2,
                run_context=cast(Any, run_context),
                invariant_evaluation=invariant_evaluation,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "OpenRouterDatasetSettings", _settings)

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target-url",
            "https://sandbox.example.test/execute",
            "--invariants",
            str(invariant_path),
            "--repetitions",
            "1",
            "--resume",
            str(evidence),
        ],
    )

    assert result.exit_code == expected_exit_code, result.output
    assert "Nothing to do" in result.output


def test_resume_rejects_forged_invariant_outcome(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    invariant_path = tmp_path / "invariants.json"
    _write_dataset(dataset, [_record()])
    _write_invariant_suite(invariant_path)
    invariant_suite = main.load_dataset_invariant_suite(invariant_path)
    evaluation_result = _evaluation_result("interaction-1")
    run_context = _run_context(
        (evaluation_result.source,),
        invariant_suite=invariant_suite,
    )
    evidence.write_text(
        json.dumps(
            main._customer_evidence_record(
                evaluation_result,
                repetitions=1,
                max_target_calls=2,
                planned_target_calls=2,
                run_context=cast(Any, run_context),
                invariant_evaluation=_invariant_evaluation(
                    "violated",
                    interaction_id="interaction-1",
                    suite_sha256=invariant_suite.sha256,
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target-url",
            "https://sandbox.example.test/execute",
            "--invariants",
            str(invariant_path),
            "--repetitions",
            "1",
            "--resume",
            str(evidence),
            "--dry-run",
        ],
    )

    assert result.exit_code == 2
    assert "cannot safely resume evidence" in result.output


def test_resume_snapshot_detects_same_summary_content_change() -> None:
    evaluation_result = _evaluation_result("interaction-1")
    run_context = _run_context((evaluation_result.source,))

    def validated_snapshot(max_target_calls: int) -> main.DatasetResumeEvidence:
        raw_evidence = (
            json.dumps(
                main._customer_evidence_record(
                    evaluation_result,
                    repetitions=1,
                    max_target_calls=max_target_calls,
                    planned_target_calls=2,
                    run_context=cast(Any, run_context),
                )
            )
            + "\n"
        ).encode()
        return main.validate_dataset_resume_evidence(
            raw_evidence,
            expected_context=cast(Any, run_context),
            selected_records=(evaluation_result.source,),
            invariant_suite=None,
            evidence_projector=main._customer_evidence_record,
        )

    first_snapshot = validated_snapshot(2)
    changed_snapshot = validated_snapshot(3)

    assert first_snapshot.processed_ids == changed_snapshot.processed_ids
    assert first_snapshot.has_review_findings == changed_snapshot.has_review_findings
    assert first_snapshot.raw_evidence_sha256 != changed_snapshot.raw_evidence_sha256
    assert first_snapshot != changed_snapshot


def test_resume_accepts_extended_invariant_evidence_schema() -> None:
    evaluation_result = _evaluation_result("interaction-1")
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
                literal="current",
            ),
        ),
    )
    baseline_trial = evaluation_result.baseline.trial_set.trials[0].model_copy(
        update={"target_output": ObservedAgentOutput(raw_output={"approval": "current"})}
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
    invariant_evaluation = main.evaluate_dataset_invariants(evaluation_result, suite)
    run_context = _run_context((evaluation_result.source,), invariant_suite=suite)
    raw_evidence = (
        json.dumps(
            main._customer_evidence_record(
                evaluation_result,
                repetitions=1,
                max_target_calls=2,
                planned_target_calls=2,
                run_context=cast(Any, run_context),
                invariant_evaluation=invariant_evaluation,
            )
        )
        + "\n"
    ).encode()

    snapshot = main.validate_dataset_resume_evidence(
        raw_evidence,
        expected_context=cast(Any, run_context),
        selected_records=(evaluation_result.source,),
        invariant_suite=suite,
        evidence_projector=main._customer_evidence_record,
    )

    assert snapshot.processed_ids == frozenset({"interaction-1"})
    assert snapshot.invariant_evaluations == (invariant_evaluation,)


def test_resume_rejects_legacy_unbound_evidence(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    _write_dataset(dataset, [_record()])
    evidence.write_text(
        json.dumps({"schema_version": "1.4.0", "interaction_id": "interaction-1"}) + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target-url",
            "https://sandbox.example.test/execute",
            "--repetitions",
            "1",
            "--resume",
            str(evidence),
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "cannot safely resume evidence" in result.output


def test_resume_rejects_changed_evaluation_plan(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    _write_dataset(dataset, [_record()])
    evaluation_result = _evaluation_result("interaction-1")
    run_context = _run_context((evaluation_result.source,))
    evidence.write_text(
        json.dumps(
            main._customer_evidence_record(
                evaluation_result,
                repetitions=1,
                max_target_calls=2,
                planned_target_calls=2,
                run_context=cast(Any, run_context),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target-url",
            "https://sandbox.example.test/execute",
            "--operator",
            "tone.frustrated",
            "--repetitions",
            "1",
            "--resume",
            str(evidence),
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "incompatible with the current evaluation plan" in result.output


def test_resume_rejects_evidence_without_terminal_newline(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    _write_dataset(dataset, [_record()])
    evaluation_result = _evaluation_result("interaction-1")
    run_context = _run_context((evaluation_result.source,))
    evidence.write_text(
        json.dumps(
            main._customer_evidence_record(
                evaluation_result,
                repetitions=1,
                max_target_calls=2,
                planned_target_calls=2,
                run_context=cast(Any, run_context),
            )
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target-url",
            "https://sandbox.example.test/execute",
            "--repetitions",
            "1",
            "--resume",
            str(evidence),
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "must end with a newline" in result.output


@pytest.mark.skipif(sys.platform == "win32", reason="Unix permission semantics")
def test_resume_dry_run_accepts_read_only_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    _write_dataset(dataset, [_record()])
    evaluation_result = _evaluation_result("interaction-1")
    run_context = _run_context((evaluation_result.source,))
    evidence.write_text(
        json.dumps(
            main._customer_evidence_record(
                evaluation_result,
                repetitions=1,
                max_target_calls=2,
                planned_target_calls=2,
                run_context=cast(Any, run_context),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    evidence.chmod(0o400)
    monkeypatch.setattr(main, "OpenRouterDatasetSettings", _settings)

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target-url",
            "https://sandbox.example.test/execute",
            "--repetitions",
            "1",
            "--resume",
            str(evidence),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert stat.S_IMODE(evidence.stat().st_mode) == 0o400


def test_resume_rejects_mismatched_output_path(
    tmp_path: Path,
) -> None:
    """--resume and --output must point to the same file when both are given."""
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    other_output = tmp_path / "other.jsonl"

    _write_dataset(dataset, [_record()])
    evidence.write_text(
        json.dumps({"schema_version": "1.4.0", "interaction_id": "x", "cases": []}) + "\n",
        encoding="utf-8",
    )
    other_output.write_text("", encoding="utf-8")

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target-url",
            "https://sandbox.example.test/execute",
            "--allow-target-network",
            "--confirm-isolated-sandbox",
            "--confirm-fresh-state",
            "--resume",
            str(evidence),
            "--output",
            str(other_output),
        ],
    )

    assert result.exit_code != 0
    assert "same file" in result.output


@pytest.mark.parametrize(
    "operators",
    [
        ("intent.self_correction", "intent.self_correction"),
        ("intent.self-correction",),
    ],
)
def test_cli_rejects_duplicate_or_unknown_self_correction_operator_before_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operators: tuple[str, ...],
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record()])

    def unexpected_settings() -> None:
        raise AssertionError("invalid operator selection reached model setup")

    monkeypatch.setattr(main, "OpenRouterDatasetSettings", unexpected_settings)
    arguments = ["dataset", "evaluate", str(dataset)]
    for operator_id in operators:
        arguments.extend(("--operator", operator_id))

    result = runner.invoke(root_app, arguments)

    assert result.exit_code != 0
    assert "operator" in result.output.casefold()


def test_customer_evidence_keeps_summary_and_nested_technical_details() -> None:
    expected_effect = SimpleNamespace(
        id="effect-1",
        evidence=(),
        confidence=0.9,
        status="completed",
        request_unit_ids=("request-1",),
        position=0,
        kind="action",
        predicate="transfer",
        fields={"amount": 100},
        propositions=(),
        model_dump=lambda **kwargs: {"kind": "action", "predicate": "transfer"},
    )
    finding = SimpleNamespace(
        category="duplicate_effect",
        message="A duplicate action needs review.",
        expected_effects=(expected_effect,),
        observed_effects=(expected_effect, expected_effect),
        grounded_field_names=("amount",),
    )
    candidate = SimpleNamespace(
        operator_id="surface.disfluency_repeat",
        operator_version="1.0.0",
        augmented_input="transfer transfer 100 to Alice",
        passed=True,
        failure_reasons=(),
    )
    case = SimpleNamespace(
        candidate=candidate,
        verdict="divergence_needs_review",
        trial_set=_trial_set(representative_effect=expected_effect),
        findings=(finding,),
        inconclusive_reasons=(),
    )
    result = cast(
        DatasetEvaluationResult,
        SimpleNamespace(
            source=SimpleNamespace(id="case-1", raw_input="transfer 100 to Alice"),
            baseline=SimpleNamespace(
                verdict="no_divergence",
                trial_set=_trial_set(representative_effect=expected_effect),
                inconclusive_reasons=(),
            ),
            cases=(case,),
            model_dump=lambda **kwargs: {"full": "technical evidence"},
        ),
    )

    evidence = main._customer_evidence_record(
        result,
        repetitions=3,
        max_target_calls=100,
        planned_target_calls=6,
    )

    assert main._result_needs_review(result) is True
    assert evidence["interaction_id"] == "case-1"
    assert evidence["original_input"] == "transfer 100 to Alice"
    assert evidence["schema_version"] == "1.4.0"
    assert evidence["invariant_evaluation"] is None
    assert evidence["current_baseline"]["status"] == "ORIGINAL REPLAY STABLE (3/3 OBSERVED)"
    assert "findings" not in evidence["current_baseline"]
    assert evidence["current_baseline"]["observations"]["outcome_group_count"] == 1
    assert evidence["current_baseline"]["observations"]["outcome_groups"][0]["repetitions"] == [
        1,
        2,
        3,
    ]
    assert evidence["current_baseline"]["observations"]["outcome_groups"][0]["count"] == 3
    assert evidence["current_baseline"]["observations"]["observed_repetitions"] == 3
    assert evidence["current_baseline"]["observations"]["inconclusive_repetitions"] == 0
    assert evidence["cases"][0]["status"] == "REPEATABLE DIFFERENCE — REVIEW"
    assert evidence["cases"][0]["findings"][0]["reference_effects"] == [
        {"kind": "action", "predicate": "transfer"}
    ]
    assert evidence["cases"][0]["findings"][0]["finding_id"] == (
        "ulf_v1_963abb59403f4bb8ec4e9bb81c8eb38ddc693dc5a28153569723a484baac4625"
    )
    assert evidence["cases"][0]["findings"][0]["grounded_field_names"] == ["amount"]
    assert evidence["cases"][0]["findings"][0]["severity"] == "unrated"
    assert evidence["cases"][0]["findings"][0]["review_status"] == "needs_review"
    assert evidence["execution_plan"] == {
        "repetitions": 3,
        "max_target_calls": 100,
        "dataset_planned_target_calls": 6,
    }
    assert "does not determine" in evidence["limitations"]
    assert "caused" in evidence["limitations"]
    assert "production failure rate" in evidence["limitations"]
    assert evidence["technical_details"] == {"full": "technical evidence"}


def test_customer_evidence_keeps_invariants_separate_from_behavioral_findings() -> None:
    result = cast(
        DatasetEvaluationResult,
        SimpleNamespace(
            source=SimpleNamespace(id="case-1", raw_input="Correct amount to 100."),
            baseline=SimpleNamespace(
                trial_set=_trial_set(requested_repetitions=1),
                inconclusive_reasons=(),
            ),
            cases=(),
            model_dump=lambda **kwargs: {"technical": "behavioral evidence"},
        ),
    )
    invariant_evaluation = _invariant_evaluation("satisfied", "violated")

    evidence = main._customer_evidence_record(
        result,
        repetitions=1,
        max_target_calls=2,
        planned_target_calls=2,
        invariant_evaluation=invariant_evaluation,
    )

    assert evidence["schema_version"] == "1.4.0"
    assert evidence["cases"] == []
    stored_invariants = cast(dict[str, Any], evidence["invariant_evaluation"])
    assert stored_invariants["baseline"]["rules"][0]["status"] == "satisfied"
    assert stored_invariants["variations"][0]["rules"][0]["status"] == "violated"
    assert evidence["technical_details"] == {"technical": "behavioral evidence"}


@pytest.mark.parametrize(
    ("evaluations", "expected_exit_code"),
    [
        ((), 0),
        ((_invariant_evaluation("satisfied"),), 0),
        ((_invariant_evaluation("not_evaluable"),), 2),
        ((_invariant_evaluation("satisfied", "not_evaluable"),), 2),
        ((_invariant_evaluation("violated"),), 1),
        ((_invariant_evaluation("not_evaluable", "violated"),), 1),
    ],
)
def test_invariant_exit_code_precedence(
    evaluations: tuple[DatasetInvariantEvaluation, ...], expected_exit_code: int
) -> None:
    assert main._invariant_exit_code(evaluations) == expected_exit_code


def test_finding_id_ignores_volatile_evidence_and_semantic_ordering() -> None:
    first_effect = SimpleNamespace(
        id="generated-effect-1",
        evidence=(SimpleNamespace(json_pointer="/actions/0"),),
        confidence=0.72,
        status="failed",
        request_unit_ids=("generated-request-1",),
        position=0,
        kind="action",
        predicate="transfer",
        fields={"recipient": "Alice", "details": {"currency": "USD", "amount": 100}},
        propositions=("authorized", "settled"),
    )
    same_effect_with_volatile_changes = SimpleNamespace(
        id="generated-effect-99",
        evidence=(SimpleNamespace(json_pointer="/tool_calls/4"),),
        confidence=0.99,
        status="completed",
        request_unit_ids=("generated-request-42",),
        position=8,
        kind="action",
        predicate="transfer",
        fields={"details": {"amount": 100, "currency": "USD"}, "recipient": "Alice"},
        propositions=("settled", "authorized"),
    )
    second_effect = SimpleNamespace(
        id="generated-effect-2",
        evidence=(),
        confidence=0.8,
        status="completed",
        request_unit_ids=(),
        position=1,
        kind="action",
        predicate="notify",
        fields={"recipient": "Alice"},
        propositions=(),
    )
    reordered_second_effect = SimpleNamespace(**vars(second_effect))
    finding = cast(
        Any,
        SimpleNamespace(
            category="changed_grounded_effect_argument",
            grounded_field_names=("recipient", "amount"),
            expected_effects=(first_effect, second_effect),
            observed_effects=(second_effect, first_effect),
        ),
    )
    semantically_identical_finding = cast(
        Any,
        SimpleNamespace(
            category="changed_grounded_effect_argument",
            grounded_field_names=("amount", "recipient"),
            expected_effects=(reordered_second_effect, same_effect_with_volatile_changes),
            observed_effects=(same_effect_with_volatile_changes, reordered_second_effect),
        ),
    )

    finding_id = main._finding_id(
        interaction_id="case-1",
        original_input="Transfer 100 to Alice.",
        operator_id="surface.rephrase",
        operator_version="1.0.0",
        augmented_input="Please transfer 100 to Alice.",
        finding=finding,
    )
    identical_finding_id = main._finding_id(
        interaction_id="case-1",
        original_input="Transfer 100 to Alice.",
        operator_id="surface.rephrase",
        operator_version="1.0.0",
        augmented_input="Please transfer 100 to Alice.",
        finding=semantically_identical_finding,
    )

    assert finding_id == identical_finding_id
    assert re.fullmatch(r"ulf_v1_[0-9a-f]{64}", finding_id)


def test_finding_id_changes_for_meaningful_variation_or_behavior() -> None:
    reference_effect = SimpleNamespace(
        status="completed",
        kind="action",
        predicate="transfer",
        fields={"amount": 100, "recipient": "Alice"},
        propositions=(),
    )
    changed_effect = SimpleNamespace(
        status="completed",
        kind="action",
        predicate="transfer",
        fields={"amount": 200, "recipient": "Alice"},
        propositions=(),
    )
    finding = cast(
        Any,
        SimpleNamespace(
            category="changed_grounded_effect_argument",
            grounded_field_names=("amount",),
            expected_effects=(reference_effect,),
            observed_effects=(changed_effect,),
        ),
    )

    finding_id = main._finding_id(
        interaction_id="case-1",
        original_input="Transfer 100 to Alice.",
        operator_id="surface.rephrase",
        operator_version="1.0.0",
        augmented_input="Please transfer 100 to Alice.",
        finding=finding,
    )
    changed_variation_id = main._finding_id(
        interaction_id="case-1",
        original_input="Transfer 100 to Alice.",
        operator_id="surface.rephrase",
        operator_version="1.0.0",
        augmented_input="Could you transfer 100 to Alice?",
        finding=finding,
    )
    changed_behavior = cast(
        Any,
        SimpleNamespace(
            category="changed_grounded_effect_argument",
            grounded_field_names=("amount",),
            expected_effects=(reference_effect,),
            observed_effects=(
                SimpleNamespace(
                    status="completed",
                    kind="action",
                    predicate="transfer",
                    fields={"amount": 300, "recipient": "Alice"},
                    propositions=(),
                ),
            ),
        ),
    )
    changed_behavior_id = main._finding_id(
        interaction_id="case-1",
        original_input="Transfer 100 to Alice.",
        operator_id="surface.rephrase",
        operator_version="1.0.0",
        augmented_input="Please transfer 100 to Alice.",
        finding=changed_behavior,
    )

    assert finding_id != changed_variation_id
    assert finding_id != changed_behavior_id


def test_duplicate_semantic_findings_get_stable_unique_reportable_ids(tmp_path: Path) -> None:
    def effect(identifier: str, amount: int, position: int) -> SimpleNamespace:
        payload = {
            "id": identifier,
            "evidence": [],
            "confidence": 0.9,
            "status": "observed",
            "request_unit_ids": [],
            "position": position,
            "kind": "action",
            "predicate": "transfer",
            "fields": {"amount": amount, "recipient": "Alice"},
            "propositions": [],
        }
        return SimpleNamespace(
            **{**payload, "propositions": ()},
            model_dump=lambda **kwargs: payload,
        )

    first_reference = effect("reference-1", 100, 0)
    second_reference = effect("reference-2", 100, 1)
    first_observed = effect("observed-1", 200, 0)
    second_observed = effect("observed-2", 200, 1)
    first_finding = SimpleNamespace(
        category="changed_grounded_effect_argument",
        message="The variation changed a grounded transfer amount.",
        expected_effects=(first_reference,),
        observed_effects=(first_observed,),
        grounded_field_names=("amount",),
    )
    second_finding = SimpleNamespace(
        category="changed_grounded_effect_argument",
        message="The variation changed a grounded transfer amount.",
        expected_effects=(second_reference,),
        observed_effects=(second_observed,),
        grounded_field_names=("amount",),
    )
    finding_context = {
        "interaction_id": "case-1",
        "original_input": "Transfer 100 to Alice.",
        "operator_id": "surface.rephrase",
        "operator_version": "1.0.0",
        "augmented_input": "Please transfer 100 to Alice.",
    }

    customer_findings = main._customer_findings(
        cast(Any, (first_finding, second_finding)),
        **finding_context,
    )
    reordered_customer_findings = main._customer_findings(
        cast(Any, (second_finding, first_finding)),
        **finding_context,
    )
    finding_ids = {cast(dict[str, Any], finding)["finding_id"] for finding in customer_findings}
    reordered_finding_ids = {
        cast(dict[str, Any], finding)["finding_id"] for finding in reordered_customer_findings
    }

    assert len(finding_ids) == 2
    assert finding_ids == reordered_finding_ids
    assert all(re.fullmatch(r"ulf_v1_[0-9a-f]{64}", finding_id) for finding_id in finding_ids)

    candidate = SimpleNamespace(
        operator_id="surface.rephrase",
        operator_version="1.0.0",
        augmented_input="Please transfer 100 to Alice.",
        passed=True,
        failure_reasons=(),
    )
    case = SimpleNamespace(
        candidate=candidate,
        verdict="divergence_needs_review",
        trial_set=_trial_set(representative_effect=first_observed),
        findings=(first_finding, second_finding),
        inconclusive_reasons=(),
    )
    result = cast(
        DatasetEvaluationResult,
        SimpleNamespace(
            source=SimpleNamespace(id="case-1", raw_input="Transfer 100 to Alice."),
            baseline=SimpleNamespace(
                verdict="no_divergence",
                trial_set=_trial_set(representative_effect=first_reference),
                inconclusive_reasons=(),
            ),
            cases=(case,),
            model_dump=lambda **kwargs: {"fixture": "duplicate semantic findings"},
        ),
    )
    evidence = main._customer_evidence_record(
        result,
        repetitions=3,
        max_target_calls=6,
        planned_target_calls=6,
    )
    evidence_path = tmp_path / "evidence.jsonl"
    evidence_path.write_text(json.dumps(evidence) + "\n", encoding="utf-8")

    report = runner.invoke(root_app, ["dataset", "report", str(evidence_path)])

    assert report.exit_code == 0, report.output
    assert "Dataset finding report: 2 finding(s)" in report.output


def test_stored_output_drift_does_not_require_review_or_appear_in_original_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finding = SimpleNamespace(
        category="changed_grounded_effect_argument",
        message="The live control changed an action value.",
        expected_effects=(),
        observed_effects=(),
    )
    candidate = SimpleNamespace(
        operator_id="surface.rephrase",
        operator_version="1.0.0",
        augmented_input="Please transfer 100 to Alice.",
        passed=True,
        failure_reasons=(),
    )
    case = SimpleNamespace(
        candidate=candidate,
        verdict="no_divergence",
        trial_set=_trial_set(),
        findings=(),
        inconclusive_reasons=(),
    )
    result = cast(
        DatasetEvaluationResult,
        SimpleNamespace(
            source=SimpleNamespace(id="case-1", raw_input="transfer 100 to Alice"),
            baseline=SimpleNamespace(
                verdict="no_divergence",
                trial_set=_trial_set(),
                findings=(finding,),
                inconclusive_reasons=(),
            ),
            cases=(case,),
        ),
    )
    printed_rows: list[tuple[str, ...]] = []

    class CapturingTable:
        def add_column(self, *args: object, **kwargs: object) -> None:
            pass

        def add_row(self, *values: str) -> None:
            printed_rows.append(values)

    monkeypatch.setattr(main, "Table", lambda **kwargs: CapturingTable())
    monkeypatch.setattr(main.console, "print", lambda *args, **kwargs: None)

    main._print_dataset_results((result,), tmp_path / "evidence.jsonl")

    assert main._result_needs_review(result) is False
    assert printed_rows == [
        (
            "1",
            "original replay",
            "ORIGINAL REPLAY STABLE (3/3 OBSERVED)",
            "stable",
            "3 / 1",
            "—",
        ),
        (
            "2",
            "surface.rephrase",
            "NO OBSERVED DIFFERENCE",
            "stable",
            "3 / 1",
            "—",
        ),
    ]


def test_customer_statuses_distinguish_potential_repeatable_and_unstable_results() -> None:
    baseline = SimpleNamespace(
        verdict="no_divergence",
        trial_set=_trial_set(),
        inconclusive_reasons=(),
    )
    result = cast(DatasetEvaluationResult, SimpleNamespace(baseline=baseline))

    one_trial_difference = SimpleNamespace(
        verdict="divergence_needs_review",
        trial_set=_trial_set(requested_repetitions=1),
    )
    repeated_difference = SimpleNamespace(
        verdict="divergence_needs_review",
        trial_set=_trial_set(requested_repetitions=3),
    )
    unstable_variation = SimpleNamespace(
        verdict="divergence_needs_review",
        trial_set=_trial_set(
            stability="unstable",
            outcome_group_repetitions=((1, 2), (3,)),
        ),
    )

    assert main._case_customer_status(result, one_trial_difference) == (
        "POTENTIAL DIFFERENCE — REVIEW"
    )
    assert main._case_customer_status(result, repeated_difference) == (
        "REPEATABLE DIFFERENCE — REVIEW"
    )
    assert main._case_customer_status(result, unstable_variation) == ("UNSTABLE VARIATION — REVIEW")

    unstable_original_result = cast(
        DatasetEvaluationResult,
        SimpleNamespace(
            baseline=SimpleNamespace(
                verdict="inconclusive",
                trial_set=_trial_set(
                    stability="unstable",
                    outcome_group_repetitions=((1, 2), (3,)),
                ),
                inconclusive_reasons=("original repetitions produced multiple outcomes",),
            )
        ),
    )
    assert (
        main._baseline_customer_status(unstable_original_result)
        == "UNSTABLE ORIGINAL — INCONCLUSIVE"
    )
    assert main._case_customer_status(unstable_original_result, repeated_difference) == (
        "UNSTABLE ORIGINAL — INCONCLUSIVE"
    )

    assert main._case_customer_status(unstable_original_result, unstable_variation) == (
        "UNSTABLE ORIGINAL AND VARIATION — INCONCLUSIVE"
    )
    incomplete_variation = SimpleNamespace(
        verdict="inconclusive",
        trial_set=_trial_set(
            stability="inconclusive",
            outcome_group_repetitions=((1, 2),),
        ),
    )
    assert main._case_customer_status(unstable_original_result, incomplete_variation) == (
        "COULDN'T DETERMINE"
    )
