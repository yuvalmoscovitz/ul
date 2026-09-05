from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner
from ul_cli import dataset_campaign as campaign_module
from ul_cli.dataset.evaluation import execution as execution_module
from ul_cli.dataset.evaluation import preparation as preparation_module
from ul_cli.dataset.evaluation import runner as runner_module
from ul_cli.main import app as root_app

from ._factories import (
    _evaluation_result,
    _run_config,
    _settings,
)
from ._files import (
    _record,
    _write_dataset,
    _write_stateful_target_config,
    _write_target_config,
)

runner = CliRunner()
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def test_dry_run_validates_and_makes_no_external_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record(), _record("interaction-2")])
    _write_target_config(target_config)

    def unexpected_deconstructor(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run constructed a semantic model client")

    def unexpected_target(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run constructed a target client")

    monkeypatch.setattr(
        runner_module, "create_semantic_model_deconstructor", unexpected_deconstructor
    )
    monkeypatch.setattr(
        execution_module.JsonHttpEnvironmentConnection, "from_config", unexpected_target
    )
    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--operator",
            "input.surface.disfluency_repeat",
            "--limit",
            "1",
            "--environment-config",
            str(target_config),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Dataset valid: 2 interaction(s)" in result.output
    assert "Selected interactions: 1" in result.output
    assert "Evaluation mode: variance" in result.output
    assert "Repetitions: 3 per original and accepted variation" in result.output
    assert "Potential semantic model calls: up to 16" in result.output
    assert "Potential environment API calls: up to 30" in result.output
    assert "authorized maximum: 100" in result.output
    assert "Semantic models receive historical inputs and outputs" in result.output
    assert "generated variations" in result.output
    assert "live control responses" in result.output
    assert (
        "Every test case invokes and validates the configured environment reset contract"
        in result.output
    )
    assert "stateful target has no fixture identity" in result.output
    assert "do not determine correctness" in result.output
    assert "identify causality" in result.output
    assert "estimate a production failure rate" in result.output
    assert "No model or environment API requests sent." in result.output
    assert "Transfer 100" not in result.output


@pytest.mark.parametrize("timeout", ("0", "nan", "3601"))
def test_dry_run_rejects_unbounded_target_timeout(
    tmp_path: Path,
    timeout: str,
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
            "--environment-config",
            str(target_config),
            "--target-timeout-seconds",
            timeout,
            "--dry-run",
        ],
    )

    assert result.exit_code == 2
    normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", result.output).split())
    assert "target-timeout-seconds" in normalized_output


def test_dry_run_json_exposes_per_example_campaign_and_exact_call_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config)

    def unexpected_deconstructor(*args: object, **kwargs: object) -> None:
        raise AssertionError("campaign planning constructed a semantic model client")

    def unexpected_target(*args: object, **kwargs: object) -> None:
        raise AssertionError("campaign planning constructed a target client")

    monkeypatch.setattr(
        runner_module, "create_semantic_model_deconstructor", unexpected_deconstructor
    )
    monkeypatch.setattr(
        execution_module.JsonHttpEnvironmentConnection, "from_config", unexpected_target
    )
    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--operator",
            "input.surface.disfluency_repeat",
            "--environment-config",
            str(target_config),
            "--repetitions",
            "2",
            "--target-timeout-seconds",
            "75",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == "1.4.0"
    assert payload["evaluation_mode"] == "variance"
    assert payload["fixture"] == {"status": "missing", "id": None, "version": None}
    assert any("no fixture identity" in warning.casefold() for warning in payload["warnings"])
    assert payload["inspection_model_calls"] == 0
    assert payload["inspection_environment_calls"] == 0
    assert payload["calls"] == {
        "basis": "authorized_maximum",
        "baseline": 2,
        "variation": 2,
        "repetitions": 2,
        "repetition_executions": 4,
        "retries": 1,
        "preflight": 4,
        "evaluators": 7,
        "materiality": 1,
        "variation_generation": 1,
        "total_semantic_model": 14,
        "total_environment_api": 20,
    }
    assert payload["calls"]["preflight"] == len(payload["preflight_profiles"])
    assert [profile["roles"] for profile in payload["preflight_profiles"]] == [
        ["deconstruct"],
        ["render"],
        ["equivalence"],
        ["materiality"],
    ]
    assert (
        sum(profile["max_completion_tokens"] for profile in payload["preflight_profiles"]) == 3_072
    )
    assert payload["calls"]["total_semantic_model"] == (
        payload["calls"]["preflight"]
        + payload["calls"]["evaluators"]
        + payload["calls"]["materiality"]
        + payload["calls"]["variation_generation"]
        + payload["calls"]["retries"]
    )
    assert payload["timing"] == {
        "target_trial_timeout_seconds": 75.0,
        "target_request_concurrency": 1,
        "maximum_wall_time_seconds": 1140.0,
    }
    planned_operator = next(
        operator
        for operator in payload["examples"][0]["operators"]
        if operator["id"] == "input.surface.disfluency_repeat"
    )
    assert planned_operator["status"] == "conditional"
    assert planned_operator["selected"] is True
    assert "candidate generation requires a semantic model call" in planned_operator["reasons"]
    assert payload["tokens"] == {
        "minimum": 0,
        "maximum": 54_784,
        "scope": "completion_tokens",
    }
    assert payload["money"] is None


def test_sensitive_candidate_output_requires_dry_run(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record()])

    result = runner.invoke(
        root_app,
        ["dataset", "evaluate", str(dataset), "--show-sensitive-values"],
    )

    assert result.exit_code == 2
    normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", result.output).split())
    assert "--show-sensitive-values requires" in normalized_output
    assert "--dry-run" in normalized_output


@pytest.mark.parametrize(
    ("status", "fixture_id", "fixture_version"),
    [
        ("configured", "standard-account", "v3"),
        ("missing", None, None),
        ("not_required", None, None),
    ],
)
def test_campaign_plan_json_preserves_fixture_contract(
    status: str,
    fixture_id: str | None,
    fixture_version: str | None,
) -> None:
    plan = campaign_module.create_dataset_campaign_plan(
        records=(_evaluation_result("interaction-1").source,),
        selected_operator_ids=("input.surface.rephrase",),
        run_config=_run_config(),
        settings=cast(Any, _settings()),
        fixture_status=cast(Any, status),
        fixture_id=fixture_id,
        fixture_version=fixture_version,
    )

    payload = plan.model_dump(mode="json")
    assert payload["evaluation_mode"] == "variance"
    assert payload["fixture"] == {
        "status": status,
        "id": fixture_id,
        "version": fixture_version,
    }
    assert any("no fixture identity" in warning.casefold() for warning in payload["warnings"]) == (
        status == "missing"
    )


def test_campaign_plan_exposes_precomputed_candidate_without_new_generation() -> None:
    evaluation_result = _evaluation_result("interaction-1", has_review_finding=True)
    plan = campaign_module.create_dataset_campaign_plan(
        records=(evaluation_result.source,),
        selected_operator_ids=("input.surface.rephrase",),
        run_config=_run_config(),
        settings=preparation_module.load_dataset_semantic_settings(),
        saved_augmentations={evaluation_result.source.id: evaluation_result.augmentation},
    )

    planned_operator = next(
        operator
        for operator in plan.examples[0].operators
        if operator.id == "input.surface.rephrase"
    )
    assert planned_operator.status == "eligible"
    assert planned_operator.candidate_input_available is True
    assert planned_operator.candidate_input is None
    assert plan.calls.variation_generation == 0
    assert any("available but omitted" in warning for warning in plan.warnings)

    sensitive_plan = campaign_module.create_dataset_campaign_plan(
        records=(evaluation_result.source,),
        selected_operator_ids=("input.surface.rephrase",),
        run_config=_run_config(),
        settings=preparation_module.load_dataset_semantic_settings(),
        saved_augmentations={evaluation_result.source.id: evaluation_result.augmentation},
        show_sensitive_values=True,
    )
    sensitive_operator = next(
        operator
        for operator in sensitive_plan.examples[0].operators
        if operator.id == "input.surface.rephrase"
    )
    assert (
        sensitive_operator.candidate_input
        == evaluation_result.augmentation.candidates[0].augmented_input
    )
    assert any("sensitive data" in warning for warning in sensitive_plan.warnings)


def test_campaign_plan_derives_execution_totals_and_timeout_from_run_config() -> None:
    run_config = _run_config(
        repetitions=2,
        environment_api_calls_per_trial=3,
        planned_environment_api_calls=12,
        max_environment_api_calls=12,
        trial_timeout_seconds=47,
    )

    plan = campaign_module.create_dataset_campaign_plan(
        records=(_evaluation_result("interaction-1").source,),
        selected_operator_ids=("input.surface.rephrase",),
        run_config=run_config,
        settings=preparation_module.load_dataset_semantic_settings(),
    )

    assert plan.calls.baseline == 2
    assert plan.calls.variation == 2
    assert plan.calls.total_environment_api == 12
    assert plan.timing.target_trial_timeout_seconds == 47
    with pytest.raises(ValueError, match="frozen"):
        run_config.repetitions = 3  # type: ignore[misc]


def test_campaign_plan_counts_grammar_error_as_llm_generation() -> None:
    plan = campaign_module.create_dataset_campaign_plan(
        records=(_evaluation_result("interaction-1").source,),
        selected_operator_ids=("input.surface.grammar_error",),
        run_config=_run_config(),
        settings=preparation_module.load_dataset_semantic_settings(),
    )

    grammar_operator = next(
        operator
        for operator in plan.examples[0].operators
        if operator.id == "input.surface.grammar_error"
    )
    assert plan.calls.variation_generation == 1
    assert plan.calls.retries == 1
    assert "candidate generation requires a semantic model call" in grammar_operator.reasons


def test_campaign_plan_counts_tone_safety_validation() -> None:
    plan = campaign_module.create_dataset_campaign_plan(
        records=(_evaluation_result("interaction-1").source,),
        selected_operator_ids=("input.tone.angry",),
        run_config=_run_config(),
        settings=preparation_module.load_dataset_semantic_settings(),
    )

    assert plan.calls.variation_generation == 1
    assert plan.calls.evaluators == 6
    assert plan.calls.retries == 1
    assert plan.calls.total_semantic_model == 13


@pytest.mark.parametrize("candidate_state", ["rejected", "missing"])
def test_campaign_plan_does_not_count_known_non_executable_variations(
    candidate_state: str,
) -> None:
    evaluation_result = _evaluation_result("interaction-1")
    saved_augmentation = evaluation_result.augmentation
    if candidate_state == "missing":
        saved_augmentation = saved_augmentation.model_copy(update={"candidates": ()})

    plan = campaign_module.create_dataset_campaign_plan(
        records=(evaluation_result.source,),
        selected_operator_ids=("input.surface.rephrase",),
        run_config=_run_config(
            repetitions=3,
            environment_api_calls_per_trial=5,
            planned_environment_api_calls=30,
        ),
        settings=preparation_module.load_dataset_semantic_settings(),
        saved_augmentations={evaluation_result.source.id: saved_augmentation},
    )

    assert plan.calls.baseline == 3
    assert plan.calls.variation == 0
    assert plan.calls.repetition_executions == 3
    assert plan.calls.evaluators == 3
    assert plan.calls.total_semantic_model == 7
    assert plan.calls.total_environment_api == 15


def test_campaign_plan_keeps_unattempted_operators_conditional() -> None:
    evaluation_result = _evaluation_result("interaction-1")
    saved_augmentation = evaluation_result.augmentation.model_copy(update={"candidates": ()})

    plan = campaign_module.create_dataset_campaign_plan(
        records=(evaluation_result.source,),
        selected_operator_ids=("input.surface.rephrase",),
        run_config=_run_config(),
        settings=preparation_module.load_dataset_semantic_settings(),
        saved_augmentations={evaluation_result.source.id: saved_augmentation},
    )

    operators = {operator.id: operator for operator in plan.examples[0].operators}
    attempted_operator = operators["input.surface.rephrase"]
    assert attempted_operator.status == "ineligible"
    assert attempted_operator.reasons == ("saved semantic qualification produced no candidate",)

    unattempted_operator = operators["input.surface.typing_noise"]
    assert unattempted_operator.status == "conditional"
    assert unattempted_operator.applicability_profile == "broad"
    assert unattempted_operator.selected is False
    assert "operator was not selected" in unattempted_operator.reasons

    self_correction = operators["input.intent.self_correction"]
    assert self_correction.applicability_profile == "conditional"
    assert "numeric, monetary, date, or duration" in self_correction.applicability_rule


def test_human_dry_run_escapes_untrusted_ids_and_summarizes_unselected_catalog(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    unsafe_id = "[bold]spoof[/bold]\x1b]8;;https://example.test\x07link"
    _write_dataset(
        dataset,
        [_record(unsafe_id), *(_record(f"interaction-{index}") for index in range(2, 101))],
    )

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--limit",
            "100",
            "--max-environment-api-calls",
            "600",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output
    assert "\\u001b" in result.output
    assert "[bold]spoof[/bold]" in result.output
    assert "Unselected catalog operators:" in result.output
    assert "0 eligible, 11 conditional, 10 ineligible" in result.output
    assert "use --json for full detail" in " ".join(result.output.split())
    assert "input.surface.rephrase@" not in result.output


def test_campaign_plan_warns_about_missing_review_and_provider_parameters() -> None:
    source = _evaluation_result("interaction-1").source
    plan = campaign_module.create_dataset_campaign_plan(
        records=(source,),
        selected_operator_ids=("input.intent.self_correction",),
        run_config=_run_config(),
        settings=cast(
            Any,
            _settings(
                semantic_provider_id="private-provider",
                semantic_provider_type="openai-compatible",
            ),
        ),
    )

    assert any("no automatic customer evaluator" in warning for warning in plan.warnings)
    assert any("strict JSON-schema" in warning for warning in plan.warnings)


def test_dry_run_prints_configured_fixture_identity(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config)
    raw_config = json.loads(target_config.read_text(encoding="utf-8"))
    raw_config["fixture_id"] = "standard-account"
    raw_config["fixture_version"] = "v3"
    target_config.write_text(json.dumps(raw_config), encoding="utf-8")

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Fixture: standard-account@v3" in result.output
    assert "no fixture identity" not in result.output


@pytest.mark.parametrize("evaluation_mode", ["correctness", "preference"])
def test_unimplemented_evaluation_mode_fails_before_calls_or_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evaluation_mode: str,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    _write_dataset(dataset, [_record()])

    def unexpected_settings() -> None:
        raise AssertionError("unsupported evaluation mode reached semantic settings")

    monkeypatch.setattr(preparation_module, "load_dataset_semantic_settings", unexpected_settings)

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--evaluation-mode",
            evaluation_mode,
            "--output",
            str(output),
        ],
        terminal_width=240,
    )

    assert result.exit_code == 2
    normalized_output = " ".join(
        _ANSI_ESCAPE_PATTERN.sub("", result.output).replace("│", " ").split()
    )
    assert f"evaluation mode '{evaluation_mode}' is not implemented" in normalized_output
    assert "Historical dataset output is grounding evidence, not an expected answer" in (
        normalized_output
    )
    assert not output.exists()


def test_augmentation_persistence_options_are_discoverable_at_80_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLUMNS", "80")

    result = runner.invoke(root_app, ["dataset", "evaluate", "--help"])

    assert result.exit_code == 0, result.output
    normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", result.output).split())
    assert "--augmentations-input" in normalized_output
    assert "--augmentations-output" in normalized_output
    assert "--no-save-augmentations" in normalized_output


def test_dry_run_plans_default_augmentations_output_without_creating_files(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "results.jsonl"
    augmentations = tmp_path / "results.augmentations.jsonl"
    _write_dataset(dataset, [_record()])

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--output",
            str(evidence),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", result.output).split())
    assert f"Augmentations destination: {augmentations}" in normalized_output
    assert "may contain sensitive inputs and derived semantic data" in normalized_output
    assert "retain" in normalized_output
    assert "data policy" in normalized_output
    assert not evidence.exists()
    assert not augmentations.exists()


def test_dry_run_supports_custom_or_disabled_augmentation_persistence(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "results.jsonl"
    custom_augmentations = tmp_path / "review" / "accepted.jsonl"
    _write_dataset(dataset, [_record()])

    custom = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--output",
            str(evidence),
            "--augmentations-output",
            str(custom_augmentations),
            "--dry-run",
        ],
    )
    disabled = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--output",
            str(evidence),
            "--no-save-augmentations",
            "--dry-run",
        ],
    )

    assert custom.exit_code == 0, custom.output
    assert f"Augmentations destination: {custom_augmentations}" in " ".join(
        _ANSI_ESCAPE_PATTERN.sub("", custom.output).split()
    )
    assert disabled.exit_code == 0, disabled.output
    assert "Augmentations destination:" not in disabled.output
    assert "Augmentations will not be saved." in " ".join(
        _ANSI_ESCAPE_PATTERN.sub("", disabled.output).split()
    )
    assert not evidence.exists()
    assert not custom_augmentations.exists()


def test_custom_augmentations_output_conflicts_with_no_save(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "results.jsonl"
    augmentations = tmp_path / "accepted.jsonl"
    _write_dataset(dataset, [_record()])

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--output",
            str(evidence),
            "--augmentations-output",
            str(augmentations),
            "--no-save-augmentations",
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", result.output).split())
    assert "--augmentations-output" in normalized_output
    assert "used with --no-save-augmentations" in normalized_output
    assert not evidence.exists()
    assert not augmentations.exists()


def test_augmentation_input_conflicts_with_augmentation_output(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "results.jsonl"
    augmentation_input = tmp_path / "accepted.jsonl"
    augmentation_output = tmp_path / "new.jsonl"
    _write_dataset(dataset, [_record()])
    augmentation_input.touch(mode=0o600)

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--output",
            str(evidence),
            "--augmentations-input",
            str(augmentation_input),
            "--augmentations-output",
            str(augmentation_output),
            "--dry-run",
        ],
    )

    assert result.exit_code == 2
    normalized_output = " ".join(
        _ANSI_ESCAPE_PATTERN.sub("", result.output).replace("│", " ").split()
    )
    assert "cannot be combined" in normalized_output
    assert not evidence.exists()
    assert not augmentation_output.exists()


def test_augmentation_input_requires_complete_selected_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "results.jsonl"
    augmentation_input = tmp_path / "accepted.jsonl"
    _write_dataset(dataset, [_record()])
    augmentation_input.touch(mode=0o600)

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--output",
            str(evidence),
            "--augmentations-input",
            str(augmentation_input),
            "--dry-run",
        ],
    )

    assert result.exit_code == 2
    normalized_output = " ".join(
        _ANSI_ESCAPE_PATTERN.sub("", result.output).replace("│", " ").split()
    )
    assert "augmentation input must contain every selected interaction exactly once" in (
        normalized_output
    )
    assert not evidence.exists()


def test_redaction_dry_run_reports_value_free_coverage_without_key_or_state(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    policy = tmp_path / "redaction.json"
    secret = "customer-secret-value"
    _write_dataset(
        dataset,
        [
            {
                "id": "private-interaction",
                "input": f"Use {secret}",
                "output": {"private": secret},
            }
        ],
    )
    policy.write_text(
        json.dumps(
            {
                "version": 1,
                "rules": [
                    {
                        "name": "customer_secret",
                        "locations": ["input", "output"],
                        "selector": "$text",
                        "literal": secret,
                        "action": "pseudonymize",
                    }
                ],
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
            "--redaction-policy",
            str(policy),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Redaction policy sha256:" in result.output
    assert "Redaction coverage (input): 1 selected value(s) across 1 path(s)" in result.output
    assert "Redaction coverage (output): 1 selected value(s) across 1 path(s)" in result.output
    assert secret not in result.output
    assert not (tmp_path / "pseudonyms.json").exists()


def test_stateful_target_dry_run_counts_physical_lifecycle_calls(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_stateful_target_config(target_config)

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--repetitions",
            "2",
            "--max-environment-api-calls",
            "24",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Potential environment API calls: up to 24 (authorized maximum: 24)" in " ".join(
        result.output.split()
    )
    assert "Lifecycle calls per execution: 6" in result.output
    assert (
        "Every test case invokes and validates the configured environment reset contract"
        in " ".join(result.output.split())
    )
