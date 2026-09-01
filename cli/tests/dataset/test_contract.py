from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner
from ul_cli.dataset.evaluation import command as command_module
from ul_cli.main import app as root_app

from ._files import (
    _record,
    _write_dataset,
)

runner = CliRunner()
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def test_help_explains_dataset_environment_and_operator_contract() -> None:
    result = runner.invoke(root_app, ["dataset", "evaluate", "--help"])

    assert result.exit_code == 0, result.output
    assert '"id"' in result.output
    assert '"input"' in result.output
    assert '"output"' in result.output
    normalized_help = " ".join(_ANSI_ESCAPE_PATTERN.sub("", result.output).split())
    assert "explicit reset/setup/execute/snapshot lifecycle" in normalized_help
    assert "UL_LIVE" in result.output
    assert "Fresh-state" in normalized_help
    assert "executions" in normalized_help
    assert "target executions" not in normalized_help
    assert "Maximum customer" in normalized_help
    assert "environment API" in normalized_help
    assert "requests" in normalized_help
    assert "Discover operators: ul augmentations list --mode dataset_variation" in normalized_help
    assert "--environment-config" in normalized_help
    help_text = " ".join(normalized_help.replace("│", "").split())
    assert "Pass --target URL for an existing response-only JSON agent" in help_text
    assert "Isolated-response HTTP(S) URL" in help_text
    assert "generic-json mapping by default" in help_text
    assert "must isolate every request" in help_text
    assert "both default to input.surface.typing_noise" in help_text
    assert "--http-preset" in help_text
    assert "--request-json" in help_text
    assert "--response-json" in help_text
    assert "--agent-model" in help_text
    assert "--header-from" in help_text
    assert "configuration" in normalized_help
    assert "structured multi-turn cases" in help_text
    assert "customer's agent environment API" in help_text

    init_help = runner.invoke(root_app, ["dataset", "init", "--help"])
    assert init_help.exit_code == 0, init_help.output
    normalized_init_help = " ".join(_ANSI_ESCAPE_PATTERN.sub("", init_help.output).split())
    assert "environment_config" in normalized_init_help
    assert "--url" in normalized_init_help
    assert "private connection config" in normalized_init_help
    assert "customer-managed environment API" in normalized_init_help

    operators = runner.invoke(root_app, ["dataset", "operators"])
    assert operators.exit_code == 0, operators.output
    assert "input.surface.disfluency_repeat" in operators.output
    assert "input.intent.self_correction" in operators.output


def test_legacy_operator_list_delegates_to_catalog_and_counts_deterministic_correction(
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
        "input.intent.self_correction@1.1.0:",
        "input.style.terse@1.0.0:",
        "input.style.verbose@1.0.0:",
        "input.surface.case_variation@1.0.0:",
        "input.surface.disfluency_repeat@1.0.0:",
        "input.surface.fragmented_syntax@1.0.0:",
        "input.surface.grammar_error@1.0.0:",
        "input.surface.punctuation_noise@1.0.0:",
        "input.surface.rephrase@1.0.0:",
        "input.surface.typing_noise@1.0.0:",
        "input.tone.angry@1.0.0:",
        "input.tone.argumentative@1.0.0:",
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
            "input.intent.self_correction@1.1.0",
            "--dry-run",
        ],
    )

    assert dry_run.exit_code == 0, dry_run.output
    assert "Operators: input.intent.self_correction@1.1.0" in dry_run.output
    assert "Potential semantic model calls: up to 14" in dry_run.output
    assert "Potential environment API calls: up to 6" in dry_run.output

    for tone_operator in ("input.tone.angry", "input.tone.argumentative"):
        tone_dry_run = runner.invoke(
            root_app,
            [
                "dataset",
                "evaluate",
                str(dataset),
                "--operator",
                f"{tone_operator}@1.0.0",
                "--dry-run",
            ],
        )
        assert tone_dry_run.exit_code == 0, tone_dry_run.output
        assert f"Operators: {tone_operator}@1.0.0" in tone_dry_run.output
        assert "Potential semantic model calls: up to 16" in tone_dry_run.output
        assert "Potential environment API calls: up to 6" in tone_dry_run.output

    wrong_version = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--operator",
            "input.intent.self_correction@2.0.0",
            "--dry-run",
        ],
    )
    assert wrong_version.exit_code == 2
    assert "unknown augmentation operator reference" in wrong_version.output


@pytest.mark.parametrize(
    "operators",
    [
        ("input.intent.self_correction", "input.intent.self_correction"),
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

    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", unexpected_settings)
    arguments = ["dataset", "evaluate", str(dataset)]
    for operator_id in operators:
        arguments.extend(("--operator", operator_id))

    result = runner.invoke(root_app, arguments)

    assert result.exit_code != 0
    assert "operator" in result.output.casefold()
