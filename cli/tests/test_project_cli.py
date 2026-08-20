from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner
from ul_cli import project
from ul_cli.main import app

runner = CliRunner()


def _write_dataset(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "id": "interaction-1",
                "input": "Transfer 100 to Alice.",
                "output": {
                    "actions": [{"action": "transfer", "amount": 100, "recipient": "Alice"}]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _initialize(tmp_path: Path, *extra_arguments: str) -> Path:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset)
    result = runner.invoke(
        app,
        [
            "init",
            str(dataset),
            "--sandbox-url",
            "https://sandbox.example",
            "--allow-sandbox-network-egress",
            "--confirm-isolated-sandbox",
            *extra_arguments,
        ],
    )
    assert result.exit_code == 0, result.output
    return dataset


def test_root_help_exposes_simple_workflow() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "init" in result.output
    assert "run" in result.output
    assert "report" in result.output


def test_init_creates_private_project_and_generated_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = _initialize(
        tmp_path,
        "--operator",
        "input.surface.rephrase",
        "--limit",
        "1",
        "--repetitions",
        "2",
        "--max-sandbox-api-calls",
        "12",
    )

    config_path = tmp_path / ".ul" / "config.json"
    sandbox_path = tmp_path / ".ul" / "sandbox.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config == {
        "schema_version": 1,
        "dataset": dataset.name,
        "sandbox_config": ".ul/sandbox.json",
        "invariants": None,
        "operators": ["input.surface.rephrase"],
        "limit": 1,
        "repetitions": 2,
        "max_sandbox_api_calls": 12,
        "allow_sandbox_network_egress": True,
        "confirm_isolated_sandbox": True,
        "allow_insecure_http": False,
    }
    assert sandbox_path.is_file()
    if hasattr(stat, "S_IMODE"):
        assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(sandbox_path.stat().st_mode) == 0o600
        assert stat.S_IMODE((tmp_path / ".ul").stat().st_mode) == 0o700
        assert stat.S_IMODE((tmp_path / ".ul" / "runs").stat().st_mode) == 0o700


@pytest.mark.parametrize(
    ("missing_flag", "expected_message"),
    [
        ("--allow-sandbox-network-egress", "allow-sandbox-network-egress"),
        ("--confirm-isolated-sandbox", "confirm-isolated-sandbox"),
    ],
)
def test_init_requires_one_time_safety_acknowledgements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_flag: str,
    expected_message: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset)
    arguments = [
        "init",
        str(dataset),
        "--sandbox-url",
        "https://sandbox.example",
        "--allow-sandbox-network-egress",
        "--confirm-isolated-sandbox",
    ]
    arguments.remove(missing_flag)

    result = runner.invoke(app, arguments)

    assert result.exit_code == 2
    assert expected_message in result.output
    assert not (tmp_path / ".ul").exists()


def test_init_reuses_existing_sandbox_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = _initialize(tmp_path)
    sandbox = tmp_path / ".ul" / "sandbox.json"

    result = runner.invoke(
        app,
        [
            "init",
            str(dataset),
            "--sandbox-config",
            str(sandbox),
            "--allow-sandbox-network-egress",
            "--confirm-isolated-sandbox",
        ],
    )

    assert result.exit_code == 2
    assert "will not overwrite" in result.output


def test_run_discovers_parent_project_and_applies_one_run_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = _initialize(tmp_path, "--limit", "2", "--repetitions", "3")
    nested_directory = tmp_path / "src" / "package"
    nested_directory.mkdir(parents=True)
    monkeypatch.chdir(nested_directory)
    received: dict[str, Any] = {}

    def fake_evaluate_dataset(**arguments: Any) -> None:
        received.update(arguments)
        arguments["output"].write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(project, "evaluate_dataset", fake_evaluate_dataset)

    result = runner.invoke(
        app,
        ["run", "--limit", "1", "--repetitions", "4", "--operator", "input.surface.rephrase"],
    )

    assert result.exit_code == 0, result.output
    assert received["data"] == dataset
    assert received["sandbox_config"] == tmp_path / ".ul" / "sandbox.json"
    assert received["limit"] == 1
    assert received["repetitions"] == 4
    assert received["operator"] == ["input.surface.rephrase"]
    assert received["allow_sandbox_network_egress"] is True
    assert received["confirm_isolated_sandbox"] is True
    state = json.loads((tmp_path / ".ul" / "state.json").read_text(encoding="utf-8"))
    assert state["latest_evidence"].startswith(".ul/runs/")
    assert (tmp_path / state["latest_evidence"]).is_file()
    assert "Next: ul report" in result.output


@pytest.mark.parametrize(("exit_code", "records_state"), [(1, True), (2, False)])
def test_run_records_only_reportable_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
    records_state: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    _initialize(tmp_path)

    def fake_evaluate_dataset(**arguments: Any) -> None:
        arguments["output"].write_text("{}\n", encoding="utf-8")
        raise typer.Exit(exit_code)

    monkeypatch.setattr(project, "evaluate_dataset", fake_evaluate_dataset)

    result = runner.invoke(app, ["run"])

    assert result.exit_code == exit_code
    assert (tmp_path / ".ul" / "state.json").exists() is records_state


def test_report_uses_latest_evidence_and_explicit_path_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _initialize(tmp_path)
    latest = tmp_path / ".ul" / "runs" / "latest.jsonl"
    latest.write_text("{}\n", encoding="utf-8")
    (tmp_path / ".ul" / "state.json").write_text(
        json.dumps({"schema_version": 1, "latest_evidence": ".ul/runs/latest.jsonl"}),
        encoding="utf-8",
    )
    explicit = tmp_path / "explicit.jsonl"
    explicit.write_text("{}\n", encoding="utf-8")
    reported: list[Path] = []

    def fake_report_dataset_evidence(**arguments: Any) -> None:
        reported.append(arguments["evidence"])

    monkeypatch.setattr(project, "report_dataset_evidence", fake_report_dataset_evidence)

    latest_result = runner.invoke(app, ["report"])
    explicit_result = runner.invoke(app, ["report", str(explicit)])

    assert latest_result.exit_code == 0, latest_result.output
    assert explicit_result.exit_code == 0, explicit_result.output
    assert reported == [latest, explicit]


def test_report_without_completed_run_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _initialize(tmp_path)

    result = runner.invoke(app, ["report"])

    assert result.exit_code == 2
    assert "run 'ul run' first or pass EVIDENCE" in result.output


def test_malformed_or_unknown_project_config_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    project_directory = tmp_path / ".ul"
    project_directory.mkdir()
    (project_directory / "config.json").write_text(
        json.dumps({"schema_version": 1, "unknown": True}), encoding="utf-8"
    )

    result = runner.invoke(app, ["run", "--dry-run"])

    assert result.exit_code == 2
    assert "invalid UL project config" in result.output


def test_real_dry_run_uses_saved_configuration_without_network_or_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _initialize(tmp_path, "--limit", "1", "--repetitions", "1")

    result = runner.invoke(app, ["run", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "No model or sandbox API requests sent." in result.output
    assert "Selected interactions: 1" in result.output
    assert not (tmp_path / ".ul" / "state.json").exists()
