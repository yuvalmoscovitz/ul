from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner
from ul_cli import project
from ul_cli.main import app

runner = CliRunner()


def _write_private_file(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o600)


def _accept_test_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(project, "is_reportable_dataset_evidence", lambda path: True)


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
            "--environment-url",
            "https://environment.example",
            "--allow-environment-network",
            "--confirm-test-environment",
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


def test_init_creates_private_project_and_generated_environment(
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
        "--max-environment-api-calls",
        "12",
    )

    config_path = tmp_path / ".ul" / "config.json"
    environment_path = tmp_path / ".ul" / "environment.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    environment_config_sha256 = config.pop("environment_config_sha256")

    assert config == {
        "schema_version": 2,
        "dataset": dataset.name,
        "environment_config": ".ul/environment.json",
        "environment_origin": "https://environment.example",
        "invariants": None,
        "redaction_policy": None,
        "redaction_policy_sha256": None,
        "redaction_state": None,
        "save_augmentations": True,
        "operators": ["input.surface.rephrase"],
        "limit": 1,
        "repetitions": 2,
        "max_environment_api_calls": 12,
        "allow_environment_network": True,
        "confirm_test_environment": True,
        "allow_insecure_http": False,
    }
    assert len(environment_config_sha256) == 64
    assert environment_path.is_file()
    assert (tmp_path / ".ul" / ".gitignore").read_text(encoding="utf-8") == "*\n"
    if hasattr(stat, "S_IMODE"):
        assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(environment_path.stat().st_mode) == 0o600
        assert stat.S_IMODE((tmp_path / ".ul").stat().st_mode) == 0o700
        assert stat.S_IMODE((tmp_path / ".ul" / "runs").stat().st_mode) == 0o700


def test_init_defaults_fit_generated_environment_call_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "interactions.jsonl"
    dataset.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"interaction-{index}",
                    "input": f"Transfer {index} to Alice.",
                    "output": {"status": "recorded"},
                }
            )
            + "\n"
            for index in range(1, 4)
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "init",
            str(dataset),
            "--environment-url",
            "https://environment.example",
            "--allow-environment-network",
            "--confirm-test-environment",
        ],
    )
    assert result.exit_code == 0, result.output

    dry_run = runner.invoke(app, ["run", "--dry-run"])

    assert dry_run.exit_code == 0, dry_run.output
    assert "Selected interactions: 3" in dry_run.output
    assert "Potential environment API calls: up to 90 (authorized maximum: 120)" in dry_run.output


@pytest.mark.parametrize(
    "missing_flag",
    [
        "--allow-environment-network",
        "--confirm-test-environment",
    ],
)
def test_init_requires_one_time_safety_acknowledgements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_flag: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset)
    arguments = [
        "init",
        str(dataset),
        "--environment-url",
        "https://environment.example",
        "--allow-environment-network",
        "--confirm-test-environment",
    ]
    arguments.remove(missing_flag)

    result = runner.invoke(app, arguments, terminal_width=160)

    assert result.exit_code == 2
    if missing_flag == "--confirm-test-environment":
        normalized_output = " ".join(result.output.split())
        assert "the agent may change external state" in normalized_output
        assert "--confirm-test-environment to" in normalized_output
    assert not (tmp_path / ".ul").exists()


def test_init_reuses_existing_environment_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = _initialize(tmp_path)
    environment = tmp_path / ".ul" / "environment.json"

    result = runner.invoke(
        app,
        [
            "init",
            str(dataset),
            "--environment-config",
            str(environment),
            "--allow-environment-network",
            "--confirm-test-environment",
        ],
    )

    assert result.exit_code == 2
    assert "will not overwrite" in result.output


def test_run_rejects_same_origin_environment_template_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _initialize(tmp_path)
    environment_path = tmp_path / ".ul" / "environment.json"
    environment_config = json.loads(environment_path.read_text(encoding="utf-8"))
    environment_config["reset"]["request_json_template"]["fixture"] = "customer-fixture"
    environment_path.write_text(json.dumps(environment_config), encoding="utf-8")

    result = runner.invoke(app, ["run", "--dry-run"])

    assert result.exit_code == 2
    assert "environment configuration changed since 'ul init'" in result.output


def test_run_rejects_environment_origin_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _initialize(tmp_path)
    environment_path = tmp_path / ".ul" / "environment.json"
    environment_config = json.loads(environment_path.read_text(encoding="utf-8"))
    for operation in ("reset", "execute_turn", "snapshot"):
        environment_config[operation]["url"] = environment_config[operation]["url"].replace(
            "environment.example", "other.example"
        )
    environment_path.write_text(json.dumps(environment_config), encoding="utf-8")

    result = runner.invoke(app, ["run", "--dry-run"], terminal_width=160)

    assert result.exit_code == 2
    assert "environment origin changed since 'ul init'" in result.output
    assert not (tmp_path / ".ul" / "state.json").exists()


def test_failed_generated_environment_init_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset)
    common_arguments = [
        "init",
        str(dataset),
        "--allow-environment-network",
        "--confirm-test-environment",
    ]

    failed = runner.invoke(app, [*common_arguments, "--environment-url", "not-a-url"])
    retried = runner.invoke(
        app, [*common_arguments, "--environment-url", "https://environment.example"]
    )

    assert failed.exit_code == 2
    assert retried.exit_code == 0, retried.output


def test_failed_init_does_not_delete_existing_environment_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset)
    project_directory = tmp_path / ".ul"
    project_directory.mkdir(mode=0o700)
    environment = project_directory / "environment.json"
    _write_private_file(environment, "customer-owned\n")

    result = runner.invoke(
        app,
        [
            "init",
            str(dataset),
            "--environment-url",
            "https://environment.example",
            "--allow-environment-network",
            "--confirm-test-environment",
        ],
    )

    assert result.exit_code == 2
    assert environment.read_text(encoding="utf-8") == "customer-owned\n"


def test_interrupted_init_removes_only_owned_scaffolding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset)

    def interrupt_environment_creation(*arguments: Any, **keywords: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(project, "initialize_dataset_environment", interrupt_environment_creation)

    result = runner.invoke(
        app,
        [
            "init",
            str(dataset),
            "--environment-url",
            "https://environment.example",
            "--allow-environment-network",
            "--confirm-test-environment",
        ],
    )

    assert result.exit_code != 0
    assert not (tmp_path / ".ul").exists()


def test_run_discovers_parent_project_and_applies_one_run_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = _initialize(tmp_path, "--limit", "2", "--repetitions", "3")
    nested_directory = tmp_path / "src" / "package"
    nested_directory.mkdir(parents=True)
    monkeypatch.chdir(nested_directory)
    _accept_test_evidence(monkeypatch)
    received: dict[str, Any] = {}

    def fake_evaluate_dataset(**arguments: Any) -> None:
        received.update(arguments)
        _write_private_file(arguments["output"], "{}\n")

    monkeypatch.setattr(project, "evaluate_dataset", fake_evaluate_dataset)

    result = runner.invoke(
        app,
        ["run", "--limit", "1", "--repetitions", "4", "--operator", "input.surface.rephrase"],
    )

    assert result.exit_code == 0, result.output
    assert received["data"] == dataset
    assert received["environment_config"] == tmp_path / ".ul" / "environment.json"
    assert received["limit"] == 1
    assert received["repetitions"] == 4
    assert received["operator"] == ["input.surface.rephrase"]
    assert received["allow_environment_network"] is True
    assert received["confirm_test_environment"] is True
    assert received["expected_environment_origin"] == "https://environment.example"
    state = json.loads((tmp_path / ".ul" / "state.json").read_text(encoding="utf-8"))
    assert state["latest_evidence"].startswith(".ul/runs/")
    assert (tmp_path / state["latest_evidence"]).is_file()
    assert "Next: ul report" in result.output


def test_init_persists_redaction_and_retention_choices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset)
    policy = tmp_path / "redaction.json"
    policy.write_text(
        json.dumps(
            {
                "version": 1,
                "rules": [
                    {
                        "name": "account",
                        "locations": ["input", "output"],
                        "selector": "$text",
                        "literal": "Alice",
                        "action": "pseudonymize",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "redaction-state.json"
    initialized = runner.invoke(
        app,
        [
            "init",
            str(dataset),
            "--environment-url",
            "https://environment.example",
            "--allow-environment-network",
            "--confirm-test-environment",
            "--redaction-policy",
            str(policy),
            "--redaction-state",
            str(state),
            "--no-save-augmentations",
        ],
    )
    assert initialized.exit_code == 0, initialized.output
    received: dict[str, Any] = {}
    real_evaluate_dataset = project.evaluate_dataset

    def fake_evaluate_dataset(**arguments: Any) -> None:
        received.update(arguments)

    monkeypatch.setattr(project, "evaluate_dataset", fake_evaluate_dataset)

    result = runner.invoke(app, ["run", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert received["redaction_policy"] == policy
    assert received["redaction_state"] == state
    assert received["no_save_augmentations"] is True
    assert len(received["expected_redaction_policy_sha256"]) == 64

    policy_payload = json.loads(policy.read_text(encoding="utf-8"))
    policy_payload["rules"][0]["literal"] = "never-matches"
    policy.write_text(json.dumps(policy_payload), encoding="utf-8")
    monkeypatch.setattr(project, "evaluate_dataset", real_evaluate_dataset)

    changed_policy = runner.invoke(app, ["run", "--dry-run"], terminal_width=160)

    assert changed_policy.exit_code == 2
    assert "redaction policy changed since 'ul init'" in changed_policy.output


def test_run_resumes_latest_reportable_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _initialize(tmp_path)
    _accept_test_evidence(monkeypatch)

    def interrupted_evaluation(**arguments: Any) -> None:
        _write_private_file(arguments["output"], "{}\n")
        raise typer.Exit(2)

    monkeypatch.setattr(project, "evaluate_dataset", interrupted_evaluation)
    first_run = runner.invoke(app, ["run"])
    assert first_run.exit_code == 2
    recorded_state = json.loads((tmp_path / ".ul" / "state.json").read_text(encoding="utf-8"))
    evidence = tmp_path / recorded_state["latest_evidence"]
    received: dict[str, Any] = {}

    def resumed_evaluation(**arguments: Any) -> None:
        received.update(arguments)

    monkeypatch.setattr(project, "evaluate_dataset", resumed_evaluation)
    resumed = runner.invoke(app, ["run", "--resume"])

    assert resumed.exit_code == 0, resumed.output
    assert received["resume"] == evidence
    assert received["output"] == evidence


def test_run_resume_without_evidence_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _initialize(tmp_path)

    result = runner.invoke(app, ["run", "--resume"])

    assert result.exit_code == 2
    assert "cannot resume: latest evidence is missing or changed" in result.output


@pytest.mark.parametrize("exit_code", [0, 1, 2])
def test_run_records_any_nonempty_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
) -> None:
    monkeypatch.chdir(tmp_path)
    _initialize(tmp_path)
    _accept_test_evidence(monkeypatch)

    def fake_evaluate_dataset(**arguments: Any) -> None:
        _write_private_file(arguments["output"], "{}\n")
        raise typer.Exit(exit_code)

    monkeypatch.setattr(project, "evaluate_dataset", fake_evaluate_dataset)

    result = runner.invoke(app, ["run"])

    assert result.exit_code == exit_code
    assert (tmp_path / ".ul" / "state.json").is_file()
    assert "Next: ul report" in result.output


def test_run_does_not_record_empty_exit_two_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _initialize(tmp_path)

    def fake_evaluate_dataset(**arguments: Any) -> None:
        arguments["output"].touch()
        raise typer.Exit(2)

    monkeypatch.setattr(project, "evaluate_dataset", fake_evaluate_dataset)

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 2
    assert not (tmp_path / ".ul" / "state.json").exists()


def test_run_records_evidence_before_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _initialize(tmp_path)
    _accept_test_evidence(monkeypatch)

    def interrupted_evaluation(**arguments: Any) -> None:
        _write_private_file(arguments["output"], "{}\n")
        raise KeyboardInterrupt

    monkeypatch.setattr(project, "evaluate_dataset", interrupted_evaluation)

    result = runner.invoke(app, ["run"])

    assert result.exit_code != 0
    assert (tmp_path / ".ul" / "state.json").is_file()
    assert "Next: ul report" in result.output


def test_run_does_not_replace_latest_with_malformed_partial_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _initialize(tmp_path)
    monkeypatch.setattr(
        project,
        "is_reportable_dataset_evidence",
        lambda path: path.read_text(encoding="utf-8") == "{}\n",
    )

    def completed_evaluation(**arguments: Any) -> None:
        _write_private_file(arguments["output"], "{}\n")

    monkeypatch.setattr(project, "evaluate_dataset", completed_evaluation)
    completed = runner.invoke(app, ["run"])
    assert completed.exit_code == 0, completed.output
    state_path = tmp_path / ".ul" / "state.json"
    original_state = state_path.read_bytes()

    def malformed_evaluation(**arguments: Any) -> None:
        _write_private_file(arguments["output"], "{partial")
        raise OSError("simulated interruption")

    monkeypatch.setattr(project, "evaluate_dataset", malformed_evaluation)
    interrupted = runner.invoke(app, ["run"])

    assert interrupted.exit_code != 0
    assert state_path.read_bytes() == original_state
    assert "Next: ul report" not in interrupted.output


def test_arbitrary_json_object_is_not_reportable_evidence(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "not-evidence.jsonl"
    _write_private_file(evidence, '{"diagnostic":"interrupted"}\n')

    assert not project._has_nonempty_evidence(evidence)


def test_report_uses_latest_evidence_and_explicit_path_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _initialize(tmp_path)
    latest = tmp_path / ".ul" / "runs" / "latest.jsonl"
    _write_private_file(latest, "{}\n")
    _write_private_file(
        tmp_path / ".ul" / "state.json",
        json.dumps(
            {
                "schema_version": 1,
                "latest_evidence": ".ul/runs/latest.jsonl",
                "latest_evidence_sha256": hashlib.sha256(b"{}\n").hexdigest(),
            }
        ),
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


def test_report_without_run_evidence_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _initialize(tmp_path)

    result = runner.invoke(app, ["report"])

    assert result.exit_code == 2
    assert "no run evidence found" in result.output


def test_malformed_or_unknown_project_config_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    project_directory = tmp_path / ".ul"
    project_directory.mkdir(mode=0o700)
    (project_directory / "runs").mkdir(mode=0o700)
    sensitive_value = "private-project-config-value"
    _write_private_file(
        project_directory / "config.json",
        json.dumps({"schema_version": 2, "unknown": sensitive_value}),
    )

    result = runner.invoke(app, ["run", "--dry-run"])

    assert result.exit_code == 2
    assert "invalid UL project config" in result.output
    assert "unknown" in result.output
    assert sensitive_value not in result.output
    assert "input_value" not in result.output


def test_project_config_cannot_disable_saved_redaction_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset)
    policy = tmp_path / "redaction.json"
    policy.write_text(
        json.dumps(
            {
                "version": 1,
                "rules": [
                    {
                        "name": "account",
                        "locations": ["input"],
                        "selector": "$text",
                        "literal": "Alice",
                        "action": "pseudonymize",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    initialized = runner.invoke(
        app,
        [
            "init",
            str(dataset),
            "--environment-url",
            "https://environment.example",
            "--allow-environment-network",
            "--confirm-test-environment",
            "--redaction-policy",
            str(policy),
            "--redaction-state",
            str(tmp_path / "redaction-state.json"),
        ],
    )
    assert initialized.exit_code == 0, initialized.output
    config_path = tmp_path / ".ul" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    del config["redaction_policy_sha256"]
    _write_private_file(config_path, json.dumps(config))

    result = runner.invoke(app, ["run", "--dry-run"], terminal_width=160)

    assert result.exit_code == 2
    assert "redaction_policy_sha256" in result.output


def test_deep_project_config_is_rejected_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    project_directory = tmp_path / ".ul"
    project_directory.mkdir(mode=0o700)
    (project_directory / "runs").mkdir(mode=0o700)
    _write_private_file(project_directory / "config.json", "[" * 10_000 + "]" * 10_000)

    with pytest.raises(ValueError, match="project file exceeds the nesting limit"):
        project._read_private_json(project_directory / "config.json")

    result = runner.invoke(app, ["run", "--dry-run"], terminal_width=160)

    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_private_json_reader_rejects_symlinks_without_no_follow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable")
    monkeypatch.setattr(project.os, "O_NOFOLLOW", 0, raising=False)

    with pytest.raises(OSError, match="symbolic link"):
        project._read_private_json(link)


def test_private_json_creation_skips_fchmod_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "private.json"

    def fail_fchmod(descriptor: int, mode: int) -> None:
        del descriptor, mode
        raise AssertionError("fchmod must not be used on Windows")

    monkeypatch.setattr(project.sys, "platform", "win32")
    monkeypatch.setattr(project.os, "fchmod", fail_fchmod, raising=False)

    project._create_private_json(path, {"safe": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"safe": True}


def test_real_dry_run_uses_saved_configuration_without_network_or_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _initialize(tmp_path, "--limit", "1", "--repetitions", "1")

    result = runner.invoke(app, ["run", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "No model or environment API requests sent." in result.output
    assert "Selected interactions: 1" in result.output
    assert not (tmp_path / ".ul" / "state.json").exists()
