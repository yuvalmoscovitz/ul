from __future__ import annotations

import json
import re
import stat
from pathlib import Path

from typer.testing import CliRunner
from ul_cli.main import app as root_app

runner = CliRunner()
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


runner = CliRunner()
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def test_init_creates_private_strict_starter_config(tmp_path: Path) -> None:
    target_config = tmp_path / "target.json"

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "init",
            str(target_config),
            "--url",
            "https://environment.example.test",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(target_config.read_text(encoding="utf-8")) == {
        "version": 5,
        "environment_id": "replace-with-stable-environment-id",
        "headers_from_env": {},
        "reset": {
            "url": "https://environment.example.test/reset",
            "request_json_template": {"case_id": "{{case_id}}"},
            "reset_session": True,
            "reset_env": True,
            "case_id_json_pointer": "/case_id",
            "environment_id_json_pointer": "/environment_id",
            "generation_json_pointer": "/generation",
            "clean_state_json_pointer": "/clean",
            "clean_state_value": True,
        },
        "execute_turn": {
            "url": "https://environment.example.test/execute",
            "request_json_template": {
                "case_id": "{{case_id}}",
                "turn_id": "{{turn_id}}",
                "input": "{{input}}",
            },
            "response_json_pointer": "/response",
            "case_id_json_pointer": "/case_id",
            "turn_id_json_pointer": "/turn_id",
            "environment_id_json_pointer": "/environment_id",
        },
        "snapshot": {
            "url": "https://environment.example.test/snapshot",
            "request_json_template": {"case_id": "{{case_id}}", "turn_id": "{{turn_id}}"},
            "response_json_pointer": "/state",
            "case_id_json_pointer": "/case_id",
            "turn_id_json_pointer": "/turn_id",
            "environment_id_json_pointer": "/environment_id",
        },
    }
    assert stat.S_IMODE(target_config.stat().st_mode) == 0o600
    assert "clean agent session" in result.output
    assert "clean external" in result.output
    assert "headers_from_env" in result.output
    assert '"reset_session":true,"reset_env":true' in result.output
    assert '"generation":1,"clean":true' in result.output
    assert "--dry-run" in result.output


def test_init_records_stateful_fixture_identity(tmp_path: Path) -> None:
    target_config = tmp_path / "target.json"

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "init",
            str(target_config),
            "--url",
            "https://environment.example.test",
            "--fixture-id",
            "standard-account",
            "--fixture-version",
            "v3",
        ],
    )

    assert result.exit_code == 0, result.output
    config = json.loads(target_config.read_text(encoding="utf-8"))
    assert config["fixture_id"] == "standard-account"
    assert config["fixture_version"] == "v3"


def test_init_rejects_partial_fixture_identity(tmp_path: Path) -> None:
    target_config = tmp_path / "target.json"

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "init",
            str(target_config),
            "--url",
            "https://environment.example.test",
            "--fixture-id",
            "standard-account",
        ],
    )

    assert result.exit_code == 2
    normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", result.output).split())
    assert "--fixture-id and --fixture-version" in normalized_output
    assert not target_config.exists()


def test_init_translates_custom_isolated_json_contract(tmp_path: Path) -> None:
    target_config = tmp_path / "target.json"

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "init",
            str(target_config),
            "--url",
            "https://agent.example.test/chat",
            "--adapter-tier",
            "isolated-response",
            "--confirm-request-isolation",
            "--confirm-safe-test-target",
            "--request-json-template",
            '{"query":"{{input}}","options":{"mode":"safe"}}',
            "--response-json-pointer",
            "/result/answer",
            "--header-from-env",
            "X-Agent-Key=UL_ENVIRONMENT_AGENT_KEY",
        ],
    )

    assert result.exit_code == 0, result.output
    config = json.loads(target_config.read_text(encoding="utf-8"))
    assert config["environment_id"] == "isolated-response:agent.example.test"
    assert config["headers_from_env"] == {"X-Agent-Key": "UL_ENVIRONMENT_AGENT_KEY"}
    assert config["execute"] == {
        "url": "https://agent.example.test/chat",
        "request_json_template": {"query": "{{input}}", "options": {"mode": "safe"}},
        "response_json_pointer": "/result/answer",
    }
    assert "no UL-specific endpoint" in result.output


def test_init_allows_root_response_json_pointer(tmp_path: Path) -> None:
    target_config = tmp_path / "target.json"

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "init",
            str(target_config),
            "--url",
            "https://agent.example.test/chat",
            "--adapter-tier",
            "isolated-response",
            "--confirm-request-isolation",
            "--confirm-safe-test-target",
            "--response-json-pointer",
            "",
        ],
    )

    assert result.exit_code == 0, result.output
    config = json.loads(target_config.read_text(encoding="utf-8"))
    assert config["execute"]["response_json_pointer"] == ""


def test_init_rejects_invalid_isolated_mapping_before_creating_file(tmp_path: Path) -> None:
    target_config = tmp_path / "target.json"

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "init",
            str(target_config),
            "--url",
            "https://agent.example.test/chat",
            "--adapter-tier",
            "isolated-response",
            "--confirm-request-isolation",
            "--confirm-safe-test-target",
            "--request-json-template",
            '{"query":"missing placeholder"}',
        ],
    )

    assert result.exit_code != 0
    assert "exactly one {{input}}" in result.output
    assert not target_config.exists()


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
            "https://environment.example.test/execute",
        ],
    )

    assert collision.exit_code != 0
    normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", collision.output).split())
    assert "will not" in normalized_output
    assert "overwrite it" in normalized_output
    assert existing_config.read_text(encoding="utf-8") == "keep me"
