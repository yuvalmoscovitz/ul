from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner
from ul_cli.main import app

runner = CliRunner()


def _write_inputs(tmp_path: Path, *, observation_authority: str) -> tuple[Path, Path, Path]:
    case_path = tmp_path / "case.json"
    target_path = tmp_path / "target.json"
    invariants_path = tmp_path / "invariants.json"
    case_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "id": "invoice-retry-after-commit",
                "operator_id": "conversation.retry_after_successful_commit",
                "operator_version": "1.0.0",
                "conversation": [
                    {"id": "initial-payment", "role": "user", "content": "Pay AC-100."},
                    {
                        "id": "explicit-retry",
                        "role": "user",
                        "content": "Retry the same payment for AC-100.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    target_path.write_text(
        json.dumps(
            {
                "version": 5,
                "environment_id": "retry-test-environment",
                "reset": {
                    "url": "http://127.0.0.1:8765/reset",
                    "generation_json_pointer": "/generation",
                    "clean_state_json_pointer": "/clean",
                    "clean_state_value": True,
                },
                "setup": {"url": "http://127.0.0.1:8765/setup"},
                "execute_turn": {
                    "url": "http://127.0.0.1:8765/execute",
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "turn_id": "{{turn_id}}",
                        "input": "{{input}}",
                    },
                },
                "snapshot": {
                    "url": "http://127.0.0.1:8765/snapshot",
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "turn_id": "{{turn_id}}",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    invariants_path.write_text(
        json.dumps(
            {
                "schema_version": "1.1.0",
                "observation_source": "target_output",
                "observation_authority": observation_authority,
                "rules": [
                    {
                        "type": "json_value_equals_literal",
                        "id": "exactly-one-committed-payment",
                        "version": "1.0.0",
                        "description": "The invoice has exactly one committed payment.",
                        "severity": "critical",
                        "value_pointer": "/committed_effect_count",
                        "literal": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return case_path, target_path, invariants_path


def test_retry_dry_run_reports_versioned_complete_plan_without_calls(tmp_path: Path) -> None:
    case_path, target_path, invariants_path = _write_inputs(
        tmp_path, observation_authority="committed_state_snapshot"
    )

    result = runner.invoke(
        app,
        [
            "stress",
            "retry-after-successful-commit",
            str(case_path),
            "--environment-config",
            str(target_path),
            "--invariants",
            str(invariants_path),
            "--allow-insecure-http",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "conversation.retry_after_successful_commit@1.0.0" in result.output
    assert "initial committed operation -> explicit retry" in result.output
    assert "Target calls per paired repetition: 14" in result.output
    assert "Potential environment API calls: 42" in result.output
    assert "External calls: none" in result.output


def test_retry_rejects_agent_response_rules_before_network_calls(tmp_path: Path) -> None:
    case_path, target_path, invariants_path = _write_inputs(
        tmp_path, observation_authority="agent_response"
    )

    result = runner.invoke(
        app,
        [
            "stress",
            "retry-after-successful-commit",
            str(case_path),
            "--environment-config",
            str(target_path),
            "--invariants",
            str(invariants_path),
            "--allow-environment-network",
            "--confirm-test-environment",
            "--allow-insecure-http",
            "--output",
            str(tmp_path / "evidence.json"),
        ],
    )

    assert result.exit_code == 2
    assert "requires committed-state invariant" in result.output
    assert "observation" in result.output
    assert not (tmp_path / "evidence.json").exists()
