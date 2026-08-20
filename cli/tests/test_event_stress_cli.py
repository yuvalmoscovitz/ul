from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner
from ul_cli.main import app

runner = CliRunner()


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    case = tmp_path / "case.json"
    target = tmp_path / "target.json"
    invariants = tmp_path / "invariants.json"
    case.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "id": "invoice-correction",
                "operator_id": "conversation.correction_after_first_response",
                "operator_version": "1.0.0",
                "conversation": [
                    {"id": "initial", "role": "user", "content": "Pay AC-100."},
                    {
                        "id": "correction",
                        "role": "user",
                        "content": "Correction: pay AC-101 instead.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    target.write_text(
        json.dumps(
            {
                "version": 4,
                "sandbox_id": "test-sandbox",
                "reset": {
                    "url": "http://127.0.0.1:8765/reset",
                    "request_json_template": {"case_id": "{{case_id}}"},
                    "case_id_json_pointer": "/case_id",
                    "generation_json_pointer": "/generation",
                    "clean_state_json_pointer": "/clean",
                    "clean_state_value": True,
                },
                "setup": {
                    "url": "http://127.0.0.1:8765/setup",
                    "request_json_template": {"case_id": "{{case_id}}"},
                    "case_id_json_pointer": "/case_id",
                },
                "execute_turn": {
                    "url": "http://127.0.0.1:8765/execute",
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "turn_id": "{{turn_id}}",
                        "input": "{{input}}",
                    },
                    "case_id_json_pointer": "/case_id",
                    "turn_id_json_pointer": "/turn_id",
                },
                "snapshot": {
                    "url": "http://127.0.0.1:8765/snapshot",
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "turn_id": "{{turn_id}}",
                    },
                    "case_id_json_pointer": "/case_id",
                    "turn_id_json_pointer": "/turn_id",
                },
            }
        ),
        encoding="utf-8",
    )
    invariants.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "observation_source": "target_output",
                "observation_authority": "committed_state_snapshot",
                "rules": [
                    {
                        "type": "json_values_equal",
                        "id": "committed-follows-request",
                        "version": "1.0.0",
                        "description": "Committed invoice follows latest request.",
                        "severity": "critical",
                        "left_pointer": "/committed_invoice",
                        "right_pointer": "/requested_invoice",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return case, target, invariants


def test_correction_dry_run_reports_complete_plan_without_calls(tmp_path: Path) -> None:
    case, target, invariants = _write_inputs(tmp_path)

    result = runner.invoke(
        app,
        [
            "stress",
            "correction",
            str(case),
            "--sandbox-config",
            str(target),
            "--invariants",
            str(invariants),
            "--allow-insecure-http",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "conversation.correction_after_first_response@1.0.0" in result.output
    assert "Target calls per paired repetition: 14" in result.output
    assert "Potential sandbox API calls: 42" in result.output
    assert "External calls: none" in result.output


def test_save_creates_replayable_multi_turn_case(tmp_path: Path) -> None:
    case, target, invariants = _write_inputs(tmp_path)
    output = tmp_path / "regression.json"

    result = runner.invoke(
        app,
        [
            "stress",
            "save",
            str(case),
            "--sandbox-config",
            str(target),
            "--invariants",
            str(invariants),
            "--output",
            str(output),
            "--confirm-versioned-input",
        ],
    )

    assert result.exit_code == 0, result.output
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["case_id"].startswith("ulmc_v1_")
    assert saved["stress_case"]["conversation"][1]["id"] == "correction"
