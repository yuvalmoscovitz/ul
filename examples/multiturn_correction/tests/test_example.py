from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from threading import Thread
from typing import cast

from typer.testing import CliRunner
from ul_cli.main import app

from examples.multiturn_correction.defective_agent import create_server

_EXAMPLE_DIRECTORY = Path(__file__).parents[1]
_PROJECT_DIRECTORY = _EXAMPLE_DIRECTORY.parents[1]


def test_runnable_example_reports_real_repeatable_correction_failure(tmp_path: Path) -> None:
    server = create_server(0)
    host, port = cast(tuple[str, int], server.server_address)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    target = json.loads((_EXAMPLE_DIRECTORY / "target.json").read_text(encoding="utf-8"))
    for phase, endpoint in {
        "reset": "reset",
        "setup": "setup",
        "execute_turn": "execute",
        "snapshot": "snapshot",
    }.items():
        target[phase]["url"] = f"http://{host}:{port}/{endpoint}"
    target_path = tmp_path / "target.json"
    target_path.write_text(json.dumps(target), encoding="utf-8")
    evidence_path = tmp_path / "evidence.json"
    server_thread.start()
    try:
        result = CliRunner().invoke(
            app,
            [
                "stress",
                "correction",
                str(_EXAMPLE_DIRECTORY / "case.json"),
                "--environment-config",
                str(target_path),
                "--invariants",
                str(_EXAMPLE_DIRECTORY / "invariants.json"),
                "--output",
                str(evidence_path),
                "--allow-environment-network",
                "--allow-insecure-http",
                "--confirm-test-environment",
                "--max-environment-api-calls",
                "42",
            ],
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()

    assert result.exit_code == 1, result.output
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["status"] == "failed"
    assert evidence["case"]["operator_id"] == "conversation.correction_after_first_response"
    assert evidence["case"]["operator_version"] == "1.0.0"
    assert evidence["first_response_divergence_turn_id"] == "corrected-request"
    assert evidence["first_committed_state_divergence_turn_id"] == "corrected-request"
    assert evidence["baseline_invariant_rules"][0]["status"] == "satisfied"
    assert evidence["corrected_invariant_rules"][0]["status"] == "violated"
    assert evidence["trials"][0]["variation"][1]["committed_state_snapshot"] == {
        "committed_invoice": "AC-100",
        "requested_invoice": "AC-101",
    }
    report = CliRunner().invoke(app, ["report", str(evidence_path), "--json"])
    assert report.exit_code == 1, report.output
    report_payload = json.loads(report.output)
    assert report_payload["evidence_type"] == "correction_after_first_response"
    assert report_payload["status"] == "failed"
    assert report_payload["summary"]["finding_count"] == 1
    assert "AC-100" not in report.output
    assert "AC-101" not in report.output
    human_report = CliRunner().invoke(app, ["report", str(evidence_path)])
    assert human_report.exit_code == 1, human_report.output
    assert "Evidence type: correction after first response" in human_report.output
    assert "Status: failed (exit 1)" in human_report.output
    assert "The agent violated a customer-defined rule." in human_report.output
    assert "no dedicated stateful detail command is available" in human_report.output
    assert "AC-100" not in human_report.output
    assert "AC-101" not in human_report.output


def test_one_command_runner_confirms_finding_without_model_configuration() -> None:
    checked_in_target_before = (_EXAMPLE_DIRECTORY / "target.json").read_bytes()
    environment = dict(os.environ)
    for name in tuple(environment):
        if name in {
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "OPENAI_API_KEY",
            "OPEN_ROUTER_API_KEY",
            "UL_LIVE",
        } or name.startswith("UL_DATASET_"):
            environment.pop(name)

    completed_process = subprocess.run(
        [sys.executable, "-m", "examples.multiturn_correction.run"],
        cwd=_PROJECT_DIRECTORY,
        env=environment,
        check=False,
        shell=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed_process.returncode == 0, completed_process.stderr
    assert "Confirmed deterministic finding: the later correction was ignored." in (
        completed_process.stdout
    )
    assert "Repetitions: 3/3" in completed_process.stdout
    assert "severity=critical; baseline=satisfied; corrected=violated" in (completed_process.stdout)
    assert "No API key or UL semantic-model calls used." in completed_process.stdout
    assert "Pay invoice" not in completed_process.stdout
    assert "AC-100" not in completed_process.stdout
    evidence_match = re.search(r"^Evidence: (.+)$", completed_process.stdout, re.MULTILINE)
    assert evidence_match is not None
    evidence_path = Path(evidence_match.group(1))
    try:
        assert evidence_path.is_absolute()
        assert evidence_path.parent.name.startswith("multiturn-correction-")
        if os.name != "nt":
            assert evidence_path.stat().st_mode & 0o777 == 0o600
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        assert evidence["status"] == "failed"
        assert evidence["case"]["operator_version"] == "1.0.0"
        assert evidence["baseline_invariant_rules"][0]["status"] == "satisfied"
        assert evidence["corrected_invariant_rules"][0]["status"] == "violated"
        assert not (evidence_path.parent / "target.json").exists()
        assert (_EXAMPLE_DIRECTORY / "target.json").read_bytes() == checked_in_target_before
    finally:
        evidence_path.unlink(missing_ok=True)
        evidence_path.parent.rmdir()
