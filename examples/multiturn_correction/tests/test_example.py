from __future__ import annotations

import json
from pathlib import Path
from threading import Thread
from typing import cast

from typer.testing import CliRunner
from ul_cli.main import app

from examples.multiturn_correction.defective_agent import create_server

_EXAMPLE_DIRECTORY = Path(__file__).parents[1]


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
                "--sandbox-config",
                str(target_path),
                "--invariants",
                str(_EXAMPLE_DIRECTORY / "invariants.json"),
                "--output",
                str(evidence_path),
                "--allow-sandbox-network-egress",
                "--allow-insecure-http",
                "--confirm-isolated-sandbox",
                "--max-sandbox-api-calls",
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
    assert evidence["first_response_divergence_turn_id"] == "corrected-request"
    assert evidence["first_committed_state_divergence_turn_id"] == "corrected-request"
    assert evidence["baseline_invariant_rules"][0]["status"] == "satisfied"
    assert evidence["corrected_invariant_rules"][0]["status"] == "violated"
    assert evidence["trials"][0]["variation"][1]["committed_state_snapshot"] == {
        "committed_invoice": "AC-100",
        "requested_invoice": "AC-101",
    }
