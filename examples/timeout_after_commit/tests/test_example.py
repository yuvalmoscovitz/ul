from __future__ import annotations

import json
from pathlib import Path
from threading import Thread
from typing import Literal, cast

import pytest
from typer.testing import CliRunner
from ul_cli.main import app

from examples.timeout_after_commit.sandbox import create_server

_EXAMPLE_DIRECTORY = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("retry_mode", "fire_event", "expected_exit", "expected_status", "expected_count"),
    [
        ("defective", True, 1, "failed", 2),
        ("safe", True, 0, "passed", 1),
        ("defective", False, 2, "inconclusive", 1),
    ],
)
def test_runnable_timeout_after_commit_variants(
    tmp_path: Path,
    retry_mode: Literal["safe", "defective"],
    fire_event: bool,
    expected_exit: int,
    expected_status: str,
    expected_count: int,
) -> None:
    server = create_server(0, retry_mode=retry_mode, fire_event=fire_event)
    host, port = cast(tuple[str, int], server.server_address)
    target_path = _write_target(tmp_path, host, port)
    evidence_path = tmp_path / "evidence.json"
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        result = _invoke(target_path, evidence_path)
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()

    assert result.exit_code == expected_exit, result.output
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["status"] == expected_status
    assert len(evidence["trials"]) == 3
    for trial in evidence["trials"]:
        execution = trial["execution_evidence"]
        event = execution["timeout_after_commit_event"]
        assert event == {
            "operator_id": "tool.timeout_after_commit",
            "operator_version": "1.0.0",
            "event_id": "lose-first-payment-acknowledgement",
            "turn_id": "submit-payment",
            "action_id": "execute-payment",
            "authority": "sandbox_self_reported",
            "requested": True,
            "armed": True,
            "trigger_status": "fired" if fire_event else "not_fired",
            "cleaned": True,
        }
        assert execution["final_state"]["value"]["matching_payment_count"] == expected_count
    expected_rule_status = (
        "not_evaluable" if not fire_event else "satisfied" if retry_mode == "safe" else "violated"
    )
    assert evidence["invariant_rules"][0]["status"] == expected_rule_status


def test_unsupported_capability_fails_preflight_without_a_request(tmp_path: Path) -> None:
    server = create_server(0)
    host, port = cast(tuple[str, int], server.server_address)
    target_path = _write_target(tmp_path, host, port, include_capability=False)
    result = _invoke(target_path, tmp_path / "evidence.json")

    assert result.exit_code == 2
    assert "does not support tool.timeout_after_commit@1.0.0" in result.output
    assert server.state.request_count == 0
    server.server_close()


def _invoke(target_path: Path, evidence_path: Path):
    return CliRunner().invoke(
        app,
        [
            "stress",
            "timeout-after-commit",
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
            "27",
        ],
    )


def _write_target(
    tmp_path: Path,
    host: str,
    port: int,
    *,
    include_capability: bool = True,
) -> Path:
    target = json.loads((_EXAMPLE_DIRECTORY / "target.json").read_text(encoding="utf-8"))
    for phase, endpoint in {
        "reset": "reset",
        "setup": "setup",
        "execute_turn": "execute",
        "snapshot": "snapshot",
        "timeout_after_commit": "timeout-after-commit",
    }.items():
        target[phase]["url"] = f"http://{host}:{port}/{endpoint}"
    if not include_capability:
        del target["timeout_after_commit"]
    target_path = tmp_path / "target.json"
    target_path.write_text(json.dumps(target), encoding="utf-8")
    return target_path
