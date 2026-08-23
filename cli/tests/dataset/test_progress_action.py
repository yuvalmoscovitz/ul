from __future__ import annotations

import json
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
from typer.testing import CliRunner
from ul_cli import progress_action as progress_action_module
from ul_cli.dataset.progress import (
    CampaignProgressEvent,
    CampaignProgressTracker,
    create_campaign_next_commands,
)
from ul_cli.main import app as root_app


def test_dataset_terminal_actions_are_opaque_private_and_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_directory = tmp_path / "action-state"
    monkeypatch.setattr(
        progress_action_module,
        "_action_receipt_directory",
        lambda: receipt_directory,
    )
    private_canary = "PRIVATE_PATH_CANARY\nwith-control-\x1b[31m"
    evidence_path = tmp_path / private_canary
    next_commands = create_campaign_next_commands(evidence_path)
    expected_by_status: dict[
        Literal["paused", "cancelled", "completed"],
        tuple[Literal["resume", "diagnose", "inspect_findings"], tuple[str, ...]],
    ] = {
        "paused": (
            "resume",
            ("ul", "dataset", "evaluate", "--resume", str(evidence_path.resolve())),
        ),
        "cancelled": (
            "diagnose",
            (
                "ul",
                "dataset",
                "evaluate",
                "--resume",
                str(evidence_path.resolve()),
                "--dry-run",
            ),
        ),
        "completed": (
            "inspect_findings",
            ("ul", "dataset", "report", str(evidence_path.resolve())),
        ),
    }
    runner = CliRunner()

    for status, (expected_action, expected_argv) in expected_by_status.items():
        events: list[CampaignProgressEvent] = []
        clock_values = iter((1.0, 2.0))

        def clock(values: object = clock_values) -> float:
            return next(values)  # type: ignore[call-overload]

        tracker = CampaignProgressTracker(
            case_count=1,
            work_upper_bound=1,
            target_call_budget=1,
            semantic_call_budget=1,
            environment_call_budget=1,
            token_budget=1,
            maximum_wall_time_seconds=10,
            next_commands=next_commands,
            publish=events.append,
            clock=clock,
        )
        event = tracker.emit(status=status, stage="terminal")

        assert event.next_command is not None
        assert event.next_command.action == expected_action
        assert event.next_command.argv[:2] == ("ul", "action")
        public_payload = event.model_dump_json()
        assert private_canary not in public_payload
        assert "\x1b" not in public_payload

        executed: list[tuple[tuple[str, ...], bool, str]] = []

        def capture_run(
            argv: tuple[str, ...],
            *,
            check: bool,
            cwd: str,
            calls: list[tuple[tuple[str, ...], bool, str]] = executed,
        ) -> SimpleNamespace:
            calls.append((argv, check, cwd))
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(progress_action_module.subprocess, "run", capture_run)
        result = runner.invoke(root_app, list(event.next_command.argv[1:]))

        assert result.exit_code == 0, result.output
        assert executed == [
            (
                (
                    progress_action_module.sys.executable,
                    "-m",
                    "ul_cli.main",
                    *expected_argv[1:],
                ),
                False,
                str(Path.cwd().resolve()),
            )
        ]

    assert stat.S_IMODE(receipt_directory.stat().st_mode) == 0o700
    receipts = tuple(receipt_directory.glob("*.json"))
    assert len(receipts) == 3
    assert all(stat.S_IMODE(receipt.stat().st_mode) == 0o600 for receipt in receipts)
    receipt_payloads = [json.loads(receipt.read_text(encoding="utf-8")) for receipt in receipts]
    assert all(payload["schema_version"] == "ul.progress-action.v1" for payload in receipt_payloads)


def test_action_rejects_invalid_id_without_exposing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        progress_action_module,
        "_action_receipt_directory",
        lambda: tmp_path / "action-state",
    )

    result = CliRunner().invoke(root_app, ["action", "../../PRIVATE_CANARY"])

    assert result.exit_code == 1
    assert "Unable to resolve the progress action safely." in result.output
    assert "PRIVATE_CANARY" not in result.output


def test_windows_action_store_uses_hardened_path_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(progress_action_module, "_WINDOWS", True)
    monkeypatch.setattr(
        progress_action_module,
        "_action_receipt_directory",
        lambda: tmp_path / "action-state",
    )
    public_argv = progress_action_module.create_progress_action(
        ("ul", "dataset", "report", str(tmp_path / "evidence.jsonl"))
    )

    receipt = progress_action_module._read_progress_action(public_argv[-1])

    assert receipt.argv[:3] == ("ul", "dataset", "report")
    assert receipt.working_directory == str(Path.cwd().resolve())
