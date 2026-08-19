from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import TextIO, cast

import typer
from pydantic import ValidationError
from ul.event_stress import CorrectionStressResult

from examples.multiturn_correction.defective_agent import create_server

_EXAMPLE_DIRECTORY = Path(__file__).resolve().parent
_PROJECT_DIRECTORY = _EXAMPLE_DIRECTORY.parents[1]
_CASE_PATH = _EXAMPLE_DIRECTORY / "case.json"
_INVARIANTS_PATH = _EXAMPLE_DIRECTORY / "invariants.json"
_TARGET_TEMPLATE_PATH = _EXAMPLE_DIRECTORY / "target.json"
_MAXIMUM_EVIDENCE_BYTES = 1_000_000


def _create_private_file(path: Path) -> TextIO:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("artifact is not a regular file")
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "w", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise


def _load_target_config(base_url: str) -> dict[str, object]:
    with _TARGET_TEMPLATE_PATH.open(encoding="utf-8") as target_file:
        untyped_target: object = json.load(target_file)
    if type(untyped_target) is not dict:
        raise ValueError("target configuration must be a JSON object")
    target = cast(dict[str, object], untyped_target)
    for phase, endpoint in {
        "reset": "reset",
        "setup": "setup",
        "execute_turn": "execute",
        "snapshot": "snapshot",
    }.items():
        phase_config = target.get(phase)
        if type(phase_config) is not dict:
            raise ValueError(f"target configuration is missing {phase}")
        cast(dict[str, object], phase_config)["url"] = f"{base_url}/{endpoint}"
    return target


def _subprocess_environment() -> dict[str, str]:
    environment = {"PYTHONUTF8": "1"}
    for name in ("SYSTEMROOT", "WINDIR"):
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


def _evidence_confirms_expected_finding(evidence_path: Path) -> bool:
    try:
        evidence_stat = evidence_path.lstat()
        if (
            not stat.S_ISREG(evidence_stat.st_mode)
            or evidence_stat.st_size > _MAXIMUM_EVIDENCE_BYTES
            or (os.name != "nt" and evidence_stat.st_mode & 0o777 != 0o600)
        ):
            return False
        result = CorrectionStressResult.model_validate_json(
            evidence_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValidationError):
        return False
    baseline_rules = result.baseline_invariant_rules
    corrected_rules = result.corrected_invariant_rules
    return (
        result.status == "failed"
        and result.case.operator_id == "event.correction_after_first_response"
        and result.requested_repetitions == 3
        and result.first_response_divergence_turn_id == "corrected-request"
        and result.first_committed_state_divergence_turn_id == "corrected-request"
        and result.response_divergence_stability == "stable"
        and result.committed_state_divergence_stability == "stable"
        and result.response_divergence_counts == {"initial-request": 0, "corrected-request": 3}
        and result.committed_state_divergence_counts
        == {"initial-request": 0, "corrected-request": 3}
        and not result.baseline_drift_observed
        and len(baseline_rules) == 1
        and len(corrected_rules) == 1
        and baseline_rules[0].rule_id == "committed-invoice-follows-latest-request"
        and baseline_rules[0].severity == "critical"
        and baseline_rules[0].status == "satisfied"
        and corrected_rules[0].rule_id == "committed-invoice-follows-latest-request"
        and corrected_rules[0].severity == "critical"
        and corrected_rules[0].status == "violated"
        and all(trial.inconclusive_reason is None for trial in result.trials)
    )


def main() -> None:
    temporary_root = _PROJECT_DIRECTORY / "tmp"
    temporary_root.mkdir(exist_ok=True)
    artifact_directory = Path(tempfile.mkdtemp(prefix="multiturn-correction-", dir=temporary_root))
    target_config_path = artifact_directory / "target.json"
    evidence_path = artifact_directory / "evidence.json"
    server: ThreadingHTTPServer | None = None
    server_thread: Thread | None = None
    completed_process: subprocess.CompletedProcess[str] | None = None

    try:
        server = create_server(0)
        host, port = cast(tuple[str, int], server.server_address)
        target_config = _load_target_config(f"http://{host}:{port}")
        with _create_private_file(target_config_path) as target_file:
            json.dump(target_config, target_file, indent=2)
            target_file.write("\n")

        server_thread = Thread(
            target=server.serve_forever,
            name="multiturn-correction-agent",
            daemon=True,
        )
        server_thread.start()
        completed_process = subprocess.run(
            [
                sys.executable,
                "-m",
                "ul_cli.main",
                "stress",
                "correction",
                str(_CASE_PATH),
                "--sandbox-config",
                str(target_config_path),
                "--invariants",
                str(_INVARIANTS_PATH),
                "--output",
                str(evidence_path),
                "--allow-sandbox-network-egress",
                "--allow-insecure-http",
                "--confirm-isolated-sandbox",
                "--max-sandbox-api-calls",
                "42",
            ],
            cwd=_PROJECT_DIRECTORY,
            env=_subprocess_environment(),
            check=False,
            shell=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired):
        typer.echo("Offline finding example could not complete setup or execution.", err=True)
        raise typer.Exit(code=2) from None
    finally:
        if server is not None:
            if server_thread is not None and server_thread.ident is not None:
                server.shutdown()
            server.server_close()
            if server_thread is not None and server_thread.ident is not None:
                server_thread.join(timeout=5)
        target_config_path.unlink(missing_ok=True)
        if not evidence_path.exists():
            artifact_directory.rmdir()

    if completed_process.returncode == 1 and _evidence_confirms_expected_finding(evidence_path):
        typer.echo("Confirmed deterministic finding: the later correction was ignored.")
        typer.echo("Repetitions: 3/3 produced the same response and committed-state failure.")
        typer.echo(
            "Invariant: committed-invoice-follows-latest-request; severity=critical; "
            "baseline=satisfied; corrected=violated"
        )
        typer.echo(f"Evidence: {evidence_path.resolve()}")
        typer.echo(
            "No API key or UL semantic-model calls used. Sandbox traffic stayed on loopback."
        )
        return

    if evidence_path.exists():
        typer.echo(
            f"The expected deterministic finding was not confirmed. Evidence: "
            f"{evidence_path.resolve()}",
            err=True,
        )
    else:
        typer.echo("The example did not produce evidence.", err=True)
    if completed_process.returncode not in {0, 1}:
        raise typer.Exit(code=2)
    raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
