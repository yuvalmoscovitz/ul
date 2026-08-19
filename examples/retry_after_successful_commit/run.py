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
from typing import Annotated, TextIO, cast

import typer
from pydantic import ValidationError
from ul.event_stress import RetryAfterSuccessfulCommitStressResult

from examples.retry_after_successful_commit.defective_agent import create_server

_EXAMPLE_DIRECTORY = Path(__file__).resolve().parent
_PROJECT_DIRECTORY = _EXAMPLE_DIRECTORY.parents[1]
_CASE_PATH = _EXAMPLE_DIRECTORY / "case.json"
_INVARIANTS_PATH = _EXAMPLE_DIRECTORY / "invariants.json"
_TARGET_TEMPLATE_PATH = _EXAMPLE_DIRECTORY / "target.json"
_EXPECTED_RULE_IDS = {
    "exactly-one-committed-payment",
    "committed-payments-unique-by-invoice",
}


def _create_private_file(path: Path) -> TextIO:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("example artifact is not a regular file")
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "w", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise


def load_target_template(base_url: str) -> dict[str, object]:
    with _TARGET_TEMPLATE_PATH.open(encoding="utf-8") as target_template_file:
        untyped_target_config: object = json.load(target_template_file)
    if type(untyped_target_config) is not dict:
        raise ValueError("retry example target configuration must be a JSON object")
    target_config = cast(dict[str, object], untyped_target_config)
    for phase, endpoint in {
        "reset": "reset",
        "setup": "setup",
        "execute_turn": "execute",
        "snapshot": "snapshot",
    }.items():
        phase_config = cast(dict[str, object], target_config[phase])
        phase_config["url"] = f"{base_url}/{endpoint}"
    return target_config


def _subprocess_environment() -> dict[str, str]:
    environment = {"PYTHONUTF8": "1"}
    for name in ("SYSTEMROOT", "WINDIR"):
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


def _evidence_confirms_repeatable_duplicate(evidence_path: Path) -> bool:
    try:
        result = RetryAfterSuccessfulCommitStressResult.model_validate_json(
            evidence_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValidationError, ValueError):
        return False
    checkpoints = (
        result.baseline_invariant_rules,
        result.successful_commit_invariant_rules,
        result.retried_invariant_rules,
    )
    if any({rule.rule_id for rule in rules} != _EXPECTED_RULE_IDS for rules in checkpoints):
        return False
    return (
        result.case.operator_id == "event.retry_after_successful_commit"
        and result.case.operator_version == "1.0.0"
        and result.status == "failed"
        and result.requested_repetitions == 3
        and not result.baseline_drift_observed
        and all(trial.inconclusive_reason is None for trial in result.trials)
        and all(rule.status == "satisfied" for rule in result.baseline_invariant_rules)
        and all(rule.status == "satisfied" for rule in result.successful_commit_invariant_rules)
        and all(rule.status == "violated" for rule in result.retried_invariant_rules)
        and all(
            len(rule.trials) == 3 and all(trial.status == "violated" for trial in rule.trials)
            for rule in result.retried_invariant_rules
        )
        and all(
            len(trial.variation) == 2
            and _committed_effect_count(trial.variation[0].committed_state_snapshot) == 1
            and _committed_effect_count(trial.variation[1].committed_state_snapshot) == 2
            for trial in result.trials
        )
    )


def _committed_effect_count(snapshot: object) -> int | None:
    if type(snapshot) is not dict:
        return None
    value = cast(dict[str, object], snapshot).get("committed_effect_count")
    return value if type(value) is int else None


def main(
    output: Annotated[
        Path | None,
        typer.Option(help="New private evidence file; defaults to a unique path under tmp."),
    ] = None,
) -> None:
    if output is not None and output.exists():
        typer.echo("Retry example could not run: output already exists", err=True)
        raise typer.Exit(code=2)
    temporary_root = _PROJECT_DIRECTORY / "tmp"
    temporary_root.mkdir(exist_ok=True)
    artifact_directory = Path(tempfile.mkdtemp(prefix="retry-after-commit-", dir=temporary_root))
    target_config_path = artifact_directory / "target.json"
    evidence_path = output.resolve() if output is not None else artifact_directory / "evidence.json"
    server: ThreadingHTTPServer | None = None
    server_thread: Thread | None = None
    completed_process: subprocess.CompletedProcess[str]

    try:
        server = create_server(0)
        server_host, server_port = cast(tuple[str, int], server.server_address)
        target_config = load_target_template(f"http://{server_host}:{server_port}")
        with _create_private_file(target_config_path) as target_config_file:
            json.dump(target_config, target_config_file, indent=2)
            target_config_file.write("\n")
        server_thread = Thread(
            target=server.serve_forever,
            name="retry-after-successful-commit-agent",
            daemon=True,
        )
        server_thread.start()
        command = [
            sys.executable,
            "-m",
            "ul_cli.main",
            "stress",
            "retry-after-successful-commit",
            str(_CASE_PATH),
            "--sandbox-config",
            str(target_config_path),
            "--invariants",
            str(_INVARIANTS_PATH),
            "--output",
            str(evidence_path),
            "--repetitions",
            "3",
            "--max-sandbox-api-calls",
            "42",
            "--allow-sandbox-network-egress",
            "--allow-insecure-http",
            "--confirm-isolated-sandbox",
        ]
        completed_process = subprocess.run(
            command,
            cwd=_PROJECT_DIRECTORY,
            env=_subprocess_environment(),
            check=False,
            shell=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        typer.echo(f"Retry example could not run: {error.__class__.__name__}", err=True)
        raise typer.Exit(code=2) from None
    finally:
        if server is not None:
            if server_thread is not None and server_thread.ident is not None:
                server.shutdown()
            server.server_close()
            if server_thread is not None and server_thread.ident is not None:
                server_thread.join(timeout=5)
        target_config_path.unlink(missing_ok=True)
        if artifact_directory.exists() and not any(artifact_directory.iterdir()):
            artifact_directory.rmdir()

    if completed_process.returncode == 1 and _evidence_confirms_repeatable_duplicate(evidence_path):
        typer.echo(
            "Confirmed: after a successful committed payment, the explicit retry created a "
            "second payment in all 3 repetitions."
        )
        typer.echo("Critical rules: exactly-once count and unique invoice effect both violated.")
        typer.echo("Semantic-model calls: none")
        typer.echo(f"Evidence: {evidence_path}")
        return

    if evidence_path.exists():
        typer.echo(
            f"UL did not confirm the expected retry finding. Review: {evidence_path}", err=True
        )
    else:
        typer.echo("UL did not produce retry evidence.", err=True)
    raise typer.Exit(code=completed_process.returncode or 1)


if __name__ == "__main__":
    typer.run(main)
