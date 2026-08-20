from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from threading import Lock, Thread
from typing import ClassVar, TextIO, cast

import typer
from platformdirs import user_data_path
from pydantic import ValidationError
from ul.event_stress import RetryAfterSuccessfulCommitStressResult

_EXPECTED_RULE_IDS = {
    "exactly-one-committed-payment",
    "committed-payments-unique-by-invoice",
    "one-new-payment-per-turn",
}


class DemoRetryHandler(BaseHTTPRequestHandler):
    environment_id: ClassVar[str] = "retry-after-successful-commit-demo"
    state_lock: ClassVar[Lock] = Lock()
    generation: ClassVar[int] = 0
    committed_effects: ClassVar[list[dict[str, str]]] = []

    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("content-length", "0"))
        except ValueError:
            self._send(HTTPStatus.BAD_REQUEST, {"error": "invalid content length"})
            return
        if content_length < 0 or content_length > 100_000:
            self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"})
            return
        try:
            raw_request: object = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, ValueError):
            self._send(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON"})
            return
        if not isinstance(raw_request, dict):
            self._send(HTTPStatus.BAD_REQUEST, {"error": "JSON object required"})
            return
        request = cast(dict[str, object], raw_request)
        if self.path == "/reset":
            with type(self).state_lock:
                type(self).generation += 1
                type(self).committed_effects = []
                generation = type(self).generation
            self._send(
                HTTPStatus.OK,
                {
                    "case_id": request.get("case_id"),
                    "generation": generation,
                    "clean": True,
                    "reset_session": request.get("reset_session") is True,
                    "reset_env": request.get("reset_env") is True,
                },
            )
        elif self.path == "/setup":
            self._send(HTTPStatus.OK, {"case_id": request.get("case_id")})
        elif self.path == "/snapshot":
            with type(self).state_lock:
                committed_effects = [dict(effect) for effect in type(self).committed_effects]
            self._send(
                HTTPStatus.OK,
                {
                    "case_id": request.get("case_id"),
                    "turn_id": request.get("turn_id"),
                    "state": {
                        "committed_effect_count": len(committed_effects),
                        "committed_effects": committed_effects,
                    },
                },
            )
        elif self.path == "/execute":
            raw_input = request.get("input")
            if not isinstance(raw_input, str) or "AC-100" not in raw_input:
                self._send(HTTPStatus.BAD_REQUEST, {"error": "AC-100 input required"})
                return
            with type(self).state_lock:
                attempt = len(type(self).committed_effects) + 1
                type(self).committed_effects.append(
                    {
                        "payment_id": f"payment-{attempt}",
                        "invoice_reference": "AC-100",
                        "idempotency_key": f"invoice:AC-100:attempt-{attempt}",
                    }
                )
            response = (
                "Payment committed." if attempt == 1 else "Payment retried and committed again."
            )
            self._send(
                HTTPStatus.OK,
                {
                    "case_id": request.get("case_id"),
                    "turn_id": request.get("turn_id"),
                    "response": response,
                },
            )
        else:
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _send(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(
            {**payload, "environment_id": self.environment_id}, separators=(",", ":")
        ).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def create_server(port: int = 0) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), DemoRetryHandler)


def _create_private_file(path: Path) -> TextIO:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("demo artifact is not a regular file")
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "w", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise


def _create_demo_artifact_directory() -> Path:
    demo_directory = user_data_path("ul", appauthor=False) / "demo"
    demo_directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        demo_directory.mkdir(mode=0o700)
    except FileExistsError:
        directory_status = demo_directory.lstat()
        if stat.S_ISLNK(directory_status.st_mode) or not stat.S_ISDIR(directory_status.st_mode):
            raise ValueError("UL demo data path must be a directory, not a symlink") from None
        if os.name != "nt":
            os.chmod(demo_directory, 0o700)
    return Path(tempfile.mkdtemp(prefix="retry-after-commit-", dir=demo_directory))


def _load_target_template(base_url: str) -> dict[str, object]:
    target_resource = resources.files("ul_cli.demo_assets").joinpath("target.json")
    untyped_target_config: object = json.loads(target_resource.read_text(encoding="utf-8"))
    if type(untyped_target_config) is not dict:
        raise ValueError("demo target configuration must be a JSON object")
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
        result.case.operator_id == "conversation.retry_after_successful_commit"
        and result.case.operator_version == "1.0.0"
        and result.status == "failed"
        and result.requested_repetitions == 3
        and not result.baseline_drift_observed
        and all(trial.inconclusive_reason is None for trial in result.trials)
        and all(rule.status == "satisfied" for rule in result.baseline_invariant_rules)
        and all(rule.status == "satisfied" for rule in result.successful_commit_invariant_rules)
        and all(
            rule.status
            == ("satisfied" if rule.rule_id == "one-new-payment-per-turn" else "violated")
            for rule in result.retried_invariant_rules
        )
        and all(
            len(rule.trials) == 3
            and all(
                trial.status
                == ("satisfied" if rule.rule_id == "one-new-payment-per-turn" else "violated")
                for trial in rule.trials
            )
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


def run_demo(output: Path | None = None) -> None:
    if output is not None and output.exists():
        typer.echo("Demo could not run: output already exists", err=True)
        raise typer.Exit(code=2)
    artifact_directory: Path | None = None
    target_config_path: Path | None = None
    evidence_path: Path | None = None
    server: ThreadingHTTPServer | None = None
    server_thread: Thread | None = None
    completed_process: subprocess.CompletedProcess[str]

    try:
        artifact_directory = _create_demo_artifact_directory()
        target_config_path = artifact_directory / "target.json"
        evidence_path = (
            output.resolve() if output is not None else artifact_directory / "evidence.json"
        )
        server = create_server()
        server_host, server_port = cast(tuple[str, int], server.server_address)
        target_config = _load_target_template(f"http://{server_host}:{server_port}")
        with _create_private_file(target_config_path) as target_config_file:
            json.dump(target_config, target_config_file, indent=2)
            target_config_file.write("\n")
        server_thread = Thread(
            target=server.serve_forever,
            name="ul-demo-agent",
            daemon=True,
        )
        server_thread.start()
        with ExitStack() as resource_stack:
            asset_root = resources.files("ul_cli.demo_assets")
            case_path = resource_stack.enter_context(
                resources.as_file(asset_root.joinpath("case.json"))
            )
            invariants_path = resource_stack.enter_context(
                resources.as_file(asset_root.joinpath("invariants.json"))
            )
            command = [
                sys.executable,
                "-m",
                "ul_cli.main",
                "stress",
                "retry-after-successful-commit",
                str(case_path),
                "--environment-config",
                str(target_config_path),
                "--invariants",
                str(invariants_path),
                "--output",
                str(evidence_path),
                "--repetitions",
                "3",
                "--max-environment-api-calls",
                "42",
                "--allow-environment-network",
                "--allow-insecure-http",
                "--confirm-test-environment",
            ]
            completed_process = subprocess.run(
                command,
                cwd=Path.cwd(),
                env=_subprocess_environment(),
                check=False,
                shell=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
    except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        typer.echo(f"Demo could not run: {error.__class__.__name__}", err=True)
        raise typer.Exit(code=2) from None
    finally:
        if server is not None:
            if server_thread is not None and server_thread.ident is not None:
                server.shutdown()
            server.server_close()
            if server_thread is not None and server_thread.ident is not None:
                server_thread.join(timeout=5)
        if target_config_path is not None:
            target_config_path.unlink(missing_ok=True)
        if (
            artifact_directory is not None
            and artifact_directory.exists()
            and not any(artifact_directory.iterdir())
        ):
            artifact_directory.rmdir()

    assert evidence_path is not None
    if completed_process.returncode == 1 and _evidence_confirms_repeatable_duplicate(evidence_path):
        typer.echo(
            "Confirmed: after a successful committed payment, the explicit retry created a "
            "second payment in all 3 repetitions."
        )
        typer.echo("Critical rules: exactly-once count and unique invoice effect both violated.")
        typer.echo("Transition rule: each individual turn appended exactly one effect.")
        typer.echo("Semantic-model calls: none")
        typer.echo("External-network calls: none (synthetic localhost environment only)")
        typer.echo(f"Evidence: {evidence_path}")
        typer.echo("Next: run 'ul init --help' to connect your own test environment.")
        return

    if evidence_path.exists():
        typer.echo(f"UL did not confirm the expected finding. Review: {evidence_path}", err=True)
    else:
        typer.echo("UL did not produce demo evidence.", err=True)
    raise typer.Exit(code=completed_process.returncode or 1)
