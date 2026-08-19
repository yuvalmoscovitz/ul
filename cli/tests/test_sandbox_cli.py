from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner
from ul_cli.main import app
from ul_cli.sandbox import _diagnose_failure

runner = CliRunner()


class _SandboxServer(ThreadingHTTPServer):
    requests: list[tuple[str, dict[str, Any]]]
    generation: int
    execute_content_type: str
    execute_status: int
    cleanup_generation_changes: bool


@contextmanager
def _sandbox_server(
    *,
    execute_content_type: str = "application/json",
    execute_status: int = 200,
    cleanup_generation_changes: bool = True,
) -> Iterator[_SandboxServer]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers["Content-Length"])
            request_body = json.loads(self.rfile.read(content_length))
            server = self.server
            assert isinstance(server, _SandboxServer)
            server.requests.append((self.path, request_body))
            if self.path == "/reset":
                if server.generation == 0 or server.cleanup_generation_changes:
                    server.generation += 1
                response = {
                    "sandbox_id": "check-sandbox",
                    "case_id": request_body["case_id"],
                    "generation": server.generation,
                    "clean": True,
                }
            elif self.path == "/setup":
                response = {
                    "sandbox_id": "check-sandbox",
                    "case_id": request_body["case_id"],
                }
            elif self.path == "/execute":
                response = {
                    "sandbox_id": "check-sandbox",
                    "case_id": request_body["case_id"],
                    "turn_id": request_body["turn_id"],
                    "response": {"private_response": "not for terminal"},
                }
            elif self.path == "/snapshot":
                response = {
                    "sandbox_id": "check-sandbox",
                    "case_id": request_body["case_id"],
                    "turn_id": request_body["turn_id"],
                    "state": {"private_state": "not for terminal"},
                }
            else:
                self.send_error(404)
                return
            encoded_response = json.dumps(response).encode()
            status = server.execute_status if self.path == "/execute" else 200
            self.send_response(status)
            content_type = (
                server.execute_content_type if self.path == "/execute" else "application/json"
            )
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded_response)))
            self.end_headers()
            self.wfile.write(encoded_response)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = _SandboxServer(("127.0.0.1", 0), Handler)
    server.requests = []
    server.generation = 0
    server.execute_content_type = execute_content_type
    server.execute_status = execute_status
    server.cleanup_generation_changes = cleanup_generation_changes
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _write_config(tmp_path: Path, server: _SandboxServer) -> Path:
    base_url = f"http://127.0.0.1:{server.server_port}"
    config = tmp_path / "sandbox.json"
    config.write_text(
        json.dumps(
            {
                "version": 3,
                "sandbox_id": "check-sandbox",
                "headers_from_env": {},
                "reset": {
                    "url": f"{base_url}/reset",
                    "request_json_template": {"case_id": "{{case_id}}"},
                    "sandbox_id_json_pointer": "/sandbox_id",
                    "case_id_json_pointer": "/case_id",
                    "generation_json_pointer": "/generation",
                    "clean_state_json_pointer": "/clean",
                    "clean_state_value": True,
                },
                "setup": {
                    "url": f"{base_url}/setup",
                    "request_json_template": {"case_id": "{{case_id}}"},
                    "sandbox_id_json_pointer": "/sandbox_id",
                    "case_id_json_pointer": "/case_id",
                },
                "execute_turn": {
                    "url": f"{base_url}/execute",
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "turn_id": "{{turn_id}}",
                        "input": "{{input}}",
                    },
                    "response_json_pointer": "/response",
                    "sandbox_id_json_pointer": "/sandbox_id",
                    "case_id_json_pointer": "/case_id",
                    "turn_id_json_pointer": "/turn_id",
                },
                "snapshot": {
                    "url": f"{base_url}/snapshot",
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "turn_id": "{{turn_id}}",
                    },
                    "response_json_pointer": "/state",
                    "sandbox_id_json_pointer": "/sandbox_id",
                    "case_id_json_pointer": "/case_id",
                    "turn_id_json_pointer": "/turn_id",
                },
            }
        ),
        encoding="utf-8",
    )
    return config


def _check_arguments(config: Path, *, output_json: bool = False) -> list[str]:
    arguments = [
        "sandbox",
        "check",
        str(config),
        "--probe",
        "connection check only",
        "--allow-sandbox-network-egress",
        "--confirm-isolated-sandbox",
        "--confirm-harmless-probe",
        "--allow-insecure-http",
    ]
    if output_json:
        arguments.append("--json")
    return arguments


def test_check_runs_complete_model_free_lifecycle(tmp_path: Path) -> None:
    with _sandbox_server() as server:
        config = _write_config(tmp_path, server)
        result = runner.invoke(app, _check_arguments(config, output_json=True))

    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["status"] == "ready"
    assert summary["sandbox_api_calls"] == 6
    assert summary["ul_semantic_model_calls"] == 0
    assert summary["probe_and_observations"] == "not_printed"
    assert "connection check only" not in result.output
    assert "private_response" not in result.output
    assert "private_state" not in result.output
    assert [path for path, _ in server.requests] == [
        "/reset",
        "/setup",
        "/snapshot",
        "/execute",
        "/snapshot",
        "/reset",
    ]
    case_ids = {request["case_id"] for _, request in server.requests}
    assert len(case_ids) == 1
    assert server.requests[2][1]["turn_id"] == "__ul_initial_state__"
    assert server.requests[3][1]["turn_id"] == server.requests[4][1]["turn_id"]


def test_check_reports_precise_phase_and_protocol_error(tmp_path: Path) -> None:
    with _sandbox_server(execute_content_type="text/plain") as server:
        config = _write_config(tmp_path, server)
        result = runner.invoke(app, _check_arguments(config, output_json=True))

    assert result.exit_code == 2, result.output
    summary = json.loads(result.output)
    assert summary["status"] == "not_ready"
    assert summary["failed_phase"] == "execute_turn"
    assert summary["error_code"] == "response_content_type"
    assert summary["cleanup"] == "succeeded"
    assert summary["ul_semantic_model_calls"] == 0
    assert [path for path, _ in server.requests] == [
        "/reset",
        "/setup",
        "/snapshot",
        "/execute",
        "/reset",
    ]


def test_check_reports_authentication_rejection(tmp_path: Path) -> None:
    with _sandbox_server(execute_status=401) as server:
        config = _write_config(tmp_path, server)
        result = runner.invoke(app, _check_arguments(config, output_json=True))

    assert result.exit_code == 2, result.output
    summary = json.loads(result.output)
    assert summary["failed_phase"] == "execute_turn"
    assert summary["error_code"] == "authentication_rejected"
    assert summary["reason"] == "sandbox API returned HTTP 401"
    assert summary["cleanup"] == "succeeded"


def test_check_reports_cleanup_failure_and_uncertain_state(tmp_path: Path) -> None:
    with _sandbox_server(cleanup_generation_changes=False) as server:
        config = _write_config(tmp_path, server)
        result = runner.invoke(app, _check_arguments(config, output_json=True))

    assert result.exit_code == 2, result.output
    summary = json.loads(result.output)
    assert summary["failed_phase"] == "cleanup_reset"
    assert summary["error_code"] == "reset_generation_reused"
    assert summary["cleanup"] == "failed"
    assert summary["sandbox_state_uncertain"] is True


def test_check_without_setup_uses_five_calls(tmp_path: Path) -> None:
    with _sandbox_server() as server:
        config = _write_config(tmp_path, server)
        raw_config = json.loads(config.read_text(encoding="utf-8"))
        raw_config["setup"] = None
        config.write_text(json.dumps(raw_config), encoding="utf-8")
        result = runner.invoke(app, _check_arguments(config, output_json=True))

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["sandbox_api_calls"] == 5
    assert [path for path, _ in server.requests] == [
        "/reset",
        "/snapshot",
        "/execute",
        "/snapshot",
        "/reset",
    ]


def test_check_reports_missing_credential_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CHECK_SANDBOX_TOKEN", raising=False)
    with _sandbox_server() as server:
        config = _write_config(tmp_path, server)
        raw_config = json.loads(config.read_text(encoding="utf-8"))
        raw_config["headers_from_env"] = {"Authorization": "CHECK_SANDBOX_TOKEN"}
        config.write_text(json.dumps(raw_config), encoding="utf-8")
        result = runner.invoke(app, _check_arguments(config, output_json=True))

    assert result.exit_code == 2
    summary = json.loads(result.output)
    assert summary["failed_phase"] == "preflight"
    assert summary["error_code"] == "credential_configuration"
    assert summary["reason"] == "sandbox API header environment variable is not set"
    assert server.requests == []


def test_check_reports_invalid_config_as_safe_json(tmp_path: Path) -> None:
    config = tmp_path / "sandbox.json"
    config.write_text("{", encoding="utf-8")

    result = runner.invoke(app, _check_arguments(config, output_json=True))

    assert result.exit_code == 2
    summary = json.loads(result.output)
    assert summary["failed_phase"] == "preflight"
    assert summary["error_code"] == "sandbox_config_invalid"
    assert summary["reason"] == "sandbox API config contains invalid JSON"


def test_check_sanitizes_config_errors_for_terminal(tmp_path: Path) -> None:
    config = tmp_path / "sandbox.json"
    config.write_text(json.dumps({"evil\u001bfield": True}), encoding="utf-8")

    result = runner.invoke(app, _check_arguments(config))

    assert result.exit_code == 2
    assert "\u001b" not in result.output
    assert "\\u001b" in result.output


def test_check_requires_explicit_safety_opt_ins_before_network(tmp_path: Path) -> None:
    with _sandbox_server() as server:
        config = _write_config(tmp_path, server)
        result = runner.invoke(
            app,
            ["sandbox", "check", str(config), "--probe", "connection check only"],
        )

    assert result.exit_code == 2
    assert "--allow-sandbox-network-egress" in result.output
    assert server.requests == []


def test_check_help_states_scope_and_limits() -> None:
    result = runner.invoke(app, ["sandbox", "check", "--help"])

    assert result.exit_code == 0
    assert "complete lifecycle" in result.output
    assert "--probe" in result.output
    assert "Attest" in result.output


@pytest.mark.parametrize(
    ("reason", "expected_code"),
    (
        ("sandbox API request timed out", "request_timeout"),
        ("sandbox API request write timed out", "write_timeout"),
        ("sandbox API connection pool timed out", "pool_timeout"),
        ("sandbox API DNS resolution failed", "dns_resolution"),
        ("sandbox API TLS connection failed", "tls_connection"),
    ),
)
def test_check_maps_transport_diagnostics(reason: str, expected_code: str) -> None:
    assert _diagnose_failure(reason, "execute_turn")[0] == expected_code
