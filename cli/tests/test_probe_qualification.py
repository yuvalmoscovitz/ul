from __future__ import annotations

import http.client
import os
import secrets
import stat
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from cli.tests.fixtures.probe_qualification_targets import QualificationRequestHandler

_REPOSITORY_ROOT = Path(__file__).parents[2]
_DATASET = (
    _REPOSITORY_ROOT / "cli" / "tests" / "fixtures" / "probe_qualification_interactions.jsonl"
)
_FIXTURE_ROOT = _DATASET.parent


def _run_smoke(
    arguments: list[str], *, working_directory: Path, environment: dict[str, str]
) -> tuple[str, float]:
    started_at = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-m", "ul_cli.main", *arguments],
        input="y\nn\n",
        text=True,
        capture_output=True,
        cwd=working_directory,
        env=environment,
        check=False,
        timeout=30,
    )
    elapsed_seconds = time.monotonic() - started_at
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout + result.stderr, elapsed_seconds


@pytest.mark.parametrize("target_kind", ("callable", "authenticated_http"))
def test_public_probe_smoke_is_one_target_call_and_zero_semantic_calls(
    tmp_path: Path,
    target_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "target-calls.jsonl"
    environment = os.environ.copy()
    environment.pop("OPEN_ROUTER_API_KEY", None)
    environment["UL_QUALIFICATION_RECEIPT"] = str(receipt)
    server: ThreadingHTTPServer | None = None
    server_thread: threading.Thread | None = None
    server_started = False
    try:
        if target_kind == "callable":
            arguments = [
                "probe",
                str(_DATASET),
                "--target",
                "probe_qualification_targets:invoke",
                "--target-working-directory",
                str(_FIXTURE_ROOT),
                "--target-environment-variable",
                "UL_QUALIFICATION_RECEIPT",
                "--limit",
                "1",
            ]
        else:
            token = f"Bearer {secrets.token_urlsafe(24)}"
            environment["UL_ENVIRONMENT_AGENT_TOKEN"] = token
            monkeypatch.setenv("UL_ENVIRONMENT_AGENT_TOKEN", token)
            monkeypatch.setenv("UL_QUALIFICATION_RECEIPT", str(receipt))
            server = ThreadingHTTPServer(("127.0.0.1", 0), QualificationRequestHandler)
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            server_started = True
            unauthenticated_client = http.client.HTTPConnection("127.0.0.1", server.server_port)
            try:
                unauthenticated_client.request(
                    "POST",
                    "/invoke",
                    body='{"input":"Return the status for ticket 42."}',
                    headers={"Content-Type": "application/json"},
                )
                assert unauthenticated_client.getresponse().status == 401
            finally:
                unauthenticated_client.close()
            arguments = [
                "probe",
                str(_DATASET),
                "--target",
                f"http://127.0.0.1:{server.server_port}/invoke",
                "--header-from-env",
                "Authorization=UL_ENVIRONMENT_AGENT_TOKEN",
                "--allow-insecure-http",
                "--limit",
                "1",
            ]
        output, elapsed_seconds = _run_smoke(
            arguments, working_directory=tmp_path, environment=environment
        )
    finally:
        if server is not None:
            if server_started:
                server.shutdown()
            server.server_close()
        if server_thread is not None and server_started:
            server_thread.join()
    normalized_output = " ".join(output.split())
    assert "Smoke target invocation succeeded" in normalized_output
    assert "Command-wide environment API requests: 3 (includes smoke)" in normalized_output
    assert "Semantic-model calls: up to 8" in normalized_output
    assert "No semantic-model calls were made" in normalized_output
    assert elapsed_seconds < 300
    assert len(receipt.read_text(encoding="utf-8").splitlines()) == 1
    if os.name != "nt":
        assert stat.S_IMODE(receipt.stat().st_mode) == 0o600
    if target_kind == "authenticated_http":
        assert environment["UL_ENVIRONMENT_AGENT_TOKEN"] not in output
