from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Annotated, TextIO, cast

import typer

from examples.quickstart.defective_agent import create_server

_EXAMPLE_DIRECTORY = Path(__file__).resolve().parent
_PROJECT_DIRECTORY = _EXAMPLE_DIRECTORY.parents[1]
_INVARIANTS_PATH = _EXAMPLE_DIRECTORY / "invariants.json"

_REQUIRED_LIVE_ENVIRONMENT = (
    "OPEN_ROUTER_API_KEY",
    "UL_DATASET_LIVE_CALLS",
    "UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING",
)
_UL_LIVE_ALIASES = {
    "UL_DATASET_LIVE_CALLS": "UL_LIVE",
    "UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING": "UL_LIVE",
}
_MODEL_ENVIRONMENT = {
    "UL_DATASET_MODEL": "x-ai/grok-4.6",
    "UL_DATASET_RENDER_MODEL": "x-ai/grok-4.6",
    "UL_DATASET_EQUIVALENCE_MODEL": "x-ai/grok-4.6",
}

_TENANTS = [
    "acme-corp",
    "globex-inc",
    "initech-llc",
    "umbrella-co",
    "cyberdyne-ltd",
]

_PAYMENT_RESULT = {
    "action": "payment_committed",
    "payment_id": "pay-0001",
    "invoice_reference": "AC-100",
    "requested_invoice_reference": "AC-100",
    "amount": "12500",
    "currency": "USD",
    "source_bank_account_id": "bank-main",
    "idempotency_key": "invoice:AC-100:1",
}

_TARGET_TEMPLATE = {
    "version": 1,
    "headers_from_env": {},
    "request_json_template": {
        "request": {"message": "{{input}}"},
        "settings": {"mode": "sandbox"},
    },
    "response_json_pointer": "/result",
}


def _make_dataset() -> list[dict]:
    return [
        {
            "id": f"ap-{tenant}",
            "input": "Pay AC-100.",
            "output": _PAYMENT_RESULT,
        }
        for tenant in _TENANTS
    ]


def _create_private_file(path: Path) -> TextIO:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    if sys.platform != "win32":
        os.fchmod(descriptor, 0o600)
    return os.fdopen(descriptor, "w", encoding="utf-8")


def _subprocess_environment(*, dry_run: bool) -> dict[str, str]:
    environment = dict(_MODEL_ENVIRONMENT)
    if dry_run:
        if "PYTHONPATH" in os.environ:
            environment["PYTHONPATH"] = os.environ["PYTHONPATH"]
        return environment
    api_key = os.environ.get("OPEN_ROUTER_API_KEY", "")
    if not api_key.strip():
        raise ValueError("set OPEN_ROUTER_API_KEY before running the live e2e test")
    environment["OPEN_ROUTER_API_KEY"] = api_key
    ul_live = os.environ.get("UL_LIVE", "").casefold() == "true"
    for name in _REQUIRED_LIVE_ENVIRONMENT[1:]:
        val = os.environ.get(name, "")
        alias = _UL_LIVE_ALIASES.get(name)
        if val.casefold() != "true" and not (alias and ul_live):
            raise ValueError(
                f"set UL_LIVE=true (or {name}=true) before running the live e2e test"
            )
        if val.casefold() == "true":
            environment[name] = val
        else:
            environment["UL_LIVE"] = os.environ["UL_LIVE"]
    return environment


def _ul(*args: str, env: dict[str, str]) -> int:
    return subprocess.run(
        [sys.executable, "-m", "ul_cli.main", *args],
        cwd=_PROJECT_DIRECTORY,
        env={**os.environ, **env},
        check=False,
        shell=False,
    ).returncode


def _extract_finding_ids(evidence_path: Path) -> list[str]:
    finding_ids: list[str] = []
    for line in evidence_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for case in cast(list, record.get("cases") or []):
            for finding in cast(list, cast(dict, case).get("findings") or []):
                fid = cast(dict, finding).get("finding_id")
                if isinstance(fid, str) and fid not in finding_ids:
                    finding_ids.append(fid)
    return finding_ids


def main(
    dry_run: Annotated[
        bool,
        typer.Option(help="Validate dataset and plan without model or target calls."),
    ] = False,
) -> None:
    try:
        env = _subprocess_environment(dry_run=dry_run)
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from None

    tmp_root = _PROJECT_DIRECTORY / "tmp"
    tmp_root.mkdir(exist_ok=True)
    artifact_dir = Path(tempfile.mkdtemp(prefix="e2e-regression-", dir=tmp_root))
    dataset_path = artifact_dir / "dataset.jsonl"
    target_config_path = artifact_dir / "target.json"
    evidence_path = artifact_dir / "evidence.jsonl"
    regressions_dir = artifact_dir / "regressions"
    regressions_dir.mkdir()
    run_result_path = artifact_dir / "run-result.json"

    with _create_private_file(dataset_path) as f:
        for entry in _make_dataset():
            f.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
    typer.echo(f"[1/5] Dataset: {len(_TENANTS)} interactions ({', '.join(_TENANTS)})")

    target_url = "http://127.0.0.1:9999/execute" if dry_run else None

    if dry_run:
        target_config = {**_TARGET_TEMPLATE, "url": target_url}
        with _create_private_file(target_config_path) as f:
            json.dump(target_config, f, indent=2)
        _ul(
            "dataset", "evaluate", str(dataset_path),
            "--target-config", str(target_config_path),
            "--invariants", str(_INVARIANTS_PATH),
            "--operator", "surface.disfluency_repeat",
            "--limit", "5",
            "--repetitions", "2",
            "--max-target-calls", "20",
            "--output", str(evidence_path),
            "--allow-target-network",
            "--confirm-isolated-sandbox",
            "--confirm-fresh-state",
            "--allow-insecure-http",
            "--dry-run",
            env=env,
        )
        typer.echo("Dry-run complete. No model or target calls sent.")
        return

    server: ThreadingHTTPServer = create_server()
    host, port = cast(tuple[str, int], server.server_address)
    server_thread = Thread(target=server.serve_forever, name="e2e-agent", daemon=True)
    server_thread.start()
    target_config = {**_TARGET_TEMPLATE, "url": f"http://{host}:{port}/execute"}
    with _create_private_file(target_config_path) as f:
        json.dump(target_config, f, indent=2)
    typer.echo(f"[2/5] Defective agent started on {host}:{port}")

    try:
        typer.echo("[3/5] Evaluating 5 interactions (up to 20 target calls) …")
        rc = _ul(
            "dataset", "evaluate", str(dataset_path),
            "--target-config", str(target_config_path),
            "--invariants", str(_INVARIANTS_PATH),
            "--operator", "surface.disfluency_repeat",
            "--limit", "5",
            "--repetitions", "2",
            "--max-target-calls", "20",
            "--output", str(evidence_path),
            "--allow-target-network",
            "--confirm-isolated-sandbox",
            "--confirm-fresh-state",
            "--allow-insecure-http",
            env=env,
        )
        if rc not in (0, 1):
            typer.echo(f"Evaluate failed (exit {rc})", err=True)
            raise typer.Exit(code=rc)

        finding_ids = _extract_finding_ids(evidence_path)
        if not finding_ids:
            typer.echo("No findings — the defect was not detected.", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"      {len(finding_ids)} finding(s) detected")

        typer.echo(f"[4/5] Reviewing and saving {len(finding_ids)} regression case(s) …")
        for i, fid in enumerate(finding_ids, start=1):
            rc = _ul(
                "dataset", "review", str(evidence_path), fid,
                "--status", "confirmed",
                "--severity", "high",
                "--reviewer", "e2e-test",
                "--reason", "The variation committed payment for the wrong invoice.",
                env=env,
            )
            if rc != 0:
                typer.echo(f"Review of finding {i} failed (exit {rc})", err=True)
                raise typer.Exit(code=rc)

            case_path = regressions_dir / f"case-{i:02d}.json"
            rc = _ul(
                "regression", "save", str(evidence_path), fid,
                "--rule", "committed-invoice-matches-request",
                "--target-config", str(target_config_path),
                "--output", str(case_path),
                "--confirm-versioned-input",
                env=env,
            )
            if rc != 0:
                typer.echo(f"Save case {i} failed (exit {rc})", err=True)
                raise typer.Exit(code=rc)
            typer.echo(f"      case-{i:02d}.json  ← {fid[:28]}…")

        saved = len(list(regressions_dir.glob("*.json")))
        typer.echo(f"[5/5] ul regression run — {saved} cases, up to {saved * 2} target calls …")
        rc = _ul(
            "regression", "run", str(regressions_dir),
            "--target-config", str(target_config_path),
            "--output", str(run_result_path),
            "--allow-target-network",
            "--confirm-isolated-sandbox",
            "--confirm-fresh-state",
            "--allow-insecure-http",
            "--max-target-calls", "20",
            env=env,
        )

        typer.echo(f"\nArtifacts: {artifact_dir}")
        raise typer.Exit(code=rc)

    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


if __name__ == "__main__":
    typer.run(main)
