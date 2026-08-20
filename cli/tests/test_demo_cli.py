from __future__ import annotations

import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner
from ul.event_stress import RetryAfterSuccessfulCommitStressResult
from ul_cli import demo_runner
from ul_cli.main import app

_PROJECT_ROOT = Path(__file__).parents[2]
_EXPECTED_CONFIRMATION = "explicit retry created a second payment in all 3 repetitions"


def test_root_help_exposes_model_free_demo() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "demo" in result.output
    assert "model-free demonstration" in result.output


@pytest.mark.skipif(os.name == "nt", reason="symlink creation may require Windows privileges")
def test_demo_rejects_symlinked_private_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    application_data_directory = tmp_path / "application-data"
    application_data_directory.mkdir()
    (application_data_directory / "demo").symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.setattr(
        demo_runner, "user_data_path", lambda *args, **kwargs: application_data_directory
    )

    with pytest.raises(ValueError, match="not a symlink"):
        demo_runner._create_demo_artifact_directory()


def test_demo_confirms_duplicate_payment_without_model_credentials_and_retains_evidence(
    tmp_path: Path,
) -> None:
    working_directory = tmp_path / "read-only-working-directory"
    working_directory.mkdir()
    if os.name != "nt":
        working_directory.chmod(0o500)
    try:
        completed_process = subprocess.run(
            [sys.executable, "-m", "ul_cli.main", "demo"],
            cwd=working_directory,
            env={
                "HOME": str(tmp_path),
                "PYTHONPATH": os.pathsep.join(
                    [
                        str(_PROJECT_ROOT),
                        str(_PROJECT_ROOT / "core/src"),
                        str(_PROJECT_ROOT / "sdk/src"),
                        str(_PROJECT_ROOT / "cli/src"),
                    ]
                ),
                "PYTHONUTF8": "1",
                "TMPDIR": str(tmp_path),
            },
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            shell=False,
        )
    finally:
        if os.name != "nt":
            working_directory.chmod(0o700)

    assert completed_process.returncode == 0, completed_process.stdout + completed_process.stderr
    assert "Confirmed:" in completed_process.stdout
    assert _EXPECTED_CONFIRMATION in completed_process.stdout
    assert "Semantic-model calls: none" in completed_process.stdout
    assert "External-network calls: none" in completed_process.stdout
    evidence_match = re.search(r"^Evidence: (.+)$", completed_process.stdout, re.MULTILINE)
    assert evidence_match is not None

    evidence_path = Path(evidence_match.group(1))
    assert evidence_path.is_file()
    evidence = RetryAfterSuccessfulCommitStressResult.model_validate_json(
        evidence_path.read_text(encoding="utf-8")
    )
    assert evidence.status == "failed"
    assert evidence.requested_repetitions == 3
    assert {rule.status for rule in evidence.baseline_invariant_rules} == {"satisfied"}
    assert {rule.status for rule in evidence.successful_commit_invariant_rules} == {"satisfied"}
    assert {rule.rule_id: rule.status for rule in evidence.retried_invariant_rules} == {
        "exactly-one-committed-payment": "violated",
        "committed-payments-unique-by-invoice": "violated",
        "one-new-payment-per-turn": "satisfied",
    }


def test_built_wheel_contains_the_complete_demo(tmp_path: Path) -> None:
    completed_process = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(tmp_path)],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        shell=False,
    )

    assert completed_process.returncode == 0, completed_process.stdout + completed_process.stderr
    wheel_path = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
    assert {
        "examples/retry_after_successful_commit/__init__.py",
        "examples/retry_after_successful_commit/case.json",
        "examples/retry_after_successful_commit/defective_agent.py",
        "examples/retry_after_successful_commit/invariants.json",
        "examples/retry_after_successful_commit/run.py",
        "examples/retry_after_successful_commit/target.json",
        "ul_cli/demo_assets/__init__.py",
        "ul_cli/demo_assets/case.json",
        "ul_cli/demo_assets/invariants.json",
        "ul_cli/demo_assets/target.json",
        "ul_cli/demo.py",
        "ul_cli/demo_runner.py",
        "ul_cli/main.py",
    } <= names
    assert not any(
        name.startswith("examples/retry_after_successful_commit/tests/") for name in names
    )
