from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import zipfile
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner
from ul_cli import demo_runner
from ul_cli.demo_scenario import run_demo_evaluations
from ul_cli.main import app

_PROJECT_ROOT = Path(__file__).parents[2]


def test_root_help_leads_with_demo_and_observed_interaction_probe() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    normalized_output = " ".join(result.output.split())
    assert "Start with 'ul demo'" in normalized_output
    assert "probe observed interactions with 'ul probe'" in normalized_output
    assert "See UL's model-free augment-and-compare workflow" in normalized_output
    assert "Probe observed interactions against a safe callable or HTTP" in normalized_output
    assert "Configure an advanced stateful-evidence project" in normalized_output
    assert [command.name for command in app.registered_commands[:2]] == ["probe", "demo"]


def test_readme_quickstart_leads_with_a_real_probe_and_keeps_demo_secondary() -> None:
    readme = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    quickstart = readme[readme.index("## Quickstart") : readme.index("## How it works")]
    normalized_quickstart = " ".join(quickstart.split())

    assert "--target support_agent:invoke" in normalized_quickstart
    assert "--target-environment-variable OPEN_ROUTER_API_KEY" in normalized_quickstart
    assert "--target https://agent.test/invoke" in normalized_quickstart
    assert "--header-from-env Authorization=UL_ENVIRONMENT_AGENT_TOKEN" in normalized_quickstart
    assert "reference evidence, not as a correct answer" in normalized_quickstart
    assert "UNKNOWN AND UNBOUNDED" in normalized_quickstart
    assert "stop unless that risk is acceptable" in normalized_quickstart
    assert "ul demo" not in normalized_quickstart
    assert "it is not a real-agent onboarding or qualification run" in readme
    assert readme.index("## Quickstart") < readme.index("## Stateful and trace-based testing")


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


def test_demo_shows_three_generic_findings_without_credentials_and_retains_valid_evidence(
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
    assert "UL evaluation report" in completed_process.stdout
    assert "2 customer requests" in completed_process.stdout
    assert "3 augmentations" in completed_process.stdout
    assert "3 repeatable findings" in completed_process.stdout
    assert completed_process.stdout.count("Baseline input") == 2
    assert completed_process.stdout.count("Augmentation") == 3
    assert "Typing errors" in completed_process.stdout
    assert "Frustrated customer" in completed_process.stdout
    assert "Short message" in completed_process.stdout
    normalized_output = " ".join(completed_process.stdout.split())
    assert "subscription cancellation scheduled action is missing" in normalized_output
    assert "cancellation timing changed from end of the billing period" in normalized_output
    assert "address type changed from delivery address to billing address" in normalized_output
    assert normalized_output.count("Detected by UL's action comparison") == 3
    assert normalized_output.count("Seen in 3/3 runs") == 3
    assert "no custom rules are used here" in normalized_output
    assert "ul probe interactions.jsonl --target agent:invoke" in normalized_output
    assert "--target https://agent.test/invoke" in normalized_output
    assert "--header-from-env Authorization=UL_ENVIRONMENT_AGENT_TOKEN" in normalized_output
    assert "observed reference evidence, not a correctness oracle" in normalized_output
    assert "does not verify trajectory or committed state" in normalized_output
    assert "confirm the exact test target and bounded paid/network campaign" in normalized_output
    assert "Use ul init and ul run" in normalized_output
    assert normalized_output.index("ul probe interactions.jsonl") < normalized_output.index(
        "Use ul init and ul run"
    )
    evidence_match = re.search(
        r"^Technical evidence saved  (.+)$", completed_process.stdout, re.MULTILINE
    )
    assert evidence_match is not None

    evidence_path = Path(evidence_match.group(1))
    assert evidence_path.is_file()
    assert len((evidence_path.parent / ".ul" / "review-history.key").read_bytes()) == 32
    evidence_records = [
        json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(evidence_records) == 2
    assert [len(record["cases"]) for record in evidence_records] == [2, 1]
    assert (
        sum(len(case["findings"]) for record in evidence_records for case in record["cases"]) == 3
    )
    assert all(
        trial["execution_evidence"] is not None
        for record in evidence_records
        for case in record["technical_details"]["cases"]
        for trial in case["trial_set"]["trials"]
    )
    assert {
        record["execution_plan"]["dataset_planned_target_calls"] for record in evidence_records
    } == {15}

    report_result = CliRunner().invoke(app, ["report", str(evidence_path)])
    assert report_result.exit_code == 1, report_result.output
    assert "UL run report" in report_result.output
    assert "Findings: 3 total; 3 actionable" in report_result.output
    assert "ul dataset review" in report_result.output


def test_demo_report_uses_terminal_colors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = StringIO()
    monkeypatch.setattr(
        demo_runner,
        "console",
        Console(file=output, force_terminal=True, color_system="standard", width=120),
    )

    demo_runner._print_report(asyncio.run(run_demo_evaluations()), tmp_path / "evidence.jsonl")

    rendered = output.getvalue()
    assert "\x1b[" in rendered
    assert "3 repeatable findings" in rendered


@pytest.mark.skipif(os.name == "nt", reason="symlink creation may require Windows privileges")
def test_demo_rejects_a_dangling_output_symlink(tmp_path: Path) -> None:
    output = tmp_path / "evidence.jsonl"
    symlink_target = tmp_path / "target.jsonl"
    output.symlink_to(symlink_target)

    result = CliRunner().invoke(app, ["demo", "--output", str(output)])

    assert result.exit_code == 2
    assert not symlink_target.exists()


def test_demo_escapes_an_unsafe_output_path(monkeypatch: pytest.MonkeyPatch) -> None:
    output = StringIO()
    monkeypatch.setattr(demo_runner, "console", Console(file=output, force_terminal=False))
    unsafe_path = Path("/tmp/evidence; touch hacked\x1b.jsonl")

    demo_runner._print_report(asyncio.run(run_demo_evaluations()), unsafe_path)

    rendered = output.getvalue()
    assert "\x1b" not in rendered
    assert "\\u001b" in rendered
    assert "ul report" not in rendered


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
        "ul_cli/demo.py",
        "ul_cli/demo_runner.py",
        "ul_cli/demo_scenario.py",
        "ul_cli/main.py",
    } <= names
    assert not any(
        name.startswith("examples/retry_after_successful_commit/tests/") for name in names
    )
