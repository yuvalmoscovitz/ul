from __future__ import annotations

import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

from ul.event_stress import RetryAfterSuccessfulCommitStressResult

_PROJECT_ROOT = Path(__file__).parents[3]


def test_documented_one_command_finds_repeatable_duplicate_without_model_calls(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "retry-evidence.json"
    environment = {
        "HOME": str(tmp_path),
        "PYTHONPATH": os.pathsep.join(
            [
                str(_PROJECT_ROOT / "core/src"),
                str(_PROJECT_ROOT / "sdk/src"),
                str(_PROJECT_ROOT / "cli/src"),
            ]
        ),
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.retry_after_successful_commit.run",
            "--output",
            str(evidence_path),
        ],
        cwd=_PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        shell=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "explicit retry created a second payment in all 3 repetitions" in completed.stdout
    assert "Semantic-model calls: none" in completed.stdout
    result = RetryAfterSuccessfulCommitStressResult.model_validate_json(
        evidence_path.read_text(encoding="utf-8")
    )
    assert result.status == "failed"
    assert {rule.status for rule in result.baseline_invariant_rules} == {"satisfied"}
    assert {rule.status for rule in result.successful_commit_invariant_rules} == {"satisfied"}
    assert {rule.status for rule in result.retried_invariant_rules} == {"violated"}
    committed_effect_counts: list[object] = []
    for trial in result.trials:
        retried_checkpoint = trial.variation[1].committed_state_snapshot
        assert isinstance(retried_checkpoint, dict)
        committed_effect_counts.append(retried_checkpoint["committed_effect_count"])
    assert committed_effect_counts == [2, 2, 2]
    if os.name != "nt":
        assert stat.S_IMODE(evidence_path.stat().st_mode) == 0o600


def test_built_wheel_contains_the_complete_retry_example(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(tmp_path)],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        shell=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheel_path = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
    assert {
        "examples/retry_after_successful_commit/__init__.py",
        "examples/retry_after_successful_commit/README.md",
        "examples/retry_after_successful_commit/case.json",
        "examples/retry_after_successful_commit/defective_agent.py",
        "examples/retry_after_successful_commit/invariants.json",
        "examples/retry_after_successful_commit/run.py",
        "examples/retry_after_successful_commit/target.json",
    } <= names
    assert not any(
        name.startswith("examples/retry_after_successful_commit/tests/") for name in names
    )
