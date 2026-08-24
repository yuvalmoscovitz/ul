from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_private_augmentation_example_uses_only_the_public_sdk() -> None:
    repository_root = Path(__file__).parents[2]

    completed = subprocess.run(
        [sys.executable, str(repository_root / "examples" / "private_augmentation.py")],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == [
        "private.customer_workflow",
        "deterministic_transform",
        "semantic_renderer",
        "conversation_modifier",
        "environment_schedule",
        "fault_control",
        "validator",
    ]
