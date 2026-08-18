from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

import pytest

from ul_browser_use_adapter.adapter import (
    BrowserUseRunResult,
    BrowserUseSettings,
    BrowserUseTarget,
    SubprocessBrowserUseRuntime,
)


class FakeRuntime:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.tasks: list[str] = []

    async def execute(self, task: str, settings: BrowserUseSettings) -> BrowserUseRunResult:
        self.tasks.append(task)
        assert settings.max_steps == 7
        if self.error is not None:
            raise self.error
        return BrowserUseRunResult(
            final_result="completed",
            final_url="http://127.0.0.1:8765/receipt",
            successful=True,
            action_names=("navigate", "click"),
            total_action_count=2,
        )


def settings(*, maximum_result_characters: int = 4_000) -> BrowserUseSettings:
    return BrowserUseSettings(
        allowed_origins=("http://127.0.0.1:8765",),
        model="gpt-5-mini",
        worker_python=Path(sys.executable),
        openai_api_key="test-key",
        max_steps=7,
        maximum_result_characters=maximum_result_characters,
    )


@pytest.mark.asyncio
async def test_target_runs_each_input_and_returns_bounded_evidence() -> None:
    runtime = FakeRuntime()
    target = BrowserUseTarget(settings(maximum_result_characters=5), runtime)

    first = await target.execute("Inspect http://127.0.0.1:8765")
    second = await target.execute("Inspect the invoice again")

    assert len(runtime.tasks) == 2
    assert "isolated test environment: http://127.0.0.1:8765/" in runtime.tasks[0]
    assert "Do not download or upload files" in runtime.tasks[0]
    assert "Test task: Inspect http://127.0.0.1:8765/" in runtime.tasks[0]
    assert first.raw_output == {
        "response": "compl",
        "final_url": "http://127.0.0.1:8765/receipt",
        "success": True,
        "action_names": ["navigate", "click"],
        "action_count": 2,
    }
    assert first.metadata["result_truncated"] is True
    assert second.metadata["browser_state_reset"] == "fresh_worker_and_ephemeral_session"


@pytest.mark.asyncio
async def test_target_sanitizes_runtime_failure() -> None:
    runtime = FakeRuntime(error=RuntimeError("provider leaked a secret"))
    target = BrowserUseTarget(settings(), runtime)

    with pytest.raises(RuntimeError, match=r"^Browser Use target execution failed$") as error:
        await target.execute("Inspect the invoice")

    assert "secret" not in str(error.value)


@pytest.mark.asyncio
async def test_target_rejects_out_of_scope_urls_before_starting_worker() -> None:
    runtime = FakeRuntime()
    target = BrowserUseTarget(settings(), runtime)

    with pytest.raises(ValueError, match="outside the configured origins"):
        await target.execute("Visit https://example.com")

    assert runtime.tasks == []


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group regression")
async def test_failed_worker_cannot_leave_a_child_process_running(tmp_path: Path) -> None:
    worker = tmp_path / "failing-worker"
    child_pid_file = tmp_path / "child.pid"
    worker.write_text(
        f"""#!{sys.executable}
import subprocess
import sys
from pathlib import Path

child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
Path({str(child_pid_file)!r}).write_text(str(child.pid), encoding="utf-8")
raise SystemExit(2)
""",
        encoding="utf-8",
    )
    worker.chmod(0o700)
    configuration = BrowserUseSettings(
        allowed_origins=("http://127.0.0.1:8765",),
        model="gpt-5-mini",
        worker_python=worker,
        openai_api_key="test-key",
    )

    with pytest.raises(RuntimeError, match="Browser Use worker failed"):
        await SubprocessBrowserUseRuntime().execute("Inspect the invoice", configuration)

    child_pid = int(child_pid_file.read_text(encoding="utf-8"))
    try:
        for _ in range(100):
            if not _process_exists(child_pid):
                break
            await asyncio.sleep(0.01)
        assert not _process_exists(child_pid)
    finally:
        if _process_exists(child_pid):
            os.kill(child_pid, signal.SIGKILL)


def _process_exists(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {
            "UL_BROWSER_USE_ISOLATED_TEST_ENVIRONMENT": (
                "I_CONFIRM_THIS_IS_AN_ISOLATED_TEST_ENVIRONMENT"
            ),
            "UL_BROWSER_USE_ALLOWED_ORIGINS": "https://sandbox.example.test/path",
            "UL_BROWSER_USE_MODEL": "gpt-5-mini",
            "UL_BROWSER_USE_WORKER_PYTHON": sys.executable,
            "OPENAI_API_KEY": "test-key",
        },
    ],
)
def test_environment_requires_explicit_safety_configuration(
    environment: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        BrowserUseSettings.from_environment(environment)


def test_environment_builds_exact_origin_settings_without_exposing_key() -> None:
    configuration = BrowserUseSettings.from_environment(
        {
            "UL_BROWSER_USE_ISOLATED_TEST_ENVIRONMENT": (
                "I_CONFIRM_THIS_IS_AN_ISOLATED_TEST_ENVIRONMENT"
            ),
            "UL_BROWSER_USE_ALLOWED_ORIGINS": "http://127.0.0.1:8765/",
            "UL_BROWSER_USE_MODEL": "gpt-5-mini",
            "UL_BROWSER_USE_WORKER_PYTHON": sys.executable,
            "OPENAI_API_KEY": "test-key",
        }
    )

    assert configuration.allowed_origins == ("http://127.0.0.1:8765",)
    assert "test-key" not in repr(configuration)
