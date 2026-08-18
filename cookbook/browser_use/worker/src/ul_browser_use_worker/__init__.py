from __future__ import annotations

import asyncio
import importlib
import json
import os
import signal
import subprocess
import sys
from typing import Never, Protocol, cast


class BrowserUseHistory(Protocol):
    def action_names(self) -> list[str]: ...

    def final_result(self) -> str | None: ...

    def is_successful(self) -> bool | None: ...


class BrowserUseSession(Protocol):
    async def get_current_page_url(self) -> str: ...

    async def kill(self) -> None: ...


class BrowserUseAgent(Protocol):
    async def run(self, *, max_steps: int) -> BrowserUseHistory: ...


class BrowserUseModule(Protocol):
    def Agent(self, **arguments: object) -> object: ...

    def BrowserProfile(self, **arguments: object) -> object: ...

    def BrowserSession(self, **arguments: object) -> object: ...

    def ChatOpenAI(self, **arguments: object) -> object: ...


async def run_request(
    request: dict[str, object], browser_use: BrowserUseModule
) -> dict[str, object]:
    task, allowed_origins, model, max_steps, maximum_result_characters = _validate_request(request)
    profile = browser_use.BrowserProfile(
        allowed_domains=[f"{origin}/" for origin in allowed_origins],
        auto_download_pdfs=False,
        enable_default_extensions=False,
        headless=True,
        keep_alive=False,
        user_data_dir=None,
    )
    session = cast(BrowserUseSession, browser_use.BrowserSession(browser_profile=profile))
    execution_failed = False
    try:
        agent = cast(
            BrowserUseAgent,
            browser_use.Agent(
                task=task,
                llm=browser_use.ChatOpenAI(model=model),
                browser_session=session,
                generate_gif=False,
            ),
        )
        history = await agent.run(max_steps=max_steps)
        all_action_names = history.action_names()
        action_names = [name[:200] for name in all_action_names[:100]]
        return {
            "final_result": (history.final_result() or "")[:maximum_result_characters],
            "final_url": (await session.get_current_page_url())[:2_000],
            "successful": history.is_successful(),
            "action_names": action_names,
            "total_action_count": len(all_action_names),
        }
    except asyncio.CancelledError:
        execution_failed = True
        raise
    except Exception:
        execution_failed = True
        raise RuntimeError("Browser Use worker execution failed") from None
    finally:
        try:
            async with asyncio.timeout(15):
                await session.kill()
        except Exception:
            if not execution_failed:
                raise RuntimeError("Browser Use worker cleanup failed") from None


def main() -> None:
    result_output = os.dup(sys.stdout.fileno())
    try:
        with open(os.devnull, "w", encoding="utf-8") as discarded_output:
            os.dup2(discarded_output.fileno(), sys.stdout.fileno())
            os.dup2(discarded_output.fileno(), sys.stderr.fileno())
            try:
                browser_use = cast(BrowserUseModule, importlib.import_module("browser_use"))
                request = cast(dict[str, object], json.loads(sys.stdin.buffer.read(20_001)))
                response = asyncio.run(run_request(request, browser_use))
                encoded_response = json.dumps(response, separators=(",", ":")).encode("utf-8")
            except BaseException:
                _terminate_process_tree()
        os.write(result_output, encoded_response)
    finally:
        os.close(result_output)


def _terminate_process_tree() -> Never:
    if os.name == "posix" and os.getpgrp() == os.getpid():
        os.killpg(os.getpgrp(), signal.SIGKILL)
    system_root = os.environ.get("SYSTEMROOT")
    if system_root:
        taskkill = os.path.join(system_root, "System32", "taskkill.exe")
        if os.path.isfile(taskkill):
            subprocess.run(
                [taskkill, "/PID", str(os.getpid()), "/T", "/F"],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
    os._exit(1)


def _validate_request(
    request: dict[str, object],
) -> tuple[str, tuple[str, ...], str, int, int]:
    if set(request) != {
        "task",
        "allowed_origins",
        "model",
        "max_steps",
        "maximum_result_characters",
    }:
        raise ValueError("invalid worker request")
    task = request["task"]
    origins = request["allowed_origins"]
    model = request["model"]
    max_steps = request["max_steps"]
    maximum_result_characters = request["maximum_result_characters"]
    if (
        not isinstance(task, str)
        or not task
        or len(task) > 5_000
        or not isinstance(origins, list)
        or not origins
        or not isinstance(model, str)
        or not model
        or type(max_steps) is not int
        or not 1 <= max_steps <= 25
        or type(maximum_result_characters) is not int
        or not 1 <= maximum_result_characters <= 20_000
    ):
        raise ValueError("invalid worker request")
    origins = cast(list[object], origins)
    typed_origins: list[str] = []
    for origin in origins:
        if not isinstance(origin, str):
            raise ValueError("invalid worker request")
        typed_origins.append(origin)
    return (
        task,
        tuple(typed_origins),
        model,
        max_steps,
        maximum_result_characters,
    )
