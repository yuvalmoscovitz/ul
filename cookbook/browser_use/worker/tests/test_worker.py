from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import pytest

from ul_browser_use_worker import run_request


@dataclass
class FakeHistory:
    def action_names(self) -> list[str]:
        return ["navigate", "click"]

    def final_result(self) -> str | None:
        return "completed"

    def is_successful(self) -> bool | None:
        return True


@dataclass
class FakeSession:
    killed: bool = False

    async def get_current_page_url(self) -> str:
        return "http://127.0.0.1:8765/receipt"

    async def kill(self) -> None:
        self.killed = True


@dataclass
class FakeAgent:
    error: Exception | None
    max_steps: int | None = None

    async def run(self, *, max_steps: int) -> FakeHistory:
        self.max_steps = max_steps
        if self.error is not None:
            raise self.error
        return FakeHistory()


@dataclass
class FakeBrowserUse:
    error: Exception | None = None
    sessions: list[FakeSession] = field(default_factory=lambda: list[FakeSession]())
    agents: list[FakeAgent] = field(default_factory=lambda: list[FakeAgent]())
    profile_arguments: list[dict[str, object]] = field(
        default_factory=lambda: list[dict[str, object]]()
    )

    def BrowserProfile(self, **arguments: object) -> object:
        self.profile_arguments.append(arguments)
        return object()

    def BrowserSession(self, **arguments: object) -> FakeSession:
        assert "browser_profile" in arguments
        session = FakeSession()
        self.sessions.append(session)
        return session

    def ChatOpenAI(self, **arguments: object) -> object:
        assert arguments == {"model": "gpt-5-mini"}
        return object()

    def Agent(self, **arguments: object) -> FakeAgent:
        assert arguments["browser_session"] is self.sessions[-1]
        agent = FakeAgent(self.error)
        self.agents.append(agent)
        return agent


def request() -> dict[str, object]:
    return {
        "task": "Inspect the invoice",
        "allowed_origins": ["http://127.0.0.1:8765"],
        "model": "gpt-5-mini",
        "max_steps": 7,
        "maximum_result_characters": 5,
    }


@pytest.mark.asyncio
async def test_worker_uses_fresh_ephemeral_session_and_returns_evidence() -> None:
    browser_use = FakeBrowserUse()

    first = await run_request(request(), browser_use)
    await run_request(request(), browser_use)

    assert len(browser_use.sessions) == 2
    assert all(session.killed for session in browser_use.sessions)
    assert browser_use.agents[0].max_steps == 7
    assert browser_use.profile_arguments[0] == {
        "allowed_domains": ["http://127.0.0.1:8765/"],
        "auto_download_pdfs": False,
        "enable_default_extensions": False,
        "headless": True,
        "keep_alive": False,
        "user_data_dir": None,
    }
    navigation_patterns = browser_use.profile_arguments[0]["allowed_domains"]
    assert isinstance(navigation_patterns, list)
    navigation_pattern = cast(list[object], navigation_patterns)[0]
    assert isinstance(navigation_pattern, str)
    assert not "http://127.0.0.1:87650/escape".startswith(navigation_pattern)
    assert not "http://127.0.0.1.evil:8765/escape".startswith(navigation_pattern)
    assert first == {
        "final_result": "compl",
        "final_url": "http://127.0.0.1:8765/receipt",
        "successful": True,
        "action_names": ["navigate", "click"],
        "total_action_count": 2,
    }


@pytest.mark.asyncio
async def test_worker_kills_session_when_agent_fails() -> None:
    browser_use = FakeBrowserUse(error=RuntimeError("provider failure"))

    with pytest.raises(RuntimeError, match=r"^Browser Use worker execution failed$"):
        await run_request(request(), browser_use)

    assert browser_use.sessions[0].killed is True
