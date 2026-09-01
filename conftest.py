from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-live-llm",
        action="store_true",
        help="fail instead of skip when a live LLM development test cannot run",
    )


@pytest.fixture(autouse=True)
def configure_test_semantic_model(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    if request.node.get_closest_marker("live_llm") is not None:
        return
    monkeypatch.setenv("UL_DATASET_MODEL", "test/default-model")
