from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _configure_openrouter_provider_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UL_DATASET_OPENROUTER_PROVIDER", "test-provider")
