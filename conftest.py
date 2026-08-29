from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def configure_test_semantic_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UL_DATASET_MODEL", "test/default-model")
