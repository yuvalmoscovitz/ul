from __future__ import annotations

import pytest
from ul import EvaluatorModelPreflight
from ul_cli.dataset.evaluation import execution as execution_module

from ._factories import _evaluator_preflight


@pytest.fixture(autouse=True)
def _stub_evaluator_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    async def successful_preflight(settings: object) -> EvaluatorModelPreflight:
        del settings
        return _evaluator_preflight()

    monkeypatch.setattr(execution_module, "preflight_evaluator", successful_preflight)
