from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from ul.python_target import (
    load_python_dataset_target,
    validate_python_target_factory_reference,
)


def _write_factory_module(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_python_target_factory_executes_and_enforces_call_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path = tmp_path / "customer_sdk_target.py"
    _write_factory_module(
        module_path,
        """
from ul import ObservedAgentOutput, SafetyEnvelope

class Target:
    safety_envelope = SafetyEnvelope(
        description="Disposable test workspace",
        isolated=True,
        allows_network_egress=False,
        allows_business_side_effects=False,
    )
    fresh_state_per_execution = True

    async def execute(self, raw_input):
        return ObservedAgentOutput(raw_output={"answer": raw_input})

def create_target():
    return Target()
""",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    target = load_python_dataset_target("customer_sdk_target:create_target", max_target_calls=1)
    output = asyncio.run(target.execute("hello"))

    assert output.raw_output == {"answer": "hello"}
    with pytest.raises(RuntimeError, match="call budget exhausted"):
        asyncio.run(target.execute("again"))


@pytest.mark.parametrize(
    "reference",
    ("", "module", "module:", ":factory", "module:factory.attr", "../module:factory"),
)
def test_python_target_factory_reference_is_strict(reference: str) -> None:
    with pytest.raises(ValueError, match=r"package\.module:create_target"):
        validate_python_target_factory_reference(reference)


def test_python_target_factory_sanitizes_import_and_initialization_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(RuntimeError, match="could not be imported") as import_error:
        load_python_dataset_target("missing_private_module:create_target")
    assert "missing_private_module" not in str(import_error.value)

    _write_factory_module(
        tmp_path / "broken_target.py",
        """
def create_target():
    raise RuntimeError("secret customer detail")
""",
    )
    with pytest.raises(RuntimeError, match="failed during initialization") as factory_error:
        load_python_dataset_target("broken_target:create_target")
    assert "secret customer detail" not in str(factory_error.value)


def test_python_target_factory_rejects_invalid_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_factory_module(
        tmp_path / "invalid_target.py",
        """
def create_target():
    return object()
""",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ValueError, match="invalid dataset target"):
        load_python_dataset_target("invalid_target:create_target")


def test_python_target_factory_closes_async_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_factory_module(
        tmp_path / "closable_target.py",
        """
from ul import ObservedAgentOutput, SafetyEnvelope

closed = False

class Target:
    safety_envelope = SafetyEnvelope(
        description="Disposable test workspace",
        isolated=True,
        allows_network_egress=False,
        allows_business_side_effects=False,
    )
    fresh_state_per_execution = True

    async def execute(self, raw_input):
        return ObservedAgentOutput(raw_output=raw_input)

    async def aclose(self):
        global closed
        closed = True

def create_target():
    return Target()
""",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    target = load_python_dataset_target("closable_target:create_target")

    asyncio.run(target.aclose())

    assert sys.modules["closable_target"].closed is True


def test_python_target_factory_rejects_wrong_output_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_factory_module(
        tmp_path / "wrong_output_target.py",
        """
from ul import SafetyEnvelope

class Target:
    safety_envelope = SafetyEnvelope(
        description="Disposable test workspace",
        isolated=True,
        allows_network_egress=False,
        allows_business_side_effects=False,
    )
    fresh_state_per_execution = True

    async def execute(self, raw_input):
        return {"answer": raw_input}

def create_target():
    return Target()
""",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    target = load_python_dataset_target("wrong_output_target:create_target")

    with pytest.raises(RuntimeError, match="invalid observation"):
        asyncio.run(target.execute("hello"))


def test_python_target_factory_sanitizes_execution_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_factory_module(
        tmp_path / "failing_execution_target.py",
        """
from ul import ObservedAgentOutput, SafetyEnvelope

class Target:
    safety_envelope = SafetyEnvelope(
        description="Disposable test workspace",
        isolated=True,
        allows_network_egress=False,
        allows_business_side_effects=False,
    )
    fresh_state_per_execution = True

    async def execute(self, raw_input):
        raise RuntimeError("private runtime detail")

def create_target():
    return Target()
""",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    target = load_python_dataset_target("failing_execution_target:create_target")

    with pytest.raises(RuntimeError, match="execution failed") as error:
        asyncio.run(target.execute("hello"))
    assert "private runtime detail" not in str(error.value)
