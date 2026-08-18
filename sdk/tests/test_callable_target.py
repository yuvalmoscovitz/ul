from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
from ul.callable_target import callable_target_factory
from ul_core.models import SafetyEnvelope

_SAFE_ENVELOPE = SafetyEnvelope(
    description="Disposable test agent state.",
    isolated=True,
    allows_network_egress=False,
    allows_business_side_effects=False,
)


@pytest.mark.asyncio
async def test_wraps_existing_callable_without_changing_its_result() -> None:
    calls: list[object] = []

    def reset() -> None:
        calls.append("reset")

    def invoke(raw_input: str) -> dict[str, object]:
        calls.append(("invoke", raw_input))
        return {"answer": "approved", "count": 1}

    def snapshot(result: object) -> dict[str, object]:
        calls.append(("snapshot", result))
        return {"invoice": "approved"}

    create_target = callable_target_factory(
        invoke,
        reset=reset,
        snapshot=snapshot,
        safety_envelope=_SAFE_ENVELOPE,
        fresh_state_per_execution=True,
    )

    target = create_target()
    output = await target.execute("approve invoice")
    second_output = await target.execute("approve another invoice")

    assert output.raw_output == {"answer": "approved", "count": 1}
    assert second_output.raw_output == output.raw_output
    assert output.metadata == {"committed_state_snapshot": {"invoice": "approved"}}
    assert calls == [
        "reset",
        ("invoke", "approve invoice"),
        ("snapshot", {"answer": "approved", "count": 1}),
        "reset",
        ("invoke", "approve another invoice"),
        ("snapshot", {"answer": "approved", "count": 1}),
    ]


@pytest.mark.asyncio
async def test_supports_async_invoke_and_hooks() -> None:
    calls: list[str] = []

    async def reset() -> None:
        calls.append("reset")

    async def invoke(raw_input: str) -> str:
        calls.append(raw_input)
        return "done"

    async def cleanup() -> None:
        calls.append("cleanup")

    create_target = callable_target_factory(
        invoke,
        reset=reset,
        cleanup=cleanup,
        safety_envelope=_SAFE_ENVELOPE,
        fresh_state_per_execution=True,
    )
    target = create_target()

    assert (await target.execute("run")).raw_output == "done"
    await cast(Any, target).aclose()
    assert calls == ["reset", "run", "cleanup"]


@pytest.mark.asyncio
async def test_serializes_reset_invoke_and_snapshot() -> None:
    cycle_active = False

    async def reset() -> None:
        nonlocal cycle_active
        assert cycle_active is False
        cycle_active = True
        await asyncio.sleep(0)

    async def invoke(raw_input: str) -> str:
        assert cycle_active is True
        await asyncio.sleep(0)
        return raw_input

    async def snapshot(result: object) -> dict[str, object]:
        nonlocal cycle_active
        assert cycle_active is True
        await asyncio.sleep(0)
        cycle_active = False
        return {"result": result}

    target = callable_target_factory(
        invoke,
        reset=reset,
        snapshot=snapshot,
        safety_envelope=_SAFE_ENVELOPE,
        fresh_state_per_execution=True,
    )()

    first, second = await asyncio.gather(target.execute("first"), target.execute("second"))

    assert first.raw_output == "first"
    assert second.raw_output == "second"
    assert cycle_active is False


def test_requires_explicit_fresh_state_declaration() -> None:
    with pytest.raises(ValueError, match="explicitly declare fresh state"):
        callable_target_factory(
            lambda raw_input: raw_input,
            safety_envelope=_SAFE_ENVELOPE,
            fresh_state_per_execution=False,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["reset", "invoke", "snapshot"])
async def test_sanitizes_hook_errors(phase: str) -> None:
    def fail(*args: object) -> object:
        raise RuntimeError("customer-secret")

    create_target = callable_target_factory(
        fail if phase == "invoke" else lambda raw_input: "ok",
        reset=fail if phase == "reset" else None,
        snapshot=fail if phase == "snapshot" else None,
        safety_envelope=_SAFE_ENVELOPE,
        fresh_state_per_execution=True,
    )

    with pytest.raises(RuntimeError, match=f"Callable target {phase} failed") as error:
        await create_target().execute("run")
    assert "customer-secret" not in str(error.value)


@pytest.mark.asyncio
async def test_rejects_invalid_and_oversized_results() -> None:
    invalid_target = callable_target_factory(
        lambda raw_input: object(),
        safety_envelope=_SAFE_ENVELOPE,
        fresh_state_per_execution=True,
    )()
    oversized_target = callable_target_factory(
        lambda raw_input: "x" * 1_000_001,
        safety_envelope=_SAFE_ENVELOPE,
        fresh_state_per_execution=True,
    )()

    with pytest.raises(RuntimeError, match="invalid result"):
        await invalid_target.execute("run")
    with pytest.raises(RuntimeError, match="result exceeded 1 MB"):
        await oversized_target.execute("run")


@pytest.mark.asyncio
async def test_rejects_invalid_snapshot_and_sanitizes_cleanup_error() -> None:
    def cleanup() -> None:
        raise RuntimeError("cleanup-secret")

    target = callable_target_factory(
        lambda raw_input: "ok",
        snapshot=lambda result: object(),
        cleanup=cleanup,
        safety_envelope=_SAFE_ENVELOPE,
        fresh_state_per_execution=True,
    )()

    with pytest.raises(RuntimeError, match="invalid snapshot"):
        await target.execute("run")
    with pytest.raises(RuntimeError, match="Callable target cleanup failed") as error:
        await cast(Any, target).aclose()
    assert "cleanup-secret" not in str(error.value)
