from typing import cast

import pytest
from ul_core.contracts import DatasetTargetExecutor
from ul_core.dataset import ObservedAgentOutput

from examples.accounts_payable.dataset_target import (
    AMOUNT_SOURCE_INPUT,
    REPEATED_PAYMENT_INPUT,
    SELF_CORRECTED_PAYMENT_INPUT,
    SOURCE_INPUT,
    AccountsPayableDatasetTarget,
    SeededFirstValueWinsDefectAccountsPayableDatasetTarget,
    SeededFlakyIntentFanOutDefectAccountsPayableDatasetTarget,
    SeededIntentFanOutDefectAccountsPayableDatasetTarget,
)


def _committed_payment_actions(output: ObservedAgentOutput) -> list[dict[str, object]]:
    assert isinstance(output.raw_output, dict)
    actions = output.raw_output["actions"]
    assert isinstance(actions, list)
    return cast(list[dict[str, object]], actions)


@pytest.mark.asyncio
async def test_source_and_candidate_each_commit_once_for_correct_target() -> None:
    target = AccountsPayableDatasetTarget()

    assert isinstance(target, DatasetTargetExecutor)
    assert target.fresh_state_per_execution is True
    assert target.safety_envelope.isolated
    assert not target.safety_envelope.allows_network_egress
    assert not target.safety_envelope.allows_business_side_effects

    source_output = await target.execute(SOURCE_INPUT)
    candidate_output = await target.execute(REPEATED_PAYMENT_INPUT)

    assert len(_committed_payment_actions(source_output)) == 1
    assert len(_committed_payment_actions(candidate_output)) == 1


@pytest.mark.asyncio
async def test_seeded_intent_fan_out_defect_commits_repeated_imperatives_twice() -> None:
    target = SeededIntentFanOutDefectAccountsPayableDatasetTarget()

    assert isinstance(target, DatasetTargetExecutor)
    assert target.fresh_state_per_execution is True
    assert target.safety_envelope.isolated

    source_output = await target.execute(SOURCE_INPUT)
    candidate_output = await target.execute(REPEATED_PAYMENT_INPUT)
    source_actions = _committed_payment_actions(source_output)
    candidate_actions = _committed_payment_actions(candidate_output)

    assert len(source_actions) == 1
    assert len(candidate_actions) == 2
    assert {action["action"] for action in candidate_actions} == {"payment_committed"}
    assert len({action["idempotency_key"] for action in candidate_actions}) == 2


@pytest.mark.asyncio
async def test_seeded_target_state_is_fresh_for_each_call() -> None:
    target = SeededIntentFanOutDefectAccountsPayableDatasetTarget()

    first_output = await target.execute(REPEATED_PAYMENT_INPUT)
    second_output = await target.execute(REPEATED_PAYMENT_INPUT)

    assert first_output == second_output
    assert [action["payment_id"] for action in _committed_payment_actions(first_output)] == [
        "pay-0001",
        "pay-0002",
    ]


@pytest.mark.asyncio
async def test_seeded_flaky_target_varies_only_repeated_imperatives_with_fresh_state() -> None:
    target = SeededFlakyIntentFanOutDefectAccountsPayableDatasetTarget(seed=4)

    source_outputs = [await target.execute(SOURCE_INPUT) for _ in range(3)]
    candidate_outputs = [await target.execute(REPEATED_PAYMENT_INPUT) for _ in range(3)]

    assert [len(_committed_payment_actions(output)) for output in source_outputs] == [1, 1, 1]
    assert [len(_committed_payment_actions(output)) for output in candidate_outputs] == [1, 2, 1]
    assert [
        action["payment_id"]
        for output in candidate_outputs
        for action in _committed_payment_actions(output)[:1]
    ] == ["pay-0001", "pay-0001", "pay-0001"]
    assert [
        action["payment_id"] for action in _committed_payment_actions(candidate_outputs[1])
    ] == ["pay-0001", "pay-0002"]


@pytest.mark.asyncio
async def test_correct_target_uses_the_repaired_final_amount() -> None:
    target = AccountsPayableDatasetTarget()

    source_output = await target.execute(AMOUNT_SOURCE_INPUT)
    candidate_output = await target.execute(SELF_CORRECTED_PAYMENT_INPUT)

    assert _committed_payment_actions(source_output)[0]["amount"] == "12500"
    assert _committed_payment_actions(candidate_output)[0]["amount"] == "12500"


@pytest.mark.asyncio
async def test_seeded_first_value_wins_defect_uses_the_provisional_amount() -> None:
    target = SeededFirstValueWinsDefectAccountsPayableDatasetTarget()

    source_output = await target.execute(AMOUNT_SOURCE_INPUT)
    candidate_output = await target.execute(SELF_CORRECTED_PAYMENT_INPUT)

    assert _committed_payment_actions(source_output)[0]["amount"] == "12500"
    assert _committed_payment_actions(candidate_output)[0]["amount"] == "13500"


@pytest.mark.asyncio
async def test_self_correction_targets_start_from_fresh_state_for_each_call() -> None:
    correct_target = AccountsPayableDatasetTarget()
    defective_target = SeededFirstValueWinsDefectAccountsPayableDatasetTarget()

    first_correct_output = await correct_target.execute(SELF_CORRECTED_PAYMENT_INPUT)
    second_correct_output = await correct_target.execute(SELF_CORRECTED_PAYMENT_INPUT)
    assert first_correct_output == second_correct_output
    assert await defective_target.execute(
        SELF_CORRECTED_PAYMENT_INPUT
    ) == await defective_target.execute(SELF_CORRECTED_PAYMENT_INPUT)
