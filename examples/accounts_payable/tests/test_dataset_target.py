from typing import cast

import pytest
from ul_core.contracts import DatasetTargetExecutor
from ul_core.dataset import ObservedAgentOutput

from examples.accounts_payable.dataset_target import (
    REPEATED_PAYMENT_INPUT,
    SOURCE_INPUT,
    AccountsPayableDatasetTarget,
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
