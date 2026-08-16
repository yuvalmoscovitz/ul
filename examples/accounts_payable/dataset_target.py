from __future__ import annotations

import re
from decimal import Decimal
from random import Random

from ul_core.dataset import ObservedAgentOutput
from ul_core.models import SafetyEnvelope

from examples.accounts_payable.environment import AccountsPayableEnvironment
from examples.accounts_payable.scenarios import get_seed_scenario
from examples.accounts_payable.tool_schemas import ExecutePaymentArguments

SOURCE_INPUT = "Pay AC-100."
REPEATED_PAYMENT_INPUT = "Pay pay AC-100."
AMOUNT_SOURCE_INPUT = "pay 12500$ to AC-100"
SELF_CORRECTED_PAYMENT_INPUT = "pay 13500$, sorry 12500$ to AC-100"
_SAFETY_ENVELOPE = SafetyEnvelope(
    description="Isolated synthetic accounts-payable ledger.",
    isolated=True,
    allows_network_egress=False,
    allows_business_side_effects=False,
)


class AccountsPayableDatasetTarget:
    safety_envelope = _SAFETY_ENVELOPE
    fresh_state_per_execution = True

    async def execute(self, raw_input: str) -> ObservedAgentOutput:
        payment_amounts = _self_corrected_payment_amounts(raw_input)
        if payment_amounts:
            return _execute_payments((payment_amounts[-1],))
        _payment_imperative_count(raw_input)
        return _execute_payments((_expected_payment_amount(),))


class SeededIntentFanOutDefectAccountsPayableDatasetTarget:
    safety_envelope = _SAFETY_ENVELOPE
    fresh_state_per_execution = True

    async def execute(self, raw_input: str) -> ObservedAgentOutput:
        return _execute_payments(
            (_expected_payment_amount(),) * _payment_imperative_count(raw_input)
        )


class SeededFlakyIntentFanOutDefectAccountsPayableDatasetTarget:
    safety_envelope = _SAFETY_ENVELOPE
    fresh_state_per_execution = True

    def __init__(self, seed: int = 4) -> None:
        self._random = Random(seed)

    async def execute(self, raw_input: str) -> ObservedAgentOutput:
        imperative_count = _payment_imperative_count(raw_input)
        payment_count = 1 if imperative_count == 1 else 1 + self._random.randrange(imperative_count)
        return _execute_payments((_expected_payment_amount(),) * payment_count)


class SeededFirstValueWinsDefectAccountsPayableDatasetTarget:
    safety_envelope = _SAFETY_ENVELOPE
    fresh_state_per_execution = True

    async def execute(self, raw_input: str) -> ObservedAgentOutput:
        payment_amounts = _self_corrected_payment_amounts(raw_input)
        if not payment_amounts:
            raise ValueError("target requires a visible payment amount")
        return _execute_payments((payment_amounts[0],))


def _payment_imperative_count(raw_input: str) -> int:
    if re.fullmatch(r"(?:pay\s+)+AC-100\.", raw_input, flags=re.IGNORECASE) is None:
        raise ValueError("target only supports payment requests for AC-100")
    return len(re.findall(r"\bpay\b", raw_input, flags=re.IGNORECASE))


def _self_corrected_payment_amounts(raw_input: str) -> tuple[Decimal, ...]:
    if (
        re.search(r"\bpay\b", raw_input, flags=re.IGNORECASE) is None
        or re.search(r"\bAC-100\b", raw_input, flags=re.IGNORECASE) is None
    ):
        return ()
    amounts = tuple(
        Decimal(match.group("amount"))
        for match in re.finditer(r"(?P<amount>\d+(?:\.\d+)?)\$", raw_input)
    )
    if not amounts:
        return ()
    if len(amounts) > 2:
        raise ValueError("target supports one amount and one optional correction")
    if (
        len(amounts) == 2
        and re.search(r"\b(?:sorry|actually|i\s+mean)\b", raw_input, flags=re.IGNORECASE) is None
    ):
        raise ValueError("two payment amounts require an explicit correction")
    return amounts


def _expected_payment_amount() -> Decimal:
    return get_seed_scenario("single-approved-invoice").expected.payments[0].amount


def _execute_payments(payment_amounts: tuple[Decimal, ...]) -> ObservedAgentOutput:
    scenario = get_seed_scenario("single-approved-invoice")
    expected_payment = scenario.expected.payments[0]
    environment = AccountsPayableEnvironment(scenario.state)
    for payment_number, payment_amount in enumerate(payment_amounts, start=1):
        environment.execute_payment(
            ExecutePaymentArguments(
                invoice_id=expected_payment.invoice_id,
                source_bank_account_id=expected_payment.source_bank_account_id,
                amount=payment_amount,
                currency=expected_payment.currency,
                idempotency_key=f"dataset-target-{payment_number}",
            )
        )
    return ObservedAgentOutput(
        raw_output={
            "actions": [
                {
                    "action": "payment_committed",
                    "payment_id": payment.id,
                    "invoice_reference": scenario.state.invoices[0].reference,
                    "amount": str(payment.amount),
                    "currency": payment.currency.value,
                    "source_bank_account_id": payment.source_bank_account_id,
                    "idempotency_key": payment.idempotency_key,
                }
                for payment in environment.new_payments
            ]
        },
        metadata={"fixture_id": scenario.id, "isolated": True},
    )
