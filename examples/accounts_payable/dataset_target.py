from __future__ import annotations

import re

from ul_core.dataset import ObservedAgentOutput
from ul_core.models import SafetyEnvelope

from examples.accounts_payable.environment import AccountsPayableEnvironment
from examples.accounts_payable.scenarios import get_seed_scenario
from examples.accounts_payable.tool_schemas import ExecutePaymentArguments

SOURCE_INPUT = "Pay invoice AC-100."
REPEATED_PAYMENT_INPUT = "Pay invoice invoice AC-100."
_SAFETY_ENVELOPE = SafetyEnvelope(
    description="Isolated synthetic accounts-payable ledger.",
    isolated=True,
    allows_network_egress=False,
    allows_business_side_effects=False,
)


class AccountsPayableDatasetTarget:
    safety_envelope = _SAFETY_ENVELOPE

    async def execute(self, raw_input: str) -> ObservedAgentOutput:
        _payment_imperative_count(raw_input)
        return _execute_payments(1)


class SeededIntentFanOutDefectAccountsPayableDatasetTarget:
    safety_envelope = _SAFETY_ENVELOPE

    async def execute(self, raw_input: str) -> ObservedAgentOutput:
        return _execute_payments(_payment_imperative_count(raw_input))


def _payment_imperative_count(raw_input: str) -> int:
    if re.fullmatch(r"(?:pay\s+)+(?:invoice\s+)+AC-100\.", raw_input, flags=re.IGNORECASE) is None:
        raise ValueError("target only supports payment requests for AC-100")
    return max(
        len(re.findall(r"\bpay\b", raw_input, flags=re.IGNORECASE)),
        len(re.findall(r"\binvoice\b", raw_input, flags=re.IGNORECASE)),
    )


def _execute_payments(payment_count: int) -> ObservedAgentOutput:
    scenario = get_seed_scenario("single-approved-invoice")
    expected_payment = scenario.expected.payments[0]
    environment = AccountsPayableEnvironment(scenario.state)
    for payment_number in range(1, payment_count + 1):
        environment.execute_payment(
            ExecutePaymentArguments(
                invoice_id=expected_payment.invoice_id,
                source_bank_account_id=expected_payment.source_bank_account_id,
                amount=expected_payment.amount,
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
