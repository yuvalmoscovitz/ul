from decimal import Decimal

import pytest

from examples.accounts_payable.environment import (
    AccountsPayableEnvironment,
    PaymentToolTimeout,
)
from examples.accounts_payable.models import ApprovalStatus, Currency, ToolFaultKind
from examples.accounts_payable.scenarios import get_seed_scenario
from examples.accounts_payable.tool_schemas import (
    ExecutePaymentArguments,
    GetApprovalStatusArguments,
    ListPaymentsArguments,
)


def test_timeout_after_commit_is_safe_with_same_idempotency_key() -> None:
    scenario = get_seed_scenario("timeout-after-commit")
    environment = AccountsPayableEnvironment(scenario.state)
    arguments = _payment_arguments(
        invoice_id="inv-acme-110",
        amount=Decimal("42000"),
        idempotency_key="stable-timeout-key",
    )

    with pytest.raises(PaymentToolTimeout):
        environment.execute_payment(arguments)

    first_payment = environment.new_payments[0]
    operation = environment.execute_payment(arguments)

    assert operation.payment_id == first_payment.id
    assert len(environment.new_payments) == 1


def test_new_idempotency_key_after_timeout_creates_duplicate() -> None:
    scenario = get_seed_scenario("timeout-after-commit")
    environment = AccountsPayableEnvironment(scenario.state)

    with pytest.raises(PaymentToolTimeout):
        environment.execute_payment(
            _payment_arguments(
                invoice_id="inv-acme-110",
                amount=Decimal("42000"),
                idempotency_key="first-timeout-key",
            )
        )
    environment.execute_payment(
        _payment_arguments(
            invoice_id="inv-acme-110",
            amount=Decimal("42000"),
            idempotency_key="unsafe-new-retry-key",
        )
    )

    assert len(environment.new_payments) == 2


def test_dispatch_rejects_unknown_tools_and_extra_arguments() -> None:
    scenario = get_seed_scenario("single-approved-invoice")
    environment = AccountsPayableEnvironment(scenario.state)

    unknown_result = environment.dispatch_json("delete_vendor", "{}")
    invalid_result = environment.dispatch_json(
        "get_invoice", '{"invoice_id":"inv-acme-100","unexpected":true}'
    )

    assert '"code": "unknown_tool"' in unknown_result
    assert '"code": "invalid_arguments"' in invalid_result


def test_list_payments_finds_uncertain_commit_by_idempotency_key() -> None:
    scenario = get_seed_scenario("timeout-after-commit")
    environment = AccountsPayableEnvironment(scenario.state)
    idempotency_key = "lookup-after-timeout"

    with pytest.raises(PaymentToolTimeout):
        environment.execute_payment(
            _payment_arguments(
                invoice_id="inv-acme-110",
                amount=Decimal("42000"),
                idempotency_key=idempotency_key,
            )
        )

    payments = environment.list_payments(ListPaymentsArguments(idempotency_key=idempotency_key))
    assert [payment.id for payment in payments] == [environment.new_payments[0].id]


def test_stale_read_can_hide_a_concurrent_approval_revocation() -> None:
    scenario = get_seed_scenario("single-approved-invoice")
    scenario.state.tool_faults["get_approval_status"] = [ToolFaultKind.STALE_READ]
    environment = AccountsPayableEnvironment(scenario.state)
    environment.revoke_approval("inv-acme-100")

    stale_approval = environment.get_approval_status(
        GetApprovalStatusArguments(invoice_id="inv-acme-100")
    )
    current_approval = environment.get_approval_status(
        GetApprovalStatusArguments(invoice_id="inv-acme-100")
    )

    assert stale_approval.status == ApprovalStatus.APPROVED
    assert current_approval.status == ApprovalStatus.REVOKED


def _payment_arguments(
    *, invoice_id: str, amount: Decimal, idempotency_key: str
) -> ExecutePaymentArguments:
    return ExecutePaymentArguments(
        invoice_id=invoice_id,
        source_bank_account_id="account-us-usd",
        amount=amount,
        currency=Currency.USD,
        idempotency_key=idempotency_key,
    )
