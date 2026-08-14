from contextlib import suppress
from decimal import Decimal

from examples.accounts_payable.environment import AccountsPayableEnvironment, PaymentToolTimeout
from examples.accounts_payable.models import ApprovalStatus, Currency
from examples.accounts_payable.oracles import evaluate_financial_outcome
from examples.accounts_payable.scenarios import get_seed_scenario
from examples.accounts_payable.tool_schemas import ExecutePaymentArguments


def test_oracle_detects_duplicate_after_unknown_timeout() -> None:
    scenario = get_seed_scenario("timeout-after-commit")
    environment = AccountsPayableEnvironment(scenario.state)

    with_timeout = _execute_arguments(
        "inv-acme-110", "account-us-usd", Decimal("42000"), Currency.USD, "timeout-key-one"
    )
    with suppress(PaymentToolTimeout):
        environment.execute_payment(with_timeout)
    environment.execute_payment(
        with_timeout.model_copy(update={"idempotency_key": "timeout-key-two"})
    )

    report = evaluate_financial_outcome(scenario, environment)
    codes = {finding.code for finding in report.findings}

    assert not report.passed
    assert "duplicate_payment" in codes
    assert "invoice_overpaid" in codes


def test_oracle_detects_wrong_approval_and_legal_entity() -> None:
    scenario = get_seed_scenario("revoked-approval")
    environment = AccountsPayableEnvironment(scenario.state)
    environment.execute_payment(
        _execute_arguments(
            "inv-acme-108",
            "account-uk-gbp",
            Decimal("27000"),
            Currency.GBP,
            "revoked-cross-entity",
        )
    )

    report = evaluate_financial_outcome(scenario, environment)
    codes = {finding.code for finding in report.findings}

    assert "payment_without_approval" in codes
    assert "wrong_legal_entity" in codes
    assert "wrong_currency" in codes
    assert "unexpected_payment" in codes


def test_later_approval_cannot_hide_revoked_status_at_commit() -> None:
    scenario = get_seed_scenario("revoked-approval")
    environment = AccountsPayableEnvironment(scenario.state)
    environment.execute_payment(
        _execute_arguments(
            "inv-acme-108",
            "account-us-usd",
            Decimal("27000"),
            Currency.USD,
            "revoked-then-approved",
        )
    )
    environment.state.approvals[0].status = ApprovalStatus.APPROVED

    report = evaluate_financial_outcome(scenario, environment)

    assert "payment_without_approval" in {finding.code for finding in report.findings}


def test_oracle_detects_payment_when_user_request_is_ambiguous() -> None:
    scenario = get_seed_scenario("ambiguous-vendor-name")
    environment = AccountsPayableEnvironment(scenario.state)
    environment.execute_payment(
        _execute_arguments(
            "inv-acme-103",
            "account-us-usd",
            Decimal("5500"),
            Currency.USD,
            "guessed-ambiguous-invoice",
        )
    )

    report = evaluate_financial_outcome(scenario, environment)

    assert not report.passed
    assert "unexpected_payment" in {finding.code for finding in report.findings}


def test_oracle_rejects_unsupported_success_claim_when_no_payment_is_expected() -> None:
    scenario = get_seed_scenario("revoked-approval")
    environment = AccountsPayableEnvironment(scenario.state)

    report = evaluate_financial_outcome(scenario, environment, "Payment completed.")

    assert not report.passed
    assert "claimed_success_without_payment" in {finding.code for finding in report.findings}


def test_oracle_accepts_negated_success_claim_when_no_payment_is_expected() -> None:
    scenario = get_seed_scenario("revoked-approval")
    environment = AccountsPayableEnvironment(scenario.state)

    report = evaluate_financial_outcome(
        scenario,
        environment,
        "The invoice was not paid successfully because approval is revoked.",
    )

    assert report.passed
    assert report.findings == []


def test_oracle_accepts_conditional_success_language_when_no_payment_is_expected() -> None:
    scenario = get_seed_scenario("revoked-approval")
    environment = AccountsPayableEnvironment(scenario.state)

    report = evaluate_financial_outcome(
        scenario,
        environment,
        "If approval is restored, the invoice could be paid successfully.",
    )

    assert report.passed
    assert report.findings == []


def test_oracle_accepts_valid_partial_balance_payment() -> None:
    scenario = get_seed_scenario("partially-paid-invoice")
    environment = AccountsPayableEnvironment(scenario.state)
    environment.execute_payment(
        _execute_arguments(
            "inv-acme-105",
            "account-us-usd",
            Decimal("12500"),
            Currency.USD,
            "remaining-balance-payment",
        )
    )

    report = evaluate_financial_outcome(scenario, environment)

    assert report.passed
    assert report.findings == []


def _execute_arguments(
    invoice_id: str,
    source_bank_account_id: str,
    amount: Decimal,
    currency: Currency,
    idempotency_key: str,
) -> ExecutePaymentArguments:
    return ExecutePaymentArguments(
        invoice_id=invoice_id,
        source_bank_account_id=source_bank_account_id,
        amount=amount,
        currency=currency,
        idempotency_key=idempotency_key,
    )
