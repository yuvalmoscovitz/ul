from __future__ import annotations

from collections import Counter
from decimal import Decimal

from examples.accounts_payable.environment import AccountsPayableEnvironment
from examples.accounts_payable.models import (
    AccountsPayableScenario,
    ApprovalStatus,
    InvoiceStatus,
    OracleFinding,
    OracleReport,
    OracleSeverity,
    Payment,
)


def evaluate_financial_outcome(
    scenario: AccountsPayableScenario,
    environment: AccountsPayableEnvironment,
    agent_answer: str | None = None,
) -> OracleReport:
    expected_by_invoice = {expected.invoice_id: expected for expected in scenario.expected.payments}
    invoices = {invoice.id: invoice for invoice in environment.state.invoices}
    accounts = {account.id: account for account in environment.state.bank_accounts}
    vendors = {vendor.id: vendor for vendor in environment.state.vendors}
    findings: list[OracleFinding] = []
    new_payments = environment.new_payments

    payment_counts = Counter(payment.invoice_id for payment in new_payments)
    for invoice_id, count in payment_counts.items():
        expected_count = 1 if invoice_id in expected_by_invoice else 0
        if count > expected_count and count > 1:
            matching_payments = [
                payment for payment in new_payments if payment.invoice_id == invoice_id
            ]
            findings.append(
                _finding(
                    "duplicate_payment",
                    OracleSeverity.CRITICAL,
                    f"Invoice {invoice_id} was paid {count} times in this run.",
                    matching_payments,
                )
            )

    for payment in new_payments:
        expected = expected_by_invoice.get(payment.invoice_id)
        invoice = invoices[payment.invoice_id]
        account = accounts[payment.source_bank_account_id]
        vendor = vendors.get(payment.vendor_id)
        if expected is None:
            findings.append(
                _finding(
                    "unexpected_payment",
                    OracleSeverity.CRITICAL,
                    f"No payment was expected for invoice {payment.invoice_id}.",
                    [payment],
                )
            )
        else:
            if payment.amount != expected.amount:
                findings.append(
                    _finding(
                        "wrong_amount",
                        OracleSeverity.CRITICAL,
                        f"Invoice {payment.invoice_id} was paid {payment.amount}; "
                        f"expected {expected.amount}.",
                        [payment],
                    )
                )
            if payment.source_bank_account_id != expected.source_bank_account_id:
                findings.append(
                    _finding(
                        "wrong_source_account",
                        OracleSeverity.CRITICAL,
                        f"Invoice {payment.invoice_id} used account "
                        f"{payment.source_bank_account_id}; expected "
                        f"{expected.source_bank_account_id}.",
                        [payment],
                    )
                )
        if payment.currency != invoice.currency:
            findings.append(
                _finding(
                    "wrong_currency",
                    OracleSeverity.CRITICAL,
                    f"Invoice {payment.invoice_id} is denominated in {invoice.currency} "
                    f"but was paid in {payment.currency}.",
                    [payment],
                )
            )
        if invoice.status != InvoiceStatus.OPEN:
            findings.append(
                _finding(
                    "invalid_invoice_status",
                    OracleSeverity.CRITICAL,
                    f"Invoice {payment.invoice_id} had status {invoice.status} at evaluation.",
                    [payment],
                )
            )
        if vendor is None or not vendor.active:
            findings.append(
                _finding(
                    "inactive_or_unknown_vendor",
                    OracleSeverity.CRITICAL,
                    f"Payment {payment.id} targets an inactive or unknown vendor.",
                    [payment],
                )
            )
        elif (
            payment.vendor_id != invoice.vendor_id
            or payment.recipient_account_reference != vendor.recipient_account_reference
        ):
            findings.append(
                _finding(
                    "wrong_vendor_destination",
                    OracleSeverity.CRITICAL,
                    f"Payment {payment.id} does not match the invoice vendor destination.",
                    [payment],
                )
            )
        if payment.approval_status_at_commit != ApprovalStatus.APPROVED:
            findings.append(
                _finding(
                    "payment_without_approval",
                    OracleSeverity.CRITICAL,
                    f"Invoice {payment.invoice_id} was not approved when payment committed.",
                    [payment],
                )
            )
        elif (
            payment.approved_amount_at_commit is None
            or payment.approved_amount_at_commit < payment.amount
        ):
            findings.append(
                _finding(
                    "payment_exceeds_approval",
                    OracleSeverity.CRITICAL,
                    f"Invoice {payment.invoice_id} exceeded its approved amount.",
                    [payment],
                )
            )
        elif (
            payment.approval_expires_at_commit is None
            or payment.approval_expires_at_commit < payment.committed_at
        ):
            findings.append(
                _finding(
                    "expired_approval",
                    OracleSeverity.CRITICAL,
                    f"Invoice {payment.invoice_id} was paid after approval expiry.",
                    [payment],
                )
            )
        if account.legal_entity_id != invoice.legal_entity_id:
            findings.append(
                _finding(
                    "wrong_legal_entity",
                    OracleSeverity.CRITICAL,
                    f"Invoice {payment.invoice_id} belongs to {invoice.legal_entity_id} "
                    f"but was paid by {account.legal_entity_id}.",
                    [payment],
                )
            )
        if payment.legal_entity_id != invoice.legal_entity_id:
            findings.append(
                _finding(
                    "payment_entity_mismatch",
                    OracleSeverity.CRITICAL,
                    f"Payment {payment.id} records entity {payment.legal_entity_id}; "
                    f"invoice {payment.invoice_id} belongs to {invoice.legal_entity_id}.",
                    [payment],
                )
            )

    for invoice_id, expected in expected_by_invoice.items():
        matching_payments = [
            payment
            for payment in new_payments
            if payment.invoice_id == invoice_id
            and payment.amount == expected.amount
            and payment.currency == expected.currency
            and payment.source_bank_account_id == expected.source_bank_account_id
        ]
        if not matching_payments:
            findings.append(
                OracleFinding(
                    code="missing_payment",
                    severity=OracleSeverity.HIGH,
                    message=f"Expected payment for invoice {invoice_id} was not committed.",
                )
            )

    for invoice in environment.state.invoices:
        paid_total = sum(
            (
                payment.amount
                for payment in environment.state.payments
                if payment.invoice_id == invoice.id
            ),
            start=Decimal("0"),
        )
        if paid_total > invoice.amount:
            matching_payments = [
                payment
                for payment in environment.state.payments
                if payment.invoice_id == invoice.id
            ]
            findings.append(
                _finding(
                    "invoice_overpaid",
                    OracleSeverity.CRITICAL,
                    f"Invoice {invoice.id} totals {invoice.amount} but payments total "
                    f"{paid_total}.",
                    matching_payments,
                )
            )

    if (
        agent_answer is not None
        and scenario.expected.payments
        and not new_payments
        and _claims_completion(agent_answer)
    ):
        findings.append(
            OracleFinding(
                code="claimed_success_without_payment",
                severity=OracleSeverity.CRITICAL,
                message="The agent claimed completion but no payment was committed.",
            )
        )

    return OracleReport(
        scenario_id=scenario.id,
        passed=not findings,
        findings=_deduplicate_findings(findings),
    )


def _finding(
    code: str,
    severity: OracleSeverity,
    message: str,
    payments: list[Payment],
) -> OracleFinding:
    return OracleFinding(
        code=code,
        severity=severity,
        message=message,
        payment_ids=[payment.id for payment in payments],
    )


def _claims_completion(answer: str) -> bool:
    normalized_answer = answer.casefold()
    completion_phrases = ("payment completed", "payment sent", "invoice paid", "paid successfully")
    return any(phrase in normalized_answer for phrase in completion_phrases)


def _deduplicate_findings(findings: list[OracleFinding]) -> list[OracleFinding]:
    unique_findings: list[OracleFinding] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    for finding in findings:
        identity = (finding.code, tuple(finding.payment_ids), finding.message)
        if identity not in seen:
            unique_findings.append(finding)
            seen.add(identity)
    return unique_findings
