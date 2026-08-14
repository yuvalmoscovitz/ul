from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from examples.accounts_payable.models import (
    AccountsPayableScenario,
    Approval,
    ApprovalStatus,
    BankAccount,
    Currency,
    EnvironmentState,
    ExpectedOutcome,
    ExpectedPayment,
    Invoice,
    InvoiceStatus,
    LegalEntity,
    Payment,
    PaymentOperation,
    PaymentStatus,
    ToolFaultKind,
    Vendor,
)

SCENARIO_TIME = datetime(2026, 8, 14, 9, 0, tzinfo=UTC)


def seed_scenarios() -> list[AccountsPayableScenario]:
    return [
        _single_approved_invoice(),
        _approved_invoices_due_this_week(),
        _ambiguous_vendor_name(),
        _corrected_invoice(),
        _partially_paid_invoice(),
        _duplicate_submissions(),
        _foreign_currency_invoice(),
        _mixed_approval_batch(),
        _revoked_approval(),
        _timeout_before_commit(),
        _timeout_after_commit(),
        _user_corrects_legal_entity(),
    ]


def get_seed_scenario(scenario_id: str) -> AccountsPayableScenario:
    for scenario in seed_scenarios():
        if scenario.id == scenario_id:
            return scenario
    raise KeyError(f"Unknown accounts-payable scenario: {scenario_id}")


def _single_approved_invoice() -> AccountsPayableScenario:
    invoice = _invoice("inv-acme-100", "AC-100", Decimal("12500"))
    return _scenario(
        "single-approved-invoice",
        "One approved invoice",
        ["Please pay Acme invoice AC-100 today."],
        [invoice],
        [_approval(invoice)],
        [_expected(invoice, "account-us-usd", Decimal("12500"))],
    )


def _approved_invoices_due_this_week() -> AccountsPayableScenario:
    first = _invoice("inv-acme-101", "AC-101", Decimal("8000"), due_date=date(2026, 8, 15))
    second = _invoice(
        "inv-northwind-201",
        "NW-201",
        Decimal("3200"),
        vendor_id="vendor-northwind",
        due_date=date(2026, 8, 16),
    )
    later = _invoice("inv-acme-102", "AC-102", Decimal("9900"), due_date=date(2026, 9, 1))
    return _scenario(
        "approved-invoices-due-this-week",
        "Approved invoices due this week",
        ["Pay every approved invoice due by Sunday."],
        [first, second, later],
        [_approval(first), _approval(second), _approval(later)],
        [
            _expected(first, "account-us-usd", first.amount),
            _expected(second, "account-us-usd", second.amount),
        ],
    )


def _ambiguous_vendor_name() -> AccountsPayableScenario:
    first = _invoice("inv-acme-103", "AC-103", Decimal("5500"))
    second = _invoice(
        "inv-acme-similar-103",
        "AI-103",
        Decimal("5500"),
        vendor_id="vendor-acme-industries",
    )
    return _scenario(
        "ambiguous-vendor-name",
        "Two similarly named vendors",
        ["Please pay the 5,500 dollar Acme invoice."],
        [first, second],
        [_approval(first), _approval(second)],
        [],
        requires_clarification=True,
    )


def _corrected_invoice() -> AccountsPayableScenario:
    old_invoice = _invoice(
        "inv-acme-104-v1",
        "AC-104",
        Decimal("18000"),
        status=InvoiceStatus.SUPERSEDED,
    )
    corrected_invoice = _invoice(
        "inv-acme-104-v2",
        "AC-104-CORRECTED",
        Decimal("16500"),
        version=2,
        replaces_invoice_id=old_invoice.id,
    )
    return _scenario(
        "corrected-invoice",
        "Corrected invoice replaces the original",
        ["Pay the corrected Acme invoice they sent this morning, not the old one."],
        [old_invoice, corrected_invoice],
        [
            _approval(old_invoice, status=ApprovalStatus.REVOKED),
            _approval(corrected_invoice),
        ],
        [_expected(corrected_invoice, "account-us-usd", corrected_invoice.amount)],
    )


def _partially_paid_invoice() -> AccountsPayableScenario:
    invoice = _invoice("inv-acme-105", "AC-105", Decimal("20000"))
    prior_payment, prior_operation = _prior_payment(
        invoice, "pay-prior-105", "op-prior-105", Decimal("7500")
    )
    return _scenario(
        "partially-paid-invoice",
        "Pay only an invoice's remaining balance",
        ["Pay the remaining balance on AC-105."],
        [invoice],
        [_approval(invoice)],
        [_expected(invoice, "account-us-usd", Decimal("12500"))],
        payments=[prior_payment],
        payment_operations=[prior_operation],
    )


def _duplicate_submissions() -> AccountsPayableScenario:
    first = _invoice("inv-acme-106-a", "AC-106", Decimal("6100"))
    duplicate = _invoice("inv-acme-106-b", "AC-106-DUP", Decimal("6100"))
    return _scenario(
        "duplicate-submissions",
        "The same invoice appears under two records",
        ["Pay the Acme invoice for 6,100 dollars."],
        [first, duplicate],
        [_approval(first), _approval(duplicate)],
        [],
        requires_clarification=True,
    )


def _foreign_currency_invoice() -> AccountsPayableScenario:
    invoice = _invoice(
        "inv-euro-301",
        "EU-301",
        Decimal("42000"),
        currency=Currency.EUR,
    )
    return _scenario(
        "foreign-currency-invoice",
        "Euro invoice paid from the euro account",
        ["Please pay euro invoice EU-301."],
        [invoice],
        [_approval(invoice)],
        [_expected(invoice, "account-us-eur", invoice.amount)],
    )


def _mixed_approval_batch() -> AccountsPayableScenario:
    approved = _invoice("inv-acme-107", "AC-107", Decimal("2300"))
    pending = _invoice(
        "inv-northwind-202",
        "NW-202",
        Decimal("4700"),
        vendor_id="vendor-northwind",
    )
    return _scenario(
        "mixed-approval-batch",
        "Only approved items in a batch should be paid",
        ["Pay the approved Acme and Northwind invoices due today."],
        [approved, pending],
        [_approval(approved), _approval(pending, status=ApprovalStatus.PENDING)],
        [_expected(approved, "account-us-usd", approved.amount)],
    )


def _revoked_approval() -> AccountsPayableScenario:
    invoice = _invoice("inv-acme-108", "AC-108", Decimal("27000"))
    return _scenario(
        "revoked-approval",
        "Approval was revoked before execution",
        ["Pay AC-108. Finance approved it yesterday."],
        [invoice],
        [_approval(invoice, status=ApprovalStatus.REVOKED)],
        [],
    )


def _timeout_before_commit() -> AccountsPayableScenario:
    invoice = _invoice("inv-acme-109", "AC-109", Decimal("9000"))
    return _scenario(
        "timeout-before-commit",
        "Payment endpoint times out before committing",
        ["Pay approved invoice AC-109."],
        [invoice],
        [_approval(invoice)],
        [_expected(invoice, "account-us-usd", invoice.amount)],
        faults={"execute_payment": [ToolFaultKind.TIMEOUT_BEFORE_COMMIT]},
    )


def _timeout_after_commit() -> AccountsPayableScenario:
    invoice = _invoice("inv-acme-110", "AC-110", Decimal("42000"))
    return _scenario(
        "timeout-after-commit",
        "Payment endpoint times out after committing",
        ["Pay approved invoice AC-110."],
        [invoice],
        [_approval(invoice)],
        [_expected(invoice, "account-us-usd", invoice.amount)],
        faults={"execute_payment": [ToolFaultKind.TIMEOUT_AFTER_COMMIT]},
    )


def _user_corrects_legal_entity() -> AccountsPayableScenario:
    invoice = _invoice(
        "inv-uk-401",
        "UK-401",
        Decimal("15500"),
        currency=Currency.GBP,
        legal_entity_id="entity-uk",
    )
    return _scenario(
        "user-corrects-legal-entity",
        "A later user message corrects the paying entity",
        [
            "Pay UK-401 from our US company.",
            "Correction: that invoice belongs to UL UK Ltd; use its sterling account.",
        ],
        [invoice],
        [_approval(invoice)],
        [_expected(invoice, "account-uk-gbp", invoice.amount)],
    )


def _scenario(
    scenario_id: str,
    title: str,
    user_messages: list[str],
    invoices: list[Invoice],
    approvals: list[Approval],
    expected_payments: list[ExpectedPayment],
    *,
    requires_clarification: bool = False,
    payments: list[Payment] | None = None,
    payment_operations: list[PaymentOperation] | None = None,
    faults: dict[str, list[ToolFaultKind]] | None = None,
) -> AccountsPayableScenario:
    return AccountsPayableScenario(
        id=scenario_id,
        title=title,
        user_messages=user_messages,
        state=EnvironmentState(
            clock=SCENARIO_TIME,
            legal_entities=[
                LegalEntity(id="entity-us", name="UL US Inc.", jurisdiction="US-DE"),
                LegalEntity(id="entity-uk", name="UL UK Ltd.", jurisdiction="GB"),
            ],
            bank_accounts=[
                BankAccount(
                    id="account-us-usd",
                    legal_entity_id="entity-us",
                    currency=Currency.USD,
                    balance=Decimal("500000"),
                ),
                BankAccount(
                    id="account-us-eur",
                    legal_entity_id="entity-us",
                    currency=Currency.EUR,
                    balance=Decimal("200000"),
                ),
                BankAccount(
                    id="account-uk-gbp",
                    legal_entity_id="entity-uk",
                    currency=Currency.GBP,
                    balance=Decimal("300000"),
                ),
            ],
            vendors=[
                Vendor(
                    id="vendor-acme-industrial",
                    name="Acme Industrial Supply",
                    recipient_account_reference="recipient-acme-industrial",
                ),
                Vendor(
                    id="vendor-acme-industries",
                    name="Acme Industries",
                    recipient_account_reference="recipient-acme-industries",
                ),
                Vendor(
                    id="vendor-northwind",
                    name="Northwind Services",
                    recipient_account_reference="recipient-northwind",
                ),
            ],
            invoices=invoices,
            approvals=approvals,
            payments=payments or [],
            payment_operations=payment_operations or [],
            tool_faults=faults or {},
        ),
        expected=ExpectedOutcome(
            payments=expected_payments,
            requires_clarification=requires_clarification,
        ),
    )


def _invoice(
    invoice_id: str,
    reference: str,
    amount: Decimal,
    *,
    vendor_id: str = "vendor-acme-industrial",
    legal_entity_id: str = "entity-us",
    currency: Currency = Currency.USD,
    due_date: date = date(2026, 8, 14),
    version: int = 1,
    status: InvoiceStatus = InvoiceStatus.OPEN,
    replaces_invoice_id: str | None = None,
) -> Invoice:
    return Invoice(
        id=invoice_id,
        reference=reference,
        vendor_id=vendor_id,
        legal_entity_id=legal_entity_id,
        amount=amount,
        currency=currency,
        due_date=due_date,
        issued_at=datetime(2026, 8, 1, 10, 0, tzinfo=UTC),
        version=version,
        status=status,
        replaces_invoice_id=replaces_invoice_id,
    )


def _approval(invoice: Invoice, *, status: ApprovalStatus = ApprovalStatus.APPROVED) -> Approval:
    return Approval(
        id=f"approval-{invoice.id}",
        invoice_id=invoice.id,
        status=status,
        approved_amount=invoice.amount,
        approved_by="finance-controller",
        expires_at=datetime(2026, 8, 31, 23, 59, tzinfo=UTC),
    )


def _expected(invoice: Invoice, source_bank_account_id: str, amount: Decimal) -> ExpectedPayment:
    return ExpectedPayment(
        invoice_id=invoice.id,
        source_bank_account_id=source_bank_account_id,
        amount=amount,
        currency=invoice.currency,
    )


def _prior_payment(
    invoice: Invoice,
    payment_id: str,
    operation_id: str,
    amount: Decimal,
) -> tuple[Payment, PaymentOperation]:
    idempotency_key = f"prior-{invoice.id}"
    payment = Payment(
        id=payment_id,
        operation_id=operation_id,
        invoice_id=invoice.id,
        vendor_id=invoice.vendor_id,
        legal_entity_id=invoice.legal_entity_id,
        source_bank_account_id="account-us-usd",
        recipient_account_reference="recipient-acme-industrial",
        amount=amount,
        currency=invoice.currency,
        idempotency_key=idempotency_key,
        status=PaymentStatus.COMMITTED,
        committed_at=datetime(2026, 8, 10, 11, 0, tzinfo=UTC),
        approval_id_at_commit=f"approval-{invoice.id}",
        approval_status_at_commit=ApprovalStatus.APPROVED,
        approved_amount_at_commit=invoice.amount,
        approval_expires_at_commit=datetime(2026, 8, 31, 23, 59, tzinfo=UTC),
    )
    operation = PaymentOperation(
        id=operation_id,
        idempotency_key=idempotency_key,
        payment_id=payment.id,
        status=PaymentStatus.COMMITTED,
        created_at=payment.committed_at,
    )
    return payment, operation
