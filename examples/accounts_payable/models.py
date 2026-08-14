from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class Currency(StrEnum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"


class InvoiceStatus(StrEnum):
    OPEN = "open"
    SUPERSEDED = "superseded"
    VOID = "void"


class ApprovalStatus(StrEnum):
    APPROVED = "approved"
    REVOKED = "revoked"
    PENDING = "pending"


class PaymentStatus(StrEnum):
    COMMITTED = "committed"
    CANCELLED = "cancelled"


class ToolFaultKind(StrEnum):
    STALE_READ = "stale_read"
    TIMEOUT_BEFORE_COMMIT = "timeout_before_commit"
    TIMEOUT_AFTER_COMMIT = "timeout_after_commit"


class LegalEntity(StrictModel):
    id: str
    name: str
    jurisdiction: str


class BankAccount(StrictModel):
    id: str
    legal_entity_id: str
    currency: Currency
    balance: Decimal = Field(ge=0)


class Vendor(StrictModel):
    id: str
    name: str
    recipient_account_reference: str
    active: bool = True


class Invoice(StrictModel):
    id: str
    reference: str
    vendor_id: str
    legal_entity_id: str
    amount: Decimal = Field(gt=0)
    currency: Currency
    due_date: date
    issued_at: datetime
    version: int = Field(ge=1)
    status: InvoiceStatus = InvoiceStatus.OPEN
    replaces_invoice_id: str | None = None


class Approval(StrictModel):
    id: str
    invoice_id: str
    status: ApprovalStatus
    approved_amount: Decimal = Field(gt=0)
    approved_by: str
    expires_at: datetime


class Payment(StrictModel):
    id: str
    operation_id: str
    invoice_id: str
    vendor_id: str
    legal_entity_id: str
    source_bank_account_id: str
    recipient_account_reference: str
    amount: Decimal = Field(gt=0)
    currency: Currency
    idempotency_key: str
    status: PaymentStatus
    committed_at: datetime
    approval_id_at_commit: str | None = None
    approval_status_at_commit: ApprovalStatus | None = None
    approved_amount_at_commit: Decimal | None = Field(default=None, gt=0)
    approval_expires_at_commit: datetime | None = None


class PaymentOperation(StrictModel):
    id: str
    idempotency_key: str
    payment_id: str
    status: PaymentStatus
    created_at: datetime


class Event(StrictModel):
    sequence: int = Field(ge=1)
    occurred_at: datetime
    kind: str
    details: dict[str, Any]


class EnvironmentState(StrictModel):
    clock: datetime
    legal_entities: list[LegalEntity]
    bank_accounts: list[BankAccount]
    vendors: list[Vendor]
    invoices: list[Invoice]
    approvals: list[Approval]
    payments: list[Payment] = Field(default_factory=lambda: list[Payment]())
    payment_operations: list[PaymentOperation] = Field(
        default_factory=lambda: list[PaymentOperation]()
    )
    tool_faults: dict[str, list[ToolFaultKind]] = Field(
        default_factory=lambda: dict[str, list[ToolFaultKind]]()
    )


class ExpectedPayment(StrictModel):
    invoice_id: str
    source_bank_account_id: str
    amount: Decimal = Field(gt=0)
    currency: Currency


class ExpectedOutcome(StrictModel):
    payments: list[ExpectedPayment] = Field(default_factory=lambda: list[ExpectedPayment]())
    requires_clarification: bool = False


class AccountsPayableScenario(StrictModel):
    id: str
    title: str
    user_messages: list[str] = Field(min_length=1)
    state: EnvironmentState
    expected: ExpectedOutcome


class OracleSeverity(StrEnum):
    HIGH = "high"
    CRITICAL = "critical"


class OracleFinding(StrictModel):
    code: str
    severity: OracleSeverity
    message: str
    payment_ids: list[str] = Field(default_factory=lambda: list[str]())


class OracleReport(StrictModel):
    scenario_id: str
    passed: bool
    findings: list[OracleFinding]
