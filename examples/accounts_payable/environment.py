from __future__ import annotations

import json
from copy import deepcopy
from datetime import timedelta
from typing import Any

from pydantic import BaseModel, ValidationError

from examples.accounts_payable.models import (
    Approval,
    ApprovalStatus,
    BankAccount,
    EnvironmentState,
    Event,
    Invoice,
    Payment,
    PaymentOperation,
    PaymentStatus,
    ToolFaultKind,
    Vendor,
)
from examples.accounts_payable.tool_schemas import (
    TOOL_ARGUMENT_MODELS,
    ExecutePaymentArguments,
    GetApprovalStatusArguments,
    GetInvoiceArguments,
    GetPaymentAccountArguments,
    GetPaymentStatusArguments,
    GetVendorArguments,
    ListPaymentsArguments,
    SearchInvoicesArguments,
)


class PaymentToolTimeout(TimeoutError):
    def __init__(self, fault: ToolFaultKind) -> None:
        self.fault = fault
        super().__init__("Payment service timed out; the operation outcome is unknown.")


class ToolExecutionError(RuntimeError):
    pass


class AccountsPayableEnvironment:
    def __init__(self, state: EnvironmentState) -> None:
        self.state = state.model_copy(deep=True)
        self.initial_state = state.model_copy(deep=True)
        self.events: list[Event] = []
        self.initial_payment_ids = {payment.id for payment in state.payments}
        self._invoices = {invoice.id: invoice for invoice in self.state.invoices}
        self._initial_invoices = {invoice.id: invoice for invoice in self.initial_state.invoices}
        self._vendors = {vendor.id: vendor for vendor in self.state.vendors}
        self._bank_accounts = {account.id: account for account in self.state.bank_accounts}
        self._approvals_by_invoice = {
            approval.invoice_id: approval for approval in self.state.approvals
        }
        self._initial_approvals_by_invoice = {
            approval.invoice_id: approval for approval in self.initial_state.approvals
        }
        self._payments = {payment.id: payment for payment in self.state.payments}
        self._operations = {operation.id: operation for operation in self.state.payment_operations}
        self._operations_by_idempotency_key = {
            operation.idempotency_key: operation for operation in self.state.payment_operations
        }

    @property
    def new_payments(self) -> list[Payment]:
        return [
            payment for payment in self.state.payments if payment.id not in self.initial_payment_ids
        ]

    def search_invoices(self, arguments: SearchInvoicesArguments) -> list[Invoice]:
        invoices = self._read_invoices("search_invoices")
        normalized_query = arguments.query.casefold().strip()
        due_on_or_before = (
            None
            if arguments.due_on_or_before is None
            else self._parse_iso_date(arguments.due_on_or_before)
        )
        matches: list[Invoice] = []
        for invoice in invoices.values():
            vendor = self._vendors[invoice.vendor_id]
            searchable_text = " ".join(
                [invoice.id, invoice.reference, vendor.id, vendor.name]
            ).casefold()
            if normalized_query and normalized_query not in searchable_text:
                continue
            if due_on_or_before is not None and invoice.due_date > due_on_or_before:
                continue
            matches.append(invoice)
        return sorted(matches, key=lambda invoice: (invoice.due_date, invoice.id))

    def get_invoice(self, arguments: GetInvoiceArguments) -> Invoice:
        invoices = self._read_invoices("get_invoice")
        return self._require(invoices, arguments.invoice_id, "invoice")

    def get_vendor(self, arguments: GetVendorArguments) -> Vendor:
        self._consume_unsupported_read_fault("get_vendor")
        return self._require(self._vendors, arguments.vendor_id, "vendor")

    def get_approval_status(self, arguments: GetApprovalStatusArguments) -> Approval:
        use_initial_state = self._consume_fault("get_approval_status") == ToolFaultKind.STALE_READ
        approvals = (
            self._initial_approvals_by_invoice if use_initial_state else self._approvals_by_invoice
        )
        return self._require(approvals, arguments.invoice_id, "approval")

    def get_payment_account(self, arguments: GetPaymentAccountArguments) -> BankAccount:
        self._consume_unsupported_read_fault("get_payment_account")
        matches = [
            account
            for account in self._bank_accounts.values()
            if account.legal_entity_id == arguments.legal_entity_id
            and account.currency == arguments.currency
        ]
        if len(matches) != 1:
            raise ToolExecutionError(
                f"Expected exactly one matching payment account; found {len(matches)}."
            )
        return matches[0]

    def list_payments(self, arguments: ListPaymentsArguments) -> list[Payment]:
        self._consume_unsupported_read_fault("list_payments")
        operation_payment_ids: set[str] | None = None
        if arguments.idempotency_key is not None:
            operation = self._operations_by_idempotency_key.get(arguments.idempotency_key)
            operation_payment_ids = set() if operation is None else {operation.payment_id}
        return [
            payment
            for payment in self.state.payments
            if (arguments.invoice_id is None or payment.invoice_id == arguments.invoice_id)
            and (operation_payment_ids is None or payment.id in operation_payment_ids)
        ]

    def execute_payment(self, arguments: ExecutePaymentArguments) -> PaymentOperation:
        existing_operation = self._operations_by_idempotency_key.get(arguments.idempotency_key)
        if existing_operation is not None:
            existing_payment = self._payments[existing_operation.payment_id]
            if not self._matches_request(existing_payment, arguments):
                raise ToolExecutionError(
                    "The idempotency key was already used with different payment parameters."
                )
            return existing_operation

        fault = self._consume_fault("execute_payment")
        if fault == ToolFaultKind.TIMEOUT_BEFORE_COMMIT:
            self._record_event(
                "payment_timeout_before_commit", {"invoice_id": arguments.invoice_id}
            )
            raise PaymentToolTimeout(fault)
        if fault == ToolFaultKind.STALE_READ:
            raise ToolExecutionError("stale_read is not valid for execute_payment")

        invoice = self._require(self._invoices, arguments.invoice_id, "invoice")
        source_account = self._require(
            self._bank_accounts, arguments.source_bank_account_id, "bank account"
        )
        vendor = self._require(self._vendors, invoice.vendor_id, "vendor")
        approval = self._approvals_by_invoice.get(invoice.id)
        if not vendor.active:
            raise ToolExecutionError("The vendor is inactive.")
        if source_account.currency != arguments.currency:
            raise ToolExecutionError("The source account does not support the requested currency.")
        if source_account.balance < arguments.amount:
            raise ToolExecutionError("The source account has insufficient funds.")

        sequence = len(self.state.payment_operations) + 1
        operation_id = f"op-{sequence:04d}"
        payment_id = f"pay-{sequence:04d}"
        committed_at = self._next_time()
        payment = Payment(
            id=payment_id,
            operation_id=operation_id,
            invoice_id=invoice.id,
            vendor_id=vendor.id,
            legal_entity_id=invoice.legal_entity_id,
            source_bank_account_id=source_account.id,
            recipient_account_reference=vendor.recipient_account_reference,
            amount=arguments.amount,
            currency=arguments.currency,
            idempotency_key=arguments.idempotency_key,
            status=PaymentStatus.COMMITTED,
            committed_at=committed_at,
            approval_id_at_commit=None if approval is None else approval.id,
            approval_status_at_commit=None if approval is None else approval.status,
            approved_amount_at_commit=(None if approval is None else approval.approved_amount),
            approval_expires_at_commit=None if approval is None else approval.expires_at,
        )
        operation = PaymentOperation(
            id=operation_id,
            idempotency_key=arguments.idempotency_key,
            payment_id=payment.id,
            status=PaymentStatus.COMMITTED,
            created_at=committed_at,
        )
        source_account.balance -= arguments.amount
        self.state.payments.append(payment)
        self.state.payment_operations.append(operation)
        self._payments[payment.id] = payment
        self._operations[operation.id] = operation
        self._operations_by_idempotency_key[operation.idempotency_key] = operation
        self._record_event(
            "payment_committed",
            {
                "payment_id": payment.id,
                "invoice_id": invoice.id,
                "amount": str(payment.amount),
                "currency": payment.currency.value,
            },
        )
        if fault == ToolFaultKind.TIMEOUT_AFTER_COMMIT:
            self._record_event("payment_timeout_after_commit", {"payment_id": payment.id})
            raise PaymentToolTimeout(fault)
        return operation

    def get_payment_status(self, arguments: GetPaymentStatusArguments) -> PaymentOperation:
        self._consume_unsupported_read_fault("get_payment_status")
        return self._require(self._operations, arguments.operation_id, "payment operation")

    def revoke_approval(self, invoice_id: str) -> None:
        approval = self._require(self._approvals_by_invoice, invoice_id, "approval")
        approval.status = ApprovalStatus.REVOKED
        self._record_event("approval_revoked", {"invoice_id": invoice_id})

    def dispatch_json(self, tool_name: str, arguments_json: str) -> str:
        argument_model = TOOL_ARGUMENT_MODELS.get(tool_name)
        if argument_model is None:
            return self._error_json("unknown_tool", f"Tool {tool_name!r} is not allowlisted.")
        try:
            arguments = argument_model.model_validate_json(arguments_json)
            result = self._dispatch_validated(tool_name, arguments)
        except ValidationError as error:
            return self._error_json("invalid_arguments", str(error))
        except PaymentToolTimeout as error:
            return self._error_json(
                "timeout",
                str(error),
                outcome="unknown",
                fault=error.fault.value,
            )
        except ToolExecutionError as error:
            return self._error_json("tool_error", str(error))
        if isinstance(result, BaseModel):
            serialized_result: Any = result.model_dump(mode="json")
        else:
            serialized_result = [item.model_dump(mode="json") for item in result]
        return json.dumps({"ok": True, "result": serialized_result}, sort_keys=True)

    def _dispatch_validated(self, tool_name: str, arguments: BaseModel) -> Any:
        method = getattr(self, tool_name)
        return method(arguments)

    def _read_invoices(self, tool_name: str) -> dict[str, Invoice]:
        use_initial_state = self._consume_fault(tool_name) == ToolFaultKind.STALE_READ
        return self._initial_invoices if use_initial_state else self._invoices

    def _consume_unsupported_read_fault(self, tool_name: str) -> None:
        fault = self._consume_fault(tool_name)
        if fault is not None:
            raise ToolExecutionError(f"{fault.value} is not supported by {tool_name}")

    def _consume_fault(self, tool_name: str) -> ToolFaultKind | None:
        faults = self.state.tool_faults.get(tool_name)
        if not faults:
            return None
        return faults.pop(0)

    def _record_event(self, kind: str, details: dict[str, Any]) -> None:
        self.events.append(
            Event(
                sequence=len(self.events) + 1,
                occurred_at=self.state.clock,
                kind=kind,
                details=deepcopy(details),
            )
        )

    def _next_time(self):
        self.state.clock += timedelta(milliseconds=1)
        return self.state.clock

    @staticmethod
    def _matches_request(payment: Payment, arguments: ExecutePaymentArguments) -> bool:
        return (
            payment.invoice_id == arguments.invoice_id
            and payment.source_bank_account_id == arguments.source_bank_account_id
            and payment.amount == arguments.amount
            and payment.currency == arguments.currency
        )

    @staticmethod
    def _require(items: dict[str, Any], item_id: str, kind: str):
        try:
            return items[item_id]
        except KeyError as error:
            raise ToolExecutionError(f"Unknown {kind}: {item_id}") from error

    @staticmethod
    def _parse_iso_date(value: str):
        from datetime import date

        try:
            return date.fromisoformat(value)
        except ValueError as error:
            raise ToolExecutionError("due_on_or_before must be an ISO date") from error

    @staticmethod
    def _error_json(code: str, message: str, **details: str) -> str:
        return json.dumps(
            {"ok": False, "error": {"code": code, "message": message, **details}},
            sort_keys=True,
        )
