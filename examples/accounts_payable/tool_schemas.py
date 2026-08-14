from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import Field

from examples.accounts_payable.models import Currency, StrictModel


class SearchInvoicesArguments(StrictModel):
    query: str = ""
    due_on_or_before: str | None = None


class GetInvoiceArguments(StrictModel):
    invoice_id: str


class GetVendorArguments(StrictModel):
    vendor_id: str


class GetApprovalStatusArguments(StrictModel):
    invoice_id: str


class GetPaymentAccountArguments(StrictModel):
    legal_entity_id: str
    currency: Currency


class ListPaymentsArguments(StrictModel):
    invoice_id: str | None = None
    idempotency_key: str | None = None


class ExecutePaymentArguments(StrictModel):
    invoice_id: str
    source_bank_account_id: str
    amount: Decimal = Field(gt=0)
    currency: Currency
    idempotency_key: str = Field(min_length=8, max_length=255)


class GetPaymentStatusArguments(StrictModel):
    operation_id: str


TOOL_ARGUMENT_MODELS = {
    "search_invoices": SearchInvoicesArguments,
    "get_invoice": GetInvoiceArguments,
    "get_vendor": GetVendorArguments,
    "get_approval_status": GetApprovalStatusArguments,
    "get_payment_account": GetPaymentAccountArguments,
    "list_payments": ListPaymentsArguments,
    "execute_payment": ExecutePaymentArguments,
    "get_payment_status": GetPaymentStatusArguments,
}


TOOL_DESCRIPTIONS = {
    "search_invoices": "Search invoices by vendor name, reference, or invoice ID.",
    "get_invoice": "Get the current invoice record by its exact ID.",
    "get_vendor": "Get a vendor and its payment destination.",
    "get_approval_status": "Get the current approval for an invoice.",
    "get_payment_account": "Find a source account for a legal entity and currency.",
    "list_payments": "List prior payments, optionally by invoice or idempotency key.",
    "execute_payment": "Commit a synthetic payment. A timeout can leave the outcome unknown.",
    "get_payment_status": "Get the status of a known payment operation.",
}


def openrouter_tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": TOOL_DESCRIPTIONS[tool_name],
                "parameters": argument_model.model_json_schema(),
            },
        }
        for tool_name, argument_model in TOOL_ARGUMENT_MODELS.items()
    ]
