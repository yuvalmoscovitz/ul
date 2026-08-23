from __future__ import annotations

import pytest
from pydantic import ValidationError
from ul.outcome_projection import OutcomeProjection, OutcomeProjectionError


def test_acme_business_fields_are_projected_deterministically() -> None:
    projection = OutcomeProjection(
        action="/result/action",
        status="/result/status",
        resource_id="/result/order_id",
        decision="/result/decision",
        amount="/result/amount",
    )
    response = {
        "assistant": {"message": "Refund approved", "private_reasoning": "customer-secret"},
        "result": {
            "action": "refund",
            "status": "completed",
            "order_id": "order-123",
            "decision": "approved",
            "amount": 42.5,
        },
        "tool_events": [{"name": "issue_refund"}],
    }

    assert projection.project(response) == {
        "action": "refund",
        "status": "completed",
        "resource_id": "order-123",
        "decision": "approved",
        "amount": 42.5,
    }
    assert "customer-secret" not in str(projection.project(response))
    assert len(projection.digest) == 64


def test_complete_result_has_independently_filtered_public_view() -> None:
    projection = OutcomeProjection(
        complete_result="/result",
        private_json_pointers=("/customer/email", "/internal_note"),
    )
    normalized = projection.project(
        {
            "result": {
                "decision": "approved",
                "customer": {"email": "secret@example.test"},
                "internal_note": "never disclose",
            },
            "raw_secret": "not selected",
        }
    )

    assert normalized["customer"] == {"email": "secret@example.test"}
    public = projection.public_result(normalized)
    assert public["customer"] == {"email": "[PRIVATE]"}
    assert public["internal_note"] == "[PRIVATE]"
    assert "secret@example.test" not in str(public)
    assert "never disclose" not in str(public)
    assert "not selected" not in str(public)


@pytest.mark.parametrize(
    ("projection", "response", "message"),
    [
        (
            OutcomeProjection(action="/result/action"),
            {"result": {}},
            "'action' at selector '/result/action' does not resolve",
        ),
        (
            OutcomeProjection(action="/result/action"),
            {"result": {"action": 7}},
            "'action' at selector '/result/action' must resolve to a non-empty string",
        ),
        (
            OutcomeProjection(amount="/result/amount"),
            {"result": {"amount": True}},
            "'amount' at selector '/result/amount' must resolve to a finite number",
        ),
        (
            OutcomeProjection(complete_result="/result"),
            {"result": []},
            "'complete_result' at selector '/result' must resolve to a JSON object",
        ),
    ],
)
def test_invalid_projection_identifies_exact_field_and_selector(
    projection: OutcomeProjection,
    response: object,
    message: str,
) -> None:
    with pytest.raises(OutcomeProjectionError, match=message.replace("/", r"\/")):
        projection.project(response)  # type: ignore[arg-type]


def test_projection_rejects_oversized_normalized_result() -> None:
    projection = OutcomeProjection(complete_result="/result")

    with pytest.raises(OutcomeProjectionError, match="64000-byte normalized-result limit"):
        projection.project({"result": {"value": "x" * 64_001}})


def test_projection_rejects_overlapping_private_pointers() -> None:
    with pytest.raises(ValidationError, match="private outcome pointers must not overlap"):
        OutcomeProjection(
            complete_result="/result",
            private_json_pointers=("/customer", "/customer/email"),
        )


def test_public_projection_normalizes_unresolved_redaction_failure() -> None:
    projection = OutcomeProjection(
        complete_result="/result",
        private_json_pointers=("/customer/email",),
    )

    with pytest.raises(
        OutcomeProjectionError,
        match="'private_json_pointers' at selector '/customer/email' does not resolve",
    ):
        projection.public_result({"customer": {}})
