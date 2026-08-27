from __future__ import annotations

import json

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


def test_openai_compatible_tool_call_is_projected_to_a_canonical_action() -> None:
    projection = OutcomeProjection.model_validate(
        {
            "tool_call": {
                "name": "/choices/0/message/tool_calls/0/function/name",
                "arguments": "/choices/0/message/tool_calls/0/function/arguments",
            },
            "private_json_pointers": ("/patient_id",),
        }
    )
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "record_observation",
                                "arguments": json.dumps(
                                    {
                                        "patient_id": "private-patient",
                                        "value": "118/77 mmHg",
                                        "body": {
                                            "subject": {"reference": "Patient/private-patient"},
                                            "category": [{"code": "vital-signs"}],
                                        },
                                        "metadata": {},
                                        "items": [],
                                    }
                                ),
                            },
                        }
                    ]
                }
            }
        ]
    }

    normalized = projection.project(response)

    assert normalized == {
        "action": "record_observation",
        "patient_id": "private-patient",
        "value": "118/77 mmHg",
        "body.subject.reference": "Patient/private-patient",
        "body.category[0].code": "vital-signs",
        "metadata": {},
        "items": [],
    }
    assert projection.public_result(normalized) == {
        "action": "record_observation",
        "patient_id": "[PRIVATE]",
        "value": "118/77 mmHg",
        "body.subject.reference": "Patient/private-patient",
        "body.category[0].code": "vital-signs",
        "metadata": {},
        "items": [],
    }


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ("not JSON", "valid JSON-encoded object string"),
        ("[]", "JSON-encoded object string"),
        ('{"ticket":1,"ticket":2}', "valid JSON-encoded object string"),
        ('{"action":"override"}', "must not contain the reserved action field"),
        ('{"value":NaN}', "valid JSON-encoded object string"),
        ('{"body":{"value":1},"body.value":2}', "ambiguous field names"),
    ],
)
def test_tool_call_projection_rejects_invalid_arguments_without_echoing_them(
    arguments: str,
    message: str,
) -> None:
    projection = OutcomeProjection.model_validate(
        {"tool_call": {"name": "/call/name", "arguments": "/call/arguments"}}
    )

    with pytest.raises(OutcomeProjectionError, match=message) as caught:
        projection.project({"call": {"name": "lookup", "arguments": arguments}})

    assert arguments not in str(caught.value)


def test_tool_call_projection_rejects_unpaired_surrogate_with_safe_diagnostic() -> None:
    projection = OutcomeProjection.model_validate(
        {"tool_call": {"name": "/call/name", "arguments": "/call/arguments"}}
    )
    arguments = '{"value":"' + chr(0xD800) + '"}'

    with pytest.raises(
        OutcomeProjectionError,
        match="must resolve to a valid JSON-encoded object string",
    ) as caught:
        projection.project({"call": {"name": "lookup", "arguments": arguments}})

    assert caught.value.__cause__ is None


def test_tool_call_projection_modes_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="requires exactly one"):
        OutcomeProjection.model_validate(
            {
                "action": "/result/action",
                "tool_call": {"name": "/call/name", "arguments": "/call/arguments"},
            }
        )


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({"call": {}}, "'tool_call.name' at selector '/call/name' does not resolve"),
        (
            {"call": {"name": 7, "arguments": "{}"}},
            "'tool_call.name' at selector '/call/name' must resolve to a non-empty string",
        ),
        (
            {"call": {"name": "lookup", "arguments": {}}},
            "'tool_call.arguments' at selector '/call/arguments' must resolve to a "
            "JSON-encoded object string",
        ),
    ],
)
def test_tool_call_projection_reports_missing_or_invalid_selected_fields(
    response: object,
    message: str,
) -> None:
    projection = OutcomeProjection.model_validate(
        {"tool_call": {"name": "/call/name", "arguments": "/call/arguments"}}
    )

    with pytest.raises(OutcomeProjectionError, match=message.replace("/", r"\/")):
        projection.project(response)  # type: ignore[arg-type]


def test_tool_call_projection_bounds_nested_argument_expansion() -> None:
    nested: object = "value"
    for _ in range(101):
        nested = {"nested": nested}
    projection = OutcomeProjection.model_validate(
        {"tool_call": {"name": "/call/name", "arguments": "/call/arguments"}}
    )

    with pytest.raises(OutcomeProjectionError, match="canonical flattening limits"):
        projection.project({"call": {"name": "lookup", "arguments": json.dumps({"root": nested})}})


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
