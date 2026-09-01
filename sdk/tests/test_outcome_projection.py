from __future__ import annotations

import json
from typing import cast

import pytest
import ul.outcome_projection as outcome_projection
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


def test_construct_projects_json_string_spread_to_a_canonical_action() -> None:
    projection = _encoded_construct(private_json_pointers=("/patient_id",))
    response = {
        "call": {
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
        }
    }

    normalized = projection.project(response)

    expected = {
        "action": "record_observation",
        "patient_id": "private-patient",
        "value": "118/77 mmHg",
        "body.subject.reference": "Patient/private-patient",
        "body.category[0].code": "vital-signs",
        "metadata": {},
        "items": [],
    }
    assert normalized == expected
    assert projection.public_result(normalized) == {**expected, "patient_id": "[PRIVATE]"}


def test_construct_spreads_existing_object_without_provider_specific_decoding() -> None:
    projection = OutcomeProjection.model_validate(
        {
            "compose": {
                "fields": {"action": "/content/0/name"},
                "spread": {"selector": "/content/0/input", "flatten": True},
            }
        }
    )

    assert projection.project(
        {
            "content": [
                {
                    "name": "create_ticket",
                    "input": {"customer": {"tier": "gold"}, "urgent": True},
                }
            ]
        }
    ) == {
        "action": "create_ticket",
        "customer.tier": "gold",
        "urgent": True,
    }


@pytest.mark.parametrize(
    ("spread_value", "message"),
    [
        ("not JSON", "valid JSON-encoded object string"),
        ("[]", "JSON-encoded object string"),
        ('{"ticket":1,"ticket":2}', "valid JSON-encoded object string"),
        ('{"action":"override"}', "collides with a selected composed field"),
        ('{"value":NaN}', "valid JSON-encoded object string"),
        ('{"body":{"value":1},"body.value":2}', "ambiguous field names"),
    ],
)
def test_construct_rejects_invalid_encoded_spread_without_echoing_it(
    spread_value: str,
    message: str,
) -> None:
    projection = _encoded_construct()

    with pytest.raises(OutcomeProjectionError, match=message) as caught:
        projection.project({"call": {"name": "lookup", "arguments": spread_value}})

    assert spread_value not in str(caught.value)


def test_construct_rejects_unpaired_surrogate_with_safe_diagnostic() -> None:
    projection = _encoded_construct()
    spread_value = '{"value":"' + chr(0xD800) + '"}'

    with pytest.raises(
        OutcomeProjectionError,
        match="must resolve to a valid JSON-encoded object string",
    ) as caught:
        projection.project({"call": {"name": "lookup", "arguments": spread_value}})

    assert caught.value.__cause__ is None


def test_construct_projection_modes_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="requires exactly one"):
        OutcomeProjection.model_validate(
            {
                "action": "/result/action",
                "compose": {"fields": {"action": "/call/name"}},
            }
        )


def test_compose_fields_are_immutable_after_confirmation() -> None:
    projection = OutcomeProjection.model_validate({"compose": {"fields": {"action": "/call/name"}}})
    assert projection.compose is not None
    original_digest = projection.digest

    with pytest.raises(TypeError):
        cast(dict[str, str], projection.compose.fields)["action"] = "/other"

    assert projection.digest == original_digest
    assert projection.project({"call": {"name": "lookup"}}) == {"action": "lookup"}


def test_compose_rejects_oversized_field_selector() -> None:
    with pytest.raises(ValidationError, match="selectors must contain at most 1000"):
        OutcomeProjection.model_validate({"compose": {"fields": {"action": "/" + "x" * 1_000}}})


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            {"call": {}},
            "'compose.fields.action' at selector '/call/name' does not resolve",
        ),
        (
            {"call": {"name": 7, "arguments": "{}"}},
            "'action' at selector '/call/name' must resolve to a non-empty string",
        ),
        (
            {"call": {"name": "lookup", "arguments": {}}},
            "'compose.spread' at selector '/call/arguments' must resolve to a "
            "JSON-encoded object string",
        ),
    ],
)
def test_construct_reports_missing_or_invalid_selected_fields(
    response: object,
    message: str,
) -> None:
    projection = _encoded_construct()

    with pytest.raises(OutcomeProjectionError, match=message.replace("/", r"\/")):
        projection.project(response)  # type: ignore[arg-type]


def test_construct_bounds_nested_spread_expansion() -> None:
    nested: object = "value"
    for _ in range(101):
        nested = {"nested": nested}
    projection = _encoded_construct()

    with pytest.raises(OutcomeProjectionError, match="canonical flattening limits"):
        projection.project({"call": {"name": "lookup", "arguments": json.dumps({"root": nested})}})


@pytest.mark.parametrize("flatten", [False, True])
def test_construct_bounds_wide_existing_object_incrementally(flatten: bool) -> None:
    projection = OutcomeProjection.model_validate(
        {
            "compose": {
                "spread": {
                    "selector": "/input",
                    "flatten": flatten,
                }
            }
        }
    )
    wide = {f"field_{index}": index for index in range(10_001)}

    with pytest.raises(
        OutcomeProjectionError,
        match=r"normalized-result limit|canonical flattening limits",
    ):
        projection.project({"input": wide})


def _encoded_construct(
    *,
    private_json_pointers: tuple[str, ...] = (),
) -> OutcomeProjection:
    return OutcomeProjection.model_validate(
        {
            "compose": {
                "fields": {"action": "/call/name"},
                "spread": {
                    "selector": "/call/arguments",
                    "decode": "json_string",
                    "flatten": True,
                },
            },
            "private_json_pointers": private_json_pointers,
        }
    )


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


@pytest.mark.parametrize("action_count", [12, 14])
def test_projection_accepts_bounded_action_arrays(action_count: int) -> None:
    projection = OutcomeProjection.model_validate({"compose": {"fields": {"actions": "/actions"}}})
    actions = [
        {
            "action": "slack.message",
            "path": "data/slack/slack.json",
            "pointer": f"/messages/C006/{index}",
            "text": f"Message {index}: " + "x" * 60,
            "channel_pointer": f"/messages/C006/{index}",
            "target": "unread emails",
            "procedure": "SOP-FIN-AP-004",
        }
        for index in range(action_count)
    ]
    encoded_size = len(json.dumps({"actions": actions}, separators=(",", ":")).encode("utf-8"))

    assert encoded_size < 4_000
    assert projection.project({"actions": actions}) == {"actions": actions}


def test_projection_reports_exact_depth_limit() -> None:
    projection = OutcomeProjection(complete_result="/result")
    nested: object = "value"
    for _ in range(101):
        nested = {"nested": nested}

    with pytest.raises(
        OutcomeProjectionError,
        match="has depth 101, exceeding the 100-level normalized-result limit",
    ):
        projection.project({"result": nested})


def test_projection_reports_exact_node_limit() -> None:
    projection = OutcomeProjection(complete_result="/result")

    with pytest.raises(
        OutcomeProjectionError,
        match="has 10001 nodes, exceeding the 10000-node normalized-result limit",
    ):
        projection.project({"result": {str(index): index for index in range(10_000)}})


def test_projection_checks_wide_arrays_incrementally() -> None:
    class CountingList(list[object]):
        yielded = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            for value in super().__iter__():
                self.yielded += 1
                yield value

    projection = OutcomeProjection.model_validate({"compose": {"fields": {"actions": "/actions"}}})
    actions = CountingList(range(20_000))

    with pytest.raises(OutcomeProjectionError, match="10000-node normalized-result limit"):
        projection.project({"actions": actions})  # type: ignore[arg-type]

    assert actions.yielded < len(actions)


def test_projection_rejects_character_lower_bound_before_json_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = OutcomeProjection.model_validate({"compose": {"fields": {"value": "/value"}}})

    def unexpected_encoding(_: object) -> int:
        raise AssertionError("oversized strings must fail before JSON encoding")

    monkeypatch.setattr(outcome_projection, "_encoded_json_size", unexpected_encoding)

    with pytest.raises(OutcomeProjectionError, match="64000-byte normalized-result limit"):
        projection.project({"value": "\u0000" * 64_001})


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
