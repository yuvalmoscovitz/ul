from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError
from ul.deconstruction import (
    _decode_semantic_json_fields,
    _RenderedInput,
    _semantic_equivalence_response_schema,
    _semantic_frame_response_schema,
)
from ul_core.dataset import SemanticFrame


def _assert_closed_required_objects(node: Any, root: dict[str, Any]) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
        if "$ref" in node:
            assert node["$ref"].removeprefix("#/$defs/") in root["$defs"]
        for value in node.values():
            _assert_closed_required_objects(value, root)
    elif isinstance(node, list):
        for value in node:
            _assert_closed_required_objects(value, root)


@pytest.mark.parametrize(
    "schema",
    [
        _semantic_frame_response_schema(observed_output_present=True),
        _semantic_frame_response_schema(observed_output_present=False),
        _semantic_equivalence_response_schema(),
        _RenderedInput.model_json_schema(),
    ],
)
def test_generation_schemas_satisfy_strict_object_contract(schema: dict[str, Any]) -> None:
    _assert_closed_required_objects(schema, schema)
    assert "metadata" not in schema["properties"]


@pytest.mark.parametrize("value", [None, True, 12, 1.5, "0012", {"nested": [1, "two"]}, []])
def test_wire_json_preserves_customer_value_types(value: Any) -> None:
    frame = {
        "interaction_id": "case-1",
        "extractor_version": "test",
        "factors": [
            {
                "id": "factor-1",
                "confidence": 1,
                "status": "explicit",
                "kind": "entity",
                "role": "customer_value",
                "value_json": json.dumps(value),
            }
        ],
        "outcomes": [
            {
                "id": "outcome-1",
                "confidence": 1,
                "status": "observed",
                "position": 0,
                "kind": "action",
                "predicate": "store",
                "fields_json": '{"customer.key":{"items":[1,true,null]}}',
            }
        ],
    }
    _decode_semantic_json_fields(frame)
    validated = SemanticFrame.model_validate_json(json.dumps(frame))
    assert validated.factors[0].value == value
    assert type(validated.factors[0].value) is type(value)
    assert validated.outcomes[0].fields == {"customer.key": {"items": [1, True, None]}}


@pytest.mark.parametrize("encoded", [None, 123, "invalid JSON"])
def test_malformed_wire_value_is_rejected(encoded: Any) -> None:
    with pytest.raises(ValueError):
        _decode_semantic_json_fields({"factors": [{"value_json": encoded}]})


def test_ambiguous_wire_and_domain_fields_are_rejected() -> None:
    with pytest.raises(ValueError):
        _decode_semantic_json_fields({"factors": [{"value_json": '"a"', "value": "b"}]})


def test_wire_object_still_passes_through_domain_validation() -> None:
    frame = {
        "interaction_id": "case-1",
        "extractor_version": "test",
        "communication_acts": [
            {
                "id": "act-1",
                "confidence": 1,
                "status": "explicit",
                "kind": "terse",
                "attributes_json": '["not an object"]',
            }
        ],
    }
    _decode_semantic_json_fields(frame)
    with pytest.raises(ValidationError):
        SemanticFrame.model_validate_json(json.dumps(frame))
