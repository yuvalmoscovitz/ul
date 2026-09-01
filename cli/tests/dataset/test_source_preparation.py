from __future__ import annotations

import io
import json

import pytest
from ul import DatasetSemanticPreparationError, InteractionRecord
from ul_cli.dataset.source_preparation import (
    build_source_preparation_failure_event,
    persist_source_preparation_failure_event,
)

from ._factories import _run_context


def test_source_preparation_event_is_safe_typed_and_persisted_before_consumption() -> None:
    private_input = "PRIVATE_SOURCE_CONTENT"
    source = InteractionRecord(
        id="interaction-1",
        raw_input=private_input,
        raw_observed_output={"private": "PRIVATE_OUTPUT_CONTENT"},
    )
    event = build_source_preparation_failure_event(
        source,
        DatasetSemanticPreparationError(),
        repetitions=1,
        max_environment_api_calls=10,
        planned_target_calls=4,
        run_context=_run_context((source,)),
    )
    output = io.StringIO()
    persisted_lines: list[dict[str, object]] = []

    def durable_flush() -> None:
        persisted_lines.extend(json.loads(line) for line in output.getvalue().splitlines())

    persisted_event = persist_source_preparation_failure_event(
        event,
        output,
        durable_flush,
    )

    assert event.durability_state == "pending"
    assert persisted_event.durability_state == "persisted"
    assert persisted_event.failure_stage == "semantic_preparation"
    assert persisted_event.failure_category == "source_semantic_preparation_failed"
    assert persisted_event.retry_disposition == "do_not_retry_in_campaign"
    assert persisted_event.review_disposition == "inspect_source"
    assert persisted_lines == [persisted_event.evidence.model_dump(mode="json", exclude_none=True)]
    assert private_input not in persisted_event.model_dump_json()
    assert "PRIVATE_OUTPUT_CONTENT" not in persisted_event.model_dump_json()

    with pytest.raises(ValueError, match="already persisted"):
        persist_source_preparation_failure_event(
            persisted_event,
            output,
            durable_flush,
        )
