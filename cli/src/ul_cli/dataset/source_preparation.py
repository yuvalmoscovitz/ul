from __future__ import annotations

import json
from collections.abc import Callable
from typing import Literal, TextIO

from pydantic import BaseModel, ConfigDict, model_validator
from ul import DatasetSourcePreparationError, InteractionRecord

from ul_cli.dataset_review import (
    DatasetEvidenceRunContext,
    DatasetSourcePreparationFailureEvidence,
)

from .evidence.customer import build_source_preparation_failure_evidence


class DatasetSourcePreparationFailureEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    interaction_id: str
    failure_stage: Literal["semantic_preparation"]
    failure_category: Literal[
        "source_semantic_preparation_failed",
        "source_outcome_projection_failed",
        "source_comparison_surface_incompatible",
    ]
    safe_diagnostic: str
    durability_state: Literal["pending", "persisted"] = "pending"
    retry_disposition: Literal["do_not_retry_in_campaign"] = "do_not_retry_in_campaign"
    review_disposition: Literal["inspect_source"] = "inspect_source"
    evidence: DatasetSourcePreparationFailureEvidence

    @model_validator(mode="after")
    def validate_evidence(self) -> DatasetSourcePreparationFailureEvent:
        if (
            self.interaction_id,
            self.failure_stage,
            self.failure_category,
            self.safe_diagnostic,
        ) != (
            self.evidence.interaction_id,
            self.evidence.failure_stage,
            self.evidence.reason_code,
            self.evidence.summary,
        ):
            raise ValueError("source preparation event must match its evidence")
        return self


def build_source_preparation_failure_event(
    source: InteractionRecord,
    error: DatasetSourcePreparationError,
    *,
    repetitions: int,
    max_environment_api_calls: int,
    planned_target_calls: int,
    run_context: DatasetEvidenceRunContext,
) -> DatasetSourcePreparationFailureEvent:
    evidence = build_source_preparation_failure_evidence(
        source,
        error,
        repetitions=repetitions,
        max_environment_api_calls=max_environment_api_calls,
        planned_target_calls=planned_target_calls,
        run_context=run_context,
    )
    return DatasetSourcePreparationFailureEvent(
        interaction_id=evidence.interaction_id,
        failure_stage=evidence.failure_stage,
        failure_category=evidence.reason_code,
        safe_diagnostic=evidence.summary,
        evidence=evidence,
    )


def persist_source_preparation_failure_event(
    event: DatasetSourcePreparationFailureEvent,
    output_stream: TextIO,
    durable_flush: Callable[[], None],
) -> DatasetSourcePreparationFailureEvent:
    if event.durability_state != "pending":
        raise ValueError("source preparation failure event was already persisted")
    output_stream.write(
        json.dumps(
            event.evidence.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
        )
        + "\n"
    )
    durable_flush()
    return event.model_copy(update={"durability_state": "persisted"})
