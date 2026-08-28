from __future__ import annotations

from typing import Literal

from pydantic import JsonValue
from ul import DatasetSemanticSettings, InteractionRecord, semantic_deconstructor_identity
from ul.dataset_invariants import DatasetInvariantSuite
from ul.http_environment import JsonHttpTargetConfig

from ul_cli.dataset_review import (
    DatasetEvidenceRedactionCoverage,
    DatasetEvidenceRunContext,
    DatasetEvidenceSemanticSettings,
    create_dataset_evidence_run_context,
)

from ..evaluation.operators import dataset_operator_identity


def build_dataset_evidence_run_context(
    *,
    selected_records: tuple[InteractionRecord, ...],
    selected_operator_ids: tuple[str, ...],
    evaluation_mode: Literal["variance"] = "variance",
    repetitions: int,
    target_timeout_seconds: float = 30.0,
    invariant_suite: DatasetInvariantSuite | None,
    target_config: JsonHttpTargetConfig | None,
    target_receipt: dict[str, JsonValue] | None = None,
    settings: DatasetSemanticSettings,
    redaction_policy_sha256: str | None = None,
    redaction_coverage: tuple[DatasetEvidenceRedactionCoverage, ...] = (),
) -> DatasetEvidenceRunContext:
    return create_dataset_evidence_run_context(
        selected_records=selected_records,
        operators=tuple(
            dataset_operator_identity(reference) for reference in selected_operator_ids
        ),
        evaluation_mode=evaluation_mode,
        repetitions=repetitions,
        target_timeout_seconds=target_timeout_seconds,
        invariant_suite_sha256=(invariant_suite.sha256 if invariant_suite is not None else None),
        target_config=target_config,
        target_receipt=target_receipt,
        semantic_settings=DatasetEvidenceSemanticSettings(
            provider=settings.semantic_provider_id,
            endpoint_sha256=settings.semantic_endpoint_sha256,
            model=settings.model,
            render_model=settings.render_model,
            equivalence_model=settings.equivalence_model,
            deconstruct_reasoning=settings.deconstruct_reasoning,
            render_reasoning=settings.render_reasoning,
            equivalence_reasoning=settings.equivalence_reasoning,
            max_input_chars=settings.max_input_chars,
            max_output_tokens=settings.max_output_tokens,
            max_render_tokens=settings.max_render_tokens,
            max_response_bytes=settings.max_response_bytes,
            timeout_seconds=settings.timeout_seconds,
            deconstructor_identity=semantic_deconstructor_identity(settings),
        ),
        redaction_policy_sha256=redaction_policy_sha256,
        redaction_coverage=redaction_coverage,
    )
