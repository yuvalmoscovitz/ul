from __future__ import annotations

from pydantic import JsonValue
from ul import (
    DatasetSemanticSettings,
    InteractionRecord,
    semantic_deconstructor_identity,
)
from ul.dataset_invariants import DatasetInvariantSuite
from ul.http_environment import JsonHttpTargetConfig
from ul.llm import llm_client_config_from_dataset_settings
from ul.material_variance import material_variance_evaluator_version_from_llm_config

from ul_cli.dataset_review import (
    DatasetEvidenceRedactionCoverage,
    DatasetEvidenceRunContext,
    DatasetEvidenceSemanticSettings,
    create_dataset_evidence_run_context,
)
from ul_cli.dataset_run_config import DatasetRunConfig

from ..evaluation.operators import dataset_operator_identity


def build_dataset_evidence_run_context(
    *,
    selected_records: tuple[InteractionRecord, ...],
    selected_operator_ids: tuple[str, ...],
    run_config: DatasetRunConfig,
    invariant_suite: DatasetInvariantSuite | None,
    target_config: JsonHttpTargetConfig | None,
    target_receipt: dict[str, JsonValue] | None = None,
    settings: DatasetSemanticSettings,
    redaction_policy_sha256: str | None = None,
    redaction_coverage: tuple[DatasetEvidenceRedactionCoverage, ...] = (),
) -> DatasetEvidenceRunContext:
    semantic_settings = dataset_evidence_semantic_settings(settings)
    return create_dataset_evidence_run_context(
        selected_records=selected_records,
        operators=tuple(
            dataset_operator_identity(reference) for reference in selected_operator_ids
        ),
        run_config=run_config,
        invariant_suite_sha256=(invariant_suite.sha256 if invariant_suite is not None else None),
        target_config=target_config,
        target_receipt=target_receipt,
        semantic_settings=semantic_settings,
        redaction_policy_sha256=redaction_policy_sha256,
        redaction_coverage=redaction_coverage,
    )


def dataset_evidence_semantic_settings(
    settings: DatasetSemanticSettings,
) -> DatasetEvidenceSemanticSettings:
    llm_config = llm_client_config_from_dataset_settings(settings)
    return DatasetEvidenceSemanticSettings(
        llm_client=llm_config.evidence_identity(),
        max_input_chars=settings.max_input_chars,
        deconstructor_identity=semantic_deconstructor_identity(settings),
        materiality_evaluator_version_id=(
            material_variance_evaluator_version_from_llm_config(llm_config).id
        ),
    )
