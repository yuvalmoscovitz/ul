from __future__ import annotations

from ul import (
    DatasetSemanticSettings,
    OpenAICompatibleDatasetSettings,
    OpenRouterDatasetSettings,
)

from ul_cli.dataset_review import DatasetEvidenceRunContext
from ul_cli.dataset_trial_journal import DatasetRunManifest


def manifest_incompatibility_reason(
    recorded: DatasetEvidenceRunContext,
    requested: DatasetEvidenceRunContext,
) -> str | None:
    checks = (
        ("fixture", recorded.fixture, requested.fixture),
        ("target", recorded.target, requested.target),
        ("projection", recorded.invariant_suite_sha256, requested.invariant_suite_sha256),
        ("operators", recorded.operators, requested.operators),
        (
            "target_timeout_seconds",
            recorded.target_timeout_seconds,
            requested.target_timeout_seconds,
        ),
        (
            "evaluator.llm_client",
            recorded.semantic_settings.llm_client,
            requested.semantic_settings.llm_client,
        ),
        (
            "evaluator.materiality_version",
            recorded.semantic_settings.materiality_evaluator_version_id,
            requested.semantic_settings.materiality_evaluator_version_id,
        ),
        (
            "evaluator.max_input_chars",
            recorded.semantic_settings.max_input_chars,
            requested.semantic_settings.max_input_chars,
        ),
        ("dataset", recorded.selected_dataset_sha256, requested.selected_dataset_sha256),
        ("redaction", recorded.redaction_policy_sha256, requested.redaction_policy_sha256),
    )
    return next((reason for reason, left, right in checks if left != right), None)


def restore_recorded_semantic_settings(manifest: DatasetRunManifest) -> DatasetSemanticSettings:
    recorded = manifest.run_context.semantic_settings
    llm_client = recorded.llm_client
    deconstruct = llm_client.role_config("deconstruct")
    render = llm_client.role_config("render")
    equivalence = llm_client.role_config("equivalence")
    materiality = llm_client.role_config("materiality")
    command = manifest.effective_command
    if command.semantic_provider_type == "openai-compatible":
        return OpenAICompatibleDatasetSettings(
            live_calls=command.semantic_live_calls,
            allow_external_data_processing=command.semantic_allow_external_data_processing,
            model=deconstruct.model,
            render_model=render.model,
            equivalence_model=equivalence.model,
            materiality_model=materiality.model,
            deconstruct_reasoning=deconstruct.reasoning_mode,
            render_reasoning=render.reasoning_mode,
            equivalence_reasoning=equivalence.reasoning_mode,
            max_input_chars=recorded.max_input_chars,
            max_output_tokens=deconstruct.max_output_tokens,
            max_render_tokens=render.max_output_tokens,
            max_response_bytes=llm_client.max_response_bytes,
            timeout_seconds=llm_client.timeout_seconds,
            provider_id=llm_client.provider_id,
            base_url=command.semantic_base_url,
        )
    assert llm_client.upstream_provider is not None
    return OpenRouterDatasetSettings(
        live_calls=command.semantic_live_calls,
        allow_external_data_processing=command.semantic_allow_external_data_processing,
        model=deconstruct.model,
        render_model=render.model,
        equivalence_model=equivalence.model,
        materiality_model=materiality.model,
        deconstruct_reasoning=deconstruct.reasoning_mode,
        render_reasoning=render.reasoning_mode,
        equivalence_reasoning=equivalence.reasoning_mode,
        max_input_chars=recorded.max_input_chars,
        max_output_tokens=deconstruct.max_output_tokens,
        max_render_tokens=render.max_output_tokens,
        max_response_bytes=llm_client.max_response_bytes,
        timeout_seconds=llm_client.timeout_seconds,
        upstream_provider=llm_client.upstream_provider,
    )


def effective_command_incompatibility_reason(
    recorded: DatasetRunManifest,
    requested: DatasetRunManifest,
) -> str | None:
    left = recorded.effective_command
    right = requested.effective_command
    checks = (
        ("run_config", left.run_config, right.run_config),
        ("invariant_suite_source", left.invariant_suite_source, right.invariant_suite_source),
        ("redaction_policy_source", left.redaction_policy_source, right.redaction_policy_source),
        ("redaction_state_path", left.redaction_state_path, right.redaction_state_path),
        ("redaction_state_sha256", left.redaction_state_sha256, right.redaction_state_sha256),
        ("augmentations_input_path", left.augmentations_input_path, right.augmentations_input_path),
        (
            "augmentations_input_sha256",
            left.augmentations_input_sha256,
            right.augmentations_input_sha256,
        ),
        (
            "augmentations_output_path",
            left.augmentations_output_path,
            right.augmentations_output_path,
        ),
    )
    return next((name for name, old, new in checks if old != new), None)
