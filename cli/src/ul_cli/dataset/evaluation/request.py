from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ul import DatasetEvaluationMode

from ul_cli.dataset_trial_journal import (
    DatasetRunManifest,
    journal_anchor_path,
    journal_path,
    manifest_path,
    read_dataset_run_manifest,
)

from ..evidence.persistence import (
    default_augmentations_output,
    durable_evidence_marker_manifest_sha256,
)

_MAXIMUM_REPETITIONS = 100
_DEFAULT_MAXIMUM_ENVIRONMENT_API_CALLS = 100
_DEFAULT_TARGET_TIMEOUT_SECONDS = 30.0


class DatasetRequestError(ValueError):
    def __init__(self, message: str, *, parameter: str | None = None) -> None:
        super().__init__(message)
        self.parameter = parameter


@dataclass(frozen=True)
class DatasetEvaluationRequest:
    data: Path | None
    environment_config: Path | None
    target: str | None
    target_artifacts: tuple[Path, ...]
    http_preset: Literal["generic-json", "openai-chat"] | None
    request_json_template: str | None
    response_json_pointer: str | None
    agent_model: str | None
    headers_from_env: tuple[str, ...]
    confirm_target: str | None
    output: Path | None
    augmentations_input: Path | None
    augmentations_output: Path | None
    no_save_augmentations: bool
    invariants: Path | None
    evaluation_mode: DatasetEvaluationMode
    operators: tuple[str, ...] | None
    limit: int | None
    repetitions: int | None
    concurrency: int | None
    target_timeout_seconds: float | None
    max_environment_api_calls: int | None
    allow_environment_network: bool
    confirm_test_environment: bool
    confirm_request_isolation: bool
    confirm_safe_test_target: bool
    allow_insecure_http: bool
    dry_run: bool
    json_output: bool
    progress_json: bool
    show_sensitive_values: bool
    resume: Path | None
    resolve_quarantine_after: Literal["environment-reset", "environment-replacement"] | None
    redaction_policy: Path | None
    redaction_state: Path | None
    expected_environment_origin: str | None
    expected_environment_config_sha256: str | None
    expected_redaction_policy_sha256: str | None
    show_report_guidance: bool


@dataclass(frozen=True)
class NormalizedDatasetEvaluationRequest:
    requested: DatasetEvaluationRequest
    recorded_manifest: DatasetRunManifest | None
    output: Path | None
    augmentations_input: Path | None
    augmentations_output: Path | None
    redaction_state: Path | None
    evaluation_mode: Literal["variance"]
    repetitions: int
    concurrency: int
    target_timeout_seconds: float
    max_environment_api_calls: int
    limit: int
    operators: tuple[str, ...] | None
    allow_environment_network: bool
    confirm_test_environment: bool
    allow_insecure_http: bool
    augmentations_input_was_explicit: bool
    augmentations_output_was_explicit: bool
    redaction_state_was_explicit: bool


def normalize_dataset_evaluation_request(
    request: DatasetEvaluationRequest,
) -> NormalizedDatasetEvaluationRequest:
    augmentations_input_was_explicit = request.augmentations_input is not None
    augmentations_output_was_explicit = request.augmentations_output is not None
    redaction_state_was_explicit = request.redaction_state is not None
    recorded_manifest = _load_recorded_manifest(request.resume)

    repetitions = request.repetitions
    concurrency = request.concurrency
    max_environment_api_calls = request.max_environment_api_calls
    allow_environment_network = request.allow_environment_network
    confirm_test_environment = request.confirm_test_environment
    allow_insecure_http = request.allow_insecure_http
    operators = request.operators
    limit = request.limit
    no_save_augmentations = request.no_save_augmentations
    augmentations_output = request.augmentations_output
    redaction_state = request.redaction_state
    if recorded_manifest is not None:
        recorded_command = recorded_manifest.effective_command
        recorded_run_config = recorded_command.run_config
        repetitions = repetitions or recorded_run_config.repetitions
        concurrency = concurrency or recorded_run_config.concurrency
        max_environment_api_calls = (
            max_environment_api_calls or recorded_run_config.target.max_environment_api_calls
        )
        allow_environment_network = (
            allow_environment_network or recorded_run_config.target.allow_network_egress
        )
        confirm_test_environment = (
            confirm_test_environment or recorded_run_config.target.test_environment_confirmed
        )
        allow_insecure_http = allow_insecure_http or recorded_run_config.target.allow_insecure_http
        if operators is None:
            operators = recorded_manifest.selected_operator_ids
        if request.data is None:
            limit = len(recorded_manifest.selected_records)
        no_save_augmentations = not recorded_command.save_augmentations
        if augmentations_output is None and recorded_command.augmentations_output_path is not None:
            augmentations_output = Path(recorded_command.augmentations_output_path)
        if redaction_state is None and recorded_command.redaction_state_path is not None:
            redaction_state = Path(recorded_command.redaction_state_path)

    repetitions = repetitions or 3
    concurrency = concurrency or 1
    target_timeout_seconds = _resolve_target_timeout_seconds(
        request.target_timeout_seconds, recorded_manifest
    )
    max_environment_api_calls = (
        max_environment_api_calls or _DEFAULT_MAXIMUM_ENVIRONMENT_API_CALLS
    )
    limit = limit or 10
    _validate_request_options(
        request,
        recorded_manifest=recorded_manifest,
        repetitions=repetitions,
    )

    augmentations_input = _restore_recorded_augmentation_input(
        request.augmentations_input, recorded_manifest
    )
    output = request.output
    if request.resume is not None:
        if output is not None and output.resolve() != request.resume.resolve():
            raise DatasetRequestError(
                "--output must point to the same file as --resume, or be omitted",
                parameter="--output",
            )
        output = request.resume
    augmentations_output = _resolve_augmentation_output(
        augmentations_input=augmentations_input,
        augmentations_output=augmentations_output,
        no_save_augmentations=no_save_augmentations,
        evidence_output=output,
    )
    if (
        output is not None
        and augmentations_output is not None
        and output.resolve() == augmentations_output.resolve()
    ):
        raise DatasetRequestError(
            "--augmentations-output must differ from --output",
            parameter="--augmentations-output",
        )

    return NormalizedDatasetEvaluationRequest(
        requested=request,
        recorded_manifest=recorded_manifest,
        output=output,
        augmentations_input=augmentations_input,
        augmentations_output=augmentations_output,
        redaction_state=redaction_state,
        evaluation_mode="variance",
        repetitions=repetitions,
        concurrency=concurrency,
        target_timeout_seconds=target_timeout_seconds,
        max_environment_api_calls=max_environment_api_calls,
        limit=limit,
        operators=operators,
        allow_environment_network=allow_environment_network,
        confirm_test_environment=confirm_test_environment,
        allow_insecure_http=allow_insecure_http,
        augmentations_input_was_explicit=augmentations_input_was_explicit,
        augmentations_output_was_explicit=augmentations_output_was_explicit,
        redaction_state_was_explicit=redaction_state_was_explicit,
    )


def _load_recorded_manifest(resume: Path | None) -> DatasetRunManifest | None:
    if resume is None:
        return None
    durable_paths = (manifest_path(resume), journal_path(resume), journal_anchor_path(resume))
    durable_path_presence = tuple(os.path.lexists(path) for path in durable_paths)
    try:
        evidence_manifest_sha256 = durable_evidence_marker_manifest_sha256(resume)
    except (OSError, ValueError) as error:
        message = str(error) if isinstance(error, ValueError) else error.__class__.__name__
        raise DatasetRequestError(
            f"cannot safely inspect resume evidence ({message})", parameter="--resume"
        ) from None
    if evidence_manifest_sha256 is not None and not all(durable_path_presence):
        raise DatasetRequestError(
            "durable evidence requires its manifest, journal, and anchor sidecars; restore all "
            "three together because legacy replay is unsafe",
            parameter="--resume",
        )
    if any(durable_path_presence) and not all(durable_path_presence):
        raise DatasetRequestError(
            "durable resume sidecars are incomplete; restore the manifest, journal, and anchor "
            "together",
            parameter="--resume",
        )
    if not all(durable_path_presence):
        return None
    try:
        recorded_manifest = read_dataset_run_manifest(manifest_path(resume))
    except (OSError, ValueError) as error:
        message = str(error) if isinstance(error, ValueError) else error.__class__.__name__
        raise DatasetRequestError(
            f"cannot safely read recorded run manifest ({message})", parameter="--resume"
        ) from None
    if (
        evidence_manifest_sha256 is not None
        and evidence_manifest_sha256 != recorded_manifest.manifest_sha256
    ):
        raise DatasetRequestError(
            "primary evidence marker does not match its durable manifest", parameter="--resume"
        )
    return recorded_manifest


def _resolve_target_timeout_seconds(
    requested: float | None, recorded_manifest: DatasetRunManifest | None
) -> float:
    recorded = (
        recorded_manifest.effective_command.run_config.target.trial_timeout_seconds
        if recorded_manifest is not None
        else _DEFAULT_TARGET_TIMEOUT_SECONDS
    )
    resolved = recorded if requested is None else requested
    if not math.isfinite(resolved) or resolved <= 0:
        raise DatasetRequestError(
            "target timeout must be positive and finite",
            parameter="--target-timeout-seconds",
        )
    return resolved


def _validate_request_options(
    request: DatasetEvaluationRequest,
    *,
    recorded_manifest: DatasetRunManifest | None,
    repetitions: int,
) -> None:
    direct_http_options_used = (
        request.http_preset is not None
        or request.request_json_template is not None
        or request.response_json_pointer is not None
        or request.agent_model is not None
        or bool(request.headers_from_env)
    )
    if request.environment_config is not None and request.target is not None:
        raise DatasetRequestError(
            "--target cannot be combined with --environment-config", parameter="--target"
        )
    if request.target_artifacts and request.target is None:
        raise DatasetRequestError(
            "--target-artifact requires --target", parameter="--target-artifact"
        )
    if direct_http_options_used and request.target is None:
        raise DatasetRequestError(
            "direct HTTP mapping options require --target with an HTTP URL",
            parameter="--target",
        )
    if request.confirm_target is not None and request.target is None:
        raise DatasetRequestError(
            "--confirm-target requires --target", parameter="--confirm-target"
        )
    if repetitions > _MAXIMUM_REPETITIONS:
        raise DatasetRequestError(
            f"repetitions cannot exceed {_MAXIMUM_REPETITIONS}", parameter="--repetitions"
        )
    if request.data is None and recorded_manifest is None:
        raise DatasetRequestError(
            "DATA is required unless --resume has a durable run manifest", parameter="DATA"
        )
    if (request.json_output or request.show_sensitive_values) and not request.dry_run:
        option = "--show-sensitive-values" if request.show_sensitive_values else "--json"
        raise DatasetRequestError(f"{option} requires --dry-run", parameter=option)
    if request.evaluation_mode != "variance":
        raise DatasetRequestError(
            f"evaluation mode '{request.evaluation_mode}' is not implemented; use 'variance'. "
            "Historical dataset output is grounding evidence, not an expected answer.",
            parameter="--evaluation-mode",
        )


def _resolve_augmentation_output(
    *,
    augmentations_input: Path | None,
    augmentations_output: Path | None,
    no_save_augmentations: bool,
    evidence_output: Path | None,
) -> Path | None:
    if augmentations_output is not None and no_save_augmentations:
        raise DatasetRequestError(
            "--augmentations-output cannot be used with --no-save-augmentations",
            parameter="--augmentations-output",
        )
    if augmentations_input is not None and augmentations_output is not None:
        raise DatasetRequestError(
            "--augmentations-input cannot be combined with --augmentations-output",
            parameter="--augmentations-input",
        )
    if (
        augmentations_output is not None
        or no_save_augmentations
        or augmentations_input is not None
        or evidence_output is None
    ):
        return augmentations_output
    return default_augmentations_output(evidence_output)


def _restore_recorded_augmentation_input(
    requested: Path | None,
    recorded_manifest: DatasetRunManifest | None,
) -> Path | None:
    if requested is not None or recorded_manifest is None:
        return requested
    recorded_path = recorded_manifest.effective_command.augmentations_input_path
    return Path(recorded_path) if recorded_path is not None else None
