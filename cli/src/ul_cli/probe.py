from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, cast

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

import httpx
import typer
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError
from ul import (
    DatasetEvaluationResult,
    DatasetSemanticSettings,
    EvaluationCase,
    EvaluatorModelPreflight,
    ExecutionEvidence,
    InteractionRecord,
    load_dataset_semantic_settings,
)
from ul.environment import validate_execution_evidence
from ul.http_environment import (
    JsonHttpEnvironmentConnection,
    JsonHttpTargetConfig,
    json_http_environment_calls_per_execution,
    json_http_environment_capabilities,
    json_http_environment_config_sha256,
    json_http_environment_config_urls,
    load_json_http_environment_config,
    validate_json_http_environment_configuration,
)
from ul.local_target import (
    LocalTargetConfig,
    LocalTargetConnection,
)
from ul.outcome_projection import OutcomeProjection, OutcomeProjectionError
from ul.probe_execution import OutcomeProjectionExecutionError
from ul_core.models import ConversationRole, ConversationTurn

from ul_cli.dataset.environment.initialize import create_isolated_response_target_config
from ul_cli.dataset.evaluation.command import preflight_evaluator
from ul_cli.dataset.evaluation.operators import dataset_operator_identity, validate_operator_ids
from ul_cli.dataset.evaluation.records import (
    DatasetInputError,
    load_interaction_records,
    validate_model_input_bounds,
)
from ul_cli.dataset.evaluation.runner import evaluate_interaction_records
from ul_cli.dataset.evidence.persistence import (
    create_durable_evidence_output,
    default_augmentations_output,
    durable_evidence_marker_manifest_sha256,
    load_evaluator_preflight,
    open_resume_output,
    persist_evaluator_preflight,
)
from ul_cli.dataset.presentation.evaluation import print_dataset_results
from ul_cli.dataset.presentation.runtime import console, print_dataset_plain
from ul_cli.dataset.progress import (
    CampaignControlRequested,
    CampaignProgressRuntime,
    CampaignSignalControl,
    create_campaign_progress_runtime,
    create_probe_next_commands,
)
from ul_cli.dataset.storage.private_files import create_private_output, open_resume_descriptor
from ul_cli.dataset_augmentation_ledger import (
    DatasetAugmentationLedger,
    DatasetAugmentationLedgerSemanticSettings,
    create_dataset_augmentation_generation_context,
    create_private_augmentation_ledger,
    open_augmentation_ledger_for_resume,
)
from ul_cli.dataset_campaign import DatasetCampaignPlan, create_dataset_campaign_plan
from ul_cli.dataset_review import (
    DatasetEvidenceRunContext,
    DatasetEvidenceSemanticSettings,
    create_dataset_evidence_run_context,
)
from ul_cli.dataset_trial_journal import (
    DatasetTrialJournal,
    create_dataset_run_manifest,
    create_dataset_trial_journal,
    journal_path,
    manifest_path,
    open_dataset_trial_journal,
    persist_dataset_run_manifest,
    read_dataset_run_manifest,
)
from ul_cli.local_target_resolution import ResolvedLocalTarget, resolve_local_target
from ul_cli.pattern_identity import (
    ensure_project_pattern_identity_key,
    ensure_project_review_history_key,
)
from ul_cli.report import report_evidence

_DEFAULT_LIMIT = 10
_DEFAULT_REPETITIONS = 1
_DEFAULT_OPERATOR = "input.surface.typing_noise"
_MAXIMUM_DATASET_RECORDS = 100
_MAXIMUM_REPETITIONS = 100
_TARGET_TIMEOUT_SECONDS = 30.0
_PROJECT_DIRECTORY = ".ul"
_PROBE_CONFIG = "probe.json"
_PROBE_QUARANTINE_PREFIX = "probe-quarantine"
_DEFAULT_EVIDENCE = ".ul/runs/probe-evidence.jsonl"

type ProbeTargetConnection = JsonHttpEnvironmentConnection | LocalTargetConnection
type ProbeTargetConfig = JsonHttpTargetConfig | LocalTargetConfig
type ProbeStage = Literal[
    "observation import",
    "target load",
    "smoke invocation",
    "augmentation preparation",
    "probe execution",
    "evaluation",
    "analysis",
    "output",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProbeProjectConfig(_StrictModel):
    schema_version: Literal[3] = 3
    dataset: str = Field(min_length=1)
    target: str = Field(min_length=1)
    target_kind: Literal["python_callable", "command", "http"]
    target_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_confirmation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome_projection_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class _ArtifactIdentity(_StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _EnvironmentIdentity(_StrictModel):
    name: str
    value_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _TargetConfirmation(_StrictModel):
    schema_version: Literal[1] = 1
    kind: Literal["python_callable", "command", "http"]
    reference: str
    config_sha256: str
    executable: _ArtifactIdentity | None = None
    artifacts: tuple[_ArtifactIdentity, ...] = ()
    environment: tuple[_EnvironmentIdentity, ...] = ()
    callable: str | None = None


class _CampaignConfirmation(_StrictModel):
    schema_version: Literal[1] = 1
    target_confirmation_sha256: str
    semantic_provider_id: str
    semantic_provider_type: str
    semantic_endpoint_sha256: str
    semantic_settings_sha256: str
    campaign_plan_sha256: str
    case_limit: int = Field(ge=1, le=_MAXIMUM_DATASET_RECORDS)
    data_policy: dict[str, object]
    command_environment_api_requests: int
    semantic_model_calls: int
    maximum_completion_tokens: int
    maximum_wall_seconds: float
    monetary_cost_status: Literal["bounded", "unknown_unbounded"]
    maximum_cost_usd: float | None = None


class _ProbeSafetyState(_StrictModel):
    schema_version: Literal[1] = 1
    target_confirmation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["reusable", "quarantined"]
    reason_code: str = Field(min_length=1, max_length=100)


class _ProbeCheckpoint(_StrictModel):
    schema_version: Literal[1] = 1
    target_confirmation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    campaign_confirmation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    smoke_evidence: ExecutionEvidence
    smoke_elapsed_seconds: float = Field(ge=0)
    smoke_case_id: str = Field(min_length=1)
    smoke_turn_id: str = Field(min_length=1)
    smoke_request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class _ResolvedTarget:
    reference: str
    kind: Literal["python_callable", "command", "http"]
    config: ProbeTargetConfig
    config_sha256: str
    calls_per_execution: int
    maximum_executions: int
    maximum_active_target_seconds: float | None
    supports_state_observation: bool
    confirmation: _TargetConfirmation
    confirmation_sha256: str
    create_connection: Callable[[int, float | None], ProbeTargetConnection]
    revalidate_identity: Callable[[], None]


@dataclass(frozen=True)
class _SmokeResult:
    evidence: ExecutionEvidence
    elapsed_seconds: float
    case_id: str
    turn_id: str
    request_sha256: str


class ProbeFailure(RuntimeError):
    def __init__(
        self,
        stage: ProbeStage,
        reason_code: str,
        explanation: str,
        remediation: str,
        *,
        target_safe_to_reuse: bool,
    ) -> None:
        super().__init__(explanation)
        self.stage = stage
        self.reason_code = reason_code
        self.explanation = explanation
        self.remediation = remediation
        self.target_safe_to_reuse = target_safe_to_reuse


def _model_sha256(model: BaseModel) -> str:
    encoded = json.dumps(
        model.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _probe_checkpoint_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.probe-checkpoint.json")


def _probe_checkpoint_binding(
    *,
    records: tuple[InteractionRecord, ...],
    resolved_target: _ResolvedTarget,
    campaign_confirmation: _CampaignConfirmation,
    output: Path,
) -> tuple[str, str, str, str]:
    return (
        resolved_target.confirmation_sha256,
        _model_sha256(campaign_confirmation),
        _json_sha256([record.model_dump(mode="json") for record in records]),
        hashlib.sha256(str(output.resolve()).encode()).hexdigest(),
    )


def _persist_probe_checkpoint(
    path: Path,
    *,
    records: tuple[InteractionRecord, ...],
    resolved_target: _ResolvedTarget,
    campaign_confirmation: _CampaignConfirmation,
    output: Path,
    smoke_result: _SmokeResult,
) -> str:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _ensure_private_project_directory(path.parent)
    target_digest, campaign_digest, records_digest, output_digest = _probe_checkpoint_binding(
        records=records,
        resolved_target=resolved_target,
        campaign_confirmation=campaign_confirmation,
        output=output,
    )
    checkpoint = _ProbeCheckpoint(
        target_confirmation_sha256=target_digest,
        campaign_confirmation_sha256=campaign_digest,
        records_sha256=records_digest,
        output_sha256=output_digest,
        smoke_evidence=smoke_result.evidence,
        smoke_elapsed_seconds=smoke_result.elapsed_seconds,
        smoke_case_id=smoke_result.case_id,
        smoke_turn_id=smoke_result.turn_id,
        smoke_request_sha256=smoke_result.request_sha256,
    )
    encoded = (checkpoint.model_dump_json() + "\n").encode()
    with create_private_output(path) as stream:
        stream.write(encoded.decode())
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_probe_project_directory(path.parent)
    return hashlib.sha256(encoded).hexdigest()


def _remove_probe_checkpoint(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_probe_project_directory(path.parent)


def _load_probe_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    records: tuple[InteractionRecord, ...],
    resolved_target: _ResolvedTarget,
    campaign_confirmation: _CampaignConfirmation,
    output: Path,
) -> _SmokeResult:
    descriptor = open_resume_descriptor(path, writable=False)
    try:
        status = os.fstat(descriptor)
        if (
            status.st_nlink != 1
            or status.st_size > 10_000_000
            or (sys.platform != "win32" and stat.S_IMODE(status.st_mode) & 0o077)
            or (hasattr(os, "getuid") and status.st_uid != os.getuid())
        ):
            raise ValueError("probe checkpoint must be a private owner-only file")
        encoded = os.read(descriptor, 10_000_001)
    finally:
        os.close(descriptor)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("probe checkpoint digest is invalid")
    if hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise ValueError("probe checkpoint integrity check failed")
    checkpoint = _ProbeCheckpoint.model_validate_json(encoded)
    expected_binding = _probe_checkpoint_binding(
        records=records,
        resolved_target=resolved_target,
        campaign_confirmation=campaign_confirmation,
        output=output,
    )
    recorded_binding = (
        checkpoint.target_confirmation_sha256,
        checkpoint.campaign_confirmation_sha256,
        checkpoint.records_sha256,
        checkpoint.output_sha256,
    )
    if recorded_binding != expected_binding:
        raise ValueError("probe checkpoint does not match this target, plan, data, and output")
    expected_case = _smoke_case(records[0], resolved_target)
    expected_request_sha256 = _json_sha256(
        {
            "case_id": expected_case.id,
            "turn": expected_case.turns[0].model_dump(mode="json"),
            "probe_context": expected_case.probe_context,
        }
    )
    if (
        checkpoint.smoke_case_id != expected_case.id
        or checkpoint.smoke_turn_id != expected_case.turns[0].id
        or checkpoint.smoke_request_sha256 != expected_request_sha256
        or checkpoint.smoke_evidence.lifecycle.terminal_status != "succeeded"
    ):
        raise ValueError("probe checkpoint smoke identity or lifecycle is invalid")
    validation_connection = resolved_target.create_connection(
        resolved_target.calls_per_execution,
        resolved_target.maximum_active_target_seconds,
    )
    try:
        validate_execution_evidence(
            expected_case,
            validation_connection,
            checkpoint.smoke_evidence,
        )
    finally:
        asyncio.run(validation_connection.aclose())
    return _SmokeResult(
        evidence=checkpoint.smoke_evidence,
        elapsed_seconds=checkpoint.smoke_elapsed_seconds,
        case_id=checkpoint.smoke_case_id,
        turn_id=checkpoint.smoke_turn_id,
        request_sha256=checkpoint.smoke_request_sha256,
    )


def _probe_quarantine_path(resolved_target: _ResolvedTarget) -> Path:
    return (
        Path.cwd()
        / _PROJECT_DIRECTORY
        / (f"{_PROBE_QUARANTINE_PREFIX}-{resolved_target.confirmation_sha256[:16]}.json")
    )


def _probe_target_lock_path(resolved_target: _ResolvedTarget) -> Path:
    return _probe_quarantine_path(resolved_target).with_suffix(".lock")


def _open_probe_target_lock(resolved_target: _ResolvedTarget) -> tuple[int, bool]:
    project_directory = Path.cwd() / _PROJECT_DIRECTORY
    _ensure_private_project_directory(project_directory)
    path = _probe_target_lock_path(resolved_target)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    created = False
    while True:
        try:
            path_status = os.lstat(path)
        except FileNotFoundError:
            try:
                descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | no_follow, 0o600)
                created = True
                path_status = os.lstat(path)
                _fsync_probe_project_directory(project_directory)
                break
            except FileExistsError:
                continue
        if not stat.S_ISREG(path_status.st_mode):
            raise OSError("probe target lock must be a regular private file")
        descriptor = os.open(path, os.O_RDWR | no_follow)
        break
    try:
        descriptor_status = os.fstat(descriptor)
        if (
            not os.path.samestat(path_status, descriptor_status)
            or not stat.S_ISREG(descriptor_status.st_mode)
            or descriptor_status.st_nlink != 1
            or (sys.platform != "win32" and stat.S_IMODE(descriptor_status.st_mode) & 0o077)
            or (hasattr(os, "getuid") and descriptor_status.st_uid != os.getuid())
        ):
            raise OSError("probe target lock must be a regular private owner-only file")
        if sys.platform == "win32" and descriptor_status.st_size == 0:
            os.write(descriptor, b"0")
            os.fsync(descriptor)
        if sys.platform == "win32":
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        return descriptor, created
    except BaseException:
        os.close(descriptor)
        raise


def _close_probe_target_lock(descriptor: int) -> None:
    try:
        if sys.platform == "win32":
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _fsync_probe_project_directory(project_directory: Path) -> None:
    if sys.platform == "win32":
        return
    descriptor = os.open(
        project_directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_probe_safety_state(path: Path) -> _ProbeSafetyState:
    path_status = os.lstat(path)
    if (
        not stat.S_ISREG(path_status.st_mode)
        or path_status.st_size > 10_000
        or path_status.st_nlink != 1
        or (sys.platform != "win32" and stat.S_IMODE(path_status.st_mode) & 0o077)
        or (hasattr(os, "getuid") and path_status.st_uid != os.getuid())
    ):
        raise ValueError("probe safety state must be a regular private owner-only file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor_status = os.fstat(descriptor)
        if (
            not os.path.samestat(path_status, descriptor_status)
            or not stat.S_ISREG(descriptor_status.st_mode)
            or descriptor_status.st_nlink != 1
        ):
            raise ValueError("probe safety state changed while opening")
        raw = os.read(descriptor, 10_001)
    finally:
        os.close(descriptor)
    if len(raw) > 10_000:
        raise ValueError("probe safety state exceeds its size limit")
    return _ProbeSafetyState.model_validate_json(raw)


def _persist_probe_safety_state(
    resolved_target: _ResolvedTarget,
    *,
    status: Literal["reusable", "quarantined"],
    reason_code: str,
) -> None:
    project_directory = Path.cwd() / _PROJECT_DIRECTORY
    _ensure_private_project_directory(project_directory)
    path = _probe_quarantine_path(resolved_target)
    temporary_path = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    state = _ProbeSafetyState(
        target_confirmation_sha256=resolved_target.confirmation_sha256,
        status=status,
        reason_code=reason_code,
    )
    descriptor = os.open(
        temporary_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        encoded = (state.model_dump_json() + "\n").encode()
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary_path, path)
        _fsync_probe_project_directory(project_directory)
    except BaseException:
        with suppress(OSError):
            temporary_path.unlink()
        raise


def _enforce_probe_quarantine(
    resolved_target: _ResolvedTarget,
    resolution: Literal["environment-reset", "environment-replacement"] | None,
    *,
    lock_created: bool,
) -> None:
    path = _probe_quarantine_path(resolved_target)
    try:
        state = _load_probe_safety_state(path)
    except FileNotFoundError:
        if lock_created:
            _persist_probe_safety_state(
                resolved_target,
                status="reusable",
                reason_code="target_safety_initialized",
            )
            return
        if resolution is not None:
            _persist_probe_safety_state(
                resolved_target,
                status="reusable",
                reason_code=f"operator_attested_{resolution.replace('-', '_')}",
            )
            return
        raise ProbeFailure(
            "target load",
            "PROBE_QUARANTINE_RECEIPT_MISSING",
            "The durable target safety state is missing after a prior probe attempt.",
            "After an operator resets or replaces the test environment, rerun with the matching "
            "--resolve-quarantine-after attestation.",
            target_safe_to_reuse=False,
        ) from None
    except (OSError, ValidationError, ValueError):
        if resolution is not None:
            _persist_probe_safety_state(
                resolved_target,
                status="reusable",
                reason_code=f"operator_attested_{resolution.replace('-', '_')}",
            )
            return
        raise ProbeFailure(
            "target load",
            "PROBE_QUARANTINE_RECEIPT_INVALID",
            "The bound target safety state is invalid or unreadable.",
            "After an operator resets or replaces the test environment, restore safety with the "
            "matching --resolve-quarantine-after attestation.",
            target_safe_to_reuse=False,
        ) from None
    if state.target_confirmation_sha256 != resolved_target.confirmation_sha256:
        raise ProbeFailure(
            "target load",
            "PROBE_QUARANTINE_RECEIPT_MISMATCH",
            "The target safety state does not match the resolved target identity.",
            "Do not call the target; restore its matching private safety state.",
            target_safe_to_reuse=False,
        )
    if state.status == "reusable":
        return
    if resolution is None:
        raise ProbeFailure(
            "target load",
            "PROBE_TARGET_QUARANTINED",
            "A previous probe may have reached the target with uncertain delivery.",
            "After an operator resets or replaces the test environment, rerun with "
            "--resolve-quarantine-after environment-reset (or environment-replacement).",
            target_safe_to_reuse=False,
        )
    _persist_probe_safety_state(
        resolved_target,
        status="reusable",
        reason_code=f"operator_attested_{resolution.replace('-', '_')}",
    )
    console.print(
        "Recorded target quarantine cleared after operator attestation; durable safety state "
        "was updated."
    )


def _persist_probe_quarantine(
    resolved_target: _ResolvedTarget,
    reason_code: str,
) -> None:
    _persist_probe_safety_state(
        resolved_target,
        status="quarantined",
        reason_code=reason_code,
    )


def _semantic_settings_snapshot(
    settings: DatasetSemanticSettings,
) -> DatasetEvidenceSemanticSettings:
    return DatasetEvidenceSemanticSettings(
        provider=settings.semantic_provider_id,
        endpoint_sha256=settings.semantic_endpoint_sha256,
        model=settings.model,
        render_model=settings.render_model,
        equivalence_model=settings.equivalence_model,
        max_input_chars=settings.max_input_chars,
        max_output_tokens=settings.max_output_tokens,
        max_render_tokens=settings.max_render_tokens,
        max_response_bytes=settings.max_response_bytes,
        timeout_seconds=settings.timeout_seconds,
    )


def _target_evidence_receipt(resolved_target: _ResolvedTarget) -> dict[str, JsonValue]:
    confirmation = resolved_target.confirmation
    outcome_projection = _outcome_projection(resolved_target)
    return {
        "kind": confirmation.kind,
        "config_sha256": confirmation.config_sha256,
        "confirmation_sha256": resolved_target.confirmation_sha256,
        "supports_state_observation": resolved_target.supports_state_observation,
        "executable_sha256": (
            confirmation.executable.sha256 if confirmation.executable is not None else None
        ),
        "artifact_sha256": [artifact.sha256 for artifact in confirmation.artifacts],
        "environment": [item.model_dump(mode="json") for item in confirmation.environment],
        "callable": confirmation.callable,
        "outcome_projection": (
            outcome_projection.model_dump(mode="json") if outcome_projection is not None else None
        ),
        "outcome_projection_sha256": (
            outcome_projection.digest if outcome_projection is not None else None
        ),
    }


def _outcome_projection(resolved_target: _ResolvedTarget) -> OutcomeProjection | None:
    return resolved_target.config.outcome


def probe(
    data: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Interaction JSONL."),
    ],
    target: Annotated[
        str,
        typer.Option(
            help=("HTTP(S) URL, Python module:callable, or local/HTTP target configuration JSON.")
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(help="New normal UL evidence JSONL file."),
    ] = Path(_DEFAULT_EVIDENCE),
    operator: Annotated[
        list[str] | None,
        typer.Option(
            "--operator",
            help=(
                "Available dataset augmentation ID; repeat as needed. Run "
                "'ul augmentations list --mode dataset_variation' for values."
            ),
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(min=1, max=_MAXIMUM_DATASET_RECORDS, help="Interactions to evaluate."),
    ] = _DEFAULT_LIMIT,
    repetitions: Annotated[
        int,
        typer.Option(
            min=1,
            max=_MAXIMUM_REPETITIONS,
            help="Target executions per original input and accepted variation.",
        ),
    ] = _DEFAULT_REPETITIONS,
    target_artifact: Annotated[
        list[Path] | None,
        typer.Option(help="Additional command worker artifact to hash and bind; repeat as needed."),
    ] = None,
    http_preset: Annotated[
        Literal["generic-json", "openai-chat"] | None,
        typer.Option(help="Request/response shape for an HTTP URL; defaults to generic-json."),
    ] = None,
    request_json_template: Annotated[
        str | None,
        typer.Option(help="JSON containing one {{input}} value; overrides the direct HTTP preset."),
    ] = None,
    response_json_pointer: Annotated[
        str | None,
        typer.Option(help="RFC 6901 pointer to the direct HTTP response value."),
    ] = None,
    agent_model: Annotated[
        str | None,
        typer.Option(help="Model sent by the direct HTTP openai-chat preset."),
    ] = None,
    header_from_env: Annotated[
        list[str] | None,
        typer.Option(
            "--header-from-env",
            help="HTTP_HEADER=UL_ENVIRONMENT_VARIABLE; repeat for credentials or routing.",
        ),
    ] = None,
    confirm_target: Annotated[
        str | None,
        typer.Option(
            "--confirm-target",
            help="Non-interactively confirm this exact displayed target digest.",
        ),
    ] = None,
    confirm_paid_execution: Annotated[
        str | None,
        typer.Option(
            "--confirm-paid-execution",
            help="Non-interactively authorize this exact displayed campaign digest.",
        ),
    ] = None,
    allow_insecure_http: Annotated[
        bool,
        typer.Option(help="Allow a local plain-HTTP test target."),
    ] = False,
    diagnostic_artifact: Annotated[
        Path | None,
        typer.Option(
            help="Opt in to a private JSON diagnostic that may contain customer input/output."
        ),
    ] = None,
    show_smoke_response: Annotated[
        bool,
        typer.Option(help="Print private smoke response content instead of only its digest."),
    ] = False,
    progress_json: Annotated[
        bool,
        typer.Option(
            "--progress-json",
            help="Write stable privacy-safe progress JSON Lines to stderr.",
        ),
    ] = False,
    resolve_quarantine_after: Annotated[
        Literal["environment-reset", "environment-replacement"] | None,
        typer.Option(
            "--resolve-quarantine-after",
            help="Attest that an operator reset or replaced a quarantined test environment.",
        ),
    ] = None,
    resume_checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--resume-checkpoint",
            exists=True,
            dir_okay=False,
            readable=True,
            hidden=True,
        ),
    ] = None,
    resume_checkpoint_sha256: Annotated[
        str | None,
        typer.Option("--resume-checkpoint-sha256", hidden=True),
    ] = None,
) -> None:
    """Smoke a real target, then optionally run one bounded active-probe campaign."""
    progress_runtime: CampaignProgressRuntime | None = None
    resolved_target: _ResolvedTarget | None = None
    target_lock_descriptor: int | None = None
    try:
        records = _load_campaign_records(data, limit=limit)
        try:
            selected_operator_ids = validate_operator_ids(operator or [_DEFAULT_OPERATOR])
        except DatasetInputError as error:
            raise typer.BadParameter(str(error), param_hint="--operator") from None
        resolved_target = _resolve_target(
            target,
            allow_insecure_http=allow_insecure_http,
            explicit_artifacts=tuple(target_artifact or ()),
            http_preset=http_preset,
            request_json_template=request_json_template,
            response_json_pointer=response_json_pointer,
            agent_model=agent_model,
            header_from_env=header_from_env,
        )
        existing_config = _check_probe_config_binding(data, resolved_target)
        try:
            os.lstat(_probe_target_lock_path(resolved_target))
            target_lock_preexists = True
        except FileNotFoundError:
            target_lock_preexists = False
        if target_lock_preexists:
            try:
                target_lock_descriptor, lock_created = _open_probe_target_lock(resolved_target)
            except OSError:
                raise ProbeFailure(
                    "target load",
                    "PROBE_TARGET_LOCK_UNAVAILABLE",
                    "The private target safety lock could not be acquired.",
                    "Fix the private .ul directory before making any target call.",
                    target_safe_to_reuse=True,
                ) from None
            _enforce_probe_quarantine(
                resolved_target,
                resolve_quarantine_after,
                lock_created=lock_created,
            )
        _confirm_target(resolved_target, confirmed_digest=confirm_target)
        if target_lock_descriptor is None:
            try:
                target_lock_descriptor, lock_created = _open_probe_target_lock(resolved_target)
            except OSError:
                raise ProbeFailure(
                    "target load",
                    "PROBE_TARGET_LOCK_UNAVAILABLE",
                    "The private target safety lock could not be acquired.",
                    "Fix the private .ul directory before making any target call.",
                    target_safe_to_reuse=True,
                ) from None
            _enforce_probe_quarantine(
                resolved_target,
                resolve_quarantine_after,
                lock_created=lock_created,
            )
        try:
            settings = load_dataset_semantic_settings()
            validate_model_input_bounds(records, settings.max_input_chars)
            plan = create_dataset_campaign_plan(
                records=records,
                selected_operator_ids=selected_operator_ids,
                repetitions=repetitions,
                target_calls_per_execution=resolved_target.calls_per_execution,
                settings=settings,
            )
            campaign_confirmation = _campaign_confirmation(
                plan, settings, resolved_target, case_limit=limit
            )
            checkpoint_path = _probe_checkpoint_path(output)
            target_path = Path(target)
            action_target = str(target_path.resolve()) if target_path.is_file() else target
            resume_argv = [
                "ul",
                "probe",
                str(data.resolve()),
                "--target",
                action_target,
                "--output",
                str(output.resolve()),
                "--confirm-target",
                resolved_target.confirmation_sha256,
                "--confirm-paid-execution",
                _model_sha256(campaign_confirmation),
                "--resume-checkpoint",
                str(checkpoint_path.resolve()),
            ]
            for artifact in target_artifact or ():
                resume_argv.extend(("--target-artifact", str(artifact.resolve())))
            for operator_reference in selected_operator_ids:
                resume_argv.extend(("--operator", operator_reference))
            resume_argv.extend(("--limit", str(limit), "--repetitions", str(repetitions)))
            if allow_insecure_http:
                resume_argv.append("--allow-insecure-http")
            if http_preset is not None:
                resume_argv.extend(("--http-preset", http_preset))
            if request_json_template is not None:
                resume_argv.extend(("--request-json-template", request_json_template))
            if response_json_pointer is not None:
                resume_argv.extend(("--response-json-pointer", response_json_pointer))
            if agent_model is not None:
                resume_argv.extend(("--agent-model", agent_model))
            for mapping in header_from_env or ():
                resume_argv.extend(("--header-from-env", mapping))
            if diagnostic_artifact is not None:
                resume_argv.extend(("--diagnostic-artifact", str(diagnostic_artifact.resolve())))
            if show_smoke_response:
                resume_argv.append("--show-smoke-response")
            if progress_json:
                resume_argv.append("--progress-json")
            run_context = create_dataset_evidence_run_context(
                selected_records=records,
                operators=tuple(
                    dataset_operator_identity(reference) for reference in selected_operator_ids
                ),
                repetitions=repetitions,
                invariant_suite_sha256=None,
                target_receipt=_target_evidence_receipt(resolved_target),
                semantic_settings=_semantic_settings_snapshot(settings),
            )
            progress_runtime = create_campaign_progress_runtime(
                case_count=len(records),
                work_upper_bound=(len(records) * repetitions * (1 + len(selected_operator_ids))),
                target_call_budget=(
                    resolved_target.calls_per_execution + plan.calls.total_environment_api
                ),
                semantic_call_budget=plan.calls.total_semantic_model,
                environment_call_budget=(
                    resolved_target.calls_per_execution + plan.calls.total_environment_api
                ),
                token_budget=plan.tokens.maximum,
                maximum_wall_time_seconds=campaign_confirmation.maximum_wall_seconds,
                next_commands=create_probe_next_commands(evidence_path=output),
                json_output=progress_json,
            )
        except ProbeFailure:
            raise
        except Exception:
            raise ProbeFailure(
                "augmentation preparation",
                "PROBE_AUGMENTATION_PREPARATION_FAILED",
                "The bounded campaign could not be prepared safely.",
                "Review semantic settings and target limits, then rerun.",
                target_safe_to_reuse=True,
            ) from None
        progress_runtime.tracker.record_usage(
            target_calls=1,
            semantic_calls=None,
            environment_calls=resolved_target.calls_per_execution,
            tokens=None,
        )
        progress_runtime.tracker.emit(status="running", stage="smoke")
        evaluator_preflight: EvaluatorModelPreflight | None = None
        if resume_checkpoint is not None:
            if resume_checkpoint_sha256 is None:
                raise ProbeFailure(
                    "target load",
                    "PROBE_CHECKPOINT_DIGEST_MISSING",
                    "The private probe checkpoint digest is required for resume.",
                    "Use only the opaque resume action emitted by the paused campaign.",
                    target_safe_to_reuse=True,
                )
            try:
                smoke_result = _load_probe_checkpoint(
                    resume_checkpoint,
                    expected_sha256=resume_checkpoint_sha256,
                    records=records,
                    resolved_target=resolved_target,
                    campaign_confirmation=campaign_confirmation,
                    output=output,
                )
            except (OSError, ValidationError, ValueError):
                raise ProbeFailure(
                    "target load",
                    "PROBE_CHECKPOINT_INVALID",
                    "The private bound smoke checkpoint is invalid.",
                    "Keep the target stopped and inspect the private smoke checkpoint.",
                    target_safe_to_reuse=True,
                ) from None
            try:
                evaluator_preflight, _ = asyncio.run(load_evaluator_preflight(output, settings))
            except (OSError, ValidationError, ValueError):
                raise ProbeFailure(
                    "target load",
                    "PROBE_PREFLIGHT_CHECKPOINT_INVALID",
                    "The private bound evaluator preflight checkpoint is invalid.",
                    "Keep the target stopped and inspect the private preflight checkpoint.",
                    target_safe_to_reuse=True,
                ) from None
            console.print("Reusing durable smoke and evaluator preflight checkpoints.")
            checkpoint_sha256 = resume_checkpoint_sha256
        else:
            if resume_checkpoint_sha256 is not None:
                raise ProbeFailure(
                    "target load",
                    "PROBE_CHECKPOINT_PATH_MISSING",
                    "A probe checkpoint digest cannot be used without its private checkpoint.",
                    "Use only the opaque resume action emitted by the paused campaign.",
                    target_safe_to_reuse=True,
                )
            with progress_runtime.signal_control.installed():
                smoke_result = asyncio.run(
                    _run_smoke(
                        records[0],
                        resolved_target,
                        signal_control=progress_runtime.signal_control,
                    )
                )
            _print_smoke(
                smoke_result,
                resolved_target,
                show_response=show_smoke_response,
            )
            _save_probe_config(
                data=data,
                resolved_target=resolved_target,
                existing_config=existing_config,
            )
            try:
                checkpoint_sha256 = _persist_probe_checkpoint(
                    checkpoint_path,
                    records=records,
                    resolved_target=resolved_target,
                    campaign_confirmation=campaign_confirmation,
                    output=output,
                    smoke_result=smoke_result,
                )
            except (OSError, ValueError):
                raise ProbeFailure(
                    "output",
                    "PROBE_CHECKPOINT_WRITE_FAILED",
                    "The successful smoke checkpoint could not be persisted durably.",
                    "Do not continue to paid campaign work; fix private output storage first.",
                    target_safe_to_reuse=True,
                ) from None
        resume_argv.extend(("--resume-checkpoint-sha256", checkpoint_sha256))
        progress_runtime.tracker.replace_next_commands(
            create_probe_next_commands(
                evidence_path=output,
                resume_argv=tuple(resume_argv),
            )
        )
        remaining_target_seconds = _validate_campaign_target_budget(
            plan, resolved_target, smoke_result.elapsed_seconds
        )
        _print_campaign_budget(
            plan,
            campaign_confirmation,
            selected_operator_ids=selected_operator_ids,
            limit=limit,
        )
        if not _confirm_paid_execution(
            confirmation=campaign_confirmation,
            confirmed_digest=confirm_paid_execution,
        ):
            console.print("Stopped after smoke. No semantic-model calls were made.")
            _remove_probe_checkpoint(checkpoint_path)
            progress_runtime.tracker.emit(status="cancelled", stage="terminal")
            return
        _validate_paid_execution_settings(settings)
        results = _run_campaign(
            records=records,
            selected_operator_ids=selected_operator_ids,
            resolved_target=resolved_target,
            settings=settings,
            plan=plan,
            output=output,
            repetitions=repetitions,
            remaining_target_seconds=remaining_target_seconds,
            run_context=run_context,
            progress_runtime=progress_runtime,
            evaluator_preflight=evaluator_preflight,
            resume_campaign=resume_checkpoint is not None,
            allow_insecure_http=allow_insecure_http,
            resolve_quarantine_after=resolve_quarantine_after,
        )
        projection_failure = _campaign_projection_failure(results, output)
        if projection_failure is not None:
            raise projection_failure
        try:
            progress_runtime.tracker.emit(status="running", stage="report")
            print_dataset_results(results, output, show_report_guidance=False)
        except Exception:
            raise ProbeFailure(
                "evaluation",
                "PROBE_RESULT_PRESENTATION_FAILED",
                "The campaign evidence was written, but its result summary could not be shown.",
                "Use the saved evidence file with `ul report` after fixing terminal output.",
                target_safe_to_reuse=True,
            ) from None
        try:
            _print_stronger_run(
                data,
                target,
                output,
                allow_insecure_http=allow_insecure_http,
                target_artifacts=tuple(target_artifact or ()),
                http_preset=http_preset,
                request_json_template=request_json_template,
                response_json_pointer=response_json_pointer,
                agent_model=agent_model,
                header_from_env=tuple(header_from_env or ()),
                selected_operator_ids=selected_operator_ids,
                limit=limit,
                repetitions=repetitions,
            )
            console.print("")
            report_evidence(output)
            progress_runtime.tracker.emit(status="completed", stage="terminal")
        except Exception:
            raise ProbeFailure(
                "analysis",
                "PROBE_REPORT_FAILED",
                "The campaign evidence was written, but the report could not be produced.",
                "Run `ul report` against the saved private evidence file.",
                target_safe_to_reuse=True,
            ) from None
    except ProbeFailure as failure:
        if not failure.target_safe_to_reuse and resolved_target is not None:
            try:
                _persist_probe_quarantine(resolved_target, failure.reason_code)
            except (OSError, ValueError):
                failure = ProbeFailure(
                    "output",
                    "PROBE_QUARANTINE_PERSIST_FAILED",
                    "UL could not persist the target quarantine receipt after an unsafe stop.",
                    "Do not call this target; fix the private .ul directory before retrying.",
                    target_safe_to_reuse=False,
                )
        if progress_runtime is not None and not progress_runtime.tracker.terminal_emitted:
            progress_runtime.tracker.emit(
                status="failed",
                stage="terminal",
                environment="reusable" if failure.target_safe_to_reuse else "quarantined",
            )
        _print_failure(failure, diagnostic_artifact=diagnostic_artifact)
        exit_code = (
            130
            if failure.reason_code
            in {"PROBE_PAUSED_AFTER_PREFLIGHT", "PROBE_PAUSED_DURING_CAMPAIGN"}
            else 2
        )
        raise typer.Exit(code=exit_code) from None
    finally:
        if target_lock_descriptor is not None:
            _close_probe_target_lock(target_lock_descriptor)


def _load_campaign_records(
    data: Path, *, limit: int = _DEFAULT_LIMIT
) -> tuple[InteractionRecord, ...]:
    try:
        return load_interaction_records(data)[:limit]
    except (DatasetInputError, ValidationError, ValueError) as error:
        raise ProbeFailure(
            "observation import",
            "PROBE_OBSERVATION_INVALID",
            str(error),
            "Fix the JSONL record named above and run the same command again.",
            target_safe_to_reuse=True,
        ) from None


def _resolve_target(
    target: str,
    *,
    allow_insecure_http: bool,
    explicit_artifacts: tuple[Path, ...] = (),
    http_preset: Literal["generic-json", "openai-chat"] | None = None,
    request_json_template: str | None = None,
    response_json_pointer: str | None = None,
    agent_model: str | None = None,
    header_from_env: list[str] | None = None,
) -> _ResolvedTarget:
    try:
        direct_http_options_used = (
            http_preset is not None
            or request_json_template is not None
            or response_json_pointer is not None
            or agent_model is not None
            or bool(header_from_env)
        )
        if target.casefold().startswith(("https://", "http://")):
            if explicit_artifacts:
                raise ValueError("--target-artifact applies only to local targets")
            http_config = create_isolated_response_target_config(
                target,
                isolated_preset=http_preset or "generic-json",
                environment_id="probe-http-" + hashlib.sha256(target.encode()).hexdigest()[:16],
                request_json_template=request_json_template,
                response_json_pointer=response_json_pointer,
                agent_model=agent_model,
                header_from_env=header_from_env,
            )
            return _http_target(target, http_config, allow_insecure_http=allow_insecure_http)
        if direct_http_options_used:
            raise ValueError("direct HTTP mapping options require an HTTP URL target")
        target_path = Path(target)
        if target_path.is_file():
            return _resolve_configured_target(
                target_path,
                allow_insecure_http=allow_insecure_http,
                explicit_artifacts=explicit_artifacts,
            )
        if allow_insecure_http:
            raise ValueError("--allow-insecure-http requires an HTTP URL or config target")
        return _local_target(resolve_local_target(target, explicit_artifacts=explicit_artifacts))
    except (OSError, RuntimeError, ValidationError, ValueError) as error:
        raise ProbeFailure(
            "target load",
            "PROBE_TARGET_INVALID",
            str(error),
            "Use an HTTP(S) URL, importable module:callable, or validate the target config with "
            "its advanced check.",
            target_safe_to_reuse=True,
        ) from None


def _resolve_configured_target(
    path: Path,
    *,
    allow_insecure_http: bool,
    explicit_artifacts: tuple[Path, ...],
) -> _ResolvedTarget:
    try:
        local_target = resolve_local_target(str(path), explicit_artifacts=explicit_artifacts)
    except ValueError:
        http_config = load_json_http_environment_config(path)
        validate_json_http_environment_configuration(
            http_config,
            test_environment_confirmed=True,
            allow_insecure_http=allow_insecure_http,
        )
        return _http_target(str(path), http_config, allow_insecure_http=allow_insecure_http)
    if allow_insecure_http:
        raise ValueError("--allow-insecure-http requires an HTTP URL or config target")
    return _local_target(local_target)


def _http_target(
    reference: str,
    config: JsonHttpTargetConfig,
    *,
    allow_insecure_http: bool,
) -> _ResolvedTarget:
    validate_json_http_environment_configuration(
        config,
        test_environment_confirmed=True,
        allow_insecure_http=allow_insecure_http,
    )
    digest = json_http_environment_config_sha256(config)
    confirmation = _TargetConfirmation(kind="http", reference=reference, config_sha256=digest)
    return _ResolvedTarget(
        reference=reference,
        kind="http",
        config=config,
        config_sha256=digest,
        calls_per_execution=json_http_environment_calls_per_execution(config),
        maximum_executions=1_000,
        maximum_active_target_seconds=None,
        supports_state_observation=json_http_environment_capabilities(
            config
        ).supports_state_observation,
        confirmation=confirmation,
        confirmation_sha256=_model_sha256(confirmation),
        create_connection=lambda maximum_calls, maximum_seconds: (
            JsonHttpEnvironmentConnection.from_config(
                config,
                test_environment_confirmed=True,
                allow_insecure_http=allow_insecure_http,
                max_environment_api_calls=maximum_calls,
            )
        ),
        revalidate_identity=lambda: None,
    )


def _local_target(
    local_target: ResolvedLocalTarget,
) -> _ResolvedTarget:
    local_confirmation = local_target.confirmation
    confirmation = _TargetConfirmation(
        kind=local_confirmation.kind,
        reference=local_confirmation.reference,
        config_sha256=local_confirmation.config_sha256,
        executable=_ArtifactIdentity(
            path=local_confirmation.executable.path,
            sha256=local_confirmation.executable.sha256,
        ),
        artifacts=tuple(
            _ArtifactIdentity(path=artifact.path, sha256=artifact.sha256)
            for artifact in local_confirmation.artifacts
        ),
        environment=tuple(
            _EnvironmentIdentity(
                name=environment.name,
                value_sha256=environment.value_sha256,
            )
            for environment in local_confirmation.environment
        ),
        callable=local_confirmation.callable,
    )

    def revalidate_identity() -> None:
        try:
            local_target.revalidate_identity()
        except ValueError:
            raise ProbeFailure(
                "target load",
                "PROBE_TARGET_IDENTITY_CHANGED",
                "A confirmed executable or target artifact changed before launch.",
                "Review the new target identity and rerun the command.",
                target_safe_to_reuse=True,
            ) from None

    return _ResolvedTarget(
        reference=local_target.reference,
        kind=local_target.kind,
        config=local_target.config,
        config_sha256=local_target.config_sha256,
        calls_per_execution=1,
        maximum_executions=local_target.maximum_executions,
        maximum_active_target_seconds=local_target.maximum_active_target_seconds,
        supports_state_observation=False,
        confirmation=confirmation,
        confirmation_sha256=_model_sha256(confirmation),
        create_connection=local_target.create_connection,
        revalidate_identity=revalidate_identity,
    )


def _confirm_target(resolved_target: _ResolvedTarget, *, confirmed_digest: str | None) -> None:
    console.print("UL active-probe target")
    console.print(f"  Kind: {resolved_target.kind}")
    console.print(f"  Config sha256: {resolved_target.config_sha256}")
    console.print(f"  Confirmation sha256: {resolved_target.confirmation_sha256}")
    if resolved_target.confirmation.executable is not None:
        executable = resolved_target.confirmation.executable
        console.print(f"  Executable: {executable.path} ({executable.sha256})")
    for artifact in resolved_target.confirmation.artifacts:
        console.print(f"  Artifact: {artifact.path} ({artifact.sha256})")
    for environment in resolved_target.confirmation.environment:
        console.print(f"  Environment: {environment.name} value sha256 {environment.value_sha256}")
    if resolved_target.kind == "http":
        http_config = cast(JsonHttpTargetConfig, resolved_target.config)
        console.print(f"  Endpoint: {json_http_environment_config_urls(http_config)[0]}")
        if not resolved_target.supports_state_observation:
            console.print("Every request must start from isolated test state.")
    console.print("Use only a dedicated test target that cannot cause real-world effects.")
    if confirmed_digest is not None:
        if confirmed_digest != resolved_target.confirmation_sha256:
            raise ProbeFailure(
                "target load",
                "PROBE_TARGET_CONFIRMATION_CHANGED",
                "The supplied target confirmation digest does not match the current target.",
                "Review the newly displayed identity and confirm its exact digest.",
                target_safe_to_reuse=True,
            )
        return
    if not typer.confirm(
        "Trust this exact target digest, attest the conditions above, and continue with one "
        "smoke call?"
    ):
        raise ProbeFailure(
            "target load",
            "PROBE_TARGET_NOT_CONFIRMED",
            "Target safety was not confirmed; UL made no target or semantic-model calls.",
            "Review the target and rerun when its displayed digest is trusted.",
            target_safe_to_reuse=True,
        )


def _smoke_case(
    record: InteractionRecord,
    resolved_target: _ResolvedTarget,
) -> EvaluationCase:
    return EvaluationCase(
        id=f"{record.id}:smoke",
        turns=(
            ConversationTurn(
                id=f"{record.id}:smoke:turn-1",
                role=ConversationRole.USER,
                content=record.raw_input,
            ),
        ),
        max_environment_api_calls=resolved_target.calls_per_execution,
        timeout_seconds=_TARGET_TIMEOUT_SECONDS,
        probe_context=record.probe_context(record.raw_input),
    )


async def _run_smoke(
    record: InteractionRecord,
    resolved_target: _ResolvedTarget,
    *,
    signal_control: CampaignSignalControl | None = None,
) -> _SmokeResult:
    resolved_target.revalidate_identity()
    connection = resolved_target.create_connection(
        resolved_target.calls_per_execution,
        resolved_target.maximum_active_target_seconds,
    )
    case = _smoke_case(record, resolved_target)
    try:
        started_at = time.monotonic()
        async with connection:
            task = asyncio.current_task()
            if signal_control is not None and task is not None:
                signal_control.target_call_started(task)
            evidence = await connection.execute(case)
            if signal_control is not None:
                signal_control.target_call_finished()
        elapsed_seconds = time.monotonic() - started_at
    except OutcomeProjectionExecutionError as error:
        raise ProbeFailure(
            "smoke invocation",
            "PROBE_OUTCOME_PROJECTION_INVALID",
            str(error),
            _projection_failure_remediation(error, error.target_safe_to_reuse),
            target_safe_to_reuse=error.target_safe_to_reuse,
        ) from None
    except asyncio.CancelledError:
        if signal_control is not None:
            signal_control.target_call_finished()
        raise ProbeFailure(
            "smoke invocation",
            "PROBE_SMOKE_DELIVERY_UNCERTAIN",
            "Smoke delivery may have begun before interruption; UL did not retry it.",
            "Quarantine the target until an operator resets or replaces the test environment.",
            target_safe_to_reuse=False,
        ) from None
    except (OSError, RuntimeError, TimeoutError, ValueError, httpx.HTTPError) as error:
        raise ProbeFailure(
            "smoke invocation",
            "PROBE_SMOKE_FAILED",
            "The target did not complete the original smoke interaction.",
            "Run the target's advanced environment/local dry-run check, then retry.",
            target_safe_to_reuse=False,
        ) from error
    try:
        validate_execution_evidence(case, connection, evidence)
    except OutcomeProjectionError as error:
        target_safe_to_reuse = (
            evidence.evidence_scope == "response_and_state"
            and not evidence.lifecycle.environment_state_uncertain
            and evidence.lifecycle.cleanup == "succeeded"
        )
        raise ProbeFailure(
            "smoke invocation",
            "PROBE_OUTCOME_PROJECTION_INVALID",
            str(error),
            _projection_failure_remediation(error, target_safe_to_reuse),
            target_safe_to_reuse=target_safe_to_reuse,
        ) from None
    if evidence.lifecycle.terminal_status != "succeeded":
        raise ProbeFailure(
            "smoke invocation",
            "PROBE_SMOKE_INCONCLUSIVE",
            "The target returned incomplete smoke evidence.",
            "Inspect the target lifecycle and retry only after restoring a known-safe state.",
            target_safe_to_reuse=not evidence.lifecycle.environment_state_uncertain,
        )
    return _SmokeResult(
        evidence=evidence,
        elapsed_seconds=elapsed_seconds,
        case_id=case.id,
        turn_id=case.turns[0].id,
        request_sha256=_json_sha256(
            {
                "case_id": case.id,
                "turn": case.turns[0].model_dump(mode="json"),
                "probe_context": case.probe_context,
            }
        ),
    )


def _print_smoke(
    smoke_result: _SmokeResult,
    resolved_target: _ResolvedTarget,
    *,
    show_response: bool,
) -> None:
    evidence = smoke_result.evidence
    console.print("")
    console.print("Smoke target invocation succeeded")
    console.print(f"  Target: {evidence.environment_id}")
    console.print(f"  Case: {smoke_result.case_id}")
    console.print(f"  Turn: {smoke_result.turn_id}")
    console.print(f"  Request sha256: {smoke_result.request_sha256}")
    console.print(f"  Evidence level: {evidence.evidence_scope.replace('_', ' ')}")
    encoded_response = json.dumps(
        evidence.final_response, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    console.print(
        f"  Response structure: {type(evidence.final_response).__name__}; "
        f"{len(encoded_response)} bytes"
    )
    console.print(f"  Response sha256: {hashlib.sha256(encoded_response).hexdigest()}")
    projection = _outcome_projection(resolved_target)
    if projection is not None:
        if evidence.normalized_result is None:
            raise AssertionError("successful projected smoke evidence requires a normalized result")
        public_result = evidence.public_normalized_result
        if public_result is None:
            raise AssertionError("projected smoke evidence requires a public normalized result")
        console.print(f"  Outcome projection sha256: {projection.digest}")
        console.print(
            "  Target-reported normalized result preview: "
            + json.dumps(public_result, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        )
    if show_response:
        console.print("  Private raw target response: " + encoded_response.decode())
        if evidence.normalized_result is not None:
            console.print(
                "  Private normalized result: "
                + json.dumps(
                    evidence.normalized_result,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
    observation_count = sum(
        observation.status != "missing" for observation in evidence.observations
    )
    console.print(f"  Trajectory observations: {observation_count}")
    console.print(
        "  State summary: "
        + (
            "before/after available"
            if resolved_target.supports_state_observation
            else "unverified (no state observation configured)"
        )
    )


def _projection_failure_remediation(
    error: OutcomeProjectionError, target_safe_to_reuse: bool
) -> str:
    selector_fix = f"Correct outcome selector {error.selector!r} for field {error.field!r}."
    if target_safe_to_reuse:
        return f"{selector_fix} Verified cleanup succeeded; then retry."
    return f"{selector_fix} Restore a known-safe fixture before retrying."


def _campaign_projection_failure(
    results: tuple[DatasetEvaluationResult, ...], output: Path
) -> ProbeFailure | None:
    trials = (
        trial
        for result in results
        for trial_set in (
            result.baseline.trial_set,
            *(case.trial_set for case in result.cases if case.trial_set is not None),
        )
        for trial in trial_set.trials
    )
    for trial in trials:
        lifecycle_failure = trial.lifecycle_failure
        if lifecycle_failure is None or lifecycle_failure.failed_phase != "outcome_projection":
            continue
        exact_reason = trial.inconclusive_reasons[0]
        target_safe_to_reuse = not lifecycle_failure.environment_state_may_remain
        remediation = (
            "Correct the named projection selector and retry with a new output."
            if target_safe_to_reuse
            else "Correct the named projection selector, restore a known-safe fixture, and retry "
            "with a new output."
        )
        return ProbeFailure(
            "evaluation",
            "PROBE_OUTCOME_PROJECTION_INVALID",
            f"{exact_reason} Paid preparation and target work already occurred; partial evidence "
            f"remains in {output}.",
            remediation,
            target_safe_to_reuse=target_safe_to_reuse,
        )
    return None


def _save_probe_config(
    *,
    data: Path,
    resolved_target: _ResolvedTarget,
    existing_config: ProbeProjectConfig | None,
) -> None:
    project_directory = Path.cwd() / _PROJECT_DIRECTORY
    config_path = project_directory / _PROBE_CONFIG
    try:
        config = ProbeProjectConfig(
            dataset=str(data.resolve()),
            target=resolved_target.reference,
            target_kind=resolved_target.kind,
            target_config_sha256=resolved_target.config_sha256,
            target_confirmation_sha256=resolved_target.confirmation_sha256,
            outcome_projection_sha256=(
                projection.digest
                if (projection := _outcome_projection(resolved_target)) is not None
                else None
            ),
        )
        if existing_config is not None:
            ensure_project_pattern_identity_key(project_directory)
            ensure_project_review_history_key(project_directory)
            console.print(f"  Using saved project config: {config_path}")
            return
        _ensure_private_project_directory(project_directory)
        ensure_project_pattern_identity_key(project_directory)
        ensure_project_review_history_key(project_directory)
        with create_private_output(config_path) as stream:
            json.dump(config.model_dump(mode="json"), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except FileExistsError:
        raise ProbeFailure(
            "analysis",
            "PROBE_CONFIG_EXISTS",
            ".ul/probe.json already exists; UL will not overwrite it.",
            "Move the existing config or run from a new project directory.",
            target_safe_to_reuse=True,
        ) from None
    except OSError:
        raise ProbeFailure(
            "analysis",
            "PROBE_CONFIG_WRITE_FAILED",
            "The successful smoke configuration could not be saved privately.",
            "Check project-directory permissions and retry; the target remains reusable.",
            target_safe_to_reuse=True,
        ) from None
    console.print(f"  Saved project config: {config_path}")


def _check_probe_config_binding(
    data: Path, resolved_target: _ResolvedTarget
) -> ProbeProjectConfig | None:
    project_directory = Path.cwd() / _PROJECT_DIRECTORY
    path = project_directory / _PROBE_CONFIG
    if project_directory.is_symlink():
        raise ProbeFailure(
            "target load",
            "PROBE_CONFIG_EXISTS",
            ".ul must be a private project directory, not a symbolic link.",
            "Replace it with a private local directory before running a target.",
            target_safe_to_reuse=True,
        )
    if not path.exists() and not path.is_symlink():
        return None
    expected = ProbeProjectConfig(
        dataset=str(data.resolve()),
        target=resolved_target.reference,
        target_kind=resolved_target.kind,
        target_config_sha256=resolved_target.config_sha256,
        target_confirmation_sha256=resolved_target.confirmation_sha256,
        outcome_projection_sha256=(
            projection.digest
            if (projection := _outcome_projection(resolved_target)) is not None
            else None
        ),
    )
    try:
        existing = _load_existing_probe_config(path)
    except (OSError, ValidationError, ValueError):
        existing = None
    if existing != expected:
        raise ProbeFailure(
            "target load",
            "PROBE_CONFIG_EXISTS",
            ".ul/probe.json does not bind this exact target and dataset.",
            "Move the existing config or run from a new project directory.",
            target_safe_to_reuse=True,
        )
    return existing


def _ensure_private_project_directory(path: Path) -> None:
    with suppress(FileExistsError):
        os.mkdir(path, 0o700)
    path_status = os.lstat(path)
    if os.name == "nt":
        if not stat.S_ISDIR(path_status.st_mode):
            raise OSError("project path is not a directory")
        return
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        descriptor_status = os.fstat(descriptor)
        if not stat.S_ISDIR(descriptor_status.st_mode) or not os.path.samestat(
            path_status, descriptor_status
        ):
            raise OSError("project path is not a directory")
        if hasattr(os, "getuid") and descriptor_status.st_uid != os.getuid():
            raise OSError("project directory is not owned by the current user")
        if os.name != "nt":
            os.fchmod(descriptor, 0o700)
    finally:
        os.close(descriptor)


def _load_existing_probe_config(path: Path) -> ProbeProjectConfig:
    path_status = os.lstat(path)
    if not stat.S_ISREG(path_status.st_mode) or path_status.st_size > 1_000_000:
        raise FileExistsError
    if os.name != "nt" and stat.S_IMODE(path_status.st_mode) & 0o077:
        raise FileExistsError
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor_status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_status.st_mode)
            or descriptor_status.st_nlink != 1
            or not os.path.samestat(path_status, descriptor_status)
        ):
            raise FileExistsError
        if hasattr(os, "getuid") and descriptor_status.st_uid != os.getuid():
            raise FileExistsError
        encoded = os.read(descriptor, 1_000_001)
    finally:
        os.close(descriptor)
    if len(encoded) > 1_000_000:
        raise FileExistsError
    return ProbeProjectConfig.model_validate_json(encoded)


def _validate_campaign_target_budget(
    plan: DatasetCampaignPlan,
    resolved_target: _ResolvedTarget,
    smoke_elapsed_seconds: float,
) -> float | None:
    if 1 + plan.calls.total_environment_api > resolved_target.maximum_executions:
        raise ProbeFailure(
            "augmentation preparation",
            "PROBE_TARGET_CALL_LIMIT_TOO_LOW",
            "The target's configured execution limit is below the displayed campaign plan.",
            "Raise the test target's max_executions or use fewer grounded examples.",
            target_safe_to_reuse=True,
        )
    if resolved_target.maximum_active_target_seconds is None:
        return None
    remaining = resolved_target.maximum_active_target_seconds - smoke_elapsed_seconds
    if remaining <= 0:
        raise ProbeFailure(
            "augmentation preparation",
            "PROBE_TARGET_WALL_LIMIT_EXHAUSTED",
            "The smoke call exhausted the command-wide target wall-time limit.",
            "Raise total_execution_timeout_seconds and rerun.",
            target_safe_to_reuse=True,
        )
    return remaining


def _print_campaign_budget(
    plan: DatasetCampaignPlan,
    confirmation: _CampaignConfirmation,
    *,
    selected_operator_ids: tuple[str, ...],
    limit: int,
) -> None:
    console.print("")
    console.print("Bounded active-probe campaign")
    console.print(f"  Source interactions: {len(plan.examples)} (limit {limit})")
    console.print(f"  Operators: {', '.join(selected_operator_ids)}")
    console.print(f"  Repetitions: {plan.calls.repetitions}")
    console.print(f"  Original agent invocations: {plan.calls.baseline}")
    console.print(f"  Probe agent invocations: {plan.calls.variation}")
    console.print(
        "  Command-wide environment API requests: "
        f"{confirmation.command_environment_api_requests} (includes smoke)"
    )
    console.print(f"  Semantic-model calls: up to {plan.calls.total_semantic_model}")
    console.print(f"  Completion tokens: 0..{plan.tokens.maximum}")
    if confirmation.monetary_cost_status == "bounded":
        assert confirmation.maximum_cost_usd is not None
        console.print(f"  Maximum monetary cost: ${confirmation.maximum_cost_usd:.6f} USD")
    else:
        console.print("  Monetary cost: UNKNOWN AND UNBOUNDED (no trusted pricing configured)")
    console.print(f"  Semantic provider: {confirmation.semantic_provider_id}")
    console.print(f"  Semantic endpoint sha256: {confirmation.semantic_endpoint_sha256}")
    console.print(f"  Semantic settings sha256: {confirmation.semantic_settings_sha256}")
    console.print(f"  Campaign plan sha256: {confirmation.campaign_plan_sha256}")
    console.print("  Data policy: " + json.dumps(confirmation.data_policy, sort_keys=True))
    console.print(f"  Maximum active wall time: {confirmation.maximum_wall_seconds:.1f} seconds")
    console.print(f"  Campaign confirmation sha256: {_model_sha256(confirmation)}")


def _campaign_confirmation(
    plan: DatasetCampaignPlan,
    settings: DatasetSemanticSettings,
    resolved_target: _ResolvedTarget,
    *,
    case_limit: int,
) -> _CampaignConfirmation:
    planned_target_seconds = (
        resolved_target.calls_per_execution + plan.calls.repetition_executions
    ) * _TARGET_TIMEOUT_SECONDS
    bounded_target_seconds = (
        resolved_target.maximum_active_target_seconds
        if resolved_target.maximum_active_target_seconds is not None
        else planned_target_seconds
    )
    if settings.semantic_provider_type == "openrouter":
        data_policy: dict[str, object] = {
            "external_processing": True,
            "provider_policy_declared": True,
            "data_collection": "deny",
            "zero_data_retention_required": True,
            "implication": (
                "The configured route requires data collection to be denied and zero data "
                "retention; the evaluator request is still processed externally."
            ),
        }
    else:
        data_policy = {
            "external_processing": True,
            "provider_policy_declared": False,
            "implication": (
                "The configured endpoint receives evaluator prompts and sample data; UL cannot "
                "verify its retention or training policy."
            ),
        }
    return _CampaignConfirmation(
        target_confirmation_sha256=resolved_target.confirmation_sha256,
        semantic_provider_id=settings.semantic_provider_id,
        semantic_provider_type=settings.semantic_provider_type,
        semantic_endpoint_sha256=settings.semantic_endpoint_sha256,
        semantic_settings_sha256=_model_sha256(_semantic_settings_snapshot(settings)),
        campaign_plan_sha256=_model_sha256(plan),
        case_limit=case_limit,
        data_policy=data_policy,
        command_environment_api_requests=(
            resolved_target.calls_per_execution + plan.calls.total_environment_api
        ),
        semantic_model_calls=plan.calls.total_semantic_model,
        maximum_completion_tokens=plan.tokens.maximum,
        maximum_wall_seconds=(
            bounded_target_seconds + plan.calls.total_semantic_model * settings.timeout_seconds
        ),
        monetary_cost_status=("bounded" if plan.money is not None else "unknown_unbounded"),
        maximum_cost_usd=plan.money.maximum if plan.money is not None else None,
    )


def _confirm_paid_execution(
    *, confirmation: _CampaignConfirmation, confirmed_digest: str | None
) -> bool:
    digest = _model_sha256(confirmation)
    if confirmed_digest is not None:
        if confirmed_digest != digest:
            raise ProbeFailure(
                "augmentation preparation",
                "PROBE_CAMPAIGN_CONFIRMATION_CHANGED",
                "The supplied campaign digest does not match the current provider or budget.",
                "Review the displayed provider, data policy, and budget; confirm its exact digest.",
                target_safe_to_reuse=True,
            )
        return True
    return typer.confirm("Run this exact paid/network campaign budget?", default=False)


def _validate_paid_execution_settings(settings: DatasetSemanticSettings) -> None:
    if not settings.live_calls or not settings.allow_external_data_processing:
        raise ProbeFailure(
            "augmentation preparation",
            "PROBE_SEMANTIC_CALLS_DISABLED",
            "Semantic-model execution is disabled.",
            "Set UL_LIVE=true after reviewing the displayed budget, then rerun.",
            target_safe_to_reuse=True,
        )
    if settings.api_key_required and (
        settings.api_key is None or not settings.api_key.get_secret_value().strip()
    ):
        raise ProbeFailure(
            "augmentation preparation",
            "PROBE_PROVIDER_CREDENTIAL_MISSING",
            f"The configured semantic provider requires {settings.api_key_environment_variable}.",
            f"Set {settings.api_key_environment_variable} and rerun.",
            target_safe_to_reuse=True,
        )


def _run_campaign(
    *,
    records: tuple[InteractionRecord, ...],
    selected_operator_ids: tuple[str, ...],
    resolved_target: _ResolvedTarget,
    settings: DatasetSemanticSettings,
    plan: DatasetCampaignPlan,
    output: Path,
    repetitions: int,
    remaining_target_seconds: float | None,
    run_context: DatasetEvidenceRunContext,
    progress_runtime: CampaignProgressRuntime,
    evaluator_preflight: EvaluatorModelPreflight | None,
    resume_campaign: bool,
    allow_insecure_http: bool,
    resolve_quarantine_after: Literal["environment-reset", "environment-replacement"] | None,
) -> tuple[DatasetEvaluationResult, ...]:
    resolved_target.revalidate_identity()
    if output.exists() and not resume_campaign:
        raise ProbeFailure(
            "output",
            "PROBE_EVIDENCE_EXISTS",
            "The evidence output already exists; UL will not overwrite it.",
            "Choose a new --output path.",
            target_safe_to_reuse=True,
        )
    try:
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError:
        raise ProbeFailure(
            "output",
            "PROBE_OUTPUT_PREPARATION_FAILED",
            "The private evidence directory could not be prepared.",
            "Check the output path and permissions, then rerun.",
            target_safe_to_reuse=True,
        ) from None
    augmentations_output = default_augmentations_output(output)
    augmentation_generation_context = create_dataset_augmentation_generation_context(
        selected_records=records,
        operators=tuple(
            dataset_operator_identity(operator_reference)
            for operator_reference in selected_operator_ids
        ),
        semantic_settings=DatasetAugmentationLedgerSemanticSettings(
            provider=settings.semantic_provider_id,
            endpoint_sha256=settings.semantic_endpoint_sha256,
            model=settings.model,
            render_model=settings.render_model,
            equivalence_model=settings.equivalence_model,
            max_input_chars=settings.max_input_chars,
            max_output_tokens=settings.max_output_tokens,
            max_render_tokens=settings.max_render_tokens,
            max_response_bytes=settings.max_response_bytes,
            timeout_seconds=settings.timeout_seconds,
        ),
    )
    expected_manifest = create_dataset_run_manifest(
        run_context=run_context,
        selected_records=records,
        selected_operator_ids=selected_operator_ids,
        repetitions=repetitions,
        max_environment_api_calls=plan.calls.total_environment_api,
        allow_environment_network=True,
        confirm_test_environment=True,
        allow_insecure_http=allow_insecure_http,
        save_augmentations=True,
        semantic_provider_type=settings.semantic_provider_type,
        semantic_base_url=settings.semantic_base_url,
        semantic_live_calls=settings.live_calls,
        semantic_allow_external_data_processing=settings.allow_external_data_processing,
        augmentations_output_path=str(augmentations_output.resolve()),
    )
    trial_journal: DatasetTrialJournal | None = None
    augmentation_ledger: DatasetAugmentationLedger | None = None
    try:
        if resume_campaign:
            recorded_manifest = read_dataset_run_manifest(manifest_path(output))
            if recorded_manifest != expected_manifest:
                raise ValueError("probe campaign manifest does not match the resumed campaign")
            trial_journal = open_dataset_trial_journal(journal_path(output), recorded_manifest)
            if trial_journal.snapshot.quarantined_unit_ids:
                if resolve_quarantine_after is None:
                    try:
                        _persist_probe_quarantine(
                            resolved_target,
                            "target_delivery_or_cleanup_uncertain",
                        )
                    except (OSError, ValueError):
                        raise ProbeFailure(
                            "output",
                            "PROBE_QUARANTINE_PERSIST_FAILED",
                            "UL could not persist target quarantine after interrupted delivery.",
                            "Do not call this target; fix the private .ul directory first.",
                            target_safe_to_reuse=False,
                        ) from None
                    raise ProbeFailure(
                        "target load",
                        "PROBE_TARGET_QUARANTINED",
                        "An interrupted probe trial has uncertain target delivery.",
                        "After an operator resets or replaces the test environment, rerun with "
                        "--resolve-quarantine-after environment-reset (or "
                        "environment-replacement).",
                        target_safe_to_reuse=False,
                    )
                _persist_probe_safety_state(
                    resolved_target,
                    status="reusable",
                    reason_code=(f"operator_attested_{resolve_quarantine_after.replace('-', '_')}"),
                )
            if durable_evidence_marker_manifest_sha256(output) != recorded_manifest.manifest_sha256:
                raise ValueError("probe evidence marker does not match the resumed campaign")
            augmentation_ledger = open_augmentation_ledger_for_resume(
                augmentations_output,
                expected_context=augmentation_generation_context,
                selected_records=records,
            )
        else:
            persist_dataset_run_manifest(manifest_path(output), expected_manifest)
            trial_journal = create_dataset_trial_journal(journal_path(output), expected_manifest)
            create_durable_evidence_output(output, expected_manifest.manifest_sha256)
            augmentation_ledger = create_private_augmentation_ledger(
                augmentations_output,
                generation_context=augmentation_generation_context,
                selected_records=records,
            )
    except ProbeFailure:
        if trial_journal is not None:
            trial_journal.close()
        raise
    except (OSError, ValueError):
        if augmentation_ledger is not None:
            augmentation_ledger.close()
        if trial_journal is not None:
            trial_journal.close()
        raise ProbeFailure(
            "output",
            "PROBE_DURABLE_STATE_INVALID",
            "The durable campaign manifest, journal, or evidence output could not be "
            "opened safely.",
            "Keep the target stopped and inspect the private campaign artifacts.",
            target_safe_to_reuse=not resume_campaign,
        ) from None

    terminal_states = trial_journal.snapshot.terminal_states
    if terminal_states:
        progress_runtime.tracker.hydrate_terminal_states(terminal_states)
        attempted_target_calls = sum(
            state in {"completed", "errored", "inconclusive", "quarantined"}
            for state in terminal_states.values()
        )
        progress_runtime.tracker.record_usage(
            target_calls=1 + attempted_target_calls,
            semantic_calls=None,
            environment_calls=(resolved_target.calls_per_execution * (1 + attempted_target_calls)),
            tokens=None,
        )

    def flush_progress_boundary() -> None:
        assert trial_journal is not None
        trial_journal.flush()

    progress_runtime.tracker.emit(status="running", stage="preflight")
    try:
        if evaluator_preflight is None:
            with progress_runtime.signal_control.installed():
                evaluator_preflight = asyncio.run(preflight_evaluator(settings))
            persist_evaluator_preflight(output, evaluator_preflight)
        if not progress_runtime.tracker.safe_boundary(
            progress_runtime.control,
            flush_progress_boundary,
        ):
            action = progress_runtime.control.requested_action()
            assert action is not None
            raise CampaignControlRequested(action)
    except CampaignControlRequested:
        augmentation_ledger.close()
        trial_journal.close()
        raise ProbeFailure(
            "augmentation preparation",
            "PROBE_PAUSED_AFTER_PREFLIGHT",
            "The campaign paused at the durable boundary after evaluator preflight.",
            "Run the opaque resume action to reuse completed smoke and preflight checkpoints.",
            target_safe_to_reuse=True,
        ) from None
    except Exception:
        augmentation_ledger.close()
        trial_journal.close()
        raise ProbeFailure(
            "augmentation preparation",
            "PROBE_AUGMENTATION_PREPARATION_FAILED",
            "The semantic evaluator preflight failed before target campaign execution.",
            "Verify provider settings and model compatibility, then rerun.",
            target_safe_to_reuse=True,
        ) from None
    try:
        connection = resolved_target.create_connection(
            plan.calls.total_environment_api,
            remaining_target_seconds,
        )
    except Exception:
        augmentation_ledger.close()
        trial_journal.close()
        raise ProbeFailure(
            "probe execution",
            "PROBE_TARGET_CONNECTION_FAILED",
            "The confirmed target connection could not be prepared.",
            "Revalidate the target receipt and advanced target configuration.",
            target_safe_to_reuse=True,
        ) from None
    try:
        try:
            output_stream, resume_evidence = open_resume_output(
                output,
                expected_context=run_context,
                selected_records=records,
                invariant_suite=None,
            )
        except (OSError, ValueError):
            raise ProbeFailure(
                "output",
                "PROBE_EVIDENCE_WRITE_FAILED",
                "The durable evidence output could not be opened safely.",
                "Keep the target stopped and inspect the private campaign artifacts.",
                target_safe_to_reuse=True,
            ) from None
        campaign_records = tuple(
            record for record in records if record.id not in resume_evidence.processed_ids
        )
        saved_augmentations = {
            record.source.id: record.augmentation for record in augmentation_ledger.snapshot.records
        }
        with output_stream:
            try:
                results = asyncio.run(
                    evaluate_interaction_records(
                        campaign_records,
                        selected_operator_ids,
                        settings,
                        connection,
                        output_stream,
                        repetitions=repetitions,
                        max_environment_api_calls=plan.calls.total_environment_api,
                        planned_target_calls=plan.calls.total_environment_api,
                        progress_plan=plan,
                        progress_runtime=progress_runtime,
                        complete_progress=False,
                        environment_calls_per_target_call=(resolved_target.calls_per_execution),
                        run_context=run_context,
                        evaluator_preflight=evaluator_preflight,
                        trial_journal=trial_journal,
                        augmentation_ledger=augmentation_ledger,
                        saved_augmentations=saved_augmentations,
                    )
                )
            except CampaignControlRequested:
                raise ProbeFailure(
                    "evaluation",
                    "PROBE_PAUSED_DURING_CAMPAIGN",
                    "The campaign paused at a durable trial boundary.",
                    "Run the opaque resume action to continue only unfinished trials.",
                    target_safe_to_reuse=True,
                ) from None
            except Exception:
                raise ProbeFailure(
                    "evaluation",
                    "PROBE_EVALUATION_FAILED",
                    "Evaluation stopped after campaign execution began; completed evidence "
                    "remains local.",
                    "Inspect the saved evidence and provider diagnostics before using a new "
                    "output.",
                    target_safe_to_reuse=False,
                ) from None
    except ProbeFailure:
        raise
    except Exception:
        raise ProbeFailure(
            "output",
            "PROBE_EVIDENCE_WRITE_FAILED",
            "The private evidence output could not be completed.",
            "Check disk space and permissions; inspect any retained evidence before retrying.",
            target_safe_to_reuse=False,
        ) from None
    finally:
        augmentation_ledger.close()
        trial_journal.close()
    return results


def _print_stronger_run(
    data: Path,
    target: str,
    output: Path,
    *,
    allow_insecure_http: bool,
    target_artifacts: tuple[Path, ...],
    http_preset: Literal["generic-json", "openai-chat"] | None,
    request_json_template: str | None,
    response_json_pointer: str | None,
    agent_model: str | None,
    header_from_env: tuple[str, ...],
    selected_operator_ids: tuple[str, ...],
    limit: int,
    repetitions: int,
) -> None:
    arguments = [
        "ul",
        "probe",
        str(data),
        "--target",
        target,
        "--output",
        str(output.with_name(output.stem + "-confirmation.jsonl")),
        "--limit",
        str(limit),
        "--repetitions",
        str(max(3, repetitions)),
    ]
    for operator_reference in selected_operator_ids:
        arguments.extend(("--operator", operator_reference))
    if allow_insecure_http:
        arguments.append("--allow-insecure-http")
    if http_preset is not None:
        arguments.extend(("--http-preset", http_preset))
    if request_json_template is not None:
        arguments.extend(("--request-json-template", request_json_template))
    if response_json_pointer is not None:
        arguments.extend(("--response-json-pointer", response_json_pointer))
    if agent_model is not None:
        arguments.extend(("--agent-model", agent_model))
    for mapping in header_from_env:
        arguments.extend(("--header-from-env", mapping))
    for artifact in target_artifacts:
        arguments.extend(("--target-artifact", str(artifact)))
    command = subprocess.list2cmdline(arguments) if os.name == "nt" else shlex.join(arguments)
    console.print(f"Stronger confirmation: rerun after adding more grounded examples: {command}")


def _print_failure(failure: ProbeFailure, *, diagnostic_artifact: Path | None) -> None:
    print_dataset_plain("UL probe stopped safely")
    print_dataset_plain(f"Stage: {failure.stage}")
    print_dataset_plain(f"Reason: {failure.reason_code}")
    print_dataset_plain(f"Explanation: {failure.explanation}")
    print_dataset_plain(f"Remediation: {failure.remediation}")
    print_dataset_plain(
        "Target safe to reuse: " + ("yes" if failure.target_safe_to_reuse else "no")
    )
    if diagnostic_artifact is not None:
        try:
            diagnostic_artifact.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with create_private_output(diagnostic_artifact) as stream:
                json.dump(
                    {
                        "stage": failure.stage,
                        "reason_code": failure.reason_code,
                        "explanation": failure.explanation,
                        "remediation": failure.remediation,
                        "target_safe_to_reuse": failure.target_safe_to_reuse,
                    },
                    stream,
                    indent=2,
                )
                stream.write("\n")
            print_dataset_plain(f"Private diagnostic: {diagnostic_artifact}")
        except OSError:
            print_dataset_plain("Private diagnostic could not be written.")
