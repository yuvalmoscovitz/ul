from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, cast

import httpx
import typer
import ul.local_target as local_target_module
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError
from ul import (
    DatasetEvaluationResult,
    DatasetSemanticSettings,
    EvaluationCase,
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
    CommandTargetConfig,
    LocalTargetConfig,
    LocalTargetConnection,
    PythonCallableTargetConfig,
    create_local_target_dry_run_plan,
    load_local_target_config,
)
from ul_core.models import ConversationRole, ConversationTurn

from ul_cli.dataset.evaluation.command import preflight_evaluator
from ul_cli.dataset.evaluation.operators import dataset_operator_identity, validate_operator_ids
from ul_cli.dataset.evaluation.records import (
    DatasetInputError,
    load_interaction_records,
    validate_model_input_bounds,
)
from ul_cli.dataset.evaluation.runner import evaluate_interaction_records
from ul_cli.dataset.presentation.evaluation import print_dataset_results
from ul_cli.dataset.presentation.runtime import console, print_dataset_plain
from ul_cli.dataset.storage.private_files import create_private_output
from ul_cli.dataset_campaign import DatasetCampaignPlan, create_dataset_campaign_plan
from ul_cli.dataset_review import (
    DatasetEvidenceRunContext,
    DatasetEvidenceSemanticSettings,
    create_dataset_evidence_run_context,
)
from ul_cli.report import report_evidence

_PILOT_LIMIT = 10
_PILOT_REPETITIONS = 1
_CONFIRMATION_REPETITIONS = 3
_PILOT_OPERATOR = "input.surface.typing_noise"
_TARGET_TIMEOUT_SECONDS = 30.0
_PROJECT_DIRECTORY = ".ul"
_PROBE_CONFIG = "probe.json"
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
    schema_version: Literal[2] = 2
    dataset: str = Field(min_length=1)
    target: str = Field(min_length=1)
    target_kind: Literal["python_callable", "command", "http"]
    target_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_confirmation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    operator: Literal["input.surface.typing_noise"] = _PILOT_OPERATOR
    limit: Literal[10] = _PILOT_LIMIT
    repetitions: Literal[1] = _PILOT_REPETITIONS


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
    data_policy: dict[str, object]
    command_environment_api_requests: int
    semantic_model_calls: int
    maximum_completion_tokens: int
    maximum_wall_seconds: float
    monetary_cost_status: Literal["bounded", "unknown_unbounded"]
    maximum_cost_usd: float | None = None


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
    }


def _artifact_identity(path: Path) -> _ArtifactIdentity:
    resolved = path.resolve(strict=True)
    path_status = os.lstat(resolved)
    descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor_status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_status.st_mode)
            or descriptor_status.st_size > 256 * 1024 * 1024
            or not os.path.samestat(path_status, descriptor_status)
        ):
            raise ValueError("target artifact must be a stable bounded regular file")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return _ArtifactIdentity(path=str(resolved), sha256=digest.hexdigest())


def _resolve_executable(value: Path | str, working_directory: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute() and candidate.parent != Path("."):
        candidate = working_directory / candidate
    if candidate.parent == Path("."):
        found = shutil.which(str(candidate))
        if found is None:
            raise ValueError("target executable was not found")
        candidate = Path(found)
    return candidate.resolve(strict=True)


def _python_module_artifacts(
    config: PythonCallableTargetConfig, explicit_artifacts: tuple[Path, ...]
) -> tuple[_ArtifactIdentity, ...]:
    module_name = config.target.partition(":")[0]
    current = config.working_directory.resolve(strict=True)
    paths: list[Path] = []
    parts = module_name.split(".")
    for index, part in enumerate(parts):
        module_file = current / f"{part}.py"
        package = current / part
        final = index == len(parts) - 1
        if final and module_file.is_file():
            paths.append(module_file)
            break
        init_file = package / "__init__.py"
        if not init_file.is_file():
            raise ValueError("Python target module must resolve to source inside working_directory")
        paths.append(init_file)
        current = package
    worker = Path(local_target_module.__file__).with_name("_local_worker.py")
    paths.append(worker)
    for path in explicit_artifacts:
        candidate = path if path.is_absolute() else config.working_directory / path
        paths.append(candidate)
    identities: dict[str, _ArtifactIdentity] = {}
    for path in paths:
        identity = _artifact_identity(path)
        identities[identity.path] = identity
    return tuple(identities.values())


def _command_artifacts(
    config: CommandTargetConfig, explicit_artifacts: tuple[Path, ...]
) -> tuple[_ArtifactIdentity, ...]:
    executable = _resolve_executable(config.argv[0], config.working_directory)
    artifacts = [_artifact_identity(executable)]
    for argument in config.argv[1:]:
        candidate = Path(argument)
        if not candidate.is_absolute():
            candidate = config.working_directory / candidate
        if candidate.is_file():
            identity = _artifact_identity(candidate)
            if identity not in artifacts:
                artifacts.append(identity)
    for path in explicit_artifacts:
        candidate = path if path.is_absolute() else config.working_directory / path
        identity = _artifact_identity(candidate)
        if identity not in artifacts:
            artifacts.append(identity)
    generic_prefixes = ("bash", "node", "python", "ruby", "sh")
    if executable.name.startswith(generic_prefixes) and len(artifacts) == 1:
        raise ValueError(
            "generic command workers require --target-artifact for their script or artifact"
        )
    return tuple(artifacts)


def _local_target_confirmation(
    reference: str,
    config: LocalTargetConfig,
    digest: str,
    explicit_artifacts: tuple[Path, ...] = (),
) -> _TargetConfirmation:
    if isinstance(config, PythonCallableTargetConfig):
        executable = _artifact_identity(
            _resolve_executable(config.interpreter, config.working_directory)
        )
        artifacts = _python_module_artifacts(config, explicit_artifacts)
        callable_name = config.target
    else:
        artifacts = _command_artifacts(config, explicit_artifacts)
        executable = artifacts[0]
        callable_name = None
    return _TargetConfirmation(
        kind=config.kind,
        reference=reference,
        config_sha256=digest,
        executable=executable,
        artifacts=artifacts,
        environment=tuple(
            _EnvironmentIdentity(
                name=name,
                value_sha256=hashlib.sha256(os.environ[name].encode()).hexdigest(),
            )
            for name in sorted(config.environment_allowlist)
        ),
        callable=callable_name,
    )


def _require_same_confirmation(
    expected: _TargetConfirmation,
    reference: str,
    config: LocalTargetConfig,
    digest: str,
    explicit_artifacts: tuple[Path, ...],
) -> None:
    try:
        current = _local_target_confirmation(reference, config, digest, explicit_artifacts)
    except (OSError, ValueError):
        current = None
    if current != expected:
        raise ProbeFailure(
            "target load",
            "PROBE_TARGET_IDENTITY_CHANGED",
            "A confirmed executable or target artifact changed before launch.",
            "Review the new target identity and rerun the command.",
            target_safe_to_reuse=True,
        )


def probe(
    data: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Interaction JSONL."),
    ],
    target: Annotated[
        str,
        typer.Option(
            help=("Python module:callable, or a local/HTTP target configuration JSON file.")
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(help="New normal UL evidence JSONL file."),
    ] = Path(_DEFAULT_EVIDENCE),
    target_artifact: Annotated[
        list[Path] | None,
        typer.Option(help="Additional command worker artifact to hash and bind; repeat as needed."),
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
    confirmation_run: Annotated[
        bool,
        typer.Option(
            "--confirmation-run",
            help="After a pilot, repeat every original/probe arm three times.",
        ),
    ] = False,
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
) -> None:
    """Smoke a real target, then optionally run one bounded active-probe pilot."""
    try:
        records = _load_pilot_records(data)
        resolved_target = _resolve_target(
            target,
            allow_insecure_http=allow_insecure_http,
            explicit_artifacts=tuple(target_artifact or ()),
        )
        existing_config = _check_probe_config_binding(data, resolved_target)
        _confirm_target(resolved_target, confirmed_digest=confirm_target)
        smoke_result = asyncio.run(_run_smoke(records[0], resolved_target))
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
            settings = load_dataset_semantic_settings()
            validate_model_input_bounds(records, settings.max_input_chars)
            selected_operator_ids = validate_operator_ids([_PILOT_OPERATOR])
            repetitions = _CONFIRMATION_REPETITIONS if confirmation_run else _PILOT_REPETITIONS
            plan = create_dataset_campaign_plan(
                records=records,
                selected_operator_ids=selected_operator_ids,
                repetitions=repetitions,
                target_calls_per_execution=resolved_target.calls_per_execution,
                settings=settings,
            )
            remaining_target_seconds = _validate_campaign_target_budget(
                plan, resolved_target, smoke_result.elapsed_seconds
            )
            campaign_confirmation = _campaign_confirmation(plan, settings, resolved_target)
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
        _print_pilot_budget(plan, campaign_confirmation)
        if not _confirm_paid_execution(
            confirmation=campaign_confirmation,
            confirmed_digest=confirm_paid_execution,
        ):
            console.print("Stopped after smoke. No semantic-model calls were made.")
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
        )
        try:
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
            )
            console.print("")
            report_evidence(output)
        except Exception:
            raise ProbeFailure(
                "analysis",
                "PROBE_REPORT_FAILED",
                "The campaign evidence was written, but the report could not be produced.",
                "Run `ul report` against the saved private evidence file.",
                target_safe_to_reuse=True,
            ) from None
    except ProbeFailure as failure:
        _print_failure(failure, diagnostic_artifact=diagnostic_artifact)
        raise typer.Exit(code=2) from None


def _load_pilot_records(data: Path) -> tuple[InteractionRecord, ...]:
    try:
        return load_interaction_records(data)[:_PILOT_LIMIT]
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
) -> _ResolvedTarget:
    try:
        target_path = Path(target)
        if target_path.is_file():
            return _resolve_configured_target(
                target_path,
                allow_insecure_http=allow_insecure_http,
                explicit_artifacts=explicit_artifacts,
            )
        if ":" not in target:
            raise ValueError("target must be module:callable or a target configuration JSON file")
        config = PythonCallableTargetConfig(
            target_id="probe-" + hashlib.sha256(target.encode()).hexdigest()[:16],
            working_directory=Path.cwd().resolve(),
            interpreter=Path(sys.executable).resolve(),
            target=target,
        )
        plan = create_local_target_dry_run_plan(config)
        return _local_target(target, config, plan.config_sha256, explicit_artifacts)
    except (OSError, RuntimeError, ValidationError, ValueError) as error:
        raise ProbeFailure(
            "target load",
            "PROBE_TARGET_INVALID",
            str(error),
            "Use an importable module:callable or validate the target config with its "
            "advanced check.",
            target_safe_to_reuse=True,
        ) from None


def _resolve_configured_target(
    path: Path,
    *,
    allow_insecure_http: bool,
    explicit_artifacts: tuple[Path, ...],
) -> _ResolvedTarget:
    try:
        local_config = load_local_target_config(path)
    except ValueError:
        http_config = load_json_http_environment_config(path)
        validate_json_http_environment_configuration(
            http_config,
            test_environment_confirmed=True,
            allow_insecure_http=allow_insecure_http,
        )
        digest = json_http_environment_config_sha256(http_config)
        confirmation = _TargetConfirmation(kind="http", reference=str(path), config_sha256=digest)
        return _ResolvedTarget(
            reference=str(path),
            kind="http",
            config=http_config,
            config_sha256=digest,
            calls_per_execution=json_http_environment_calls_per_execution(http_config),
            maximum_executions=1_000,
            maximum_active_target_seconds=None,
            supports_state_observation=json_http_environment_capabilities(
                http_config
            ).supports_state_observation,
            confirmation=confirmation,
            confirmation_sha256=_model_sha256(confirmation),
            create_connection=lambda maximum_calls, maximum_seconds: (
                JsonHttpEnvironmentConnection.from_config(
                    http_config,
                    test_environment_confirmed=True,
                    allow_insecure_http=allow_insecure_http,
                    max_environment_api_calls=maximum_calls,
                )
            ),
            revalidate_identity=lambda: None,
        )
    plan = create_local_target_dry_run_plan(local_config)
    return _local_target(str(path), local_config, plan.config_sha256, explicit_artifacts)


def _local_target(
    reference: str,
    config: LocalTargetConfig,
    digest: str,
    explicit_artifacts: tuple[Path, ...] = (),
) -> _ResolvedTarget:
    confirmation = _local_target_confirmation(reference, config, digest, explicit_artifacts)
    return _ResolvedTarget(
        reference=reference,
        kind=config.kind,
        config=config,
        config_sha256=digest,
        calls_per_execution=1,
        maximum_executions=config.limits.max_executions,
        maximum_active_target_seconds=config.limits.total_execution_timeout_seconds,
        supports_state_observation=False,
        confirmation=confirmation,
        confirmation_sha256=_model_sha256(confirmation),
        create_connection=lambda maximum_calls, maximum_seconds: LocalTargetConnection.from_config(
            config.model_copy(
                update={
                    "limits": config.limits.model_copy(
                        update={
                            "max_executions": maximum_calls,
                            "total_execution_timeout_seconds": (
                                maximum_seconds
                                if maximum_seconds is not None
                                else config.limits.total_execution_timeout_seconds
                            ),
                        }
                    )
                }
            ),
            customer_code_execution_confirmed=True,
        ),
        revalidate_identity=lambda: _require_same_confirmation(
            confirmation, reference, config, digest, explicit_artifacts
        ),
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
    if not typer.confirm("Trust this exact target digest and continue with one smoke call?"):
        raise ProbeFailure(
            "target load",
            "PROBE_TARGET_NOT_CONFIRMED",
            "Target safety was not confirmed; UL made no target or semantic-model calls.",
            "Review the target and rerun when its displayed digest is trusted.",
            target_safe_to_reuse=True,
        )


async def _run_smoke(
    record: InteractionRecord,
    resolved_target: _ResolvedTarget,
) -> _SmokeResult:
    resolved_target.revalidate_identity()
    connection = resolved_target.create_connection(
        resolved_target.calls_per_execution,
        resolved_target.maximum_active_target_seconds,
    )
    case = EvaluationCase(
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
    try:
        started_at = time.monotonic()
        async with connection:
            evidence = await connection.execute(case)
        elapsed_seconds = time.monotonic() - started_at
        validate_execution_evidence(case, connection, evidence)
    except (OSError, RuntimeError, TimeoutError, ValueError, httpx.HTTPError) as error:
        raise ProbeFailure(
            "smoke invocation",
            "PROBE_SMOKE_FAILED",
            "The target did not complete the original smoke interaction.",
            "Run the target's advanced environment/local dry-run check, then retry.",
            target_safe_to_reuse=False,
        ) from error
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
    if show_response:
        console.print("  Private normalized response: " + encoded_response.decode())
    observation_count = sum(
        observation.status != "missing" for observation in evidence.observations
    )
    console.print(f"  Trajectory observations: {observation_count}")
    console.print(
        "  State summary: "
        + (
            "before/after available"
            if resolved_target.supports_state_observation
            else "not available"
        )
    )


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
        )
        if existing_config is not None:
            console.print(f"  Using saved project config: {config_path}")
            return
        _ensure_private_project_directory(project_directory)
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
    if os.name == "nt":
        if not stat.S_ISDIR(os.lstat(path).st_mode):
            raise OSError("project path is not a directory")
        return
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        path_status = os.fstat(descriptor)
        if not stat.S_ISDIR(path_status.st_mode):
            raise OSError("project path is not a directory")
        if hasattr(os, "getuid") and path_status.st_uid != os.getuid():
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


def _print_pilot_budget(
    plan: DatasetCampaignPlan,
    confirmation: _CampaignConfirmation,
) -> None:
    console.print("")
    console.print("Bounded active-probe pilot")
    console.print(f"  Source interactions: {len(plan.examples)} (maximum {_PILOT_LIMIT})")
    console.print(f"  Operator: {_PILOT_OPERATOR}")
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
    console.print("  Data policy: " + json.dumps(confirmation.data_policy, sort_keys=True))
    console.print(f"  Maximum active wall time: {confirmation.maximum_wall_seconds:.1f} seconds")
    console.print(f"  Campaign confirmation sha256: {_model_sha256(confirmation)}")


def _campaign_confirmation(
    plan: DatasetCampaignPlan,
    settings: DatasetSemanticSettings,
    resolved_target: _ResolvedTarget,
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
) -> tuple[DatasetEvaluationResult, ...]:
    resolved_target.revalidate_identity()
    if output.exists():
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
    try:
        evaluator_preflight = asyncio.run(preflight_evaluator(settings))
    except Exception:
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
        raise ProbeFailure(
            "probe execution",
            "PROBE_TARGET_CONNECTION_FAILED",
            "The confirmed target connection could not be prepared.",
            "Revalidate the target receipt and advanced target configuration.",
            target_safe_to_reuse=True,
        ) from None
    try:
        with create_private_output(output) as output_stream:
            try:
                results = asyncio.run(
                    evaluate_interaction_records(
                        records,
                        selected_operator_ids,
                        settings,
                        connection,
                        output_stream,
                        repetitions=repetitions,
                        max_environment_api_calls=plan.calls.total_environment_api,
                        planned_target_calls=plan.calls.total_environment_api,
                        run_context=run_context,
                        evaluator_preflight=evaluator_preflight,
                    )
                )
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
    return results


def _print_stronger_run(
    data: Path,
    target: str,
    output: Path,
    *,
    allow_insecure_http: bool,
    target_artifacts: tuple[Path, ...],
) -> None:
    arguments = [
        "ul",
        "probe",
        str(data),
        "--target",
        target,
        "--output",
        str(output.with_name(output.stem + "-confirmation.jsonl")),
        "--confirmation-run",
    ]
    if allow_insecure_http:
        arguments.append("--allow-insecure-http")
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
