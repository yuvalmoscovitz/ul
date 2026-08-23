from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, cast

import httpx
import typer
from pydantic import BaseModel, ConfigDict, Field, ValidationError
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
    LocalTargetConfig,
    LocalTargetConnection,
    PythonCallableTargetConfig,
    create_local_target_dry_run_plan,
    load_local_target_config,
)
from ul_core.models import ConversationRole, ConversationTurn

from ul_cli.dataset.evaluation.command import preflight_evaluator
from ul_cli.dataset.evaluation.operators import validate_operator_ids
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
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProbeProjectConfig(_StrictModel):
    schema_version: Literal[1] = 1
    dataset: str = Field(min_length=1)
    target: str = Field(min_length=1)
    target_kind: Literal["python_callable", "command", "http"]
    target_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    operator: Literal["input.surface.typing_noise"] = _PILOT_OPERATOR
    limit: Literal[10] = _PILOT_LIMIT
    repetitions: Literal[1] = _PILOT_REPETITIONS


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
    create_connection: Callable[[int], ProbeTargetConnection]


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
    confirm_target: Annotated[
        bool,
        typer.Option(
            "--confirm-target",
            help="Confirm the displayed digest identifies a safe dedicated test target.",
        ),
    ] = False,
    confirm_paid_execution: Annotated[
        bool,
        typer.Option(
            "--confirm-paid-execution",
            help="Authorize the displayed semantic/network campaign budget.",
        ),
    ] = False,
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
) -> None:
    """Smoke a real target, then optionally run one bounded active-probe pilot."""
    try:
        records = _load_pilot_records(data)
        resolved_target = _resolve_target(
            target,
            allow_insecure_http=allow_insecure_http,
        )
        _confirm_target(resolved_target, confirmed=confirm_target)
        smoke_evidence = asyncio.run(_run_smoke(records[0], resolved_target))
        _print_smoke(smoke_evidence, resolved_target)
        _save_probe_config(
            data=data,
            resolved_target=resolved_target,
        )
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
        _validate_campaign_target_budget(plan, resolved_target)
        _print_pilot_budget(plan, settings, resolved_target)
        if not _confirm_paid_execution(confirmed=confirm_paid_execution):
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
        )
        print_dataset_results(results, output, show_report_guidance=False)
        _print_stronger_run(
            data,
            target,
            output,
            allow_insecure_http=allow_insecure_http,
        )
        console.print("")
        report_evidence(output)
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


def _resolve_target(target: str, *, allow_insecure_http: bool) -> _ResolvedTarget:
    try:
        target_path = Path(target)
        if target_path.is_file():
            return _resolve_configured_target(target_path, allow_insecure_http=allow_insecure_http)
        if ":" not in target:
            raise ValueError("target must be module:callable or a target configuration JSON file")
        config = PythonCallableTargetConfig(
            target_id="probe-" + hashlib.sha256(target.encode()).hexdigest()[:16],
            working_directory=Path.cwd().resolve(),
            interpreter=Path(sys.executable).resolve(),
            target=target,
        )
        plan = create_local_target_dry_run_plan(config)
        return _local_target(target, config, plan.config_sha256)
    except (OSError, RuntimeError, ValidationError, ValueError) as error:
        raise ProbeFailure(
            "target load",
            "PROBE_TARGET_INVALID",
            str(error),
            "Use an importable module:callable or validate the target config with its "
            "advanced check.",
            target_safe_to_reuse=True,
        ) from None


def _resolve_configured_target(path: Path, *, allow_insecure_http: bool) -> _ResolvedTarget:
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
            create_connection=lambda maximum_calls: JsonHttpEnvironmentConnection.from_config(
                http_config,
                test_environment_confirmed=True,
                allow_insecure_http=allow_insecure_http,
                max_environment_api_calls=maximum_calls,
            ),
        )
    plan = create_local_target_dry_run_plan(local_config)
    return _local_target(str(path), local_config, plan.config_sha256)


def _local_target(reference: str, config: LocalTargetConfig, digest: str) -> _ResolvedTarget:
    return _ResolvedTarget(
        reference=reference,
        kind=config.kind,
        config=config,
        config_sha256=digest,
        calls_per_execution=1,
        maximum_executions=config.limits.max_executions,
        maximum_active_target_seconds=config.limits.total_execution_timeout_seconds,
        supports_state_observation=False,
        create_connection=lambda maximum_calls: LocalTargetConnection.from_config(
            config.model_copy(
                update={
                    "limits": config.limits.model_copy(update={"max_executions": maximum_calls})
                }
            ),
            customer_code_execution_confirmed=True,
        ),
    )


def _confirm_target(resolved_target: _ResolvedTarget, *, confirmed: bool) -> None:
    console.print("UL active-probe target")
    console.print(f"  Kind: {resolved_target.kind}")
    console.print(f"  Config sha256: {resolved_target.config_sha256}")
    if resolved_target.kind == "http":
        http_config = cast(JsonHttpTargetConfig, resolved_target.config)
        console.print(f"  Endpoint: {json_http_environment_config_urls(http_config)[0]}")
    console.print("Use only a dedicated test target that cannot cause real-world effects.")
    if confirmed:
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
) -> ExecutionEvidence:
    connection = resolved_target.create_connection(resolved_target.calls_per_execution)
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
        async with connection:
            evidence = await connection.execute(case)
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
    return evidence


def _print_smoke(evidence: ExecutionEvidence, resolved_target: _ResolvedTarget) -> None:
    console.print("")
    console.print("Smoke target invocation succeeded")
    console.print(f"  Target: {evidence.environment_id}")
    console.print(f"  Evidence level: {evidence.evidence_scope.replace('_', ' ')}")
    console.print(
        "  Normalized response: "
        + json.dumps(evidence.final_response, ensure_ascii=False, separators=(",", ":"))
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
            else "not available"
        )
    )


def _save_probe_config(
    *,
    data: Path,
    resolved_target: _ResolvedTarget,
) -> None:
    project_directory = Path.cwd() / _PROJECT_DIRECTORY
    config_path = project_directory / _PROBE_CONFIG
    try:
        _ensure_private_project_directory(project_directory)
        config = ProbeProjectConfig(
            dataset=str(data.resolve()),
            target=resolved_target.reference,
            target_kind=resolved_target.kind,
            target_config_sha256=resolved_target.config_sha256,
        )
        if config_path.exists():
            try:
                existing_config = _load_existing_probe_config(config_path)
            except (OSError, ValidationError, ValueError):
                raise FileExistsError from None
            if existing_config != config:
                raise FileExistsError
            console.print(f"  Using saved project config: {config_path}")
            return
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
) -> None:
    if plan.calls.total_environment_api > resolved_target.maximum_executions:
        raise ProbeFailure(
            "augmentation preparation",
            "PROBE_TARGET_CALL_LIMIT_TOO_LOW",
            "The target's configured execution limit is below the displayed campaign plan.",
            "Raise the test target's max_executions or use fewer grounded examples.",
            target_safe_to_reuse=True,
        )


def _print_pilot_budget(
    plan: DatasetCampaignPlan,
    settings: DatasetSemanticSettings,
    resolved_target: _ResolvedTarget,
) -> None:
    planned_target_seconds = plan.calls.repetition_executions * _TARGET_TIMEOUT_SECONDS
    bounded_target_seconds = (
        min(planned_target_seconds, resolved_target.maximum_active_target_seconds)
        if resolved_target.maximum_active_target_seconds is not None
        else planned_target_seconds
    )
    maximum_wall_seconds = (
        bounded_target_seconds + plan.calls.total_semantic_model * settings.timeout_seconds
    )
    console.print("")
    console.print("Bounded active-probe pilot")
    console.print(f"  Source interactions: {len(plan.examples)} (maximum {_PILOT_LIMIT})")
    console.print(f"  Operator: {_PILOT_OPERATOR}")
    console.print(f"  Repetitions: {plan.calls.repetitions}")
    console.print(f"  Original agent invocations: {plan.calls.baseline}")
    console.print(f"  Probe agent invocations: {plan.calls.variation}")
    console.print(f"  Environment API requests: {plan.calls.total_environment_api}")
    console.print(f"  Semantic-model calls: up to {plan.calls.total_semantic_model}")
    console.print(f"  Completion tokens: 0..{plan.tokens.maximum}")
    console.print("  Monetary cost: unavailable (no trusted pricing configured)")
    console.print(f"  Maximum active wall time: {maximum_wall_seconds:.1f} seconds")


def _confirm_paid_execution(*, confirmed: bool) -> bool:
    if confirmed:
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
) -> tuple[DatasetEvaluationResult, ...]:
    if output.exists():
        raise ProbeFailure(
            "analysis",
            "PROBE_EVIDENCE_EXISTS",
            "The evidence output already exists; UL will not overwrite it.",
            "Choose a new --output path.",
            target_safe_to_reuse=True,
        )
    try:
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        evaluator_preflight = asyncio.run(preflight_evaluator(settings))
    except (OSError, RuntimeError, ValueError):
        raise ProbeFailure(
            "augmentation preparation",
            "PROBE_AUGMENTATION_PREPARATION_FAILED",
            "The semantic evaluator preflight failed before target campaign execution.",
            "Verify provider settings and model compatibility, then rerun.",
            target_safe_to_reuse=True,
        ) from None
    connection = resolved_target.create_connection(plan.calls.total_environment_api)
    try:
        with create_private_output(output) as output_stream:
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
                    evaluator_preflight=evaluator_preflight,
                )
            )
    except (OSError, RuntimeError, TimeoutError, ValueError, httpx.HTTPError) as error:
        raise ProbeFailure(
            "probe execution",
            "PROBE_CAMPAIGN_FAILED",
            "The bounded campaign stopped; complete evidence written before failure remains local.",
            "Inspect the evidence and restore the test target before retrying with a new output.",
            target_safe_to_reuse=False,
        ) from error
    return results


def _print_stronger_run(
    data: Path,
    target: str,
    output: Path,
    *,
    allow_insecure_http: bool,
) -> None:
    arguments = [
        "ul",
        "probe",
        str(data),
        "--target",
        target,
        "--output",
        str(output.with_name(output.stem + "-confirmation.jsonl")),
        "--confirm-target",
        "--confirm-paid-execution",
        "--confirmation-run",
    ]
    if allow_insecure_http:
        arguments.append("--allow-insecure-http")
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
