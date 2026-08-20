from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import typer
from ul import DatasetSemanticSettings, load_dataset_invariant_suite, load_dataset_semantic_settings
from ul.http_environment import (
    json_http_environment_capabilities,
    json_http_environment_config_sha256,
    json_http_environment_origin,
    load_json_http_environment_config,
    validate_json_http_environment_configuration,
)
from ul_core.augmentation_catalog import (
    AugmentationBinding,
    AugmentationMode,
    AugmentationRef,
    AugmentationScope,
    BuiltinAugmentationSpec,
    builtin_augmentation_catalog,
)
from ul_core.evaluation import EnvironmentCapabilities

from ul_cli.dataset import validate_interaction_dataset
from ul_cli.project import ProjectConfig, load_project, resolve_project_path

app = typer.Typer(help="Inspect UL's built-in augmentation library.")

_SCOPES: tuple[AugmentationScope, ...] = ("input", "conversation", "environment")
_MODES: tuple[AugmentationMode, ...] = (
    "dataset_variation",
    "scenario_materialization",
    "conversation_stress",
    "environment_fault",
)
_REFERENCE_PATTERN = re.compile(
    r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+(?:@(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))?$"
)

_ReadinessStatus = Literal["ready", "blocked", "manual"]
_ProjectStatus = Literal["ready", "missing", "invalid"]


@dataclass(frozen=True)
class _ReadinessReason:
    code: str
    message: str

    def as_json(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class _ProjectReadiness:
    status: _ProjectStatus
    reason: str | None
    available_source_features: frozenset[str]
    dataset_reasons: tuple[_ReadinessReason, ...]
    semantic_model_reasons: tuple[_ReadinessReason, ...]
    environment_reasons: tuple[_ReadinessReason, ...]
    environment_capabilities: EnvironmentCapabilities | None
    customer_evaluator_reasons: tuple[_ReadinessReason, ...]


@dataclass(frozen=True)
class _PlannedAugmentation:
    ref: AugmentationRef
    scope: AugmentationScope
    mode: AugmentationMode
    status: _ReadinessStatus
    reasons: tuple[_ReadinessReason, ...]
    command: str | None

    def as_json(self) -> dict[str, object]:
        return {
            "ref": self.ref.model_dump(mode="json"),
            "scope": self.scope,
            "mode": self.mode,
            "status": self.status,
            "reasons": [reason.as_json() for reason in self.reasons],
            "command": self.command,
        }


@app.command("list")
def list_augmentations(
    scope: Annotated[str | None, typer.Option(help="Only show one augmentation scope.")] = None,
    mode: Annotated[str | None, typer.Option(help="Only show one execution mode.")] = None,
    cli_only: Annotated[
        bool, typer.Option("--cli-only", help="Only show augmentations with a CLI command.")
    ] = False,
    all_versions: Annotated[
        bool, typer.Option("--all-versions", help="Show every registered version.")
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit stable machine-readable JSON.")
    ] = False,
) -> None:
    """List every built-in augmentation through one catalog."""
    selected_scope = _parse_scope(scope)
    selected_mode = _parse_mode(mode)
    augmentations = builtin_augmentation_catalog().list(
        scope=selected_scope,
        mode=selected_mode,
        cli_only=cli_only,
        latest_only=not all_versions,
    )
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "augmentations": [_catalog_item(item) for item in augmentations],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not augmentations:
        typer.echo("No augmentations matched.")
        return
    typer.echo("Built-in augmentations")
    for augmentation in augmentations:
        modes = ",".join(binding.mode for binding in augmentation.bindings)
        cli_available = "yes" if augmentation.cli_available else "no"
        typer.echo(f"{augmentation.ref.id}@{augmentation.ref.version}")
        typer.echo(f"  scope={augmentation.scope} modes={modes} cli_available={cli_available}")
        typer.echo(f"  {augmentation.summary}")
    typer.echo("Use 'ul augmentations show ID[@VERSION]' for requirements and safety details.")


@app.command("plan")
def plan_augmentations(
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit stable machine-readable JSON.")
    ] = False,
) -> None:
    """Classify built-in augmentations for the current project without external calls."""
    project = _load_project_readiness()
    augmentations = tuple(
        _plan_augmentation(augmentation, project)
        for augmentation in builtin_augmentation_catalog().list()
    )
    summary = {
        status: sum(augmentation.status == status for augmentation in augmentations)
        for status in ("ready", "blocked", "manual")
    }
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "project": {"status": project.status, "reason": project.reason},
                    "summary": summary,
                    "inspection": {
                        "model_calls": 0,
                        "environment_calls": 0,
                        "network_requests": 0,
                    },
                    "augmentations": [augmentation.as_json() for augmentation in augmentations],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    typer.echo(
        "Augmentation readiness: "
        f"{summary['ready']} ready, {summary['blocked']} blocked, {summary['manual']} manual"
    )
    if project.reason is not None:
        typer.echo(f"Project: {project.reason}")
    for augmentation in augmentations:
        typer.echo(
            f"{augmentation.status.upper()} {augmentation.ref.id}@{augmentation.ref.version}"
        )
        for reason in augmentation.reasons:
            typer.echo(f"  Reason: {reason.message}")
        if augmentation.command is not None:
            typer.echo(f"  Command: {augmentation.command}")
    typer.echo("Inspection only: 0 model calls, 0 environment calls, 0 network requests.")


@app.command("show")
def show_augmentation(
    reference: Annotated[str, typer.Argument(help="Augmentation ID with optional @VERSION.")],
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit stable machine-readable JSON.")
    ] = False,
) -> None:
    """Show execution requirements for one built-in augmentation."""
    augmentation_id, version = _parse_reference(reference)
    try:
        augmentation = builtin_augmentation_catalog().get(augmentation_id, version)
    except KeyError:
        raise typer.BadParameter(
            "unknown augmentation; use ID or ID@VERSION from 'ul augmentations list'",
            param_hint="REFERENCE",
        ) from None
    if json_output:
        typer.echo(
            json.dumps(
                {"schema_version": "1.0.0", "augmentation": _catalog_item(augmentation)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    typer.echo(f"{augmentation.ref.id}@{augmentation.ref.version}")
    typer.echo(f"Summary: {augmentation.summary}")
    typer.echo(f"Scope: {augmentation.scope}")
    typer.echo(f"CLI execution available: {'yes' if augmentation.cli_available else 'no'}")
    for binding in augmentation.bindings:
        typer.echo(f"Mode: {binding.mode}")
        typer.echo(f"  Stages: {', '.join(binding.stages)}")
        typer.echo(f"  Execution owner: {_execution_owner_label(binding.execution_owner)}")
        typer.echo(f"  CLI command: {binding.command or 'unavailable'}")
        requirements = binding.requirements
        typer.echo(
            "  Requires: "
            + _format_requirements(
                requirements.required_source_features,
                semantic_model=requirements.semantic_model,
                environment=requirements.environment,
                conversations=requirements.conversations,
                state_observation=requirements.state_observation,
                customer_evaluator=requirements.customer_evaluator,
                environment_capabilities=requirements.environment_capabilities,
                human_review=requirements.human_review,
            )
        )
    typer.echo("Catalog inspection: 0 model calls, 0 environment calls, 0 network requests.")
    typer.echo("Execution safety and call cost are enforced by the selected command's planner.")


def _load_project_readiness() -> _ProjectReadiness:
    try:
        project_root, config = load_project()
    except typer.BadParameter as error:
        if str(error) == "no UL project found; run 'ul init' first":
            status: _ProjectStatus = "missing"
            reason = "No UL project found; run 'ul init' first."
        else:
            status = "invalid"
            reason = "The UL project configuration is invalid."
        return _ProjectReadiness(
            status=status,
            reason=reason,
            available_source_features=frozenset(),
            dataset_reasons=(),
            semantic_model_reasons=(),
            environment_reasons=(),
            environment_capabilities=None,
            customer_evaluator_reasons=(),
        )
    available_source_features, dataset_reasons = _dataset_readiness(project_root, config)
    environment_reasons, environment_capabilities = _environment_readiness(project_root, config)
    return _ProjectReadiness(
        status="ready",
        reason=None,
        available_source_features=available_source_features,
        dataset_reasons=dataset_reasons,
        semantic_model_reasons=_semantic_model_readiness(),
        environment_reasons=environment_reasons,
        environment_capabilities=environment_capabilities,
        customer_evaluator_reasons=_customer_evaluator_readiness(project_root, config),
    )


def _dataset_readiness(
    project_root: Path,
    config: ProjectConfig,
) -> tuple[frozenset[str], tuple[_ReadinessReason, ...]]:
    try:
        validate_interaction_dataset(resolve_project_path(config.dataset, project_root))
    except ValueError:
        return frozenset(), (
            _ReadinessReason(
                code="dataset_unavailable",
                message="The configured interaction dataset is missing or invalid.",
            ),
        )
    return frozenset(("production interaction",)), ()


def _semantic_model_readiness() -> tuple[_ReadinessReason, ...]:
    try:
        settings = load_dataset_semantic_settings()
    except ValueError:
        return (
            _ReadinessReason(
                code="semantic_model_configuration_invalid",
                message="The semantic model configuration is invalid.",
            ),
        )
    reasons: list[_ReadinessReason] = []
    if not settings.live_calls:
        reasons.append(
            _ReadinessReason(
                code="semantic_model_calls_disabled",
                message=(
                    "Semantic model calls are disabled; set UL_LIVE=true or "
                    "UL_DATASET_LIVE_CALLS=true."
                ),
            )
        )
    if not settings.allow_external_data_processing:
        reasons.append(
            _ReadinessReason(
                code="external_data_processing_disabled",
                message=(
                    "External semantic processing is disabled; set UL_LIVE=true or "
                    "UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING=true."
                ),
            )
        )
    if settings.api_key_required and not _semantic_api_key(settings):
        reasons.append(
            _ReadinessReason(
                code="semantic_model_credentials_missing",
                message=f"The semantic model requires {settings.api_key_environment_variable}.",
            )
        )
    return tuple(reasons)


def _semantic_api_key(settings: DatasetSemanticSettings) -> str:
    return settings.api_key.get_secret_value().strip() if settings.api_key is not None else ""


def _environment_readiness(
    project_root: Path,
    config: ProjectConfig,
) -> tuple[tuple[_ReadinessReason, ...], EnvironmentCapabilities | None]:
    reasons: list[_ReadinessReason] = []
    if not config.allow_environment_network:
        reasons.append(
            _ReadinessReason(
                code="environment_network_not_authorized",
                message="The project has not authorized environment network access.",
            )
        )
    if not config.confirm_test_environment:
        reasons.append(
            _ReadinessReason(
                code="test_environment_not_confirmed",
                message="The project has not confirmed a resettable test environment.",
            )
        )
    try:
        environment_config = load_json_http_environment_config(
            resolve_project_path(config.environment_config, project_root)
        )
    except (RuntimeError, ValueError):
        reasons.append(
            _ReadinessReason(
                code="environment_config_unavailable",
                message="The configured environment connection is missing or invalid.",
            )
        )
        return tuple(reasons), None
    trusted = True
    if json_http_environment_config_sha256(environment_config) != config.environment_config_sha256:
        trusted = False
        reasons.append(
            _ReadinessReason(
                code="environment_config_changed",
                message="The environment configuration changed after 'ul init'.",
            )
        )
    if json_http_environment_origin(environment_config) != config.environment_origin:
        trusted = False
        reasons.append(
            _ReadinessReason(
                code="environment_origin_changed",
                message="The environment origin changed after 'ul init'.",
            )
        )
    if not trusted:
        return tuple(reasons), None
    try:
        validate_json_http_environment_configuration(
            environment_config,
            test_environment_confirmed=True,
            allow_insecure_http=config.allow_insecure_http,
        )
    except (RuntimeError, ValueError):
        reasons.append(
            _ReadinessReason(
                code="environment_configuration_invalid",
                message=(
                    "The environment connection is not locally executable; check transport and "
                    "header environment variables."
                ),
            )
        )
        return tuple(reasons), None
    return tuple(reasons), json_http_environment_capabilities(environment_config)


def _customer_evaluator_readiness(
    project_root: Path,
    config: ProjectConfig,
) -> tuple[_ReadinessReason, ...]:
    if config.invariants is None:
        return (
            _ReadinessReason(
                code="customer_evaluator_unavailable",
                message="The project does not configure customer invariants.",
            ),
        )
    try:
        load_dataset_invariant_suite(resolve_project_path(config.invariants, project_root))
    except (RuntimeError, ValueError):
        return (
            _ReadinessReason(
                code="customer_evaluator_unavailable",
                message="The configured customer invariants are missing or invalid.",
            ),
        )
    return ()


def _plan_augmentation(
    augmentation: BuiltinAugmentationSpec,
    project: _ProjectReadiness,
) -> _PlannedAugmentation:
    cli_bindings = tuple(binding for binding in augmentation.bindings if binding.cli_available)
    if not cli_bindings:
        reasons = [
            _ReadinessReason(
                code="cli_unavailable",
                message="No CLI command is available; use the SDK augmentation registry.",
            )
        ]
        source_features = tuple(
            dict.fromkeys(
                feature
                for binding in augmentation.bindings
                for feature in binding.requirements.required_source_features
            )
        )
        if source_features:
            reasons.append(_manual_source_feature_reason(source_features))
        return _PlannedAugmentation(
            ref=augmentation.ref,
            scope=augmentation.scope,
            mode=augmentation.bindings[0].mode,
            status="manual",
            reasons=tuple(reasons),
            command=None,
        )
    binding = cli_bindings[0]
    blocking_reasons = _binding_blocking_reasons(binding, project)
    if blocking_reasons:
        return _PlannedAugmentation(
            ref=augmentation.ref,
            scope=augmentation.scope,
            mode=binding.mode,
            status="blocked",
            reasons=blocking_reasons,
            command=None,
        )
    manual_reasons = _binding_manual_reasons(binding, project)
    command = _project_command(augmentation, binding)
    if manual_reasons:
        return _PlannedAugmentation(
            ref=augmentation.ref,
            scope=augmentation.scope,
            mode=binding.mode,
            status="manual",
            reasons=manual_reasons,
            command=command,
        )
    return _PlannedAugmentation(
        ref=augmentation.ref,
        scope=augmentation.scope,
        mode=binding.mode,
        status="ready",
        reasons=(
            _ReadinessReason(
                code="requirements_satisfied",
                message="The current project satisfies every declared CLI requirement.",
            ),
        ),
        command=command,
    )


def _binding_blocking_reasons(
    binding: AugmentationBinding,
    project: _ProjectReadiness,
) -> tuple[_ReadinessReason, ...]:
    if project.status != "ready":
        return (_project_blocking_reason(project.status),)
    requirements = binding.requirements
    reasons: list[_ReadinessReason] = []
    if (
        "production interaction" in requirements.required_source_features
        and "production interaction" not in project.available_source_features
    ):
        reasons.extend(project.dataset_reasons)
    if requirements.semantic_model:
        reasons.extend(project.semantic_model_reasons)
    if requirements.environment:
        reasons.extend(project.environment_reasons)
    capabilities = project.environment_capabilities
    if (
        requirements.conversations
        and capabilities is not None
        and not capabilities.supports_conversations
    ):
        reasons.append(
            _ReadinessReason(
                code="environment_conversations_unsupported",
                message="The configured environment does not support conversations.",
            )
        )
    if (
        requirements.state_observation
        and capabilities is not None
        and not capabilities.supports_state_observation
    ):
        reasons.append(
            _ReadinessReason(
                code="environment_state_observation_unsupported",
                message="The configured environment does not support state observation.",
            )
        )
    if requirements.customer_evaluator:
        reasons.extend(project.customer_evaluator_reasons)
    available_capabilities = _environment_capability_references(capabilities)
    for capability in requirements.environment_capabilities:
        if capability not in available_capabilities:
            reasons.append(
                _ReadinessReason(
                    code="environment_capability_missing",
                    message=f"The configured environment does not declare {capability}.",
                )
            )
    return _deduplicate_reasons(reasons)


def _binding_manual_reasons(
    binding: AugmentationBinding,
    project: _ProjectReadiness,
) -> tuple[_ReadinessReason, ...]:
    reasons: list[_ReadinessReason] = []
    manual_source_features = tuple(
        feature
        for feature in binding.requirements.required_source_features
        if feature not in project.available_source_features
    )
    if manual_source_features:
        reasons.append(_manual_source_feature_reason(manual_source_features))
    if binding.execution_owner == "stress_cli":
        reasons.append(
            _ReadinessReason(
                code="stress_case_not_configured",
                message=(
                    "The project does not configure the case and output paths required by this "
                    "stress command."
                ),
            )
        )
    if binding.requirements.human_review:
        reasons.append(
            _ReadinessReason(
                code="human_review_required",
                message="This augmentation requires human review of generated candidates.",
            )
        )
    return tuple(reasons)


def _manual_source_feature_reason(source_features: tuple[str, ...]) -> _ReadinessReason:
    return _ReadinessReason(
        code="source_feature_requires_manual_selection",
        message=f"Select or materialize source data with: {', '.join(source_features)}.",
    )


def _project_blocking_reason(status: _ProjectStatus) -> _ReadinessReason:
    if status == "missing":
        return _ReadinessReason(
            code="project_not_configured",
            message="No UL project was found; run 'ul init' first.",
        )
    return _ReadinessReason(
        code="project_invalid",
        message="The UL project configuration is invalid.",
    )


def _environment_capability_references(
    capabilities: EnvironmentCapabilities | None,
) -> frozenset[str]:
    if capabilities is None or capabilities.timeout_after_commit_version is None:
        return frozenset()
    return frozenset(
        (f"environment.tool.timeout_after_commit@{capabilities.timeout_after_commit_version}",)
    )


def _project_command(
    augmentation: BuiltinAugmentationSpec,
    binding: AugmentationBinding,
) -> str | None:
    if binding.execution_owner == "dataset_cli":
        return f"ul run --operator {augmentation.ref.id}@{augmentation.ref.version}"
    if binding.execution_owner == "stress_cli":
        return None
    return binding.command


def _deduplicate_reasons(
    reasons: list[_ReadinessReason],
) -> tuple[_ReadinessReason, ...]:
    return tuple(dict.fromkeys(reasons))


def _catalog_item(augmentation: BuiltinAugmentationSpec) -> dict[str, object]:
    return {
        "ref": augmentation.ref.model_dump(mode="json"),
        "scope": augmentation.scope,
        "summary": augmentation.summary,
        "bindings": [
            {
                **binding.model_dump(mode="json"),
                "cli_available": binding.cli_available,
            }
            for binding in augmentation.bindings
        ],
        "cli_available": augmentation.cli_available,
    }


def _execution_owner_label(owner: str) -> str:
    return {
        "dataset_cli": "dataset CLI",
        "augmentation_registry": "SDK augmentation registry",
        "stress_cli": "stress CLI",
    }[owner]


def _parse_scope(value: str | None) -> AugmentationScope | None:
    if value is None:
        return None
    if value not in _SCOPES:
        raise typer.BadParameter(
            f"unknown scope; choose one of: {', '.join(_SCOPES)}", param_hint="--scope"
        )
    return value


def _parse_mode(value: str | None) -> AugmentationMode | None:
    if value is None:
        return None
    if value not in _MODES:
        raise typer.BadParameter(
            f"unknown mode; choose one of: {', '.join(_MODES)}", param_hint="--mode"
        )
    return value


def _parse_reference(reference: str) -> tuple[str, str | None]:
    if len(reference) > 251 or _REFERENCE_PATTERN.fullmatch(reference) is None:
        raise typer.BadParameter("augmentation reference must be ID or ID@VERSION")
    if "@" not in reference:
        return reference, None
    augmentation_id, version = reference.rsplit("@", 1)
    if not augmentation_id or not version:
        raise typer.BadParameter("augmentation reference must be ID or ID@VERSION")
    return augmentation_id, version


def _format_requirements(
    source_features: tuple[str, ...],
    *,
    semantic_model: bool,
    environment: bool,
    conversations: bool,
    state_observation: bool,
    customer_evaluator: bool,
    environment_capabilities: tuple[str, ...],
    human_review: bool,
) -> str:
    values = [*source_features]
    values.extend(
        label
        for required, label in (
            (semantic_model, "semantic model"),
            (environment, "test environment"),
            (conversations, "conversation support"),
            (state_observation, "committed-state observation"),
            (customer_evaluator, "customer evaluator"),
            (human_review, "human review"),
        )
        if required
    )
    values.extend(f"environment capability {capability}" for capability in environment_capabilities)
    return ", ".join(values) if values else "none"
