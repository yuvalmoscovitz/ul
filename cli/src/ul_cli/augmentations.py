from __future__ import annotations

import json
import re
from collections.abc import Callable
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
from ul_core.augmentations.definitions import (
    AugmentationBinding,
    AugmentationMode,
    AugmentationRef,
    AugmentationScope,
    AugmentationSurface,
    BuiltinAugmentationSpec,
    builtin_augmentation_catalog,
)
from ul_core.augmentations.projections import ProjectionContract
from ul_core.evaluation import EnvironmentCapabilities

from ul_cli.dataset import validate_dataset_operator_ids, validate_interaction_dataset
from ul_cli.project import (
    DEFAULT_PROJECT_OPERATORS,
    ProjectConfig,
    load_project,
    resolve_project_path,
    update_project_config,
)

app = typer.Typer(help="Inspect UL's built-in augmentation library.")

_SCOPES: tuple[AugmentationScope, ...] = ("input", "conversation", "environment")
_SURFACES: tuple[AugmentationSurface, ...] = (
    "human_behavior",
    "task_semantics",
    "conversation_workflow",
    "world_business_state",
    "tool_execution",
    "trust_policy_authorization",
)
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


@app.command("guide")
def augmentation_guide() -> None:
    """Show every augmentation grouped by the business area it tests."""
    catalog = builtin_augmentation_catalog()
    for surface in _SURFACES:
        typer.echo(_surface_label(surface))
        for definition in catalog.list(surface=surface):
            typer.echo(
                f"  {definition.ref.id}@{definition.ref.version} "
                f"[{definition.implementation_status}; {definition.qualification_status}]"
            )
            typer.echo(f"    {definition.summary}")
            typer.echo(f"    Expected: {definition.expected_relation}")
        typer.echo()
    typer.echo("Use 'ul augmentations show ID' for applicability and runtime details.")


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
    enabled_references: frozenset[tuple[str, str]]


@dataclass(frozen=True)
class _PlannedAugmentation:
    ref: AugmentationRef
    scope: AugmentationScope
    mode: AugmentationMode
    status: _ReadinessStatus
    reasons: tuple[_ReadinessReason, ...]
    command: str | None
    enabled: bool
    projection: ProjectionContract

    def as_json(self) -> dict[str, object]:
        return {
            "ref": self.ref.model_dump(mode="json"),
            "scope": self.scope,
            "mode": self.mode,
            "status": self.status,
            "reasons": [reason.as_json() for reason in self.reasons],
            "command": self.command,
            "enabled": self.enabled,
            "projection": self.projection.model_dump(mode="json"),
        }


@app.command("list")
def list_augmentations(
    scope: Annotated[str | None, typer.Option(help="Only show one augmentation scope.")] = None,
    surface: Annotated[
        str | None,
        typer.Option(help="Only show one business-risk surface."),
    ] = None,
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
    selected_surface = _parse_surface(surface)
    selected_mode = _parse_mode(mode)
    augmentations = builtin_augmentation_catalog().list(
        scope=selected_scope,
        surface=selected_surface,
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
        typer.echo(
            f"  surface={augmentation.surface} scope={augmentation.scope} "
            f"modes={modes} cli_available={cli_available}"
        )
        typer.echo(f"  applicability={augmentation.applicability_profile}")
        typer.echo(
            f"  implementation={augmentation.implementation_status} "
            f"qualification={augmentation.qualification_status}"
        )
        typer.echo(f"  {augmentation.summary}")
    typer.echo("Use 'ul augmentations show ID[@VERSION]' for requirements and runtime details.")


@app.command("plan")
def plan_augmentations(
    reference: Annotated[
        str | None,
        typer.Argument(help="Optional augmentation ID with optional @VERSION."),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit stable machine-readable JSON.")
    ] = False,
) -> None:
    """Classify built-in augmentations for the current project without external calls."""
    project = _load_project_readiness()
    catalog_augmentations = (
        (_resolve_augmentation(reference),)
        if reference is not None
        else builtin_augmentation_catalog().list()
    )
    augmentations = tuple(
        _plan_augmentation(augmentation, project) for augmentation in catalog_augmentations
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
    elif project.enabled_references:
        typer.echo(
            "Enabled for ul run: "
            + ", ".join(
                f"{item_id}@{version}" for item_id, version in sorted(project.enabled_references)
            )
        )
    for augmentation in augmentations:
        enabled = " [enabled]" if augmentation.enabled else ""
        typer.echo(
            f"{augmentation.status.upper()} "
            f"{augmentation.ref.id}@{augmentation.ref.version}{enabled}"
        )
        if augmentation.status == "blocked":
            typer.echo("  Next steps:")
            for step_number, reason in enumerate(augmentation.reasons, start=1):
                typer.echo(f"  {step_number}. {reason.message}")
        else:
            for reason in augmentation.reasons:
                typer.echo(f"  Reason: {reason.message}")
        if augmentation.command is not None:
            typer.echo(f"  Command: {augmentation.command}")
        typer.echo(
            "  Projection: reads="
            + ",".join(augmentation.projection.reads)
            + " writes="
            + ",".join(augmentation.projection.writes)
        )
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
    typer.echo(f"Surface: {augmentation.surface}")
    typer.echo(f"Scope: {augmentation.scope}")
    typer.echo(f"Applicability: {augmentation.applicability_profile}")
    typer.echo(f"Applicability rule: {augmentation.applicability_rule}")
    typer.echo(f"CLI execution available: {'yes' if augmentation.cli_available else 'no'}")
    typer.echo(f"Expected relation: {augmentation.expected_relation}")
    typer.echo(f"Implementation: {augmentation.implementation_status}")
    typer.echo(f"Qualification: {augmentation.qualification_status}")
    for binding in augmentation.bindings:
        typer.echo(f"Mode: {binding.mode}")
        typer.echo(f"  Stages: {', '.join(binding.stages)}")
        typer.echo(f"  Execution owner: {_execution_owner_label(binding.execution_owner)}")
        typer.echo(f"  Runtime: {binding.runtime}")
        typer.echo(f"  CLI command: {binding.command or 'unavailable'}")
        typer.echo(
            "  Projection: reads="
            + ",".join(binding.projection.reads)
            + " writes="
            + ",".join(binding.projection.writes)
        )
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
    typer.echo("This command inspects metadata. It does not execute the augmentation.")


@app.command("enabled")
def enabled_augmentations(
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit stable machine-readable JSON.")
    ] = False,
) -> None:
    """Show augmentations enabled for `ul run` in the current project."""
    _, config = load_project()
    augmentations = _configured_augmentations(config)
    project = _load_project_readiness()
    planned = tuple(_plan_augmentation(augmentation, project) for augmentation in augmentations)
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "augmentations": [augmentation.as_json() for augmentation in planned],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    typer.echo("Enabled augmentations")
    for augmentation in planned:
        typer.echo(f"{augmentation.ref.id}@{augmentation.ref.version} status={augmentation.status}")
    typer.echo("Use 'ul augmentations plan ID' for required data and configuration steps.")


@app.command("add")
@app.command("enable")
def enable_augmentation(
    reference: Annotated[str, typer.Argument(help="Augmentation ID with optional @VERSION.")],
) -> None:
    """Enable a built-in dataset augmentation for `ul run`."""
    augmentation = _resolve_augmentation(reference)
    _require_project_configurable(augmentation)
    project_root, _ = load_project()
    identity = (augmentation.ref.id, augmentation.ref.version)
    stored_reference = (
        f"{augmentation.ref.id}@{augmentation.ref.version}"
        if "@" in reference
        else augmentation.ref.id
    )

    def add_operator(config: ProjectConfig) -> tuple[str, ...]:
        configured = _configured_augmentations(config)
        if identity in {(item.ref.id, item.ref.version) for item in configured}:
            return config.operators
        return (*config.operators, stored_reference)

    if not _update_operators(project_root, add_operator):
        typer.echo(f"Already enabled: {augmentation.ref.id}@{augmentation.ref.version}")
        return
    typer.echo(f"Enabled: {augmentation.ref.id}@{augmentation.ref.version}")
    _print_selected_readiness(augmentation)


@app.command("remove")
@app.command("disable")
def disable_augmentation(
    reference: Annotated[str, typer.Argument(help="Augmentation ID with optional @VERSION.")],
) -> None:
    """Disable a built-in dataset augmentation for `ul run`."""
    augmentation = _resolve_augmentation(reference)
    _require_project_configurable(augmentation)
    project_root, _ = load_project()
    identity = (augmentation.ref.id, augmentation.ref.version)

    def remove_operator(config: ProjectConfig) -> tuple[str, ...]:
        configured = _configured_augmentations(config)
        remaining = tuple(
            stored_reference
            for stored_reference, item in zip(config.operators, configured, strict=True)
            if (item.ref.id, item.ref.version) != identity
        )
        if not remaining:
            raise typer.BadParameter(
                "at least one augmentation must remain enabled; enable another or run "
                "'ul augmentations reset'"
            )
        return remaining

    if not _update_operators(project_root, remove_operator):
        typer.echo(f"Already disabled: {augmentation.ref.id}@{augmentation.ref.version}")
        return
    typer.echo(f"Disabled: {augmentation.ref.id}@{augmentation.ref.version}")


@app.command("reset")
def reset_augmentations() -> None:
    """Restore the recommended project augmentation defaults."""
    project_root, _ = load_project()

    def reset_operators(config: ProjectConfig) -> tuple[str, ...]:
        _configured_augmentations(config)
        return DEFAULT_PROJECT_OPERATORS

    _update_operators(project_root, reset_operators)
    typer.echo("Restored recommended defaults: input.surface.rephrase@1.0.0")
    _print_selected_readiness(builtin_augmentation_catalog().get("input.surface.rephrase"))


def _configured_augmentations(config: ProjectConfig) -> tuple[BuiltinAugmentationSpec, ...]:
    try:
        validate_dataset_operator_ids(list(config.operators))
    except ValueError as error:
        raise typer.BadParameter(f"invalid configured augmentations: {error}") from None
    catalog = builtin_augmentation_catalog()
    configured: list[BuiltinAugmentationSpec] = []
    for reference in config.operators:
        augmentation_id, separator, version = reference.partition("@")
        try:
            configured.append(catalog.get(augmentation_id, version if separator else None))
        except KeyError:
            raise typer.BadParameter(
                "invalid configured augmentations: unknown augmentation operator reference"
            ) from None
    return tuple(configured)


def _resolve_augmentation(reference: str) -> BuiltinAugmentationSpec:
    augmentation_id, version = _parse_reference(reference)
    try:
        return builtin_augmentation_catalog().get(augmentation_id, version)
    except KeyError:
        raise typer.BadParameter(
            "unknown augmentation; use ID or ID@VERSION from 'ul augmentations list'",
            param_hint="REFERENCE",
        ) from None


def _require_project_configurable(augmentation: BuiltinAugmentationSpec) -> None:
    if any(binding.execution_owner == "dataset_cli" for binding in augmentation.bindings):
        return
    reference = f"{augmentation.ref.id}@{augmentation.ref.version}"
    raise typer.BadParameter(
        f"{reference} cannot be enabled for 'ul run'; run "
        f"'ul augmentations plan {reference}' for its required data, configuration, and command"
    )


def _update_operators(
    project_root: Path,
    update: Callable[[ProjectConfig], tuple[str, ...]],
) -> bool:
    changed = False

    def update_config(config: ProjectConfig) -> ProjectConfig:
        nonlocal changed
        operators = update(config)
        if operators == config.operators:
            return config
        changed = True
        validate_dataset_operator_ids(list(operators))
        return ProjectConfig.model_validate(
            {**config.model_dump(mode="json"), "operators": operators}
        )

    try:
        update_project_config(project_root, update_config)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(f"cannot update augmentation configuration: {error}") from None
    return changed


def _print_selected_readiness(augmentation: BuiltinAugmentationSpec) -> None:
    planned = _plan_augmentation(augmentation, _load_project_readiness())
    if planned.status == "ready":
        typer.echo("Ready for ul run.")
        return
    typer.echo(f"Configured, but currently {planned.status}.")
    typer.echo("Next steps:")
    for step_number, reason in enumerate(planned.reasons, start=1):
        typer.echo(f"  {step_number}. {reason.message}")
    typer.echo(f"Recheck: ul augmentations plan {augmentation.ref.id}@{augmentation.ref.version}")


def _load_project_readiness() -> _ProjectReadiness:
    try:
        project_root, config = load_project()
        configured = _configured_augmentations(config)
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
            enabled_references=frozenset(),
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
        enabled_references=frozenset(
            (augmentation.ref.id, augmentation.ref.version) for augmentation in configured
        ),
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
            enabled=(augmentation.ref.id, augmentation.ref.version) in project.enabled_references,
            projection=augmentation.bindings[0].projection,
        )
    binding = cli_bindings[0]
    command = _project_command(augmentation, binding)
    blocking_reasons = _binding_blocking_reasons(binding, project)
    if blocking_reasons:
        return _PlannedAugmentation(
            ref=augmentation.ref,
            scope=augmentation.scope,
            mode=binding.mode,
            status="blocked",
            reasons=blocking_reasons,
            command=command if binding.execution_owner == "stress_cli" else None,
            enabled=(augmentation.ref.id, augmentation.ref.version) in project.enabled_references,
            projection=binding.projection,
        )
    manual_reasons = _binding_manual_reasons(binding, project)
    if manual_reasons:
        return _PlannedAugmentation(
            ref=augmentation.ref,
            scope=augmentation.scope,
            mode=binding.mode,
            status="manual",
            reasons=manual_reasons,
            command=command,
            enabled=(augmentation.ref.id, augmentation.ref.version) in project.enabled_references,
            projection=binding.projection,
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
        enabled=(augmentation.ref.id, augmentation.ref.version) in project.enabled_references,
        projection=binding.projection,
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
        assert binding.command is not None
        return (
            f"{binding.command} CASE.json --environment-config ENVIRONMENT.json "
            "--invariants INVARIANTS.json --dry-run"
        )
    return binding.command


def _deduplicate_reasons(
    reasons: list[_ReadinessReason],
) -> tuple[_ReadinessReason, ...]:
    return tuple(dict.fromkeys(reasons))


def _catalog_item(augmentation: BuiltinAugmentationSpec) -> dict[str, object]:
    return {
        "ref": augmentation.ref.model_dump(mode="json"),
        "scope": augmentation.scope,
        "surface": augmentation.surface,
        "summary": augmentation.summary,
        "expected_relation": augmentation.expected_relation,
        "applicability_profile": augmentation.applicability_profile,
        "applicability_rule": augmentation.applicability_rule,
        "implementation_status": augmentation.implementation_status,
        "qualification_status": augmentation.qualification_status,
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


def _parse_surface(value: str | None) -> AugmentationSurface | None:
    if value is None:
        return None
    normalized = value.replace("-", "_")
    if normalized not in _SURFACES:
        raise typer.BadParameter(
            f"unknown surface; choose one of: {', '.join(_SURFACES)}",
            param_hint="--surface",
        )
    return normalized


def _surface_label(surface: AugmentationSurface) -> str:
    return {
        "human_behavior": "Human behavior",
        "task_semantics": "Task semantics",
        "conversation_workflow": "Conversation and workflow",
        "world_business_state": "World and business state",
        "tool_execution": "Tool and execution",
        "trust_policy_authorization": "Trust, policy, and authorization",
    }[surface]


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
