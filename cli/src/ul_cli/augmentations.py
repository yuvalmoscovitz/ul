from __future__ import annotations

import json
import re
from typing import Annotated

import typer
from ul_core.augmentation_catalog import (
    AugmentationMode,
    AugmentationScope,
    BuiltinAugmentationSpec,
    builtin_augmentation_catalog,
)

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
