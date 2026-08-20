from __future__ import annotations

import json
import os
import stat
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Never, cast
from uuid import uuid4

import typer
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from rich.console import Console
from ul import load_dataset_invariant_suite
from ul.http_sandbox import load_json_http_sandbox_config

from ul_cli.dataset import (
    evaluate_dataset,
    initialize_dataset_sandbox,
    validate_dataset_operator_ids,
    validate_interaction_dataset,
)
from ul_cli.dataset_review import report_dataset_evidence

console = Console()

_PROJECT_DIRECTORY = ".ul"
_PROJECT_CONFIG = "config.json"
_PROJECT_STATE = "state.json"
_MAXIMUM_PROJECT_FILE_BYTES = 1_000_000


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProjectConfig(_StrictModel):
    schema_version: Literal[1] = 1
    dataset: str = Field(min_length=1)
    sandbox_config: str = Field(min_length=1)
    invariants: str | None = None
    operators: tuple[str, ...] = Field(default=("input.surface.rephrase",), min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    repetitions: int = Field(default=3, ge=1)
    max_sandbox_api_calls: int = Field(default=100, ge=1)
    allow_sandbox_network_egress: bool
    confirm_isolated_sandbox: bool
    allow_insecure_http: bool = False

    @field_validator("operators", mode="before")
    @classmethod
    def parse_json_operators(cls, value: object) -> object:
        return tuple(cast(list[object], value)) if isinstance(value, list) else value


class ProjectState(_StrictModel):
    schema_version: Literal[1] = 1
    latest_evidence: str = Field(min_length=1)


def initialize_project(
    dataset: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Historical interaction dataset JSONL.",
        ),
    ],
    sandbox_config: Annotated[
        Path | None,
        typer.Option(
            "--sandbox-config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Existing customer-managed sandbox connection config.",
        ),
    ] = None,
    sandbox_url: Annotated[
        str | None,
        typer.Option(
            "--sandbox-url",
            help="Create .ul/sandbox.json for this sandbox API base URL.",
        ),
    ] = None,
    invariants: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = None,
    operator: Annotated[
        list[str] | None,
        typer.Option("--operator", help="Augmentation ID; repeat as needed."),
    ] = None,
    limit: Annotated[int, typer.Option(min=1, max=100)] = 10,
    repetitions: Annotated[int, typer.Option(min=1)] = 3,
    max_sandbox_api_calls: Annotated[int, typer.Option("--max-sandbox-api-calls", min=1)] = 100,
    allow_sandbox_network_egress: Annotated[
        bool,
        typer.Option(
            "--allow-sandbox-network-egress",
            help="Save permission for UL runs to call this sandbox API.",
        ),
    ] = False,
    confirm_isolated_sandbox: Annotated[
        bool,
        typer.Option(
            "--confirm-isolated-sandbox",
            help="Attest once that the endpoint is an isolated non-production sandbox.",
        ),
    ] = False,
    allow_insecure_http: Annotated[
        bool,
        typer.Option(help="Allow a local HTTP sandbox API."),
    ] = False,
) -> None:
    """Configure this project once for `ul run` and `ul report`."""
    if (sandbox_config is None) == (sandbox_url is None):
        raise typer.BadParameter("provide exactly one of --sandbox-config or --sandbox-url")
    if not allow_sandbox_network_egress:
        raise typer.BadParameter(
            "initialization requires --allow-sandbox-network-egress",
            param_hint="--allow-sandbox-network-egress",
        )
    if not confirm_isolated_sandbox:
        raise typer.BadParameter(
            "initialization requires --confirm-isolated-sandbox",
            param_hint="--confirm-isolated-sandbox",
        )

    project_root = Path.cwd().resolve()
    project_directory = project_root / _PROJECT_DIRECTORY
    project_config_path = project_directory / _PROJECT_CONFIG
    if project_config_path.exists():
        raise typer.BadParameter(".ul/config.json already exists; UL will not overwrite it")

    try:
        validate_interaction_dataset(dataset)
        selected_operators = validate_dataset_operator_ids(operator)
        if invariants is not None:
            load_dataset_invariant_suite(invariants)
        if sandbox_config is not None:
            load_json_http_sandbox_config(sandbox_config)
    except (OSError, RuntimeError, ValidationError, ValueError) as error:
        raise typer.BadParameter(str(error)) from None

    try:
        _ensure_private_directory(project_directory)
        runs_directory = project_directory / "runs"
        _ensure_private_directory(runs_directory)

        selected_sandbox_config = sandbox_config
        if sandbox_url is not None:
            selected_sandbox_config = project_directory / "sandbox.json"
            initialize_dataset_sandbox(selected_sandbox_config, sandbox_url, show_guidance=False)
        assert selected_sandbox_config is not None

        config = ProjectConfig(
            dataset=_relative_project_path(dataset, project_root),
            sandbox_config=_relative_project_path(selected_sandbox_config, project_root),
            invariants=(
                _relative_project_path(invariants, project_root) if invariants is not None else None
            ),
            operators=selected_operators,
            limit=limit,
            repetitions=repetitions,
            max_sandbox_api_calls=max_sandbox_api_calls,
            allow_sandbox_network_egress=allow_sandbox_network_egress,
            confirm_isolated_sandbox=confirm_isolated_sandbox,
            allow_insecure_http=allow_insecure_http,
        )
        _create_private_json(project_config_path, config.model_dump(mode="json"))
    except (OSError, ValueError) as error:
        raise typer.BadParameter(f"cannot create UL project: {error}") from None
    console.print(f"Configured UL project: {project_config_path}")
    if sandbox_url is not None:
        console.print(
            "Before running: review .ul/sandbox.json and match its lifecycle requests and "
            "response pointers to your sandbox API."
        )
    console.print("Next: set semantic-provider credentials and UL_LIVE=true, then run 'ul run'.")


def run_project(
    dry_run: Annotated[
        bool,
        typer.Option(help="Validate and show the execution plan without external calls."),
    ] = False,
    limit: Annotated[int | None, typer.Option(min=1, max=100)] = None,
    repetitions: Annotated[int | None, typer.Option(min=1)] = None,
    max_sandbox_api_calls: Annotated[
        int | None, typer.Option("--max-sandbox-api-calls", min=1)
    ] = None,
    operator: Annotated[
        list[str] | None,
        typer.Option("--operator", help="Override configured augmentations; repeat as needed."),
    ] = None,
) -> None:
    """Run the configured project, with optional one-run overrides."""
    project_root, config = _load_project()
    output = project_root / _PROJECT_DIRECTORY / "runs" / _new_evidence_name()
    selected_operators = operator if operator is not None else list(config.operators)

    try:
        evaluate_dataset(
            data=_resolve_project_path(config.dataset, project_root),
            sandbox_config=_resolve_project_path(config.sandbox_config, project_root),
            output=output,
            augmentations_output=None,
            no_save_augmentations=False,
            invariants=(
                _resolve_project_path(config.invariants, project_root)
                if config.invariants is not None
                else None
            ),
            operator=selected_operators,
            limit=limit if limit is not None else config.limit,
            repetitions=repetitions if repetitions is not None else config.repetitions,
            max_sandbox_api_calls=(
                max_sandbox_api_calls
                if max_sandbox_api_calls is not None
                else config.max_sandbox_api_calls
            ),
            allow_sandbox_network_egress=config.allow_sandbox_network_egress,
            confirm_isolated_sandbox=config.confirm_isolated_sandbox,
            allow_insecure_http=config.allow_insecure_http,
            dry_run=dry_run,
            resume=None,
            redaction_policy=None,
            redaction_state=None,
            show_report_guidance=False,
        )
    except typer.Exit as exit_error:
        if exit_error.exit_code in {0, 1} and output.is_file():
            _save_latest_evidence(project_root, output)
            console.print("Next: ul report")
        raise
    if output.is_file():
        _save_latest_evidence(project_root, output)
        console.print("Next: ul report")


def report_project(
    evidence: Annotated[
        Path | None,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Evidence JSONL; defaults to the latest completed ul run.",
        ),
    ] = None,
    reviews: Annotated[Path | None, typer.Option(help="Review JSONL sidecar.")] = None,
    show_sensitive_values: Annotated[bool, typer.Option()] = False,
    finding: Annotated[str | None, typer.Option("--finding")] = None,
) -> None:
    """Report findings from an explicit or latest completed run."""
    selected_evidence = evidence
    if selected_evidence is None:
        project_root, _ = _load_project()
        state_path = project_root / _PROJECT_DIRECTORY / _PROJECT_STATE
        try:
            state = ProjectState.model_validate(_read_private_json(state_path))
        except (OSError, ValueError, ValidationError) as error:
            raise typer.BadParameter(
                "no completed run found; run 'ul run' first or pass EVIDENCE"
            ) from error
        selected_evidence = _resolve_project_path(state.latest_evidence, project_root)
        if not selected_evidence.is_file():
            raise typer.BadParameter(
                "latest evidence no longer exists; run 'ul run' or pass EVIDENCE"
            )
    report_dataset_evidence(
        evidence=selected_evidence,
        reviews=reviews,
        show_sensitive_values=show_sensitive_values,
        sensitive_finding_id=finding,
    )


def _load_project() -> tuple[Path, ProjectConfig]:
    for candidate_root in (Path.cwd(), *Path.cwd().parents):
        config_path = candidate_root / _PROJECT_DIRECTORY / _PROJECT_CONFIG
        if not config_path.exists() and not config_path.is_symlink():
            continue
        try:
            return candidate_root, ProjectConfig.model_validate(_read_private_json(config_path))
        except (OSError, ValueError, ValidationError) as error:
            raise typer.BadParameter(f"invalid UL project config: {error}") from None
    raise typer.BadParameter("no UL project found; run 'ul init' first")


def _read_private_json(path: Path) -> object:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        file_status = os.fstat(descriptor)
        if not stat.S_ISREG(file_status.st_mode):
            raise ValueError("project file must be a regular file, not a symlink")
        if file_status.st_size > _MAXIMUM_PROJECT_FILE_BYTES:
            raise ValueError("project file exceeds the size limit")
        encoded = os.read(descriptor, _MAXIMUM_PROJECT_FILE_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(encoded) > _MAXIMUM_PROJECT_FILE_BYTES:
        raise ValueError("project file exceeds the size limit")
    return json.loads(
        encoded,
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonstandard_json_constant,
    )


def _create_private_json(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def _replace_private_json(path: Path, value: object) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        _create_private_json(temporary_path, value)
        os.replace(temporary_path, path)
    finally:
        with suppress(FileNotFoundError):
            temporary_path.unlink()


def _save_latest_evidence(project_root: Path, evidence: Path) -> None:
    state = ProjectState(latest_evidence=_relative_project_path(evidence, project_root))
    _replace_private_json(
        project_root / _PROJECT_DIRECTORY / _PROJECT_STATE,
        state.model_dump(mode="json"),
    )


def _relative_project_path(path: Path, project_root: Path) -> str:
    return os.path.relpath(path.resolve(), project_root)


def _resolve_project_path(path: str, project_root: Path) -> Path:
    return (project_root / path).resolve()


def _new_evidence_name() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid4().hex[:8]}.jsonl"


def _ensure_private_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        path_status = path.lstat()
        if stat.S_ISLNK(path_status.st_mode) or not stat.S_ISDIR(path_status.st_mode):
            raise ValueError(f"{path} must be a directory, not a symlink") from None
        os.chmod(path, 0o700)


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("project file contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> Never:
    raise ValueError(f"project file contains invalid JSON constant {value}")
