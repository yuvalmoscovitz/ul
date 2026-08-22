from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Never, cast
from uuid import uuid4

import typer
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from rich.console import Console
from ul import DatasetEvaluationMode, load_dataset_invariant_suite, load_redaction_policy
from ul.http_environment import (
    ENVIRONMENT_ID_PLACEHOLDER,
    JsonHttpIsolatedResponseConfig,
    JsonHttpTargetConfig,
    json_http_environment_config_sha256,
    json_http_environment_origin,
    load_json_http_environment_config,
)

from ul_cli.dataset import (
    evaluate_dataset,
    initialize_dataset_environment,
    validate_dataset_operator_ids,
    validate_interaction_dataset,
)
from ul_cli.dataset_review import is_reportable_dataset_evidence
from ul_cli.environment import TEST_ENVIRONMENT_CONFIRMATION_MESSAGE
from ul_cli.report import report_evidence

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

console = Console()

_PROJECT_DIRECTORY = ".ul"
_PROJECT_CONFIG = "config.json"
_PROJECT_CONFIG_LOCK = ".config.json.lock"
_PROJECT_STATE = "state.json"
_MAXIMUM_PROJECT_FILE_BYTES = 1_000_000
DEFAULT_PROJECT_OPERATORS = ("input.surface.rephrase",)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProjectConfig(_StrictModel):
    schema_version: Literal[2] = 2
    dataset: str = Field(min_length=1)
    environment_config: str = Field(min_length=1)
    environment_origin: str = Field(min_length=1)
    environment_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    invariants: str | None = None
    evaluation_mode: DatasetEvaluationMode = "variance"
    redaction_policy: str | None = None
    redaction_policy_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    redaction_state: str | None = None
    save_augmentations: bool = True
    operators: tuple[str, ...] = Field(default=DEFAULT_PROJECT_OPERATORS, min_length=1)
    limit: int = Field(default=3, ge=1, le=100)
    repetitions: int = Field(default=3, ge=1)
    max_environment_api_calls: int = Field(default=120, ge=1)
    allow_environment_network: bool
    confirm_test_environment: bool
    allow_insecure_http: bool = False

    @field_validator("operators", mode="before")
    @classmethod
    def parse_json_operators(cls, value: object) -> object:
        return tuple(cast(list[object], value)) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_redaction_binding(self) -> ProjectConfig:
        redaction_values = (
            self.redaction_policy,
            self.redaction_policy_sha256,
            self.redaction_state,
        )
        if any(value is not None for value in redaction_values) and not all(
            value is not None for value in redaction_values
        ):
            raise ValueError(
                "redaction_policy, redaction_policy_sha256, and redaction_state must be "
                "configured together"
            )
        return self


class ProjectState(_StrictModel):
    schema_version: Literal[1] = 1
    latest_evidence: str = Field(min_length=1)
    latest_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


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
    environment_config: Annotated[
        Path | None,
        typer.Option(
            "--environment-config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Existing customer-managed environment connection config.",
        ),
    ] = None,
    environment_url: Annotated[
        str | None,
        typer.Option(
            "--environment-url",
            help=(
                "Stateful API base URL, or an isolated-response POST URL without credentials, "
                "query, or fragment."
            ),
        ),
    ] = None,
    adapter_tier: Annotated[
        Literal["stateful-lifecycle", "isolated-response"],
        typer.Option(
            help=(
                "Generated adapter evidence tier. Isolated-response uses one endpoint and "
                "cannot verify state or conversations."
            )
        ),
    ] = "stateful-lifecycle",
    confirm_request_isolation: Annotated[
        bool,
        typer.Option(
            help="Attest every isolated-response request starts fresh and cannot affect another."
        ),
    ] = False,
    confirm_safe_test_target: Annotated[
        bool,
        typer.Option(help="Attest this isolated target cannot cause real-world effects."),
    ] = False,
    isolated_preset: Annotated[
        Literal["generic-json", "openai-chat"],
        typer.Option(help="Known JSON shape for an existing isolated agent endpoint."),
    ] = "generic-json",
    environment_id: Annotated[
        str | None,
        typer.Option(help="Stable evidence name; isolated mode defaults to the endpoint host."),
    ] = None,
    fixture_id: Annotated[
        str | None,
        typer.Option(help="Stable stateful fixture name recorded with every run."),
    ] = None,
    fixture_version: Annotated[
        str | None,
        typer.Option(help="Version of the stateful fixture recorded with every run."),
    ] = None,
    request_json_template: Annotated[
        str | None,
        typer.Option(help="JSON request template for an existing isolated endpoint."),
    ] = None,
    response_json_pointer: Annotated[
        str | None,
        typer.Option(help="RFC 6901 pointer to the isolated endpoint's response value."),
    ] = None,
    agent_model: Annotated[
        str | None,
        typer.Option(help="Agent model sent by the openai-chat preset."),
    ] = None,
    header_from_env: Annotated[
        list[str] | None,
        typer.Option(
            "--header-from-env",
            help="HTTP_HEADER=UL_ENVIRONMENT_VARIABLE; repeat as needed.",
        ),
    ] = None,
    invariants: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = None,
    evaluation_mode: Annotated[
        DatasetEvaluationMode,
        typer.Option(
            "--evaluation-mode",
            help=(
                "Evaluation intent. Variance is available now; correctness and preference "
                "evaluators are not implemented."
            ),
        ),
    ] = "variance",
    redaction_policy: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Provider redaction policy JSON; schema and example are in docs/privacy.md.",
        ),
    ] = None,
    redaction_state: Annotated[
        Path | None,
        typer.Option(help="Private pseudonym mapping file used with --redaction-policy."),
    ] = None,
    save_augmentations: Annotated[
        bool,
        typer.Option(
            "--save-augmentations/--no-save-augmentations",
            help="Retain generated augmentations for safe resume.",
        ),
    ] = True,
    operator: Annotated[
        list[str] | None,
        typer.Option("--operator", help="Augmentation ID; repeat as needed."),
    ] = None,
    limit: Annotated[int, typer.Option(min=1, max=100)] = 3,
    repetitions: Annotated[int, typer.Option(min=1)] = 3,
    max_environment_api_calls: Annotated[
        int, typer.Option("--max-environment-api-calls", min=1)
    ] = 120,
    allow_environment_network: Annotated[
        bool,
        typer.Option(
            "--allow-environment-network",
            help="Save permission for UL runs to call this environment API.",
        ),
    ] = False,
    confirm_test_environment: Annotated[
        bool,
        typer.Option(
            "--confirm-test-environment",
            help="Confirm this is a dedicated test environment, not a production target.",
        ),
    ] = False,
    allow_insecure_http: Annotated[
        bool,
        typer.Option(help="Allow a local HTTP environment API."),
    ] = False,
) -> None:
    """Configure this project once for `ul run` and `ul report`."""
    if evaluation_mode != "variance":
        raise typer.BadParameter(
            f"evaluation mode '{evaluation_mode}' is not implemented; use 'variance'. "
            "Historical dataset output is grounding evidence, not an expected answer.",
            param_hint="--evaluation-mode",
        )
    if (environment_config is None) == (environment_url is None):
        raise typer.BadParameter("provide exactly one of --environment-config or --environment-url")
    generated_mapping_options_used = (
        isolated_preset != "generic-json"
        or environment_id is not None
        or fixture_id is not None
        or fixture_version is not None
        or request_json_template is not None
        or response_json_pointer is not None
        or agent_model is not None
        or bool(header_from_env)
    )
    if environment_config is not None and generated_mapping_options_used:
        raise typer.BadParameter(
            "generated adapter options require --environment-url",
            param_hint="--environment-url",
        )
    loaded_environment_config: JsonHttpTargetConfig | None = None
    if environment_config is not None:
        try:
            loaded_environment_config = load_json_http_environment_config(environment_config)
        except (OSError, RuntimeError, ValueError) as error:
            raise typer.BadParameter(str(error), param_hint="--environment-config") from None
    isolated_response_selected = (
        adapter_tier == "isolated-response"
        if environment_url is not None
        else isinstance(loaded_environment_config, JsonHttpIsolatedResponseConfig)
    )
    if not allow_environment_network:
        raise typer.BadParameter(
            "initialization requires --allow-environment-network",
            param_hint="--allow-environment-network",
        )
    if not confirm_test_environment:
        raise typer.BadParameter(
            (
                "UL will send requests to this target. Use a dedicated test target, not a "
                "production system. Re-run with --confirm-test-environment to continue."
                if isolated_response_selected
                else TEST_ENVIRONMENT_CONFIRMATION_MESSAGE
            ),
            param_hint="--confirm-test-environment",
        )
    if isolated_response_selected and not confirm_request_isolation:
        raise typer.BadParameter(
            "isolated-response setup requires --confirm-request-isolation",
            param_hint="--confirm-request-isolation",
        )
    if isolated_response_selected and not confirm_safe_test_target:
        raise typer.BadParameter(
            "isolated-response setup requires --confirm-safe-test-target",
            param_hint="--confirm-safe-test-target",
        )
    if (redaction_policy is None) != (redaction_state is None):
        raise typer.BadParameter(
            "--redaction-policy and --redaction-state must be configured together"
        )

    project_root = Path.cwd().resolve()
    project_directory = project_root / _PROJECT_DIRECTORY
    project_config_path = project_directory / _PROJECT_CONFIG
    if project_config_path.exists():
        raise typer.BadParameter(".ul/config.json already exists; UL will not overwrite it")

    loaded_redaction_policy = None
    try:
        validate_interaction_dataset(dataset)
        selected_operators = validate_dataset_operator_ids(operator)
        if invariants is not None:
            load_dataset_invariant_suite(invariants)
        if redaction_policy is not None:
            loaded_redaction_policy = load_redaction_policy(redaction_policy)
    except ValidationError as error:
        raise typer.BadParameter(_format_validation_error(error)) from None
    except (OSError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from None

    project_directory_created = not project_directory.exists()
    runs_directory = project_directory / "runs"
    runs_directory_created = not runs_directory.exists()
    ignore_file_created = False
    generated_environment_path: Path | None = None
    project_directory_status: os.stat_result | None = None
    project_directory_descriptor: int | None = None
    try:
        _ensure_private_directory(project_directory)
        project_directory_status = project_directory.lstat()
        project_directory_descriptor = _open_private_directory(project_directory)
        _ensure_private_directory(runs_directory)
        _create_private_text(project_directory / ".gitignore", "*\n")
        ignore_file_created = True

        selected_environment_config = environment_config
        if environment_url is not None:
            selected_environment_config = project_directory / "environment.json"
            initialize_dataset_environment(
                selected_environment_config,
                environment_url,
                adapter_tier=adapter_tier,
                confirm_request_isolation=confirm_request_isolation,
                confirm_safe_test_target=confirm_safe_test_target,
                isolated_preset=isolated_preset,
                environment_id=environment_id,
                fixture_id=fixture_id,
                fixture_version=fixture_version,
                request_json_template=request_json_template,
                response_json_pointer=response_json_pointer,
                agent_model=agent_model,
                header_from_env=header_from_env,
                show_guidance=False,
            )
            generated_environment_path = selected_environment_config
        assert selected_environment_config is not None
        if loaded_environment_config is None:
            loaded_environment_config = load_json_http_environment_config(
                selected_environment_config
            )

        config = ProjectConfig(
            dataset=_relative_project_path(dataset, project_root),
            environment_config=_relative_project_path(selected_environment_config, project_root),
            environment_origin=json_http_environment_origin(loaded_environment_config),
            environment_config_sha256=json_http_environment_config_sha256(
                loaded_environment_config
            ),
            invariants=(
                _relative_project_path(invariants, project_root) if invariants is not None else None
            ),
            evaluation_mode=evaluation_mode,
            redaction_policy=(
                _relative_project_path(redaction_policy, project_root)
                if redaction_policy is not None
                else None
            ),
            redaction_policy_sha256=(
                loaded_redaction_policy.digest if loaded_redaction_policy is not None else None
            ),
            redaction_state=(
                _relative_project_path(redaction_state, project_root)
                if redaction_state is not None
                else None
            ),
            save_augmentations=save_augmentations,
            operators=selected_operators,
            limit=limit,
            repetitions=repetitions,
            max_environment_api_calls=max_environment_api_calls,
            allow_environment_network=allow_environment_network,
            confirm_test_environment=confirm_test_environment,
            allow_insecure_http=allow_insecure_http,
        )
        _create_private_json(project_config_path, config.model_dump(mode="json"))
    except typer.BadParameter:
        _discard_incomplete_project(
            project_directory,
            runs_directory=runs_directory,
            generated_environment_path=generated_environment_path,
            project_directory_created=project_directory_created,
            runs_directory_created=runs_directory_created,
            ignore_file_created=ignore_file_created,
            project_directory_status=project_directory_status,
            project_directory_descriptor=project_directory_descriptor,
        )
        _close_descriptor(project_directory_descriptor)
        raise
    except ValidationError as error:
        _discard_incomplete_project(
            project_directory,
            runs_directory=runs_directory,
            generated_environment_path=generated_environment_path,
            project_directory_created=project_directory_created,
            runs_directory_created=runs_directory_created,
            ignore_file_created=ignore_file_created,
            project_directory_status=project_directory_status,
            project_directory_descriptor=project_directory_descriptor,
        )
        _close_descriptor(project_directory_descriptor)
        raise typer.BadParameter(
            f"cannot create UL project: {_format_validation_error(error)}"
        ) from None
    except (OSError, RuntimeError, ValueError) as error:
        _discard_incomplete_project(
            project_directory,
            runs_directory=runs_directory,
            generated_environment_path=generated_environment_path,
            project_directory_created=project_directory_created,
            runs_directory_created=runs_directory_created,
            ignore_file_created=ignore_file_created,
            project_directory_status=project_directory_status,
            project_directory_descriptor=project_directory_descriptor,
        )
        _close_descriptor(project_directory_descriptor)
        raise typer.BadParameter(f"cannot create UL project: {error}") from None
    except BaseException:
        _discard_incomplete_project(
            project_directory,
            runs_directory=runs_directory,
            generated_environment_path=generated_environment_path,
            project_directory_created=project_directory_created,
            runs_directory_created=runs_directory_created,
            ignore_file_created=ignore_file_created,
            project_directory_status=project_directory_status,
            project_directory_descriptor=project_directory_descriptor,
        )
        _close_descriptor(project_directory_descriptor)
        raise
    _close_descriptor(project_directory_descriptor)
    console.print(f"Configured UL project: {project_config_path}")
    assert loaded_environment_config is not None
    if loaded_environment_config.environment_id == ENVIRONMENT_ID_PLACEHOLDER:
        console.print(
            f"Before running: replace environment_id '{ENVIRONMENT_ID_PLACEHOLDER}' in "
            f"{selected_environment_config} with a stable name for this test environment."
        )
    if environment_url is not None:
        if isinstance(loaded_environment_config, JsonHttpIsolatedResponseConfig):
            console.print(
                "Existing one-request JSON endpoint configured; no UL-specific endpoint is "
                "required. Set any configured UL_ENVIRONMENT_* variables. This tier records "
                "response evidence only; it cannot verify committed state, conversations, or "
                "state-dependent stress tests."
            )
        else:
            console.print(
                "Before running: implement the reset, execute-turn, and snapshot contract "
                "generated in .ul/environment.json."
            )
        console.print("Then verify it with 'ul environment check .ul/environment.json --help'.")
    elif isinstance(loaded_environment_config, JsonHttpIsolatedResponseConfig):
        console.print(
            "Adapter limitation: isolated-response records response evidence only. It does not "
            "verify committed state, cleanup, conversations, or state-dependent stress tests."
        )
    console.print("Next: set semantic-provider credentials and UL_LIVE=true, then run 'ul run'.")


def run_project(
    dry_run: Annotated[
        bool,
        typer.Option(help="Validate and show the execution plan without external calls."),
    ] = False,
    limit: Annotated[int | None, typer.Option(min=1, max=100)] = None,
    repetitions: Annotated[int | None, typer.Option(min=1)] = None,
    max_environment_api_calls: Annotated[
        int | None, typer.Option("--max-environment-api-calls", min=1)
    ] = None,
    operator: Annotated[
        list[str] | None,
        typer.Option("--operator", help="Override configured augmentations; repeat as needed."),
    ] = None,
    save_augmentations: Annotated[
        bool | None,
        typer.Option(
            "--save-augmentations/--no-save-augmentations",
            help="Override augmentation retention for this run.",
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(help="Resume the latest run with evidence instead of starting a new run."),
    ] = False,
) -> None:
    """Run the configured project, with optional one-run overrides."""
    project_root, config = load_project()
    try:
        output = (
            _load_latest_evidence(project_root)
            if resume
            else project_root / _PROJECT_DIRECTORY / "runs" / _new_evidence_name()
        )
    except (OSError, ValueError, ValidationError):
        raise typer.BadParameter(
            "cannot resume: latest evidence is missing or changed; start a new run"
        ) from None
    selected_operators = operator if operator is not None else list(config.operators)

    try:
        evaluate_dataset(
            data=resolve_project_path(config.dataset, project_root),
            environment_config=resolve_project_path(config.environment_config, project_root),
            output=output,
            augmentations_output=None,
            no_save_augmentations=not (
                config.save_augmentations if save_augmentations is None else save_augmentations
            ),
            invariants=(
                resolve_project_path(config.invariants, project_root)
                if config.invariants is not None
                else None
            ),
            evaluation_mode=config.evaluation_mode,
            operator=selected_operators,
            limit=limit if limit is not None else config.limit,
            repetitions=repetitions if repetitions is not None else config.repetitions,
            max_environment_api_calls=(
                max_environment_api_calls
                if max_environment_api_calls is not None
                else config.max_environment_api_calls
            ),
            allow_environment_network=config.allow_environment_network,
            confirm_test_environment=config.confirm_test_environment,
            allow_insecure_http=config.allow_insecure_http,
            dry_run=dry_run,
            resume=output if resume else None,
            redaction_policy=(
                resolve_project_path(config.redaction_policy, project_root)
                if config.redaction_policy is not None
                else None
            ),
            redaction_state=(
                resolve_project_path(config.redaction_state, project_root)
                if config.redaction_state is not None
                else None
            ),
            expected_environment_origin=config.environment_origin,
            expected_environment_config_sha256=config.environment_config_sha256,
            expected_redaction_policy_sha256=config.redaction_policy_sha256,
            show_report_guidance=False,
        )
    except BaseException:
        if _has_nonempty_evidence(output):
            _save_latest_evidence(project_root, output)
            console.print("Next: ul report")
        raise
    if _has_nonempty_evidence(output):
        _save_latest_evidence(project_root, output)
        console.print("Next: ul report")


def report_project(
    evidence: Annotated[
        Path | None,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="UL evidence; defaults to the latest ul run with evidence.",
        ),
    ] = None,
    reviews: Annotated[Path | None, typer.Option(help="Review JSONL sidecar.")] = None,
    show_sensitive_values: Annotated[bool, typer.Option()] = False,
    finding: Annotated[str | None, typer.Option("--finding")] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit stable non-sensitive JSON.")
    ] = False,
) -> None:
    """Report findings from supported explicit or latest run evidence."""
    selected_evidence = evidence
    if selected_evidence is None:
        project_root, _ = load_project()
        try:
            selected_evidence = _load_latest_evidence(project_root)
        except (OSError, ValueError, ValidationError) as error:
            raise typer.BadParameter(
                "no run evidence found; run 'ul run' first or pass EVIDENCE"
            ) from error
    report_evidence(
        selected_evidence,
        reviews=reviews,
        show_sensitive_values=show_sensitive_values,
        finding=finding,
        json_output=json_output,
    )


def load_project() -> tuple[Path, ProjectConfig]:
    for candidate_root in (Path.cwd(), *Path.cwd().parents):
        config_path = candidate_root / _PROJECT_DIRECTORY / _PROJECT_CONFIG
        if not config_path.exists() and not config_path.is_symlink():
            continue
        try:
            project_directory = candidate_root / _PROJECT_DIRECTORY
            project_descriptor = _open_private_directory(project_directory)
            try:
                runs_descriptor = _open_private_directory_at(
                    project_descriptor, "runs", project_directory / "runs"
                )
                os.close(runs_descriptor)
                config = ProjectConfig.model_validate(
                    _read_private_json_at(project_descriptor, _PROJECT_CONFIG, config_path)
                )
            finally:
                os.close(project_descriptor)
            return candidate_root, config
        except ValidationError as error:
            raise typer.BadParameter(
                f"invalid UL project config: {_format_validation_error(error)}"
            ) from None
        except (OSError, ValueError) as error:
            raise typer.BadParameter(f"invalid UL project config: {error}") from None
    raise typer.BadParameter("no UL project found; run 'ul init' first")


def update_project_config(
    project_root: Path,
    update: Callable[[ProjectConfig], ProjectConfig],
) -> ProjectConfig:
    """Update a project config under an exclusive lock and atomically replace it."""
    project_directory = project_root / _PROJECT_DIRECTORY
    project_descriptor = _open_private_directory(project_directory)
    lock_descriptor: int | None = None
    config_locked = False
    try:
        lock_descriptor = _open_project_config_lock(project_directory)
        _lock_descriptor(lock_descriptor)
        config_locked = True
        config = ProjectConfig.model_validate(
            _read_private_json_at(
                project_descriptor,
                _PROJECT_CONFIG,
                project_directory / _PROJECT_CONFIG,
            )
        )
        updated_config = update(config)
        if updated_config != config:
            _replace_private_json(
                project_directory / _PROJECT_CONFIG,
                updated_config.model_dump(mode="json"),
            )
        return updated_config
    finally:
        if lock_descriptor is not None:
            if config_locked:
                _unlock_descriptor(lock_descriptor)
            os.close(lock_descriptor)
        os.close(project_descriptor)


def _open_project_config_lock(project_directory: Path) -> int:
    lock_path = project_directory / _PROJECT_CONFIG_LOCK
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    while True:
        try:
            path_status = os.lstat(lock_path)
        except FileNotFoundError:
            try:
                descriptor = os.open(
                    lock_path,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | no_follow_flag,
                    0o600,
                )
                path_status = os.lstat(lock_path)
                break
            except FileExistsError:
                continue
        if not stat.S_ISREG(path_status.st_mode):
            raise OSError("project config lock must be a regular private file")
        descriptor = os.open(lock_path, os.O_RDWR | no_follow_flag)
        break
    try:
        descriptor_status = os.fstat(descriptor)
        if not os.path.samestat(path_status, descriptor_status):
            raise OSError("project config lock changed while opening")
        if not stat.S_ISREG(descriptor_status.st_mode) or descriptor_status.st_nlink != 1:
            raise OSError("project config lock must be a regular private file")
        if sys.platform != "win32" and stat.S_IMODE(descriptor_status.st_mode) & 0o077:
            raise OSError("project config lock permissions must not allow group or other access")
        if hasattr(os, "getuid") and descriptor_status.st_uid != os.getuid():
            raise OSError("project config lock must be owned by the current user")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _lock_descriptor(descriptor: int) -> None:
    if sys.platform == "win32":
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
    else:
        fcntl.flock(descriptor, fcntl.LOCK_EX)


def _unlock_descriptor(descriptor: int) -> None:
    if sys.platform == "win32":
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(descriptor, fcntl.LOCK_UN)


def _read_private_json(path: Path) -> object:
    descriptor = _open_private_regular_file(path)
    return _read_private_json_descriptor(descriptor)


def _read_private_json_at(directory_descriptor: int, name: str, path: Path) -> object:
    if sys.platform == "win32":
        return _read_private_json(path)
    descriptor = _open_private_regular_file_at(directory_descriptor, name)
    return _read_private_json_descriptor(descriptor)


def _read_private_json_descriptor(descriptor: int) -> object:
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
    try:
        return json.loads(
            encoded,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except RecursionError:
        raise ValueError("project file exceeds the nesting limit") from None


def _open_private_regular_file(path: Path) -> int:
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    requires_identity_check = no_follow_flag == 0
    initial_path_status: os.stat_result | None = None
    if requires_identity_check:
        initial_path_status = os.lstat(path)
        if stat.S_ISLNK(initial_path_status.st_mode):
            raise OSError("project file must not be a symbolic link")

    binary_flag = os.O_BINARY if sys.platform == "win32" else 0
    descriptor = os.open(path, os.O_RDONLY | no_follow_flag | binary_flag)
    try:
        descriptor_status = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_status.st_mode):
            raise OSError("project file must be a regular file")
        if descriptor_status.st_nlink != 1:
            raise OSError("project file must not have multiple hard links")
        if sys.platform != "win32" and stat.S_IMODE(descriptor_status.st_mode) & 0o077:
            raise OSError("project file permissions must not allow group or other access")
        if hasattr(os, "getuid") and descriptor_status.st_uid != os.getuid():
            raise OSError("project file must be owned by the current user")
        if requires_identity_check:
            assert initial_path_status is not None
            current_path_status = os.lstat(path)
            if (
                stat.S_ISLNK(current_path_status.st_mode)
                or not os.path.samestat(initial_path_status, descriptor_status)
                or not os.path.samestat(current_path_status, descriptor_status)
            ):
                raise OSError("project file changed while it was opened")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_private_regular_file_at(directory_descriptor: int, name: str) -> int:
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    binary_flag = os.O_BINARY if sys.platform == "win32" else 0
    descriptor = os.open(
        name,
        os.O_RDONLY | no_follow_flag | binary_flag,
        dir_fd=directory_descriptor,
    )
    try:
        descriptor_status = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_status.st_mode):
            raise OSError("project file must be a regular file")
        if descriptor_status.st_nlink != 1:
            raise OSError("project file must not have multiple hard links")
        if sys.platform != "win32" and stat.S_IMODE(descriptor_status.st_mode) & 0o077:
            raise OSError("project file permissions must not allow group or other access")
        if hasattr(os, "getuid") and descriptor_status.st_uid != os.getuid():
            raise OSError("project file must be owned by the current user")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _create_private_json(path: Path, value: object) -> None:
    _create_private_text(path, json.dumps(value, indent=2) + "\n")


def _create_private_text(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    created_status = os.fstat(descriptor)
    try:
        if sys.platform != "win32":
            os.fchmod(descriptor, 0o600)
        output = os.fdopen(descriptor, "w", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        _unlink_if_same(path, created_status)
        raise
    try:
        with output:
            output.write(value)
    except BaseException:
        _unlink_if_same(path, created_status)
        raise


def _replace_private_json(path: Path, value: object) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        _create_private_json(temporary_path, value)
        os.replace(temporary_path, path)
    finally:
        with suppress(FileNotFoundError):
            temporary_path.unlink()


def _save_latest_evidence(project_root: Path, evidence: Path) -> None:
    state = ProjectState(
        latest_evidence=_relative_project_path(evidence, project_root),
        latest_evidence_sha256=_private_file_sha256(evidence),
    )
    _replace_private_json(
        project_root / _PROJECT_DIRECTORY / _PROJECT_STATE,
        state.model_dump(mode="json"),
    )


def _has_nonempty_evidence(evidence: Path) -> bool:
    return is_reportable_dataset_evidence(evidence)


def _load_latest_evidence(project_root: Path) -> Path:
    state_path = project_root / _PROJECT_DIRECTORY / _PROJECT_STATE
    state = ProjectState.model_validate(_read_private_json(state_path))
    evidence = resolve_project_path(state.latest_evidence, project_root)
    if _private_file_sha256(evidence) != state.latest_evidence_sha256:
        raise ValueError("latest evidence changed after it was recorded")
    return evidence


def _private_file_sha256(path: Path) -> str:
    descriptor = _open_private_regular_file(path)
    digest = hashlib.sha256()
    try:
        while chunk := os.read(descriptor, 65_536):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _relative_project_path(path: Path, project_root: Path) -> str:
    return os.path.relpath(path.resolve(), project_root)


def resolve_project_path(path: str, project_root: Path) -> Path:
    return Path(os.path.abspath(project_root / path))


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


def _open_private_directory(path: Path) -> int:
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, os.O_RDONLY | no_follow_flag | directory_flag)
    try:
        _validate_private_directory_descriptor(descriptor)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_private_directory_at(parent_descriptor: int, name: str, path: Path) -> int:
    if sys.platform == "win32":
        return _open_private_directory(path)
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(
        name,
        os.O_RDONLY | no_follow_flag | directory_flag,
        dir_fd=parent_descriptor,
    )
    try:
        _validate_private_directory_descriptor(descriptor)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validate_private_directory_descriptor(descriptor: int) -> None:
    directory_status = os.fstat(descriptor)
    if not stat.S_ISDIR(directory_status.st_mode):
        raise OSError("project directory must be a directory")
    if sys.platform != "win32" and stat.S_IMODE(directory_status.st_mode) & 0o077:
        raise OSError("project directory permissions must not allow group or other access")
    if hasattr(os, "getuid") and directory_status.st_uid != os.getuid():
        raise OSError("project directory must be owned by the current user")


def _require_same_path(path: Path, expected_status: os.stat_result) -> None:
    current_status = path.lstat()
    if stat.S_ISLNK(current_status.st_mode) or not os.path.samestat(
        current_status, expected_status
    ):
        raise OSError(f"{path} changed while the project was loaded")


def _unlink_if_same(path: Path, expected_status: os.stat_result) -> None:
    try:
        current_status = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISLNK(current_status.st_mode) and os.path.samestat(
        current_status, expected_status
    ):
        path.unlink()


def _discard_incomplete_project(
    project_directory: Path,
    *,
    runs_directory: Path,
    generated_environment_path: Path | None,
    project_directory_created: bool,
    runs_directory_created: bool,
    ignore_file_created: bool,
    project_directory_status: os.stat_result | None,
    project_directory_descriptor: int | None,
) -> None:
    if project_directory_status is None or project_directory_descriptor is None:
        return
    if sys.platform == "win32":
        try:
            _require_same_path(project_directory, project_directory_status)
        except (FileNotFoundError, OSError):
            return
        if generated_environment_path is not None:
            with suppress(FileNotFoundError):
                generated_environment_path.unlink()
        if ignore_file_created:
            with suppress(FileNotFoundError):
                (project_directory / ".gitignore").unlink()
        if runs_directory_created:
            with suppress(OSError):
                runs_directory.rmdir()
        if project_directory_created:
            with suppress(OSError):
                project_directory.rmdir()
        return
    if generated_environment_path is not None:
        with suppress(FileNotFoundError):
            os.unlink(generated_environment_path.name, dir_fd=project_directory_descriptor)
    if ignore_file_created:
        with suppress(FileNotFoundError):
            os.unlink(".gitignore", dir_fd=project_directory_descriptor)
    if runs_directory_created:
        with suppress(OSError):
            os.rmdir(runs_directory.name, dir_fd=project_directory_descriptor)
    if project_directory_created:
        try:
            _require_same_path(project_directory, project_directory_status)
        except (FileNotFoundError, OSError):
            return
        with suppress(OSError):
            project_directory.rmdir()


def _close_descriptor(descriptor: int | None) -> None:
    if descriptor is not None:
        os.close(descriptor)


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("project file contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> Never:
    raise ValueError(f"project file contains invalid JSON constant {value}")


def _format_validation_error(error: ValidationError) -> str:
    reasons: list[str] = []
    for issue in error.errors(include_url=False, include_context=False, include_input=False):
        field_path = ".".join(str(part) for part in issue["loc"])
        message = str(issue["msg"]).removeprefix("Value error, ")
        reasons.append(f"{field_path}: {message}" if field_path else message)
    return "; ".join(reasons) or "validation failed"
