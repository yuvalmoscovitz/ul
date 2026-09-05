from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Annotated, Literal

import typer
from platformdirs import user_state_path
from pydantic import ConfigDict, Field
from ul_core.models import ULModel

_ACTION_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAXIMUM_RECEIPT_BYTES = 16 * 1024
_WINDOWS = sys.platform == "win32"


class _ProgressActionReceipt(ULModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: str = Field(pattern=r"^ul\.progress-action\.v1$")
    action_kind: Literal[
        "dataset_report",
        "dataset_resume",
        "dataset_diagnose",
        "probe_report",
        "probe_resume",
        "probe_diagnose",
    ]
    argv: tuple[str, ...] = Field(min_length=2, max_length=512)
    working_directory: str = Field(min_length=1)
    nonce: str = Field(pattern=r"^[0-9a-f]{32}$")


def _action_receipt_directory() -> Path:
    return user_state_path("ul", appauthor=False) / "progress-actions"


def _validate_private_directory(path: Path) -> None:
    path_status = path.lstat()
    if stat.S_ISLNK(path_status.st_mode) or not stat.S_ISDIR(path_status.st_mode):
        raise OSError("progress action storage must be a private directory")
    if not _WINDOWS and stat.S_IMODE(path_status.st_mode) & 0o077:
        raise OSError("progress action storage permissions must be private")
    if hasattr(os, "getuid") and path_status.st_uid != os.getuid():
        raise OSError("progress action storage must be owned by the current user")


def _open_action_directory() -> tuple[Path, int | None]:
    directory = _action_receipt_directory()
    directory.parent.mkdir(parents=True, exist_ok=True)
    with suppress(FileExistsError):
        directory.mkdir(mode=0o700)
    _validate_private_directory(directory)
    if _WINDOWS:
        return directory, None
    descriptor = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        descriptor_status = os.fstat(descriptor)
        current_status = directory.lstat()
        if not stat.S_ISDIR(descriptor_status.st_mode) or not os.path.samestat(
            current_status, descriptor_status
        ):
            raise OSError("progress action storage changed while opening")
        return directory, descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validate_action_argv(
    action_kind: Literal[
        "dataset_report",
        "dataset_resume",
        "dataset_diagnose",
        "probe_report",
        "probe_resume",
        "probe_diagnose",
    ],
    argv: tuple[str, ...],
) -> None:
    if any("\x00" in argument for argument in argv):
        raise ValueError("progress action arguments cannot contain null bytes")
    fixed_shapes = {
        "dataset_report": ("ul", "dataset", "report"),
        "dataset_resume": ("ul", "dataset", "evaluate", "--resume"),
        "dataset_diagnose": (
            "ul",
            "dataset",
            "evaluate",
            "--resume",
        ),
        "probe_report": ("ul", "report"),
    }
    if action_kind in fixed_shapes:
        prefix = fixed_shapes[action_kind]
        minimum_length = len(prefix) + 1 + (action_kind == "dataset_diagnose")
        if argv[: len(prefix)] != prefix or len(argv) < minimum_length:
            raise ValueError("progress action arguments do not match their action kind")
        if argv[len(prefix)].startswith("--"):
            raise ValueError("progress action path is invalid")
        if action_kind == "dataset_diagnose" and argv[-1] != "--dry-run":
            raise ValueError("dataset diagnosis must be a dry run")
        if action_kind in {"dataset_resume", "dataset_diagnose"}:
            value_options = {"--target", "--confirm-target", "--target-artifact"}
            seen_bound_options: set[str] = set()
            index = len(prefix) + 1
            options_end = len(argv) - (action_kind == "dataset_diagnose")
            while index < options_end:
                option = argv[index]
                if option not in value_options or index + 1 >= options_end:
                    raise ValueError("dataset resume contains an unsupported option")
                if argv[index + 1].startswith("--"):
                    raise ValueError("dataset resume option is missing its value")
                if (
                    option == "--confirm-target"
                    and re.fullmatch(r"[0-9a-f]{64}", argv[index + 1]) is None
                ):
                    raise ValueError("dataset resume target confirmation is invalid")
                if option == "--target-artifact" and not Path(argv[index + 1]).is_absolute():
                    raise ValueError("dataset resume target artifact must be absolute")
                if option != "--target-artifact":
                    if option in seen_bound_options:
                        raise ValueError("dataset resume repeats a required bound option")
                    seen_bound_options.add(option)
                index += 2
            if seen_bound_options not in (set(), {"--target", "--confirm-target"}):
                raise ValueError("dataset resume local target binding is incomplete")
        elif len(argv) != minimum_length:
            raise ValueError("progress action arguments do not match their action kind")
        return
    if action_kind == "probe_resume":
        if argv[:2] != ("ul", "probe") or len(argv) < 13 or argv[2].startswith("--"):
            raise ValueError("probe resume arguments are invalid")
        value_options = {
            "--target",
            "--output",
            "--confirm-target",
            "--confirm-paid-execution",
            "--resume-checkpoint",
            "--resume-checkpoint-sha256",
            "--target-artifact",
            "--diagnostic-artifact",
            "--http-preset",
            "--request-json-template",
            "--response-json-pointer",
            "--agent-model",
            "--header-from-env",
            "--operator",
            "--limit",
            "--repetitions",
            "--target-working-directory",
            "--target-interpreter",
            "--target-environment-variable",
        }
        flag_options = {
            "--allow-insecure-http",
            "--show-smoke-response",
            "--progress-json",
        }
        required = {
            "--target",
            "--output",
            "--confirm-target",
            "--confirm-paid-execution",
            "--resume-checkpoint",
            "--resume-checkpoint-sha256",
        }
        single_value_options = required | {"--limit", "--repetitions"}
        seen_required: set[str] = set()
        seen_single_value_options: set[str] = set()
        index = 3
        while index < len(argv):
            option = argv[index]
            if option in value_options:
                if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                    raise ValueError("probe resume option is missing its value")
                if option in single_value_options:
                    if option in seen_single_value_options:
                        raise ValueError("probe resume repeats a single-value option")
                    seen_single_value_options.add(option)
                if option in required:
                    seen_required.add(option)
                index += 2
                continue
            if option in flag_options:
                index += 1
                continue
            raise ValueError("probe resume contains an unsupported option")
        if seen_required != required:
            raise ValueError("probe resume is missing a required bound option")
        return
    if argv != ("ul", "probe-diagnose"):
        raise ValueError("probe diagnosis must be the read-only built-in action")


def _validate_working_directory(working_directory: str) -> None:
    if "\x00" in working_directory or not Path(working_directory).is_absolute():
        raise ValueError("progress action working directory must be an absolute path")


def create_progress_action(
    action_kind: Literal[
        "dataset_report",
        "dataset_resume",
        "dataset_diagnose",
        "probe_report",
        "probe_resume",
        "probe_diagnose",
    ],
    argv: tuple[str, ...],
) -> tuple[str, ...]:
    _validate_action_argv(action_kind, argv)
    _directory, directory_descriptor = _open_action_directory()
    try:
        _validate_working_directory(str(Path.cwd().resolve()))
        while True:
            receipt = _ProgressActionReceipt(
                schema_version="ul.progress-action.v1",
                action_kind=action_kind,
                argv=argv,
                working_directory=str(Path.cwd().resolve()),
                nonce=os.urandom(16).hex(),
            )
            encoded = receipt.model_dump_json().encode("utf-8") + b"\n"
            action_id = hashlib.sha256(encoded).hexdigest()
            if len(encoded) > _MAXIMUM_RECEIPT_BYTES:
                raise ValueError("progress action receipt exceeds its size limit")
            try:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
                descriptor = (
                    os.open(_directory / f"{action_id}.json", flags, 0o600)
                    if directory_descriptor is None
                    else os.open(
                        f"{action_id}.json",
                        flags,
                        0o600,
                        dir_fd=directory_descriptor,
                    )
                )
            except FileExistsError:
                continue
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise OSError("progress action receipt must be a regular file")
                view = memoryview(encoded)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if directory_descriptor is not None:
                os.fsync(directory_descriptor)
            return ("ul", "action", action_id)
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)


def _read_progress_action(action_id: str) -> _ProgressActionReceipt:
    if _ACTION_ID_PATTERN.fullmatch(action_id) is None:
        raise ValueError("progress action ID is invalid")
    directory, directory_descriptor = _open_action_directory()
    try:
        path = directory / f"{action_id}.json"
        initial_status = path.lstat()
        if stat.S_ISLNK(initial_status.st_mode):
            raise OSError("progress action receipt must not be a symbolic link")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = (
            os.open(path, flags)
            if directory_descriptor is None
            else os.open(
                f"{action_id}.json",
                flags,
                dir_fd=directory_descriptor,
            )
        )
        try:
            descriptor_status = os.fstat(descriptor)
            if (
                not stat.S_ISREG(descriptor_status.st_mode)
                or descriptor_status.st_nlink != 1
                or not os.path.samestat(initial_status, descriptor_status)
            ):
                raise OSError("progress action receipt is not a private regular file")
            if not _WINDOWS and stat.S_IMODE(descriptor_status.st_mode) & 0o077:
                raise OSError("progress action receipt permissions must be private")
            if hasattr(os, "getuid") and descriptor_status.st_uid != os.getuid():
                raise OSError("progress action receipt must be owned by the current user")
            if descriptor_status.st_size > _MAXIMUM_RECEIPT_BYTES:
                raise ValueError("progress action receipt exceeds its size limit")
            encoded = os.read(descriptor, _MAXIMUM_RECEIPT_BYTES + 1)
        finally:
            os.close(descriptor)
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
    if len(encoded) > _MAXIMUM_RECEIPT_BYTES:
        raise ValueError("progress action receipt exceeds its size limit")
    if hashlib.sha256(encoded).hexdigest() != action_id:
        raise ValueError("progress action receipt integrity check failed")
    try:
        receipt = _ProgressActionReceipt.model_validate_json(encoded)
    except Exception:
        raise ValueError("progress action receipt is invalid") from None
    _validate_action_argv(receipt.action_kind, receipt.argv)
    _validate_working_directory(receipt.working_directory)
    return receipt


def execute_progress_action(
    action_id: Annotated[
        str,
        typer.Argument(help="Opaque action ID emitted by campaign progress."),
    ],
    resolve_quarantine_after: Annotated[
        Literal["environment-reset", "environment-replacement"] | None,
        typer.Option(
            help="Attest cleanup before resuming a quarantined probe target.",
        ),
    ] = None,
) -> None:
    try:
        receipt = _read_progress_action(action_id)
        if resolve_quarantine_after is not None and receipt.action_kind != "probe_resume":
            raise ValueError("quarantine attestation is only valid for probe resume actions")
        if receipt.action_kind == "probe_diagnose":
            typer.echo(
                "Probe target calls are stopped. Inspect the private probe diagnostic and "
                "safety state; restart only after an explicit environment reset or replacement."
            )
            return
        private_argv = receipt.argv
        if resolve_quarantine_after is not None:
            private_argv = (
                *private_argv,
                "--resolve-quarantine-after",
                resolve_quarantine_after,
            )
        completed = subprocess.run(
            (sys.executable, "-I", "-m", "ul_cli.main", *private_argv[1:]),
            check=False,
            cwd=receipt.working_directory,
        )
    except (OSError, ValueError):
        typer.echo("Unable to resolve the progress action safely.", err=True)
        raise typer.Exit(code=1) from None
    raise typer.Exit(code=completed.returncode)
