from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Annotated

import typer
from platformdirs import user_state_path
from pydantic import ConfigDict, Field
from ul_core.models import ULModel

_ACTION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32}$")
_MAXIMUM_RECEIPT_BYTES = 16 * 1024
_WINDOWS = sys.platform == "win32"


class _ProgressActionReceipt(ULModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: str = Field(pattern=r"^ul\.progress-action\.v1$")
    argv: tuple[str, ...] = Field(min_length=2, max_length=32)
    working_directory: str = Field(min_length=1)


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


def _validate_action_argv(argv: tuple[str, ...]) -> None:
    allowed_prefixes = (
        ("ul", "dataset", "evaluate"),
        ("ul", "dataset", "report"),
        ("ul", "probe"),
        ("ul", "report"),
    )
    if not any(argv[: len(prefix)] == prefix for prefix in allowed_prefixes):
        raise ValueError("progress actions must use a supported UL command")
    if any("\x00" in argument for argument in argv):
        raise ValueError("progress action arguments cannot contain null bytes")


def _validate_working_directory(working_directory: str) -> None:
    if "\x00" in working_directory or not Path(working_directory).is_absolute():
        raise ValueError("progress action working directory must be an absolute path")


def create_progress_action(argv: tuple[str, ...]) -> tuple[str, ...]:
    _validate_action_argv(argv)
    _directory, directory_descriptor = _open_action_directory()
    try:
        _validate_working_directory(str(Path.cwd().resolve()))
        while True:
            action_id = os.urandom(24).hex()[:32]
            receipt = _ProgressActionReceipt(
                schema_version="ul.progress-action.v1",
                argv=argv,
                working_directory=str(Path.cwd().resolve()),
            )
            encoded = receipt.model_dump_json().encode("utf-8") + b"\n"
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
    try:
        receipt = _ProgressActionReceipt.model_validate_json(encoded)
    except Exception:
        raise ValueError("progress action receipt is invalid") from None
    _validate_action_argv(receipt.argv)
    _validate_working_directory(receipt.working_directory)
    return receipt


def execute_progress_action(
    action_id: Annotated[
        str,
        typer.Argument(help="Opaque action ID emitted by campaign progress."),
    ],
) -> None:
    try:
        receipt = _read_progress_action(action_id)
        completed = subprocess.run(
            (sys.executable, "-m", "ul_cli.main", *receipt.argv[1:]),
            check=False,
            cwd=receipt.working_directory,
        )
    except (OSError, ValueError):
        typer.echo("Unable to resolve the progress action safely.", err=True)
        raise typer.Exit(code=1) from None
    raise typer.Exit(code=completed.returncode)
