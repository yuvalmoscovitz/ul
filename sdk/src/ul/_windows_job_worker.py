from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
import stat
import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from ctypes import wintypes
from pathlib import Path
from typing import Any, Protocol, cast

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_MAXIMUM_EXECUTABLE_BYTES = 256 * 1024 * 1024


class _CFunction(Protocol):
    argtypes: tuple[object, ...]
    restype: object

    def __call__(self, *arguments: object) -> Any: ...


class _Kernel32(Protocol):
    CreateJobObjectW: _CFunction
    SetInformationJobObject: _CFunction
    AssignProcessToJobObject: _CFunction
    GetCurrentProcess: _CFunction
    CloseHandle: _CFunction


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _create_kill_on_close_job() -> int:
    win_dll = getattr(ctypes, "WinDLL")  # noqa: B009
    get_last_error = getattr(ctypes, "get_last_error")  # noqa: B009
    kernel32 = cast(_Kernel32, win_dll("kernel32", use_last_error=True))
    kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(get_last_error(), "CreateJobObjectW failed")
    information = _JobObjectExtendedLimitInformation()
    information.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        job,
        _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        error = get_last_error()
        kernel32.CloseHandle(job)
        raise OSError(error, "SetInformationJobObject failed")
    if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
        error = get_last_error()
        kernel32.CloseHandle(job)
        raise OSError(error, "AssignProcessToJobObject failed")
    return int(job)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _descriptor_sha256(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


@contextmanager
def _open_executable_identity(
    path: Path,
) -> Generator[tuple[int, str, tuple[int, int, int, int]], None, None]:
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode):
        raise OSError("executable cannot be a symbolic link")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise OSError("executable is not a regular file")
        if descriptor_stat.st_size > _MAXIMUM_EXECUTABLE_BYTES:
            raise OSError("executable exceeds the identity limit")
        identity = _stat_identity(descriptor_stat)
        digest = _descriptor_sha256(descriptor)
        if _stat_identity(os.fstat(descriptor)) != identity:
            raise OSError("executable changed while hashing")
        if _stat_identity(os.lstat(path)) != identity:
            raise OSError("executable path changed while opening")
        yield descriptor, digest, identity
    finally:
        os.close(descriptor)


def _identity_still_matches_path(
    path: Path,
    descriptor: int,
    digest: str,
    identity: tuple[int, int, int, int],
) -> bool:
    try:
        current = os.lstat(path)
        return (
            not stat.S_ISLNK(current.st_mode)
            and _stat_identity(current) == identity
            and _stat_identity(os.fstat(descriptor)) == identity
            and _descriptor_sha256(descriptor) == digest
        )
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--expected-executable-sha256", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command
    if not command or command[0] != "--" or len(command) < 2:
        return 126
    command = command[1:]
    executable = Path(command[0])
    try:
        if not executable.is_absolute() or executable.resolve(strict=True) != executable:
            return 126
        with _open_executable_identity(executable) as (descriptor, digest, identity):
            if digest != arguments.expected_executable_sha256:
                return 126
            _job_handle = _create_kill_on_close_job()
            child = subprocess.Popen(command, shell=False, close_fds=False)
            if not _identity_still_matches_path(
                executable,
                descriptor,
                digest,
                identity,
            ):
                child.kill()
                return 126
            return child.wait()
    except (OSError, ValueError):
        return 126


if __name__ == "__main__":
    raise SystemExit(main())
