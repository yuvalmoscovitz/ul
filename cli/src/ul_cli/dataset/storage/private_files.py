from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from typing import TextIO

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


def create_private_output(path: Path) -> TextIO:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    if sys.platform != "win32":
        os.fchmod(descriptor, 0o600)
    return os.fdopen(descriptor, "w", encoding="utf-8")


def open_private_append_output(path: Path) -> TextIO:
    descriptor = open_resume_descriptor(path, writable=True)
    if sys.platform != "win32":
        os.fchmod(descriptor, 0o600)
    os.lseek(descriptor, 0, os.SEEK_END)
    return os.fdopen(descriptor, "a", encoding="utf-8")


def open_resume_descriptor(path: Path, *, writable: bool) -> int:
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    binary_flag = os.O_BINARY if sys.platform == "win32" else 0
    path_status = os.lstat(path)
    if not stat.S_ISREG(path_status.st_mode):
        raise OSError("resume evidence is not a regular file")
    access_flags = os.O_RDWR | os.O_APPEND if writable else os.O_RDONLY
    descriptor = os.open(path, access_flags | no_follow_flag | binary_flag)
    try:
        descriptor_status = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_status.st_mode) or (
            path_status.st_dev,
            path_status.st_ino,
        ) != (descriptor_status.st_dev, descriptor_status.st_ino):
            raise OSError("resume evidence changed while opening")
        if sys.platform == "win32":
            os.lseek(descriptor, 0, os.SEEK_SET)
            lock_mode = msvcrt.LK_NBLCK if writable else msvcrt.LK_NBRLCK
            msvcrt.locking(descriptor, lock_mode, 1)
        else:
            lock_mode = fcntl.LOCK_EX if writable else fcntl.LOCK_SH
            fcntl.flock(descriptor, lock_mode | fcntl.LOCK_NB)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise
