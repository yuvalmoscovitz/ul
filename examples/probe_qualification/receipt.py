from __future__ import annotations

import json
import os
import stat
from pathlib import Path


def append_private_receipt(path: Path, value: object) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("qualification receipt is not a regular file")
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as receipt:
            descriptor = -1
            receipt.write(json.dumps({"input": value}, sort_keys=True) + "\n")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
