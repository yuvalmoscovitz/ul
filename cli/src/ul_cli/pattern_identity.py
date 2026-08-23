from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import stat
from contextlib import suppress
from pathlib import Path

PATTERN_IDENTITY_KEY_FILENAME = "pattern-identity.key"
_PATTERN_IDENTITY_KEY_BYTES = 32


class PatternIdentityKeyError(ValueError):
    pass


def create_project_pattern_identity_key(project_directory: Path) -> None:
    _validate_private_project_directory(project_directory)
    path = project_directory / PATTERN_IDENTITY_KEY_FILENAME
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow_flag,
        0o600,
    )
    created_status = os.fstat(descriptor)
    try:
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        key = secrets.token_bytes(_PATTERN_IDENTITY_KEY_BYTES)
        remaining = memoryview(key)
        while remaining:
            written = os.write(descriptor, remaining)
            if written == 0:
                raise OSError("pattern identity key write was incomplete")
            remaining = remaining[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        _unlink_if_same(path, created_status)
        raise
    os.close(descriptor)


def ensure_project_pattern_identity_key(project_directory: Path) -> bytes:
    with suppress(FileExistsError):
        create_project_pattern_identity_key(project_directory)
    return _read_private_key(
        project_directory,
        project_directory / PATTERN_IDENTITY_KEY_FILENAME,
    )


def load_pattern_identity_key(evidence: Path) -> bytes:
    checked: set[Path] = set()
    search_roots = (evidence.absolute().parent, Path.cwd().absolute())
    for search_root in search_roots:
        for candidate in (search_root, *search_root.parents):
            project_directory = candidate if candidate.name == ".ul" else candidate / ".ul"
            if project_directory in checked:
                continue
            checked.add(project_directory)
            key_path = project_directory / PATTERN_IDENTITY_KEY_FILENAME
            if key_path.exists() or key_path.is_symlink():
                return _read_private_key(project_directory, key_path)
    raise PatternIdentityKeyError(
        "project pattern identity key not found; run 'ul init' in this project"
    )


def pattern_mechanism_pseudonym(key: bytes, private_mechanism_digest: str) -> str:
    if len(key) != _PATTERN_IDENTITY_KEY_BYTES:
        raise PatternIdentityKeyError("project pattern identity key must contain 32 bytes")
    if len(private_mechanism_digest) != 64 or any(
        character not in "0123456789abcdef" for character in private_mechanism_digest
    ):
        raise ValueError("private mechanism digest must be a lowercase SHA-256 digest")
    digest = hmac.new(
        key,
        b"ul.pattern-mechanism.v1\0" + bytes.fromhex(private_mechanism_digest),
        hashlib.sha256,
    ).hexdigest()
    return f"ulpm_v1_{digest}"


def pattern_evidence_reference(key: bytes, evidence_record_sha256: str) -> str:
    if len(key) != _PATTERN_IDENTITY_KEY_BYTES:
        raise PatternIdentityKeyError("project pattern identity key must contain 32 bytes")
    if len(evidence_record_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in evidence_record_sha256
    ):
        raise ValueError("evidence record digest must be a lowercase SHA-256 digest")
    digest = hmac.new(
        key,
        b"ul.pattern-evidence.v1\0" + bytes.fromhex(evidence_record_sha256),
        hashlib.sha256,
    ).hexdigest()
    return f"ulpe_v1_{digest}"


def _read_private_key(project_directory: Path, path: Path) -> bytes:
    _validate_private_project_directory(project_directory)

    path_status = path.lstat()
    if stat.S_ISLNK(path_status.st_mode) or not stat.S_ISREG(path_status.st_mode):
        raise PatternIdentityKeyError("pattern identity key must be a regular file")
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, os.O_RDONLY | no_follow_flag)
    try:
        descriptor_status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_status.st_mode)
            or descriptor_status.st_nlink != 1
            or not os.path.samestat(path_status, descriptor_status)
        ):
            raise PatternIdentityKeyError("pattern identity key changed while opening")
        if os.name != "nt" and stat.S_IMODE(descriptor_status.st_mode) & 0o077:
            raise PatternIdentityKeyError("pattern identity key permissions must be 0600")
        if hasattr(os, "getuid") and descriptor_status.st_uid != os.getuid():
            raise PatternIdentityKeyError("pattern identity key must be owned by the current user")
        if descriptor_status.st_size != _PATTERN_IDENTITY_KEY_BYTES:
            raise PatternIdentityKeyError("pattern identity key must contain 32 bytes")
        key = os.read(descriptor, _PATTERN_IDENTITY_KEY_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(key) != _PATTERN_IDENTITY_KEY_BYTES:
        raise PatternIdentityKeyError("pattern identity key must contain 32 bytes")
    return key


def _validate_private_project_directory(project_directory: Path) -> None:
    directory_status = project_directory.lstat()
    if not stat.S_ISDIR(directory_status.st_mode) or stat.S_ISLNK(directory_status.st_mode):
        raise PatternIdentityKeyError("UL project directory must be a regular directory")
    if os.name != "nt" and stat.S_IMODE(directory_status.st_mode) & 0o077:
        raise PatternIdentityKeyError("UL project directory permissions must be 0700")
    if hasattr(os, "getuid") and directory_status.st_uid != os.getuid():
        raise PatternIdentityKeyError("UL project directory must be owned by the current user")


def _unlink_if_same(path: Path, expected_status: os.stat_result) -> None:
    try:
        current_status = path.lstat()
        if os.path.samestat(current_status, expected_status):
            path.unlink()
    except FileNotFoundError:
        pass
