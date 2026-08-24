from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
from contextlib import suppress
from pathlib import Path

PATTERN_IDENTITY_KEY_FILENAME = "pattern-identity.key"
REVIEW_HISTORY_KEY_FILENAME = "review-history.key"
_PATTERN_IDENTITY_KEY_BYTES = 32


class PatternIdentityKeyError(ValueError):
    pass


class ReviewHistoryKeyError(ValueError):
    pass


def create_project_pattern_identity_key(project_directory: Path) -> None:
    _create_project_private_key(
        project_directory,
        PATTERN_IDENTITY_KEY_FILENAME,
        "pattern identity key",
    )


def create_project_review_history_key(project_directory: Path) -> None:
    _create_project_private_key(
        project_directory,
        REVIEW_HISTORY_KEY_FILENAME,
        "review history key",
    )


def _create_project_private_key(
    project_directory: Path,
    filename: str,
    label: str,
) -> None:
    _validate_private_project_directory(project_directory)
    path = project_directory / filename
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
                raise OSError(f"{label} write was incomplete")
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
        error_type=PatternIdentityKeyError,
        label="pattern identity key",
    )


def ensure_project_review_history_key(project_directory: Path) -> bytes:
    with suppress(FileExistsError):
        create_project_review_history_key(project_directory)
    return _read_private_key(
        project_directory,
        project_directory / REVIEW_HISTORY_KEY_FILENAME,
        error_type=ReviewHistoryKeyError,
        label="review history key",
    )


def load_pattern_identity_key(evidence: Path) -> bytes:
    project_directory = _find_evidence_project_directory(evidence)
    return _read_private_key(
        project_directory,
        project_directory / PATTERN_IDENTITY_KEY_FILENAME,
        error_type=PatternIdentityKeyError,
        label="pattern identity key",
    )


def load_review_history_key(evidence: Path) -> bytes:
    project_directory = _find_evidence_project_directory(evidence)
    key_path = project_directory / REVIEW_HISTORY_KEY_FILENAME
    if not key_path.exists() and not key_path.is_symlink():
        raise ReviewHistoryKeyError(
            "project review history key not found; run 'ul init' in this project"
        )
    return _read_private_key(
        project_directory,
        key_path,
        error_type=ReviewHistoryKeyError,
        label="review history key",
    )


def _find_evidence_project_directory(evidence: Path) -> Path:
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
                _read_private_key(
                    project_directory,
                    key_path,
                    error_type=PatternIdentityKeyError,
                    label="pattern identity key",
                )
                return project_directory
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


def pattern_review_record_hmac(key: bytes, value: object) -> str:
    if len(key) != _PATTERN_IDENTITY_KEY_BYTES:
        raise ReviewHistoryKeyError("project review history key must contain 32 bytes")
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hmac.new(
        key,
        b"ul.pattern-review.v1\0" + canonical,
        hashlib.sha256,
    ).hexdigest()


def _read_private_key(
    project_directory: Path,
    path: Path,
    *,
    error_type: type[ValueError],
    label: str,
) -> bytes:
    _validate_private_project_directory(project_directory)

    path_status = path.lstat()
    if stat.S_ISLNK(path_status.st_mode) or not stat.S_ISREG(path_status.st_mode):
        raise error_type(f"{label} must be a regular file")
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, os.O_RDONLY | no_follow_flag)
    try:
        descriptor_status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_status.st_mode)
            or descriptor_status.st_nlink != 1
            or not os.path.samestat(path_status, descriptor_status)
        ):
            raise error_type(f"{label} changed while opening")
        if os.name != "nt" and stat.S_IMODE(descriptor_status.st_mode) & 0o077:
            raise error_type(f"{label} permissions must be 0600")
        if hasattr(os, "getuid") and descriptor_status.st_uid != os.getuid():
            raise error_type(f"{label} must be owned by the current user")
        if descriptor_status.st_size != _PATTERN_IDENTITY_KEY_BYTES:
            raise error_type(f"{label} must contain 32 bytes")
        key = os.read(descriptor, _PATTERN_IDENTITY_KEY_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(key) != _PATTERN_IDENTITY_KEY_BYTES:
        raise error_type(f"{label} must contain 32 bytes")
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
