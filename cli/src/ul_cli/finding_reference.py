from __future__ import annotations

import hmac
import json
import os
import secrets
import stat
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ul_cli.report_contract import FindingEvidencePackage

_REFERENCE_KEY_BYTES = 32
_MAXIMUM_REFERENCE_CONTEXT_BYTES = 200


@dataclass(frozen=True)
class FindingReferenceContext:
    key: bytes
    recorded_at: datetime


def finding_reference_key_path(finding_output: Path) -> Path:
    return finding_output.with_name(f"{finding_output.name}.key")


def create_finding_reference_context(finding_output: Path) -> FindingReferenceContext:
    reference_key = secrets.token_bytes(_REFERENCE_KEY_BYTES)
    recorded_at = datetime.now(UTC)
    key_path = finding_reference_key_path(finding_output)
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    binary_flag = os.O_BINARY if sys.platform == "win32" else 0
    descriptor = os.open(
        key_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow_flag | binary_flag,
        0o600,
    )
    created_status = os.fstat(descriptor)
    encoded_context = (reference_key.hex() + "\n" + recorded_at.isoformat() + "\n").encode("ascii")
    try:
        if sys.platform != "win32":
            os.fchmod(descriptor, 0o600)
        remaining = memoryview(encoded_context)
        while remaining:
            written = os.write(descriptor, remaining)
            if written == 0:
                raise OSError("finding reference context write was incomplete")
            remaining = remaining[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        _unlink_if_same(key_path, created_status)
        raise
    os.close(descriptor)
    _fsync_directory(finding_output.parent)
    return FindingReferenceContext(key=reference_key, recorded_at=recorded_at)


def load_finding_reference_context(finding_output: Path) -> FindingReferenceContext:
    key_path = finding_reference_key_path(finding_output)
    path_status = key_path.lstat()
    if not stat.S_ISREG(path_status.st_mode):
        raise ValueError("finding reference key must be a regular file")
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    binary_flag = os.O_BINARY if sys.platform == "win32" else 0
    descriptor = os.open(key_path, os.O_RDONLY | no_follow_flag | binary_flag)
    try:
        key_status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(key_status.st_mode)
            or (path_status.st_dev, path_status.st_ino) != (key_status.st_dev, key_status.st_ino)
            or key_status.st_nlink != 1
            or key_status.st_size > _MAXIMUM_REFERENCE_CONTEXT_BYTES
            or (
                sys.platform != "win32"
                and (key_status.st_uid != os.getuid() or stat.S_IMODE(key_status.st_mode) & 0o077)
            )
        ):
            raise ValueError("finding reference key must be private and owned by this user")
        encoded_context = os.read(descriptor, _MAXIMUM_REFERENCE_CONTEXT_BYTES).decode("ascii")
    finally:
        os.close(descriptor)
    context_lines = encoded_context.splitlines()
    if len(context_lines) != 2:
        raise ValueError("finding reference context is malformed")
    try:
        reference_key = bytes.fromhex(context_lines[0])
        recorded_at = datetime.fromisoformat(context_lines[1])
    except ValueError:
        raise ValueError("finding reference context is malformed") from None
    if (
        len(reference_key) != _REFERENCE_KEY_BYTES
        or recorded_at.tzinfo is None
        or recorded_at.utcoffset() is None
    ):
        raise ValueError("finding reference context is malformed")
    return FindingReferenceContext(key=reference_key, recorded_at=recorded_at)


def resolve_finding_reference_context(finding_output: Path) -> FindingReferenceContext:
    key_output = finding_reference_key_path(finding_output)
    finding_exists = finding_output.exists() or finding_output.is_symlink()
    key_exists = key_output.exists() or key_output.is_symlink()
    if finding_exists and not key_exists:
        raise ValueError("finding package sidecar exists without its private reference key")
    return (
        load_finding_reference_context(finding_output)
        if key_exists
        else create_finding_reference_context(finding_output)
    )


def finding_public_reference(
    reference_key: bytes,
    campaign_id: str,
    namespace: str,
    *values: str,
) -> str:
    message = json.dumps(
        [campaign_id, namespace, *values],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"ulref_v1_{hmac.digest(reference_key, message, 'sha256').hex()}"


def validate_finding_private_references(
    package: FindingEvidencePackage,
    reference_key: bytes,
) -> None:
    occurrence = package.occurrence
    private = package.private_references
    expected_references = (
        (
            occurrence.campaign_ref,
            finding_public_reference(
                reference_key,
                private.campaign_id,
                "campaign",
                private.campaign_id,
            ),
        ),
        (
            occurrence.case_ref,
            finding_public_reference(reference_key, private.campaign_id, "case", private.case_id),
        ),
        (
            occurrence.operator.id,
            finding_public_reference(
                reference_key,
                private.campaign_id,
                "operator",
                private.operator_id,
            ),
        ),
        (
            occurrence.operator.version,
            finding_public_reference(
                reference_key,
                private.campaign_id,
                "operator-version",
                private.operator_version,
            ),
        ),
    )
    if any(not hmac.compare_digest(actual, expected) for actual, expected in expected_references):
        raise ValueError("private references do not resolve the public occurrence")
    optional_references = (
        (occurrence.source_interaction_ref, "source-interaction", private.source_interaction_id),
        (
            occurrence.violated_rule.id if occurrence.violated_rule is not None else None,
            "rule",
            private.rule_id,
        ),
        (
            occurrence.violated_rule.version if occurrence.violated_rule is not None else None,
            "rule-version",
            private.rule_version,
        ),
    )
    for actual, namespace, private_value in optional_references:
        if private_value is not None and (
            actual is None
            or not hmac.compare_digest(
                actual,
                finding_public_reference(
                    reference_key,
                    private.campaign_id,
                    namespace,
                    private_value,
                ),
            )
        ):
            raise ValueError("private references do not resolve the public occurrence")


def _fsync_directory(directory: Path) -> None:
    if sys.platform == "win32":
        return
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_descriptor = os.open(directory, directory_flags)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _unlink_if_same(path: Path, created_status: os.stat_result) -> None:
    try:
        current_status = path.lstat()
    except FileNotFoundError:
        return
    if (current_status.st_dev, current_status.st_ino) == (
        created_status.st_dev,
        created_status.st_ino,
    ):
        path.unlink()
