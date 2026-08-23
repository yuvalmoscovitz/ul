from __future__ import annotations

import os
from pathlib import Path

import pytest
from ul_cli.pattern_identity import (
    PATTERN_IDENTITY_KEY_FILENAME,
    PatternIdentityKeyError,
    create_project_pattern_identity_key,
    load_pattern_identity_key,
)


def test_project_pattern_identity_key_is_private_persisted_and_stable(tmp_path: Path) -> None:
    project_directory = tmp_path / ".ul"
    project_directory.mkdir(mode=0o700)

    create_project_pattern_identity_key(project_directory)

    key_path = project_directory / PATTERN_IDENTITY_KEY_FILENAME
    first = load_pattern_identity_key(tmp_path / "evidence.jsonl")
    second = load_pattern_identity_key(tmp_path / "other.jsonl")
    assert first == second
    assert len(first) == 32
    if os.name != "nt":
        assert key_path.stat().st_mode & 0o777 == 0o600


def test_project_pattern_identity_key_rejects_insecure_storage(tmp_path: Path) -> None:
    project_directory = tmp_path / ".ul"
    project_directory.mkdir(mode=0o700)
    key_path = project_directory / PATTERN_IDENTITY_KEY_FILENAME
    key_path.write_bytes(b"x" * 32)
    if os.name == "nt":
        pytest.skip("POSIX permission validation")

    key_path.chmod(0o644)
    with pytest.raises(PatternIdentityKeyError, match="permissions must be 0600"):
        load_pattern_identity_key(tmp_path / "evidence.jsonl")


def test_project_pattern_identity_key_rejects_wrong_size(tmp_path: Path) -> None:
    project_directory = tmp_path / ".ul"
    project_directory.mkdir(mode=0o700)
    key_path = project_directory / PATTERN_IDENTITY_KEY_FILENAME
    key_path.write_bytes(b"too-short")
    key_path.chmod(0o600)

    with pytest.raises(PatternIdentityKeyError, match="must contain 32 bytes"):
        load_pattern_identity_key(tmp_path / "evidence.jsonl")
