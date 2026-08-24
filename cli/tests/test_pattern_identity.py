from __future__ import annotations

import os
from pathlib import Path

import pytest
from ul_cli.pattern_identity import (
    PATTERN_IDENTITY_KEY_FILENAME,
    REVIEW_HISTORY_KEY_FILENAME,
    PatternIdentityKeyError,
    ReviewHistoryKeyError,
    create_project_pattern_identity_key,
    create_project_review_history_key,
    load_pattern_identity_key,
    load_review_history_key,
    pattern_review_record_hmac,
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


def test_project_review_history_key_is_private_and_authentication_is_canonical(
    tmp_path: Path,
) -> None:
    project_directory = tmp_path / ".ul"
    project_directory.mkdir(mode=0o700)
    create_project_pattern_identity_key(project_directory)
    create_project_review_history_key(project_directory)

    key_path = project_directory / REVIEW_HISTORY_KEY_FILENAME
    key = load_review_history_key(tmp_path / "evidence.jsonl")
    assert len(key) == 32
    assert pattern_review_record_hmac(key, {"reason": "café", "status": "confirmed"}) == (
        pattern_review_record_hmac(key, {"status": "confirmed", "reason": "café"})
    )
    assert pattern_review_record_hmac(key, {"status": "expected"}) != (
        pattern_review_record_hmac(key, {"status": "confirmed"})
    )
    if os.name != "nt":
        assert key_path.stat().st_mode & 0o777 == 0o600


def test_project_review_history_key_rejects_insecure_storage(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("POSIX permission validation")
    project_directory = tmp_path / ".ul"
    project_directory.mkdir(mode=0o700)
    create_project_pattern_identity_key(project_directory)
    key_path = project_directory / REVIEW_HISTORY_KEY_FILENAME
    key_path.write_bytes(b"x" * 32)
    key_path.chmod(0o644)

    with pytest.raises(ReviewHistoryKeyError, match="permissions must be 0600"):
        load_review_history_key(tmp_path / "evidence.jsonl")


def test_review_history_key_cannot_fall_through_to_another_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer_project = tmp_path / ".ul"
    outer_project.mkdir(mode=0o700)
    create_project_pattern_identity_key(outer_project)
    create_project_review_history_key(outer_project)
    nested_root = tmp_path / "nested"
    nested_project = nested_root / ".ul"
    nested_project.mkdir(parents=True, mode=0o700)
    create_project_pattern_identity_key(nested_project)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ReviewHistoryKeyError, match="review history key not found"):
        load_review_history_key(nested_root / "evidence.jsonl")
