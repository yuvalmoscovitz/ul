from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

CHECKER = Path(__file__).parents[1] / "check_python_file_lengths.py"


def run(
    *arguments: str,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*arguments],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def git(repository: Path, *arguments: str) -> str:
    result = run("git", *arguments, cwd=repository)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def write_lines(repository: Path, path: str, line_count: int) -> None:
    destination = repository / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("value = 1\n" * line_count)


def commit(repository: Path, message: str) -> str:
    git(repository, "add", "--all")
    git(repository, "commit", "-m", message)
    return git(repository, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    git(tmp_path, "init", "--initial-branch=main")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test")
    write_lines(tmp_path, "short.py", 10)
    commit(tmp_path, "initial")
    return tmp_path


def check(
    repository: Path, base: str, head: str, *, github: bool = False, merge_base: bool = False
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    if github:
        environment["GITHUB_ACTIONS"] = "true"
    else:
        environment.pop("GITHUB_ACTIONS", None)
    arguments = [sys.executable, str(CHECKER), base, head]
    if merge_base:
        arguments.append("--merge-base")
    return run(*arguments, cwd=repository, environment=environment)


def test_warns_for_new_file_at_threshold(repository: Path) -> None:
    base = git(repository, "rev-parse", "HEAD")
    write_lines(repository, "large.py", 750)
    head = commit(repository, "add large file")

    result = check(repository, base, head)

    assert result.returncode == 0
    assert "warning: 'large.py': new file has 750 physical lines" in result.stdout


def test_ignores_new_file_below_threshold(repository: Path) -> None:
    base = git(repository, "rev-parse", "HEAD")
    write_lines(repository, "small.py", 749)
    head = commit(repository, "add small file")

    result = check(repository, base, head)

    assert result.returncode == 0
    assert result.stdout == ""


def test_warns_for_new_file_copied_from_existing_long_file(repository: Path) -> None:
    write_lines(repository, "source.py", 800)
    base = commit(repository, "add source file")
    (repository / "copied.py").write_bytes((repository / "source.py").read_bytes())
    head = commit(repository, "copy source file")

    result = check(repository, base, head)

    assert result.returncode == 0
    assert "'copied.py': new file has 800 physical lines" in result.stdout


def test_warns_when_existing_file_reaches_threshold(repository: Path) -> None:
    write_lines(repository, "growing.py", 749)
    base = commit(repository, "add growing file")
    write_lines(repository, "growing.py", 750)
    head = commit(repository, "grow file")

    result = check(repository, base, head)

    assert "file grew from 749 to 750 physical lines" in result.stdout


def test_warns_when_already_long_file_grows(repository: Path) -> None:
    write_lines(repository, "growing.py", 800)
    base = commit(repository, "add growing file")
    write_lines(repository, "growing.py", 801)
    head = commit(repository, "grow file")

    result = check(repository, base, head)

    assert "file grew from 800 to 801 physical lines" in result.stdout


def test_ignores_untouched_shrunk_deleted_and_rename_only_long_files(repository: Path) -> None:
    write_lines(repository, "untouched.py", 800)
    write_lines(repository, "shrunk.py", 800)
    write_lines(repository, "same_length.py", 800)
    write_lines(repository, "deleted.py", 800)
    write_lines(repository, "renamed.py", 800)
    base = commit(repository, "add legacy files")
    write_lines(repository, "shrunk.py", 799)
    (repository / "same_length.py").write_text("value = 2\n" * 800)
    (repository / "deleted.py").unlink()
    git(repository, "mv", "renamed.py", "new_name.py")
    head = commit(repository, "maintain legacy files")

    result = check(repository, base, head)

    assert result.returncode == 0
    assert result.stdout == ""


def test_ignores_non_python_destination(repository: Path) -> None:
    write_lines(repository, "legacy.py", 800)
    base = commit(repository, "add legacy file")
    git(repository, "mv", "legacy.py", "legacy.txt")
    head = commit(repository, "rename outside policy")

    result = check(repository, base, head)

    assert result.stdout == ""


def test_warns_when_long_file_is_renamed_into_python_scope(repository: Path) -> None:
    write_lines(repository, "legacy.txt", 800)
    base = commit(repository, "add legacy text file")
    git(repository, "mv", "legacy.txt", "legacy.py")
    head = commit(repository, "rename into policy")

    result = check(repository, base, head)

    assert result.returncode == 0
    assert "'legacy.py': new file has 800 physical lines" in result.stdout


def test_escapes_special_path_in_github_annotation(repository: Path) -> None:
    base = git(repository, "rev-parse", "HEAD")
    path = ":(glob) odd,100%\nfile.py"
    write_lines(repository, path, 750)
    head = commit(repository, "add oddly named file")

    result = check(repository, base, head, github=True)

    assert result.returncode == 0
    assert (
        "::warning file=%3A(glob) odd%2C100%25%0Afile.py,line=1,title=Long Python file::"
        in result.stdout
    )
    assert result.stdout.count("\n") == 1


def test_counts_blob_instead_of_following_worktree_symlink(repository: Path) -> None:
    base = git(repository, "rev-parse", "HEAD")
    (repository / "linked.py").symlink_to("short.py")
    head = commit(repository, "add Python symlink")
    write_lines(repository, "short.py", 800)

    result = check(repository, base, head)

    assert result.returncode == 0
    assert result.stdout == ""


def test_merge_base_mode_checks_only_branch_changes(repository: Path) -> None:
    branch_point = git(repository, "rev-parse", "HEAD")
    git(repository, "checkout", "-b", "feature")
    write_lines(repository, "feature.py", 750)
    feature_head = commit(repository, "feature change")
    git(repository, "checkout", "main")
    write_lines(repository, "base_only.py", 750)
    base_head = commit(repository, "base change")

    result = check(repository, base_head, feature_head, merge_base=True)

    assert result.returncode == 0
    assert "feature.py" in result.stdout
    assert "base_only.py" not in result.stdout
    assert branch_point != base_head


def test_reports_invalid_reference_as_operational_error(repository: Path) -> None:
    head = git(repository, "rev-parse", "HEAD")

    result = check(repository, "missing-reference", head)

    assert result.returncode == 1
    assert result.stdout == ""
    assert "error: unable to check Python file lengths" in result.stderr
