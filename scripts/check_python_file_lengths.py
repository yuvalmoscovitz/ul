from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Annotated

import typer

WARNING_LINE_COUNT = 750


class GitError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChangedFile:
    base_path: str | None
    head_path: str


def run_git(*arguments: str) -> bytes:
    environment = os.environ.copy()
    environment["GIT_LITERAL_PATHSPECS"] = "1"
    result = subprocess.run(
        ["git", *arguments],
        check=False,
        capture_output=True,
        env=environment,
    )
    if result.returncode != 0:
        error = os.fsdecode(result.stderr).strip()
        raise GitError(error or f"git {' '.join(arguments)} failed")
    return result.stdout


def resolve_commit(reference: str) -> str:
    return os.fsdecode(
        run_git("rev-parse", "--verify", "--end-of-options", f"{reference}^{{commit}}")
    ).strip()


def changed_python_files(base_commit: str, head_commit: str) -> list[ChangedFile]:
    output = run_git(
        "diff",
        "--name-status",
        "-z",
        "--diff-filter=AMRT",
        "--find-renames",
        base_commit,
        head_commit,
        "--",
    )
    fields = output.split(b"\0")
    if fields[-1] != b"":
        raise GitError("git diff returned malformed path data")

    changes: list[ChangedFile] = []
    index = 0
    while index < len(fields) - 1:
        status = os.fsdecode(fields[index])
        index += 1
        if status.startswith("R"):
            if index + 1 >= len(fields):
                raise GitError("git diff returned an incomplete rename")
            renamed_from = os.fsdecode(fields[index])
            head_path = os.fsdecode(fields[index + 1])
            base_path = renamed_from if renamed_from.endswith(".py") else None
            index += 2
        else:
            if index >= len(fields):
                raise GitError("git diff returned an incomplete path")
            head_path = os.fsdecode(fields[index])
            base_path = None if status == "A" else head_path
            index += 1

        if head_path.endswith(".py"):
            changes.append(ChangedFile(base_path=base_path, head_path=head_path))

    return sorted(changes, key=lambda change: os.fsencode(change.head_path))


def blob_line_count(commit: str, path: str) -> int:
    tree_entry = run_git("ls-tree", "-z", commit, "--", path)
    entries = tree_entry.split(b"\0")
    if len(entries) != 2 or entries[-1] != b"" or b"\t" not in entries[0]:
        raise GitError(f"could not find {path!r} in commit {commit}")

    metadata, listed_path = entries[0].split(b"\t", 1)
    if listed_path != os.fsencode(path):
        raise GitError(f"git returned an unexpected path for {path!r}")
    metadata_fields = metadata.split()
    if len(metadata_fields) != 3 or metadata_fields[1] != b"blob":
        raise GitError(f"{path!r} is not a file in commit {commit}")

    content = run_git("cat-file", "blob", os.fsdecode(metadata_fields[2]))
    return content.count(b"\n") + int(bool(content) and not content.endswith(b"\n"))


def escape_annotation_data(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def escape_annotation_property(value: str) -> str:
    return escape_annotation_data(value).replace(":", "%3A").replace(",", "%2C")


def emit_warning(path: str, base_lines: int | None, head_lines: int) -> None:
    if base_lines is None:
        detail = f"new file has {head_lines} physical lines"
    else:
        detail = f"file grew from {base_lines} to {head_lines} physical lines"
    message = f"{detail}; consider splitting it (warning threshold: {WARNING_LINE_COUNT})"

    if os.environ.get("GITHUB_ACTIONS") == "true":
        typer.echo(
            "::warning "
            f"file={escape_annotation_property(path)},line=1,title=Long Python file::"
            f"{escape_annotation_data(message)}"
        )
    else:
        typer.echo(f"warning: {path!r}: {message}")


def check_file_lengths(base: str, head: str, use_merge_base: bool) -> None:
    base_commit = resolve_commit(base)
    head_commit = resolve_commit(head)
    comparison_base = base_commit
    if use_merge_base:
        comparison_base = os.fsdecode(run_git("merge-base", base_commit, head_commit)).strip()
        if not comparison_base:
            raise GitError(f"{base!r} and {head!r} do not have a merge base")

    for change in changed_python_files(comparison_base, head_commit):
        head_lines = blob_line_count(head_commit, change.head_path)
        base_lines = (
            blob_line_count(comparison_base, change.base_path)
            if change.base_path is not None
            else None
        )
        newly_long = base_lines is None or base_lines < WARNING_LINE_COUNT
        grew_while_long = base_lines is not None and head_lines > base_lines
        if head_lines >= WARNING_LINE_COUNT and (newly_long or grew_while_long):
            emit_warning(change.head_path, base_lines, head_lines)


def main(
    base: Annotated[str, typer.Argument(help="Base Git commit or reference")],
    head: Annotated[str, typer.Argument(help="Head Git commit or reference")],
    merge_base: Annotated[
        bool,
        typer.Option("--merge-base", help="Compare the merge base with head"),
    ] = False,
) -> None:
    try:
        check_file_lengths(base, head, merge_base)
    except (GitError, OSError) as error:
        typer.echo(f"error: unable to check Python file lengths: {error}", err=True)
        raise typer.Exit(code=1) from error


if __name__ == "__main__":
    typer.run(main)
