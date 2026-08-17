from __future__ import annotations

import json
import math
import os
import stat
import sys
import unicodedata
from pathlib import Path
from typing import Annotated, TextIO, cast

import typer
from ul.otlp_ingest import parse_otlp_traces

app = typer.Typer(help="Import production traces as a UL dataset.")

_MAXIMUM_FILE_BYTES = 50_000_000
_MAXIMUM_RECORDS = 100
_MAXIMUM_JSON_DEPTH = 100


@app.command("otlp")
def ingest_otlp_traces(
    traces: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="OTLP JSON trace export file.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(help="New JSONL dataset file for ul dataset evaluate."),
    ],
    limit: Annotated[
        int,
        typer.Option(
            min=1,
            max=_MAXIMUM_RECORDS,
            help="Maximum interactions to extract.",
        ),
    ] = _MAXIMUM_RECORDS,
) -> None:
    """Extract LLM interactions from an OTLP trace export for ul dataset evaluate.

    Reads traces exported from Langfuse, LangSmith, Arize Phoenix, or any
    OpenTelemetry-compatible backend that emits GenAI semantic conventions.

    Example: ul dataset ingest otlp traces.json --output dataset.jsonl
    """
    if output.exists():
        raise typer.BadParameter(
            "output already exists; UL will not overwrite it", param_hint="--output"
        )

    try:
        raw_bytes = _read_bounded_file(traces, maximum_bytes=_MAXIMUM_FILE_BYTES)
    except OSError as error:
        raise typer.BadParameter(
            f"cannot read trace file ({error.__class__.__name__})",
            param_hint="TRACES",
        ) from None

    try:
        data = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
            parse_float=_parse_finite_float,
        )
        _reject_deep_json(data)
    except UnicodeDecodeError:
        raise typer.BadParameter("trace file must be UTF-8", param_hint="TRACES") from None
    except (json.JSONDecodeError, RecursionError, ValueError):
        raise typer.BadParameter("trace file contains invalid JSON", param_hint="TRACES") from None

    try:
        result = parse_otlp_traces(data, limit=limit)
    except ValueError as error:
        raise typer.BadParameter(_terminal_safe(str(error)), param_hint="TRACES") from None

    if not result.records:
        skipped_summary = _skipped_summary(result)
        raise typer.BadParameter(
            f"no usable LLM interactions found in trace file{skipped_summary}",
            param_hint="TRACES",
        )

    try:
        output_stream = _create_private_output(output)
    except OSError as error:
        raise typer.BadParameter(
            f"cannot create output file ({error.__class__.__name__})",
            param_hint="--output",
        ) from None

    with output_stream:
        for record in result.records:
            output_stream.write(
                json.dumps(
                    {"id": record.interaction_id, "input": record.input, "output": record.output},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
        output_stream.flush()
        os.fsync(output_stream.fileno())

    _print_safe(f"Extracted {len(result.records)} interaction(s) → {output}")
    skipped_summary = _skipped_summary(result)
    if skipped_summary:
        _print_safe(f"Skipped traces:{skipped_summary}")
    if result.truncated:
        _print_safe(f"Trace file contains more than {limit} interactions; use --limit to adjust.")
    _print_safe(f"Next: ul dataset evaluate {output} --target-config target.json --dry-run")


def _skipped_summary(result: object) -> str:
    from ul.otlp_ingest import OtlpIngestResult

    assert isinstance(result, OtlpIngestResult)
    parts: list[str] = []
    if result.skipped_no_gen_ai:
        parts.append(f"{result.skipped_no_gen_ai} without GenAI spans")
    if result.skipped_no_input:
        parts.append(f"{result.skipped_no_input} without extractable input")
    if result.skipped_no_output:
        parts.append(f"{result.skipped_no_output} without extractable output")
    return (" " + ", ".join(parts)) if parts else ""


def _read_bounded_file(path: Path, *, maximum_bytes: int) -> bytes:
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    requires_identity_check = no_follow_flag == 0
    if requires_identity_check and stat.S_ISLNK(os.lstat(path).st_mode):
        raise OSError("path is a symbolic link")
    binary_flag = os.O_BINARY if sys.platform == "win32" else 0
    nonblocking_flag = getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, os.O_RDONLY | no_follow_flag | nonblocking_flag | binary_flag)
    try:
        descriptor_status = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_status.st_mode):
            raise OSError("path is not a regular file")
        if requires_identity_check:
            path_status = os.lstat(path)
            if stat.S_ISLNK(path_status.st_mode) or not os.path.samestat(
                descriptor_status, path_status
            ):
                raise OSError("path changed while it was opened")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65_536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _create_private_output(path: Path) -> TextIO:
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    binary_flag = os.O_BINARY if sys.platform == "win32" else 0
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow_flag | binary_flag,
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("output is not a regular file")
        if sys.platform != "win32":
            os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "w", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _reject_deep_json(value: object, *, depth: int = 0) -> None:
    if depth > _MAXIMUM_JSON_DEPTH:
        raise ValueError("JSON exceeds the nesting limit")
    if isinstance(value, dict):
        for item in cast(dict[str, object], value).values():
            _reject_deep_json(item, depth=depth + 1)
    elif isinstance(value, list):
        for item in cast(list[object], value):
            _reject_deep_json(item, depth=depth + 1)


def _print_safe(message: str) -> None:
    typer.echo(_terminal_safe(message))


def _terminal_safe(message: str) -> str:
    return "".join(
        character
        if (ord(character) >= 32 and not 0x7F <= ord(character) <= 0x9F)
        and unicodedata.category(character) not in {"Cf", "Cs"}
        else f"\\u{ord(character):04x}"
        for character in message
    )
