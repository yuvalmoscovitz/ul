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
from pydantic import ValidationError
from ul.otlp_ingest import OtlpIngestResult, OtlpMappingConfig, parse_otlp_traces
from ul.trace_replay import TraceReplayBundle, materialize_trace_replay_bundle

from ul_cli.bfcl_ingest import BfclInputError, materialize_bfcl_cohort

app = typer.Typer(help="Import external evidence as a UL dataset.")

_MAXIMUM_FILE_BYTES = 50_000_000
_MAXIMUM_RECORDS = 100
_MAXIMUM_JSON_DEPTH = 100


class _OtlpJsonInputError(ValueError):
    pass


@app.command("bfcl")
def ingest_bfcl_dataset(
    questions: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Official BFCL V4 question JSONL file.",
        ),
    ],
    possible_answers: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Matching official BFCL V4 possible-answer JSONL file.",
        ),
    ],
    category: Annotated[
        str,
        typer.Option(help="Exact BFCL test category, such as simple_python."),
    ],
    source_revision: Annotated[
        str,
        typer.Option(help="Pinned BFCL git revision or immutable release identifier."),
    ],
    output: Annotated[
        Path | None,
        typer.Option(help="New rich UL JSONL dataset for ul dataset evaluate."),
    ] = None,
    seed: Annotated[
        int,
        typer.Option(help="Seed for deterministic SHA-256 cohort ranking."),
    ] = 0,
    limit: Annotated[
        int,
        typer.Option(
            min=1,
            max=_MAXIMUM_RECORDS,
            help=f"Nested cohort size (default: {_MAXIMUM_RECORDS}).",
        ),
    ] = _MAXIMUM_RECORDS,
    dry_run: Annotated[
        bool,
        typer.Option(help="Validate and summarize the cohort without writing content."),
    ] = False,
) -> None:
    """Prepare a reproducible BFCL V4 single-turn cohort for UL evaluation."""
    if output is None and not dry_run:
        raise typer.BadParameter(
            "--output is required unless --dry-run is used",
            param_hint="--output",
        )
    if output is not None and output.exists():
        raise typer.BadParameter(
            "output already exists; UL will not overwrite it",
            param_hint="--output",
        )
    try:
        question_bytes = _read_bounded_file(questions, maximum_bytes=_MAXIMUM_FILE_BYTES)
        answer_bytes = _read_bounded_file(possible_answers, maximum_bytes=_MAXIMUM_FILE_BYTES)
    except OSError as error:
        raise typer.BadParameter(
            f"cannot read BFCL source file ({error.__class__.__name__})",
            param_hint="QUESTIONS",
        ) from None
    if len(question_bytes) > _MAXIMUM_FILE_BYTES or len(answer_bytes) > _MAXIMUM_FILE_BYTES:
        raise typer.BadParameter(
            f"BFCL source file exceeds the {_MAXIMUM_FILE_BYTES // 1_000_000} MB limit",
            param_hint="QUESTIONS",
        )
    try:
        result = materialize_bfcl_cohort(
            question_bytes,
            answer_bytes,
            category=category,
            source_revision=source_revision,
            seed=seed,
            limit=limit,
        )
    except BfclInputError as error:
        raise typer.BadParameter(_terminal_safe(str(error)), param_hint="QUESTIONS") from None

    if dry_run:
        _print_safe(
            f"Dry run: {len(result.records)} of {result.source_record_count} BFCL case(s) ready; "
            "no benchmark content was printed or written."
        )
        _print_safe(f"Question SHA-256: {result.question_sha256}")
        _print_safe(f"Possible-answer SHA-256: {result.answer_sha256}")
        return

    assert output is not None
    try:
        output_stream = _create_private_output(output)
    except OSError as error:
        raise typer.BadParameter(
            f"cannot create output file ({error.__class__.__name__})",
            param_hint="--output",
        ) from None
    with output_stream:
        for record in result.records:
            output_stream.write(record.model_dump_json())
            output_stream.write("\n")
        output_stream.flush()
        os.fsync(output_stream.fileno())
    _print_safe(f"Prepared {len(result.records)} BFCL interaction(s) → {output}")
    _print_safe(f"Next: inspect the bounded plan with 'ul dataset evaluate {output} --dry-run'.")


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
        Path | None,
        typer.Option(help="New JSONL dataset file for ul dataset evaluate."),
    ] = None,
    replay_output: Annotated[
        Path | None,
        typer.Option(
            "--replay-output",
            help="New private trace-replay bundle with one case per eligible user turn.",
        ),
    ] = None,
    mapping: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Declarative JSON mapping for trace-native scenario ingestion.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(help="Validate and summarize ingestion without writing content."),
    ] = False,
    limit: Annotated[
        int,
        typer.Option(
            min=1,
            max=_MAXIMUM_RECORDS,
            help=f"Maximum interactions to extract (default: {_MAXIMUM_RECORDS}).",
        ),
    ] = _MAXIMUM_RECORDS,
) -> None:
    """Extract LLM interactions from an OTLP trace export for ul dataset evaluate.

    Reads traces exported from Langfuse, LangSmith, Arize Phoenix, or any
    OpenTelemetry-compatible backend that emits GenAI semantic conventions.

    Example: ul dataset ingest otlp traces.json --output dataset.jsonl
    """
    if output is None and replay_output is None and not dry_run:
        raise typer.BadParameter(
            "--output or --replay-output is required unless --dry-run is used",
            param_hint="--output",
        )
    if output is not None and output.exists():
        raise typer.BadParameter(
            "output already exists; UL will not overwrite it", param_hint="--output"
        )
    if replay_output is not None and replay_output.exists():
        raise typer.BadParameter(
            "replay output already exists; UL will not overwrite it",
            param_hint="--replay-output",
        )
    if output is not None and replay_output is not None:
        raise typer.BadParameter(
            "--output and --replay-output are separate artifact modes; choose one",
            param_hint="--replay-output",
        )

    try:
        raw_bytes = _read_bounded_file(traces, maximum_bytes=_MAXIMUM_FILE_BYTES)
    except OSError as error:
        raise typer.BadParameter(
            f"cannot read trace file ({error.__class__.__name__})",
            param_hint="TRACES",
        ) from None
    if len(raw_bytes) > _MAXIMUM_FILE_BYTES:
        raise typer.BadParameter(
            f"trace file exceeds the {_MAXIMUM_FILE_BYTES // 1_000_000} MB limit",
            param_hint="TRACES",
        )

    try:
        data = _parse_otlp_json(raw_bytes.decode("utf-8"))
        _reject_deep_json(data)
    except UnicodeDecodeError:
        raise typer.BadParameter("trace file must be UTF-8", param_hint="TRACES") from None
    except _OtlpJsonInputError as error:
        raise typer.BadParameter(_terminal_safe(str(error)), param_hint="TRACES") from None
    except (json.JSONDecodeError, RecursionError, ValueError):
        raise typer.BadParameter(
            "trace file must be one OTLP JSON object, an array of objects, or JSON Lines",
            param_hint="TRACES",
        ) from None

    mapping_config: OtlpMappingConfig | None = None
    if mapping is not None:
        try:
            mapping_bytes = _read_bounded_file(mapping, maximum_bytes=1_000_000)
            mapping_data = json.loads(
                mapping_bytes.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonstandard_constant,
                parse_float=_parse_finite_float,
            )
            _reject_deep_json(mapping_data)
            mapping_config = OtlpMappingConfig.model_validate(mapping_data)
        except OSError as error:
            raise typer.BadParameter(
                f"cannot read mapping file ({error.__class__.__name__})", param_hint="--mapping"
            ) from None
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            RecursionError,
            ValueError,
            ValidationError,
        ):
            raise typer.BadParameter(
                "mapping file is invalid; expected the UL OTLP mapping schema",
                param_hint="--mapping",
            ) from None

    try:
        result = parse_otlp_traces(data, limit=limit, mapping=mapping_config)
    except ValueError as error:
        raise typer.BadParameter(_terminal_safe(str(error)), param_hint="TRACES") from None

    replay_bundle: TraceReplayBundle | None = None
    if replay_output is not None:
        if mapping_config is None or not mapping_config.include_raw_content:
            raise typer.BadParameter(
                "--replay-output needs explicit permission to store replay content. Create "
                "mapping.json with "
                '{"schema_version":"1.0.0","include_raw_content":true,'
                '"attributes":{}} and rerun with --mapping mapping.json',
                param_hint="--replay-output",
            )
        try:
            replay_bundle = materialize_trace_replay_bundle(result.records)
        except ValueError as error:
            raise typer.BadParameter(
                _terminal_safe(str(error)), param_hint="--replay-output"
            ) from None

    if dry_run:
        _print_safe(
            f"Dry run: {len(result.records)} scenario(s) ready; "
            "no trace content was printed or written."
        )
        skipped_summary = _skipped_summary(result)
        if skipped_summary:
            _print_safe(f"Skipped traces:{skipped_summary}")
        if mapping_config is not None and not mapping_config.include_raw_content:
            _print_safe(
                "Raw content is disabled by the mapping; enable include_raw_content only for "
                "approved trace data."
            )
        if replay_bundle is not None:
            _print_safe(
                f"Replay plan: {len(replay_bundle.cases)} user-turn case(s) ready; "
                "no trace content was printed or written."
            )
        return

    if not result.records:
        skipped_summary = _skipped_summary(result)
        raise typer.BadParameter(
            f"no usable LLM interactions found in trace file{skipped_summary}; "
            "ensure your export includes GenAI semantic conventions "
            "(gen_ai.operation.name or gen_ai.prompt attributes)",
            param_hint="TRACES",
        )

    if output is not None:
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
                        {
                            "id": record.interaction_id,
                            "input": record.input,
                            "output": record.output,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            output_stream.flush()
            os.fsync(output_stream.fileno())
        _print_safe(f"Extracted {len(result.records)} interaction(s) → {output}")
    if replay_output is not None and replay_bundle is not None:
        try:
            replay_stream = _create_private_output(replay_output)
        except OSError as error:
            raise typer.BadParameter(
                f"cannot create replay output ({error.__class__.__name__})",
                param_hint="--replay-output",
            ) from None
        with replay_stream:
            replay_stream.write(replay_bundle.model_dump_json())
            replay_stream.write("\n")
            replay_stream.flush()
            os.fsync(replay_stream.fileno())
        _print_safe(f"Materialized {len(replay_bundle.cases)} replay case(s) → {replay_output}")
    skipped_summary = _skipped_summary(result)
    if skipped_summary:
        _print_safe(f"Skipped traces:{skipped_summary}")
    if result.truncated:
        _print_safe(f"Trace file contains more than {limit} interactions; use --limit to adjust.")
    if output is not None:
        _print_safe(
            "Next: create a target config with "
            "'ul dataset init target.json --url https://your-environment', "
            f"then run 'ul dataset evaluate {output} --environment-config target.json --dry-run'."
        )
    if replay_output is not None:
        _print_safe(
            f"Next: inspect the evidence-linked plan with 'ul stress trace-plan {replay_output}'."
        )


def _skipped_summary(result: OtlpIngestResult) -> str:
    parts: list[str] = []
    if result.skipped_no_gen_ai:
        parts.append(f"{result.skipped_no_gen_ai} without GenAI spans")
    if result.skipped_no_input:
        parts.append(f"{result.skipped_no_input} without extractable input")
    if result.skipped_no_output:
        parts.append(f"{result.skipped_no_output} without extractable output")
    if result.skipped_limit:
        parts.append(f"{result.skipped_limit} over trace evidence limits")
    if result.skipped_incompatible_histories:
        parts.append(f"{result.skipped_incompatible_histories} with incompatible message histories")
    return (" " + ", ".join(parts)) if parts else ""


def _parse_otlp_json(text: str) -> object:
    try:
        return _load_json_value(text)
    except json.JSONDecodeError as whole_document_error:
        batches: list[object] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                batches.append(_load_json_value(line))
            except (json.JSONDecodeError, ValueError):
                raise _OtlpJsonInputError(
                    f"invalid OTLP JSON Lines record at line {line_number}"
                ) from None
        if not batches:
            raise whole_document_error
        return batches


def _load_json_value(text: str) -> object:
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonstandard_constant,
        parse_float=_parse_finite_float,
    )


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
