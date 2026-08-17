from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner
from ul_cli.main import app

runner = CliRunner()


def _attr(key: str, string_value: str) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": string_value}}


def _prompt_event(text: str) -> dict[str, Any]:
    return {
        "timeUnixNano": "1000000000",
        "name": "gen_ai.content.prompt",
        "attributes": [_attr("gen_ai.prompt", text)],
    }


def _completion_event(text: str) -> dict[str, Any]:
    return {
        "timeUnixNano": "2000000000",
        "name": "gen_ai.content.completion",
        "attributes": [_attr("gen_ai.completion", text)],
    }


def _span(
    trace_id: str,
    span_id: str,
    *,
    parent_span_id: str = "",
    input_text: str | None = "Pay invoice AC-100.",
    output_text: str | None = "Payment committed for AC-100.",
    use_events: bool = True,
    start_nano: str = "1000000000",
) -> dict[str, Any]:
    attributes: list[dict[str, Any]] = [
        _attr("gen_ai.operation.name", "chat"),
        _attr("gen_ai.system", "openai"),
        _attr("gen_ai.request.model", "gpt-4"),
    ]
    events: list[dict[str, Any]] = []
    if use_events:
        if input_text is not None:
            events.append(_prompt_event(input_text))
        if output_text is not None:
            events.append(_completion_event(output_text))
    else:
        if input_text is not None:
            attributes.append(_attr("gen_ai.prompt", input_text))
        if output_text is not None:
            attributes.append(_attr("gen_ai.completion", output_text))
    span: dict[str, Any] = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": "chat gpt-4",
        "startTimeUnixNano": start_nano,
        "attributes": attributes,
        "events": events,
    }
    if parent_span_id:
        span["parentSpanId"] = parent_span_id
    return span


def _otlp_export(*spans: dict[str, Any]) -> dict[str, Any]:
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": []},
                "scopeSpans": [
                    {
                        "scope": {"name": "openai", "version": "1.0.0"},
                        "spans": list(spans),
                    }
                ],
            }
        ]
    }


def _write_traces(path: Path, *spans: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_otlp_export(*spans), ensure_ascii=False),
        encoding="utf-8",
    )


def _ingest_arguments(traces: Path, output: Path, *, limit: int | None = None) -> list[str]:
    args = ["dataset", "ingest", "otlp", str(traces), "--output", str(output)]
    if limit is not None:
        args += ["--limit", str(limit)]
    return args


def test_ingest_extracts_records_from_event_based_spans(tmp_path: Path) -> None:
    traces = tmp_path / "traces.json"
    output = tmp_path / "dataset.jsonl"
    _write_traces(
        traces,
        _span("aabbccdd" * 4, "11223344" * 2, use_events=True),
    )

    result = runner.invoke(app, _ingest_arguments(traces, output))

    assert result.exit_code == 0, result.output
    assert "1 interaction" in result.output
    assert "ul dataset evaluate" in result.output
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["id"] == "aabbccdd" * 4
    assert records[0]["input"] == "Pay invoice AC-100."
    assert records[0]["output"] == "Payment committed for AC-100."


def test_ingest_extracts_records_from_attribute_based_spans(tmp_path: Path) -> None:
    traces = tmp_path / "traces.json"
    output = tmp_path / "dataset.jsonl"
    _write_traces(
        traces,
        _span("aabbccdd" * 4, "11223344" * 2, use_events=False),
    )

    result = runner.invoke(app, _ingest_arguments(traces, output))

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["input"] == "Pay invoice AC-100."


def test_ingest_deduplicates_by_trace_id(tmp_path: Path) -> None:
    traces = tmp_path / "traces.json"
    output = tmp_path / "dataset.jsonl"
    trace_id = "aabbccdd" * 4
    _write_traces(
        traces,
        _span(trace_id, "11223344" * 2, start_nano="1000"),
        _span(trace_id, "55667788" * 2, start_nano="2000"),
    )

    result = runner.invoke(app, _ingest_arguments(traces, output))

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1


def test_ingest_picks_earliest_root_gen_ai_span_per_trace(tmp_path: Path) -> None:
    traces = tmp_path / "traces.json"
    output = tmp_path / "dataset.jsonl"
    trace_id = "aabbccdd" * 4
    _write_traces(
        traces,
        _span(
            trace_id,
            "11223344" * 2,
            input_text="First call.",
            output_text="First response.",
            start_nano="1000",
        ),
        _span(
            trace_id,
            "55667788" * 2,
            parent_span_id="11223344" * 2,
            input_text="Tool call.",
            output_text="Tool response.",
            start_nano="2000",
        ),
    )

    result = runner.invoke(app, _ingest_arguments(traces, output))

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["input"] == "First call."


def test_ingest_multiple_traces_produces_multiple_records(tmp_path: Path) -> None:
    traces = tmp_path / "traces.json"
    output = tmp_path / "dataset.jsonl"
    _write_traces(
        traces,
        _span("aabbccdd" * 4, "11223344" * 2, input_text="First interaction."),
        _span("eeff0011" * 4, "22334455" * 2, input_text="Second interaction."),
    )

    result = runner.invoke(app, _ingest_arguments(traces, output))

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2


def test_ingest_skips_traces_without_gen_ai_spans(tmp_path: Path) -> None:
    output = tmp_path / "dataset.jsonl"
    non_gen_ai_span: dict[str, Any] = {
        "traceId": "aabbccdd" * 4,
        "spanId": "11223344" * 2,
        "name": "http.request",
        "attributes": [_attr("http.method", "POST")],
        "events": [],
    }
    valid_span = _span("eeff0011" * 4, "22334455" * 2)
    path = tmp_path / "traces.json"
    path.write_text(json.dumps(_otlp_export(non_gen_ai_span, valid_span)), encoding="utf-8")

    result = runner.invoke(app, _ingest_arguments(path, output))

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert "without GenAI spans" in result.output


def test_ingest_skips_traces_without_output_and_reports_count(tmp_path: Path) -> None:
    traces = tmp_path / "traces.json"
    output = tmp_path / "dataset.jsonl"
    _write_traces(
        traces,
        _span("aabbccdd" * 4, "11223344" * 2, output_text=None),
        _span("eeff0011" * 4, "22334455" * 2),
    )

    result = runner.invoke(app, _ingest_arguments(traces, output))

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert "without extractable output" in result.output


def test_ingest_fails_when_no_usable_traces(tmp_path: Path) -> None:
    traces = tmp_path / "traces.json"
    output = tmp_path / "dataset.jsonl"
    traces.write_text(json.dumps({"resourceSpans": []}), encoding="utf-8")

    result = runner.invoke(app, _ingest_arguments(traces, output))

    assert result.exit_code == 2
    assert not output.exists()


def test_ingest_enforces_limit(tmp_path: Path) -> None:
    traces = tmp_path / "traces.json"
    output = tmp_path / "dataset.jsonl"
    _write_traces(
        traces,
        _span("aabbccdd" * 4, "11223344" * 2, input_text="First."),
        _span("eeff0011" * 4, "22334455" * 2, input_text="Second."),
        _span("00112233" * 4, "44556677" * 2, input_text="Third."),
    )

    result = runner.invoke(app, _ingest_arguments(traces, output, limit=2))

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert "more than 2 interactions" in result.output


def test_ingest_does_not_overwrite_existing_output(tmp_path: Path) -> None:
    traces = tmp_path / "traces.json"
    output = tmp_path / "dataset.jsonl"
    _write_traces(traces, _span("aabbccdd" * 4, "11223344" * 2))
    output.write_text("keep", encoding="utf-8")

    result = runner.invoke(app, _ingest_arguments(traces, output))

    assert result.exit_code == 2
    assert output.read_text(encoding="utf-8") == "keep"


def test_ingest_rejects_invalid_json(tmp_path: Path) -> None:
    traces = tmp_path / "traces.json"
    output = tmp_path / "dataset.jsonl"
    traces.write_text("not json", encoding="utf-8")

    result = runner.invoke(app, _ingest_arguments(traces, output))

    assert result.exit_code == 2
    assert "traceback" not in result.output.casefold()
    assert not output.exists()


def test_ingest_rejects_non_object_json(tmp_path: Path) -> None:
    traces = tmp_path / "traces.json"
    output = tmp_path / "dataset.jsonl"
    traces.write_text("[]", encoding="utf-8")

    result = runner.invoke(app, _ingest_arguments(traces, output))

    assert result.exit_code == 2
    assert not output.exists()


def test_ingest_output_is_private(tmp_path: Path) -> None:
    traces = tmp_path / "traces.json"
    output = tmp_path / "dataset.jsonl"
    _write_traces(traces, _span("aabbccdd" * 4, "11223344" * 2))

    result = runner.invoke(app, _ingest_arguments(traces, output))

    assert result.exit_code == 0, result.output
    if os.name != "nt":
        assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_ingest_output_is_valid_jsonl_for_evaluate(tmp_path: Path) -> None:
    traces = tmp_path / "traces.json"
    output = tmp_path / "dataset.jsonl"
    _write_traces(
        traces,
        _span("aabbccdd" * 4, "11223344" * 2),
    )

    runner.invoke(app, _ingest_arguments(traces, output))

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert set(record.keys()) == {"id", "input", "output"}
    assert isinstance(record["id"], str) and record["id"]
    assert isinstance(record["input"], str) and record["input"]
    assert isinstance(record["output"], str) and record["output"]


def test_ingest_escapes_terminal_controls_from_error_messages(tmp_path: Path) -> None:
    traces = tmp_path / "traces.json"
    output = tmp_path / "dataset.jsonl"
    traces.write_text(
        json.dumps({"\u001b[31mresourceSpans": []}),
        encoding="utf-8",
    )

    result = runner.invoke(app, _ingest_arguments(traces, output), color=True)

    assert result.exit_code == 2
    assert "\x1b[31m" not in result.output


@pytest.mark.parametrize("trace_id_format", ["hex", "base64"])
def test_ingest_handles_both_trace_id_encodings(tmp_path: Path, trace_id_format: str) -> None:
    traces = tmp_path / "traces.json"
    output = tmp_path / "dataset.jsonl"
    hex_id = "aabbccdd" * 4
    if trace_id_format == "base64":
        import base64

        trace_id = base64.b64encode(bytes.fromhex(hex_id)).decode()
    else:
        trace_id = hex_id
    _write_traces(traces, _span(trace_id, "11223344" * 2))

    result = runner.invoke(app, _ingest_arguments(traces, output))

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["id"] == hex_id
