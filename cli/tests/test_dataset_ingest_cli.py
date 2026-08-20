from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner
from ul_cli import event_stress
from ul_cli.main import app

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


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


def _write_mapping(path: Path, *, include_raw_content: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "include_raw_content": include_raw_content,
                "maximum_content_characters": 1000,
                "attributes": {},
            }
        ),
        encoding="utf-8",
    )


def _write_target_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 5,
                "environment_id": "test-environment",
                "reset": {
                    "url": "https://environment.example.test/reset",
                    "request_json_template": {"case_id": "{{case_id}}"},
                    "case_id_json_pointer": "/case_id",
                    "generation_json_pointer": "/generation",
                    "clean_state_json_pointer": "/clean",
                    "clean_state_value": True,
                },
                "setup": {
                    "url": "https://environment.example.test/setup",
                    "request_json_template": {"case_id": "{{case_id}}"},
                    "case_id_json_pointer": "/case_id",
                },
                "execute_turn": {
                    "url": "https://environment.example.test/execute",
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "turn_id": "{{turn_id}}",
                        "input": "{{input}}",
                    },
                    "case_id_json_pointer": "/case_id",
                    "turn_id_json_pointer": "/turn_id",
                },
                "snapshot": {
                    "url": "https://environment.example.test/snapshot",
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "turn_id": "{{turn_id}}",
                    },
                    "case_id_json_pointer": "/case_id",
                    "turn_id_json_pointer": "/turn_id",
                },
            }
        ),
        encoding="utf-8",
    )


def test_trace_native_mapping_preserves_ordered_agent_evidence(tmp_path: Path) -> None:
    output = tmp_path / "dataset.jsonl"
    mapping = tmp_path / "mapping.json"
    _write_mapping(mapping)

    result = runner.invoke(
        app,
        [
            "dataset",
            "ingest",
            "otlp",
            str(FIXTURES / "otlp_agent_trace.json"),
            "--mapping",
            str(mapping),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["input"] == "Pay approved invoice AC-100."
    scenario = record["output"]
    assert scenario["kind"] == "ul.trace_scenario"
    assert scenario["session_id"] == "session-42"
    assert scenario["agent"] == {"name": "invoice-reviewer", "version": "2026.08.18"}
    assert scenario["source"]["reference"] == "response-17"
    assert [span["span_id"] for span in scenario["spans"]] == [
        "1111111111111111",
        "2222222222222222",
        "3333333333333333",
    ]
    tool_span = scenario["spans"][2]
    assert tool_span["parent_span_id"] == "2222222222222222"
    assert tool_span["retry_attempt"] == 2
    assert tool_span["tool_calls"][0]["arguments"] == {"invoice_id": "AC-100"}
    assert tool_span["state_delta"] == {"invoice_checked": True}
    assert tool_span["errors"] == [
        {"type": "TransientTimeout", "message": "first attempt timed out"}
    ]
    assert "must-not-be-copied" not in output.read_text(encoding="utf-8")
    if os.name != "nt":
        assert stat.S_IMODE(output.stat().st_mode) == 0o600

    evaluation = runner.invoke(app, ["dataset", "evaluate", str(output), "--dry-run"])

    assert evaluation.exit_code == 0, evaluation.output
    assert "1 interaction" in evaluation.output


def test_trace_native_dry_run_does_not_write_or_print_content(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.json"
    _write_mapping(mapping)

    result = runner.invoke(
        app,
        [
            "dataset",
            "ingest",
            "otlp",
            str(FIXTURES / "otlp_agent_trace.json"),
            "--mapping",
            str(mapping),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "1 scenario(s) ready" in result.output
    assert "AC-100" not in result.output
    assert "must-not-be-copied" not in result.output


def test_trace_native_materializes_private_replay_bundle_and_dry_run_plan(
    tmp_path: Path,
) -> None:
    mapping = tmp_path / "mapping.json"
    replay_output = tmp_path / "replay.json"
    target_config = tmp_path / "target.json"
    _write_mapping(mapping)
    _write_target_config(target_config)

    ingest_result = runner.invoke(
        app,
        [
            "dataset",
            "ingest",
            "otlp",
            str(FIXTURES / "otlp_agent_trace.json"),
            "--mapping",
            str(mapping),
            "--replay-output",
            str(replay_output),
        ],
    )

    assert ingest_result.exit_code == 0, ingest_result.output
    bundle = json.loads(replay_output.read_text(encoding="utf-8"))
    assert bundle["schema_version"] == "1.0.0"
    assert len(bundle["envelopes"]) == 1
    assert len(bundle["cases"]) == 1
    assert bundle["cases"][0]["replay_user_turns"][0]["content"] == ("Pay approved invoice AC-100.")
    if os.name != "nt":
        assert stat.S_IMODE(replay_output.stat().st_mode) == 0o600

    dry_run = runner.invoke(
        app,
        [
            "stress",
            "trace",
            str(replay_output),
            "--environment-config",
            str(target_config),
            "--dry-run",
        ],
    )

    assert dry_run.exit_code == 0, dry_run.output
    assert "Ordered user turns: 1" in dry_run.output
    assert "Recorded content: not printed" in dry_run.output
    assert "AC-100" not in dry_run.output


def test_replay_output_requires_explicit_raw_content_mapping(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.json"
    replay_output = tmp_path / "replay.json"
    _write_mapping(mapping, include_raw_content=False)

    result = runner.invoke(
        app,
        [
            "dataset",
            "ingest",
            "otlp",
            str(FIXTURES / "otlp_agent_trace.json"),
            "--mapping",
            str(mapping),
            "--replay-output",
            str(replay_output),
        ],
    )

    assert result.exit_code == 2
    assert "include_raw_content" in result.output
    assert not replay_output.exists()


def test_trace_replay_output_reservation_failure_makes_no_target_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = tmp_path / "mapping.json"
    replay_bundle = tmp_path / "replay.json"
    target_config = tmp_path / "target.json"
    output = tmp_path / "evidence.json"
    _write_mapping(mapping)
    _write_target_config(target_config)
    ingest_result = runner.invoke(
        app,
        [
            "dataset",
            "ingest",
            "otlp",
            str(FIXTURES / "otlp_agent_trace.json"),
            "--mapping",
            str(mapping),
            "--replay-output",
            str(replay_bundle),
        ],
    )
    assert ingest_result.exit_code == 0, ingest_result.output

    target_constructions = 0

    class _TargetMustNotBeConstructed:
        @classmethod
        def from_config(cls, *args: object, **kwargs: object) -> object:
            nonlocal target_constructions
            target_constructions += 1
            raise AssertionError("target must not be constructed")

    def reject_output(_path: Path) -> object:
        raise OSError("simulated output failure")

    monkeypatch.setattr(event_stress, "JsonHttpEnvironmentConnection", _TargetMustNotBeConstructed)
    monkeypatch.setattr(event_stress, "_create_private_output", reject_output)

    result = runner.invoke(
        app,
        [
            "stress",
            "trace",
            str(replay_bundle),
            "--environment-config",
            str(target_config),
            "--allow-environment-network",
            "--confirm-test-environment",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "output could not be created" in result.output
    assert target_constructions == 0
    assert not output.exists()


def test_trace_native_requires_explicit_raw_content_opt_in(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.json"
    _write_mapping(mapping, include_raw_content=False)

    result = runner.invoke(
        app,
        [
            "dataset",
            "ingest",
            "otlp",
            str(FIXTURES / "otlp_agent_trace.json"),
            "--mapping",
            str(mapping),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "0 scenario(s) ready" in result.output
    assert "Raw content is disabled" in result.output


def test_trace_native_rejects_unknown_mapping_fields_without_echoing_values(
    tmp_path: Path,
) -> None:
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps({"schema_version": "1.0.0", "secret_mapping": "do-not-echo"}),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "dataset",
            "ingest",
            "otlp",
            str(FIXTURES / "otlp_agent_trace.json"),
            "--mapping",
            str(mapping),
            "--dry-run",
        ],
    )

    assert result.exit_code == 2
    assert "mapping file is invalid" in result.output
    assert "do-not-echo" not in result.output


def test_file_exporter_json_lines_batches_merge_deterministically(tmp_path: Path) -> None:
    traces = tmp_path / "traces.jsonl"
    output = tmp_path / "dataset.jsonl"
    mapping = tmp_path / "mapping.json"
    _write_mapping(mapping)
    later_trace = _otlp_export(_span("eeff0011" * 4, "22334455" * 2, input_text="Second trace"))
    earlier_trace = _otlp_export(_span("aabbccdd" * 4, "11223344" * 2, input_text="First trace"))
    traces.write_text(
        json.dumps(later_trace) + "\n" + json.dumps(earlier_trace) + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "dataset",
            "ingest",
            "otlp",
            str(traces),
            "--mapping",
            str(mapping),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["id"] for record in records] == ["aabbccdd" * 4, "eeff0011" * 4]


def test_file_exporter_json_array_is_accepted(tmp_path: Path) -> None:
    traces = tmp_path / "traces.json"
    output = tmp_path / "dataset.jsonl"
    traces.write_text(
        json.dumps([_otlp_export(_span("aabbccdd" * 4, "11223344" * 2, input_text="Array trace"))]),
        encoding="utf-8",
    )

    result = runner.invoke(app, _ingest_arguments(traces, output))

    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text(encoding="utf-8"))["input"] == "Array trace"


def test_file_exporter_json_lines_error_identifies_line_without_values(tmp_path: Path) -> None:
    traces = tmp_path / "traces.jsonl"
    output = tmp_path / "dataset.jsonl"
    traces.write_text(
        json.dumps(_otlp_export(_span("aabbccdd" * 4, "11223344" * 2)))
        + '\n{"secret":"do-not-echo"\n',
        encoding="utf-8",
    )

    result = runner.invoke(app, _ingest_arguments(traces, output))

    assert result.exit_code == 2
    assert "line 2" in result.output
    assert "do-not-echo" not in result.output


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

    result = runner.invoke(app, _ingest_arguments(traces, output))
    assert result.exit_code == 0, result.output

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

    result = runner.invoke(app, _ingest_arguments(traces, output))

    assert result.exit_code == 2
    assert "traceback" not in result.output.casefold()


def test_ingest_rejects_oversized_file_with_clear_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    traces = tmp_path / "traces.json"
    output = tmp_path / "dataset.jsonl"
    _write_traces(traces, _span("aabbccdd" * 4, "11223344" * 2))
    monkeypatch.setattr("ul_cli.dataset_ingest._MAXIMUM_FILE_BYTES", 10)

    result = runner.invoke(app, _ingest_arguments(traces, output))

    assert result.exit_code == 2
    assert "limit" in result.output.casefold() or "mb" in result.output.casefold()
    assert not output.exists()


def test_ingest_no_usable_traces_error_includes_actionable_guidance(tmp_path: Path) -> None:
    traces = tmp_path / "traces.json"
    output = tmp_path / "dataset.jsonl"
    traces.write_text(json.dumps({"resourceSpans": []}), encoding="utf-8")

    result = runner.invoke(app, _ingest_arguments(traces, output))

    assert result.exit_code == 2
    assert "gen_ai" in result.output.casefold() or "semantic" in result.output.casefold()


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
