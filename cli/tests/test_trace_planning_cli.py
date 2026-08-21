from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner
from ul.otlp_ingest import OtlpMappingConfig, parse_otlp_traces
from ul.trace_replay import (
    TraceReplayBundle,
    TraceReplayResult,
    TraceReplayTrial,
    materialize_trace_replay_bundle,
)
from ul_cli.main import app

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


def _write_trace_bundle(path: Path) -> TraceReplayBundle:
    export = json.loads((FIXTURES / "otlp_agent_trace.json").read_text(encoding="utf-8"))
    records = parse_otlp_traces(export, mapping=OtlpMappingConfig(include_raw_content=True)).records
    bundle = materialize_trace_replay_bundle(records)
    path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    return bundle


def test_trace_plan_prints_evidence_links_without_recorded_content(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundle.json"
    bundle = _write_trace_bundle(bundle_path)

    result = runner.invoke(app, ["stress", "trace-plan", str(bundle_path)])

    assert result.exit_code == 0, result.output
    assert "Trace-derived stress plan: 1 case(s)" in result.output
    assert "trace_error" in result.output
    assert "retry_attempt" in result.output
    assert "The source trace recorded an error status or error event." in result.output
    assert (
        "Suggested stress focus: error recovery, retry safety, state consistency" in result.output
    )
    assert "3333333333333333" in result.output
    assert "Pay approved invoice" not in result.output
    assert bundle.cases[0].case_id in result.output


def test_trace_plan_json_is_stable_and_machine_readable(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundle.json"
    _write_trace_bundle(bundle_path)

    first = runner.invoke(app, ["stress", "trace-plan", str(bundle_path), "--json"])
    second = runner.invoke(app, ["stress", "trace-plan", str(bundle_path), "--json"])

    assert first.exit_code == 0, first.output
    assert first.output == second.output
    payload = json.loads(first.output)
    assert payload["schema_version"] == "1.0.0"
    assert payload["cases"][0]["signals"][0]["code"] == "trace_error"


def test_trace_group_groups_inconclusive_results_and_preserves_references(
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / "bundle.json"
    bundle = _write_trace_bundle(bundle_path)
    replay_result = TraceReplayResult(
        case=bundle.cases[0],
        requested_repetitions=1,
        required_target_calls=1,
        status="inconclusive",
        response_match_count=0,
        state_match_count=None,
        trials=(
            TraceReplayTrial(
                repetition=1,
                inconclusive_reason="environment execution timed out",
            ),
        ),
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(replay_result.model_dump_json(indent=2), encoding="utf-8")

    result = runner.invoke(app, ["stress", "trace-group", str(result_path), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["failure_count"] == 1
    assert payload["groups"][0]["signature"] == "environment_execution_timeout"
    assert payload["groups"][0]["members"][0] == {
        "case_id": bundle.cases[0].case_id,
        "source_trace_id": bundle.cases[0].source_trace_id,
        "source_span_ids": list(bundle.cases[0].source_span_ids),
    }
