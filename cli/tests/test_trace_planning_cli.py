from __future__ import annotations

import json
import stat
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


def _write_trace_campaign_target(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 5,
                "environment_id": "trace-replay-campaign-test",
                "reset": {
                    "url": "http://127.0.0.1:8765/reset",
                    "request_json_template": {"case_id": "{{case_id}}"},
                    "case_id_json_pointer": "/case_id",
                    "generation_json_pointer": "/generation",
                    "clean_state_json_pointer": "/clean",
                    "clean_state_value": True,
                },
                "setup": {
                    "url": "http://127.0.0.1:8765/setup",
                    "request_json_template": {"case_id": "{{case_id}}"},
                    "case_id_json_pointer": "/case_id",
                },
                "execute_turn": {
                    "url": "http://127.0.0.1:8765/execute",
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "turn_id": "{{turn_id}}",
                        "input": "{{input}}",
                    },
                    "case_id_json_pointer": "/case_id",
                    "turn_id_json_pointer": "/turn_id",
                },
                "snapshot": {
                    "url": "http://127.0.0.1:8765/snapshot",
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
    assert f"ul stress trace {bundle_path}" in result.output
    assert "--environment-config .ul/environment.json --dry-run" in result.output
    assert "review directions, not automatically runnable augmentations" in result.output


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
    assert payload["difference_count"] == 1
    assert payload["groups"][0]["signature"] == "environment_execution_timeout"
    assert payload["groups"][0]["members"][0] == {
        "case_id": bundle.cases[0].case_id,
        "source_trace_id": bundle.cases[0].source_trace_id,
        "source_span_ids": list(bundle.cases[0].source_span_ids),
    }


def test_trace_group_explains_technical_difference_without_private_content(
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

    result = runner.invoke(app, ["stress", "trace-group", str(result_path)])

    assert result.exit_code == 0, result.output
    assert "1 difference(s)" in result.output
    assert "did not answer before timeout" in result.output
    assert "semantic agent failure" in result.output
    assert "failure(s)" not in result.output
    assert "Pay approved invoice" not in result.output


def test_trace_replay_campaign_dry_run_reports_actionable_plan_without_credentials(
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / "bundle.json"
    bundle = _write_trace_bundle(bundle_path)
    target_path = tmp_path / "environment.json"
    _write_trace_campaign_target(target_path)
    target = json.loads(target_path.read_text(encoding="utf-8"))
    target["headers_from_env"] = {"Authorization": "UL_ENVIRONMENT_MISSING_TEST_TOKEN"}
    target_path.write_text(json.dumps(target), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "stress",
            "trace-replay-campaign",
            str(bundle_path),
            "--environment-config",
            str(target_path),
            "--limit",
            "1",
            "--repetitions",
            "2",
            "--max-environment-api-calls",
            "12",
            "--allow-insecure-http",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"Trace replay campaign: 1/{len(bundle.cases)} prioritized case(s)" in result.output
    assert "Repetitions per case: 2" in result.output
    assert "Potential environment API calls: 12 / 12 authorized" in result.output
    assert "priority score" in result.output
    assert "Why:" in result.output
    assert "Calls: 6 per replay x 2 = 12" in result.output
    assert "Recorded message and state content: not printed" in result.output
    assert "External calls: none" in result.output
    assert "Next: ul stress trace-replay-campaign" in result.output
    assert "--output trace-replay-campaign.json" in result.output
    assert "--allow-environment-network --confirm-test-environment" in result.output
    assert "Pay approved invoice" not in result.output


def test_trace_replay_campaign_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundle.json"
    _write_trace_bundle(bundle_path)
    target_path = tmp_path / "environment.json"
    _write_trace_campaign_target(target_path)
    output_path = tmp_path / "campaign.json"
    output_path.write_text("keep me", encoding="utf-8")
    output_path.chmod(0o640)

    result = runner.invoke(
        app,
        [
            "stress",
            "trace-replay-campaign",
            str(bundle_path),
            "--environment-config",
            str(target_path),
            "--output",
            str(output_path),
            "--limit",
            "1",
            "--repetitions",
            "1",
            "--max-environment-api-calls",
            "8",
            "--allow-environment-network",
            "--confirm-test-environment",
            "--allow-insecure-http",
        ],
    )

    assert result.exit_code != 0
    assert "output already exists; UL will not overwrite it" in result.output
    assert output_path.read_text(encoding="utf-8") == "keep me"
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o640


def test_trace_replay_campaign_help_explains_setup_and_budget_controls() -> None:
    result = runner.invoke(app, ["stress", "trace-replay-campaign", "--help"])

    assert result.exit_code == 0, result.output
    normalized_output = " ".join(result.output.split())
    assert "-e" in normalized_output
    assert "-b" in normalized_output
    assert "-n" in normalized_output
    assert "resettable" in normalized_output
    assert "cumulative" in normalized_output
    assert "call budget" in normalized_output
