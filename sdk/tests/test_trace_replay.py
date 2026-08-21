from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import ul.trace_replay as trace_replay_module
from ul.otlp_ingest import OtlpMappingConfig, parse_otlp_traces
from ul.trace_replay import (
    TraceReplayResult,
    TraceReplayTrial,
    derive_trace_stress_plan,
    group_trace_replay_differences,
    load_trace_replay_bundle,
    load_trace_replay_result,
    load_trace_replay_results,
    materialize_trace_replay_bundle,
    run_trace_replay,
)
from ul_core.dataset import ObservedAgentOutput
from ul_core.evaluation import (
    EnvironmentCapabilities,
    EnvironmentLifecycleEvidence,
    EnvironmentResetEvidence,
    EnvironmentStateEvidence,
    EnvironmentTurnEvidence,
    EvaluationCase,
    ExecutionEvidence,
)


def _attribute(key: str, value: str) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": value}}


def _trace_records(
    *,
    include_state: bool = True,
    include_stress_signals: bool = False,
    stress_error_span_id: str = "11" * 8,
    include_future_child_signals: bool = False,
) -> tuple[Any, ...]:
    first_user = {"role": "user", "parts": [{"type": "text", "content": "Pay AC-100."}]}
    first_response = {
        "role": "assistant",
        "parts": [{"type": "text", "content": "Prepared AC-100."}],
    }
    second_user = {
        "role": "user",
        "parts": [{"type": "text", "content": "Approve and submit it."}],
    }
    second_response = {
        "role": "assistant",
        "parts": [{"type": "text", "content": "Submitted AC-100."}],
    }

    def span(
        span_id: str,
        start: str,
        end: str,
        inputs: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        state: dict[str, Any],
        parent_span_id: str | None = None,
    ) -> dict[str, Any]:
        attributes = [
            _attribute("gen_ai.operation.name", "invoke_agent"),
            _attribute("gen_ai.input.messages", json.dumps(inputs)),
            _attribute("gen_ai.output.messages", json.dumps(outputs)),
        ]
        if include_state:
            attributes.append(_attribute("ul.state.snapshot", json.dumps(state)))
        if include_stress_signals:
            attributes.extend(
                [
                    _attribute("gen_ai.tool.call.id", f"call-{span_id[:2]}"),
                    _attribute("gen_ai.tool.name", "accounting.submit_invoice"),
                ]
            )
            if span_id == stress_error_span_id:
                attributes.extend(
                    [
                        _attribute("retry.attempt", "2"),
                        _attribute("error.type", "TransientTimeout"),
                    ]
                )
        raw_span = {
            "traceId": "aa" * 16,
            "spanId": span_id,
            "name": "invoice agent turn",
            "startTimeUnixNano": start,
            "endTimeUnixNano": end,
            "attributes": attributes,
        }
        if parent_span_id is not None:
            raw_span["parentSpanId"] = parent_span_id
        return raw_span

    spans = [
        span(
            "11" * 8,
            "1",
            "2",
            [first_user],
            [first_response],
            {"invoice": "AC-100", "status": "prepared"},
        ),
        span(
            "22" * 8,
            "3",
            "4",
            [first_user, first_response, second_user],
            [second_response],
            {"invoice": "AC-100", "status": "submitted"},
        ),
    ]
    if include_future_child_signals:
        future_message_child = span(
            "33" * 8,
            "3",
            "4",
            [first_user, first_response, second_user],
            [second_response],
            {"invoice": "AC-100", "status": "submitted"},
            parent_span_id="11" * 8,
        )
        future_message_child["attributes"].append(_attribute("error.type", "FutureTurnError"))
        future_message_less_child = span(
            "44" * 8,
            "5",
            "6",
            [],
            [],
            {},
            parent_span_id="11" * 8,
        )
        future_message_less_child["attributes"].append(_attribute("retry.attempt", "2"))
        spans.extend((future_message_child, future_message_less_child))

    data = {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}
    return parse_otlp_traces(
        data,
        mapping=OtlpMappingConfig(include_raw_content=True),
    ).records


class _ReplayTarget:
    environment_id = "trace-replay-test-environment"
    config_sha256 = "0" * 64
    capabilities = EnvironmentCapabilities(
        supports_conversations=True,
        supports_state_observation=True,
        state_observation_authority="environment_self_reported",
        cancellation_guarantee="guaranteed",
    )

    def __init__(self, *, drift: bool = False, include_state: bool = True) -> None:
        self.drift = drift
        self.include_state = include_state
        self.conversations: list[tuple[str, ...]] = []

    def api_calls_for_case(self, case: EvaluationCase) -> int:
        turn_count = len(case.turns)
        return 3 + (2 * turn_count)

    async def execute(self, case: EvaluationCase) -> ExecutionEvidence:
        outputs = await self.execute_conversation(tuple(turn.content for turn in case.turns))
        snapshots = tuple(output.metadata.get("committed_state_snapshot", {}) for output in outputs)
        return ExecutionEvidence(
            case_id=case.id,
            environment_id=self.environment_id,
            environment_config_sha256=self.config_sha256,
            initial_state=EnvironmentStateEvidence(
                value={},
                authority="environment_self_reported",
            ),
            turns=tuple(
                EnvironmentTurnEvidence(
                    turn_id=turn.id,
                    response=output.raw_output,
                    state_snapshot=snapshot,
                    state_observation_authority="environment_self_reported",
                )
                for turn, output, snapshot in zip(case.turns, outputs, snapshots, strict=True)
            ),
            final_response=outputs[-1].raw_output,
            final_state=EnvironmentStateEvidence(
                value=snapshots[-1],
                authority="environment_self_reported",
            ),
            lifecycle=EnvironmentLifecycleEvidence(
                initial_reset=EnvironmentResetEvidence(
                    reset_session_requested=True,
                    reset_session_acknowledged=True,
                    reset_env_requested=True,
                    reset_env_acknowledged=True,
                ),
                cleanup_reset=EnvironmentResetEvidence(
                    reset_session_requested=True,
                    reset_session_acknowledged=True,
                    reset_env_requested=True,
                    reset_env_acknowledged=True,
                ),
                terminal_status="succeeded",
                completed_phases=("execute", "cleanup"),
                delivery="certain",
                cleanup="succeeded",
                environment_state_uncertain=False,
            ),
        )

    async def execute_conversation(
        self, raw_inputs: tuple[str, ...]
    ) -> tuple[ObservedAgentOutput, ...]:
        self.conversations.append(raw_inputs)
        first_metadata = (
            {
                "committed_state_snapshot": {
                    "invoice": "AC-100",
                    "status": "prepared",
                }
            }
            if self.include_state
            else {}
        )
        outputs = [
            ObservedAgentOutput(
                raw_output="Prepared AC-100.",
                metadata=first_metadata,
            )
        ]
        if len(raw_inputs) == 2:
            final_metadata = (
                {
                    "committed_state_snapshot": {
                        "invoice": "AC-200" if self.drift else "AC-100",
                        "status": "submitted",
                    }
                }
                if self.include_state
                else {}
            )
            outputs.append(
                ObservedAgentOutput(
                    raw_output=("Submitted AC-200." if self.drift else "Submitted AC-100."),
                    metadata=final_metadata,
                )
            )
        return tuple(outputs)


class _UncertainReplayEnvironment(_ReplayTarget):
    def __init__(self) -> None:
        super().__init__()
        self.execution_count = 0

    async def execute(self, case: EvaluationCase) -> ExecutionEvidence:
        self.execution_count += 1
        if self.execution_count > 1:
            raise AssertionError("uncertain environment must not be called again")
        return ExecutionEvidence(
            case_id=case.id,
            environment_id=self.environment_id,
            environment_config_sha256=self.config_sha256,
            lifecycle=EnvironmentLifecycleEvidence(
                initial_reset=EnvironmentResetEvidence(
                    reset_session_requested=True,
                    reset_session_acknowledged=True,
                    reset_env_requested=True,
                    reset_env_acknowledged=True,
                ),
                cleanup_reset=EnvironmentResetEvidence(
                    reset_session_requested=True,
                    reset_session_acknowledged=True,
                    reset_env_requested=True,
                    reset_env_acknowledged=True,
                ),
                terminal_status="failed",
                failed_phase="execute_turn",
                failure_code="transport_failed",
                failure_reason="environment API transport failed",
                delivery="uncertain",
                cleanup="succeeded",
                environment_state_uncertain=True,
            ),
        )


def test_materializes_one_replay_case_per_completed_user_turn() -> None:
    bundle = materialize_trace_replay_bundle(_trace_records())

    assert len(bundle.envelopes) == 1
    assert len(bundle.cases) == 2
    first, second = bundle.cases
    assert [turn.content for turn in first.replay_user_turns] == ["Pay AC-100."]
    assert first.source_span_ids == ("1111111111111111",)
    assert first.recorded_state_snapshot == {"invoice": "AC-100", "status": "prepared"}
    assert [turn.content for turn in second.replay_user_turns] == [
        "Pay AC-100.",
        "Approve and submit it.",
    ]
    assert second.recorded_terminal_response == "Submitted AC-100."
    assert second.recorded_state_snapshot_available is True
    assert second.recorded_state_snapshot == {"invoice": "AC-100", "status": "submitted"}
    assert second.source_span_ids == ("2222222222222222",)
    assert bundle.envelopes[0].scenario["messages"][-1]["content"] == "Submitted AC-100."


def test_derives_ranked_stress_plan_from_explicit_trace_evidence() -> None:
    bundle = materialize_trace_replay_bundle(_trace_records(include_stress_signals=True))

    plan = derive_trace_stress_plan(bundle)

    assert plan.case_count == 2
    first = plan.cases[0]
    assert first.case_id == bundle.cases[1].case_id
    assert [signal.code for signal in first.signals] == [
        "trace_error",
        "retry_attempt",
        "repeated_tool_call",
        "multi_turn_follow_up",
        "recorded_state_evidence",
    ]
    assert first.priority_score == 18
    assert first.recommended_focuses == (
        "error_recovery",
        "retry_safety",
        "multi_turn_change_handling",
        "state_consistency",
    )
    assert first.source_span_ids == ("2222222222222222",)
    assert first.signals[0].source_span_ids == ("1111111111111111",)
    assert first.signals[2].source_span_ids == (
        "1111111111111111",
        "2222222222222222",
    )


def test_stress_plan_falls_back_to_reproducibility_without_elevated_signals() -> None:
    bundle = materialize_trace_replay_bundle(_trace_records(include_state=False))

    plan = derive_trace_stress_plan(bundle)

    assert plan.cases[0].recommended_focuses == ("multi_turn_change_handling",)
    assert plan.cases[1].signals == ()
    assert plan.cases[1].recommended_focuses == ("production_reproducibility",)


def test_stress_plan_does_not_apply_later_span_signals_to_earlier_case() -> None:
    bundle = materialize_trace_replay_bundle(
        _trace_records(include_stress_signals=True, stress_error_span_id="22" * 8)
    )

    plan_by_case_id = {case.case_id: case for case in derive_trace_stress_plan(bundle).cases}
    earlier_signals = {signal.code for signal in plan_by_case_id[bundle.cases[0].case_id].signals}
    later_signals = {signal.code for signal in plan_by_case_id[bundle.cases[1].case_id].signals}

    assert "trace_error" not in earlier_signals
    assert "repeated_tool_call" not in earlier_signals
    assert "trace_error" in later_signals
    assert "repeated_tool_call" in later_signals


def test_stress_plan_excludes_future_children_of_an_earlier_case_span() -> None:
    bundle = materialize_trace_replay_bundle(_trace_records(include_future_child_signals=True))

    plan_by_case_id = {case.case_id: case for case in derive_trace_stress_plan(bundle).cases}
    earlier_signals = {signal.code for signal in plan_by_case_id[bundle.cases[0].case_id].signals}
    later_signals = {signal.code for signal in plan_by_case_id[bundle.cases[1].case_id].signals}

    assert "trace_error" not in earlier_signals
    assert "retry_attempt" not in earlier_signals
    assert "trace_error" in later_signals
    assert "retry_attempt" not in later_signals


@pytest.mark.asyncio
async def test_replays_selected_conversation_prefix_and_reports_reproduction() -> None:
    case = materialize_trace_replay_bundle(_trace_records()).cases[1]
    target = _ReplayTarget()

    result = await run_trace_replay(
        case, target, repetitions=2, max_target_calls=14, allow_network_egress=True
    )

    assert result.status == "reproduced"
    assert result.response_match_count == 2
    assert result.state_match_count == 2
    execution_evidence = result.trials[0].execution_evidence
    assert execution_evidence is not None
    assert execution_evidence.initial_state == EnvironmentStateEvidence(
        value={},
        authority="environment_self_reported",
    )
    assert execution_evidence.final_response == "Submitted AC-100."
    assert execution_evidence.final_state == EnvironmentStateEvidence(
        value={"invoice": "AC-100", "status": "submitted"},
        authority="environment_self_reported",
    )
    assert target.conversations == [
        ("Pay AC-100.", "Approve and submit it."),
        ("Pay AC-100.", "Approve and submit it."),
    ]


@pytest.mark.asyncio
async def test_replay_reports_observed_drift_without_claiming_correctness() -> None:
    case = materialize_trace_replay_bundle(_trace_records()).cases[1]

    result = await run_trace_replay(
        case,
        _ReplayTarget(drift=True),
        repetitions=2,
        max_target_calls=14,
        allow_network_egress=True,
    )

    assert result.status == "drifted"
    assert result.response_match_count == 0
    assert result.state_match_count == 0


@pytest.mark.asyncio
async def test_groups_replay_failures_by_stable_evidence_signature() -> None:
    stateful_case = materialize_trace_replay_bundle(_trace_records()).cases[1]
    response_only_case = materialize_trace_replay_bundle(_trace_records(include_state=False)).cases[
        1
    ]
    state_and_response_drift = await run_trace_replay(
        stateful_case,
        _ReplayTarget(drift=True),
        repetitions=1,
        max_target_calls=7,
        allow_network_egress=True,
    )
    response_drift = await run_trace_replay(
        response_only_case,
        _ReplayTarget(drift=True, include_state=False),
        repetitions=1,
        max_target_calls=7,
        allow_network_egress=True,
    )
    reproduced = await run_trace_replay(
        stateful_case,
        _ReplayTarget(),
        repetitions=1,
        max_target_calls=7,
        allow_network_egress=True,
    )

    grouping = group_trace_replay_differences(
        (state_and_response_drift, response_drift, reproduced)
    )

    assert grouping.result_count == 3
    assert grouping.difference_count == 2
    assert grouping.reproduced_count == 1
    assert [group.signature for group in grouping.groups] == [
        "response_mismatch",
        "response_mismatch+state_mismatch",
    ]
    assert grouping.groups[1].members[0].source_trace_id == "aa" * 16
    assert grouping.groups[1].members[0].source_span_ids == ("2222222222222222",)


@pytest.mark.asyncio
async def test_replay_stops_after_uncertain_environment_state() -> None:
    case = materialize_trace_replay_bundle(_trace_records()).cases[1]
    environment = _UncertainReplayEnvironment()

    result = await run_trace_replay(
        case, environment, repetitions=3, max_target_calls=21, allow_network_egress=True
    )

    assert result.status == "inconclusive"
    assert environment.execution_count == 1
    assert result.trials[0].execution_evidence is not None
    assert all(trial.inconclusive_reason is not None for trial in result.trials)

    grouping = group_trace_replay_differences((result,))

    assert grouping.groups[0].signature == (
        "environment_lifecycle_failed+environment_state_uncertain"
    )


@pytest.mark.asyncio
async def test_replay_without_recorded_state_compares_response_only() -> None:
    case = materialize_trace_replay_bundle(_trace_records(include_state=False)).cases[1]

    result = await run_trace_replay(
        case,
        _ReplayTarget(include_state=False),
        repetitions=2,
        max_target_calls=14,
        allow_network_egress=True,
    )

    assert result.status == "reproduced"
    assert result.response_match_count == 2
    assert result.state_match_count is None


@pytest.mark.asyncio
async def test_replay_result_json_round_trip_preserves_nested_tuple_models(
    tmp_path: Path,
) -> None:
    case = materialize_trace_replay_bundle(_trace_records()).cases[1]
    result = await run_trace_replay(
        case,
        _ReplayTarget(),
        repetitions=1,
        max_target_calls=7,
        allow_network_egress=True,
    )
    path = tmp_path / "result.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    loaded = load_trace_replay_result(path)

    assert loaded == result
    assert isinstance(loaded.trials, tuple)
    assert isinstance(loaded.trials[0].outputs, tuple)
    assert loaded.trials[0].execution_evidence is not None
    assert isinstance(loaded.trials[0].execution_evidence.turns, tuple)
    assert isinstance(loaded.trials[0].execution_evidence.lifecycle.completed_phases, tuple)


@pytest.mark.asyncio
async def test_replay_result_rejects_aggregate_counts_that_disagree_with_trials() -> None:
    case = materialize_trace_replay_bundle(_trace_records()).cases[1]
    result = await run_trace_replay(
        case,
        _ReplayTarget(),
        repetitions=1,
        max_target_calls=7,
        allow_network_egress=True,
    )

    with pytest.raises(ValueError, match="response match count"):
        TraceReplayResult.model_validate({**result.model_dump(), "response_match_count": 0})
    with pytest.raises(ValueError, match="state match count"):
        TraceReplayResult.model_validate({**result.model_dump(), "state_match_count": 0})
    conclusive_trial_without_state_comparison = result.trials[0].model_copy(
        update={"state_matches_recorded": None}
    )
    with pytest.raises(ValueError, match="recorded state comparison"):
        TraceReplayResult.model_validate(
            {
                **result.model_dump(exclude={"trials"}),
                "state_match_count": 0,
                "trials": (conclusive_trial_without_state_comparison,),
            }
        )


@pytest.mark.asyncio
async def test_replay_result_rejects_state_comparison_when_recorded_state_is_unavailable() -> None:
    case = materialize_trace_replay_bundle(_trace_records(include_state=False)).cases[1]
    result = await run_trace_replay(
        case,
        _ReplayTarget(include_state=False),
        repetitions=1,
        max_target_calls=7,
        allow_network_egress=True,
    )
    trial_with_state_comparison = result.trials[0].model_copy(
        update={"state_matches_recorded": True}
    )

    with pytest.raises(ValueError, match="unavailable recorded state"):
        TraceReplayResult.model_validate(
            {
                **result.model_dump(exclude={"trials"}),
                "state_match_count": None,
                "trials": (trial_with_state_comparison,),
            }
        )


def test_replay_result_rejects_excessive_trial_count() -> None:
    case = materialize_trace_replay_bundle(_trace_records()).cases[1]
    trials = tuple(
        TraceReplayTrial.model_construct(repetition=index, inconclusive_reason="unknown")
        for index in range(1, 102)
    )

    with pytest.raises(ValueError):
        TraceReplayResult(
            case=case,
            requested_repetitions=100,
            required_target_calls=101,
            status="inconclusive",
            response_match_count=0,
            state_match_count=0,
            trials=trials,
        )


@pytest.mark.asyncio
async def test_replay_rejects_excessive_repetitions_before_environment_calls() -> None:
    case = materialize_trace_replay_bundle(_trace_records()).cases[1]
    environment = _ReplayTarget()

    with pytest.raises(ValueError, match="must not exceed 100"):
        await run_trace_replay(
            case,
            environment,
            repetitions=101,
            max_target_calls=1_000,
            allow_network_egress=True,
        )

    assert environment.conversations == []


@pytest.mark.asyncio
async def test_grouping_keeps_mismatch_reasons_when_another_trial_is_inconclusive() -> None:
    case = materialize_trace_replay_bundle(_trace_records()).cases[1]
    drifted = await run_trace_replay(
        case,
        _ReplayTarget(drift=True),
        repetitions=1,
        max_target_calls=7,
        allow_network_egress=True,
    )
    mixed_result = TraceReplayResult(
        case=case,
        requested_repetitions=2,
        required_target_calls=14,
        status="inconclusive",
        response_match_count=0,
        state_match_count=0,
        trials=(
            drifted.trials[0],
            TraceReplayTrial(
                repetition=2,
                inconclusive_reason="environment execution timed out",
            ),
        ),
    )

    grouping = group_trace_replay_differences((mixed_result,))

    assert grouping.groups[0].reason_codes == (
        "response_mismatch",
        "state_mismatch",
        "environment_execution_timeout",
    )


def test_grouping_uses_stable_fallback_for_unknown_inconclusive_reason() -> None:
    case = materialize_trace_replay_bundle(_trace_records(include_state=False)).cases[1]
    result = TraceReplayResult(
        case=case,
        requested_repetitions=1,
        required_target_calls=1,
        status="inconclusive",
        response_match_count=0,
        state_match_count=None,
        trials=(TraceReplayTrial(repetition=1, inconclusive_reason="new reason"),),
    )

    grouping = group_trace_replay_differences((result,))

    assert grouping.groups[0].reason_codes == ("other_inconclusive",)


@pytest.mark.asyncio
async def test_group_result_loader_enforces_cumulative_byte_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = materialize_trace_replay_bundle(_trace_records()).cases[1]
    result = await run_trace_replay(
        case,
        _ReplayTarget(),
        repetitions=1,
        max_target_calls=7,
        allow_network_egress=True,
    )
    encoded = result.model_dump_json().encode("utf-8")
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_bytes(encoded)
    second_path.write_bytes(encoded)
    monkeypatch.setattr(trace_replay_module, "_MAXIMUM_GROUP_RESULT_BYTES", len(encoded) + 1)

    with pytest.raises(ValueError, match="cumulative size limit"):
        load_trace_replay_results((first_path, second_path))


def test_group_result_loader_rejects_too_many_paths_before_reading() -> None:
    paths = (Path(f"missing-{index}.json") for index in range(101))

    with pytest.raises(ValueError, match="at most 100 result files"):
        load_trace_replay_results(paths)


def test_private_bundle_loader_rejects_tampered_source_envelope(tmp_path: Path) -> None:
    bundle = materialize_trace_replay_bundle(_trace_records())
    path = tmp_path / "bundle.json"
    raw = bundle.model_dump(mode="json")
    raw["envelopes"][0]["scenario"]["trace_id"] = "tampered"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid"):
        load_trace_replay_bundle(path)


def test_private_bundle_loader_rejects_rehashed_case_not_derived_from_envelope(
    tmp_path: Path,
) -> None:
    raw = materialize_trace_replay_bundle(_trace_records()).model_dump(mode="json")
    case = raw["cases"][0]
    case["source_span_ids"] = ["ff" * 8]
    canonical = json.dumps(
        {key: value for key, value in case.items() if key != "case_id"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    case["case_id"] = f"ultr_v1_{hashlib.sha256(canonical).hexdigest()}"
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid"):
        load_trace_replay_bundle(path)
