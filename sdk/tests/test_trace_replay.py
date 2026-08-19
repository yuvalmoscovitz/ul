from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from ul.otlp_ingest import OtlpMappingConfig, parse_otlp_traces
from ul.trace_replay import (
    load_trace_replay_bundle,
    materialize_trace_replay_bundle,
    run_trace_replay,
)
from ul_core.dataset import ObservedAgentOutput
from ul_core.evaluation import (
    EvaluationCase,
    ExecutionEvidence,
    SandboxCapabilities,
    SandboxLifecycleEvidence,
    SandboxStateEvidence,
    SandboxTurnEvidence,
)


def _attribute(key: str, value: str) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": value}}


def _trace_records(*, include_state: bool = True) -> tuple[Any, ...]:
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
        inputs: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        attributes = [
            _attribute("gen_ai.operation.name", "invoke_agent"),
            _attribute("gen_ai.input.messages", json.dumps(inputs)),
            _attribute("gen_ai.output.messages", json.dumps(outputs)),
        ]
        if include_state:
            attributes.append(_attribute("ul.state.snapshot", json.dumps(state)))
        return {
            "traceId": "aa" * 16,
            "spanId": span_id,
            "name": "invoice agent turn",
            "startTimeUnixNano": start,
            "attributes": attributes,
        }

    data = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            span(
                                "11" * 8,
                                "1",
                                [first_user],
                                [first_response],
                                {"invoice": "AC-100", "status": "prepared"},
                            ),
                            span(
                                "22" * 8,
                                "2",
                                [first_user, first_response, second_user],
                                [second_response],
                                {"invoice": "AC-100", "status": "submitted"},
                            ),
                        ]
                    }
                ]
            }
        ]
    }
    return parse_otlp_traces(
        data,
        mapping=OtlpMappingConfig(include_raw_content=True),
    ).records


class _ReplayTarget:
    sandbox_id = "trace-replay-test-sandbox"
    config_sha256 = "0" * 64
    capabilities = SandboxCapabilities(
        supports_conversations=True,
        supports_state_observation=True,
        state_observation_authority="sandbox_self_reported",
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
            sandbox_id=self.sandbox_id,
            sandbox_config_sha256=self.config_sha256,
            initial_state=SandboxStateEvidence(
                value={},
                authority="sandbox_self_reported",
            ),
            turns=tuple(
                SandboxTurnEvidence(
                    turn_id=turn.id,
                    response=output.raw_output,
                    state_snapshot=snapshot,
                    state_observation_authority="sandbox_self_reported",
                )
                for turn, output, snapshot in zip(case.turns, outputs, snapshots, strict=True)
            ),
            final_response=outputs[-1].raw_output,
            final_state=SandboxStateEvidence(
                value=snapshots[-1],
                authority="sandbox_self_reported",
            ),
            lifecycle=SandboxLifecycleEvidence(
                terminal_status="succeeded",
                completed_phases=("execute", "cleanup"),
                delivery="certain",
                cleanup="succeeded",
                sandbox_state_uncertain=False,
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


class _UncertainReplaySandbox(_ReplayTarget):
    def __init__(self) -> None:
        super().__init__()
        self.execution_count = 0

    async def execute(self, case: EvaluationCase) -> ExecutionEvidence:
        self.execution_count += 1
        if self.execution_count > 1:
            raise AssertionError("uncertain sandbox must not be called again")
        return ExecutionEvidence(
            case_id=case.id,
            sandbox_id=self.sandbox_id,
            sandbox_config_sha256=self.config_sha256,
            lifecycle=SandboxLifecycleEvidence(
                terminal_status="failed",
                failed_phase="execute_turn",
                failure_code="transport_failed",
                failure_reason="sandbox API transport failed",
                delivery="uncertain",
                cleanup="succeeded",
                sandbox_state_uncertain=True,
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
    assert execution_evidence.initial_state == SandboxStateEvidence(
        value={},
        authority="sandbox_self_reported",
    )
    assert execution_evidence.final_response == "Submitted AC-100."
    assert execution_evidence.final_state == SandboxStateEvidence(
        value={"invoice": "AC-100", "status": "submitted"},
        authority="sandbox_self_reported",
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
async def test_replay_stops_after_uncertain_sandbox_state() -> None:
    case = materialize_trace_replay_bundle(_trace_records()).cases[1]
    sandbox = _UncertainReplaySandbox()

    result = await run_trace_replay(
        case, sandbox, repetitions=3, max_target_calls=21, allow_network_egress=True
    )

    assert result.status == "inconclusive"
    assert sandbox.execution_count == 1
    assert result.trials[0].execution_evidence is not None
    assert all(trial.inconclusive_reason is not None for trial in result.trials)


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
