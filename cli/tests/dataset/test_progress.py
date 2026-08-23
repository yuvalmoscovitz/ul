from __future__ import annotations

import asyncio
import io
import json

import pytest
from rich.console import Console
from ul import DatasetTrialProgress
from ul_cli.dataset.progress import (
    CampaignControl,
    CampaignNextCommands,
    CampaignProgressTracker,
    CampaignSignalControl,
    JsonCampaignProgressRenderer,
    SafeCampaignProgressPublisher,
    TerminalCampaignProgressRenderer,
)


def _tracker(publish: object, clock: object) -> CampaignProgressTracker:
    return CampaignProgressTracker(
        case_count=10,
        work_upper_bound=30,
        target_call_budget=60,
        semantic_call_budget=40,
        environment_call_budget=200,
        token_budget=50_000,
        maximum_wall_time_seconds=600,
        next_commands=CampaignNextCommands(
            inspect_findings="ul report evidence.jsonl",
            resume="ul dataset evaluate --resume evidence.jsonl",
            diagnose="ul dataset evaluate --resume evidence.jsonl --dry-run",
        ),
        publish=publish,
        clock=clock,
    )


def test_json_and_terminal_renderers_expose_equivalent_progress_facts() -> None:
    clock_values = iter((100.0, 102.5))
    events = []
    tracker = _tracker(events.append, lambda: next(clock_values))
    tracker.record_usage(
        target_calls=4,
        semantic_calls=7,
        environment_calls=12,
        tokens=900,
    )
    callback = tracker.trial_callback(case_number=2)
    callback(
        DatasetTrialProgress(
            status="running",
            kind="probe",
            operator_id="input.surface.rephrase",
            repetition=3,
            environment="awaiting_cleanup",
        )
    )
    event = events[-1]
    json_stream = io.StringIO()
    terminal_stream = io.StringIO()

    JsonCampaignProgressRenderer(json_stream).render(event)
    TerminalCampaignProgressRenderer(
        Console(file=terminal_stream, force_terminal=False, width=300)
    ).render(event)

    payload = json.loads(json_stream.getvalue())
    terminal = terminal_stream.getvalue()
    assert payload["schema_version"] == "ul.campaign.progress.v1"
    assert payload["position"] == {
        "stage": "probe",
        "case_number": 2,
        "case_count": 10,
        "operator_id": "input.surface.rephrase",
        "trial_kind": "probe",
        "repetition": 3,
        "attempt": 1,
    }
    for fact in (
        "stage=probe",
        "case=2/10",
        "operator=input.surface.rephrase",
        "trial=probe",
        "repetition=3",
        "attempt=1",
        "remaining<=29",
        "target_calls=4",
        "semantic_calls=7",
        "environment_calls=12",
        "tokens=900",
        "environment=awaiting_cleanup",
    ):
        assert fact in terminal


def test_progress_contract_never_accepts_or_emits_private_content() -> None:
    canaries = (
        "PRIVATE_SOURCE_CANARY",
        "PRIVATE_TARGET_CANARY",
        "PRIVATE_TOOL_ARGUMENT_CANARY",
        "PRIVATE_STATE_CANARY",
        "sk-live-secret-canary",
        "customer@example.test",
    )
    clock_values = iter((1.0, 2.0))
    events = []
    tracker = _tracker(events.append, lambda: next(clock_values))
    tracker.trial_callback(case_number=1)(
        DatasetTrialProgress(
            status="completed",
            kind="original",
            repetition=1,
            environment="reusable",
        )
    )
    serialized = events[0].model_dump_json()
    assert all(canary not in serialized for canary in canaries)
    assert set(type(events[0].position).model_fields) == {
        "stage",
        "case_number",
        "case_count",
        "operator_id",
        "trial_kind",
        "repetition",
        "attempt",
    }


def test_control_flushes_before_single_terminal_resume_command() -> None:
    clock_values = iter((10.0, 11.0))
    actions = []
    tracker = _tracker(actions.append, lambda: next(clock_values))
    control = CampaignControl()
    order = []
    control.request_pause()

    assert tracker.safe_boundary(control, lambda: order.append("flushed")) is False

    order.append("reported")
    assert order == ["flushed", "reported"]
    assert len(actions) == 1
    assert actions[0].status == "paused"
    assert actions[0].next_command == "ul dataset evaluate --resume evidence.jsonl"


def test_renderer_failure_cannot_affect_campaign_execution() -> None:
    class BrokenRenderer:
        def render(self, event: object) -> None:
            del event
            raise OSError("terminal unavailable")

    publisher = SafeCampaignProgressPublisher(BrokenRenderer())
    clock_values = iter((1.0, 1.1))
    tracker = _tracker(publisher.publish, lambda: next(clock_values))

    tracker.emit(status="running", stage="preflight")


def test_signal_requests_pause_at_safe_boundary() -> None:
    control = CampaignControl()
    signal_control = CampaignSignalControl(control)

    signal_control.interrupt()

    assert control.requested_action() == "pause"


def test_second_signal_requests_cancel_at_safe_boundary() -> None:
    control = CampaignControl()
    signal_control = CampaignSignalControl(control)

    signal_control.interrupt()
    signal_control.interrupt()

    assert control.requested_action() == "cancel"


def test_remaining_budgets_are_unknown_until_actual_usage_is_available() -> None:
    clock_values = iter((1.0, 2.0))
    events = []
    tracker = _tracker(events.append, lambda: next(clock_values))

    tracker.emit(status="running", stage="preflight")

    usage = events[0].usage
    assert usage.remaining_target_call_budget is None
    assert usage.remaining_semantic_call_budget is None
    assert usage.remaining_environment_call_budget is None
    assert usage.remaining_token_budget is None


async def _wait_for_cancellation() -> None:
    await asyncio.Event().wait()


def test_signal_cancels_inflight_target_call_for_uncertainty_handling() -> None:
    async def run() -> None:
        control = CampaignControl()
        signal_control = CampaignSignalControl(control)
        target_task = asyncio.create_task(_wait_for_cancellation())
        signal_control.target_call_started(target_task)

        signal_control.interrupt()
        with pytest.raises(asyncio.CancelledError):
            await target_task

        assert control.requested_action() is None

    asyncio.run(run())


def test_uncertain_delivery_is_terminal_and_quarantined() -> None:
    clock_values = iter((20.0, 21.0, 22.0))
    events = []
    tracker = _tracker(events.append, lambda: next(clock_values))
    unit = DatasetTrialUnit(
        interaction_id="PRIVATE_CASE_ID",
        operator_id="input.surface.rephrase",
        arm="probe",
        repetition=2,
    )
    tracker.trial_started(case_number=3, unit=unit)

    tracker.trial_delivery_uncertain(case_number=3, unit=unit)

    event = events[-1]
    assert event.status == "failed"
    assert event.work.running == 0
    assert event.work.failed == 1
    assert event.environment == "quarantined"
    assert event.delivery_uncertain is True
    assert event.next_command == "ul dataset evaluate --resume evidence.jsonl --dry-run"
    assert "PRIVATE_CASE_ID" not in event.model_dump_json()
