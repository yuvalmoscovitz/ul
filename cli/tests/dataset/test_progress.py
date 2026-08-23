from __future__ import annotations

import io
import json

from rich.console import Console
from ul import DatasetTrialProgress
from ul_cli.dataset.progress import (
    CampaignControl,
    CampaignProgressTracker,
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
    assert actions[0].next_command == "ul dataset evaluate --resume"


def test_renderer_failure_cannot_affect_campaign_execution() -> None:
    class BrokenRenderer:
        def render(self, event: object) -> None:
            del event
            raise OSError("terminal unavailable")

    publisher = SafeCampaignProgressPublisher(BrokenRenderer())
    clock_values = iter((1.0, 1.1))
    tracker = _tracker(publisher.publish, lambda: next(clock_values))

    tracker.emit(status="running", stage="preflight")
