from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from rich.console import Console
from ul import (
    DatasetEvaluationResult,
    DatasetEvaluationTrial,
    DatasetTargetDeliveryUncertain,
    DatasetTrialUnit,
    InteractionRecord,
)
from ul_cli import progress_action as progress_action_module
from ul_cli.dataset.evaluation import runner as runner_module
from ul_cli.dataset.progress import (
    CampaignControl,
    CampaignProgressTracker,
    CampaignSignalControl,
    JsonCampaignProgressRenderer,
    SafeCampaignProgressPublisher,
    TerminalCampaignProgressRenderer,
    create_campaign_next_commands,
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
        next_commands=create_campaign_next_commands(Path.cwd() / "tmp" / "test-evidence.jsonl"),
        publish=publish,
        clock=clock,
    )


@pytest.fixture(autouse=True)
def _private_progress_action_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        progress_action_module,
        "_action_receipt_directory",
        lambda: tmp_path / "action-state",
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
    tracker.trial_started(
        case_number=2,
        unit=DatasetTrialUnit(
            interaction_id="PRIVATE_CASE_ID",
            operator_id="input.surface.rephrase",
            arm="probe",
            repetition=3,
        ),
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
    tracker.trial_terminal(
        case_number=1,
        unit=DatasetTrialUnit(
            interaction_id="PRIVATE_CASE_ID",
            operator_id="current_baseline",
            arm="original",
            repetition=1,
        ),
        trial=DatasetEvaluationTrial(
            repetition=1,
            inconclusive_reasons=("sanitized failure",),
        ),
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
    assert actions[0].next_command is not None
    assert actions[0].next_command.action == "resume"
    assert actions[0].next_command.argv[:2] == ("ul", "action")
    assert len(actions[0].next_command.argv[2]) == 32


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


def test_resume_hydrates_mixed_durable_terminal_states_before_first_event() -> None:
    clock_values = iter((1.0, 2.0))
    events = []
    tracker = _tracker(events.append, lambda: next(clock_values))
    tracker.hydrate_terminal_states(
        {
            "completed": "completed",
            "skipped": "rejected",
            "failed": "inconclusive",
            "quarantined": "quarantined",
        }
    )

    tracker.emit(status="running", stage="preflight")

    event = events[0]
    assert event.work.completed == 1
    assert event.work.skipped == 1
    assert event.work.failed == 2
    assert event.work.remaining_upper_bound == 26
    assert event.environment == "quarantined"


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
    assert event.next_command is not None
    assert event.next_command.action == "diagnose"
    assert "PRIVATE_CASE_ID" not in event.model_dump_json()


def test_quarantine_status_is_sticky_until_external_cleanup() -> None:
    clock_values = iter((1.0, 2.0, 3.0, 4.0))
    events = []
    tracker = _tracker(events.append, lambda: next(clock_values))
    uncertain = DatasetTrialUnit(
        interaction_id="PRIVATE_CASE_ID",
        operator_id="current_baseline",
        arm="original",
        repetition=1,
    )
    skipped = DatasetTrialUnit(
        interaction_id="PRIVATE_CASE_ID",
        operator_id="input.surface.rephrase",
        arm="probe",
        repetition=1,
    )

    tracker.trial_started(case_number=1, unit=uncertain)
    tracker.trial_delivery_uncertain(case_number=1, unit=uncertain)
    tracker.trial_skipped(case_number=1, unit=skipped)

    assert events[-1].environment == "quarantined"


def test_eta_excludes_smoke_preflight_and_human_confirmation_idle() -> None:
    clock_values = iter((0.0, 100.0, 900.0, 1_000.0, 1_010.0, 1_020.0, 1_030.0, 1_040.0, 1_050.0))
    events = []
    tracker = _tracker(events.append, lambda: next(clock_values))
    tracker.emit(status="running", stage="smoke")
    tracker.emit(status="running", stage="preflight")
    for repetition in range(1, 4):
        unit = DatasetTrialUnit(
            interaction_id="PRIVATE_CASE_ID",
            operator_id="current_baseline",
            arm="original",
            repetition=repetition,
        )
        tracker.trial_started(case_number=1, unit=unit)
        tracker.trial_terminal(
            case_number=1,
            unit=unit,
            trial=DatasetEvaluationTrial(
                repetition=repetition,
                inconclusive_reasons=("sanitized failure",),
            ),
        )

    assert events[-1].timing.eta_seconds == pytest.approx(450.0)


def test_cancellation_after_delivery_before_semantic_completion_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    post_delivery_semantic_started = asyncio.Event()
    run_calls = 0
    journal_transitions: list[tuple[str, str]] = []

    class FakeJournal:
        snapshot = SimpleNamespace(recovered_trials={})

        def start(self, unit: DatasetTrialUnit) -> None:
            journal_transitions.append((unit.id, "running"))

        def is_terminal(self, _unit: DatasetTrialUnit) -> bool:
            return False

        def terminal(self, unit: DatasetTrialUnit, state: str, _reason: str) -> None:
            journal_transitions.append((unit.id, state))

        def flush(self) -> None:
            pass

    class AsyncContext:
        async def __aenter__(self) -> object:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        def reuse_preflight(self, _result: object) -> None:
            pass

    class FakeRunner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def run(self, *args: object, **kwargs: object) -> DatasetEvaluationResult:
            nonlocal run_calls
            run_calls += 1
            unit = DatasetTrialUnit(
                interaction_id="PRIVATE_CASE_ID",
                operator_id="current_baseline",
                arm="original",
                repetition=1,
            )
            cast(Any, kwargs["trial_started_callback"])(unit)
            post_delivery_semantic_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    monkeypatch.setattr(
        runner_module,
        "create_semantic_model_deconstructor",
        lambda settings: AsyncContext(),
    )
    monkeypatch.setattr(runner_module, "DatasetAugmentationEngine", lambda *args: object())
    monkeypatch.setattr(runner_module, "DatasetEvaluationRunner", FakeRunner)
    output = tmp_path / "evidence.jsonl"

    async def run() -> None:
        with output.open("w", encoding="utf-8") as output_stream:
            task = asyncio.create_task(
                runner_module.evaluate_interaction_records(
                    (
                        InteractionRecord.model_validate(
                            {
                                "id": "PRIVATE_CASE_ID",
                                "raw_input": "PRIVATE_SOURCE_CANARY",
                                "raw_observed_output": {"value": "PRIVATE_TARGET_CANARY"},
                            }
                        ),
                    ),
                    (),
                    cast(Any, SimpleNamespace()),
                    cast(Any, AsyncContext()),
                    output_stream,
                    repetitions=1,
                    max_environment_api_calls=1,
                    planned_target_calls=1,
                    evaluator_preflight=cast(Any, object()),
                    trial_journal=cast(Any, FakeJournal()),
                )
            )
            await post_delivery_semantic_started.wait()
            task.cancel()
            with pytest.raises(DatasetTargetDeliveryUncertain):
                await task

    asyncio.run(run())

    progress_output = capsys.readouterr().err
    assert run_calls == 1
    assert "delivery_uncertain=True" in progress_output
    assert "environment=quarantined" in progress_output
    assert "PRIVATE_CASE_ID" not in progress_output
    assert "PRIVATE_SOURCE_CANARY" not in progress_output
    assert "PRIVATE_TARGET_CANARY" not in progress_output
    assert [state for _, state in journal_transitions] == ["running", "quarantined"]
