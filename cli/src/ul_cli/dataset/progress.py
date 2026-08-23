from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable
from typing import Literal, Protocol

from pydantic import ConfigDict, Field, model_validator
from rich.console import Console
from ul import DatasetTrialProgress
from ul_core.models import ULModel

CampaignStage = Literal[
    "smoke",
    "preflight",
    "augmentation",
    "original",
    "probe",
    "evidence",
    "report",
    "terminal",
]


class ProgressTextStream(Protocol):
    def write(self, value: str) -> object: ...

    def flush(self) -> object: ...


class _ProgressModel(ULModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class CampaignWork(_ProgressModel):
    completed: int = Field(ge=0)
    running: int = Field(ge=0)
    skipped: int = Field(ge=0)
    failed: int = Field(ge=0)
    remaining_upper_bound: int = Field(ge=0)
    remaining_is_upper_bound: Literal[True] = True


class CampaignUsage(_ProgressModel):
    target_calls: int | None = Field(default=None, ge=0)
    semantic_calls: int | None = Field(default=None, ge=0)
    environment_calls: int | None = Field(default=None, ge=0)
    tokens: int | None = Field(default=None, ge=0)
    remaining_target_call_budget: int = Field(ge=0)
    remaining_semantic_call_budget: int = Field(ge=0)
    remaining_environment_call_budget: int = Field(ge=0)
    remaining_token_budget: int = Field(ge=0)


class CampaignTiming(_ProgressModel):
    elapsed_seconds: float = Field(ge=0)
    eta_seconds: float | None = Field(default=None, ge=0)
    maximum_wall_time_seconds: float = Field(gt=0)


class CampaignPosition(_ProgressModel):
    stage: CampaignStage
    case_number: int | None = Field(default=None, ge=1)
    case_count: int = Field(ge=1)
    operator_id: str | None = None
    trial_kind: Literal["original", "probe"] | None = None
    repetition: int | None = Field(default=None, ge=1)
    attempt: int | None = Field(default=None, ge=1)


class CampaignProgressEvent(_ProgressModel):
    schema_version: Literal["ul.campaign.progress.v1"] = "ul.campaign.progress.v1"
    sequence: int = Field(ge=1)
    status: Literal["running", "completed", "paused", "cancelled", "failed"]
    position: CampaignPosition
    work: CampaignWork
    usage: CampaignUsage
    timing: CampaignTiming
    environment: Literal["reusable", "quarantined", "awaiting_cleanup"]
    delivery_uncertain: bool = False
    next_command: Literal["ul report", "ul dataset evaluate --resume", "ul diagnose"] | None = None

    @model_validator(mode="after")
    def validate_terminal_command(self) -> CampaignProgressEvent:
        terminal = self.status != "running"
        if terminal != (self.next_command is not None):
            raise ValueError("terminal progress must contain exactly one next command")
        if self.delivery_uncertain and self.environment != "quarantined":
            raise ValueError("uncertain delivery requires a quarantined environment")
        return self


class CampaignProgressRenderer(Protocol):
    def render(self, event: CampaignProgressEvent) -> None: ...


class JsonCampaignProgressRenderer:
    def __init__(self, stream: ProgressTextStream) -> None:
        self._stream = stream

    def render(self, event: CampaignProgressEvent) -> None:
        self._stream.write(json.dumps(event.model_dump(mode="json"), separators=(",", ":")) + "\n")
        self._stream.flush()


class TerminalCampaignProgressRenderer:
    def __init__(self, console: Console) -> None:
        self._console = console

    def render(self, event: CampaignProgressEvent) -> None:
        payload = event.model_dump(mode="json")
        position = payload["position"]
        work = payload["work"]
        usage = payload["usage"]
        timing = payload["timing"]
        eta = "unknown" if timing["eta_seconds"] is None else f"~{timing['eta_seconds']:.1f}s"
        case = (
            "none"
            if position["case_number"] is None
            else f"{position['case_number']}/{position['case_count']}"
        )
        operator = position["operator_id"] or "none"
        trial = position["trial_kind"] or "none"
        repetition = position["repetition"] or "none"
        attempt = position["attempt"] or "none"
        actual_usage = ",".join(
            f"{name}={usage[name] if usage[name] is not None else 'unknown'}"
            for name in ("target_calls", "semantic_calls", "environment_calls", "tokens")
        )
        remaining_budget = ",".join(
            f"{name.removeprefix('remaining_')}={usage[name]}"
            for name in (
                "remaining_target_call_budget",
                "remaining_semantic_call_budget",
                "remaining_environment_call_budget",
                "remaining_token_budget",
            )
        )
        next_command = f" next={event.next_command}" if event.next_command is not None else ""
        self._console.print(
            f"[{event.sequence}] schema={event.schema_version} {event.status} "
            f"stage={position['stage']} case={case} "
            f"operator={operator} trial={trial} repetition={repetition} attempt={attempt} "
            f"work=completed:{work['completed']},running:{work['running']},"
            f"skipped:{work['skipped']},failed:{work['failed']},"
            f"remaining<={work['remaining_upper_bound']} usage={actual_usage} "
            f"budget={remaining_budget} elapsed={timing['elapsed_seconds']:.1f}s eta={eta} "
            f"max_wall={timing['maximum_wall_time_seconds']:.1f}s "
            f"environment={event.environment} delivery_uncertain={event.delivery_uncertain}"
            f"{next_command}",
            markup=False,
            highlight=False,
            soft_wrap=True,
        )


class SafeCampaignProgressPublisher:
    def __init__(self, renderer: CampaignProgressRenderer) -> None:
        self._renderer = renderer

    def publish(self, event: CampaignProgressEvent) -> None:
        try:
            self._renderer.render(event)
        except Exception:
            return


class CampaignControl:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request: Literal["pause", "cancel"] | None = None

    def request_pause(self) -> None:
        with self._lock:
            if self._request is None:
                self._request = "pause"

    def request_cancel(self) -> None:
        with self._lock:
            self._request = "cancel"

    def requested_action(self) -> Literal["pause", "cancel"] | None:
        with self._lock:
            return self._request


class CampaignProgressTracker:
    def __init__(
        self,
        *,
        case_count: int,
        work_upper_bound: int,
        target_call_budget: int,
        semantic_call_budget: int,
        environment_call_budget: int,
        token_budget: int,
        maximum_wall_time_seconds: float,
        publish: Callable[[CampaignProgressEvent], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if case_count < 1 or work_upper_bound < 1:
            raise ValueError("campaign progress requires positive case and work counts")
        if any(
            budget < 0
            for budget in (
                target_call_budget,
                semantic_call_budget,
                environment_call_budget,
                token_budget,
            )
        ):
            raise ValueError("campaign progress budgets cannot be negative")
        if not math.isfinite(maximum_wall_time_seconds) or maximum_wall_time_seconds <= 0:
            raise ValueError("maximum wall time must be positive and finite")
        self._case_count = case_count
        self._work_upper_bound = work_upper_bound
        self._budgets = (
            target_call_budget,
            semantic_call_budget,
            environment_call_budget,
            token_budget,
        )
        self._maximum_wall_time_seconds = maximum_wall_time_seconds
        self._publish = publish
        self._clock = clock
        self._started_at = clock()
        self._sequence = 0
        self._completed = 0
        self._running = 0
        self._skipped = 0
        self._failed = 0
        self._actual_usage: tuple[int | None, int | None, int | None, int | None] = (
            None,
            None,
            None,
            None,
        )

    def trial_callback(self, *, case_number: int) -> Callable[[DatasetTrialProgress], None]:
        def report(progress: DatasetTrialProgress) -> None:
            if progress.status == "running":
                self._running = 1
            else:
                self._running = 0
                if progress.status == "completed":
                    self._completed += 1
                elif progress.status == "skipped":
                    self._skipped += 1
                else:
                    self._failed += 1
            self.emit(
                status="running",
                stage=progress.kind,
                case_number=case_number,
                operator_id=progress.operator_id,
                trial_kind=progress.kind,
                repetition=progress.repetition,
                attempt=progress.attempt,
                environment=progress.environment,
                delivery_uncertain=progress.delivery_uncertain,
            )

        return report

    def record_usage(
        self,
        *,
        target_calls: int | None,
        semantic_calls: int | None,
        environment_calls: int | None,
        tokens: int | None,
    ) -> None:
        values = (target_calls, semantic_calls, environment_calls, tokens)
        if any(value is not None and value < 0 for value in values):
            raise ValueError("campaign usage cannot be negative")
        self._actual_usage = values

    def emit(
        self,
        *,
        status: Literal["running", "completed", "paused", "cancelled", "failed"],
        stage: CampaignStage,
        case_number: int | None = None,
        operator_id: str | None = None,
        trial_kind: Literal["original", "probe"] | None = None,
        repetition: int | None = None,
        attempt: int | None = None,
        environment: Literal["reusable", "quarantined", "awaiting_cleanup"] = "reusable",
        delivery_uncertain: bool = False,
    ) -> CampaignProgressEvent:
        self._sequence += 1
        elapsed = max(0.0, self._clock() - self._started_at)
        terminal_count = self._completed + self._skipped + self._failed
        remaining = max(0, self._work_upper_bound - terminal_count - self._running)
        eta = None
        if terminal_count >= 3 and remaining > 0:
            eta = elapsed / terminal_count * remaining
        actual_target, actual_semantic, actual_environment, actual_tokens = self._actual_usage
        target_budget, semantic_budget, environment_budget, token_budget = self._budgets
        next_command = None
        if status == "completed":
            next_command = "ul report"
        elif status in {"paused", "cancelled"}:
            next_command = "ul dataset evaluate --resume"
        elif status == "failed":
            next_command = "ul diagnose"
        event = CampaignProgressEvent(
            sequence=self._sequence,
            status=status,
            position=CampaignPosition(
                stage=stage,
                case_number=case_number,
                case_count=self._case_count,
                operator_id=operator_id,
                trial_kind=trial_kind,
                repetition=repetition,
                attempt=attempt,
            ),
            work=CampaignWork(
                completed=self._completed,
                running=self._running,
                skipped=self._skipped,
                failed=self._failed,
                remaining_upper_bound=remaining,
                remaining_is_upper_bound=True,
            ),
            usage=CampaignUsage(
                target_calls=actual_target,
                semantic_calls=actual_semantic,
                environment_calls=actual_environment,
                tokens=actual_tokens,
                remaining_target_call_budget=max(0, target_budget - (actual_target or 0)),
                remaining_semantic_call_budget=max(0, semantic_budget - (actual_semantic or 0)),
                remaining_environment_call_budget=max(
                    0, environment_budget - (actual_environment or 0)
                ),
                remaining_token_budget=max(0, token_budget - (actual_tokens or 0)),
            ),
            timing=CampaignTiming(
                elapsed_seconds=elapsed,
                eta_seconds=eta,
                maximum_wall_time_seconds=self._maximum_wall_time_seconds,
            ),
            environment=environment,
            delivery_uncertain=delivery_uncertain,
            next_command=next_command,
        )
        self._publish(event)
        return event

    def safe_boundary(
        self,
        control: CampaignControl,
        durable_flush: Callable[[], None],
    ) -> bool:
        action = control.requested_action()
        if action is None:
            return True
        durable_flush()
        self.emit(
            status="paused" if action == "pause" else "cancelled",
            stage="terminal",
        )
        return False
