from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import ConfigDict, Field, JsonValue, model_validator

from ul_core.models import ConversationTurn, ULModel

StateObservationAuthority = Literal["sandbox_self_reported", "independent_observer"]
TimeoutAfterCommitTriggerStatus = Literal["unknown", "not_fired", "fired"]
SandboxLifecycleFailureCode = Literal[
    "authentication_rejected",
    "rate_limited",
    "http_status",
    "response_content_type",
    "response_content_encoding",
    "invalid_json",
    "null_json",
    "response_mapping",
    "sandbox_identity",
    "case_identity",
    "turn_identity",
    "reset_generation",
    "reset_generation_reused",
    "reset_session_not_acknowledged",
    "reset_env_not_acknowledged",
    "reset_not_clean",
    "request_too_large",
    "response_too_large",
    "call_budget",
    "request_timeout",
    "connect_timeout",
    "dns_resolution",
    "tls_connection",
    "connect_failed",
    "response_timeout",
    "write_timeout",
    "pool_timeout",
    "transport_protocol",
    "transport_failed",
    "sandbox_state_uncertain",
    "sandbox_lifecycle_error",
    "sandbox_cleanup_error",
]


class _StrictModel(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProductionObservation(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    id: str = Field(min_length=1, max_length=500)
    source_kind: str = Field(min_length=1, max_length=100)
    source_reference: str = Field(min_length=1, max_length=1_000)
    session_id: str | None = Field(default=None, min_length=1, max_length=500)
    turn_id: str | None = Field(default=None, min_length=1, max_length=500)
    observed_input: JsonValue
    observed_output: JsonValue | None = None
    observed_state: JsonValue | None = None
    observed_state_authority: (
        Literal["production_trace_self_reported", "independent_observer"] | None
    ) = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_observed_state_authority(self) -> Self:
        if (self.observed_state is None) != (self.observed_state_authority is None):
            raise ValueError("observed state and its authority must be provided together")
        return self


class ProductionSourcePage(_StrictModel):
    observations: tuple[ProductionObservation, ...]
    next_checkpoint: str | None = Field(default=None, min_length=1, max_length=1_000)


class TimeoutAfterCommitEventRequest(_StrictModel):
    operator_id: Literal["environment.tool.timeout_after_commit"] = (
        "environment.tool.timeout_after_commit"
    )
    operator_version: Literal["1.0.0"] = "1.0.0"
    event_id: str = Field(min_length=1, max_length=500)
    turn_id: str = Field(min_length=1, max_length=500)
    action_id: str = Field(min_length=1, max_length=500)


class TimeoutAfterCommitEventEvidence(TimeoutAfterCommitEventRequest):
    authority: Literal["sandbox_self_reported"] = "sandbox_self_reported"
    requested: Literal[True] = True
    armed: bool
    trigger_status: TimeoutAfterCommitTriggerStatus
    cleaned: bool

    @model_validator(mode="after")
    def validate_trigger_status(self) -> Self:
        if not self.armed and self.trigger_status != "unknown":
            raise ValueError("an unarmed timeout-after-commit event cannot have a trigger result")
        return self


class EvaluationCase(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    id: str = Field(min_length=1, max_length=500)
    turns: tuple[ConversationTurn, ...] = Field(min_length=1)
    max_sandbox_api_calls: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    required_state_observation_authority: StateObservationAuthority | None = None
    required_state_observer_id: str | None = Field(default=None, min_length=1, max_length=500)
    timeout_after_commit_event: TimeoutAfterCommitEventRequest | None = None

    @model_validator(mode="after")
    def validate_identifiers(self) -> Self:
        if not math.isfinite(self.timeout_seconds):
            raise ValueError("evaluation case timeout must be finite")
        turn_ids = tuple(turn.id for turn in self.turns)
        if len(turn_ids) != len(set(turn_ids)):
            raise ValueError("evaluation case turn identifiers must be unique")
        if (
            self.timeout_after_commit_event is not None
            and self.timeout_after_commit_event.turn_id not in turn_ids
        ):
            raise ValueError("timeout-after-commit event must reference a case turn")
        if self.required_state_observation_authority == "independent_observer":
            if self.required_state_observer_id is None:
                raise ValueError("independent state requirements need an observer identifier")
        elif self.required_state_observer_id is not None:
            raise ValueError("self-reported state requirements cannot name an observer")
        return self


class SandboxCapabilities(_StrictModel):
    isolation: Literal["customer_managed"] = "customer_managed"
    supports_conversations: bool
    supports_state_observation: bool
    state_observation_authority: StateObservationAuthority | None = None
    state_observer_id: str | None = Field(default=None, min_length=1, max_length=500)
    cancellation_guarantee: Literal["none", "best_effort", "guaranteed"]
    timeout_after_commit_version: Literal["1.0.0"] | None = None

    @model_validator(mode="after")
    def validate_state_observation_authority(self) -> Self:
        if self.supports_state_observation != (self.state_observation_authority is not None):
            raise ValueError(
                "state observation support and its authority must be declared together"
            )
        if self.state_observation_authority == "independent_observer":
            if self.state_observer_id is None:
                raise ValueError("independent state support requires an observer identifier")
        elif self.state_observer_id is not None:
            raise ValueError("self-reported state support cannot name an observer")
        return self


class SandboxTurnEvidence(_StrictModel):
    turn_id: str = Field(min_length=1, max_length=500)
    response: JsonValue
    state_snapshot: JsonValue | None = None
    state_observation_authority: StateObservationAuthority | None = None
    state_observer_id: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_state_evidence(self) -> Self:
        if (self.state_snapshot is None) != (self.state_observation_authority is None):
            raise ValueError("state snapshot and its authority must be provided together")
        if self.state_observation_authority == "independent_observer":
            if self.state_observer_id is None:
                raise ValueError("independent state evidence requires an observer identifier")
        elif self.state_observer_id is not None:
            raise ValueError("self-reported state evidence cannot name an independent observer")
        return self


class SandboxStateEvidence(_StrictModel):
    value: JsonValue
    authority: StateObservationAuthority
    observer_id: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_observer(self) -> Self:
        if self.authority == "independent_observer":
            if self.observer_id is None:
                raise ValueError("independent state evidence requires an observer identifier")
        elif self.observer_id is not None:
            raise ValueError("self-reported state evidence cannot name an observer")
        return self


class SandboxLifecycleEvidence(_StrictModel):
    terminal_status: Literal["succeeded", "failed", "timed_out", "cancelled"]
    completed_phases: tuple[str, ...] = ()
    failed_phase: str | None = Field(default=None, min_length=1, max_length=200)
    failure_code: SandboxLifecycleFailureCode | None = None
    failure_reason: str | None = Field(default=None, min_length=1, max_length=500)
    delivery: Literal["certain", "uncertain"]
    cleanup: Literal["succeeded", "failed", "not_attempted"]
    cleanup_failure_code: SandboxLifecycleFailureCode | None = None
    cleanup_failure_reason: str | None = Field(default=None, min_length=1, max_length=500)
    sandbox_state_uncertain: bool
    reset_session_requested: bool = True
    reset_session_acknowledged: bool = True
    reset_env_requested: bool = True
    reset_env_acknowledged: bool = True

    @model_validator(mode="after")
    def validate_terminal_status(self) -> Self:
        if self.reset_session_acknowledged and not self.reset_session_requested:
            raise ValueError("session reset cannot be acknowledged when it was not requested")
        if self.reset_env_acknowledged and not self.reset_env_requested:
            raise ValueError("environment reset cannot be acknowledged when it was not requested")
        if self.terminal_status == "succeeded" and (
            self.reset_session_requested != self.reset_session_acknowledged
            or self.reset_env_requested != self.reset_env_acknowledged
        ):
            raise ValueError("successful lifecycle evidence requires every requested reset")
        if self.terminal_status == "succeeded" and (
            self.failed_phase is not None
            or self.failure_code is not None
            or self.failure_reason is not None
        ):
            raise ValueError("successful lifecycle evidence cannot name a failure")
        if self.terminal_status == "succeeded" and (
            self.delivery != "certain"
            or self.cleanup != "succeeded"
            or self.sandbox_state_uncertain
        ):
            raise ValueError("successful lifecycle evidence requires certain delivery and cleanup")
        if self.terminal_status != "succeeded" and self.failed_phase is None:
            raise ValueError("unsuccessful lifecycle evidence requires a failed phase")
        if self.terminal_status != "succeeded" and (
            self.failure_code is None or self.failure_reason is None
        ):
            raise ValueError("unsuccessful lifecycle evidence requires a failure code and reason")
        if self.delivery == "uncertain" and not self.sandbox_state_uncertain:
            raise ValueError("uncertain delivery must mark sandbox state uncertain")
        if self.cleanup == "failed" and not self.sandbox_state_uncertain:
            raise ValueError("failed cleanup must mark sandbox state uncertain")
        if (self.cleanup == "failed") != (self.cleanup_failure_reason is not None):
            raise ValueError("cleanup failure detail must match cleanup status")
        if (self.cleanup_failure_code is None) != (self.cleanup_failure_reason is None):
            raise ValueError("cleanup failure code and reason must be provided together")
        return self


class ExecutionEvidence(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    case_id: str = Field(min_length=1, max_length=500)
    sandbox_id: str = Field(min_length=1, max_length=500)
    sandbox_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    initial_state: SandboxStateEvidence | None = None
    turns: tuple[SandboxTurnEvidence, ...] = ()
    final_response: JsonValue | None = None
    final_state: SandboxStateEvidence | None = None
    timeout_after_commit_event: TimeoutAfterCommitEventEvidence | None = None
    lifecycle: SandboxLifecycleEvidence

    @model_validator(mode="after")
    def validate_successful_evidence(self) -> Self:
        if self.lifecycle.terminal_status == "succeeded" and not self.turns:
            raise ValueError("successful execution evidence requires turn evidence")
        if self.lifecycle.terminal_status == "succeeded":
            if (
                self.initial_state is None
                or self.final_response is None
                or self.final_state is None
            ):
                raise ValueError("successful execution evidence requires explicit pre/post state")
            final_turn = self.turns[-1]
            if self.final_response != final_turn.response:
                raise ValueError("final response must match the last turn")
            if (
                self.final_state.value != final_turn.state_snapshot
                or self.final_state.authority != final_turn.state_observation_authority
                or self.final_state.observer_id != final_turn.state_observer_id
            ):
                raise ValueError("final state must match the last turn")
            if self.timeout_after_commit_event is not None and (
                not self.timeout_after_commit_event.armed
                or self.timeout_after_commit_event.trigger_status == "unknown"
                or not self.timeout_after_commit_event.cleaned
            ):
                raise ValueError(
                    "successful timeout-after-commit evidence requires a completed event"
                )
        return self
