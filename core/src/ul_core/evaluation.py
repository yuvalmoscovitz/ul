from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import ConfigDict, Field, JsonValue, model_validator

from ul_core.evaluators import EvaluatorSpec
from ul_core.models import ConversationTurn, ULModel

StateObservationAuthority = Literal["environment_self_reported", "independent_observer"]
TimeoutAfterCommitTriggerStatus = Literal["unknown", "not_fired", "fired"]
EnvironmentLifecycleFailureCode = Literal[
    "authentication_rejected",
    "rate_limited",
    "http_status",
    "response_content_type",
    "response_content_encoding",
    "invalid_json",
    "null_json",
    "response_contains_credential",
    "response_mapping",
    "environment_identity",
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
    "environment_state_uncertain",
    "environment_lifecycle_error",
    "environment_cleanup_error",
]
EvidenceFact = Literal[
    "response_observed",
    "trajectory_observed",
    "committed_state_verified",
    "deterministic_replay_verified",
]
EvidenceAuthority = Literal[
    "invoker_self_reported",
    "source_self_reported",
    "independent_observer",
    "environment_self_reported",
]


class _StrictModel(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProbeTurn(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    id: str = Field(min_length=1)
    input: JsonValue
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ProbeRequest(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    case_id: str = Field(min_length=1, max_length=500)
    session_id: str = Field(min_length=1, max_length=500)
    correlation_id: str = Field(min_length=1, max_length=500)
    turn: ProbeTurn
    context: dict[str, JsonValue] = Field(default_factory=dict)


class ProbeExecutionEvent(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    id: str = Field(min_length=1, max_length=500)
    correlation_id: str = Field(min_length=1, max_length=500)
    kind: str = Field(min_length=1, max_length=200)
    payload: JsonValue


class ProbeResult(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    id: str = Field(min_length=1, max_length=500)
    correlation_id: str = Field(min_length=1, max_length=500)
    response: JsonValue
    response_size_bytes: int | None = Field(default=None, ge=0)
    response_truncated: bool = False
    execution_events: tuple[ProbeExecutionEvent, ...] = Field(default=(), max_length=1_000)

    @model_validator(mode="after")
    def validate_event_correlations(self) -> Self:
        if any(event.correlation_id != self.correlation_id for event in self.execution_events):
            raise ValueError("execution events must match the result correlation identifier")
        event_ids = tuple(event.id for event in self.execution_events)
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("execution event identifiers must be unique")
        return self


class ObservationRequest(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    case_id: str = Field(min_length=1, max_length=500)
    session_id: str = Field(min_length=1, max_length=500)
    correlation_id: str = Field(min_length=1, max_length=500)
    checkpoint: str | None = Field(default=None, min_length=1, max_length=1_000)
    context: dict[str, JsonValue] = Field(default_factory=dict)


class ProbeObservation(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    id: str = Field(min_length=1, max_length=500)
    source_id: str = Field(min_length=1, max_length=500)
    correlation_id: str = Field(min_length=1, max_length=500)
    authority: Literal["source_self_reported", "independent_observer"]
    status: Literal["complete", "incomplete", "missing"] = "complete"
    limitation: str | None = Field(default=None, min_length=1, max_length=500)
    traces: tuple[JsonValue, ...] = Field(default=(), max_length=1_000)
    tool_calls: tuple[JsonValue, ...] = Field(default=(), max_length=1_000)
    handoffs: tuple[JsonValue, ...] = Field(default=(), max_length=1_000)
    errors: tuple[JsonValue, ...] = Field(default=(), max_length=1_000)
    usage: JsonValue | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    next_checkpoint: str | None = Field(default=None, min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def validate_limitation(self) -> Self:
        if self.status == "complete" and self.limitation is not None:
            raise ValueError("complete observations cannot declare a limitation")
        if self.status != "complete" and self.limitation is None:
            raise ValueError("incomplete or missing observations require a limitation")
        if self.status == "missing" and any(
            (
                self.traces,
                self.tool_calls,
                self.handoffs,
                self.errors,
                self.usage is not None,
                self.metadata,
            )
        ):
            raise ValueError("missing observations cannot contain observed evidence")
        return self


class ProbeExecutionIdentity(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    campaign_id: str = Field(min_length=1, max_length=500)
    case_id: str = Field(min_length=1, max_length=500)
    probe_id: str = Field(min_length=1, max_length=500)
    attempt_id: str = Field(min_length=1, max_length=500)
    session_id: str = Field(min_length=1, max_length=500)
    turn_ids: tuple[str, ...]
    variation_id: str | None = Field(default=None, min_length=1, max_length=500)
    repetition: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_turn_ids(self) -> Self:
        if len(self.turn_ids) != len(set(self.turn_ids)):
            raise ValueError("probe execution turn identifiers must be unique")
        return self


class StateFixtureRequest(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    fixture_id: str = Field(min_length=1, max_length=500)
    case_id: str = Field(min_length=1, max_length=500)
    session_id: str = Field(min_length=1, max_length=500)
    correlation_id: str = Field(min_length=1, max_length=500)
    turn_id: str | None = Field(default=None, min_length=1)
    configuration: dict[str, JsonValue] = Field(default_factory=dict)


class StateSnapshot(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    id: str = Field(min_length=1, max_length=500)
    fixture_id: str = Field(min_length=1, max_length=500)
    correlation_id: str = Field(min_length=1, max_length=500)
    source_id: str = Field(min_length=1, max_length=500)
    value: JsonValue
    authority: StateObservationAuthority
    observer_id: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_observer(self) -> Self:
        if self.authority == "independent_observer":
            if self.observer_id is None:
                raise ValueError("independent state snapshots require an observer identifier")
        elif self.observer_id is not None:
            raise ValueError("self-reported state snapshots cannot name an observer")
        return self


class StateOperationResult(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    id: str = Field(min_length=1, max_length=500)
    fixture_id: str = Field(min_length=1, max_length=500)
    correlation_id: str = Field(min_length=1, max_length=500)
    operation: Literal["reset", "setup", "cleanup"]
    succeeded: bool
    reset_session_requested: bool = False
    reset_session_acknowledged: bool = False
    reset_environment_requested: bool = False
    reset_environment_acknowledged: bool = False
    state_uncertain: bool = False
    failure_code: str | None = Field(default=None, min_length=1, max_length=200)
    failure_reason: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.reset_session_acknowledged and not self.reset_session_requested:
            raise ValueError("session reset cannot be acknowledged when it was not requested")
        if self.reset_environment_acknowledged and not self.reset_environment_requested:
            raise ValueError("environment reset cannot be acknowledged when it was not requested")
        if self.operation == "setup" and any(
            (
                self.reset_session_requested,
                self.reset_session_acknowledged,
                self.reset_environment_requested,
                self.reset_environment_acknowledged,
            )
        ):
            raise ValueError("setup results cannot contain reset acknowledgements")
        if self.succeeded and (
            self.state_uncertain or self.failure_code is not None or self.failure_reason is not None
        ):
            raise ValueError("successful state operations cannot contain failure claims")
        if not self.succeeded and (self.failure_code is None or self.failure_reason is None):
            raise ValueError("failed state operations require a failure code and reason")
        if (self.failure_code is None) != (self.failure_reason is None):
            raise ValueError("state operation failure code and reason must be provided together")
        if self.succeeded and (
            self.reset_session_requested != self.reset_session_acknowledged
            or self.reset_environment_requested != self.reset_environment_acknowledged
        ):
            raise ValueError("successful state operations require every requested reset")
        return self


class ProbeInvokerCapabilities(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    invoker_id: str = Field(min_length=1, max_length=500)
    response_size_limit_bytes: int = Field(ge=1)
    execution_events_size_limit_bytes: int = Field(default=1_000_000, ge=1)
    supports_structured_execution_events: bool = False
    supports_conversations: bool = False
    request_isolation: Literal["not_attested", "per_request_attested"] = "not_attested"
    cancellation_guarantee: Literal["none", "best_effort", "guaranteed"] = "none"


class ObservationSourceCapabilities(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    source_id: str = Field(min_length=1, max_length=500)
    authority: Literal["source_self_reported", "independent_observer"]
    observation_size_limit_bytes: int = Field(default=1_000_000, ge=1)
    supports_traces: bool = False
    supports_tool_calls: bool = False
    supports_handoffs: bool = False
    supports_errors: bool = False
    supports_usage: bool = False
    supports_metadata: bool = False
    counts_toward_environment_api_calls: bool = True

    @property
    def supports_trajectory(self) -> bool:
        return any(
            (
                self.supports_traces,
                self.supports_tool_calls,
                self.supports_handoffs,
                self.supports_errors,
            )
        )


class StateEnvironmentCapabilities(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    environment_id: str = Field(min_length=1, max_length=500)
    snapshot_size_limit_bytes: int = Field(default=1_000_000, ge=1)
    supports_reset: bool = False
    supports_setup: bool = False
    supports_snapshot: bool = False
    supports_cleanup: bool = False
    state_observation_authority: StateObservationAuthority | None = None
    state_observer_id: str | None = Field(default=None, min_length=1, max_length=500)
    supports_deterministic_replay: bool = False

    @model_validator(mode="after")
    def validate_capabilities(self) -> Self:
        if self.supports_snapshot != (self.state_observation_authority is not None):
            raise ValueError("snapshot support and state authority must be declared together")
        if self.state_observation_authority == "independent_observer":
            if self.state_observer_id is None:
                raise ValueError("independent state support requires an observer identifier")
        elif self.state_observer_id is not None:
            raise ValueError("self-reported state support cannot name an observer")
        if self.supports_deterministic_replay and not all(
            (
                self.supports_reset,
                self.supports_snapshot,
                self.supports_cleanup,
            )
        ):
            raise ValueError("deterministic replay requires reset, snapshot, and cleanup support")
        return self


class ProbeCapabilities(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    invoker: ProbeInvokerCapabilities
    observation_source: ObservationSourceCapabilities | None = None
    state_environment: StateEnvironmentCapabilities | None = None


class EvidenceProfile(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    basis: Literal["declared_capabilities"] = "declared_capabilities"
    available_facts: frozenset[EvidenceFact]
    sources: dict[EvidenceFact, str]
    authorities: dict[EvidenceFact, EvidenceAuthority]

    @model_validator(mode="after")
    def validate_fact_provenance(self) -> Self:
        if set(self.sources) != set(self.available_facts):
            raise ValueError("every available evidence fact requires exactly one source")
        if set(self.authorities) != set(self.available_facts):
            raise ValueError("every available evidence fact requires exactly one authority")
        return self


def evidence_profile_from_capabilities(capabilities: ProbeCapabilities) -> EvidenceProfile:
    supported_facts: set[EvidenceFact] = {"response_observed"}
    sources: dict[EvidenceFact, str] = {"response_observed": capabilities.invoker.invoker_id}
    authorities: dict[EvidenceFact, EvidenceAuthority] = {
        "response_observed": "invoker_self_reported"
    }
    observation_source = capabilities.observation_source
    if observation_source is not None and observation_source.supports_trajectory:
        supported_facts.add("trajectory_observed")
        sources["trajectory_observed"] = observation_source.source_id
        authorities["trajectory_observed"] = observation_source.authority
    state_environment = capabilities.state_environment
    if state_environment is not None and state_environment.supports_snapshot:
        supported_facts.add("committed_state_verified")
        sources["committed_state_verified"] = state_environment.environment_id
        authorities["committed_state_verified"] = (
            "independent_observer"
            if state_environment.state_observation_authority == "independent_observer"
            else "environment_self_reported"
        )
    if state_environment is not None and state_environment.supports_deterministic_replay:
        supported_facts.add("deterministic_replay_verified")
        sources["deterministic_replay_verified"] = state_environment.environment_id
        authorities["deterministic_replay_verified"] = (
            "independent_observer"
            if state_environment.state_observation_authority == "independent_observer"
            else "environment_self_reported"
        )
    return EvidenceProfile(
        available_facts=frozenset(supported_facts),
        sources=sources,
        authorities=authorities,
    )


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
    authority: Literal["environment_self_reported"] = "environment_self_reported"
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
    schema_version: Literal["1.1.0"] = "1.1.0"
    id: str = Field(min_length=1, max_length=500)
    turns: tuple[ConversationTurn, ...] = Field(min_length=1)
    max_environment_api_calls: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    required_state_observation_authority: StateObservationAuthority | None = None
    required_state_observer_id: str | None = Field(default=None, min_length=1, max_length=500)
    timeout_after_commit_event: TimeoutAfterCommitEventRequest | None = None
    evaluators: tuple[EvaluatorSpec, ...] = ()

    @model_validator(mode="after")
    def validate_identifiers(self) -> Self:
        if not math.isfinite(self.timeout_seconds):
            raise ValueError("evaluation case timeout must be finite")
        turn_ids = tuple(turn.id for turn in self.turns)
        if len(turn_ids) != len(set(turn_ids)):
            raise ValueError("evaluation case turn identifiers must be unique")
        evaluator_ids = tuple(evaluator.id for evaluator in self.evaluators)
        if len(evaluator_ids) != len(set(evaluator_ids)):
            raise ValueError("evaluation case evaluator identifiers must be unique")
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


class EnvironmentCapabilities(_StrictModel):
    isolation: Literal["customer_managed"] = "customer_managed"
    request_isolation: Literal["not_attested", "per_request_attested"] = "not_attested"
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


class EnvironmentTurnEvidence(_StrictModel):
    turn_id: str = Field(min_length=1)
    response: JsonValue
    response_source_id: str | None = Field(default=None, min_length=1, max_length=500)
    correlation_id: str | None = Field(default=None, min_length=1, max_length=500)
    state_snapshot: JsonValue | None = None
    state_observation_authority: StateObservationAuthority | None = None
    state_observer_id: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_state_evidence(self) -> Self:
        if (self.response_source_id is None) != (self.correlation_id is None):
            raise ValueError(
                "response source and correlation identifiers must be provided together"
            )
        if (self.state_snapshot is None) != (self.state_observation_authority is None):
            raise ValueError("state snapshot and its authority must be provided together")
        if self.state_observation_authority == "independent_observer":
            if self.state_observer_id is None:
                raise ValueError("independent state evidence requires an observer identifier")
        elif self.state_observer_id is not None:
            raise ValueError("self-reported state evidence cannot name an independent observer")
        return self


class EnvironmentStateEvidence(_StrictModel):
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


class EnvironmentLifecycleEvidence(_StrictModel):
    terminal_status: Literal["succeeded", "failed", "timed_out", "cancelled"]
    completed_phases: tuple[str, ...] = ()
    failed_phase: str | None = Field(default=None, min_length=1, max_length=200)
    failure_code: EnvironmentLifecycleFailureCode | None = None
    failure_reason: str | None = Field(default=None, min_length=1, max_length=500)
    delivery: Literal["certain", "uncertain"]
    cleanup: Literal["succeeded", "failed", "not_attempted"]
    cleanup_failure_code: EnvironmentLifecycleFailureCode | None = None
    cleanup_failure_reason: str | None = Field(default=None, min_length=1, max_length=500)
    environment_state_uncertain: bool
    initial_reset: EnvironmentResetEvidence | None = None
    cleanup_reset: EnvironmentResetEvidence | None = None

    @model_validator(mode="after")
    def validate_terminal_status(self) -> Self:
        if (self.cleanup == "not_attempted") != (self.cleanup_reset is None):
            raise ValueError("cleanup reset receipt must match whether cleanup was attempted")
        if (
            self.cleanup == "succeeded"
            and self.cleanup_reset is not None
            and (
                self.cleanup_reset.reset_session_requested
                != self.cleanup_reset.reset_session_acknowledged
                or self.cleanup_reset.reset_env_requested
                != self.cleanup_reset.reset_env_acknowledged
            )
        ):
            raise ValueError("successful cleanup requires every requested reset")
        if self.terminal_status == "succeeded" and any(
            receipt.reset_session_requested != receipt.reset_session_acknowledged
            or receipt.reset_env_requested != receipt.reset_env_acknowledged
            for receipt in (self.initial_reset, self.cleanup_reset)
            if receipt is not None
        ):
            raise ValueError("successful lifecycle evidence requires every requested reset")
        if self.terminal_status == "succeeded" and (
            self.failed_phase is not None
            or self.failure_code is not None
            or self.failure_reason is not None
        ):
            raise ValueError("successful lifecycle evidence cannot name a failure")
        if self.terminal_status == "succeeded" and (
            self.delivery != "certain" or self.environment_state_uncertain
        ):
            raise ValueError("successful lifecycle evidence requires certain delivery")
        if self.terminal_status != "succeeded" and self.failed_phase is None:
            raise ValueError("unsuccessful lifecycle evidence requires a failed phase")
        if self.terminal_status != "succeeded" and (
            self.failure_code is None or self.failure_reason is None
        ):
            raise ValueError("unsuccessful lifecycle evidence requires a failure code and reason")
        if self.delivery == "uncertain" and not self.environment_state_uncertain:
            raise ValueError("uncertain delivery must mark environment state uncertain")
        if self.cleanup == "failed" and not self.environment_state_uncertain:
            raise ValueError("failed cleanup must mark environment state uncertain")
        if (self.cleanup == "failed") != (self.cleanup_failure_reason is not None):
            raise ValueError("cleanup failure detail must match cleanup status")
        if (self.cleanup_failure_code is None) != (self.cleanup_failure_reason is None):
            raise ValueError("cleanup failure code and reason must be provided together")
        return self


class EnvironmentResetEvidence(_StrictModel):
    reset_session_requested: bool
    reset_session_acknowledged: bool
    reset_env_requested: bool
    reset_env_acknowledged: bool

    @model_validator(mode="after")
    def validate_acknowledgements(self) -> Self:
        if self.reset_session_acknowledged and not self.reset_session_requested:
            raise ValueError("session reset cannot be acknowledged when it was not requested")
        if self.reset_env_acknowledged and not self.reset_env_requested:
            raise ValueError("environment reset cannot be acknowledged when it was not requested")
        return self


class ExecutionEvidence(_StrictModel):
    schema_version: Literal["1.4.0"] = "1.4.0"
    evidence_scope: Literal["response_only", "response_and_state"] = "response_and_state"
    case_id: str = Field(min_length=1, max_length=500)
    environment_id: str = Field(min_length=1, max_length=500)
    environment_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    initial_state: EnvironmentStateEvidence | None = None
    turns: tuple[EnvironmentTurnEvidence, ...] = ()
    final_response: JsonValue | None = None
    final_state: EnvironmentStateEvidence | None = None
    timeout_after_commit_event: TimeoutAfterCommitEventEvidence | None = None
    observations: tuple[ProbeObservation, ...] = ()
    execution_events: tuple[ProbeExecutionEvent, ...] = ()
    probe_identity: ProbeExecutionIdentity | None = None
    lifecycle: EnvironmentLifecycleEvidence

    @model_validator(mode="after")
    def validate_successful_evidence(self) -> Self:
        if self.probe_identity is not None:
            if self.probe_identity.case_id != self.case_id:
                raise ValueError("probe identity must match the evidence case")
            evidence_turn_ids = tuple(turn.turn_id for turn in self.turns)
            if self.probe_identity.turn_ids[: len(evidence_turn_ids)] != evidence_turn_ids:
                raise ValueError("probe identity turns must contain the evidence turns in order")
        if self.evidence_scope == "response_only":
            if (
                self.initial_state is not None
                or self.final_state is not None
                or any(turn.state_snapshot is not None for turn in self.turns)
            ):
                raise ValueError("response-only evidence cannot contain state observations")
            if self.lifecycle.initial_reset is not None or self.lifecycle.cleanup_reset is not None:
                raise ValueError("response-only evidence cannot contain reset receipts")
            if self.lifecycle.cleanup != "not_attempted":
                raise ValueError("response-only evidence must record cleanup as not attempted")
            if any(phase != "execute_turn" for phase in self.lifecycle.completed_phases):
                raise ValueError("response-only evidence cannot contain stateful phases")
            if self.lifecycle.failed_phase not in {None, "execute_turn"}:
                raise ValueError("response-only evidence cannot fail during a stateful phase")
            if self.timeout_after_commit_event is not None:
                raise ValueError("response-only evidence cannot contain state-dependent events")
        else:
            if self.lifecycle.initial_reset is None:
                raise ValueError("response-and-state evidence requires an initial reset receipt")
            if (
                self.lifecycle.terminal_status == "succeeded"
                and self.lifecycle.cleanup != "succeeded"
            ):
                raise ValueError("successful response-and-state evidence requires cleanup")
        if self.lifecycle.terminal_status == "succeeded" and not self.turns:
            raise ValueError("successful execution evidence requires turn evidence")
        if self.lifecycle.terminal_status == "succeeded":
            if self.final_response is None:
                raise ValueError("successful execution evidence requires a final response")
            final_turn = self.turns[-1]
            if self.final_response != final_turn.response:
                raise ValueError("final response must match the last turn")
            if self.evidence_scope == "response_and_state":
                if self.initial_state is None or self.final_state is None:
                    raise ValueError("response-and-state evidence requires explicit pre/post state")
                if any(turn.state_snapshot is None for turn in self.turns):
                    raise ValueError("response-and-state evidence requires state for every turn")
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
