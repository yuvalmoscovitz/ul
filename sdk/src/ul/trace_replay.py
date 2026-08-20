from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import secrets
import stat
import sys
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import ConfigDict, Field, JsonValue, ValidationError, field_validator, model_validator
from ul_core.contracts import EnvironmentExecutor
from ul_core.dataset import ObservedAgentOutput
from ul_core.evaluation import ExecutionEvidence
from ul_core.models import ConversationRole, ConversationTurn, ULModel

from ul.dataset_evaluation import DatasetTargetLifecycleFailure
from ul.environment import (
    evaluation_case_from_inputs,
    execution_evidence_requires_quarantine,
    observed_outputs_from_evidence,
    validate_execution_evidence,
)
from ul.http_environment import (
    JsonHttpEnvironmentConfig,
    json_http_environment_calls_per_conversation,
)
from ul.otlp_ingest import OtlpInteractionRecord

_MAXIMUM_BUNDLE_BYTES = 50_000_000
_MAXIMUM_JSON_DEPTH = 100
_MAXIMUM_REPLAY_CASES = 1_000
_MAXIMUM_REPLAY_TRACES = 100
_MAXIMUM_MESSAGES_PER_TRACE = 512
_MAXIMUM_SPANS_PER_TRACE = 256
_CASE_ID_PATTERN = r"^ultr_v1_[0-9a-f]{64}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _StrictModel(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TraceReplayEnvelope(_StrictModel):
    trace_id: str = Field(min_length=1, max_length=128)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    scenario: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        if self.scenario.get("kind") != "ul.trace_scenario":
            raise ValueError("trace replay envelopes require a UL trace scenario")
        if self.scenario.get("trace_id") != self.trace_id:
            raise ValueError("trace replay envelope trace ID does not match its scenario")
        if self.sha256 != _canonical_sha256(cast(JsonValue, self.scenario)):
            raise ValueError("trace replay envelope digest does not match its scenario")
        return self


class TraceReplayCase(_StrictModel):
    case_id: str = Field(pattern=_CASE_ID_PATTERN)
    source_trace_id: str = Field(min_length=1, max_length=128)
    source_envelope_sha256: str = Field(pattern=_SHA256_PATTERN)
    selected_user_message_index: int = Field(ge=0)
    terminal_response_message_index: int = Field(ge=0)
    conversation_prefix: tuple[ConversationTurn, ...] = Field(min_length=2)
    replay_user_turns: tuple[ConversationTurn, ...] = Field(min_length=1)
    recorded_terminal_response: str = Field(min_length=1, max_length=100_000)
    recorded_state_snapshot_available: bool
    recorded_state_snapshot: JsonValue = None
    source_span_ids: tuple[str, ...] = ()

    @field_validator("conversation_prefix", "replay_user_turns", mode="before")
    @classmethod
    def accept_json_turn_arrays(cls, turns: object) -> object:
        return tuple(cast(list[object], turns)) if isinstance(turns, list) else turns

    @field_validator("source_span_ids", mode="before")
    @classmethod
    def accept_json_span_array(cls, span_ids: object) -> object:
        return tuple(cast(list[object], span_ids)) if isinstance(span_ids, list) else span_ids

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        if self.selected_user_message_index >= self.terminal_response_message_index:
            raise ValueError("trace replay response must follow its selected user message")
        if self.conversation_prefix[-1].role != ConversationRole.ASSISTANT:
            raise ValueError("trace replay conversation must end at an assistant response")
        if any(turn.role != ConversationRole.USER for turn in self.replay_user_turns):
            raise ValueError("trace replay inputs must contain only user turns")
        expected_replay_turns = tuple(
            turn for turn in self.conversation_prefix if turn.role == ConversationRole.USER
        )
        if self.replay_user_turns != expected_replay_turns:
            raise ValueError("trace replay inputs must match the conversation prefix")
        if len({turn.id for turn in self.conversation_prefix}) != len(self.conversation_prefix):
            raise ValueError("trace replay conversation turn IDs must be unique")
        if not self.recorded_state_snapshot_available and self.recorded_state_snapshot is not None:
            raise ValueError("unavailable recorded state must not contain a value")
        content = self.model_dump(mode="json", exclude={"case_id"})
        if self.case_id != _trace_replay_case_id(cast(dict[str, JsonValue], content)):
            raise ValueError("trace replay case ID must match its canonical content")
        return self


class TraceReplayBundle(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    envelopes: tuple[TraceReplayEnvelope, ...] = Field(min_length=1)
    cases: tuple[TraceReplayCase, ...] = Field(min_length=1, max_length=_MAXIMUM_REPLAY_CASES)

    @field_validator("envelopes", "cases", mode="before")
    @classmethod
    def accept_json_arrays(cls, values: object) -> object:
        return tuple(cast(list[object], values)) if isinstance(values, list) else values

    @model_validator(mode="after")
    def validate_bundle(self) -> Self:
        envelope_by_trace_id = {envelope.trace_id: envelope for envelope in self.envelopes}
        if len(envelope_by_trace_id) != len(self.envelopes):
            raise ValueError("trace replay bundle contains duplicate trace IDs")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("trace replay bundle contains duplicate case IDs")
        expected_cases: list[TraceReplayCase] = []
        for case in self.cases:
            envelope = envelope_by_trace_id.get(case.source_trace_id)
            if envelope is None or envelope.sha256 != case.source_envelope_sha256:
                raise ValueError("trace replay case references an unknown source envelope")
        for envelope in self.envelopes:
            expected_cases.extend(_materialize_trace_cases(envelope))
        if self.cases != tuple(expected_cases):
            raise ValueError("trace replay cases do not match their source envelopes")
        if len(self.model_dump_json().encode("utf-8")) > _MAXIMUM_BUNDLE_BYTES:
            raise ValueError("trace replay bundle exceeds the size limit")
        return self


class TraceReplayPlan(_StrictModel):
    case_id: str
    source_trace_id: str
    replay_turn_count: int = Field(ge=1)
    repetitions: int = Field(ge=1)
    target_calls_per_repetition: int = Field(ge=1)
    required_target_calls: int = Field(ge=1)


class TraceReplayTrial(_StrictModel):
    repetition: int = Field(ge=1)
    execution_evidence: ExecutionEvidence | None = None
    outputs: tuple[ObservedAgentOutput, ...] = ()
    response_matches_recorded: bool | None = None
    state_matches_recorded: bool | None = None
    inconclusive_reason: str | None = None
    lifecycle_failure: DatasetTargetLifecycleFailure | None = None

    @field_validator("outputs", mode="before")
    @classmethod
    def accept_json_output_array(cls, outputs: object) -> object:
        return tuple(cast(list[object], outputs)) if isinstance(outputs, list) else outputs

    @model_validator(mode="after")
    def validate_trial(self) -> Self:
        if self.inconclusive_reason is None:
            if (
                self.execution_evidence is None
                or self.execution_evidence.lifecycle.terminal_status != "succeeded"
                or not self.outputs
                or self.response_matches_recorded is None
            ):
                raise ValueError("conclusive trace replay trials require comparison evidence")
        elif self.response_matches_recorded is not None or self.state_matches_recorded is not None:
            raise ValueError("inconclusive trace replay trials must not claim a comparison")
        return self


class TraceReplayResult(_StrictModel):
    schema_version: Literal["1.1.0"] = "1.1.0"
    case: TraceReplayCase
    requested_repetitions: int = Field(ge=1)
    required_target_calls: int = Field(ge=1)
    status: Literal["reproduced", "drifted", "inconclusive"]
    response_match_count: int = Field(ge=0)
    state_match_count: int | None = Field(default=None, ge=0)
    trials: tuple[TraceReplayTrial, ...] = Field(min_length=1)

    @field_validator("trials", mode="before")
    @classmethod
    def accept_json_trial_array(cls, trials: object) -> object:
        return tuple(cast(list[object], trials)) if isinstance(trials, list) else trials

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if len(self.trials) != self.requested_repetitions:
            raise ValueError("trace replay result must preserve every repetition")
        if tuple(trial.repetition for trial in self.trials) != tuple(
            range(1, self.requested_repetitions + 1)
        ):
            raise ValueError("trace replay repetitions must remain ordered")
        expected_status: Literal["reproduced", "drifted", "inconclusive"]
        if any(trial.inconclusive_reason is not None for trial in self.trials):
            expected_status = "inconclusive"
        elif self.response_match_count == self.requested_repetitions and (
            self.state_match_count is None or self.state_match_count == self.requested_repetitions
        ):
            expected_status = "reproduced"
        else:
            expected_status = "drifted"
        if self.status != expected_status:
            raise ValueError("trace replay status must match its trial evidence")
        return self


def materialize_trace_replay_bundle(
    records: tuple[OtlpInteractionRecord, ...],
) -> TraceReplayBundle:
    if len(records) > _MAXIMUM_REPLAY_TRACES:
        raise ValueError("trace export contains too many traces for one replay bundle")
    envelopes: list[TraceReplayEnvelope] = []
    cases: list[TraceReplayCase] = []
    for record in records:
        if not isinstance(record.output, dict) or record.output.get("kind") != "ul.trace_scenario":
            continue
        scenario = cast(dict[str, JsonValue], record.output)
        envelope_sha256 = _canonical_sha256(cast(JsonValue, scenario))
        envelope = TraceReplayEnvelope(
            trace_id=record.interaction_id,
            sha256=envelope_sha256,
            scenario=scenario,
        )
        trace_cases = _materialize_trace_cases(envelope)
        if trace_cases:
            envelopes.append(envelope)
            cases.extend(trace_cases)
        if len(cases) > _MAXIMUM_REPLAY_CASES:
            raise ValueError("trace export contains too many replayable user turns")
    if not cases:
        raise ValueError("trace export contains no replayable user turns with assistant responses")
    return TraceReplayBundle(envelopes=tuple(envelopes), cases=tuple(cases))


def select_trace_replay_case(
    bundle: TraceReplayBundle, case_id: str | None = None
) -> TraceReplayCase:
    if case_id is None:
        if len(bundle.cases) != 1:
            raise ValueError("trace replay bundle contains multiple cases; provide --case-id")
        return bundle.cases[0]
    matching = tuple(case for case in bundle.cases if case.case_id == case_id)
    if len(matching) != 1:
        raise ValueError("trace replay case ID was not found in the bundle")
    return matching[0]


def plan_trace_replay(
    case: TraceReplayCase,
    target_config: JsonHttpEnvironmentConfig,
    *,
    repetitions: int = 3,
    max_target_calls: int = 100,
) -> TraceReplayPlan:
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if type(max_target_calls) is not int or max_target_calls < 1:
        raise ValueError("max_target_calls must be a positive integer")
    target_calls_per_repetition = json_http_environment_calls_per_conversation(
        target_config, len(case.replay_user_turns)
    )
    required_target_calls = repetitions * target_calls_per_repetition
    if required_target_calls > max_target_calls:
        raise ValueError("trace replay exceeds the authorized target call budget")
    return TraceReplayPlan(
        case_id=case.case_id,
        source_trace_id=case.source_trace_id,
        replay_turn_count=len(case.replay_user_turns),
        repetitions=repetitions,
        target_calls_per_repetition=target_calls_per_repetition,
        required_target_calls=required_target_calls,
    )


async def run_trace_replay(
    case: TraceReplayCase,
    environment: EnvironmentExecutor,
    *,
    repetitions: int = 3,
    max_target_calls: int = 100,
    allow_network_egress: bool = False,
) -> TraceReplayResult:
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if type(max_target_calls) is not int or max_target_calls < 1:
        raise ValueError("max_target_calls must be a positive integer")
    if not allow_network_egress:
        raise ValueError("trace replay environment API access requires explicit network opt-in")
    planned_case = evaluation_case_from_inputs(
        case_id=f"ul-case-{secrets.token_hex(16)}",
        raw_inputs=(turn.content for turn in case.replay_user_turns),
        max_environment_api_calls=max_target_calls,
        timeout_seconds=30,
        required_state_observation_authority=(
            environment.capabilities.state_observation_authority
            if case.recorded_state_snapshot_available
            else None
        ),
        required_state_observer_id=(
            environment.capabilities.state_observer_id
            if case.recorded_state_snapshot_available
            else None
        ),
    )
    calls_per_repetition = environment.api_calls_for_case(planned_case)
    if type(calls_per_repetition) is not int or calls_per_repetition < 1:
        raise ValueError("trace replay target returned an invalid physical call count")
    required_target_calls = repetitions * calls_per_repetition
    if required_target_calls > max_target_calls:
        raise ValueError("trace replay exceeds the authorized target call budget")

    raw_inputs = tuple(turn.content for turn in case.replay_user_turns)
    trials: list[TraceReplayTrial] = []
    environment_state_uncertain = False
    for repetition in range(1, repetitions + 1):
        if environment_state_uncertain:
            trials.append(
                TraceReplayTrial(
                    repetition=repetition,
                    inconclusive_reason=(
                        "environment not called because prior execution left state uncertain"
                    ),
                )
            )
            continue
        evidence: ExecutionEvidence | None = None
        try:
            evaluation_case = evaluation_case_from_inputs(
                case_id=f"ul-case-{secrets.token_hex(16)}",
                raw_inputs=raw_inputs,
                max_environment_api_calls=calls_per_repetition,
                timeout_seconds=30,
                required_state_observation_authority=(
                    environment.capabilities.state_observation_authority
                    if case.recorded_state_snapshot_available
                    else None
                ),
                required_state_observer_id=(
                    environment.capabilities.state_observer_id
                    if case.recorded_state_snapshot_available
                    else None
                ),
            )
            async with asyncio.timeout(evaluation_case.timeout_seconds):
                evidence = await environment.execute(evaluation_case)
            validate_execution_evidence(evaluation_case, environment, evidence)
            environment_state_uncertain = execution_evidence_requires_quarantine(evidence)
            if evidence.lifecycle.terminal_status != "succeeded":
                trials.append(
                    TraceReplayTrial(
                        repetition=repetition,
                        execution_evidence=evidence,
                        inconclusive_reason="environment lifecycle failed",
                        lifecycle_failure=DatasetTargetLifecycleFailure(
                            failed_phase=evidence.lifecycle.failed_phase or "unknown",
                            completed_phases=evidence.lifecycle.completed_phases,
                            cleanup_reset_failed=evidence.lifecycle.cleanup == "failed",
                            environment_state_may_remain=(
                                evidence.lifecycle.environment_state_uncertain
                            ),
                        ),
                    )
                )
                continue
            outputs = observed_outputs_from_evidence(evidence)
            if len(outputs) != len(raw_inputs):
                raise RuntimeError("environment returned an invalid number of turn observations")
            final_output = outputs[-1]
            committed_state_present = "committed_state_snapshot" in final_output.metadata
            if case.recorded_state_snapshot_available and not committed_state_present:
                raise RuntimeError("target omitted a committed state snapshot")
            state_match = (
                final_output.metadata["committed_state_snapshot"] == case.recorded_state_snapshot
                if case.recorded_state_snapshot_available
                else None
            )
            trials.append(
                TraceReplayTrial(
                    repetition=repetition,
                    execution_evidence=evidence,
                    outputs=outputs,
                    response_matches_recorded=(
                        final_output.raw_output == case.recorded_terminal_response
                    ),
                    state_matches_recorded=state_match,
                )
            )
        except TimeoutError:
            environment_state_uncertain = (
                environment.capabilities.cancellation_guarantee != "guaranteed"
            )
            trials.append(
                TraceReplayTrial(
                    repetition=repetition,
                    inconclusive_reason="environment execution timed out",
                )
            )
        except RuntimeError:
            trials.append(
                TraceReplayTrial(
                    repetition=repetition,
                    execution_evidence=evidence,
                    inconclusive_reason="environment execution failed",
                )
            )

    response_match_count = sum(trial.response_matches_recorded is True for trial in trials)
    state_match_count = (
        sum(trial.state_matches_recorded is True for trial in trials)
        if case.recorded_state_snapshot_available
        else None
    )
    status: Literal["reproduced", "drifted", "inconclusive"]
    if any(trial.inconclusive_reason is not None for trial in trials):
        status = "inconclusive"
    elif response_match_count == repetitions and (
        state_match_count is None or state_match_count == repetitions
    ):
        status = "reproduced"
    else:
        status = "drifted"
    return TraceReplayResult(
        case=case,
        requested_repetitions=repetitions,
        required_target_calls=required_target_calls,
        status=status,
        response_match_count=response_match_count,
        state_match_count=state_match_count,
        trials=tuple(trials),
    )


def load_trace_replay_bundle(path: str | Path) -> TraceReplayBundle:
    try:
        encoded = _read_bounded_regular_file(Path(path), _MAXIMUM_BUNDLE_BYTES)
        raw = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonstandard_json_constant,
            parse_float=_parse_finite_float,
        )
        _reject_deep_json(raw)
        return TraceReplayBundle.model_validate(raw)
    except OSError:
        raise RuntimeError("trace replay bundle could not be read") from None
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValidationError, ValueError):
        raise ValueError("trace replay bundle is invalid") from None


def _materialize_trace_cases(envelope: TraceReplayEnvelope) -> tuple[TraceReplayCase, ...]:
    raw_messages = envelope.scenario.get("messages")
    raw_spans = envelope.scenario.get("spans")
    if (
        not isinstance(raw_messages, list)
        or len(raw_messages) > _MAXIMUM_MESSAGES_PER_TRACE
        or not isinstance(raw_spans, list)
        or len(raw_spans) > _MAXIMUM_SPANS_PER_TRACE
    ):
        return ()
    messages = [
        cast(dict[str, JsonValue], message)
        for message in cast(list[JsonValue], raw_messages)
        if isinstance(message, dict)
    ]
    cases: list[TraceReplayCase] = []
    user_indices = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user" and _message_content(message) is not None
    ]
    for user_position, user_index in enumerate(user_indices):
        next_user_index = (
            user_indices[user_position + 1]
            if user_position + 1 < len(user_indices)
            else len(messages)
        )
        response_indices = [
            index
            for index in range(user_index + 1, next_user_index)
            if messages[index].get("role") == "assistant"
            and _message_content(messages[index]) is not None
        ]
        if not response_indices:
            continue
        response_index = response_indices[-1]
        conversation_prefix = _conversation_prefix(envelope.trace_id, messages, response_index)
        replay_user_turns = tuple(
            turn for turn in conversation_prefix if turn.role == ConversationRole.USER
        )
        response_content = _message_content(messages[response_index])
        if response_content is None:
            continue
        source_span_ids = _source_span_ids(
            envelope.scenario,
            messages[user_index],
            messages[response_index],
        )
        state_available, state_snapshot = _recorded_state_snapshot(
            envelope.scenario, source_span_ids
        )
        content = cast(
            dict[str, JsonValue],
            {
                "source_trace_id": envelope.trace_id,
                "source_envelope_sha256": envelope.sha256,
                "selected_user_message_index": user_index,
                "terminal_response_message_index": response_index,
                "conversation_prefix": [
                    turn.model_dump(mode="json") for turn in conversation_prefix
                ],
                "replay_user_turns": [turn.model_dump(mode="json") for turn in replay_user_turns],
                "recorded_terminal_response": response_content,
                "recorded_state_snapshot_available": state_available,
                "recorded_state_snapshot": state_snapshot,
                "source_span_ids": list(source_span_ids),
            },
        )
        cases.append(
            TraceReplayCase(
                case_id=_trace_replay_case_id(content),
                source_trace_id=envelope.trace_id,
                source_envelope_sha256=envelope.sha256,
                selected_user_message_index=user_index,
                terminal_response_message_index=response_index,
                conversation_prefix=conversation_prefix,
                replay_user_turns=replay_user_turns,
                recorded_terminal_response=response_content,
                recorded_state_snapshot_available=state_available,
                recorded_state_snapshot=state_snapshot,
                source_span_ids=source_span_ids,
            )
        )
    return tuple(cases)


def _conversation_prefix(
    trace_id: str,
    messages: list[dict[str, JsonValue]],
    response_index: int,
) -> tuple[ConversationTurn, ...]:
    turns: list[ConversationTurn] = []
    for index, message in enumerate(messages[: response_index + 1]):
        raw_role = message.get("role")
        content = _message_content(message)
        if raw_role not in {role.value for role in ConversationRole} or content is None:
            continue
        metadata: dict[str, JsonValue] = {"source_message_index": index}
        direction = message.get("direction")
        if isinstance(direction, str):
            metadata["direction"] = direction
        turns.append(
            ConversationTurn(
                id=f"{trace_id}:message:{index + 1}",
                role=ConversationRole(raw_role),
                content=content,
                metadata=metadata,
            )
        )
    return tuple(turns)


def _message_content(message: dict[str, JsonValue]) -> str | None:
    content = message.get("content")
    return content if isinstance(content, str) and content.strip() else None


def _source_span_ids(
    scenario: dict[str, JsonValue],
    user_message: dict[str, JsonValue],
    response_message: dict[str, JsonValue],
) -> tuple[str, ...]:
    raw_spans = scenario.get("spans")
    raw_messages = scenario.get("messages")
    if not isinstance(raw_spans, list) or not isinstance(raw_messages, list):
        return ()
    requested_identities = {
        _message_identity(user_message),
        _message_identity(response_message),
    }
    top_level_identities = [
        _message_identity(cast(dict[str, JsonValue], message))
        for message in cast(list[JsonValue], raw_messages)
        if isinstance(message, dict)
    ]
    remaining_identities = {
        identity for identity in requested_identities if top_level_identities.count(identity) == 1
    }
    span_ids: list[str] = []
    for raw_span in cast(list[JsonValue], raw_spans):
        if not isinstance(raw_span, dict):
            continue
        span = cast(dict[str, JsonValue], raw_span)
        raw_messages = span.get("messages")
        if not isinstance(raw_messages, list):
            continue
        identities = {
            _message_identity(cast(dict[str, JsonValue], message))
            for message in cast(list[JsonValue], raw_messages)
            if isinstance(message, dict)
        }
        span_id = span.get("span_id")
        matched_identities = remaining_identities & identities
        if matched_identities and isinstance(span_id, str) and span_id:
            span_ids.append(span_id)
            remaining_identities -= matched_identities
        if not remaining_identities:
            break
    return tuple(dict.fromkeys(span_ids))


def _recorded_state_snapshot(
    scenario: dict[str, JsonValue], source_span_ids: tuple[str, ...]
) -> tuple[bool, JsonValue]:
    raw_spans = scenario.get("spans")
    if not isinstance(raw_spans, list):
        return False, None
    found = False
    snapshot: JsonValue = None
    for raw_span in cast(list[JsonValue], raw_spans):
        if not isinstance(raw_span, dict):
            continue
        span = cast(dict[str, JsonValue], raw_span)
        if span.get("span_id") not in source_span_ids or "state_snapshot" not in span:
            continue
        found = True
        snapshot = span["state_snapshot"]
    return found, snapshot


def _message_identity(message: dict[str, JsonValue]) -> str:
    semantic = {key: value for key, value in message.items() if key != "direction"}
    return json.dumps(semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _trace_replay_case_id(content: dict[str, JsonValue]) -> str:
    return f"ultr_v1_{_canonical_sha256(cast(JsonValue, content))}"


def _canonical_sha256(value: JsonValue) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_bounded_regular_file(path: Path, maximum_bytes: int) -> bytes:
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    requires_identity_check = no_follow_flag == 0
    if requires_identity_check and stat.S_ISLNK(os.lstat(path).st_mode):
        raise OSError("path is a symbolic link")
    binary_flag = os.O_BINARY if sys.platform == "win32" else 0
    descriptor = os.open(
        path,
        os.O_RDONLY | no_follow_flag | getattr(os, "O_NONBLOCK", 0) | binary_flag,
    )
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise OSError("path is not a regular file")
        if requires_identity_check:
            path_status = os.lstat(path)
            if stat.S_ISLNK(path_status.st_mode) or not os.path.samestat(status, path_status):
                raise OSError("path changed while it was opened")
        encoded = bytearray()
        while len(encoded) <= maximum_bytes:
            chunk = os.read(descriptor, min(65_536, maximum_bytes + 1 - len(encoded)))
            if not chunk:
                break
            encoded.extend(chunk)
        if len(encoded) > maximum_bytes:
            raise ValueError("trace replay bundle exceeds the size limit")
        return bytes(encoded)
    finally:
        os.close(descriptor)


def _reject_duplicate_object_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _reject_deep_json(value: object, depth: int = 0) -> None:
    if depth > _MAXIMUM_JSON_DEPTH:
        raise ValueError("JSON nesting exceeds the limit")
    if isinstance(value, dict):
        for nested in cast(dict[object, object], value).values():
            _reject_deep_json(nested, depth + 1)
    elif isinstance(value, list):
        for nested in cast(list[object], value):
            _reject_deep_json(nested, depth + 1)
