from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import secrets
import stat
import sys
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import ConfigDict, Field, JsonValue, ValidationError, field_validator, model_validator
from ul_core.contracts import EnvironmentExecutor
from ul_core.dataset import ObservedAgentOutput
from ul_core.evaluation import EvaluationCase, ExecutionEvidence, StateObservationAuthority
from ul_core.models import ULModel

from ul.dataset_evaluation import DatasetTargetLifecycleFailure
from ul.dataset_invariants import (
    DatasetInvariantRule,
    DatasetInvariantRuleEvaluation,
    DatasetInvariantRuleResult,
    DatasetInvariantTransitionRuleEvaluation,
    JsonValuesEqualInvariant,
    ObservationAuthority,
    evaluate_dataset_invariant_rules,
)
from ul.environment import (
    evaluation_case_from_inputs,
    execution_evidence_requires_quarantine,
    observed_outputs_from_evidence,
    validate_execution_evidence,
)
from ul.http_environment import JsonHttpEnvironmentConfig, json_http_environment_config_sha256

_MAXIMUM_CASE_BYTES = 1_000_000
_MAXIMUM_JSON_DEPTH = 100
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_FINDING_ID_PATTERN = r"^ulf_v1_[0-9a-f]{64}$"
_REVIEW_ID_PATTERN = r"^ulr_[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
_CASE_ID_PATTERN = r"^ulrc_v1_[0-9a-f]{64}$"
_TRANSITION_RULE_TYPES = {
    "no_new_effect",
    "exactly_one_new_effect",
    "unchanged_between_checkpoints",
}
RegressionStatus = Literal["passed", "failed", "inconclusive"]
RegressionExecutionStatus = Literal[
    "observed",
    "target_execution_timed_out",
    "target_execution_failed",
    "target_state_uncertain",
]


class _StrictModel(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DatasetRegressionLineage(_StrictModel):
    finding_id: str = Field(pattern=_FINDING_ID_PATTERN)
    evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    review_id: str = Field(pattern=_REVIEW_ID_PATTERN)


class DatasetRegressionVariation(_StrictModel):
    interaction_id: str = Field(min_length=1, max_length=500)
    operator_id: str = Field(min_length=1, max_length=200)
    operator_version: str = Field(min_length=1, max_length=100)
    original_input: str = Field(min_length=1, max_length=1_000_000)
    variation_input: str = Field(min_length=1, max_length=1_000_000)


class DatasetRegressionTargetSnapshot(_StrictModel):
    provenance: Literal["declared_at_case_creation"]
    config: JsonHttpEnvironmentConfig
    config_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_config_digest(self) -> Self:
        if self.config_sha256 != dataset_regression_target_config_sha256(self.config):
            raise ValueError("target config digest must match its canonical snapshot")
        return self


class DatasetRegressionInvariantSuite(_StrictModel):
    source_suite_sha256: str = Field(pattern=_SHA256_PATTERN)
    observation_source: Literal["target_output"]
    observation_authority: ObservationAuthority
    state_observation_authority: StateObservationAuthority | None = None
    state_observer_id: str | None = Field(default=None, min_length=1, max_length=500)
    rules: tuple[DatasetInvariantRule, ...] = Field(min_length=1, max_length=100)

    @field_validator("rules", mode="before")
    @classmethod
    def accept_json_rule_array(cls, rules: object) -> object:
        return tuple(cast(list[object], rules)) if isinstance(rules, list) else rules

    @model_validator(mode="after")
    def validate_rule_ids(self) -> Self:
        rule_ids = tuple(rule.id for rule in self.rules)
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("regression invariant rule identifiers must be unique")
        requires_state = self.observation_authority == "committed_state_snapshot"
        if not requires_state and any(rule.type in _TRANSITION_RULE_TYPES for rule in self.rules):
            raise ValueError(
                "state-transition regression rules require committed-state observation"
            )
        if requires_state != (self.state_observation_authority is not None):
            raise ValueError("committed-state regressions must pin the state authority")
        if self.state_observation_authority == "independent_observer":
            if self.state_observer_id is None:
                raise ValueError("independent regressions must pin the observer identifier")
        elif self.state_observer_id is not None:
            raise ValueError("self-reported regressions cannot name an observer")
        return self


class DatasetRegressionCase(_StrictModel):
    schema_version: Literal["1.0.0", "1.1.0", "1.2.0", "1.3.0"]
    case_id: str = Field(pattern=_CASE_ID_PATTERN)
    lineage: DatasetRegressionLineage
    variation: DatasetRegressionVariation
    target: DatasetRegressionTargetSnapshot
    invariant_suite: DatasetRegressionInvariantSuite
    discovery_repetitions: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_case_id(self) -> Self:
        if self.schema_version == "1.0.0" and any(
            not isinstance(rule, JsonValuesEqualInvariant) for rule in self.invariant_suite.rules
        ):
            raise ValueError("regression schema 1.0.0 supports only json_values_equal rules")
        if self.schema_version not in {"1.2.0", "1.3.0"} and any(
            rule.type in _TRANSITION_RULE_TYPES for rule in self.invariant_suite.rules
        ):
            raise ValueError("state-transition regression rules require schema 1.2.0")
        if self.case_id != _case_id(self.model_dump(mode="json", exclude={"case_id"})):
            raise ValueError("regression case ID must match its canonical content")
        if len(self.model_dump_json().encode("utf-8")) > _MAXIMUM_CASE_BYTES:
            raise ValueError("regression case exceeds the size limit")
        return self


class DatasetRegressionExecution(_StrictModel):
    repetition: int = Field(ge=1)
    status: RegressionExecutionStatus
    execution_evidence: ExecutionEvidence | None = None
    target_output: ObservedAgentOutput | None = None
    lifecycle_failure: DatasetTargetLifecycleFailure | None = None

    @model_validator(mode="after")
    def validate_execution(self) -> Self:
        if (self.status == "observed") != (self.target_output is not None):
            raise ValueError("only observed regression executions contain target output")
        if self.status == "observed" and self.execution_evidence is None:
            raise ValueError("observed regression executions require execution evidence")
        if (
            self.execution_evidence is not None
            and self.execution_evidence.lifecycle.terminal_status == "succeeded"
        ) != (self.status == "observed"):
            raise ValueError("regression execution status must match execution evidence")
        return self


class DatasetRegressionResult(_StrictModel):
    schema_version: Literal["1.0.0", "1.1.0", "1.2.0", "1.3.0"] = "1.3.0"
    case_id: str = Field(pattern=_CASE_ID_PATTERN)
    target_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_suite_sha256: str = Field(pattern=_SHA256_PATTERN)
    requested_repetitions: int = Field(ge=1)
    target_calls_per_execution: int = Field(default=1, ge=1)
    status: RegressionStatus
    executions: tuple[DatasetRegressionExecution, ...] = Field(min_length=1)
    rules: tuple[DatasetInvariantRuleResult, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.schema_version == "1.0.0" and any(
            not isinstance(rule, DatasetInvariantRuleEvaluation) for rule in self.rules
        ):
            raise ValueError("regression result schema 1.0.0 supports only json_values_equal rules")
        if self.schema_version not in {"1.2.0", "1.3.0"} and any(
            isinstance(rule, DatasetInvariantTransitionRuleEvaluation) for rule in self.rules
        ):
            raise ValueError("state-transition regression results require schema 1.2.0")
        if tuple(execution.repetition for execution in self.executions) != tuple(
            range(1, len(self.executions) + 1)
        ):
            raise ValueError("regression executions must preserve repetition order")
        if len(self.executions) != self.requested_repetitions:
            raise ValueError("regression executions must include every requested repetition")
        if any(
            execution.execution_evidence is not None
            and execution.execution_evidence.environment_config_sha256 != self.target_config_sha256
            for execution in self.executions
        ):
            raise ValueError("regression execution evidence must match the result config digest")
        rule_ids = tuple(rule.rule_id for rule in self.rules)
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("regression results must have unique invariant rule IDs")
        rule_statuses = {rule.status for rule in self.rules}
        if "violated" in rule_statuses:
            expected_status: RegressionStatus = "failed"
        elif "not_evaluable" in rule_statuses:
            expected_status = "inconclusive"
        else:
            expected_status = "passed"
        if self.status != expected_status:
            raise ValueError("regression status must match invariant results")
        return self


class DatasetRegressionRunCaseResult(_StrictModel):
    label: str = Field(min_length=1, max_length=500)
    result: DatasetRegressionResult

    @field_validator("label")
    @classmethod
    def validate_label(cls, label: str) -> str:
        return _validate_regression_run_label(label)


class DatasetRegressionRunResult(_StrictModel):
    schema_version: Literal["1.0.0", "1.1.0", "1.2.0", "1.3.0"] = "1.3.0"
    started_at: datetime
    completed_at: datetime
    target_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    requested_target_calls: int = Field(ge=1)
    status: RegressionStatus
    passed_case_count: int = Field(ge=0)
    failed_case_count: int = Field(ge=0)
    inconclusive_case_count: int = Field(ge=0)
    cases: tuple[DatasetRegressionRunCaseResult, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        if self.schema_version == "1.0.0" and any(
            case.result.schema_version != "1.0.0" for case in self.cases
        ):
            raise ValueError("regression run schema 1.0.0 only supports result schema 1.0.0")
        if self.schema_version == "1.1.0" and any(
            case.result.schema_version in {"1.2.0", "1.3.0"} for case in self.cases
        ):
            raise ValueError("regression run schema 1.1.0 does not support result schema 1.2.0")
        if self.schema_version == "1.2.0" and any(
            case.result.schema_version == "1.3.0" for case in self.cases
        ):
            raise ValueError("regression run schema 1.2.0 does not support result schema 1.3.0")
        for timestamp in (self.started_at, self.completed_at):
            if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(None):
                raise ValueError("regression run timestamps must use UTC")
        if self.completed_at < self.started_at:
            raise ValueError("regression run completion cannot precede its start")
        case_ids = tuple(case.result.case_id for case in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("regression run results must have unique case IDs")
        labels = tuple(case.label for case in self.cases)
        if len(labels) != len(set(labels)):
            raise ValueError("regression run results must have unique labels")
        if any(
            case.result.target_config_sha256 != self.target_config_sha256 for case in self.cases
        ):
            raise ValueError("regression run results must use one target config digest")
        if self.requested_target_calls != sum(
            case.result.requested_repetitions * case.result.target_calls_per_execution
            for case in self.cases
        ):
            raise ValueError(
                "requested environment API calls must match regression result repetitions"
            )
        expected_counts = {
            status: sum(case.result.status == status for case in self.cases)
            for status in ("passed", "failed", "inconclusive")
        }
        if (
            self.passed_case_count != expected_counts["passed"]
            or self.failed_case_count != expected_counts["failed"]
            or self.inconclusive_case_count != expected_counts["inconclusive"]
        ):
            raise ValueError("regression run counts must match case results")
        expected_status: RegressionStatus
        if self.failed_case_count:
            expected_status = "failed"
        elif self.inconclusive_case_count:
            expected_status = "inconclusive"
        else:
            expected_status = "passed"
        if self.status != expected_status:
            raise ValueError("regression run status must match case results")
        return self


def create_dataset_regression_case(
    *,
    finding_id: str,
    evidence_sha256: str,
    review_id: str,
    interaction_id: str,
    operator_id: str,
    operator_version: str,
    original_input: str,
    variation_input: str,
    target_config: JsonHttpEnvironmentConfig,
    source_suite_sha256: str,
    observation_authority: ObservationAuthority,
    state_observation_authority: StateObservationAuthority | None,
    state_observer_id: str | None = None,
    selected_rules: tuple[DatasetInvariantRule, ...],
    discovery_repetitions: int,
) -> DatasetRegressionCase:
    target = DatasetRegressionTargetSnapshot(
        provenance="declared_at_case_creation",
        config=target_config,
        config_sha256=dataset_regression_target_config_sha256(target_config),
    )
    lineage = DatasetRegressionLineage(
        finding_id=finding_id,
        evidence_sha256=evidence_sha256,
        review_id=review_id,
    )
    variation = DatasetRegressionVariation(
        interaction_id=interaction_id,
        operator_id=operator_id,
        operator_version=operator_version,
        original_input=original_input,
        variation_input=variation_input,
    )
    invariant_suite = DatasetRegressionInvariantSuite(
        source_suite_sha256=source_suite_sha256,
        observation_source="target_output",
        observation_authority=observation_authority,
        state_observation_authority=state_observation_authority,
        state_observer_id=state_observer_id,
        rules=selected_rules,
    )
    schema_version: Literal["1.0.0", "1.1.0", "1.2.0", "1.3.0"] = "1.3.0"
    serialized_content = cast(
        dict[str, JsonValue],
        {
            "schema_version": schema_version,
            "lineage": lineage.model_dump(mode="json"),
            "variation": variation.model_dump(mode="json"),
            "target": target.model_dump(mode="json"),
            "invariant_suite": invariant_suite.model_dump(mode="json"),
            "discovery_repetitions": discovery_repetitions,
        },
    )
    return DatasetRegressionCase(
        schema_version=schema_version,
        case_id=_case_id(serialized_content),
        lineage=lineage,
        variation=variation,
        target=target,
        invariant_suite=invariant_suite,
        discovery_repetitions=discovery_repetitions,
    )


def dataset_regression_target_config_sha256(config: JsonHttpEnvironmentConfig) -> str:
    return json_http_environment_config_sha256(config)


def load_dataset_regression_case(path: str | Path) -> DatasetRegressionCase:
    try:
        encoded_case = _read_bounded_regular_file(Path(path), maximum_bytes=_MAXIMUM_CASE_BYTES)
    except OSError:
        raise RuntimeError("dataset regression case could not be read") from None
    if len(encoded_case) > _MAXIMUM_CASE_BYTES:
        raise ValueError("dataset regression case exceeds the size limit")
    try:
        raw_case = json.loads(
            encoded_case.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonstandard_json_constant,
            parse_float=_parse_finite_float,
        )
        _reject_deep_json(raw_case)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise ValueError("dataset regression case contains invalid JSON") from None
    try:
        return DatasetRegressionCase.model_validate(raw_case)
    except RecursionError:
        raise ValueError("dataset regression case is invalid") from None
    except ValidationError as error:
        reasons = [
            f"{'.'.join(str(part) for part in issue['loc'])}: "
            f"{str(issue['msg']).removeprefix('Value error, ')}"
            for issue in error.errors(include_url=False, include_context=False, include_input=False)
        ]
        raise ValueError(f"dataset regression case is invalid: {'; '.join(reasons)}") from None


async def replay_dataset_regression(
    case: DatasetRegressionCase,
    environment: EnvironmentExecutor,
    *,
    target_timeout_seconds: float = 30,
    allow_network_egress: bool = False,
    max_target_calls: int = 100,
) -> DatasetRegressionResult:
    if not allow_network_egress:
        raise ValueError("regression environment API access requires explicit network opt-in")
    if isinstance(max_target_calls, bool) or max_target_calls < 1:
        raise ValueError("max_target_calls must be a positive integer")
    if not math.isfinite(target_timeout_seconds) or target_timeout_seconds <= 0:
        raise ValueError("target_timeout_seconds must be positive and finite")
    evaluation_case = _regression_evaluation_case(
        case,
        repetition=1,
        max_environment_api_calls=max_target_calls,
        timeout_seconds=target_timeout_seconds,
    )
    target_calls_per_execution = environment.api_calls_for_case(evaluation_case)
    required_target_calls = case.discovery_repetitions * target_calls_per_execution
    if required_target_calls > max_target_calls:
        raise ValueError("regression case exceeds the authorized target call budget")
    if not environment.capabilities.supports_state_observation and (
        case.invariant_suite.observation_authority == "committed_state_snapshot"
    ):
        raise ValueError("environment must support state observation for committed-state checks")
    if case.invariant_suite.state_observation_authority is not None and (
        case.invariant_suite.state_observation_authority
        != environment.capabilities.state_observation_authority
        or case.invariant_suite.state_observer_id != environment.capabilities.state_observer_id
    ):
        raise ValueError("environment state authority does not match the regression case")

    executions: list[DatasetRegressionExecution] = []
    target_state_uncertain = False
    for repetition in range(1, case.discovery_repetitions + 1):
        if target_state_uncertain:
            executions.append(
                DatasetRegressionExecution(
                    repetition=repetition,
                    status="target_state_uncertain",
                )
            )
            continue
        try:
            evaluation_case = _regression_evaluation_case(
                case,
                repetition=repetition,
                max_environment_api_calls=target_calls_per_execution,
                timeout_seconds=target_timeout_seconds,
            )
            async with asyncio.timeout(target_timeout_seconds):
                execution_evidence = await environment.execute(evaluation_case)
                validate_execution_evidence(evaluation_case, environment, execution_evidence)
        except TimeoutError:
            if environment.capabilities.cancellation_guarantee != "guaranteed":
                target_state_uncertain = True
            executions.append(
                DatasetRegressionExecution(
                    repetition=repetition,
                    status="target_execution_timed_out",
                )
            )
        except RuntimeError:
            executions.append(
                DatasetRegressionExecution(
                    repetition=repetition,
                    status="target_execution_failed",
                )
            )
        else:
            lifecycle = execution_evidence.lifecycle
            target_state_uncertain = execution_evidence_requires_quarantine(execution_evidence)
            if lifecycle.terminal_status != "succeeded":
                executions.append(
                    DatasetRegressionExecution(
                        repetition=repetition,
                        status=(
                            "target_state_uncertain"
                            if lifecycle.environment_state_uncertain
                            else (
                                "target_execution_timed_out"
                                if lifecycle.terminal_status == "timed_out"
                                else "target_execution_failed"
                            )
                        ),
                        execution_evidence=execution_evidence,
                        lifecycle_failure=DatasetTargetLifecycleFailure(
                            failed_phase=lifecycle.failed_phase or "unknown",
                            completed_phases=lifecycle.completed_phases,
                            cleanup_reset_failed=lifecycle.cleanup == "failed",
                            environment_state_may_remain=lifecycle.environment_state_uncertain,
                        ),
                    )
                )
                continue
            observed_outputs = observed_outputs_from_evidence(execution_evidence)
            if len(observed_outputs) != 1:
                raise RuntimeError("environment returned invalid regression execution evidence")
            executions.append(
                DatasetRegressionExecution(
                    repetition=repetition,
                    status="observed",
                    execution_evidence=execution_evidence,
                    target_output=observed_outputs[0],
                )
            )
    return _build_dataset_regression_result(case, target_calls_per_execution, tuple(executions))


def _build_dataset_regression_result(
    case: DatasetRegressionCase,
    target_calls_per_execution: int,
    executions: tuple[DatasetRegressionExecution, ...],
) -> DatasetRegressionResult:
    rules = evaluate_dataset_invariant_rules(
        case.invariant_suite.rules,
        tuple(execution.target_output for execution in executions),
        observation_authority=case.invariant_suite.observation_authority,
    )
    rule_statuses = {rule.status for rule in rules}
    status: RegressionStatus
    if "violated" in rule_statuses:
        status = "failed"
    elif "not_evaluable" in rule_statuses:
        status = "inconclusive"
    else:
        status = "passed"
    return DatasetRegressionResult(
        schema_version=case.schema_version,
        case_id=case.case_id,
        target_config_sha256=case.target.config_sha256,
        source_suite_sha256=case.invariant_suite.source_suite_sha256,
        requested_repetitions=case.discovery_repetitions,
        target_calls_per_execution=target_calls_per_execution,
        status=status,
        executions=executions,
        rules=rules,
    )


async def run_dataset_regressions(
    cases: tuple[DatasetRegressionCase, ...],
    environment: EnvironmentExecutor,
    *,
    case_labels: tuple[str, ...] | None = None,
    target_timeout_seconds: float = 30,
    allow_network_egress: bool = False,
    max_target_calls: int = 100,
) -> DatasetRegressionRunResult:
    if not allow_network_egress:
        raise ValueError("regression environment API access requires explicit network opt-in")
    if not cases:
        raise ValueError("regression run requires at least one case")
    if len(cases) > 100:
        raise ValueError("regression run supports at most 100 cases")
    case_ids = tuple(case.case_id for case in cases)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("regression run case IDs must be unique")
    labels = case_ids if case_labels is None else case_labels
    if len(labels) != len(cases):
        raise ValueError("regression run labels must match the cases")
    for label in labels:
        _validate_regression_run_label(label)
    if len(labels) != len(set(labels)):
        raise ValueError("regression run labels must be unique")
    target_config_sha256 = cases[0].target.config_sha256
    if any(case.target.config_sha256 != target_config_sha256 for case in cases):
        raise ValueError("regression run cases must use one target config digest")
    if isinstance(max_target_calls, bool) or max_target_calls < 1:
        raise ValueError("max_target_calls must be a positive integer")
    if not math.isfinite(target_timeout_seconds) or target_timeout_seconds <= 0:
        raise ValueError("target_timeout_seconds must be positive and finite")
    target_calls_per_case = tuple(
        environment.api_calls_for_case(
            _regression_evaluation_case(
                case,
                repetition=1,
                max_environment_api_calls=max_target_calls,
                timeout_seconds=target_timeout_seconds,
            )
        )
        for case in cases
    )
    requested_target_calls = sum(
        case.discovery_repetitions * target_calls_per_execution
        for case, target_calls_per_execution in zip(cases, target_calls_per_case, strict=True)
    )
    if requested_target_calls > max_target_calls:
        raise ValueError("regression run exceeds the authorized target call budget")

    started_at = datetime.now(UTC)
    result_list: list[DatasetRegressionResult] = []
    environment_state_uncertain = False
    for case, target_calls_per_execution in zip(cases, target_calls_per_case, strict=True):
        if environment_state_uncertain:
            result_list.append(
                _build_dataset_regression_result(
                    case,
                    target_calls_per_execution,
                    tuple(
                        DatasetRegressionExecution(
                            repetition=repetition,
                            status="target_state_uncertain",
                        )
                        for repetition in range(1, case.discovery_repetitions + 1)
                    ),
                )
            )
            continue
        result = await replay_dataset_regression(
            case,
            environment,
            target_timeout_seconds=target_timeout_seconds,
            allow_network_egress=allow_network_egress,
            max_target_calls=case.discovery_repetitions * target_calls_per_execution,
        )
        result_list.append(result)
        environment_state_uncertain = any(
            execution.status == "target_state_uncertain"
            or (
                execution.execution_evidence is not None
                and execution_evidence_requires_quarantine(execution.execution_evidence)
            )
            for execution in result.executions
        )
    results = tuple(result_list)
    completed_at = datetime.now(UTC)
    passed_case_count = sum(result.status == "passed" for result in results)
    failed_case_count = sum(result.status == "failed" for result in results)
    inconclusive_case_count = sum(result.status == "inconclusive" for result in results)
    status: RegressionStatus
    if failed_case_count:
        status = "failed"
    elif inconclusive_case_count:
        status = "inconclusive"
    else:
        status = "passed"
    return DatasetRegressionRunResult(
        schema_version=(
            "1.3.0"
            if any(result.schema_version == "1.3.0" for result in results)
            else "1.2.0"
            if any(result.schema_version == "1.2.0" for result in results)
            else "1.1.0"
            if any(result.schema_version == "1.1.0" for result in results)
            else "1.0.0"
        ),
        started_at=started_at,
        completed_at=completed_at,
        target_config_sha256=target_config_sha256,
        requested_target_calls=requested_target_calls,
        status=status,
        passed_case_count=passed_case_count,
        failed_case_count=failed_case_count,
        inconclusive_case_count=inconclusive_case_count,
        cases=tuple(
            DatasetRegressionRunCaseResult(label=label, result=result)
            for label, result in zip(labels, results, strict=True)
        ),
    )


def _regression_evaluation_case(
    case: DatasetRegressionCase,
    *,
    repetition: int,
    max_environment_api_calls: int,
    timeout_seconds: float,
) -> EvaluationCase:
    return evaluation_case_from_inputs(
        case_id=f"ul-case-{secrets.token_hex(16)}",
        raw_inputs=(case.variation.variation_input,),
        max_environment_api_calls=max_environment_api_calls,
        timeout_seconds=timeout_seconds,
        required_state_observation_authority=(case.invariant_suite.state_observation_authority),
        required_state_observer_id=case.invariant_suite.state_observer_id,
    )


def _validate_regression_run_label(label: object) -> str:
    if (
        not isinstance(label, str)
        or not label
        or len(label) > 500
        or any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in label)
    ):
        raise ValueError("regression run labels must contain 1 to 500 characters without controls")
    return label


def _case_id(content: dict[str, JsonValue]) -> str:
    return f"ulrc_v1_{_canonical_json_sha256(content)}"


def _read_bounded_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    requires_identity_check = no_follow_flag == 0
    if requires_identity_check and stat.S_ISLNK(os.lstat(path).st_mode):
        raise OSError("path is a symbolic link")
    binary_flag = os.O_BINARY if sys.platform == "win32" else 0
    nonblocking_flag = getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, os.O_RDONLY | no_follow_flag | nonblocking_flag | binary_flag)
    try:
        descriptor_status = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_status.st_mode):
            raise OSError("path is not a regular file")
        if requires_identity_check:
            path_status = os.lstat(path)
            if stat.S_ISLNK(path_status.st_mode) or not os.path.samestat(
                descriptor_status, path_status
            ):
                raise OSError("path changed while it was opened")
        chunks: list[bytes] = []
        remaining_bytes = maximum_bytes + 1
        while remaining_bytes:
            chunk = os.read(descriptor, min(remaining_bytes, 65_536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining_bytes -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _canonical_json_sha256(value: JsonValue) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_object_keys(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _reject_deep_json(value: object, *, depth: int = 0) -> None:
    if depth > _MAXIMUM_JSON_DEPTH:
        raise ValueError("JSON exceeds the nesting limit")
    if isinstance(value, dict):
        for item in cast(dict[str, object], value).values():
            _reject_deep_json(item, depth=depth + 1)
    elif isinstance(value, list):
        for item in cast(list[object], value):
            _reject_deep_json(item, depth=depth + 1)
