from __future__ import annotations

import asyncio
import json
import math
import re
import secrets
from collections import defaultdict
from collections.abc import Callable, Iterable
from decimal import Decimal, InvalidOperation
from itertools import islice
from typing import Literal, Self

from pydantic import ConfigDict, Field, JsonValue, model_validator
from ul_core.contracts import (
    EnvironmentExecutor,
    SemanticDeconstructor,
)
from ul_core.dataset import InteractionRecord, ObservedAgentOutput, ObservedOutcome, SemanticFrame
from ul_core.evaluation import EvaluationCase, ExecutionEvidence
from ul_core.models import ConversationRole, ConversationTurn, ULModel

from ul.augmentations.dataset import (
    DatasetAugmentationCandidate,
    DatasetAugmentationEngine,
    DatasetAugmentationResult,
    builtin_dataset_augmentation_operators,
)
from ul.environment import (
    environment_timeout_requires_quarantine,
    execution_evidence_requires_quarantine,
    observed_outputs_from_evidence,
    validate_execution_evidence,
)
from ul.outcome_projection import OutcomeProjectionError
from ul.probe_execution import OutcomeProjectionExecutionError

FindingCategory = Literal[
    "duplicate_effect",
    "unexpected_effect",
    "missing_effect",
    "changed_grounded_effect_argument",
    "changed_response",
]
_NUMBER_PATTERN = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?"
CaseVerdict = Literal[
    "augmentation_rejected",
    "inconclusive",
    "no_divergence",
    "divergence_needs_review",
]
BaselineVerdict = Literal["inconclusive", "no_divergence"]
TrialSetStability = Literal["stable", "unstable", "inconclusive"]
DatasetEvaluationMode = Literal["variance", "correctness", "preference"]
TrialArm = Literal["original", "probe"]
ComparisonSurface = Literal["action", "answer"]


class _StrictULModel(ULModel):
    model_config = ConfigDict(strict=True)


class DatasetTargetDeliveryUncertain(RuntimeError):
    """Execution was interrupted after target delivery may have begun."""


class DatasetEvaluationFinding(_StrictULModel):
    category: FindingCategory
    severity: Literal["unrated"] = "unrated"
    review_status: Literal["needs_review"] = "needs_review"
    message: str = Field(min_length=1)
    expected_effects: tuple[ObservedOutcome, ...] = ()
    observed_effects: tuple[ObservedOutcome, ...] = ()
    grounded_field_names: tuple[str, ...] = ()


class DatasetTargetLifecycleFailure(_StrictULModel):
    protocol_version: Literal[2] = 2
    failed_phase: str = Field(min_length=1)
    completed_phases: tuple[str, ...] = ()
    cleanup_reset_failed: bool
    environment_state_may_remain: bool


class DatasetEvaluationTrial(_StrictULModel):
    repetition: int = Field(ge=1)
    execution_evidence: ExecutionEvidence | None = None
    target_output: ObservedAgentOutput | None = None
    observed_frame: SemanticFrame | None = None
    inconclusive_reasons: tuple[str, ...] = ()
    lifecycle_failure: DatasetTargetLifecycleFailure | None = None

    @model_validator(mode="after")
    def validate_execution_state(self) -> Self:
        if self.observed_frame is not None and self.target_output is None:
            raise ValueError("an observed frame requires target output")
        if self.target_output is None and not self.inconclusive_reasons:
            raise ValueError("a trial requires target output or an inconclusive reason")
        if self.observed_frame is None and not self.inconclusive_reasons:
            raise ValueError("a trial without an observed frame requires an inconclusive reason")
        return self


class DatasetTrialUnit(_StrictULModel):
    interaction_id: str = Field(min_length=1)
    operator_id: str = Field(min_length=1)
    arm: TrialArm
    repetition: int = Field(ge=1)
    attempt: int = Field(default=1, ge=1)

    @property
    def id(self) -> str:
        return ":".join(
            (
                self.interaction_id,
                self.operator_id,
                self.arm,
                str(self.repetition),
                str(self.attempt),
            )
        )


class DatasetEvaluationOutcomeGroup(_StrictULModel):
    repetitions: tuple[int, ...] = Field(min_length=1)
    representative_effects: tuple[ObservedOutcome, ...]

    @model_validator(mode="after")
    def validate_repetitions(self) -> Self:
        if any(repetition < 1 for repetition in self.repetitions):
            raise ValueError("outcome group repetitions must be positive")
        if tuple(sorted(set(self.repetitions))) != self.repetitions:
            raise ValueError("outcome group repetitions must be unique and ordered")
        outcome_kinds = {effect.kind for effect in self.representative_effects}
        if outcome_kinds and (len(outcome_kinds) != 1 or not outcome_kinds <= {"action", "answer"}):
            raise ValueError("outcome groups require one explicit comparison surface")
        return self


class DatasetEvaluationTrialSet(_StrictULModel):
    requested_repetitions: int = Field(ge=1)
    stability: TrialSetStability
    trials: tuple[DatasetEvaluationTrial, ...]
    outcome_groups: tuple[DatasetEvaluationOutcomeGroup, ...] = ()

    @model_validator(mode="after")
    def validate_trials(self) -> Self:
        if len(self.trials) != self.requested_repetitions or any(
            trial.repetition != expected_repetition
            for expected_repetition, trial in enumerate(self.trials, start=1)
        ):
            raise ValueError("trials must preserve every requested repetition in order")
        conclusive_repetitions = tuple(
            trial.repetition for trial in self.trials if not trial.inconclusive_reasons
        )
        grouped_repetitions = tuple(
            sorted(repetition for group in self.outcome_groups for repetition in group.repetitions)
        )
        if grouped_repetitions != conclusive_repetitions:
            raise ValueError("outcome groups must partition every conclusive trial")
        for group in self.outcome_groups:
            representative_frame = self.trials[group.repetitions[0] - 1].observed_frame
            if representative_frame is None:
                raise ValueError("outcome groups require conclusive representative trials")
            comparison_kind = (
                group.representative_effects[0].kind if group.representative_effects else "action"
            )
            representative_effects = (
                _answer_outcomes(representative_frame)
                if comparison_kind == "answer"
                else tuple(
                    outcome for outcome in representative_frame.outcomes if outcome.kind == "action"
                )
            )
            if group.representative_effects != representative_effects:
                raise ValueError("outcome group effects must match their representative trial")
        expected_stability: TrialSetStability
        if len(conclusive_repetitions) != self.requested_repetitions:
            expected_stability = "inconclusive"
        elif len(self.outcome_groups) == 1:
            expected_stability = "stable"
        else:
            expected_stability = "unstable"
        if self.stability != expected_stability:
            raise ValueError("trial set stability must match its observed outcome groups")
        return self

    @property
    def representative_frame(self) -> SemanticFrame | None:
        if self.stability != "stable":
            return None
        representative_repetition = self.outcome_groups[0].repetitions[0]
        return self.trials[representative_repetition - 1].observed_frame


class DatasetEvaluationCase(_StrictULModel):
    candidate: DatasetAugmentationCandidate
    verdict: CaseVerdict
    trial_set: DatasetEvaluationTrialSet | None = None
    findings: tuple[DatasetEvaluationFinding, ...] = ()
    inconclusive_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_execution_state(self) -> Self:
        if not self.candidate.passed and (
            self.trial_set is not None or self.findings or self.inconclusive_reasons
        ):
            raise ValueError("rejected candidates cannot have evaluation results")
        if self.candidate.passed and self.trial_set is None:
            raise ValueError("accepted candidates require trials")
        if self.findings and (self.trial_set is None or self.trial_set.stability != "stable"):
            raise ValueError("findings require stable observed trials")
        if self.findings and self.inconclusive_reasons:
            raise ValueError("inconclusive cases cannot have findings")
        if (
            self.trial_set is not None
            and self.trial_set.stability == "inconclusive"
            and not self.inconclusive_reasons
        ):
            raise ValueError("inconclusive trials require an inconclusive reason")
        expected_verdict: CaseVerdict
        if not self.candidate.passed:
            expected_verdict = "augmentation_rejected"
        elif self.inconclusive_reasons:
            expected_verdict = "inconclusive"
        elif self.findings or (
            self.trial_set is not None and self.trial_set.stability == "unstable"
        ):
            expected_verdict = "divergence_needs_review"
        else:
            expected_verdict = "no_divergence"
        if self.verdict != expected_verdict:
            raise ValueError("case verdict must match augmentation and evaluation results")
        return self

    @property
    def target_output(self) -> ObservedAgentOutput | None:
        return self.trial_set.trials[0].target_output if self.trial_set is not None else None

    @property
    def observed_frame(self) -> SemanticFrame | None:
        return self.trial_set.trials[0].observed_frame if self.trial_set is not None else None


class DatasetEvaluationBaseline(_StrictULModel):
    verdict: BaselineVerdict
    trial_set: DatasetEvaluationTrialSet
    findings: tuple[DatasetEvaluationFinding, ...] = ()
    inconclusive_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_execution_state(self) -> Self:
        if self.findings:
            raise ValueError("historical output is grounding evidence, not a review oracle")
        expected_verdict: BaselineVerdict
        if self.inconclusive_reasons or self.trial_set.stability != "stable":
            expected_verdict = "inconclusive"
        else:
            expected_verdict = "no_divergence"
        if self.verdict != expected_verdict:
            raise ValueError("baseline verdict must match its evaluation results")
        return self

    @property
    def target_output(self) -> ObservedAgentOutput | None:
        return self.trial_set.trials[0].target_output

    @property
    def observed_frame(self) -> SemanticFrame | None:
        return self.trial_set.trials[0].observed_frame


class DatasetSemanticCallCounts(_StrictULModel):
    actual_calls: int = Field(ge=0)
    cache_hits: int = Field(ge=0)

    @property
    def total_requests(self) -> int:
        return self.actual_calls + self.cache_hits


class DatasetEvaluationResult(_StrictULModel):
    evaluation_mode: Literal["variance"] = "variance"
    source: InteractionRecord
    augmentation: DatasetAugmentationResult
    baseline: DatasetEvaluationBaseline
    cases: tuple[DatasetEvaluationCase, ...]
    semantic_calls: DatasetSemanticCallCounts = DatasetSemanticCallCounts(
        actual_calls=0,
        cache_hits=0,
    )

    @model_validator(mode="after")
    def validate_lineage(self) -> Self:
        if len(self.augmentation.source_frames) != 1:
            raise ValueError("dataset evaluation requires exactly one source frame")
        if self.augmentation.source_frames[0].interaction_id != self.source.id:
            raise ValueError("source frame must reference the source interaction")
        if tuple(case.candidate for case in self.cases) != self.augmentation.candidates:
            raise ValueError("evaluation cases must preserve every augmentation candidate")
        for trial in self.baseline.trial_set.trials:
            if trial.observed_frame is not None and trial.observed_frame.interaction_id != (
                f"{self.source.id}:current_baseline:round-{trial.repetition}"
            ):
                raise ValueError("baseline frame must reference its original repetition")
        for case in self.cases:
            if case.trial_set is None:
                continue
            if (
                case.trial_set.requested_repetitions
                != self.baseline.trial_set.requested_repetitions
            ):
                raise ValueError("variation and original repetition counts must match")
            for trial in case.trial_set.trials:
                if trial.observed_frame is not None and trial.observed_frame.interaction_id != (
                    f"{self.source.id}:{case.candidate.operator_id}:round-{trial.repetition}"
                ):
                    raise ValueError("variation frame must reference its repetition")
        return self


class DatasetEvaluationRunner:
    def __init__(
        self,
        augmentation_engine: DatasetAugmentationEngine,
        deconstructor: SemanticDeconstructor,
        environment: EnvironmentExecutor,
        *,
        target_timeout_seconds: float = 30,
        allow_network_egress: bool = False,
        evaluation_mode: DatasetEvaluationMode = "variance",
    ) -> None:
        if evaluation_mode != "variance":
            raise ValueError(
                f"evaluation mode '{evaluation_mode}' is not implemented; use 'variance' to "
                "compare fresh original replays with generated variations without assessing "
                "correctness"
            )
        if not math.isfinite(target_timeout_seconds) or target_timeout_seconds <= 0:
            raise ValueError("target_timeout_seconds must be positive and finite")
        if environment.capabilities.isolation != "customer_managed":
            raise ValueError("dataset execution requires a customer-managed environment")
        if not allow_network_egress:
            raise ValueError("remote environment API access requires explicit network opt-in")
        self._augmentation_engine = augmentation_engine
        self._deconstructor = deconstructor
        self._environment = environment
        self._target_timeout_seconds = target_timeout_seconds
        self._evaluation_mode: Literal["variance"] = evaluation_mode
        self._target_state_uncertain = False

    async def run(
        self,
        source: InteractionRecord,
        *,
        operator_ids: Iterable[str] = ("input.surface.rephrase",),
        repetitions: int = 3,
        precomputed_augmentation: DatasetAugmentationResult | None = None,
        augmentation_checkpoint_callback: Callable[[DatasetAugmentationResult], None] | None = None,
        prior_trials: dict[str, DatasetEvaluationTrial] | None = None,
        trial_started_callback: Callable[[DatasetTrialUnit], None] | None = None,
        trial_terminal_callback: (
            Callable[[DatasetTrialUnit, DatasetEvaluationTrial], None] | None
        ) = None,
    ) -> DatasetEvaluationResult:
        if type(repetitions) is not int or repetitions < 1:
            raise ValueError("repetitions must be a positive integer")
        starting_actual_calls, starting_cache_hits = self._semantic_call_metrics()
        if precomputed_augmentation is None:
            augmentation = await self._augmentation_engine.augment(
                (source,), operator_ids=operator_ids
            )
            _validate_precomputed_augmentation(source, augmentation, operator_ids=None)
            if augmentation_checkpoint_callback is not None:
                augmentation_checkpoint_callback(augmentation)
        else:
            augmentation = precomputed_augmentation
            _validate_precomputed_augmentation(source, augmentation, operator_ids=operator_ids)
        source_frame = augmentation.source_frames[0]
        comparison_surface = _comparison_surface(source_frame)
        if comparison_surface == "action":
            source_action_issues = _action_outcome_reliability_issues(
                source_frame,
                source.raw_observed_output,
                source.raw_input,
                require_input_grounded_fields=True,
            )
            if source_action_issues:
                raise ValueError(
                    f"source action outcomes are inconclusive: {source_action_issues[0]}"
                )
        else:
            source_answer_issues = _answer_outcome_reliability_issues(
                source_frame,
                source.raw_observed_output,
            )
            if source_answer_issues:
                raise ValueError(
                    f"source answer outcomes are inconclusive: {source_answer_issues[0]}"
                )
        accepted_candidates = tuple(
            candidate for candidate in augmentation.candidates if candidate.passed
        )
        baseline_trials: list[DatasetEvaluationTrial] = []
        candidate_trials: dict[str, list[DatasetEvaluationTrial]] = {
            candidate.operator_id: [] for candidate in accepted_candidates
        }
        recovered_trials = prior_trials or {}
        for repetition in range(1, repetitions + 1):
            baseline_unit = DatasetTrialUnit(
                interaction_id=source.id,
                operator_id="current_baseline",
                arm="original",
                repetition=repetition,
            )
            baseline_trial = recovered_trials.get(baseline_unit.id)
            if baseline_trial is None:
                if trial_started_callback is not None:
                    trial_started_callback(baseline_unit)
                baseline_trial = await self._execute_trial(
                    repetition=repetition,
                    interaction_id=f"{source.id}:current_baseline:round-{repetition}",
                    raw_input=source.raw_input,
                    reference_frame=source_frame,
                    source=source,
                    subject="current baseline",
                    variation_id="current_baseline",
                    comparison_surface=comparison_surface,
                )
                if trial_terminal_callback is not None:
                    trial_terminal_callback(baseline_unit, baseline_trial)
            baseline_trials.append(baseline_trial)
            for candidate in accepted_candidates:
                candidate_unit = DatasetTrialUnit(
                    interaction_id=source.id,
                    operator_id=candidate.operator_id,
                    arm="probe",
                    repetition=repetition,
                )
                recovered_candidate_trial = recovered_trials.get(candidate_unit.id)
                if recovered_candidate_trial is not None:
                    candidate_trials[candidate.operator_id].append(recovered_candidate_trial)
                    continue
                if baseline_trial.inconclusive_reasons:
                    skipped_trial = DatasetEvaluationTrial(
                        repetition=repetition,
                        inconclusive_reasons=(
                            "paired original repetition was inconclusive; variation not executed",
                        ),
                    )
                    candidate_trials[candidate.operator_id].append(skipped_trial)
                    if trial_terminal_callback is not None:
                        trial_terminal_callback(candidate_unit, skipped_trial)
                    continue
                reference_frame = baseline_trial.observed_frame
                if reference_frame is None:
                    raise AssertionError("conclusive baseline trial requires an observed frame")
                if trial_started_callback is not None:
                    trial_started_callback(candidate_unit)
                candidate_trial = await self._execute_trial(
                    repetition=repetition,
                    interaction_id=(f"{source.id}:{candidate.operator_id}:round-{repetition}"),
                    raw_input=candidate.augmented_input,
                    reference_frame=reference_frame,
                    source=source,
                    subject="variation",
                    variation_id=candidate.operator_id,
                    comparison_surface=comparison_surface,
                )
                candidate_trials[candidate.operator_id].append(candidate_trial)
                if trial_terminal_callback is not None:
                    trial_terminal_callback(candidate_unit, candidate_trial)

        baseline_trial_set = _group_evaluation_trials(
            tuple(baseline_trials), source.raw_input, source_frame, comparison_surface
        )
        baseline = self._classify_baseline(baseline_trial_set)
        cases: list[DatasetEvaluationCase] = []
        for candidate in augmentation.candidates:
            if not candidate.passed:
                cases.append(
                    DatasetEvaluationCase(candidate=candidate, verdict="augmentation_rejected")
                )
                continue
            trial_set = _group_evaluation_trials(
                tuple(candidate_trials[candidate.operator_id]),
                source.raw_input,
                source_frame,
                comparison_surface,
            )
            if trial_set.stability == "inconclusive":
                cases.append(
                    DatasetEvaluationCase(
                        candidate=candidate,
                        verdict="inconclusive",
                        trial_set=trial_set,
                        inconclusive_reasons=(
                            *_trial_inconclusive_reasons(trial_set),
                            *(
                                f"original repetition {trial.repetition} is inconclusive: {reason}"
                                for trial in baseline_trial_set.trials
                                for reason in trial.inconclusive_reasons
                            ),
                        ),
                    )
                )
                continue
            if baseline_trial_set.stability == "unstable":
                instability_reasons = ["original repetitions produced multiple outcomes"]
                if trial_set.stability == "unstable":
                    instability_reasons.append("variation repetitions produced multiple outcomes")
                cases.append(
                    DatasetEvaluationCase(
                        candidate=candidate,
                        verdict="inconclusive",
                        trial_set=trial_set,
                        inconclusive_reasons=tuple(instability_reasons),
                    )
                )
                continue
            if baseline_trial_set.stability == "inconclusive":
                cases.append(
                    DatasetEvaluationCase(
                        candidate=candidate,
                        verdict="inconclusive",
                        trial_set=trial_set,
                        inconclusive_reasons=tuple(
                            f"original repetition {trial.repetition} is inconclusive: {reason}"
                            for trial in baseline_trial_set.trials
                            for reason in trial.inconclusive_reasons
                        ),
                    )
                )
                continue
            if trial_set.stability == "unstable":
                cases.append(
                    DatasetEvaluationCase(
                        candidate=candidate,
                        verdict="divergence_needs_review",
                        trial_set=trial_set,
                    )
                )
                continue
            baseline_frame = baseline_trial_set.representative_frame
            observed_frame = trial_set.representative_frame
            if baseline_frame is None or observed_frame is None:
                raise AssertionError("stable trial sets require representative frames")
            findings = (
                _compare_action_outcomes(
                    baseline_frame,
                    observed_frame,
                    source.raw_input,
                    grounding_frame=source_frame,
                )
                if comparison_surface == "action"
                else _compare_answer_outcomes(baseline_frame, observed_frame)
            )
            cases.append(
                DatasetEvaluationCase(
                    candidate=candidate,
                    verdict=("divergence_needs_review" if findings else "no_divergence"),
                    trial_set=trial_set,
                    findings=findings,
                )
            )
        ending_actual_calls, ending_cache_hits = self._semantic_call_metrics()
        return DatasetEvaluationResult(
            evaluation_mode=self._evaluation_mode,
            source=source,
            augmentation=augmentation,
            baseline=baseline,
            cases=tuple(cases),
            semantic_calls=DatasetSemanticCallCounts(
                actual_calls=ending_actual_calls - starting_actual_calls,
                cache_hits=ending_cache_hits - starting_cache_hits,
            ),
        )

    def _semantic_call_metrics(self) -> tuple[int, int]:
        metrics = getattr(self._deconstructor, "semantic_call_metrics", None)
        if metrics is None:
            return (0, 0)
        actual_calls = getattr(metrics, "actual_calls", None)
        cache_hits = getattr(metrics, "cache_hits", None)
        if (
            type(actual_calls) is not int
            or actual_calls < 0
            or type(cache_hits) is not int
            or cache_hits < 0
        ):
            raise ValueError("semantic call metrics are invalid")
        return actual_calls, cache_hits

    async def _execute_trial(
        self,
        *,
        repetition: int,
        interaction_id: str,
        raw_input: str,
        reference_frame: SemanticFrame,
        source: InteractionRecord,
        subject: Literal["current baseline", "variation"],
        variation_id: str,
        comparison_surface: ComparisonSurface,
    ) -> DatasetEvaluationTrial:
        if self._target_state_uncertain:
            return DatasetEvaluationTrial(
                repetition=repetition,
                inconclusive_reasons=(
                    f"{subject} not executed because target state is uncertain; "
                    "environment state may remain",
                ),
            )
        try:
            async with asyncio.timeout(self._target_timeout_seconds):
                evaluation_case = EvaluationCase(
                    id=source.id,
                    turns=(
                        ConversationTurn(
                            id=f"turn-{secrets.token_hex(12)}",
                            role=ConversationRole.USER,
                            content=raw_input,
                        ),
                    ),
                    max_environment_api_calls=1,
                    timeout_seconds=self._target_timeout_seconds,
                    probe_context={
                        **source.probe_context(raw_input),
                        "ul.variation.id": variation_id,
                        "ul.repetition": repetition,
                    },
                )
                environment_api_calls = self._environment.api_calls_for_case(evaluation_case)
                if type(environment_api_calls) is not int or environment_api_calls < 1:
                    raise RuntimeError("environment returned an invalid API call count")
                evaluation_case = evaluation_case.model_copy(
                    update={"max_environment_api_calls": environment_api_calls}
                )
                execution_evidence = await self._environment.execute(evaluation_case)
        except OutcomeProjectionExecutionError as error:
            return self._outcome_projection_failure_trial(
                repetition=repetition,
                subject=subject,
                error=error,
                completed_phases=error.completed_phases,
                cleanup_reset_failed=error.cleanup_reset_failed,
                target_safe_to_reuse=error.target_safe_to_reuse,
            )
        except asyncio.CancelledError:
            self._target_state_uncertain = True
            raise DatasetTargetDeliveryUncertain(
                "target delivery is uncertain; environment quarantined and trial not retried"
            ) from None
        except TimeoutError:
            if environment_timeout_requires_quarantine(self._environment.capabilities):
                self._target_state_uncertain = True
            return DatasetEvaluationTrial(
                repetition=repetition,
                inconclusive_reasons=(f"{subject} execution timed out",),
            )
        except RuntimeError:
            return DatasetEvaluationTrial(
                repetition=repetition,
                inconclusive_reasons=(f"{subject} execution failed",),
            )
        try:
            validate_execution_evidence(evaluation_case, self._environment, execution_evidence)
        except OutcomeProjectionError as error:
            target_safe_to_reuse = (
                execution_evidence.evidence_scope == "response_and_state"
                and not execution_evidence_requires_quarantine(execution_evidence)
            )
            return self._outcome_projection_failure_trial(
                repetition=repetition,
                subject=subject,
                error=error,
                completed_phases=execution_evidence.lifecycle.completed_phases,
                cleanup_reset_failed=execution_evidence.lifecycle.cleanup == "failed",
                target_safe_to_reuse=target_safe_to_reuse,
            )
        lifecycle = execution_evidence.lifecycle
        if lifecycle.terminal_status != "succeeded":
            environment_state_may_remain = execution_evidence_requires_quarantine(
                execution_evidence
            )
            if environment_state_may_remain:
                self._target_state_uncertain = True
            cleanup_reason = (
                "; cleanup reset also failed; environment state may remain"
                if lifecycle.cleanup == "failed" and lifecycle.failed_phase != "cleanup_reset"
                else ("; environment state may remain" if environment_state_may_remain else "")
            )
            return DatasetEvaluationTrial(
                repetition=repetition,
                inconclusive_reasons=(
                    f"{subject} lifecycle failed during {lifecycle.failed_phase}{cleanup_reason}",
                ),
                lifecycle_failure=DatasetTargetLifecycleFailure(
                    failed_phase=lifecycle.failed_phase or "unknown",
                    completed_phases=lifecycle.completed_phases,
                    cleanup_reset_failed=lifecycle.cleanup == "failed",
                    environment_state_may_remain=environment_state_may_remain,
                ),
                execution_evidence=execution_evidence,
            )
        if len(execution_evidence.turns) != 1:
            raise RuntimeError("environment returned invalid single-turn evidence")
        target_output = observed_outputs_from_evidence(execution_evidence)[0]
        record = InteractionRecord(
            id=interaction_id,
            raw_input=raw_input,
            raw_observed_output=target_output.raw_output,
        )
        try:
            observed_frame = await self._deconstructor.deconstruct(record, reference_frame)
        except ValueError:
            return DatasetEvaluationTrial(
                repetition=repetition,
                execution_evidence=execution_evidence,
                target_output=target_output,
                inconclusive_reasons=(f"{subject} output could not be semantically deconstructed",),
            )
        if observed_frame.interaction_id != record.id:
            raise ValueError(f"observed frame must reference its {subject} interaction")
        inconclusive_reasons = (
            _action_outcome_reliability_issues(
                observed_frame,
                target_output.raw_output,
                source.raw_input,
                reference_frame=reference_frame,
            )
            if comparison_surface == "action"
            else _answer_outcome_reliability_issues(
                observed_frame,
                target_output.raw_output,
            )
        )
        if inconclusive_reasons:
            return DatasetEvaluationTrial(
                repetition=repetition,
                execution_evidence=execution_evidence,
                target_output=target_output,
                observed_frame=observed_frame,
                inconclusive_reasons=inconclusive_reasons,
            )
        return DatasetEvaluationTrial(
            repetition=repetition,
            execution_evidence=execution_evidence,
            target_output=target_output,
            observed_frame=observed_frame,
        )

    def _outcome_projection_failure_trial(
        self,
        *,
        repetition: int,
        subject: str,
        error: OutcomeProjectionError,
        completed_phases: tuple[str, ...],
        cleanup_reset_failed: bool,
        target_safe_to_reuse: bool,
    ) -> DatasetEvaluationTrial:
        environment_state_may_remain = not target_safe_to_reuse
        if environment_state_may_remain:
            self._target_state_uncertain = True
        reuse_status = (
            "verified cleanup left the target reusable"
            if target_safe_to_reuse
            else "target reuse is unverified; restore a known-safe fixture before continuing"
        )
        return DatasetEvaluationTrial(
            repetition=repetition,
            inconclusive_reasons=(
                f"{subject} target execution completed, but outcome field {error.field!r} at "
                f"selector {error.selector!r} {error.reason}; result evaluation was not run; "
                f"{reuse_status}",
            ),
            lifecycle_failure=DatasetTargetLifecycleFailure(
                failed_phase="outcome_projection",
                completed_phases=completed_phases,
                cleanup_reset_failed=cleanup_reset_failed,
                environment_state_may_remain=environment_state_may_remain,
            ),
        )

    @staticmethod
    def _classify_baseline(
        trial_set: DatasetEvaluationTrialSet,
    ) -> DatasetEvaluationBaseline:
        if trial_set.stability == "inconclusive":
            return DatasetEvaluationBaseline(
                verdict="inconclusive",
                trial_set=trial_set,
                inconclusive_reasons=_trial_inconclusive_reasons(trial_set),
            )
        if trial_set.stability == "unstable":
            return DatasetEvaluationBaseline(
                verdict="inconclusive",
                trial_set=trial_set,
                inconclusive_reasons=("original repetitions produced multiple outcomes",),
            )
        observed_frame = trial_set.representative_frame
        if observed_frame is None:
            raise AssertionError("stable baseline requires a representative frame")
        return DatasetEvaluationBaseline(
            verdict="no_divergence",
            trial_set=trial_set,
        )


def _validate_precomputed_augmentation(
    source: InteractionRecord,
    augmentation: DatasetAugmentationResult,
    *,
    operator_ids: Iterable[str] | None,
) -> None:
    if augmentation.source_records != (source,):
        raise ValueError("precomputed augmentation does not match the source interaction")
    if len(augmentation.source_frames) != 1:
        raise ValueError("dataset evaluation requires exactly one source frame")
    source_frame = augmentation.source_frames[0]
    if source_frame.interaction_id != source.id:
        raise ValueError("precomputed augmentation does not match the source interaction")

    known_operators = {
        (operator.id, operator.version): operator
        for operator in builtin_dataset_augmentation_operators()
    }
    candidate_references: list[tuple[str, str]] = []
    for candidate in augmentation.candidates:
        candidate_reference = (candidate.operator_id, candidate.operator_version)
        if candidate_reference not in known_operators:
            raise ValueError("precomputed augmentation contains an unknown operator reference")
        if candidate.source_interaction_id != source.id:
            raise ValueError("precomputed augmentation candidate does not match the source")
        if candidate.expected_input_frame.interaction_id != source.id:
            raise ValueError("precomputed augmentation candidate has invalid source lineage")
        expected_reparsed_interaction_id = f"{source.id}:{candidate.operator_id}"
        if (
            candidate.reparsed_input_frame is not None
            and candidate.reparsed_input_frame.interaction_id != expected_reparsed_interaction_id
        ):
            raise ValueError("precomputed augmentation candidate has invalid reparsed lineage")
        candidate_references.append(candidate_reference)
    if len(candidate_references) != len(set(candidate_references)):
        raise ValueError("precomputed augmentation contains duplicate operator references")

    stored_references = tuple(
        (reference.id, reference.version) for reference in augmentation.operator_references
    )
    if any(reference not in known_operators for reference in stored_references):
        raise ValueError("precomputed augmentation contains an unknown operator reference")
    requested_references = stored_references
    if operator_ids is not None:
        selected_operator_ids = tuple(islice(operator_ids, len(known_operators) + 1))
        if not selected_operator_ids:
            raise ValueError("operator_ids must contain at least one operator")
        if len(selected_operator_ids) > len(known_operators):
            raise ValueError("operator count exceeds the built-in library")
        if len(selected_operator_ids) != len(set(selected_operator_ids)):
            raise ValueError("operator identifiers must be unique")
        operators_by_id = {operator.id: operator for operator in known_operators.values()}
        if any(operator_id not in operators_by_id for operator_id in selected_operator_ids):
            raise ValueError("operator identifiers contain an unknown operator")
        requested_references = tuple(
            (operators_by_id[operator_id].id, operators_by_id[operator_id].version)
            for operator_id in selected_operator_ids
        )
        if stored_references != requested_references:
            raise ValueError("precomputed augmentation operators do not match the evaluation plan")
    requested_positions = {
        reference: position for position, reference in enumerate(requested_references)
    }
    if any(reference not in requested_positions for reference in candidate_references) or tuple(
        requested_positions[reference] for reference in candidate_references
    ) != tuple(sorted(requested_positions[reference] for reference in candidate_references)):
        raise ValueError("precomputed augmentation operators do not match the evaluation plan")


def _group_evaluation_trials(
    trials: tuple[DatasetEvaluationTrial, ...],
    source_input: str,
    source_frame: SemanticFrame,
    comparison_surface: ComparisonSurface,
) -> DatasetEvaluationTrialSet:
    grouped_repetitions: dict[tuple[str, ...], list[int]] = {}
    representative_effects: dict[tuple[str, ...], tuple[ObservedOutcome, ...]] = {}
    for trial in trials:
        if trial.inconclusive_reasons:
            continue
        observed_frame = trial.observed_frame
        if observed_frame is None:
            raise AssertionError("conclusive trials require observed frames")
        action_effects = tuple(
            outcome for outcome in observed_frame.outcomes if outcome.kind == "action"
        )
        grounded_names_by_outcome, association_issues = (
            _input_grounded_action_field_names_by_outcome(
                observed_frame,
                source_input,
                reference_frame=source_frame,
            )
        )
        if association_issues:
            raise AssertionError("reliable trials require unambiguous action grounding")
        action_keys: list[str] = []
        for effect in action_effects:
            grounded_fields: dict[str, JsonValue] = {
                name: _observable_field_key(name, value)
                for name, value in effect.fields.items()
                if name in grounded_names_by_outcome.get(effect.id, set())
            }
            action_keys.append(_json_key([effect.predicate, grounded_fields]))
        answer_effects = _answer_outcomes(observed_frame)
        answer_keys = tuple(
            _json_key(["answer", *_response_outcome_semantics(outcome)])
            for outcome in answer_effects
        )
        outcome_key = tuple(sorted(action_keys)) if comparison_surface == "action" else answer_keys
        grouped_repetitions.setdefault(outcome_key, []).append(trial.repetition)
        representative_effects.setdefault(
            outcome_key,
            action_effects if comparison_surface == "action" else answer_effects,
        )
    outcome_groups = tuple(
        DatasetEvaluationOutcomeGroup(
            repetitions=tuple(repetitions),
            representative_effects=representative_effects[outcome_key],
        )
        for outcome_key, repetitions in grouped_repetitions.items()
    )
    conclusive_trial_count = sum(not trial.inconclusive_reasons for trial in trials)
    if conclusive_trial_count != len(trials):
        stability: TrialSetStability = "inconclusive"
    elif len(outcome_groups) == 1:
        stability = "stable"
    else:
        stability = "unstable"
    return DatasetEvaluationTrialSet(
        requested_repetitions=len(trials),
        stability=stability,
        trials=trials,
        outcome_groups=outcome_groups,
    )


def _trial_inconclusive_reasons(
    trial_set: DatasetEvaluationTrialSet,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(reason for trial in trial_set.trials for reason in trial.inconclusive_reasons)
    )


def _action_outcome_reliability_issues(
    frame: SemanticFrame,
    raw_observed_output: JsonValue,
    source_input: str,
    *,
    reference_frame: SemanticFrame | None = None,
    require_input_grounded_fields: bool = False,
) -> tuple[str, ...]:
    issues: list[str] = []
    input_grounded_field_names_by_outcome, association_issues = (
        _input_grounded_action_field_names_by_outcome(
            frame,
            source_input,
            reference_frame=reference_frame,
        )
    )
    issues.extend(association_issues)
    for outcome in frame.outcomes:
        if outcome.kind != "action":
            continue
        input_grounded_field_names = input_grounded_field_names_by_outcome.get(outcome.id, set())
        if outcome.status.casefold() != "observed":
            issues.append(f"action outcome {outcome.id} is not affirmatively observed")
        if outcome.confidence < 1:
            issues.append(f"action outcome {outcome.id} has confidence below 1")
        evidenced_action_objects: list[dict[str, JsonValue]] = []
        for evidence in outcome.evidence:
            if evidence.source != "output":
                continue
            try:
                evidence_value = _resolve_output_pointer(
                    raw_observed_output,
                    evidence.json_pointer,
                )
            except ValueError:
                issues.append(f"action outcome {outcome.id} has invalid output evidence")
                continue
            if isinstance(evidence_value, dict):
                action_value = evidence_value.get("action")
                if _json_key(action_value) == _json_key(outcome.predicate):
                    evidenced_action_objects.append(evidence_value)
                    continue
                issues.append(f"action outcome {outcome.id} has non-action object evidence")
                continue
            if isinstance(evidence_value, list):
                issues.append(f"action outcome {outcome.id} has non-primitive output evidence")
                continue
            if _json_key(evidence_value) == _json_key(outcome.predicate):
                try:
                    parent_value = _resolve_output_pointer(
                        raw_observed_output,
                        evidence.json_pointer.rsplit("/", 1)[0],
                    )
                except ValueError:
                    continue
                if isinstance(parent_value, dict):
                    evidenced_action_objects.append(parent_value)
        grounded_fields = {
            name: value
            for name, value in outcome.fields.items()
            if name in input_grounded_field_names
        }
        if not evidenced_action_objects:
            issues.append(f"action outcome {outcome.id} predicate lacks coherent action evidence")
        elif grounded_fields and not any(
            all(
                name in action_object and _observable_values_equal(action_object[name], value)
                for name, value in grounded_fields.items()
            )
            for action_object in evidenced_action_objects
        ):
            issues.append(
                f"action outcome {outcome.id} grounded fields lack one coherent action record: "
                f"{', '.join(sorted(grounded_fields))}"
            )
        if require_input_grounded_fields and not grounded_fields:
            issues.append(f"action outcome {outcome.id} has no input-grounded fields")
    return tuple(issues)


def _answer_outcome_reliability_issues(
    frame: SemanticFrame,
    raw_observed_output: JsonValue,
) -> tuple[str, ...]:
    answers = _answer_outcomes(frame)
    if not answers:
        return ("observed response produced no answer outcome",)
    issues: list[str] = []
    for outcome in answers:
        if outcome.status.casefold() != "observed":
            issues.append(f"answer outcome {outcome.id} is not affirmatively observed")
        if outcome.confidence < 1:
            issues.append(f"answer outcome {outcome.id} has confidence below 1")
        output_evidence = tuple(
            evidence for evidence in outcome.evidence if evidence.source == "output"
        )
        if not output_evidence:
            issues.append(f"answer outcome {outcome.id} has no output evidence")
            continue
        try:
            for evidence in output_evidence:
                _resolve_output_pointer(raw_observed_output, evidence.json_pointer)
        except ValueError:
            issues.append(f"answer outcome {outcome.id} has invalid output evidence")
    return tuple(issues)


def _input_grounded_action_field_names_by_outcome(
    frame: SemanticFrame,
    source_input: str,
    *,
    reference_frame: SemanticFrame | None = None,
) -> tuple[dict[str, set[str]], tuple[str, ...]]:
    grounding_frame = reference_frame or frame
    grounded_names_by_reference_id = {
        outcome.id: {
            name
            for name, value in outcome.fields.items()
            if _value_appears_in_input(value, source_input)
        }
        for outcome in grounding_frame.outcomes
        if outcome.kind == "action"
    }
    if reference_frame is None or frame is reference_frame:
        return grounded_names_by_reference_id, ()

    reference_by_predicate = _action_outcomes_by_key(grounding_frame)
    grounded_names_by_outcome: dict[str, set[str]] = {}
    association_issues: list[str] = []
    frame_by_predicate = _action_outcomes_by_key(frame)
    for key, outcomes in frame_by_predicate.items():
        reference_outcomes = reference_by_predicate.get(key, ())
        if not reference_outcomes:
            grounded_names_by_outcome.update(
                (
                    outcome.id,
                    {
                        name
                        for name, value in outcome.fields.items()
                        if _value_appears_in_input(value, source_input)
                    },
                )
                for outcome in outcomes
            )
            continue
        if len(reference_outcomes) == 1:
            grounded_names_by_outcome.update(
                (outcome.id, grounded_names_by_reference_id[reference_outcomes[0].id])
                for outcome in outcomes
            )
            continue
        remaining_outcomes = list(outcomes)
        remaining_references = list(reference_outcomes)
        while remaining_outcomes and remaining_references:
            proposals: list[tuple[ObservedOutcome, ObservedOutcome]] = []
            for outcome in remaining_outcomes:
                scored_references = [
                    (
                        sum(
                            name in outcome.fields
                            and _observable_values_equal(
                                outcome.fields[name], reference.fields[name]
                            )
                            for name in grounded_names_by_reference_id[reference.id]
                        ),
                        reference,
                    )
                    for reference in remaining_references
                ]
                highest_score = max(score for score, _ in scored_references)
                best_references = [
                    reference for score, reference in scored_references if score == highest_score
                ]
                if highest_score > 0 and len(best_references) == 1:
                    proposals.append((outcome, best_references[0]))
            uniquely_proposed_references = {
                reference.id
                for _, reference in proposals
                if sum(candidate.id == reference.id for _, candidate in proposals) == 1
            }
            accepted_proposals = [
                (outcome, reference)
                for outcome, reference in proposals
                if reference.id in uniquely_proposed_references
            ]
            if not accepted_proposals:
                break
            for outcome, reference in accepted_proposals:
                grounded_names_by_outcome[outcome.id] = grounded_names_by_reference_id[reference.id]
            accepted_outcome_ids = {outcome.id for outcome, _ in accepted_proposals}
            accepted_reference_ids = {reference.id for _, reference in accepted_proposals}
            remaining_outcomes = [
                outcome for outcome in remaining_outcomes if outcome.id not in accepted_outcome_ids
            ]
            remaining_references = [
                reference
                for reference in remaining_references
                if reference.id not in accepted_reference_ids
            ]
        for outcome in remaining_outcomes:
            association_issues.append(
                f"action outcome {outcome.id} cannot be safely associated with an "
                "input-grounded source action"
            )
    return grounded_names_by_outcome, tuple(association_issues)


def _value_appears_in_input(value: JsonValue, source_input: str) -> bool:
    if value is None or isinstance(value, (dict, list)):
        return False
    if isinstance(value, str) and not value.strip():
        return False
    numeric_text: str | None = None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric_text = str(value)
    elif (
        isinstance(value, str)
        and re.fullmatch(_NUMBER_PATTERN, value.strip(), re.IGNORECASE)
        and any(marker in value.casefold() for marker in (".", ",", "e"))
    ):
        numeric_text = value.strip()
    if numeric_text is not None:
        try:
            normalized_value = Decimal(numeric_text.replace(",", ""))
        except InvalidOperation:
            return False
        bounded_number_pattern = rf"(?<![\w.]){_NUMBER_PATTERN}(?!\w|\.\d)"
        for match in re.finditer(bounded_number_pattern, source_input, re.IGNORECASE):
            try:
                candidate_value = Decimal(match.group().replace(",", ""))
            except InvalidOperation:
                continue
            if candidate_value == normalized_value:
                return True
        return False
    value_text = str(value).casefold()
    return re.search(rf"(?<!\w){re.escape(value_text)}(?!\w)", source_input.casefold()) is not None


def _resolve_output_pointer(raw_observed_output: JsonValue, pointer: str) -> JsonValue:
    if pointer == "/raw_observed_output":
        return raw_observed_output
    prefix = "/raw_observed_output/"
    if not pointer.startswith(prefix):
        raise ValueError("action evidence must point below raw_observed_output")
    current: object = raw_observed_output
    for encoded_token in pointer[len(prefix) :].split("/"):
        token = encoded_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
            continue
        valid_array_index = token == "0" or (token.isdecimal() and not token.startswith("0"))
        if isinstance(current, list) and valid_array_index and int(token) < len(current):
            current = current[int(token)]
            continue
        raise ValueError("action evidence pointer does not resolve")
    return current


def _compare_action_outcomes(
    expected_frame: SemanticFrame,
    observed_frame: SemanticFrame,
    source_input: str,
    *,
    subject: str = "augmented input",
    grounding_frame: SemanticFrame | None = None,
) -> tuple[DatasetEvaluationFinding, ...]:
    expected_by_key = _action_outcomes_by_key(expected_frame)
    observed_by_key = _action_outcomes_by_key(observed_frame)
    input_grounded_field_names_by_outcome, association_issues = (
        _input_grounded_action_field_names_by_outcome(
            expected_frame,
            source_input,
            reference_frame=grounding_frame,
        )
    )
    if association_issues:
        raise AssertionError("expected action grounding must be unambiguous")
    findings: list[DatasetEvaluationFinding] = []
    for key in sorted(expected_by_key.keys() | observed_by_key.keys()):
        expected = expected_by_key.get(key, ())
        observed = observed_by_key.get(key, ())
        if not expected:
            findings.append(
                DatasetEvaluationFinding(
                    category="unexpected_effect",
                    message=(
                        f"Needs review: the {subject} produced an unexpected {key[1]} "
                        "action effect."
                    ),
                    observed_effects=observed,
                )
            )
            continue

        unmatched_expected, unmatched_observed = _remove_grounded_matches(
            expected,
            observed,
            input_grounded_field_names_by_outcome,
        )
        changed_count = min(len(unmatched_expected), len(unmatched_observed))
        for expected_effect, observed_effect in zip(
            unmatched_expected[:changed_count],
            unmatched_observed[:changed_count],
            strict=True,
        ):
            grounded_field_names = _changed_grounded_field_names(
                expected_effect,
                observed_effect,
                input_grounded_field_names_by_outcome[expected_effect.id],
            )
            findings.append(
                DatasetEvaluationFinding(
                    category="changed_grounded_effect_argument",
                    message=(
                        f"Needs review: the {subject} changed a grounded argument of the "
                        f"{key[1]} action effect."
                    ),
                    expected_effects=(expected_effect,),
                    observed_effects=(observed_effect,),
                    grounded_field_names=grounded_field_names,
                )
            )

        missing = unmatched_expected[changed_count:]
        if missing:
            findings.append(
                DatasetEvaluationFinding(
                    category="missing_effect",
                    message=(
                        f"Needs review: the {subject} produced {len(observed)} {key[1]} "
                        f"action effects instead of {len(expected)}."
                    ),
                    expected_effects=expected,
                    observed_effects=observed,
                )
            )
        remaining_observed = unmatched_observed[changed_count:]
        duplicate = tuple(
            effect
            for effect in remaining_observed
            if any(
                _grounded_effect_matches(
                    expected_effect,
                    effect,
                    input_grounded_field_names_by_outcome[expected_effect.id],
                )
                for expected_effect in expected
            )
        )
        if duplicate:
            findings.append(
                DatasetEvaluationFinding(
                    category="duplicate_effect",
                    message=(f"Needs review: the {subject} repeated a {key[1]} action effect."),
                    expected_effects=expected,
                    observed_effects=duplicate,
                )
            )
        unexpected = tuple(effect for effect in remaining_observed if effect not in duplicate)
        if unexpected:
            findings.append(
                DatasetEvaluationFinding(
                    category="unexpected_effect",
                    message=(
                        f"Needs review: the {subject} produced an unexpected {key[1]} "
                        "action effect."
                    ),
                    expected_effects=expected,
                    observed_effects=unexpected,
                )
            )
    return tuple(findings)


def compare_action_outcomes(
    expected_frame: SemanticFrame,
    observed_frame: SemanticFrame,
    source_input: str,
    *,
    grounding_frame: SemanticFrame | None = None,
) -> tuple[DatasetEvaluationFinding, ...]:
    return _compare_action_outcomes(
        expected_frame,
        observed_frame,
        source_input,
        grounding_frame=grounding_frame,
    )


def compare_observed_outcomes(
    expected_frame: SemanticFrame,
    observed_frame: SemanticFrame,
    source_input: str,
    *,
    grounding_frame: SemanticFrame | None = None,
) -> tuple[DatasetEvaluationFinding, ...]:
    return _compare_observed_outcomes(
        expected_frame,
        observed_frame,
        source_input,
        grounding_frame=grounding_frame,
    )


def _compare_observed_outcomes(
    expected_frame: SemanticFrame,
    observed_frame: SemanticFrame,
    source_input: str,
    *,
    subject: str = "augmented input",
    grounding_frame: SemanticFrame | None = None,
) -> tuple[DatasetEvaluationFinding, ...]:
    if _comparison_surface_or_action(expected_frame) == "action":
        return _compare_action_outcomes(
            expected_frame,
            observed_frame,
            source_input,
            subject=subject,
            grounding_frame=grounding_frame,
        )
    return _compare_answer_outcomes(expected_frame, observed_frame, subject=subject)


def _compare_answer_outcomes(
    expected_frame: SemanticFrame,
    observed_frame: SemanticFrame,
    *,
    subject: str = "augmented input",
) -> tuple[DatasetEvaluationFinding, ...]:
    expected_responses = _answer_outcomes(expected_frame)
    observed_responses = _answer_outcomes(observed_frame)
    if tuple(map(_response_outcome_semantics, expected_responses)) == tuple(
        map(_response_outcome_semantics, observed_responses)
    ):
        return ()
    return (
        DatasetEvaluationFinding(
            category="changed_response",
            message=f"Needs review: the {subject} produced a different observed response.",
            expected_effects=expected_responses,
            observed_effects=observed_responses,
        ),
    )


def _response_outcome_semantics(outcome: ObservedOutcome) -> tuple[JsonValue, ...]:
    return (
        outcome.position,
        outcome.kind,
        outcome.predicate,
        outcome.fields,
        list(outcome.propositions),
    )


def _answer_outcomes(frame: SemanticFrame) -> tuple[ObservedOutcome, ...]:
    return tuple(
        sorted(
            (outcome for outcome in frame.outcomes if outcome.kind == "answer"),
            key=lambda outcome: outcome.position,
        )
    )


def _comparison_surface(frame: SemanticFrame) -> ComparisonSurface:
    if any(outcome.kind == "action" for outcome in frame.outcomes):
        return "action"
    if any(outcome.kind == "answer" for outcome in frame.outcomes):
        return "answer"
    raise ValueError("source frame requires an observable action or answer outcome")


def _comparison_surface_or_action(frame: SemanticFrame) -> ComparisonSurface:
    try:
        return _comparison_surface(frame)
    except ValueError:
        return "action"


def _action_outcomes_by_key(
    frame: SemanticFrame,
) -> dict[tuple[str, str], tuple[ObservedOutcome, ...]]:
    grouped: defaultdict[tuple[str, str], list[ObservedOutcome]] = defaultdict(list)
    for outcome in frame.outcomes:
        if outcome.kind == "action":
            grouped[(outcome.kind, outcome.predicate)].append(outcome)
    return {key: tuple(outcomes) for key, outcomes in grouped.items()}


def _remove_grounded_matches(
    expected: tuple[ObservedOutcome, ...],
    observed: tuple[ObservedOutcome, ...],
    input_grounded_field_names_by_outcome: dict[str, set[str]],
) -> tuple[tuple[ObservedOutcome, ...], tuple[ObservedOutcome, ...]]:
    compatible_observed_indexes: list[tuple[int, ...]] = []
    for expected_effect in expected:
        compatible_observed_indexes.append(
            tuple(
                index
                for index, observed_effect in enumerate(observed)
                if _grounded_effect_matches(
                    expected_effect,
                    observed_effect,
                    input_grounded_field_names_by_outcome[expected_effect.id],
                )
            )
        )

    observed_matches: list[int | None] = [None] * len(observed)

    def match(expected_index: int, visited_observed_indexes: set[int]) -> bool:
        for observed_index in compatible_observed_indexes[expected_index]:
            if observed_index in visited_observed_indexes:
                continue
            visited_observed_indexes.add(observed_index)
            previous_expected_index = observed_matches[observed_index]
            if previous_expected_index is None or match(
                previous_expected_index, visited_observed_indexes
            ):
                observed_matches[observed_index] = expected_index
                return True
        return False

    for expected_index in range(len(expected)):
        match(expected_index, set())
    matched_expected_indexes = {
        expected_index for expected_index in observed_matches if expected_index is not None
    }
    return (
        tuple(
            effect for index, effect in enumerate(expected) if index not in matched_expected_indexes
        ),
        tuple(effect for index, effect in enumerate(observed) if observed_matches[index] is None),
    )


def _grounded_effect_matches(
    expected: ObservedOutcome,
    observed: ObservedOutcome,
    input_grounded_field_names: set[str],
) -> bool:
    expected_grounded_fields = {
        name: value for name, value in expected.fields.items() if name in input_grounded_field_names
    }
    return all(
        name in observed.fields
        and _observable_field_key(name, observed.fields[name]) == _observable_field_key(name, value)
        for name, value in expected_grounded_fields.items()
    )


def _changed_grounded_field_names(
    expected: ObservedOutcome,
    observed: ObservedOutcome,
    input_grounded_field_names: set[str],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name, value in expected.fields.items()
            if name in input_grounded_field_names
            and (
                name not in observed.fields
                or _observable_field_key(name, observed.fields[name])
                != _observable_field_key(name, value)
            )
        )
    )


def _json_key(value: JsonValue) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _observable_field_key(field_name: str, value: JsonValue) -> str:
    identifier_markers = {"code", "id", "identifier", "key", "number", "reference"}
    if identifier_markers.intersection(field_name.casefold().split("_")):
        return _json_key(value)
    numeric_key = _numeric_observable_key(value)
    if numeric_key is not None:
        return numeric_key
    return _json_key(value)


def _numeric_observable_key(value: JsonValue) -> str | None:
    if isinstance(value, (bool, dict, list)) or value is None:
        return None
    numeric_text: str
    if isinstance(value, (int, float)):
        numeric_text = str(value)
    else:
        if (
            re.fullmatch(
                r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:e[-+]?\d+)?",
                value,
                re.IGNORECASE,
            )
            is None
        ):
            return None
        numeric_text = value
    try:
        decimal_value = Decimal(numeric_text)
    except InvalidOperation:
        return None
    if not decimal_value.is_finite():
        return None
    if decimal_value.is_zero():
        return "number:0"
    sign, digits, decimal_exponent = decimal_value.as_tuple()
    if not isinstance(decimal_exponent, int):
        return None
    exponent = decimal_exponent
    normalized_digits = list(digits)
    while normalized_digits[-1] == 0:
        normalized_digits.pop()
        exponent += 1
    return f"number:{sign}:{''.join(str(digit) for digit in normalized_digits)}:{exponent}"


def _observable_values_equal(left: JsonValue, right: JsonValue) -> bool:
    if _json_key(left) == _json_key(right):
        return True
    left_numeric_key = _numeric_observable_key(left)
    return left_numeric_key is not None and left_numeric_key == _numeric_observable_key(right)
