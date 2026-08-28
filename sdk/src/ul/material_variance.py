from __future__ import annotations

import json
from typing import cast

from pydantic import JsonValue
from ul_core.dataset import ObservedOutcome
from ul_core.evaluators import EvaluationSubject, EvaluatorVersion, RubricEvaluator
from ul_core.prompts import PromptManager

from ul.dataset_evaluation import (
    ComparisonSurface,
    DatasetEvaluationFinding,
    MaterialVarianceAssessment,
    MaterialVarianceDecision,
    MaterialVarianceEvidence,
    MaterialVarianceReasonCode,
)
from ul.evaluators import (
    EvaluatorJudge,
    OpenAICompatibleJudgeConfig,
    create_evaluator_version,
    evaluate,
)

_PROMPTS = PromptManager.instance()
_MAXIMUM_PAYLOAD_BYTES = 64 * 1024
_EVALUATOR = RubricEvaluator(
    id="dataset.material_variance",
    rubric=_PROMPTS.get_prompt("evaluation.material_variance"),
    minimum_score=0,
    include_sources=("answer",),
)
_LABELS: dict[str, tuple[MaterialVarianceDecision, MaterialVarianceReasonCode, float]] = {
    "material_variance:action_added": ("material_variance", "action_added", 1),
    "material_variance:action_removed": ("material_variance", "action_removed", 1),
    "material_variance:action_count_changed": (
        "material_variance",
        "action_count_changed",
        1,
    ),
    "material_variance:action_target_changed": (
        "material_variance",
        "action_target_changed",
        1,
    ),
    "material_variance:grounded_argument_changed": (
        "material_variance",
        "grounded_argument_changed",
        1,
    ),
    "material_variance:committed_state_changed": (
        "material_variance",
        "committed_state_changed",
        1,
    ),
    "material_variance:response_meaning_changed": (
        "material_variance",
        "response_meaning_changed",
        1,
    ),
    "operationally_equivalent:alias_or_representation": (
        "operationally_equivalent",
        "alias_or_representation",
        0,
    ),
    "operationally_equivalent:presentation_only": (
        "operationally_equivalent",
        "presentation_only",
        0,
    ),
    "operationally_equivalent:lookup_path_only": (
        "operationally_equivalent",
        "lookup_path_only",
        0,
    ),
    "operationally_equivalent:same_real_world_effect": (
        "operationally_equivalent",
        "same_real_world_effect",
        0,
    ),
    "insufficient_evidence:missing_comparison_evidence": (
        "insufficient_evidence",
        "missing_comparison_evidence",
        0.5,
    ),
}
_EXPLANATIONS = {
    "material_variance": "The variation changed the observed real-world action or outcome.",
    "operationally_equivalent": (
        "The observed difference represents the same real-world action or outcome."
    ),
    "insufficient_evidence": "The available comparison evidence cannot establish materiality.",
}


class DatasetMaterialVarianceJudge:
    def __init__(self, judge: EvaluatorJudge, *, max_input_chars: int) -> None:
        if max_input_chars < 1:
            raise ValueError("material variance input limit must be positive")
        self._judge = judge
        self._max_input_chars = max_input_chars
        judge_version = getattr(judge, "version", None)
        if judge_version is None:
            raise ValueError("material variance judge requires a versioned configuration")
        self._evaluator_version: EvaluatorVersion = create_evaluator_version(
            _EVALUATOR,
            judge_version=judge_version,
        )
        self._actual_calls = 0

    @property
    def evaluator_version_id(self) -> str:
        return self._evaluator_version.id

    @property
    def actual_calls(self) -> int:
        return self._actual_calls

    async def evaluate(
        self,
        comparison_surface: ComparisonSurface,
        findings: tuple[DatasetEvaluationFinding, ...],
    ) -> MaterialVarianceAssessment:
        answer = _comparison_payload(comparison_surface, findings)
        encoded_answer = json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
        if (
            not findings
            or len(encoded_answer) > self._max_input_chars
            or len(encoded_answer.encode("utf-8")) > _MAXIMUM_PAYLOAD_BYTES
        ):
            return self._insufficient("missing_comparison_evidence")
        self._actual_calls += 1
        results = await evaluate(
            EvaluationSubject(agent_status="succeeded", answer=answer),
            (_EVALUATOR,),
            judge=self._judge,
            judge_version=self._evaluator_version.judge,
        )
        result = results.results[0]
        label_contract = _LABELS.get(result.label or "")
        if result.status == "evaluator_error" or label_contract is None:
            return self._insufficient("judge_error")
        decision, reason_code, required_score = label_contract
        if result.score != required_score or not _has_required_citations(
            decision,
            tuple(evidence.json_pointer for evidence in result.evidence),
        ):
            return self._insufficient("judge_error")
        return MaterialVarianceAssessment(
            decision=decision,
            reason_code=reason_code,
            explanation=_EXPLANATIONS[decision],
            evidence=tuple(
                MaterialVarianceEvidence(json_pointer=evidence.json_pointer)
                for evidence in result.evidence
            ),
            evaluator_version_id=self.evaluator_version_id,
        )

    def _insufficient(
        self,
        reason_code: MaterialVarianceReasonCode,
    ) -> MaterialVarianceAssessment:
        return MaterialVarianceAssessment(
            decision="insufficient_evidence",
            reason_code=reason_code,
            explanation=_EXPLANATIONS["insufficient_evidence"],
            evaluator_version_id=self.evaluator_version_id,
        )


def material_variance_evaluator_version(judge: EvaluatorJudge) -> EvaluatorVersion:
    judge_version = getattr(judge, "version", None)
    if judge_version is None:
        raise ValueError("material variance judge requires a versioned configuration")
    return create_evaluator_version(_EVALUATOR, judge_version=judge_version)


def material_variance_evaluator_version_from_config(
    config: OpenAICompatibleJudgeConfig,
) -> EvaluatorVersion:
    return create_evaluator_version(
        _EVALUATOR,
        judge_version=config.evaluator_judge_version(),
    )


def _comparison_payload(
    comparison_surface: ComparisonSurface,
    findings: tuple[DatasetEvaluationFinding, ...],
) -> dict[str, JsonValue]:
    return {
        "comparison_surface": comparison_surface,
        "findings": [
            {
                "category": finding.category,
                "grounded_field_names": list(finding.grounded_field_names),
                "baseline_effects": [
                    _effect_payload(effect, finding) for effect in finding.expected_effects
                ],
                "variation_effects": [
                    _effect_payload(effect, finding) for effect in finding.observed_effects
                ],
            }
            for finding in findings
        ],
    }


def _effect_payload(
    outcome: ObservedOutcome,
    finding: DatasetEvaluationFinding,
) -> JsonValue:
    kind = outcome.kind
    fields = outcome.fields
    selected_fields = (
        {name: fields[name] for name in finding.grounded_field_names if name in fields}
        if kind == "action"
        else fields
    )
    return cast(
        JsonValue,
        {
            "kind": kind,
            "predicate": outcome.predicate,
            "fields": selected_fields,
        },
    )


def _has_required_citations(decision: str, pointers: tuple[str, ...]) -> bool:
    if decision == "insufficient_evidence":
        return bool(pointers)
    return any("/baseline_effects" in pointer for pointer in pointers) and any(
        "/variation_effects" in pointer for pointer in pointers
    )
