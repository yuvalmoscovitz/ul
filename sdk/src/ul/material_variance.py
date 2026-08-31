from __future__ import annotations

import json
import re
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
    evaluator_judge_version_from_llm_config,
)
from ul.llm import LLMClientConfig

_PROMPTS = PromptManager.instance()
_MAXIMUM_PAYLOAD_BYTES = 64 * 1024
_READ_ONLY_ACTION_TOKENS = frozenset(
    {"FETCH", "FIND", "GET", "LIST", "LOOKUP", "QUERY", "READ", "SEARCH"}
)
_RESPONSE_ENVELOPE_FIELDS = frozenset({"answer", "actions", "reward", "status", "steps", "task_id"})
_EVALUATOR = RubricEvaluator(
    id="dataset.material_variance.v2",
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
        deterministic_assessment = _deterministic_response_material_variance(
            answer,
            evaluator_version_id=self.evaluator_version_id,
        )
        if deterministic_assessment is not None:
            return deterministic_assessment
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


def material_variance_evaluator_version_from_llm_config(
    config: LLMClientConfig,
) -> EvaluatorVersion:
    return create_evaluator_version(
        _EVALUATOR,
        judge_version=evaluator_judge_version_from_llm_config(config),
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
        else _normalized_response_fields(fields)
    )
    return cast(
        JsonValue,
        {
            "kind": kind,
            "predicate": outcome.predicate,
            "fields": selected_fields,
        },
    )


def _normalized_response_fields(fields: dict[str, JsonValue]) -> dict[str, JsonValue]:
    value = fields.get("value")
    if not isinstance(value, dict) or not {"answer", "actions"} <= value.keys():
        return fields
    answer = value.get("answer")
    actions = value.get("actions")
    substantive_answer = None if _is_empty_answer(answer) else answer
    normalized: dict[str, JsonValue] = {
        "substantive_answer_state": "empty" if substantive_answer is None else "present",
        "substantive_answer": substantive_answer,
    }
    if isinstance(actions, list):
        committed_actions = [action for action in actions if not _is_read_only_action(action)]
        normalized["committed_action_count"] = len(committed_actions)
        normalized["committed_actions"] = committed_actions
    else:
        normalized["committed_action_count"] = None
        normalized["committed_actions"] = actions
    other_fields = {
        key: nested_value
        for key, nested_value in value.items()
        if key not in _RESPONSE_ENVELOPE_FIELDS
    }
    if other_fields:
        normalized["other_fields"] = other_fields
    return normalized


def _is_empty_answer(answer: JsonValue) -> bool:
    if answer is None:
        return True
    if isinstance(answer, str):
        return not answer.strip()
    if isinstance(answer, (list, dict)):
        return not answer
    return False


def _is_read_only_action(action: JsonValue) -> bool:
    if not isinstance(action, dict):
        return False
    action_name = action.get("action", action.get("name"))
    if not isinstance(action_name, str):
        return False
    separated_name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", action_name)
    tokens = set(re.findall(r"[A-Za-z]+", separated_name.upper()))
    return bool(tokens & _READ_ONLY_ACTION_TOKENS)


def _deterministic_response_material_variance(
    answer: dict[str, JsonValue],
    *,
    evaluator_version_id: str,
) -> MaterialVarianceAssessment | None:
    findings = answer.get("findings")
    if not isinstance(findings, list):
        return None
    for finding_index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            continue
        baseline_effects = finding.get("baseline_effects")
        variation_effects = finding.get("variation_effects")
        baseline_facts = _response_safety_facts(baseline_effects)
        variation_facts = _response_safety_facts(variation_effects)
        if baseline_facts is None or variation_facts is None:
            continue
        baseline_action_count, baseline_has_answer = baseline_facts
        variation_action_count, variation_has_answer = variation_facts
        if baseline_action_count != variation_action_count:
            reason_code: MaterialVarianceReasonCode
            if baseline_action_count == 0:
                reason_code = "action_added"
            elif variation_action_count == 0:
                reason_code = "action_removed"
            else:
                reason_code = "action_count_changed"
            return _deterministic_material_assessment(
                finding_index,
                reason_code,
                evaluator_version_id=evaluator_version_id,
            )
        if baseline_has_answer != variation_has_answer:
            return _deterministic_material_assessment(
                finding_index,
                "response_meaning_changed",
                evaluator_version_id=evaluator_version_id,
            )
    return None


def _response_safety_facts(effects: JsonValue) -> tuple[int, bool] | None:
    if not isinstance(effects, list) or len(effects) != 1:
        return None
    effect = effects[0]
    if not isinstance(effect, dict) or effect.get("kind") != "answer":
        return None
    fields = effect.get("fields")
    if not isinstance(fields, dict):
        return None
    committed_action_count = fields.get("committed_action_count")
    answer_state = fields.get("substantive_answer_state")
    if (
        type(committed_action_count) is not int
        or committed_action_count < 0
        or answer_state not in {"empty", "present"}
    ):
        return None
    return committed_action_count, answer_state == "present"


def _deterministic_material_assessment(
    finding_index: int,
    reason_code: MaterialVarianceReasonCode,
    *,
    evaluator_version_id: str,
) -> MaterialVarianceAssessment:
    finding_pointer = f"/payload/answer/findings/{finding_index}"
    return MaterialVarianceAssessment(
        decision="material_variance",
        reason_code=reason_code,
        explanation=_EXPLANATIONS["material_variance"],
        evidence=(
            MaterialVarianceEvidence(json_pointer=f"{finding_pointer}/baseline_effects/0"),
            MaterialVarianceEvidence(json_pointer=f"{finding_pointer}/variation_effects/0"),
        ),
        evaluator_version_id=evaluator_version_id,
    )


def _has_required_citations(decision: str, pointers: tuple[str, ...]) -> bool:
    if decision == "insufficient_evidence":
        return bool(pointers)
    return any("/baseline_effects" in pointer for pointer in pointers) and any(
        "/variation_effects" in pointer for pointer in pointers
    )
