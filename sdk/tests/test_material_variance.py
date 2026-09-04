from __future__ import annotations

from typing import cast

import pytest
from pydantic import JsonValue, ValidationError
from ul.dataset_evaluation import (
    DatasetEvaluationFinding,
    MaterialVarianceAssessment,
    MaterialVarianceEvidence,
)
from ul.evaluators import JudgeRequest
from ul.material_variance import DatasetMaterialVarianceJudge
from ul_core.dataset import ObservedOutcome
from ul_core.evaluators import EvaluatorDecision, EvaluatorEvidence, EvaluatorJudgeVersion

pytestmark = pytest.mark.asyncio


class RecordingJudge:
    version = EvaluatorJudgeVersion(
        prompt_version="a" * 64,
        model="test/materiality",
        configuration_sha256="b" * 64,
    )

    def __init__(self, decision: EvaluatorDecision | Exception) -> None:
        self.decision = decision
        self.requests: list[JudgeRequest] = []

    async def evaluate(self, request: JudgeRequest) -> EvaluatorDecision:
        self.requests.append(request)
        if isinstance(self.decision, Exception):
            raise self.decision
        return self.decision


def _decision(label: str, score: float) -> EvaluatorDecision:
    return EvaluatorDecision(
        score=score,
        label=label,
        explanation="model prose must not be persisted: private-sentinel",
        evidence=(
            EvaluatorEvidence(
                source="judge_payload",
                json_pointer="/payload/answer/findings/0/baseline_effects/0",
                description="baseline",
            ),
            EvaluatorEvidence(
                source="judge_payload",
                json_pointer="/payload/answer/findings/0/variation_effects/0",
                description="variation",
            ),
        ),
    )


def _finding(*, expected_amount: int = 100, observed_amount: int = 200) -> DatasetEvaluationFinding:
    return DatasetEvaluationFinding(
        category="changed_grounded_effect_argument",
        message="changed",
        expected_effects=(
            ObservedOutcome(
                id="private-baseline-id",
                kind="action",
                predicate="transfer",
                status="observed",
                confidence=1,
                position=0,
                fields={"amount": expected_amount, "server_id": "private-server-id"},
            ),
        ),
        observed_effects=(
            ObservedOutcome(
                id="private-variation-id",
                kind="action",
                predicate="transfer",
                status="observed",
                confidence=1,
                position=0,
                fields={"amount": observed_amount, "server_id": "other-private-id"},
            ),
        ),
        grounded_field_names=("amount",),
    )


def _response_finding(
    *,
    baseline_answer: JsonValue,
    baseline_actions: list[JsonValue],
    variation_answer: JsonValue,
    variation_actions: list[JsonValue],
) -> DatasetEvaluationFinding:
    def outcome(identifier: str, answer: JsonValue, actions: list[JsonValue]) -> ObservedOutcome:
        return ObservedOutcome(
            id=identifier,
            kind="answer",
            predicate="returned_response",
            status="observed",
            confidence=1,
            position=0,
            fields={"value": {"answer": answer, "actions": actions}},
        )

    return DatasetEvaluationFinding(
        category="changed_response",
        message="changed",
        expected_effects=(outcome("baseline-response", baseline_answer, baseline_actions),),
        observed_effects=(outcome("variation-response", variation_answer, variation_actions),),
    )


async def test_material_variance_judge_persists_closed_local_decision() -> None:
    judge = RecordingJudge(_decision("material_variance:grounded_argument_changed", 1))
    evaluator = DatasetMaterialVarianceJudge(judge, max_input_chars=50_000)

    assessment = await evaluator.evaluate("action", (_finding(),))

    assert assessment.decision == "material_variance"
    assert assessment.reason_code == "grounded_argument_changed"
    assert assessment.explanation == (
        "The variation changed the observed real-world action or outcome."
    )
    assert "private-sentinel" not in repr(assessment)
    assert evaluator.actual_calls == 1
    assert "material_variance:grounded_argument_changed" in judge.requests[0].allowed_labels
    assert judge.requests[0].label_scores["material_variance:grounded_argument_changed"] == 1
    answer = judge.requests[0].payload["answer"]
    assert answer == {
        "comparison_surface": "action",
        "findings": [
            {
                "category": "changed_grounded_effect_argument",
                "grounded_field_names": ["amount"],
                "baseline_effects": [
                    {"kind": "action", "predicate": "transfer", "fields": {"amount": 100}}
                ],
                "variation_effects": [
                    {"kind": "action", "predicate": "transfer", "fields": {"amount": 200}}
                ],
            }
        ],
    }


async def test_material_variance_judge_accepts_operational_equivalence() -> None:
    judge = RecordingJudge(_decision("operationally_equivalent:alias_or_representation", 0))

    assessment = await DatasetMaterialVarianceJudge(judge, max_input_chars=50_000).evaluate(
        "action", (_finding(),)
    )

    assert assessment.decision == "operationally_equivalent"
    assert assessment.reason_code == "alias_or_representation"


async def test_material_variance_judge_normalizes_response_envelopes() -> None:
    judge = RecordingJudge(_decision("material_variance:action_added", 1))
    finding = _response_finding(
        baseline_answer=[],
        baseline_actions=[{"action": "FHIR_GET", "url": "private-read-target"}],
        variation_answer=None,
        variation_actions=[
            {"action": "findPatient", "url": "private-read-target"},
            {"action": "FHIR_POST", "body": {"value": 118}},
        ],
    )

    assessment = await DatasetMaterialVarianceJudge(judge, max_input_chars=50_000).evaluate(
        "response", (finding,)
    )

    assert assessment.decision == "material_variance"
    assert assessment.reason_code == "action_added"
    assert tuple(evidence.json_pointer for evidence in assessment.evidence) == (
        "/payload/answer/findings/0/baseline_effects/0",
        "/payload/answer/findings/0/variation_effects/0",
    )
    assert not judge.requests


async def test_material_variance_judge_preserves_undeclared_actions_only_response() -> None:
    judge = RecordingJudge(_decision("operationally_equivalent:presentation_only", 0))

    def outcome(identifier: str, actions: list[JsonValue]) -> ObservedOutcome:
        return ObservedOutcome(
            id=identifier,
            kind="answer",
            predicate="returned_response",
            status="observed",
            confidence=1,
            position=0,
            fields={"value": {"actions": actions}},
        )

    finding = DatasetEvaluationFinding(
        category="changed_response",
        message="changed",
        expected_effects=(outcome("baseline", [{"name": "GET_APPROVAL"}]),),
        observed_effects=(outcome("variation", [{"name": "REQUEST_APPROVAL"}]),),
    )

    assessment = await DatasetMaterialVarianceJudge(judge, max_input_chars=50_000).evaluate(
        "response", (finding,)
    )

    assert assessment.decision == "operationally_equivalent"
    assert assessment.reason_code == "presentation_only"
    assert len(judge.requests) == 1
    baseline_fields = judge.requests[0].payload["answer"]["findings"][0]["baseline_effects"][0][
        "fields"
    ]
    assert baseline_fields == {"value": {"actions": [{"name": "GET_APPROVAL"}]}}


async def test_material_variance_judge_preserves_unverified_action_object_fields() -> None:
    judge = RecordingJudge(_decision("material_variance:response_meaning_changed", 1))

    def outcome(identifier: str, status: str) -> ObservedOutcome:
        return ObservedOutcome(
            id=identifier,
            kind="answer",
            predicate="returned_response",
            status="observed",
            confidence=1,
            position=0,
            fields={"value": {"actions": [], "status": status}},
        )

    finding = DatasetEvaluationFinding(
        category="changed_response",
        message="changed",
        expected_effects=(outcome("baseline", "approved"),),
        observed_effects=(outcome("variation", "rejected"),),
    )

    assessment = await DatasetMaterialVarianceJudge(judge, max_input_chars=50_000).evaluate(
        "response", (finding,)
    )

    assert assessment.decision == "material_variance"
    baseline_fields = judge.requests[0].payload["answer"]["findings"][0]["baseline_effects"][0][
        "fields"
    ]
    assert baseline_fields == {"value": {"actions": [], "status": "approved"}}


async def test_material_variance_judge_routes_unverified_same_count_action_changes() -> None:
    judge = RecordingJudge(_decision("operationally_equivalent:presentation_only", 0))
    finding = _response_finding(
        baseline_answer=None,
        baseline_actions=[
            {
                "operation": "add",
                "path": "/messages/id=generated-a",
                "value": {
                    "id": "generated-a",
                    "thread_id": "shared-thread",
                    "date": 100,
                    "to": ["billing@example.com"],
                    "subject": "Payment $300",
                    "body_plain": "Payment is processing urgently.",
                    "body_html": "<p>Payment is processing urgently.</p>",
                    "snippet": "Payment is processing urgently.",
                    "size_estimate": 1000,
                },
            }
        ],
        variation_answer=None,
        variation_actions=[
            {
                "operation": "add",
                "path": "/messages/id=generated-b",
                "value": {
                    "id": "generated-b",
                    "thread_id": "shared-thread",
                    "date": 100,
                    "to": ["billing@example.com"],
                    "subject": "Payment $300",
                    "body_plain": "Payment is processing urgently.",
                    "body_html": "<p>Payment is processing urgently.</p>",
                    "snippet": "Payment is processing urgently.",
                    "size_estimate": 1000,
                },
            }
        ],
    )

    assessment = await DatasetMaterialVarianceJudge(judge, max_input_chars=50_000).evaluate(
        "response", (finding,)
    )

    assert assessment.decision == "operationally_equivalent"
    assert len(judge.requests) == 1
    baseline_action = judge.requests[0].payload["answer"]["findings"][0]["baseline_effects"][0][
        "fields"
    ]["committed_actions"][0]
    assert baseline_action == {
        "operation": "add",
        "path": "/messages/id=generated-a",
        "value": {
            "id": "generated-a",
            "thread_id": "shared-thread",
            "date": 100,
            "to": ["billing@example.com"],
            "subject": "Payment $300",
            "body_plain": "Payment is processing urgently.",
            "body_html": "<p>Payment is processing urgently.</p>",
            "snippet": "Payment is processing urgently.",
            "size_estimate": 1000,
        },
    }


async def test_material_variance_safety_floor_catches_removed_substantive_response() -> None:
    judge = RecordingJudge(_decision("operationally_equivalent:same_real_world_effect", 0))
    finding = _response_finding(
        baseline_answer=[42],
        baseline_actions=[],
        variation_answer=[],
        variation_actions=[],
    )

    assessment = await DatasetMaterialVarianceJudge(judge, max_input_chars=50_000).evaluate(
        "response", (finding,)
    )

    assert assessment.decision == "material_variance"
    assert assessment.reason_code == "response_meaning_changed"
    assert not judge.requests


@pytest.mark.parametrize("baseline_answer", ["", "  \n", [], {}])
async def test_material_variance_safety_floor_canonicalizes_empty_answers(
    baseline_answer: JsonValue,
) -> None:
    judge = RecordingJudge(_decision("operationally_equivalent:presentation_only", 0))
    finding = _response_finding(
        baseline_answer=baseline_answer,
        baseline_actions=[],
        variation_answer=None,
        variation_actions=[],
    )

    assessment = await DatasetMaterialVarianceJudge(judge, max_input_chars=50_000).evaluate(
        "response", (finding,)
    )

    assert assessment.decision == "operationally_equivalent"
    assert assessment.reason_code == "same_real_world_effect"
    assert not judge.requests


async def test_material_variance_safety_floor_catches_removed_committed_action() -> None:
    judge = RecordingJudge(_decision("operationally_equivalent:same_real_world_effect", 0))
    finding = _response_finding(
        baseline_answer=[],
        baseline_actions=[{"action": "CREATE_RECORD", "value": 42}],
        variation_answer=[],
        variation_actions=[{"action": "FHIR_GET", "value": 42}],
    )

    assessment = await DatasetMaterialVarianceJudge(judge, max_input_chars=50_000).evaluate(
        "response", (finding,)
    )

    assert assessment.decision == "material_variance"
    assert assessment.reason_code == "action_removed"
    assert not judge.requests


async def test_material_variance_safety_floor_leaves_response_meaning_to_judge() -> None:
    judge = RecordingJudge(_decision("material_variance:response_meaning_changed", 1))
    finding = _response_finding(
        baseline_answer=["Monday"],
        baseline_actions=[],
        variation_answer=["Tuesday"],
        variation_actions=[],
    )

    assessment = await DatasetMaterialVarianceJudge(judge, max_input_chars=50_000).evaluate(
        "response", (finding,)
    )

    assert assessment.reason_code == "response_meaning_changed"
    assert len(judge.requests) == 1


async def test_material_variance_judge_keeps_unknown_actions_and_business_fields() -> None:
    judge = RecordingJudge(_decision("operationally_equivalent:same_real_world_effect", 0))
    outcome = ObservedOutcome(
        id="response",
        kind="answer",
        predicate="returned_response",
        status="observed",
        confidence=1,
        position=0,
        fields={
            "value": {
                "answer": ["A substantive answer."],
                "actions": [
                    {"action": "BUDGET_UPDATE"},
                    {"payload": 1},
                    "opaque-action",
                ],
                "business_result": {"approved": True},
                "status": "completed",
                "task_id": "private-task",
            }
        },
    )
    changed_outcome = outcome.model_copy(deep=True)
    changed_value = cast(dict[str, JsonValue], changed_outcome.fields["value"])
    changed_business_result = cast(dict[str, JsonValue], changed_value["business_result"])
    changed_business_result["approved"] = False
    finding = DatasetEvaluationFinding(
        category="changed_response",
        message="changed",
        expected_effects=(outcome,),
        observed_effects=(changed_outcome,),
    )

    await DatasetMaterialVarianceJudge(judge, max_input_chars=50_000).evaluate(
        "response", (finding,)
    )

    fields = judge.requests[0].payload["answer"]["findings"][0]["baseline_effects"][0]["fields"]
    assert fields == {
        "substantive_answer_state": "present",
        "substantive_answer": ["A substantive answer."],
        "committed_action_count": 3,
        "committed_actions": [
            {"action": "BUDGET_UPDATE"},
            {"payload": 1},
            "opaque-action",
        ],
        "other_fields": {"business_result": {"approved": True}},
    }
    assert "private-task" not in repr(fields)


async def test_material_variance_judge_fails_closed_on_invalid_or_failed_judgment() -> None:
    invalid_score = RecordingJudge(_decision("operationally_equivalent:same_real_world_effect", 1))
    failed = RecordingJudge(RuntimeError("private-provider-body"))

    invalid_assessment = await DatasetMaterialVarianceJudge(
        invalid_score, max_input_chars=50_000
    ).evaluate("action", (_finding(),))
    failed_assessment = await DatasetMaterialVarianceJudge(failed, max_input_chars=50_000).evaluate(
        "action", (_finding(),)
    )

    assert invalid_assessment.decision == "insufficient_evidence"
    assert invalid_assessment.reason_code == "judge_error"
    assert failed_assessment.decision == "insufficient_evidence"
    assert "private-provider-body" not in repr(failed_assessment)


async def test_material_variance_judge_skips_empty_or_oversized_payloads() -> None:
    judge = RecordingJudge(_decision("material_variance:response_meaning_changed", 1))
    evaluator = DatasetMaterialVarianceJudge(judge, max_input_chars=100_000)
    oversized = DatasetEvaluationFinding(
        category="changed_response",
        message="changed",
        expected_effects=(
            ObservedOutcome(
                id="answer-1",
                kind="answer",
                predicate="returned_response",
                status="observed",
                confidence=1,
                position=0,
                fields={"value": "a" * 70_000},
            ),
        ),
    )

    empty_assessment = await evaluator.evaluate("response", ())
    oversized_assessment = await evaluator.evaluate("response", (oversized,))

    assert empty_assessment.reason_code == "missing_comparison_evidence"
    assert oversized_assessment.reason_code == "missing_comparison_evidence"
    assert evaluator.actual_calls == 0
    assert not judge.requests

    configured_limit_judge = RecordingJudge(
        _decision("material_variance:grounded_argument_changed", 1)
    )
    configured_limit_assessment = await DatasetMaterialVarianceJudge(
        configured_limit_judge,
        max_input_chars=100,
    ).evaluate("action", (_finding(),))

    assert configured_limit_assessment.reason_code == "missing_comparison_evidence"
    assert not configured_limit_judge.requests


async def test_material_variance_assessment_rejects_incoherent_or_unsafe_evidence() -> None:
    with pytest.raises(ValidationError, match="reason must match"):
        MaterialVarianceAssessment(
            decision="operationally_equivalent",
            reason_code="action_added",
            explanation="invalid",
            evidence=(
                MaterialVarianceEvidence(
                    json_pointer="/payload/answer/findings/0/baseline_effects/0"
                ),
                MaterialVarianceEvidence(
                    json_pointer="/payload/answer/findings/0/variation_effects/0"
                ),
            ),
            evaluator_version_id=f"ulev_v1_{'a' * 64}",
        )
    with pytest.raises(ValidationError, match="safe finding pointer"):
        MaterialVarianceEvidence(json_pointer="/payload/answer/findings/0/private\nforged")
    with pytest.raises(ValidationError, match="requires cited comparison evidence"):
        MaterialVarianceAssessment(
            decision="material_variance",
            reason_code="action_added",
            explanation="invalid",
            evaluator_version_id=f"ulev_v1_{'a' * 64}",
        )
