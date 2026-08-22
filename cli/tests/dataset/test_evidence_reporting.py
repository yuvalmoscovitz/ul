from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from typer.testing import CliRunner
from ul import (
    AugmentationTarget,
    CaseFixtureReference,
    DatasetEvaluationResult,
    RichInteractionCase,
    project_rich_interaction_case,
)
from ul.dataset_invariants import (
    DatasetInvariantEvaluation,
)
from ul_cli.dataset.evidence import customer as customer_module
from ul_cli.dataset.presentation import evaluation as presentation_module
from ul_cli.main import app as root_app

from ._factories import (
    _evaluation_result,
    _invariant_evaluation,
    _isolated_response_target_config,
    _run_context,
    _trial_set,
)

runner = CliRunner()
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def test_rich_customer_evidence_records_source_target_original_and_lineage() -> None:
    source = project_rich_interaction_case(
        RichInteractionCase(
            id="cancel-order",
            inputs={"message": "Cancel order ord-9."},
            augmentation_targets=(
                AugmentationTarget(
                    id="message", kind="input_field", json_pointer="/inputs/message"
                ),
            ),
            fixture=CaseFixtureReference(id="orders", version="9"),
            observed_output={"status": "cancelled"},
        )
    )[0]
    candidate = SimpleNamespace(
        operator_id="input.surface.rephrase",
        operator_version="1.0.0",
        augmented_input="Please cancel order ord-9.",
        passed=True,
        failure_reasons=(),
    )
    case = SimpleNamespace(
        candidate=candidate,
        verdict="no_divergence",
        trial_set=_trial_set(requested_repetitions=1),
        findings=(),
        inconclusive_reasons=(),
    )
    result = cast(
        DatasetEvaluationResult,
        SimpleNamespace(
            source=source,
            baseline=SimpleNamespace(
                trial_set=_trial_set(requested_repetitions=1),
                inconclusive_reasons=(),
            ),
            cases=(case,),
            model_dump=lambda **kwargs: {"technical": "evidence"},
        ),
    )

    evidence = customer_module.build_customer_evidence_record(
        result,
        repetitions=1,
        max_environment_api_calls=2,
        planned_target_calls=2,
    )

    assert evidence["schema_version"] == "1.9.0"
    assert evidence["interaction_id"] == "cancel-order::message"
    assert evidence["source_record_id"] == "cancel-order"
    assert evidence["augmentation_target"] == {
        "id": "message",
        "kind": "input_field",
        "json_pointer": "/inputs/message",
        "turn_id": None,
    }
    assert evidence["cases"][0]["original_value"] == "Cancel order ord-9."


def test_unified_report_surfaces_response_only_scope_and_limitations(tmp_path: Path) -> None:
    result = _evaluation_result("interaction-1")
    run_context = _run_context((result.source,), target_config=_isolated_response_target_config())
    record = customer_module.build_customer_evidence_record(
        result,
        repetitions=1,
        max_environment_api_calls=2,
        planned_target_calls=2,
        run_context=cast(Any, run_context),
    )
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_text(json.dumps(record) + "\n", encoding="utf-8")

    json_report = runner.invoke(root_app, ["report", str(evidence), "--json"])
    human_report = runner.invoke(root_app, ["report", str(evidence)])

    assert json_report.exit_code == 0, json_report.output
    parsed_report = json.loads(json_report.output)
    assert parsed_report["evaluation_mode"] == "variance"
    assert parsed_report["evidence_scope"] == "response_only"
    assert parsed_report["capability_limitations"] == [
        "cleanup_verification",
        "conversation_replay",
        "state_observation",
    ]
    assert human_report.exit_code == 0, human_report.output
    assert "Evaluation mode: variance" in human_report.output
    assert "Evidence scope: response only" in human_report.output
    assert "Not verified: committed state, cleanup, or multi-turn conversations." in (
        human_report.output
    )


def test_customer_evidence_keeps_summary_and_nested_technical_details() -> None:
    expected_effect = SimpleNamespace(
        id="effect-1",
        evidence=(),
        confidence=0.9,
        status="completed",
        request_unit_ids=("request-1",),
        position=0,
        kind="action",
        predicate="transfer",
        fields={"amount": 100},
        propositions=(),
        model_dump=lambda **kwargs: {"kind": "action", "predicate": "transfer"},
    )
    finding = SimpleNamespace(
        category="duplicate_effect",
        message="A duplicate action needs review.",
        expected_effects=(expected_effect,),
        observed_effects=(expected_effect, expected_effect),
        grounded_field_names=("amount",),
    )
    candidate = SimpleNamespace(
        operator_id="input.surface.disfluency_repeat",
        operator_version="1.0.0",
        augmented_input="transfer transfer 100 to Alice",
        passed=True,
        failure_reasons=(),
    )
    case = SimpleNamespace(
        candidate=candidate,
        verdict="divergence_needs_review",
        trial_set=_trial_set(representative_effect=expected_effect),
        findings=(finding,),
        inconclusive_reasons=(),
    )
    result = cast(
        DatasetEvaluationResult,
        SimpleNamespace(
            source=SimpleNamespace(id="case-1", raw_input="transfer 100 to Alice"),
            baseline=SimpleNamespace(
                verdict="no_divergence",
                trial_set=_trial_set(representative_effect=expected_effect),
                inconclusive_reasons=(),
            ),
            cases=(case,),
            model_dump=lambda **kwargs: {"full": "technical evidence"},
        ),
    )

    evidence = customer_module.build_customer_evidence_record(
        result,
        repetitions=3,
        max_environment_api_calls=100,
        planned_target_calls=6,
    )

    assert presentation_module.result_needs_review(result) is True
    assert evidence["interaction_id"] == "case-1"
    assert evidence["original_input"] == "transfer 100 to Alice"
    assert evidence["schema_version"] == "1.8.0"
    assert evidence["evaluation_mode"] == "variance"
    assert evidence["invariant_evaluation"] is None
    assert evidence["current_baseline"]["status"] == "ORIGINAL REPLAY STABLE (3/3 OBSERVED)"
    assert "findings" not in evidence["current_baseline"]
    assert evidence["current_baseline"]["observations"]["outcome_group_count"] == 1
    assert evidence["current_baseline"]["observations"]["outcome_groups"][0]["repetitions"] == [
        1,
        2,
        3,
    ]
    assert evidence["current_baseline"]["observations"]["outcome_groups"][0]["count"] == 3
    assert evidence["current_baseline"]["observations"]["observed_repetitions"] == 3
    assert evidence["current_baseline"]["observations"]["inconclusive_repetitions"] == 0
    assert evidence["cases"][0]["status"] == "REPEATABLE DIFFERENCE — REVIEW"
    assert evidence["cases"][0]["findings"][0]["reference_effects"] == [
        {"kind": "action", "predicate": "transfer"}
    ]
    assert evidence["cases"][0]["findings"][0]["finding_id"] == (
        "ulf_v1_3ece170dbaff96e18428f477f3ea17e1e24f6e2d9cb0c222277699ce624d1b5e"
    )
    assert evidence["cases"][0]["findings"][0]["grounded_field_names"] == ["amount"]
    assert evidence["cases"][0]["findings"][0]["severity"] == "unrated"
    assert evidence["cases"][0]["findings"][0]["review_status"] == "needs_review"
    assert evidence["execution_plan"] == {
        "repetitions": 3,
        "max_target_calls": 100,
        "dataset_planned_target_calls": 6,
    }
    assert "does not determine" in evidence["limitations"]
    assert "caused" in evidence["limitations"]
    assert "production failure rate" in evidence["limitations"]
    assert (
        "Historical output is grounding evidence, not an expected answer" in evidence["limitations"]
    )
    assert evidence["technical_details"] == {"full": "technical evidence"}


def test_customer_evidence_keeps_invariants_separate_from_behavioral_findings() -> None:
    result = cast(
        DatasetEvaluationResult,
        SimpleNamespace(
            source=SimpleNamespace(id="case-1", raw_input="Correct amount to 100."),
            baseline=SimpleNamespace(
                trial_set=_trial_set(requested_repetitions=1),
                inconclusive_reasons=(),
            ),
            cases=(),
            model_dump=lambda **kwargs: {"technical": "behavioral evidence"},
        ),
    )
    invariant_evaluation = _invariant_evaluation("satisfied", "violated")

    evidence = customer_module.build_customer_evidence_record(
        result,
        repetitions=1,
        max_environment_api_calls=2,
        planned_target_calls=2,
        invariant_evaluation=invariant_evaluation,
    )

    assert evidence["schema_version"] == "1.8.0"
    assert evidence["evaluation_mode"] == "variance"
    assert evidence["cases"] == []
    stored_invariants = cast(dict[str, Any], evidence["invariant_evaluation"])
    assert stored_invariants["baseline"]["rules"][0]["status"] == "satisfied"
    assert stored_invariants["variations"][0]["rules"][0]["status"] == "violated"
    assert evidence["technical_details"] == {"technical": "behavioral evidence"}


@pytest.mark.parametrize(
    ("evaluations", "expected_exit_code"),
    [
        ((), 0),
        ((_invariant_evaluation("satisfied"),), 0),
        ((_invariant_evaluation("not_evaluable"),), 2),
        ((_invariant_evaluation("satisfied", "not_evaluable"),), 2),
        ((_invariant_evaluation("violated"),), 1),
        ((_invariant_evaluation("not_evaluable", "violated"),), 1),
    ],
)
def test_invariant_exit_code_precedence(
    evaluations: tuple[DatasetInvariantEvaluation, ...], expected_exit_code: int
) -> None:
    assert presentation_module.dataset_invariant_exit_code(evaluations) == expected_exit_code


def test_finding_id_ignores_volatile_evidence_and_semantic_ordering() -> None:
    first_effect = SimpleNamespace(
        id="generated-effect-1",
        evidence=(SimpleNamespace(json_pointer="/actions/0"),),
        confidence=0.72,
        status="failed",
        request_unit_ids=("generated-request-1",),
        position=0,
        kind="action",
        predicate="transfer",
        fields={"recipient": "Alice", "details": {"currency": "USD", "amount": 100}},
        propositions=("authorized", "settled"),
    )
    same_effect_with_volatile_changes = SimpleNamespace(
        id="generated-effect-99",
        evidence=(SimpleNamespace(json_pointer="/tool_calls/4"),),
        confidence=0.99,
        status="completed",
        request_unit_ids=("generated-request-42",),
        position=8,
        kind="action",
        predicate="transfer",
        fields={"details": {"amount": 100, "currency": "USD"}, "recipient": "Alice"},
        propositions=("settled", "authorized"),
    )
    second_effect = SimpleNamespace(
        id="generated-effect-2",
        evidence=(),
        confidence=0.8,
        status="completed",
        request_unit_ids=(),
        position=1,
        kind="action",
        predicate="notify",
        fields={"recipient": "Alice"},
        propositions=(),
    )
    reordered_second_effect = SimpleNamespace(**vars(second_effect))
    finding = cast(
        Any,
        SimpleNamespace(
            category="changed_grounded_effect_argument",
            grounded_field_names=("recipient", "amount"),
            expected_effects=(first_effect, second_effect),
            observed_effects=(second_effect, first_effect),
        ),
    )
    semantically_identical_finding = cast(
        Any,
        SimpleNamespace(
            category="changed_grounded_effect_argument",
            grounded_field_names=("amount", "recipient"),
            expected_effects=(reordered_second_effect, same_effect_with_volatile_changes),
            observed_effects=(same_effect_with_volatile_changes, reordered_second_effect),
        ),
    )

    finding_id = customer_module._finding_id(
        interaction_id="case-1",
        original_input="Transfer 100 to Alice.",
        operator_id="input.surface.rephrase",
        operator_version="1.0.0",
        augmented_input="Please transfer 100 to Alice.",
        finding=finding,
    )
    identical_finding_id = customer_module._finding_id(
        interaction_id="case-1",
        original_input="Transfer 100 to Alice.",
        operator_id="input.surface.rephrase",
        operator_version="1.0.0",
        augmented_input="Please transfer 100 to Alice.",
        finding=semantically_identical_finding,
    )

    assert finding_id == identical_finding_id
    assert re.fullmatch(r"ulf_v1_[0-9a-f]{64}", finding_id)


def test_finding_id_changes_for_meaningful_variation_or_behavior() -> None:
    reference_effect = SimpleNamespace(
        status="completed",
        kind="action",
        predicate="transfer",
        fields={"amount": 100, "recipient": "Alice"},
        propositions=(),
    )
    changed_effect = SimpleNamespace(
        status="completed",
        kind="action",
        predicate="transfer",
        fields={"amount": 200, "recipient": "Alice"},
        propositions=(),
    )
    finding = cast(
        Any,
        SimpleNamespace(
            category="changed_grounded_effect_argument",
            grounded_field_names=("amount",),
            expected_effects=(reference_effect,),
            observed_effects=(changed_effect,),
        ),
    )

    finding_id = customer_module._finding_id(
        interaction_id="case-1",
        original_input="Transfer 100 to Alice.",
        operator_id="input.surface.rephrase",
        operator_version="1.0.0",
        augmented_input="Please transfer 100 to Alice.",
        finding=finding,
    )
    changed_variation_id = customer_module._finding_id(
        interaction_id="case-1",
        original_input="Transfer 100 to Alice.",
        operator_id="input.surface.rephrase",
        operator_version="1.0.0",
        augmented_input="Could you transfer 100 to Alice?",
        finding=finding,
    )
    changed_behavior = cast(
        Any,
        SimpleNamespace(
            category="changed_grounded_effect_argument",
            grounded_field_names=("amount",),
            expected_effects=(reference_effect,),
            observed_effects=(
                SimpleNamespace(
                    status="completed",
                    kind="action",
                    predicate="transfer",
                    fields={"amount": 300, "recipient": "Alice"},
                    propositions=(),
                ),
            ),
        ),
    )
    changed_behavior_id = customer_module._finding_id(
        interaction_id="case-1",
        original_input="Transfer 100 to Alice.",
        operator_id="input.surface.rephrase",
        operator_version="1.0.0",
        augmented_input="Please transfer 100 to Alice.",
        finding=changed_behavior,
    )

    assert finding_id != changed_variation_id
    assert finding_id != changed_behavior_id


def test_duplicate_semantic_findings_get_stable_unique_reportable_ids(tmp_path: Path) -> None:
    def effect(identifier: str, amount: int, position: int) -> SimpleNamespace:
        payload = {
            "id": identifier,
            "evidence": [],
            "confidence": 0.9,
            "status": "observed",
            "request_unit_ids": [],
            "position": position,
            "kind": "action",
            "predicate": "transfer",
            "fields": {"amount": amount, "recipient": "Alice"},
            "propositions": [],
        }
        return SimpleNamespace(
            **{**payload, "propositions": ()},
            model_dump=lambda **kwargs: payload,
        )

    first_reference = effect("reference-1", 100, 0)
    second_reference = effect("reference-2", 100, 1)
    first_observed = effect("observed-1", 200, 0)
    second_observed = effect("observed-2", 200, 1)
    first_finding = SimpleNamespace(
        category="changed_grounded_effect_argument",
        message="The variation changed a grounded transfer amount.",
        expected_effects=(first_reference,),
        observed_effects=(first_observed,),
        grounded_field_names=("amount",),
    )
    second_finding = SimpleNamespace(
        category="changed_grounded_effect_argument",
        message="The variation changed a grounded transfer amount.",
        expected_effects=(second_reference,),
        observed_effects=(second_observed,),
        grounded_field_names=("amount",),
    )
    finding_context = {
        "interaction_id": "case-1",
        "original_input": "Transfer 100 to Alice.",
        "operator_id": "input.surface.rephrase",
        "operator_version": "1.0.0",
        "augmented_input": "Please transfer 100 to Alice.",
    }

    customer_findings = customer_module._customer_findings(
        cast(Any, (first_finding, second_finding)),
        **finding_context,
    )
    reordered_customer_findings = customer_module._customer_findings(
        cast(Any, (second_finding, first_finding)),
        **finding_context,
    )
    finding_ids = {cast(dict[str, Any], finding)["finding_id"] for finding in customer_findings}
    reordered_finding_ids = {
        cast(dict[str, Any], finding)["finding_id"] for finding in reordered_customer_findings
    }

    assert len(finding_ids) == 2
    assert finding_ids == reordered_finding_ids
    assert all(re.fullmatch(r"ulf_v1_[0-9a-f]{64}", finding_id) for finding_id in finding_ids)

    candidate = SimpleNamespace(
        operator_id="input.surface.rephrase",
        operator_version="1.0.0",
        augmented_input="Please transfer 100 to Alice.",
        passed=True,
        failure_reasons=(),
    )
    case = SimpleNamespace(
        candidate=candidate,
        verdict="divergence_needs_review",
        trial_set=_trial_set(representative_effect=first_observed),
        findings=(first_finding, second_finding),
        inconclusive_reasons=(),
    )
    result = cast(
        DatasetEvaluationResult,
        SimpleNamespace(
            source=SimpleNamespace(id="case-1", raw_input="Transfer 100 to Alice."),
            baseline=SimpleNamespace(
                verdict="no_divergence",
                trial_set=_trial_set(representative_effect=first_reference),
                inconclusive_reasons=(),
            ),
            cases=(case,),
            model_dump=lambda **kwargs: {
                "evaluation_mode": "variance",
                "fixture": "duplicate semantic findings",
            },
        ),
    )
    evidence = customer_module.build_customer_evidence_record(
        result,
        repetitions=3,
        max_environment_api_calls=6,
        planned_target_calls=6,
    )
    evidence_path = tmp_path / "evidence.jsonl"
    evidence_path.write_text(json.dumps(evidence) + "\n", encoding="utf-8")

    report = runner.invoke(root_app, ["dataset", "report", str(evidence_path)])

    assert report.exit_code == 0, report.output
    assert "Dataset finding report: 2 finding(s)" in report.output


def test_stored_output_drift_does_not_require_review_or_appear_in_original_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finding = SimpleNamespace(
        category="changed_grounded_effect_argument",
        message="The live control changed an action value.",
        expected_effects=(),
        observed_effects=(),
    )
    candidate = SimpleNamespace(
        operator_id="input.surface.rephrase",
        operator_version="1.0.0",
        augmented_input="Please transfer 100 to Alice.",
        passed=True,
        failure_reasons=(),
    )
    case = SimpleNamespace(
        candidate=candidate,
        verdict="no_divergence",
        trial_set=_trial_set(),
        findings=(),
        inconclusive_reasons=(),
    )
    result = cast(
        DatasetEvaluationResult,
        SimpleNamespace(
            source=SimpleNamespace(id="case-1", raw_input="transfer 100 to Alice"),
            baseline=SimpleNamespace(
                verdict="no_divergence",
                trial_set=_trial_set(),
                findings=(finding,),
                inconclusive_reasons=(),
            ),
            cases=(case,),
        ),
    )
    printed_rows: list[tuple[str, ...]] = []

    class CapturingTable:
        def add_column(self, *args: object, **kwargs: object) -> None:
            pass

        def add_row(self, *values: str) -> None:
            printed_rows.append(values)

    monkeypatch.setattr(presentation_module, "Table", lambda **kwargs: CapturingTable())
    monkeypatch.setattr(presentation_module.console, "print", lambda *args, **kwargs: None)

    presentation_module.print_dataset_results((result,), tmp_path / "evidence.jsonl")

    assert presentation_module.result_needs_review(result) is False
    assert printed_rows == [
        (
            "1",
            "original replay",
            "ORIGINAL REPLAY STABLE (3/3 OBSERVED)",
            "stable",
            "3 / 1",
            "—",
        ),
        (
            "2",
            "input.surface.rephrase",
            "NO OBSERVED DIFFERENCE",
            "stable",
            "3 / 1",
            "—",
        ),
    ]


def test_customer_statuses_distinguish_potential_repeatable_and_unstable_results() -> None:
    baseline = SimpleNamespace(
        verdict="no_divergence",
        trial_set=_trial_set(),
        inconclusive_reasons=(),
    )
    result = cast(DatasetEvaluationResult, SimpleNamespace(baseline=baseline))

    one_trial_difference = SimpleNamespace(
        verdict="divergence_needs_review",
        trial_set=_trial_set(requested_repetitions=1),
    )
    repeated_difference = SimpleNamespace(
        verdict="divergence_needs_review",
        trial_set=_trial_set(requested_repetitions=3),
    )
    unstable_variation = SimpleNamespace(
        verdict="divergence_needs_review",
        trial_set=_trial_set(
            stability="unstable",
            outcome_group_repetitions=((1, 2), (3,)),
        ),
    )

    assert customer_module.case_customer_status(result, one_trial_difference) == (
        "POTENTIAL DIFFERENCE — REVIEW"
    )
    assert customer_module.case_customer_status(result, repeated_difference) == (
        "REPEATABLE DIFFERENCE — REVIEW"
    )
    assert customer_module.case_customer_status(result, unstable_variation) == (
        "UNSTABLE VARIATION — REVIEW"
    )

    unstable_original_result = cast(
        DatasetEvaluationResult,
        SimpleNamespace(
            baseline=SimpleNamespace(
                verdict="inconclusive",
                trial_set=_trial_set(
                    stability="unstable",
                    outcome_group_repetitions=((1, 2), (3,)),
                ),
                inconclusive_reasons=("original repetitions produced multiple outcomes",),
            )
        ),
    )
    assert (
        customer_module.baseline_customer_status(unstable_original_result)
        == "UNSTABLE ORIGINAL — INCONCLUSIVE"
    )
    assert customer_module.case_customer_status(unstable_original_result, repeated_difference) == (
        "UNSTABLE ORIGINAL — INCONCLUSIVE"
    )

    assert customer_module.case_customer_status(unstable_original_result, unstable_variation) == (
        "UNSTABLE ORIGINAL AND VARIATION — INCONCLUSIVE"
    )
    incomplete_variation = SimpleNamespace(
        verdict="inconclusive",
        trial_set=_trial_set(
            stability="inconclusive",
            outcome_group_repetitions=((1, 2),),
        ),
    )
    assert customer_module.case_customer_status(unstable_original_result, incomplete_variation) == (
        "COULDN'T DETERMINE"
    )
