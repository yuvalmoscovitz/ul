from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner
from ul import (
    AugmentationTarget,
    CaseFixtureReference,
    DatasetEvaluationFinding,
    DatasetEvaluationOutcomeGroup,
    DatasetEvaluationResult,
    DatasetEvaluationTrialSet,
    RichInteractionCase,
    SemanticFrame,
    project_rich_interaction_case,
)
from ul.dataset_invariants import (
    DatasetInvariantEvaluation,
    DatasetInvariantSuite,
    JsonValuesEqualInvariant,
    evaluate_dataset_invariants,
)
from ul_cli import dataset_review
from ul_cli.dataset.evidence import customer as customer_module
from ul_cli.dataset.presentation import evaluation as presentation_module
from ul_cli.invariant_findings import reproduced_invariant_rule_pairs
from ul_cli.main import app as root_app
from ul_cli.pattern_identity import ensure_project_pattern_identity_key
from ul_core.dataset import ObservedOutcome
from ul_core.evaluation import (
    EnvironmentLifecycleEvidence,
    EnvironmentTurnEvidence,
    ExecutionEvidence,
    ProbeExecutionEvent,
)

from ._factories import (
    _evaluation_result,
    _invariant_evaluation,
    _isolated_response_target_config,
    _rich_evaluation_result,
    _run_context,
    _trial_set,
)

runner = CliRunner()
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _create_pattern_identity_key(tmp_path: Path) -> None:
    project_directory = tmp_path / ".ul"
    project_directory.mkdir(mode=0o700)
    ensure_project_pattern_identity_key(project_directory)


def test_rich_evidence_builds_parses_and_reports_end_to_end(tmp_path: Path) -> None:
    result = _rich_evaluation_result()
    run_context = _run_context((result.source,))
    record = customer_module.build_customer_evidence_record(
        result,
        repetitions=1,
        max_environment_api_calls=2,
        planned_target_calls=2,
        run_context=cast(Any, run_context),
    )
    evidence_path = tmp_path / "rich-evidence.jsonl"
    evidence_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    _create_pattern_identity_key(tmp_path)

    assert record["schema_version"] == "1.15.0"
    assert dataset_review.is_reportable_dataset_evidence(evidence_path) is True
    report = runner.invoke(root_app, ["report", str(evidence_path), "--json"])
    assert report.exit_code == 0, report.output
    parsed_report = json.loads(report.output)
    assert parsed_report["evidence_schema_versions"] == ["1.15.0"]
    assert parsed_report["evaluation_mode"] == "variance"


@pytest.mark.parametrize(
    ("decision", "expected_status", "expected_exit_code"),
    (
        ("material_variance", "action_required", 1),
        ("operationally_equivalent", "resolved", 0),
        ("insufficient_evidence", "inconclusive", 2),
    ),
)
def test_automatic_materiality_drives_unified_report_actionability(
    tmp_path: Path,
    decision: str,
    expected_status: str,
    expected_exit_code: int,
) -> None:
    result = _evaluation_result(
        f"automatic-{decision}",
        has_review_finding=True,
        material_variance_decision=decision,
    )
    record = customer_module.build_customer_evidence_record(
        result,
        repetitions=1,
        max_environment_api_calls=2,
        planned_target_calls=2,
        run_context=cast(Any, _run_context((result.source,))),
    )
    evidence_path = tmp_path / f"{decision}.jsonl"
    evidence_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    _create_pattern_identity_key(tmp_path)

    command_result = runner.invoke(root_app, ["report", str(evidence_path), "--json"])
    report = json.loads(command_result.output)

    assert command_result.exit_code == expected_exit_code
    assert report["review_status"] == expected_status
    assert report["exit_code"] == expected_exit_code


def test_dataset_report_explains_array_count_change_without_disclosing_values(
    tmp_path: Path,
) -> None:
    original_value = {
        "answer": "",
        "actions": [
            {"action": "POST remove-message-canary"},
            {"action": "DELETE remove-record-canary"},
            {"action": "PUT update-record-canary", "body": {"amount": 1}},
        ],
        "private-diagnostics-canary": list(range(100)),
    }
    variation_value = {
        "answer": "",
        "actions": [
            {"action": "GET private-read-only-operation-canary"},
            {"action": "PUT update-record-canary", "body": {"amount": 2}},
            {"action": "PATCH add-record-canary"},
        ],
        "private-diagnostics-canary": [],
    }

    def response_effect(identifier: str, value: dict[str, Any]) -> ObservedOutcome:
        return ObservedOutcome(
            id=identifier,
            confidence=1,
            status="observed",
            position=0,
            kind="answer",
            predicate="returned_response",
            fields={"value": cast(Any, value)},
        )

    original_effect = response_effect("original-response", original_value)
    variation_effect = response_effect("variation-response", variation_value)
    result = _evaluation_result("private-input-canary", has_review_finding=True)
    baseline_trial = result.baseline.trial_set.trials[0]
    variation_trial_set = result.cases[0].trial_set
    assert variation_trial_set is not None
    variation_trial = variation_trial_set.trials[0]
    baseline_frame = baseline_trial.observed_frame
    variation_frame = variation_trial.observed_frame
    assert baseline_frame is not None
    assert variation_frame is not None
    source_frame = result.augmentation.source_frames[0].model_copy(
        update={"outcomes": (original_effect,)}
    )
    baseline_frame = baseline_frame.model_copy(update={"outcomes": (original_effect,)})
    variation_frame = variation_frame.model_copy(update={"outcomes": (variation_effect,)})
    baseline_trial_set = result.baseline.trial_set.model_copy(
        update={
            "comparison_surface": "response",
            "trials": (baseline_trial.model_copy(update={"observed_frame": baseline_frame}),),
            "outcome_groups": (
                DatasetEvaluationOutcomeGroup(
                    repetitions=(1,), representative_effects=(original_effect,)
                ),
            ),
        }
    )
    variation_trial_set = variation_trial_set.model_copy(
        update={
            "comparison_surface": "response",
            "trials": (variation_trial.model_copy(update={"observed_frame": variation_frame}),),
            "outcome_groups": (
                DatasetEvaluationOutcomeGroup(
                    repetitions=(1,), representative_effects=(variation_effect,)
                ),
            ),
        }
    )
    finding = DatasetEvaluationFinding(
        category="changed_response",
        message=("The variation changed the observed response at /private-diagnostics-canary/0."),
        expected_effects=(original_effect,),
        observed_effects=(variation_effect,),
    )
    material_variance = result.cases[0].material_variance
    assert material_variance is not None
    result = result.model_copy(
        update={
            "comparison_surface": "response",
            "augmentation": result.augmentation.model_copy(
                update={"source_frames": (source_frame,)}
            ),
            "baseline": result.baseline.model_copy(update={"trial_set": baseline_trial_set}),
            "cases": (
                result.cases[0].model_copy(
                    update={
                        "trial_set": variation_trial_set,
                        "findings": (finding,),
                        "material_variance": material_variance.model_copy(
                            update={"reason_code": "action_count_changed"}
                        ),
                    }
                ),
            ),
        }
    )
    record = customer_module.build_customer_evidence_record(
        result,
        repetitions=1,
        max_environment_api_calls=2,
        planned_target_calls=2,
    )
    evidence_path = tmp_path / "structural-difference.jsonl"
    evidence_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    report = runner.invoke(root_app, ["dataset", "report", str(evidence_path)])

    assert report.exit_code == 0, report.output
    normalized_report = " ".join(_ANSI_ESCAPE_PATTERN.sub("", report.output).split())
    assert (
        "What changed: The agent made 3 committed actions for the original response and 2 after "
        "the test variation (1 fewer)."
    ) in normalized_report
    for private_canary in (
        "private-input-canary",
        "private-diagnostics-canary",
        "remove-message-canary",
        "remove-record-canary",
        "private-read-only-operation-canary",
        "update-record-canary",
        "add-record-canary",
    ):
        assert private_canary not in normalized_report

    finding_id = cast(dict[str, Any], cast(list[Any], record["cases"])[0])["findings"][0][
        "finding_id"
    ]
    private_report = runner.invoke(
        root_app,
        [
            "dataset",
            "report",
            str(evidence_path),
            "--finding",
            finding_id,
            "--show-sensitive-values",
        ],
    )

    assert private_report.exit_code == 0, private_report.output
    assert (
        "Changed committed action: PUT update-record-canary; /body/amount: 1 -> 2"
        in private_report.output
    )
    assert "Removed committed action: DELETE remove-record-canary" in private_report.output
    assert "Removed committed action: POST remove-message-canary" in private_report.output
    assert "Added committed action: PATCH add-record-canary" in private_report.output
    assert "private-read-only-operation-canary" not in private_report.output
    assert "private-diagnostics-canary" not in private_report.output


def _response_action_finding(
    original_actions: list[dict[str, Any]],
    variation_actions: list[dict[str, Any]],
) -> Any:
    def effect(identifier: str, actions: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "id": identifier,
            "evidence": [],
            "confidence": 1,
            "status": "observed",
            "request_unit_ids": [],
            "position": 0,
            "kind": "answer",
            "predicate": "returned_response",
            "fields": {"value": {"answer": "", "actions": actions}},
            "propositions": [],
        }

    return dataset_review._Finding.model_validate(
        {
            "finding_id": "ulf_v1_" + "0" * 64,
            "category": "changed_response",
            "grounded_field_names": [],
            "severity": "unrated",
            "review_status": "needs_review",
            "summary": "The response changed.",
            "reference_effects": [effect("original", original_actions)],
            "observed_effects": [effect("variation", variation_actions)],
        }
    )


@pytest.mark.parametrize(
    ("original_actions", "variation_actions", "expected_count_summary", "expected_diff"),
    (
        (
            [{"action": "POST remove-canary"}],
            [],
            "1 committed actions for the original response and 0 after the test variation",
            "Removed committed action: POST remove-canary",
        ),
        (
            [],
            [{"action": "POST add-canary"}],
            "0 committed actions for the original response and 1 after the test variation",
            "Added committed action: POST add-canary",
        ),
    ),
)
def test_response_action_summary_covers_complete_addition_and_removal(
    original_actions: list[dict[str, Any]],
    variation_actions: list[dict[str, Any]],
    expected_count_summary: str,
    expected_diff: str,
) -> None:
    finding = _response_action_finding(original_actions, variation_actions)

    assert expected_count_summary in dataset_review._response_action_count_summary(finding)
    assert expected_diff in dataset_review._sensitive_response_action_difference_lines(finding)


def test_response_action_diff_is_bounded_with_an_omission_summary() -> None:
    finding = _response_action_finding(
        [{"action": f"POST remove-{index}"} for index in range(51)],
        [],
    )

    lines = dataset_review._sensitive_response_action_difference_lines(finding)

    assert len(lines) == 21
    assert lines[-1] == (
        "Additional committed action differences omitted: 31; "
        "inspect the complete technical evidence."
    )


@pytest.mark.parametrize(
    ("decision", "expected_exit_code", "needs_review"),
    (
        ("material_variance", 1, True),
        ("operationally_equivalent", 0, False),
        ("insufficient_evidence", 2, False),
    ),
)
def test_automatic_materiality_drives_evaluation_exit_code(
    decision: str,
    expected_exit_code: int,
    needs_review: bool,
) -> None:
    result = _evaluation_result(
        f"automatic-exit-{decision}",
        has_review_finding=True,
        material_variance_decision=decision,
    )

    assert presentation_module.dataset_result_exit_code(result) == expected_exit_code
    assert presentation_module.result_needs_review(result) is needs_review


def test_dataset_report_leads_with_actionable_summary_and_hides_private_values(
    tmp_path: Path,
) -> None:
    clear_result = _evaluation_result("report-clear", has_review_finding=True)
    clear_result = clear_result.model_copy(
        update={
            "cases": (
                clear_result.cases[0].model_copy(
                    update={
                        "verdict": "no_divergence",
                        "findings": (),
                        "material_variance": None,
                    }
                ),
            )
        }
    )
    results = (
        _evaluation_result(
            "report-material",
            has_review_finding=True,
            material_variance_decision="material_variance",
        ),
        _evaluation_result(
            "report-equivalent",
            has_review_finding=True,
            material_variance_decision="operationally_equivalent",
        ),
        _evaluation_result(
            "report-inconclusive",
            has_review_finding=True,
            material_variance_decision="insufficient_evidence",
        ),
        clear_result,
    )
    records = [
        customer_module.build_customer_evidence_record(
            result,
            repetitions=1,
            max_environment_api_calls=2,
            planned_target_calls=2,
        )
        for result in results
    ]
    evidence_path = tmp_path / "management-report.jsonl"
    evidence_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    material_finding_id = cast(dict[str, Any], records[0]["cases"][0])["findings"][0]["finding_id"]
    equivalent_finding_id = cast(dict[str, Any], records[1]["cases"][0])["findings"][0][
        "finding_id"
    ]
    inconclusive_finding_id = cast(dict[str, Any], records[2]["cases"][0])["findings"][0][
        "finding_id"
    ]

    report = runner.invoke(root_app, ["dataset", "report", str(evidence_path)])

    assert report.exit_code == 0, report.output
    assert "Result: ACTION REQUIRED — 1 consequential behavior change found" in report.output
    assert "Semantic comparisons: total=4, completed=4, no_observed_difference=1" in report.output
    assert "Automatic decisions: consequential=1, equivalent=1, inconclusive=1" in report.output
    assert "Scope: variance; correctness and severity were not assessed" in report.output
    assert "Consequential behavior changes" in report.output
    assert "Inconclusive comparisons" in report.output
    assert "Resolved or equivalent differences" not in report.output
    assert material_finding_id in report.output
    assert inconclusive_finding_id in report.output
    assert equivalent_finding_id not in report.output
    assert "Transfer 100 to Alice." not in report.output
    assert "Please transfer 100 to Alice." not in report.output

    all_findings = runner.invoke(
        root_app,
        ["dataset", "report", str(evidence_path), "--all-findings"],
    )

    assert all_findings.exit_code == 0, all_findings.output
    assert "Resolved or equivalent differences" in all_findings.output
    assert equivalent_finding_id in all_findings.output
    assert "Transfer 100 to Alice." not in all_findings.output

    reviewed = runner.invoke(
        root_app,
        [
            "dataset",
            "review",
            str(evidence_path),
            material_finding_id,
            "--status",
            "expected",
            "--reviewer",
            "risk-team",
            "--reason",
            "Approved behavior for this workflow.",
        ],
    )

    assert reviewed.exit_code == 0, reviewed.output
    reviewed_report = runner.invoke(root_app, ["dataset", "report", str(evidence_path)])
    assert reviewed_report.exit_code == 0, reviewed_report.output
    assert "Result: INCONCLUSIVE — 1 item(s) need attention" in reviewed_report.output
    assert "ACTION REQUIRED" not in reviewed_report.output
    assert "Human review overrides: expected=1" in reviewed_report.output

    private_drilldown = runner.invoke(
        root_app,
        [
            "dataset",
            "report",
            str(evidence_path),
            "--finding",
            material_finding_id,
            "--show-sensitive-values",
        ],
    )

    assert private_drilldown.exit_code == 0, private_drilldown.output
    assert "WARNING: showing selected private values" in private_drilldown.output
    assert "Transfer 100 to Alice." in private_drilldown.output
    assert "Sensitive value disclosure:" in private_drilldown.output


def test_dataset_report_is_inconclusive_when_no_valid_variation_was_evaluated(
    tmp_path: Path,
) -> None:
    result = _evaluation_result("report-discarded")
    record = customer_module.build_customer_evidence_record(
        result,
        repetitions=1,
        max_environment_api_calls=2,
        planned_target_calls=1,
    )
    evidence_path = tmp_path / "discarded-report.jsonl"
    evidence_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    report = runner.invoke(root_app, ["dataset", "report", str(evidence_path)])

    assert report.exit_code == 0, report.output
    assert "Result: INCONCLUSIVE — no valid variations were evaluated" in report.output
    assert "Semantic comparisons: total=1, completed=0" in report.output
    assert "Result: CLEAR" not in report.output


def test_equivalent_semantic_difference_does_not_suppress_invariant_violation(
    tmp_path: Path,
) -> None:
    result = _evaluation_result(
        "equivalent-with-invariant",
        has_review_finding=True,
        material_variance_decision="operationally_equivalent",
    )
    baseline_trial = result.baseline.trial_set.trials[0]
    variation_trial = result.cases[0].trial_set.trials[0]
    assert baseline_trial.target_output is not None
    assert variation_trial.target_output is not None
    baseline_output = baseline_trial.target_output.model_copy(
        update={"raw_output": {"actual": 100, "expected": 100}}
    )
    variation_output = variation_trial.target_output.model_copy(
        update={"raw_output": {"actual": 200, "expected": 100}}
    )
    result = result.model_copy(
        update={
            "baseline": result.baseline.model_copy(
                update={
                    "trial_set": result.baseline.trial_set.model_copy(
                        update={
                            "trials": (
                                baseline_trial.model_copy(
                                    update={"target_output": baseline_output}
                                ),
                            )
                        }
                    )
                }
            ),
            "cases": (
                result.cases[0].model_copy(
                    update={
                        "trial_set": result.cases[0].trial_set.model_copy(
                            update={
                                "trials": (
                                    variation_trial.model_copy(
                                        update={"target_output": variation_output}
                                    ),
                                )
                            }
                        )
                    }
                ),
            ),
        }
    )
    invariant_suite = DatasetInvariantSuite(
        schema_version="1.0.0",
        observation_source="target_output",
        observation_authority="agent_response",
        rules=(
            JsonValuesEqualInvariant(
                type="json_values_equal",
                id="values-remain-equal",
                version="1.0.0",
                description="The committed values must remain equal.",
                severity="high",
                left_pointer="/actual",
                right_pointer="/expected",
            ),
        ),
    )
    invariant_evaluation = evaluate_dataset_invariants(result, invariant_suite)
    record = customer_module.build_customer_evidence_record(
        result,
        repetitions=1,
        max_environment_api_calls=2,
        planned_target_calls=2,
        invariant_evaluation=invariant_evaluation,
    )
    evidence_path = tmp_path / "equivalent-with-invariant.jsonl"
    evidence_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    indexed_findings = dataset_review._index_findings(dataset_review._load_evidence(evidence_path))
    semantic_finding_id = next(
        finding_id
        for finding_id, indexed in indexed_findings.items()
        if indexed.kind == "semantic_difference"
    )
    invariant_finding_id = next(
        finding_id
        for finding_id, indexed in indexed_findings.items()
        if indexed.kind == "customer_invariant_violation"
    )

    management_report = runner.invoke(root_app, ["dataset", "report", str(evidence_path)])

    assert management_report.exit_code == 0, management_report.output
    assert "Result: ACTION REQUIRED" in management_report.output
    assert "Customer invariant violations: total=1, require_attention=1" in (
        management_report.output
    )
    assert invariant_finding_id in management_report.output
    assert semantic_finding_id not in management_report.output

    _create_pattern_identity_key(tmp_path)
    unified_report = runner.invoke(root_app, ["report", str(evidence_path), "--json"])

    assert unified_report.exit_code == 1, unified_report.output
    finding_statuses = {
        finding["finding_id"]: finding["review_status"]
        for finding in json.loads(unified_report.output)["findings"]
    }
    assert finding_statuses == {
        semantic_finding_id: "expected",
        invariant_finding_id: "needs_review",
    }


def test_reproduced_invariant_surfaces_despite_full_response_instability(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _evaluation_result(
        "stable-rule-unstable-response",
        has_review_finding=True,
        material_variance_decision="operationally_equivalent",
    )
    baseline_trial = result.baseline.trial_set.trials[0]
    variation_trial_set = result.cases[0].trial_set
    assert variation_trial_set is not None
    variation_trial = variation_trial_set.trials[0]
    assert baseline_trial.target_output is not None
    assert variation_trial.target_output is not None

    baseline_trials = tuple(
        baseline_trial.model_copy(
            update={
                "repetition": repetition,
                "target_output": baseline_trial.target_output.model_copy(
                    update={
                        "raw_output": {
                            "actual": 100,
                            "expected": 100,
                            "message": f"original-{repetition}",
                        }
                    }
                ),
                "observed_frame": baseline_trial.observed_frame.model_copy(
                    update={
                        "interaction_id": (
                            f"stable-rule-unstable-response:current_baseline:round-{repetition}"
                        )
                    }
                ),
            }
        )
        for repetition in (1, 2)
    )
    variation_trials = tuple(
        variation_trial.model_copy(
            update={
                "repetition": repetition,
                "target_output": variation_trial.target_output.model_copy(
                    update={
                        "raw_output": {
                            "actual": 200,
                            "expected": 100,
                            "message": f"variation-{repetition}",
                        }
                    }
                ),
                "observed_frame": variation_trial.observed_frame.model_copy(
                    update={
                        "interaction_id": (
                            "stable-rule-unstable-response:input.surface.rephrase:"
                            f"round-{repetition}"
                        )
                    }
                ),
            }
        )
        for repetition in (1, 2)
    )
    result = result.model_copy(
        update={
            "baseline": result.baseline.model_copy(
                update={
                    "verdict": "inconclusive",
                    "trial_set": result.baseline.trial_set.model_copy(
                        update={
                            "requested_repetitions": 2,
                            "stability": "unstable",
                            "trials": baseline_trials,
                            "outcome_groups": tuple(
                                DatasetEvaluationOutcomeGroup(
                                    repetitions=(repetition,), representative_effects=()
                                )
                                for repetition in (1, 2)
                            ),
                        }
                    ),
                }
            ),
            "cases": (
                result.cases[0].model_copy(
                    update={
                        "verdict": "divergence_needs_review",
                        "findings": (),
                        "material_variance": None,
                        "trial_set": variation_trial_set.model_copy(
                            update={
                                "requested_repetitions": 2,
                                "stability": "unstable",
                                "trials": variation_trials,
                                "outcome_groups": tuple(
                                    DatasetEvaluationOutcomeGroup(
                                        repetitions=(repetition,), representative_effects=()
                                    )
                                    for repetition in (1, 2)
                                ),
                            }
                        ),
                    }
                ),
            ),
        }
    )
    invariant_suite = DatasetInvariantSuite(
        schema_version="1.0.0",
        observation_source="target_output",
        observation_authority="agent_response",
        rules=(
            JsonValuesEqualInvariant(
                type="json_values_equal",
                id="values-remain-equal",
                version="1.0.0",
                description="The committed values must remain equal.",
                severity="high",
                left_pointer="/actual",
                right_pointer="/expected",
            ),
        ),
    )
    invariant_evaluation = evaluate_dataset_invariants(result, invariant_suite)
    assert tuple(trial.status for trial in invariant_evaluation.baseline.rules[0].trials) == (
        "satisfied",
        "satisfied",
    )
    assert tuple(trial.status for trial in invariant_evaluation.variations[0].rules[0].trials) == (
        "violated",
        "violated",
    )
    assert reproduced_invariant_rule_pairs(
        invariant_evaluation, result.cases[0].candidate.operator_id
    )
    record = customer_module.build_customer_evidence_record(
        result,
        repetitions=2,
        max_environment_api_calls=4,
        planned_target_calls=4,
        invariant_evaluation=invariant_evaluation,
    )
    dataset_review._EvidenceRecord.model_validate_json(json.dumps(record))
    evidence_path = tmp_path / "stable-rule-unstable-response.jsonl"
    evidence_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    _create_pattern_identity_key(tmp_path)

    monkeypatch.setattr(presentation_module.console, "width", 240)
    presentation_module.print_dataset_results(
        (result,),
        evidence_path,
        invariant_evaluations=(invariant_evaluation,),
        show_report_guidance=False,
    )
    terminal_output = " ".join(capsys.readouterr().out.split())
    management_report = runner.invoke(root_app, ["dataset", "report", str(evidence_path)])
    unified_report = runner.invoke(root_app, ["report", str(evidence_path), "--json"])
    management_output = " ".join(management_report.output.split())

    assert "reproduced invariant: values-remain-equal" in terminal_output
    assert "Reproduced invariant finding:" in terminal_output
    assert "original satisfied=2/2; variation violated=2/2" in terminal_output
    assert "whole-task correctness not established" in terminal_output
    assert management_report.exit_code == 0, management_report.output
    assert "Invariant repetitions: original satisfied=2/2; variation violated=2/2" in (
        management_output
    )
    assert "Full-response stability: original=unstable" in management_output
    assert "whole-task correctness not established" in management_output
    assert unified_report.exit_code == 1, unified_report.output
    invariant_summary = next(
        finding
        for finding in json.loads(unified_report.output)["findings"]
        if finding["kind"] == "customer_invariant_violation" and finding["finding_id"] is not None
    )
    assert invariant_summary["requested_repetitions"] == 2
    assert invariant_summary["violated_repetitions"] == 2
    assert invariant_summary["stability"] == "stable"


def test_dataset_report_treats_unstable_behavior_as_actionable(tmp_path: Path) -> None:
    result = _evaluation_result("unstable-report", has_review_finding=True)
    baseline_trial = result.baseline.trial_set.trials[0]
    stable_baseline_trial_set = DatasetEvaluationTrialSet(
        requested_repetitions=2,
        stability="stable",
        trials=(
            baseline_trial,
            baseline_trial.model_copy(
                update={
                    "repetition": 2,
                    "observed_frame": baseline_trial.observed_frame.model_copy(
                        update={"interaction_id": "unstable-report:current_baseline:round-2"}
                    ),
                }
            ),
        ),
        outcome_groups=(
            DatasetEvaluationOutcomeGroup(repetitions=(1, 2), representative_effects=()),
        ),
    )
    stable_trial = result.cases[0].trial_set.trials[0]
    unstable_trial_set = DatasetEvaluationTrialSet(
        requested_repetitions=2,
        stability="unstable",
        trials=(
            stable_trial,
            stable_trial.model_copy(
                update={
                    "repetition": 2,
                    "observed_frame": stable_trial.observed_frame.model_copy(
                        update={
                            "interaction_id": ("unstable-report:input.surface.rephrase:round-2")
                        }
                    ),
                }
            ),
        ),
        outcome_groups=(
            DatasetEvaluationOutcomeGroup(repetitions=(1,), representative_effects=()),
            DatasetEvaluationOutcomeGroup(repetitions=(2,), representative_effects=()),
        ),
    )
    unstable_case = result.cases[0].model_copy(
        update={
            "trial_set": unstable_trial_set,
            "findings": (),
            "material_variance": None,
        }
    )
    result = result.model_copy(
        update={
            "baseline": result.baseline.model_copy(update={"trial_set": stable_baseline_trial_set}),
            "cases": (unstable_case,),
        }
    )
    result = DatasetEvaluationResult.model_validate(result.model_dump())
    record = customer_module.build_customer_evidence_record(
        result,
        repetitions=2,
        max_environment_api_calls=2,
        planned_target_calls=4,
    )
    evidence_path = tmp_path / "unstable.jsonl"
    evidence_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    report = runner.invoke(root_app, ["dataset", "report", str(evidence_path)])

    assert report.exit_code == 0, report.output
    assert "Result: ACTION REQUIRED" in report.output
    assert "Other: unstable=1" in report.output
    assert "Behavior changed inconsistently across repeated trials." in report.output


def test_dataset_report_lists_inconclusive_case_without_finding(tmp_path: Path) -> None:
    result = _evaluation_result("technical-inconclusive", has_review_finding=True)
    trial = (
        result.cases[0]
        .trial_set.trials[0]
        .model_copy(
            update={"observed_frame": None, "inconclusive_reasons": ("target_execution_failed",)}
        )
    )
    inconclusive_trial_set = DatasetEvaluationTrialSet(
        requested_repetitions=1,
        stability="inconclusive",
        trials=(trial,),
    )
    inconclusive_case = result.cases[0].model_copy(
        update={
            "verdict": "inconclusive",
            "trial_set": inconclusive_trial_set,
            "findings": (),
            "material_variance": None,
            "inconclusive_reasons": ("target_execution_failed",),
        }
    )
    result = result.model_copy(update={"cases": (inconclusive_case,)})
    record = customer_module.build_customer_evidence_record(
        result,
        repetitions=1,
        max_environment_api_calls=2,
        planned_target_calls=2,
    )
    evidence_path = tmp_path / "inconclusive.jsonl"
    evidence_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    report = runner.invoke(root_app, ["dataset", "report", str(evidence_path)])

    assert report.exit_code == 0, report.output
    assert "Result: INCONCLUSIVE" in report.output
    assert "Comparison could not be classified." in report.output
    assert "Case: technical-inconclusive" in report.output
    assert "Reasons: target_execution_failed" in report.output
    assert "Transfer 100 to Alice." not in report.output


def test_pre_response_schema_action_and_answer_evidence_still_reports(tmp_path: Path) -> None:
    result = _evaluation_result("legacy-mixed-outcomes")
    action = ObservedOutcome(
        id="transfer",
        confidence=1,
        status="observed",
        position=0,
        kind="action",
        predicate="transfer",
        fields={"amount": 100},
    )
    answer = ObservedOutcome(
        id="confirmation",
        confidence=1,
        status="observed",
        position=1,
        kind="answer",
        predicate="confirmation",
        fields={"text": "private legacy answer"},
    )
    historical_frame = SemanticFrame(
        interaction_id=result.source.id,
        outcomes=(action, answer),
        extractor_version="legacy-test",
    )
    baseline_trial = result.baseline.trial_set.trials[0]
    baseline_frame = historical_frame.model_copy(
        update={"interaction_id": f"{result.source.id}:current_baseline:round-1"}
    )
    baseline_trial_set = result.baseline.trial_set.model_copy(
        update={
            "trials": (baseline_trial.model_copy(update={"observed_frame": baseline_frame}),),
            "outcome_groups": (
                DatasetEvaluationOutcomeGroup(repetitions=(1,), representative_effects=(action,)),
            ),
        }
    )
    result = result.model_copy(
        update={
            "augmentation": result.augmentation.model_copy(
                update={"source_frames": (historical_frame,)}
            ),
            "baseline": result.baseline.model_copy(update={"trial_set": baseline_trial_set}),
        }
    )
    record = customer_module.build_customer_evidence_record(
        result,
        repetitions=1,
        max_environment_api_calls=1,
        planned_target_calls=1,
    )
    record["schema_version"] = "1.13.0"
    record["cases"][0].pop("material_variance")
    evidence_path = tmp_path / "legacy-mixed.jsonl"
    evidence_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    _create_pattern_identity_key(tmp_path)

    report = runner.invoke(root_app, ["report", str(evidence_path), "--json"])

    assert report.exit_code == 0, report.output
    assert json.loads(report.output)["evidence_schema_versions"] == ["1.13.0"]
    assert "private legacy answer" not in report.output


def test_response_evidence_cannot_be_downgraded_to_pre_response_schema() -> None:
    result = _evaluation_result("response-downgrade", has_review_finding=True)
    record = customer_module.build_customer_evidence_record(
        result,
        repetitions=1,
        max_environment_api_calls=2,
        planned_target_calls=2,
    )
    record["schema_version"] = "1.13.0"
    record["cases"][0]["findings"][0]["category"] = "changed_response"
    record["technical_details"]["comparison_surface"] = "response"
    with pytest.raises(
        ValueError,
        match=r"response comparison evidence requires schema 1\.14\.0",
    ):
        dataset_review._EvidenceRecord.model_validate_json(json.dumps(record))


def test_materiality_version_must_match_run_context() -> None:
    result = _evaluation_result(
        "materiality-version-tamper",
        has_review_finding=True,
        material_variance_decision="operationally_equivalent",
    )
    record = customer_module.build_customer_evidence_record(
        result,
        repetitions=1,
        max_environment_api_calls=2,
        planned_target_calls=2,
        run_context=cast(Any, _run_context((result.source,))),
    )
    mismatched_version = "ulev_v1_" + "f" * 64
    record["cases"][0]["material_variance"]["evaluator_version_id"] = mismatched_version
    record["technical_details"]["cases"][0]["material_variance"]["evaluator_version_id"] = (
        mismatched_version
    )

    with pytest.raises(ValidationError, match="must match the evidence run context"):
        dataset_review._EvidenceRecord.model_validate_json(json.dumps(record))


def test_cross_examination_json_and_offline_cli_present_the_same_safe_facts(
    tmp_path: Path,
) -> None:
    result = _evaluation_result("private-case-canary", has_review_finding=True)
    record = customer_module.build_customer_evidence_record(
        result,
        repetitions=1,
        max_environment_api_calls=2,
        planned_target_calls=2,
    )
    evidence_path = tmp_path / "cross-examination.jsonl"
    evidence_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    _create_pattern_identity_key(tmp_path)

    json_report = runner.invoke(root_app, ["report", str(evidence_path), "--json"])
    human_report = runner.invoke(root_app, ["report", str(evidence_path)])

    assert json_report.exit_code == 1, json_report.output
    assert human_report.exit_code == 1, human_report.output
    cross_examination = json.loads(json_report.output)["findings"][0]["cross_examination"]
    assert cross_examination["historical_reference_available"] is True
    assert cross_examination["baseline_drift"] == "not_observed"
    assert cross_examination["augmentation_sensitivity"] == "observed"
    assert cross_examination["intrinsic_instability"] == "not_observed"
    assert cross_examination["response_evidence"]["conclusion"] == "observed"
    assert cross_examination["trajectory_evidence"]["conclusion"] == "unavailable"
    assert cross_examination["committed_state_evidence"]["conclusion"] == "unavailable"
    assert "Baseline drift: not observed (descriptive divergence; not an agent failure)" in (
        human_report.output
    )
    assert "Augmentation sensitivity: observed" in human_report.output
    assert "Historical output: reference evidence only; not a correctness oracle" in (
        human_report.output
    )
    assert "Response evidence: observed" in human_report.output
    assert "Trajectory evidence: unavailable" in human_report.output
    assert "Committed-state verification: unavailable" in human_report.output
    assert "Transfer 100 to Alice" not in human_report.output
    assert "private-case-canary" not in human_report.output


def test_response_evidence_remains_observed_when_semantic_evaluation_is_inconclusive() -> None:
    result = _evaluation_result("semantic-inconclusive", has_review_finding=True)

    def semantic_failure(trial_set: DatasetEvaluationTrialSet) -> DatasetEvaluationTrialSet:
        return trial_set.model_copy(
            update={
                "trials": tuple(
                    trial.model_copy(update={"inconclusive_reasons": ("semantic_parse_failed",)})
                    for trial in trial_set.trials
                )
            }
        )

    variation_trial_set = result.cases[0].trial_set
    assert variation_trial_set is not None
    result = result.model_copy(
        update={
            "baseline": result.baseline.model_copy(
                update={"trial_set": semantic_failure(result.baseline.trial_set)}
            ),
            "cases": (
                result.cases[0].model_copy(
                    update={"trial_set": semantic_failure(variation_trial_set)}
                ),
            ),
        }
    )

    record = customer_module.build_customer_evidence_record(
        result,
        repetitions=1,
        max_environment_api_calls=2,
        planned_target_calls=2,
    )

    response_evidence = record["cases"][0]["cross_examination"]["response_evidence"]
    assert response_evidence["conclusion"] == "observed"


def test_failed_lifecycle_response_and_execution_events_reach_human_and_json_reports(
    tmp_path: Path,
) -> None:
    result = _evaluation_result("failed-lifecycle-events", has_review_finding=True)

    def with_failed_event_evidence(
        trial_set: DatasetEvaluationTrialSet,
    ) -> DatasetEvaluationTrialSet:
        return trial_set.model_copy(
            update={
                "trials": tuple(
                    trial.model_copy(
                        update={
                            "execution_evidence": ExecutionEvidence(
                                evidence_scope="response_only",
                                case_id=f"case-{trial.repetition}",
                                environment_id="response-agent",
                                environment_config_sha256="a" * 64,
                                turns=(
                                    EnvironmentTurnEvidence(
                                        turn_id="turn-1",
                                        response={"status": "captured-before-failure"},
                                    ),
                                ),
                                final_response={"status": "captured-before-failure"},
                                execution_events=(
                                    ProbeExecutionEvent(
                                        id=f"event-{trial.repetition}",
                                        correlation_id=f"correlation-{trial.repetition}",
                                        kind="tool_call",
                                        payload={"tool": "bounded-tool"},
                                    ),
                                ),
                                lifecycle=EnvironmentLifecycleEvidence(
                                    terminal_status="failed",
                                    failed_phase="execute_turn",
                                    failure_code="transport_failed",
                                    failure_reason="connection closed after response capture",
                                    delivery="certain",
                                    cleanup="not_attempted",
                                    environment_state_uncertain=False,
                                ),
                            )
                        }
                    )
                    for trial in trial_set.trials
                )
            }
        )

    variation_trial_set = result.cases[0].trial_set
    assert variation_trial_set is not None
    result = result.model_copy(
        update={
            "baseline": result.baseline.model_copy(
                update={"trial_set": with_failed_event_evidence(result.baseline.trial_set)}
            ),
            "cases": (
                result.cases[0].model_copy(
                    update={"trial_set": with_failed_event_evidence(variation_trial_set)}
                ),
            ),
        }
    )
    record = customer_module.build_customer_evidence_record(
        result,
        repetitions=1,
        max_environment_api_calls=2,
        planned_target_calls=2,
    )
    evidence_path = tmp_path / "failed-events.jsonl"
    evidence_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    _create_pattern_identity_key(tmp_path)

    json_report = runner.invoke(root_app, ["report", str(evidence_path), "--json"])
    human_report = runner.invoke(root_app, ["report", str(evidence_path)])

    assert json_report.exit_code == 1, json_report.output
    assert human_report.exit_code == 1, human_report.output
    cross_examination = json.loads(json_report.output)["findings"][0]["cross_examination"]
    assert cross_examination["response_evidence"]["conclusion"] == "observed"
    assert cross_examination["trajectory_evidence"]["conclusion"] == "observed"
    assert "Response evidence: observed" in human_report.output
    assert "Trajectory evidence: observed" in human_report.output
    assert "authorities=invoker self reported" in human_report.output


def test_customer_evidence_rejects_cross_examination_claims_not_in_technical_details(
    tmp_path: Path,
) -> None:
    result = _evaluation_result("forged-trajectory", has_review_finding=True)
    record = customer_module.build_customer_evidence_record(
        result,
        repetitions=1,
        max_environment_api_calls=2,
        planned_target_calls=2,
    )
    trajectory = record["cases"][0]["cross_examination"]["trajectory_evidence"]
    trajectory.update(
        {
            "conclusion": "observed",
            "current_baseline": "observed",
            "variation": "observed",
            "current_baseline_covered_repetitions": 1,
            "variation_covered_repetitions": 1,
            "current_baseline_authorities": ["independent_observer"],
            "variation_authorities": ["independent_observer"],
        }
    )
    evidence_path = tmp_path / "forged.jsonl"
    evidence_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    report = runner.invoke(root_app, ["report", str(evidence_path), "--json"])

    assert report.exit_code == 2
    assert "unsupported evidence" in report.output


@pytest.mark.parametrize(
    ("stability", "expected_sensitivity", "expected_instability"),
    (
        ("stable", "observed", "not_observed"),
        ("unstable", "inconclusive", "observed"),
        ("inconclusive", "inconclusive", "inconclusive"),
    ),
)
def test_cross_examination_classifies_stability_without_correctness_claims(
    stability: str,
    expected_sensitivity: str,
    expected_instability: str,
) -> None:
    finding = SimpleNamespace()
    result = cast(
        DatasetEvaluationResult,
        SimpleNamespace(
            source=SimpleNamespace(raw_input="private-input"),
            baseline=SimpleNamespace(trial_set=_trial_set(stability=stability)),
            cases=(),
        ),
    )
    case = cast(
        Any,
        SimpleNamespace(
            trial_set=_trial_set(stability=stability),
            findings=(finding,),
        ),
    )

    cross_examination = customer_module._customer_cross_examination(result, case)

    assert cross_examination["augmentation_sensitivity"] == expected_sensitivity
    assert cross_examination["intrinsic_instability"] == expected_instability
    assert cross_examination["limitations"] == [
        "causality_not_established",
        "correctness_not_verified",
        "historical_reference_not_an_oracle",
    ]


def test_missing_variation_does_not_erase_a_conclusive_baseline_comparison() -> None:
    result = _evaluation_result("rejected-variation")

    record = customer_module.build_customer_evidence_record(
        result,
        repetitions=1,
        max_environment_api_calls=1,
        planned_target_calls=1,
    )

    cross_examination = record["cases"][0]["cross_examination"]
    assert cross_examination["baseline_drift"] == "not_observed"
    assert cross_examination["augmentation_sensitivity"] == "inconclusive"


def test_customer_pattern_facets_are_copied_only_from_the_reserved_metadata_field() -> None:
    result = _evaluation_result("interaction-1")
    result = result.model_copy(
        update={
            "source": result.source.model_copy(
                update={
                    "metadata": {
                        "unrelated": {"domain": "ignored"},
                        "ul_pattern_facets": {
                            "domain": "payments",
                            "workflow": "invoice-payment",
                            "role": "approver",
                            "use_case": "pay-approved-invoice",
                        },
                    }
                }
            )
        }
    )

    record = customer_module.build_customer_evidence_record(
        result,
        repetitions=1,
        max_environment_api_calls=2,
        planned_target_calls=2,
    )

    assert record["pattern_facets"] == {
        "domain": "payments",
        "workflow": "invoice-payment",
        "role": "approver",
        "use_case": "pay-approved-invoice",
    }


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

    assert evidence["schema_version"] == "1.15.0"
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
    _create_pattern_identity_key(tmp_path)

    json_report = runner.invoke(root_app, ["report", str(evidence), "--json"])
    human_report = runner.invoke(root_app, ["report", str(evidence)])

    assert json_report.exit_code == 0, json_report.output
    parsed_report = json.loads(json_report.output)
    assert parsed_report["evaluation_mode"] == "variance"
    assert parsed_report["response_state_evidence_scope"] == "response_only"
    assert parsed_report["capability_limitations"] == [
        "cleanup_verification",
        "conversation_replay",
        "state_observation",
    ]
    assert human_report.exit_code == 0, human_report.output
    assert "Evaluation mode: variance" in human_report.output
    assert "Response/state evidence scope: response only" in human_report.output
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
    assert evidence["schema_version"] == "1.14.0"
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

    assert evidence["schema_version"] == "1.15.0"
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


def test_response_finding_id_excludes_low_entropy_private_response_values() -> None:
    def response_finding(reference: str, observed: str) -> Any:
        return SimpleNamespace(
            category="changed_response",
            grounded_field_names=(),
            expected_effects=(
                SimpleNamespace(
                    kind="answer",
                    predicate="returned_response",
                    fields={"value": reference},
                    propositions=(),
                ),
            ),
            observed_effects=(
                SimpleNamespace(
                    kind="answer",
                    predicate="returned_response",
                    fields={"value": observed},
                    propositions=(),
                ),
            ),
        )

    identifiers = {
        customer_module._finding_id(
            interaction_id="case-1",
            original_input="Should I proceed?",
            operator_id="input.surface.rephrase",
            operator_version="1.0.0",
            augmented_input="Would you proceed?",
            finding=cast(Any, response_finding(reference, observed)),
        )
        for reference, observed in (("yes", "no"), ("no", "yes"), ("allow", "deny"))
    }

    assert len(identifiers) == 1


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

    result = _evaluation_result("case-1", has_review_finding=True)
    typed_findings = tuple(
        DatasetEvaluationFinding.model_validate(
            {
                "category": finding.category,
                "message": finding.message,
                "expected_effects": [effect.model_dump() for effect in finding.expected_effects],
                "observed_effects": [effect.model_dump() for effect in finding.observed_effects],
                "grounded_field_names": finding.grounded_field_names,
            },
            strict=False,
        )
        for finding in (first_finding, second_finding)
    )
    result = result.model_copy(
        update={
            "cases": (result.cases[0].model_copy(update={"findings": typed_findings}),),
        }
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
    assert "Result: ACTION REQUIRED — 1 consequential behavior change found" in report.output
    assert "Consequential behavior changes" in report.output


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
