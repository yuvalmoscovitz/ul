from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue
from rich.console import Console
from typer.testing import CliRunner
from ul import (
    DatasetAugmentationResult,
    DatasetEvaluationBaseline,
    DatasetEvaluationCase,
    DatasetEvaluationOutcomeGroup,
    DatasetEvaluationResult,
    DatasetEvaluationTrial,
    DatasetEvaluationTrialSet,
    InteractionRecord,
    ObservedAgentOutput,
    SemanticFrame,
)
from ul.dataset_augmentation import DatasetAugmentationCandidate
from ul.dataset_invariants import DatasetInvariantSuite, JsonValuesEqualInvariant
from ul_cli import dataset_review
from ul_cli import report as report_module
from ul_cli.dataset.evaluation import command as dataset_command
from ul_cli.dataset.evaluation import runner as dataset_runner
from ul_cli.main import app
from ul_cli.pattern_identity import pattern_mechanism_pseudonym
from ul_cli.report_contract import (
    FindingSummary,
    UnifiedReport,
)

runner = CliRunner()
FINDING_ID = f"ulf_v1_{'a' * 64}"
_PATTERN_IDENTITY_KEY = bytes(range(32))
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _effect(invoice_reference: str) -> dict[str, Any]:
    return {
        "id": f"payment-{invoice_reference}",
        "evidence": [],
        "confidence": 1.0,
        "status": "completed",
        "request_unit_ids": [],
        "position": 0,
        "kind": "action",
        "predicate": "payment_committed",
        "fields": {
            "invoice_reference": invoice_reference,
            "amount": "12500",
            "currency": "USD",
        },
        "propositions": [],
    }


def _observations(effect: dict[str, Any]) -> dict[str, Any]:
    return {
        "requested_repetitions": 3,
        "stability": "stable",
        "observed_repetitions": 3,
        "inconclusive_repetitions": 0,
        "outcome_group_count": 1,
        "outcome_groups": [
            {
                "repetitions": [1, 2, 3],
                "count": 3,
                "representative_effects": [effect],
            }
        ],
        "trials": [
            {"repetition": repetition, "status": "observed", "inconclusive_reasons": []}
            for repetition in (1, 2, 3)
        ],
    }


def _evidence_record(*, finding_id: str = FINDING_ID) -> dict[str, Any]:
    original_effect = _effect("AC-100")
    changed_effect = _effect("AC-101")
    return {
        "schema_version": "1.3.0",
        "interaction_id": "quickstart-payment",
        "original_input": "Pay AC-100.",
        "execution_plan": {
            "repetitions": 3,
            "max_target_calls": 6,
            "dataset_planned_target_calls": 6,
        },
        "limitations": (
            "UL compares observed action behavior only. It does not determine correctness, "
            "causation, or a production failure rate."
        ),
        "current_baseline": {
            "status": "ORIGINAL REPLAY STABLE (3/3 OBSERVED)",
            "observations": _observations(original_effect),
            "inconclusive_reasons": [],
        },
        "cases": [
            {
                "operator_id": "input.surface.disfluency_repeat",
                "operator_version": "1.0.0",
                "augmented_input": "Pay pay AC-100.",
                "status": "REPEATABLE DIFFERENCE — REVIEW",
                "variation_accepted": True,
                "variation_rejection_reasons": [],
                "observations": _observations(changed_effect),
                "findings": [
                    {
                        "finding_id": finding_id,
                        "category": "changed_grounded_effect_argument",
                        "grounded_field_names": ["invoice_reference"],
                        "severity": "unrated",
                        "review_status": "needs_review",
                        "summary": "The live variation changed a grounded action value.",
                        "reference_effects": [original_effect],
                        "observed_effects": [changed_effect],
                    }
                ],
                "inconclusive_reasons": [],
            }
        ],
        "technical_details": _technical_details(),
    }


def _invariant_evaluation() -> dict[str, Any]:
    def rule(status: str, left: int, right: int) -> dict[str, Any]:
        return {
            "rule_type": "json_values_equal",
            "rule_id": "final-amount-matches-corrected",
            "rule_version": "1.0.0",
            "description": "Final amount equals the corrected amount.",
            "severity": "critical",
            "status": status,
            "reason_code": (
                "all_trials_satisfied" if status == "satisfied" else "one_or_more_trials_violated"
            ),
            "trials": [
                {
                    "repetition": repetition,
                    "status": status,
                    "reason_code": "values_equal" if status == "satisfied" else "values_differ",
                    "left_pointer": "/final_amount",
                    "right_pointer": "/corrected_amount",
                    "resolved_values": {"left": left, "right": right},
                }
                for repetition in (1, 2, 3)
            ],
        }

    suite = DatasetInvariantSuite(
        schema_version="1.0.0",
        observation_source="target_output",
        observation_authority="committed_state_snapshot",
        rules=(
            JsonValuesEqualInvariant(
                type="json_values_equal",
                id="final-amount-matches-corrected",
                version="1.0.0",
                description="Final amount equals the corrected amount.",
                severity="critical",
                left_pointer="/final_amount",
                right_pointer="/corrected_amount",
            ),
        ),
    )
    return {
        "interaction_id": "quickstart-payment",
        "suite_sha256": suite.sha256,
        "observation_source": "target_output",
        "observation_authority": "committed_state_snapshot",
        "baseline": {
            "arm": "baseline",
            "operator_id": None,
            "rules": [rule("satisfied", 100, 100)],
        },
        "variations": [
            {
                "arm": "variation",
                "operator_id": "input.surface.disfluency_repeat",
                "rules": [rule("violated", 200, 100)],
            }
        ],
    }


def _technical_details() -> dict[str, Any]:
    source_frame = SemanticFrame(interaction_id="quickstart-payment", extractor_version="test")
    candidate = DatasetAugmentationCandidate(
        source_interaction_id="quickstart-payment",
        operator_id="input.surface.disfluency_repeat",
        operator_version="1.0.0",
        augmented_input="Pay pay AC-100.",
        expected_input_frame=source_frame,
        reparsed_input_frame=source_frame,
        passed=True,
    )

    def trial_set(arm: str, final_amount: int, corrected_amount: int) -> DatasetEvaluationTrialSet:
        committed_state_snapshot: JsonValue = {
            "final_amount": final_amount,
            "corrected_amount": corrected_amount,
        }
        trials = tuple(
            DatasetEvaluationTrial(
                repetition=repetition,
                target_output=ObservedAgentOutput(
                    raw_output={"message": "completed"},
                    metadata={
                        "committed_state_snapshot": committed_state_snapshot,
                        "state_observation_authority": "environment_self_reported",
                    },
                ),
                observed_frame=SemanticFrame(
                    interaction_id=f"quickstart-payment:{arm}:round-{repetition}",
                    extractor_version="test",
                ),
            )
            for repetition in (1, 2, 3)
        )
        return DatasetEvaluationTrialSet(
            requested_repetitions=3,
            stability="stable",
            trials=trials,
            outcome_groups=(
                DatasetEvaluationOutcomeGroup(repetitions=(1, 2, 3), representative_effects=()),
            ),
        )

    result = DatasetEvaluationResult(
        source=InteractionRecord(
            id="quickstart-payment",
            raw_input="Pay AC-100.",
            raw_observed_output={"final_amount": 100, "corrected_amount": 100},
        ),
        augmentation=DatasetAugmentationResult(
            operator_references=({"id": candidate.operator_id, "version": "1.0.0"},),
            source_records=(
                InteractionRecord(
                    id="quickstart-payment",
                    raw_input="Pay AC-100.",
                    raw_observed_output={"final_amount": 100, "corrected_amount": 100},
                ),
            ),
            source_frames=(source_frame,),
            candidates=(candidate,),
        ),
        baseline=DatasetEvaluationBaseline(
            verdict="no_divergence",
            trial_set=trial_set("current_baseline", 100, 100),
        ),
        cases=(
            DatasetEvaluationCase(
                candidate=candidate,
                verdict="no_divergence",
                trial_set=trial_set("input.surface.disfluency_repeat", 200, 100),
            ),
        ),
    )
    return result.model_dump(mode="json")


def _write_evidence(path: Path, records: list[dict[str, Any]] | None = None) -> bytes:
    project_directory = path.parent / ".ul"
    project_directory.mkdir(mode=0o700, exist_ok=True)
    project_directory.chmod(0o700)
    identity_key_path = project_directory / "pattern-identity.key"
    if not identity_key_path.exists():
        identity_key_path.write_bytes(_PATTERN_IDENTITY_KEY)
        identity_key_path.chmod(0o600)
    raw = b"".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
        for record in (records or [_evidence_record()])
    )
    path.write_bytes(raw)
    return raw


def _review_arguments(
    evidence: Path,
    *,
    status_value: str = "confirmed",
    severity: str | None = "high",
    supersedes: str | None = None,
) -> list[str]:
    arguments = [
        "dataset",
        "review",
        str(evidence),
        FINDING_ID,
        "--status",
        status_value,
        "--reviewer",
        "payments-risk",
        "--reason",
        "The variation committed payment for a different invoice.",
    ]
    if severity is not None:
        arguments.extend(("--severity", severity))
    if supersedes is not None:
        arguments.extend(("--supersedes", supersedes))
    return arguments


def _read_reviews(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_report_review_report_journey_preserves_evidence_and_history(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.jsonl"
    original_bytes = _write_evidence(evidence)
    sidecar = tmp_path / "evidence.reviews.jsonl"

    initial_report = runner.invoke(app, ["dataset", "report", str(evidence)])

    assert initial_report.exit_code == 0, initial_report.output
    assert "Dataset finding report: 1 finding(s)" in initial_report.output
    assert "needs_review=1" in initial_report.output
    assert "Original: Pay AC-100." in initial_report.output
    assert "Variation: Pay pay AC-100." in initial_report.output
    assert '"invoice_reference": "AC-100"' in initial_report.output
    assert '"invoice_reference": "AC-101"' in initial_report.output
    assert "stable; 3/3 observed; groups=1+2+3" in initial_report.output
    assert "not proof of correctness, causation, or production frequency" in initial_report.output
    assert not sidecar.exists()

    first_review = runner.invoke(app, _review_arguments(evidence))

    assert first_review.exit_code == 0, first_review.output
    assert evidence.read_bytes() == original_bytes
    if sys.platform != "win32":
        assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600
    first_record = _read_reviews(sidecar)[0]
    assert first_record["status"] == "confirmed"
    assert first_record["severity"] == "high"
    assert first_record["finding_id"] == FINDING_ID
    assert (
        first_record["evidence_record_sha256"]
        == hashlib.sha256(original_bytes.rstrip(b"\n")).hexdigest()
    )
    assert first_record["supersedes_review_id"] is None

    reviewed_report = runner.invoke(app, ["dataset", "report", str(evidence)])
    assert reviewed_report.exit_code == 0, reviewed_report.output
    assert "confirmed=1" in reviewed_report.output
    assert "Latest review: confirmed, severity=high" in reviewed_report.output
    assert "history: 1" in reviewed_report.output

    replacement = runner.invoke(
        app,
        _review_arguments(
            evidence,
            status_value="expected",
            severity=None,
            supersedes=first_record["review_id"],
        ),
    )

    assert replacement.exit_code == 0, replacement.output
    review_history = _read_reviews(sidecar)
    assert len(review_history) == 2
    assert review_history[0] == first_record
    assert review_history[1]["status"] == "expected"
    assert review_history[1]["severity"] == "unrated"
    assert review_history[1]["supersedes_review_id"] == first_record["review_id"]
    assert evidence.read_bytes() == original_bytes

    final_report = runner.invoke(app, ["dataset", "report", str(evidence)])
    assert final_report.exit_code == 0, final_report.output
    assert "expected=1" in final_report.output
    assert "Latest review: expected, severity=unrated" in final_report.output
    assert "history: 2" in final_report.output


def test_root_json_report_is_stable_and_omits_private_dataset_fields(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.jsonl"
    _write_evidence(evidence)

    report = runner.invoke(app, ["report", str(evidence), "--json"])

    assert report.exit_code == 1, report.output
    expected = {
        "schema_version": "1.6.0",
        "evidence_type": "dataset_evaluation",
        "evidence_schema_versions": ["1.3.0"],
        "evidence_scope": "response_and_state",
        "evaluation_mode": None,
        "capability_limitations": [],
        "review_status": "action_required",
        "exit_code": 1,
        "summary": {
            "finding_count": 1,
            "actionable_finding_count": 1,
            "review_status_counts": {
                "needs_review": 1,
                "confirmed": 0,
                "expected": 0,
                "unsupported": 0,
                "inconclusive": 0,
            },
        },
        "patterns": [
            {
                "pattern_fingerprint": (
                    "ulpf_v1_b5dd705fb4db534431680decfe8b221fbebfd049d7b7aba99c2b59af966a2ca3"
                ),
                "pattern_snapshot_id": (
                    "ulps_v1_c6b003667fc2f8b325a8c753cb6caf0332fbe0e63f27134d7fb4787b87c9f093"
                ),
                "kind": "behavior_difference",
                "category": "changed_grounded_effect_argument",
                "rule_id": None,
                "rule_version": None,
                "summary": "The changed input altered an important action detail.",
                "severity": "unrated",
                "stability": "stable",
                "evidence_authorities": ["model_derived_unverified"],
                "evidence_limitations": ["semantic_model_output_not_independently_verified"],
                "horizontal_facets": {
                    "failure_type": "changed_grounded_effect_argument",
                    "affected_subject": "action",
                    "evidence_level": "model_derived_action",
                    "mechanism_pseudonym": (
                        "ulpm_v1_43c0893b3d2af633ffb223be1f918c9c2d16af24ab8c6535193025f706b866e6"
                    ),
                },
                "finding_count": 1,
                "source_case_count": 1,
                "operators": [
                    {
                        "operator_id": "input.surface.disfluency_repeat",
                        "operator_version": "1.0.0",
                        "summary": "Repeat a word as a natural disfluency.",
                    }
                ],
                "needs_review_count": 1,
                "confirmed_count": 0,
                "members": [
                    {
                        "finding_id": FINDING_ID,
                        "membership_reasons": [
                            "same_action_shape",
                            "same_evidence_authority",
                            "same_evidence_limitation",
                            "same_finding_category",
                            "same_finding_kind",
                            "same_outcome_stability",
                        ],
                        "review_status": "needs_review",
                        "review_severity": "unrated",
                    }
                ],
            }
        ],
        "findings": [
            {
                "finding_id": FINDING_ID,
                "kind": "behavior_difference",
                "category": "changed_grounded_effect_argument",
                "operator_id": "input.surface.disfluency_repeat",
                "operator_version": "1.0.0",
                "rule_id": None,
                "rule_version": None,
                "declared_severity": None,
                "review_status": "needs_review",
                "review_severity": "unrated",
                "requested_repetitions": 3,
                "conclusive_repetitions": 3,
                "inconclusive_repetitions": 0,
                "stability": "stable",
                "evidence_authorities": ["model_derived_unverified"],
                "evidence_limitations": ["semantic_model_output_not_independently_verified"],
                "violated_repetitions": None,
                "next_action": "review_dataset_finding",
                "summary": "The changed input altered an important action detail.",
            }
        ],
    }
    assert json.loads(report.output) == expected
    assert (
        report.output == json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    assert "Pay AC-100" not in report.output
    assert "Pay pay AC-100" not in report.output
    assert "AC-101" not in report.output
    assert "technical_details" not in report.output
    assert "ulpm_v1_" in report.output
    assert _PATTERN_IDENTITY_KEY.hex() not in report.output
    assert "6f5f3a4bf1b01b071a12aaf35df870dcd1fc1a4077db18efe2ab81453b6ef114" not in (report.output)
    for private_label in ("payment_committed", "invoice_reference", "completed"):
        assert hashlib.sha256(private_label.encode()).hexdigest() not in report.output


def test_root_human_report_explains_patterns_and_augmentation_names(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.jsonl"
    _write_evidence(evidence)

    report = runner.invoke(app, ["report", str(evidence)])

    assert report.exit_code == 1, report.output
    assert "Reviewable finding patterns: 1" in report.output
    assert "Patterns group similar evidence; they do not claim a root cause." in report.output
    assert "Pattern 1: The changed input altered an important action detail." in report.output
    assert "Priority: unrated" in report.output
    assert "Evidence authority: model derived unverified" in report.output
    assert "Evidence limitation: semantic model output not independently verified" in report.output
    assert "Why grouped: same finding category and private action shape." in report.output
    assert "Affected: 1 finding(s) across 1 test question(s)" in report.output
    assert "Repeat a word as a natural disfluency." in report.output
    assert "1 needs review; 0 confirmed" in report.output
    assert f"{FINDING_ID}: same action shape, same evidence authority" in report.output
    assert "review=needs_review/unrated" in report.output
    assert "Next: use the per-finding review commands below." in report.output
    assert "Pay AC-100" not in report.output
    assert "AC-101" not in report.output


def test_report_contract_rejects_pattern_review_counts_that_disagree_with_findings(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.jsonl"
    _write_evidence(evidence)
    report = runner.invoke(app, ["report", str(evidence), "--json"])
    payload = json.loads(report.output)
    payload["patterns"][0]["needs_review_count"] = 0
    payload["patterns"][0]["confirmed_count"] = 1

    with pytest.raises(ValueError, match="review counts must match member snapshots"):
        UnifiedReport.model_validate_json(json.dumps(payload))


def test_failure_patterns_group_same_mechanism_across_questions_and_augmentations() -> None:
    finding_ids = tuple(f"ulf_v1_{character * 64}" for character in "abcd")
    operators = (
        "input.surface.typing_noise",
        "input.style.terse",
        "input.surface.typing_noise",
        "input.style.verbose",
    )
    categories = (
        "changed_grounded_effect_argument",
        "changed_grounded_effect_argument",
        "changed_grounded_effect_argument",
        "duplicate_effect",
    )
    summaries = (
        "The changed input altered an important action detail.",
        "The changed input altered an important action detail.",
        "The changed input altered an important action detail.",
        "The changed input made the agent repeat an action.",
    )
    findings = tuple(
        FindingSummary(
            finding_id=finding_id,
            kind="behavior_difference",
            category=category,
            operator_id=operator_id,
            operator_version="1.0.0",
            review_status="needs_review",
            review_severity="unrated",
            requested_repetitions=3,
            conclusive_repetitions=3,
            inconclusive_repetitions=0,
            stability="stable",
            evidence_authorities=("deterministic_evaluator",),
            next_action="review_dataset_finding",
            summary=summary,
        )
        for finding_id, operator_id, category, summary in zip(
            finding_ids, operators, categories, summaries, strict=True
        )
    )
    contexts = {
        finding_ids[0]: dataset_review._PatternContext(
            "1" * 64,
            "source-a",
            operators[0],
            ("deterministic_evaluator",),
        ),
        finding_ids[1]: dataset_review._PatternContext(
            "1" * 64,
            "source-a",
            operators[1],
            ("deterministic_evaluator",),
        ),
        finding_ids[2]: dataset_review._PatternContext(
            "1" * 64,
            "source-b",
            operators[2],
            ("deterministic_evaluator",),
        ),
        finding_ids[3]: dataset_review._PatternContext(
            "2" * 64,
            "source-b",
            operators[3],
            ("deterministic_evaluator",),
        ),
    }

    patterns = dataset_review._build_failure_patterns(
        findings, contexts, pattern_identity_key=_PATTERN_IDENTITY_KEY
    )

    changed_argument_pattern = next(
        pattern for pattern in patterns if pattern.category == "changed_grounded_effect_argument"
    )
    assert changed_argument_pattern.finding_count == 3
    assert changed_argument_pattern.source_case_count == 2
    assert [operator.operator_id for operator in changed_argument_pattern.operators] == [
        "input.style.terse",
        "input.surface.typing_noise",
    ]
    assert (
        tuple(member.finding_id for member in changed_argument_pattern.members) == finding_ids[:3]
    )
    assert len(patterns) == 2


def test_pattern_fingerprint_survives_compatible_membership_changes() -> None:
    finding_ids = tuple(f"ulf_v1_{character * 64}" for character in "ab")
    findings = tuple(
        FindingSummary(
            finding_id=finding_id,
            kind="behavior_difference",
            category="duplicate_effect",
            operator_id="input.surface.typing_noise",
            operator_version="1.0.0",
            review_status="needs_review",
            review_severity="unrated",
            requested_repetitions=3,
            conclusive_repetitions=3,
            inconclusive_repetitions=0,
            stability="stable",
            evidence_authorities=("deterministic_evaluator",),
            next_action="review_dataset_finding",
            summary="The changed input made the agent repeat an action.",
        )
        for finding_id in finding_ids
    )
    contexts = {
        finding_id: dataset_review._PatternContext(
            "1" * 64,
            f"source-{index}",
            "input.surface.typing_noise",
            ("deterministic_evaluator",),
        )
        for index, finding_id in enumerate(finding_ids)
    }

    first_snapshot = dataset_review._build_failure_patterns(
        findings[:1],
        {finding_ids[0]: contexts[finding_ids[0]]},
        pattern_identity_key=_PATTERN_IDENTITY_KEY,
    )[0]
    second_snapshot = dataset_review._build_failure_patterns(
        findings, contexts, pattern_identity_key=_PATTERN_IDENTITY_KEY
    )[0]

    assert first_snapshot.pattern_fingerprint == second_snapshot.pattern_fingerprint
    assert first_snapshot.pattern_snapshot_id != second_snapshot.pattern_snapshot_id
    assert all(
        member.membership_reasons
        == (
            "same_action_shape",
            "same_evidence_authority",
            "same_evidence_limitation",
            "same_finding_category",
            "same_finding_kind",
            "same_outcome_stability",
        )
        for member in second_snapshot.members
    )


def test_private_action_shapes_split_snapshots_without_leaking_public_hashes() -> None:
    finding_ids = tuple(f"ulf_v1_{character * 64}" for character in "ab")
    findings = tuple(
        FindingSummary(
            finding_id=finding_id,
            kind="behavior_difference",
            category="duplicate_effect",
            operator_id="input.surface.typing_noise",
            operator_version="1.0.0",
            review_status="needs_review",
            review_severity="unrated",
            requested_repetitions=1,
            conclusive_repetitions=1,
            inconclusive_repetitions=0,
            stability="stable",
            evidence_authorities=("deterministic_evaluator",),
            next_action="review_dataset_finding",
            summary="The changed input made the agent repeat an action.",
        )
        for finding_id in finding_ids
    )
    private_keys = ("1" * 64, "2" * 64)
    contexts = {
        finding_id: dataset_review._PatternContext(
            private_keys[index],
            f"source-{index}",
            "input.surface.typing_noise",
            ("deterministic_evaluator",),
        )
        for index, finding_id in enumerate(finding_ids)
    }

    patterns = dataset_review._build_failure_patterns(
        findings, contexts, pattern_identity_key=_PATTERN_IDENTITY_KEY
    )
    serialized_patterns = json.dumps(
        [pattern.model_dump(mode="json") for pattern in patterns], sort_keys=True
    )

    assert len(patterns) == 2
    assert patterns[0].pattern_fingerprint != patterns[1].pattern_fingerprint
    assert patterns[0].pattern_snapshot_id != patterns[1].pattern_snapshot_id
    assert all(private_key not in serialized_patterns for private_key in private_keys)


def test_mechanism_pseudonym_resists_dictionary_checks_and_key_rotation_starts_new_identity() -> (
    None
):
    private_mechanism_digest = hashlib.sha256(b"payment_committed").hexdigest()
    pseudonym = pattern_mechanism_pseudonym(
        _PATTERN_IDENTITY_KEY,
        private_mechanism_digest,
    )
    rotated_key_pseudonym = pattern_mechanism_pseudonym(
        bytes(reversed(_PATTERN_IDENTITY_KEY)),
        private_mechanism_digest,
    )

    assert pseudonym != f"ulpm_v1_{private_mechanism_digest}"
    assert pseudonym != rotated_key_pseudonym
    assert private_mechanism_digest not in pseudonym


def test_pattern_grouping_splits_outcomes_and_authorities_but_keeps_review_cohorts() -> None:
    variants = (
        ("a", "stable", ("deterministic_evaluator",), "needs_review"),
        ("b", "unstable", ("deterministic_evaluator",), "needs_review"),
        ("c", "stable", ("independent_observer",), "needs_review"),
        ("d", "stable", ("deterministic_evaluator",), "confirmed"),
    )
    findings = tuple(
        FindingSummary(
            finding_id=f"ulf_v1_{character * 64}",
            kind="behavior_difference",
            category="duplicate_effect",
            operator_id="input.surface.typing_noise",
            operator_version="1.0.0",
            review_status=review_status,
            review_severity="unrated",
            requested_repetitions=3,
            conclusive_repetitions=3,
            inconclusive_repetitions=0,
            stability=stability,
            evidence_authorities=authorities,
            next_action="review_dataset_finding",
            summary="The changed input made the agent repeat an action.",
        )
        for character, stability, authorities, review_status in variants
    )
    contexts = {
        finding.finding_id: dataset_review._PatternContext(
            "1" * 64,
            f"source-{index}",
            "input.surface.typing_noise",
            finding.evidence_authorities,
        )
        for index, finding in enumerate(findings)
        if finding.finding_id is not None
    }

    patterns = dataset_review._build_failure_patterns(
        findings, contexts, pattern_identity_key=_PATTERN_IDENTITY_KEY
    )

    assert len(patterns) == 3
    assert len({pattern.pattern_fingerprint for pattern in patterns}) == 3
    mixed_review_pattern = next(pattern for pattern in patterns if pattern.finding_count == 2)
    assert mixed_review_pattern.needs_review_count == 1
    assert mixed_review_pattern.confirmed_count == 1
    assert {member.review_status for member in mixed_review_pattern.members} == {
        "needs_review",
        "confirmed",
    }


def test_pattern_fingerprint_survives_a_member_review_transition() -> None:
    finding = FindingSummary(
        finding_id=FINDING_ID,
        kind="behavior_difference",
        category="duplicate_effect",
        operator_id="input.surface.typing_noise",
        operator_version="1.0.0",
        review_status="needs_review",
        review_severity="unrated",
        requested_repetitions=1,
        conclusive_repetitions=1,
        inconclusive_repetitions=0,
        stability="stable",
        evidence_authorities=("deterministic_evaluator",),
        next_action="review_dataset_finding",
        summary="The changed input made the agent repeat an action.",
    )
    context = dataset_review._PatternContext(
        "1" * 64,
        "source-a",
        "input.surface.typing_noise",
        ("deterministic_evaluator",),
    )
    before_review = dataset_review._build_failure_patterns(
        (finding,),
        {FINDING_ID: context},
        pattern_identity_key=_PATTERN_IDENTITY_KEY,
    )[0]
    reviewed_finding = finding.model_copy(
        update={"review_status": "confirmed", "review_severity": "high"}
    )
    after_review = dataset_review._build_failure_patterns(
        (reviewed_finding,),
        {FINDING_ID: context},
        pattern_identity_key=_PATTERN_IDENTITY_KEY,
    )[0]
    after_key_rotation = dataset_review._build_failure_patterns(
        (reviewed_finding,),
        {FINDING_ID: context},
        pattern_identity_key=bytes(reversed(_PATTERN_IDENTITY_KEY)),
    )[0]

    assert before_review.pattern_fingerprint == after_review.pattern_fingerprint
    assert before_review.pattern_snapshot_id != after_review.pattern_snapshot_id
    assert before_review.members[0].review_status == "needs_review"
    assert after_review.members[0].review_status == "confirmed"
    assert after_review.pattern_fingerprint != after_key_rotation.pattern_fingerprint


def test_pattern_grouping_splits_conflicting_customer_rules() -> None:
    finding_ids = tuple(f"ulf_v1_{character * 64}" for character in "ab")
    findings = tuple(
        FindingSummary(
            finding_id=finding_id,
            kind="customer_invariant_violation",
            category="customer_invariant_violation",
            operator_id="input.surface.typing_noise",
            operator_version="1.0.0",
            rule_id=f"rule-{index}",
            rule_version="1.0.0",
            declared_severity="high",
            review_status="needs_review",
            review_severity="unrated",
            requested_repetitions=1,
            conclusive_repetitions=1,
            inconclusive_repetitions=0,
            stability="stable",
            evidence_authorities=("customer_declared", "deterministic_evaluator"),
            violated_repetitions=1,
            next_action="review_dataset_finding",
            summary="The agent violated a customer-defined rule.",
        )
        for index, finding_id in enumerate(finding_ids)
    )
    contexts = {
        finding_id: dataset_review._PatternContext(
            "5" * 64,
            f"source-{index}",
            "input.surface.typing_noise",
            ("customer_declared", "deterministic_evaluator"),
        )
        for index, finding_id in enumerate(finding_ids)
    }

    patterns = dataset_review._build_failure_patterns(
        findings, contexts, pattern_identity_key=_PATTERN_IDENTITY_KEY
    )

    assert len(patterns) == 2
    assert {pattern.rule_id for pattern in patterns} == {"rule-0", "rule-1"}


def test_effect_statuses_produce_distinct_pattern_signatures() -> None:
    completed_effect = dataset_review._Effect.model_validate(_effect("AC-100"))
    failed_effect = completed_effect.model_copy(update={"status": "failed"})

    assert dataset_review._bounded_effect_mechanisms([completed_effect]) != (
        dataset_review._bounded_effect_mechanisms([failed_effect])
    )


def test_behavior_pattern_signature_abstains_from_unbounded_effect_metadata() -> None:
    effect = dataset_review._Effect.model_validate(_effect("AC-100"))
    finding = dataset_review._Finding(
        finding_id=FINDING_ID,
        category="changed_grounded_effect_argument",
        grounded_field_names=["invoice_reference"],
        severity="unrated",
        review_status="needs_review",
        summary="The live variation changed a grounded action value.",
        reference_effects=[effect] * (dataset_review._MAXIMUM_PATTERN_EFFECTS + 1),
        observed_effects=[effect],
    )

    assert dataset_review._behavior_pattern_signature(finding) is None


def test_private_pattern_identity_is_nfc_normalized_and_total_size_bounded() -> None:
    decomposed_payload = _effect("cafe\u0301")
    decomposed_payload["predicate"] = "paie\u0301ment"
    composed_payload = _effect("café")
    composed_payload["predicate"] = "paiément"
    decomposed_effect = dataset_review._Effect.model_validate(decomposed_payload)
    composed_effect = dataset_review._Effect.model_validate(composed_payload)

    def finding(effect: dataset_review._Effect, grounded_field: str) -> dataset_review._Finding:
        return dataset_review._Finding(
            finding_id=FINDING_ID,
            category="changed_grounded_effect_argument",
            grounded_field_names=[grounded_field],
            severity="unrated",
            review_status="needs_review",
            summary="Needs review.",
            reference_effects=[effect],
            observed_effects=[effect],
        )

    assert dataset_review._behavior_pattern_signature(
        finding(composed_effect, "référence")
    ) == dataset_review._behavior_pattern_signature(
        finding(decomposed_effect, "re\u0301fe\u0301rence")
    )

    oversized_label = composed_effect.model_copy(update={"status": "x" * 501})
    assert dataset_review._behavior_pattern_signature(finding(oversized_label, "reference")) is None

    large_effect = composed_effect.model_copy(
        update={
            "kind": "k" * 500,
            "predicate": "p" * 500,
            "status": "s" * 500,
        }
    )
    large_identity = finding(large_effect, "reference").model_copy(
        update={"reference_effects": [large_effect] * 100, "observed_effects": [large_effect] * 100}
    )
    assert dataset_review._behavior_pattern_signature(large_identity) is None


def test_root_report_uses_a_placeholder_in_windows_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "evidence & injected.jsonl"
    _write_evidence(evidence)
    monkeypatch.setattr(report_module, "_WINDOWS", True)

    report = runner.invoke(app, ["report", str(evidence)])

    assert report.exit_code == 1, report.output
    assert f"ul dataset review EVIDENCE {FINDING_ID}" in report.output
    assert "ul dataset report EVIDENCE" in report.output
    assert f"ul dataset review {evidence}" not in report.output


def test_root_report_treats_unstable_variation_as_exit_one_finding(tmp_path: Path) -> None:
    evidence = tmp_path / "unstable.jsonl"
    record = _evidence_record()
    case = record["cases"][0]
    case["findings"] = []
    case["status"] = "UNSTABLE VARIATION — REVIEW"
    observations = case["observations"]
    observations["stability"] = "unstable"
    observations["outcome_group_count"] = 2
    observations["outcome_groups"] = [
        {
            "repetitions": [1, 2],
            "count": 2,
            "representative_effects": [_effect("AC-100")],
        },
        {
            "repetitions": [3],
            "count": 1,
            "representative_effects": [_effect("AC-101")],
        },
    ]
    _write_evidence(evidence, [record])

    report = runner.invoke(app, ["report", str(evidence), "--json"])

    assert report.exit_code == 1, report.output
    payload = json.loads(report.output)
    assert payload["review_status"] == "action_required"
    assert payload["summary"]["finding_count"] == 1
    assert payload["findings"][0]["category"] == "unstable_behavior"
    assert payload["findings"][0]["summary"] == (
        "The changed input produced inconsistent behavior across repetitions."
    )


def test_root_report_maps_incomplete_dataset_evidence_to_exit_two(tmp_path: Path) -> None:
    evidence = tmp_path / "inconclusive.jsonl"
    record = _evidence_record()
    case = record["cases"][0]
    case["findings"] = []
    case["status"] = "COULDN'T DETERMINE"
    case["inconclusive_reasons"] = ["variation execution failed"]
    observations = case["observations"]
    observations.update(
        {
            "stability": "inconclusive",
            "observed_repetitions": 0,
            "inconclusive_repetitions": 3,
            "outcome_group_count": 0,
            "outcome_groups": [],
            "trials": [
                {
                    "repetition": repetition,
                    "status": "inconclusive",
                    "inconclusive_reasons": ["variation execution failed"],
                }
                for repetition in (1, 2, 3)
            ],
        }
    )
    _write_evidence(evidence, [record])

    report = runner.invoke(app, ["report", str(evidence), "--json"])

    assert report.exit_code == 2, report.output
    payload = json.loads(report.output)
    assert payload["review_status"] == "inconclusive"
    assert payload["exit_code"] == 2
    assert payload["findings"] == []


def test_report_schema_1_4_shows_customer_invariants_separately(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.jsonl"
    record = _evidence_record()
    record["schema_version"] = "1.4.0"
    record["invariant_evaluation"] = _invariant_evaluation()
    _write_evidence(evidence, [record])

    report = runner.invoke(app, ["dataset", "report", str(evidence)])

    assert report.exit_code == 0, report.output
    normalized_output = " ".join(report.output.split())
    assert "Dataset finding report: 2 finding(s)" in normalized_output
    assert "Category: customer_invariant_violation" in normalized_output
    assert "Rule transition: original=satisfied; variation=violated" in normalized_output
    assert "Customer invariant evaluation" in normalized_output
    assert "Declared observation authority: committed_state_snapshot" in normalized_output
    assert "severity=critical; arm=original; status=satisfied" in normalized_output
    assert (
        "severity=critical; arm=variation (input.surface.disfluency_repeat); status=violated"
        in normalized_output
    )
    assert "Description: Final amount equals the corrected amount." in normalized_output
    assert "reason=one_or_more_trials_violated" in normalized_output
    assert "satisfied=0, violated=3, not_evaluable=0" in normalized_output
    assert "selected_values=" not in normalized_output
    assert "Customer rule violated against declared committed_state_snapshot." in normalized_output
    assert "agent wrong" not in normalized_output.casefold()


def test_invariant_violation_without_semantic_difference_can_be_reviewed(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.jsonl"
    record = _evidence_record()
    record["schema_version"] = "1.4.0"
    record["invariant_evaluation"] = _invariant_evaluation()
    record["cases"][0]["findings"] = []
    record["cases"][0]["status"] = "NO OBSERVED DIFFERENCE"
    _write_evidence(evidence, [record])

    report = runner.invoke(app, ["dataset", "report", str(evidence)])

    assert report.exit_code == 0, report.output
    indexed_findings = dataset_review._index_findings(dataset_review._load_evidence(evidence))
    assert len(indexed_findings) == 1
    invariant_finding_id = next(iter(indexed_findings))
    assert indexed_findings[invariant_finding_id].kind == "customer_invariant_violation"
    assert f"Finding {invariant_finding_id}" in report.output
    assert "Category: customer_invariant_violation" in report.output
    assert "Semantic comparison status: NO OBSERVED DIFFERENCE" in report.output
    assert "Invariant finding status: original=satisfied; variation=violated" in report.output
    assert "Machine status: NO OBSERVED DIFFERENCE" not in report.output
    assert "declared_severity=critical" in report.output.replace("\n", " ")
    assert "resolved_values" not in report.output
    assert "200" not in report.output

    sensitive_report = runner.invoke(
        app,
        [
            "dataset",
            "report",
            str(evidence),
            "--show-sensitive-values",
            "--finding",
            invariant_finding_id,
        ],
    )
    assert sensitive_report.exit_code == 0, sensitive_report.output
    assert "may contain secrets or PII" in sensitive_report.output
    assert 'Original trial 1 selected values: {"left":100,"right":100}' in sensitive_report.output
    assert 'Variation trial 1 selected values: {"left":200,"right":100}' in sensitive_report.output
    assert "shown=6; omitted=0" in sensitive_report.output

    review = runner.invoke(
        app,
        [
            "dataset",
            "review",
            str(evidence),
            invariant_finding_id,
            "--status",
            "confirmed",
            "--severity",
            "critical",
            "--reviewer",
            "payments-risk",
            "--reason",
            "The variation violated the declared final-amount rule.",
        ],
    )

    assert review.exit_code == 0, review.output
    reviewed_report = runner.invoke(app, ["dataset", "report", str(evidence)])
    assert reviewed_report.exit_code == 0, reviewed_report.output
    assert "confirmed=1" in reviewed_report.output
    assert "Latest review: confirmed, severity=critical" in reviewed_report.output


def test_invariant_finding_rejects_changed_values_and_id_tracks_variation_identity(
    tmp_path: Path,
) -> None:
    first_evidence = tmp_path / "first.jsonl"
    first = _evidence_record()
    first["schema_version"] = "1.4.0"
    first["invariant_evaluation"] = _invariant_evaluation()
    first["cases"][0]["findings"] = []
    _write_evidence(first_evidence, [first])
    first_id = next(
        iter(dataset_review._index_findings(dataset_review._load_evidence(first_evidence)))
    )
    expected_identity = {
        "finding_kind": "customer_invariant_violation",
        "interaction_id": "quickstart-payment",
        "original_input": "Pay AC-100.",
        "operator_id": "input.surface.disfluency_repeat",
        "operator_version": "1.0.0",
        "augmented_input": "Pay pay AC-100.",
        "suite_sha256": first["invariant_evaluation"]["suite_sha256"],
        "observation_authority": "committed_state_snapshot",
        "rule_id": "final-amount-matches-corrected",
        "rule_version": "1.0.0",
        "rule_type": "json_values_equal",
    }
    assert first_id == f"ulf_v1_{dataset_review._canonical_json_sha256(expected_identity)}"

    changed_values_evidence = tmp_path / "changed-values.jsonl"
    changed_values = json.loads(json.dumps(first))
    changed_values["invariant_evaluation"]["variations"][0]["rules"][0]["trials"][0][
        "resolved_values"
    ]["left"] = 300
    _write_evidence(changed_values_evidence, [changed_values])
    with pytest.raises(
        dataset_review._ReviewInputError,
        match="does not match the technical execution evidence",
    ):
        dataset_review._index_findings(dataset_review._load_evidence(changed_values_evidence))

    changed_input_evidence = tmp_path / "changed-input.jsonl"
    changed_input = json.loads(json.dumps(first))
    changed_input["cases"][0]["augmented_input"] = "Pay pay pay AC-100."
    changed_input["technical_details"]["augmentation"]["candidates"][0]["augmented_input"] = (
        "Pay pay pay AC-100."
    )
    changed_input["technical_details"]["cases"][0]["candidate"]["augmented_input"] = (
        "Pay pay pay AC-100."
    )
    _write_evidence(changed_input_evidence, [changed_input])
    changed_input_id = next(
        iter(dataset_review._index_findings(dataset_review._load_evidence(changed_input_evidence)))
    )

    assert first_id != changed_input_id


def test_invariant_variation_must_map_to_exactly_one_case(tmp_path: Path) -> None:
    evidence = tmp_path / "ambiguous.jsonl"
    record = _evidence_record()
    record["schema_version"] = "1.4.0"
    record["invariant_evaluation"] = _invariant_evaluation()
    record["cases"][0]["findings"] = []
    duplicate_case = json.loads(json.dumps(record["cases"][0]))
    duplicate_case["augmented_input"] = "Pay pay pay AC-100."
    record["cases"].append(duplicate_case)
    _write_evidence(evidence, [record])

    report = runner.invoke(app, ["dataset", "report", str(evidence)])

    assert report.exit_code != 0
    assert "exactly one evidence case" in report.output


def test_invariant_finding_rejects_nonexecuted_variation(tmp_path: Path) -> None:
    evidence = tmp_path / "nonexecuted.jsonl"
    record = _evidence_record()
    record["schema_version"] = "1.4.0"
    record["invariant_evaluation"] = _invariant_evaluation()
    record["cases"][0]["findings"] = []
    record["cases"][0]["variation_accepted"] = False
    record["cases"][0]["variation_rejection_reasons"] = ["independent validation rejected it"]
    record["cases"][0]["observations"] = None
    _write_evidence(evidence, [record])

    report = runner.invoke(app, ["dataset", "report", str(evidence)])

    assert report.exit_code != 0
    normalized_output = " ".join(report.output.split())
    assert "accepted" in normalized_output
    assert "executed" in normalized_output
    assert "repetition-consistent evidence" in normalized_output


def test_invariant_finding_rejects_failed_technical_trials(tmp_path: Path) -> None:
    evidence = tmp_path / "failed-technical-trials.jsonl"
    record = _evidence_record()
    record["schema_version"] = "1.4.0"
    record["invariant_evaluation"] = _invariant_evaluation()
    record["cases"][0]["findings"] = []
    observations = record["cases"][0]["observations"]
    observations.update(
        {
            "stability": "inconclusive",
            "observed_repetitions": 0,
            "inconclusive_repetitions": 3,
            "outcome_group_count": 0,
            "outcome_groups": [],
            "trials": [
                {
                    "repetition": repetition,
                    "status": "inconclusive",
                    "inconclusive_reasons": ["variation execution failed"],
                }
                for repetition in (1, 2, 3)
            ],
        }
    )
    technical_case = record["technical_details"]["cases"][0]
    technical_case["verdict"] = "inconclusive"
    technical_case["inconclusive_reasons"] = ["variation execution failed"]
    technical_case["trial_set"]["stability"] = "inconclusive"
    technical_case["trial_set"]["outcome_groups"] = []
    for trial in technical_case["trial_set"]["trials"]:
        trial["target_output"] = None
        trial["observed_frame"] = None
        trial["inconclusive_reasons"] = ["variation execution failed"]
    _write_evidence(evidence, [record])

    report = runner.invoke(app, ["dataset", "report", str(evidence)])

    assert report.exit_code != 0
    normalized_output = " ".join(report.output.split())
    assert "does not match the technical execution" in normalized_output
    assert "evidence" in normalized_output


def test_invariant_finding_rejects_rule_definitions_outside_suite_digest(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "changed-rule.jsonl"
    record = _evidence_record()
    record["schema_version"] = "1.4.0"
    record["invariant_evaluation"] = _invariant_evaluation()
    record["cases"][0]["findings"] = []
    for arm in (
        record["invariant_evaluation"]["baseline"],
        *record["invariant_evaluation"]["variations"],
    ):
        for trial in arm["rules"][0]["trials"]:
            trial["left_pointer"] = "/secret"
    _write_evidence(evidence, [record])

    report = runner.invoke(app, ["dataset", "report", str(evidence)])

    assert report.exit_code != 0
    assert "suite digest does not match its rule definitions" in " ".join(report.output.split())


def test_sensitive_invariant_lines_cover_extended_rules_and_sanitize_controls() -> None:
    common = {
        "rule_id": "rule-one",
        "rule_version": "1.0.0",
        "description": "A rule.",
        "severity": "high",
    }
    literal_rule = dataset_review.DatasetInvariantValueEqualsRuleEvaluation.model_validate(
        {
            **common,
            "rule_type": "json_value_equals_literal",
            "status": "violated",
            "reason_code": "one_or_more_trials_violated",
            "value_pointer": "/status",
            "literal": "approved",
            "trials": (
                {
                    "repetition": 1,
                    "status": "violated",
                    "reason_code": "value_differs_from_literal",
                    "value_pointer": "/status",
                    "resolved_values": {
                        "actual": (
                            "denied\n\x1b]8;;bad\x07\u2028line\u2029paragraph\u202eright-to-left"
                        )
                    },
                },
            ),
        }
    )
    set_rule = dataset_review.DatasetInvariantValueInSetRuleEvaluation.model_validate(
        {
            **common,
            "rule_type": "json_value_in_allowed_set",
            "status": "violated",
            "reason_code": "one_or_more_trials_violated",
            "value_pointer": "/status",
            "allowed_values": ("approved", "pending"),
            "trials": (
                {
                    "repetition": 1,
                    "status": "violated",
                    "reason_code": "value_not_in_allowed_set",
                    "value_pointer": "/status",
                    "resolved_values": {"actual": "denied"},
                },
            ),
        }
    )
    array_rule = dataset_review.DatasetInvariantArrayUniqueRuleEvaluation.model_validate(
        {
            **common,
            "rule_type": "json_array_items_unique_by",
            "status": "violated",
            "reason_code": "one_or_more_trials_violated",
            "array_pointer": "/payments",
            "key_pointers": ("/invoice",),
            "trials": (
                {
                    "repetition": 1,
                    "status": "violated",
                    "reason_code": "duplicate_array_items",
                    "array_pointer": "/payments",
                    "key_pointers": ("/invoice",),
                    "item_count": 2,
                    "duplicate_indices": (0, 1),
                },
            ),
        }
    )

    literal_lines = list(dataset_review._sensitive_invariant_lines(literal_rule, literal_rule))
    set_lines = list(dataset_review._sensitive_invariant_lines(set_rule, set_rule))
    array_lines = list(dataset_review._sensitive_invariant_lines(array_rule, array_rule))

    assert 'Configured invariant literal: {"literal":"approved"}' in literal_lines
    sensitive_literal_output = dataset_review._sanitize_plain_text("\n".join(literal_lines))
    assert (
        '"actual":"denied\\n\\u001b]8;;bad\\u0007\\u2028line\\u2029paragraph'
        '\\u202eright-to-left"' in sensitive_literal_output
    )
    assert set_lines[-2:] == [
        'Configured allowed value 1: {"value":"approved"}',
        'Configured allowed value 2: {"value":"pending"}',
    ]
    assert array_lines == [
        "Selected values unavailable: array uniqueness evidence intentionally retains indices "
        "and pointers only."
    ]


def test_sensitive_value_disclosure_rejects_partial_output_over_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "evidence.jsonl"
    record = _evidence_record()
    record["schema_version"] = "1.4.0"
    record["invariant_evaluation"] = _invariant_evaluation()
    record["cases"][0]["findings"] = []
    _write_evidence(evidence, [record])
    invariant_finding_id = next(
        iter(dataset_review._index_findings(dataset_review._load_evidence(evidence)))
    )
    monkeypatch.setattr(dataset_review, "_MAXIMUM_SENSITIVE_DISCLOSURE_LINES", 1)

    report = runner.invoke(
        app,
        [
            "dataset",
            "report",
            str(evidence),
            "--show-sensitive-values",
            "--finding",
            invariant_finding_id,
        ],
    )

    assert report.exit_code != 0
    assert "selected finding values exceed the safe disclosure cap" in " ".join(
        report.output.split()
    )
    assert "selected values:" not in report.output


def test_sensitive_value_disclosure_requires_one_invariant_finding(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.jsonl"
    _write_evidence(evidence)

    missing_finding = runner.invoke(
        app, ["dataset", "report", str(evidence), "--show-sensitive-values"]
    )
    semantic_finding = runner.invoke(
        app,
        [
            "dataset",
            "report",
            str(evidence),
            "--show-sensitive-values",
            "--finding",
            FINDING_ID,
        ],
    )

    assert missing_finding.exit_code != 0
    missing_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", missing_finding.output).split())
    assert "requires --finding FINDING_ID" in missing_output
    assert semantic_finding.exit_code != 0
    semantic_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", semantic_finding.output).split())
    assert "only for a reviewable" in semantic_output
    assert "invariant finding" in semantic_output


def test_sensitive_value_printer_does_not_wrap_beyond_counted_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = StringIO()
    monkeypatch.setattr(
        dataset_review,
        "console",
        Console(file=output, width=20, force_terminal=False),
    )

    dataset_review._print_sensitive_plain("x" * 4_096)

    assert output.getvalue() == "x" * 4_096 + "\n"


@pytest.mark.parametrize(
    ("status_value", "severity", "expected_exit", "expected_review_status", "actionable_count"),
    [
        ("confirmed", "critical", 1, "action_required", 1),
        ("expected", None, 0, "resolved", 0),
        ("unsupported", None, 0, "resolved", 0),
        ("inconclusive", None, 2, "inconclusive", 0),
    ],
)
def test_all_review_statuses_are_recorded(
    tmp_path: Path,
    status_value: str,
    severity: str | None,
    expected_exit: int,
    expected_review_status: str,
    actionable_count: int,
) -> None:
    evidence = tmp_path / "results.jsonl"
    _write_evidence(evidence)

    result = runner.invoke(
        app,
        _review_arguments(evidence, status_value=status_value, severity=severity),
    )

    assert result.exit_code == 0, result.output
    record = _read_reviews(tmp_path / "results.reviews.jsonl")[0]
    assert record["status"] == status_value
    assert record["severity"] == (severity or "unrated")

    report = runner.invoke(app, ["report", str(evidence), "--json"])
    assert report.exit_code == expected_exit, report.output
    payload = json.loads(report.output)
    assert payload["review_status"] == expected_review_status
    assert payload["summary"]["actionable_finding_count"] == actionable_count
    assert payload["summary"]["review_status_counts"][status_value] == 1


@pytest.mark.parametrize("status_value", ["expected", "unsupported", "inconclusive"])
def test_only_confirmed_reviews_may_have_rated_severity(tmp_path: Path, status_value: str) -> None:
    evidence = tmp_path / "evidence.jsonl"
    _write_evidence(evidence)

    result = runner.invoke(
        app,
        _review_arguments(evidence, status_value=status_value, severity="low"),
    )

    assert result.exit_code != 0
    assert "review fields are invalid" in result.output
    assert not (tmp_path / "evidence.reviews.jsonl").exists()


def test_review_requires_exact_finding_and_active_supersession_ids(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.jsonl"
    _write_evidence(evidence)

    unknown = _review_arguments(evidence)
    unknown[3] = f"ulf_v1_{'b' * 64}"
    unknown_result = runner.invoke(app, unknown)
    assert unknown_result.exit_code != 0
    assert "finding ID was not found" in unknown_result.output

    first = runner.invoke(app, _review_arguments(evidence))
    assert first.exit_code == 0, first.output
    active_id = _read_reviews(tmp_path / "evidence.reviews.jsonl")[0]["review_id"]

    missing_supersession = runner.invoke(app, _review_arguments(evidence))
    assert missing_supersession.exit_code != 0
    assert "supersedes" in missing_supersession.output
    assert active_id in missing_supersession.output

    wrong_supersession = runner.invoke(
        app,
        _review_arguments(evidence, supersedes=f"ulr_{'0' * 8}-0000-4000-8000-{'0' * 12}"),
    )
    assert wrong_supersession.exit_code != 0
    assert "supersedes" in wrong_supersession.output
    assert active_id in wrong_supersession.output
    assert len(_read_reviews(tmp_path / "evidence.reviews.jsonl")) == 1


def test_duplicate_finding_and_review_ids_are_rejected(tmp_path: Path) -> None:
    duplicate_findings = tmp_path / "duplicate-findings.jsonl"
    _write_evidence(duplicate_findings, [_evidence_record(), _evidence_record()])

    evidence_result = runner.invoke(app, ["dataset", "report", str(duplicate_findings)])
    assert evidence_result.exit_code != 0
    assert "duplicate finding ID" in evidence_result.output

    evidence = tmp_path / "evidence.jsonl"
    raw_evidence = _write_evidence(evidence).rstrip(b"\n")
    review_id = "ulr_00000000-0000-4000-8000-000000000000"
    record = _manual_review(
        finding_id=FINDING_ID,
        review_id=review_id,
        evidence_sha256=hashlib.sha256(raw_evidence).hexdigest(),
    )
    reviews = tmp_path / "evidence.reviews.jsonl"
    reviews.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")

    review_result = runner.invoke(app, ["dataset", "report", str(evidence)])
    assert review_result.exit_code != 0
    assert "duplicate review ID" in review_result.output


def _manual_review(
    *,
    finding_id: str,
    review_id: str = "ulr_00000000-0000-4000-8000-000000000000",
    evidence_sha256: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "schema_version": "1.0.0",
        "review_id": review_id,
        "evidence_record_sha256": evidence_sha256,
        "finding_id": finding_id,
        "status": "confirmed",
        "severity": "high",
        "reviewer": "risk-team",
        "reason": "Observed wrong invoice.",
        "reviewed_at": datetime.now(UTC).isoformat(),
        "supersedes_review_id": None,
    }
    record.update(extra or {})
    return record


def test_review_digest_detects_evidence_tampering(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.jsonl"
    _write_evidence(evidence)
    recorded = runner.invoke(app, _review_arguments(evidence))
    assert recorded.exit_code == 0, recorded.output

    tampered = _evidence_record()
    tampered["original_input"] = "Pay AC-999."
    _write_evidence(evidence, [tampered])

    result = runner.invoke(app, ["dataset", "report", str(evidence)])

    assert result.exit_code != 0
    assert "digest does not match" in result.output


@pytest.mark.parametrize(
    "invalid_evidence",
    [
        b'{"schema_version":',
        json.dumps({**_evidence_record(), "unexpected": True}).encode() + b"\n",
        b"\n",
    ],
)
def test_malformed_truncated_extra_field_and_empty_evidence_are_rejected(
    tmp_path: Path, invalid_evidence: bytes
) -> None:
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_bytes(invalid_evidence)

    result = runner.invoke(app, ["dataset", "report", str(evidence)])

    assert result.exit_code != 0
    assert "evidence" in result.output.casefold()


def test_invalid_evidence_diagnostic_lists_current_schema(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_bytes(b'{"schema_version":')

    result = runner.invoke(app, ["dataset", "report", str(evidence)])

    assert result.exit_code != 0
    assert "1.9.0" in result.output


def test_malformed_extra_field_and_digest_mismatch_reviews_are_rejected(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.jsonl"
    evidence_bytes = _write_evidence(evidence).rstrip(b"\n")
    reviews = tmp_path / "reviews.jsonl"

    reviews.write_bytes(b'{"schema_version":')
    malformed = runner.invoke(app, ["dataset", "report", str(evidence), "--reviews", str(reviews)])
    assert malformed.exit_code != 0
    assert "review file is not valid review JSONL" in malformed.output

    extra_field = _manual_review(
        finding_id=FINDING_ID,
        evidence_sha256=hashlib.sha256(evidence_bytes).hexdigest(),
        extra={"unexpected": True},
    )
    reviews.write_text(json.dumps(extra_field) + "\n", encoding="utf-8")
    extra = runner.invoke(app, ["dataset", "report", str(evidence), "--reviews", str(reviews)])
    assert extra.exit_code != 0
    assert "review file is not valid review JSONL" in extra.output

    wrong_digest = _manual_review(finding_id=FINDING_ID, evidence_sha256="0" * 64)
    reviews.write_text(json.dumps(wrong_digest) + "\n", encoding="utf-8")
    mismatch = runner.invoke(app, ["dataset", "report", str(evidence), "--reviews", str(reviews)])
    assert mismatch.exit_code != 0
    assert "digest does not match" in mismatch.output


def test_oversize_evidence_and_reviews_are_rejected(tmp_path: Path) -> None:
    oversized_evidence = tmp_path / "oversized.jsonl"
    with oversized_evidence.open("wb") as stream:
        stream.truncate(128_000_001)
    evidence_result = runner.invoke(app, ["dataset", "report", str(oversized_evidence)])
    assert evidence_result.exit_code != 0
    assert "128 MB limit" in evidence_result.output

    evidence = tmp_path / "evidence.jsonl"
    _write_evidence(evidence)
    oversized_reviews = tmp_path / "reviews.jsonl"
    with oversized_reviews.open("wb") as stream:
        stream.truncate(10_000_001)
    reviews_result = runner.invoke(
        app, ["dataset", "report", str(evidence), "--reviews", str(oversized_reviews)]
    )
    assert reviews_result.exit_code != 0
    assert "10 MB limit" in reviews_result.output


def test_symlink_and_nonregular_paths_are_refused(tmp_path: Path) -> None:
    real_evidence = tmp_path / "real.jsonl"
    _write_evidence(real_evidence)
    evidence_link = tmp_path / "evidence-link.jsonl"
    evidence_link.symlink_to(real_evidence)

    linked_evidence = runner.invoke(app, ["dataset", "report", str(evidence_link)])
    assert linked_evidence.exit_code != 0
    assert "cannot safely read evidence" in linked_evidence.output

    directory_result = runner.invoke(app, ["dataset", "report", str(tmp_path)])
    assert directory_result.exit_code != 0

    protected_target = tmp_path / "protected.txt"
    protected_target.write_text("do not change", encoding="utf-8")
    reviews_link = tmp_path / "reviews-link.jsonl"
    reviews_link.symlink_to(protected_target)
    linked_reviews = runner.invoke(
        app,
        [*_review_arguments(real_evidence), "--reviews", str(reviews_link)],
    )
    assert linked_reviews.exit_code != 0
    assert "cannot safely update review file" in linked_reviews.output
    assert protected_target.read_text(encoding="utf-8") == "do not change"


def test_report_and_review_make_no_model_or_network_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "evidence.jsonl"
    _write_evidence(evidence)

    def unexpected_call(*args: object, **kwargs: object) -> None:
        raise AssertionError("report/review attempted an external call")

    monkeypatch.setattr(dataset_command, "load_dataset_semantic_settings", unexpected_call)
    monkeypatch.setattr(dataset_runner, "create_semantic_model_deconstructor", unexpected_call)
    monkeypatch.setattr(dataset_command, "JsonHttpEnvironmentConnection", unexpected_call)

    report = runner.invoke(app, ["dataset", "report", str(evidence)])
    review = runner.invoke(app, _review_arguments(evidence))

    assert report.exit_code == 0, report.output
    assert review.exit_code == 0, review.output


@pytest.mark.skipif(sys.platform == "win32", reason="Unix locking implementation")
def test_unix_file_lock_helpers_use_shared_exclusive_and_unlock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        dataset_review.fcntl,
        "flock",
        lambda descriptor, mode: calls.append((descriptor, mode)),
    )

    dataset_review._lock_file(7, exclusive=False)
    dataset_review._lock_file(7, exclusive=True)
    dataset_review._unlock_file(7)

    assert calls == [
        (7, dataset_review.fcntl.LOCK_SH),
        (7, dataset_review.fcntl.LOCK_EX),
        (7, dataset_review.fcntl.LOCK_UN),
    ]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows locking implementation")
def test_windows_file_lock_helpers_use_byte_range_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locks: list[tuple[int, int, int]] = []
    seeks: list[tuple[int, int, int]] = []
    monkeypatch.setattr(
        dataset_review.msvcrt,
        "locking",
        lambda descriptor, mode, byte_count: locks.append((descriptor, mode, byte_count)),
    )
    monkeypatch.setattr(
        os,
        "lseek",
        lambda descriptor, offset, whence: seeks.append((descriptor, offset, whence)) or 0,
    )

    dataset_review._lock_file(7, exclusive=False)
    dataset_review._lock_file(7, exclusive=True)
    dataset_review._unlock_file(7)

    assert locks == [
        (7, dataset_review.msvcrt.LK_RLCK, 1),
        (7, dataset_review.msvcrt.LK_LOCK, 1),
        (7, dataset_review.msvcrt.LK_UNLCK, 1),
    ]
    assert seeks == [(7, 0, os.SEEK_SET)] * 3
