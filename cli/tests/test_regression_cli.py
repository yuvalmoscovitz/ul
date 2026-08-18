from __future__ import annotations

import json
import os
import re
import stat
import threading
from collections.abc import Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest
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
from ul.dataset_invariants import (
    DatasetInvariantSuite,
    DatasetInvariantValueEqualsRuleEvaluation,
    DatasetInvariantValueEqualsTrialEvaluation,
    JsonValueEqualsLiteralInvariant,
    JsonValuesEqualInvariant,
)
from ul_cli import dataset_regression as regression_cli
from ul_cli import dataset_review
from ul_cli.main import app

runner = CliRunner()
FINDING_ID = f"ulf_v1_{'a' * 64}"
RULE_ID = "committed-invoice-matches-request"
SECOND_RULE_ID = "committed-amount-matches-request"
TEST_SECRET = "regression-test-secret-must-not-leak"
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def test_regression_save_reconstructs_extended_rule_definition() -> None:
    result = DatasetInvariantValueEqualsRuleEvaluation(
        rule_type="json_value_equals_literal",
        rule_id="approval-is-current",
        rule_version="1.0.0",
        description="The approval must be current.",
        severity="critical",
        status="violated",
        reason_code="one_or_more_trials_violated",
        value_pointer="/approval/version",
        literal=7,
        trials=(
            DatasetInvariantValueEqualsTrialEvaluation(
                repetition=1,
                status="violated",
                reason_code="value_differs_from_literal",
                value_pointer="/approval/version",
                resolved_values={"actual": 6},
            ),
        ),
    )

    assert regression_cli._invariant_rule_definition(result) == JsonValueEqualsLiteralInvariant(
        type="json_value_equals_literal",
        id="approval-is-current",
        version="1.0.0",
        description="The approval must be current.",
        severity="critical",
        value_pointer="/approval/version",
        literal=7,
    )


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
        "fields": {"invoice_reference": invoice_reference},
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


def _rule(status: str, committed: str, requested: str) -> dict[str, Any]:
    return {
        "rule_type": "json_values_equal",
        "rule_id": RULE_ID,
        "rule_version": "1.0.0",
        "description": "The committed invoice must equal the requested invoice.",
        "severity": "high",
        "status": status,
        "reason_code": (
            "all_trials_satisfied" if status == "satisfied" else "one_or_more_trials_violated"
        ),
        "trials": [
            {
                "repetition": repetition,
                "status": status,
                "reason_code": "values_equal" if status == "satisfied" else "values_differ",
                "left_pointer": "/invoice_reference",
                "right_pointer": "/requested_invoice_reference",
                "resolved_values": {"left": committed, "right": requested},
            }
            for repetition in (1, 2, 3)
        ],
    }


def _amount_rule(status: str) -> dict[str, Any]:
    rule = _rule(status, "12500" if status == "satisfied" else "12600", "12500")
    rule["rule_id"] = SECOND_RULE_ID
    rule["description"] = "The committed amount must equal the requested amount."
    for trial in cast(list[dict[str, Any]], rule["trials"]):
        trial["left_pointer"] = "/amount"
        trial["right_pointer"] = "/requested_amount"
    return rule


def _evidence_record() -> dict[str, Any]:
    original_effect = _effect("AC-100")
    changed_effect = _effect("AC-101")
    suite = DatasetInvariantSuite(
        schema_version="1.0.0",
        observation_source="target_output",
        observation_authority="agent_response",
        rules=(
            JsonValuesEqualInvariant(
                type="json_values_equal",
                id=RULE_ID,
                version="1.0.0",
                description="The committed invoice must equal the requested invoice.",
                severity="high",
                left_pointer="/invoice_reference",
                right_pointer="/requested_invoice_reference",
            ),
            JsonValuesEqualInvariant(
                type="json_values_equal",
                id=SECOND_RULE_ID,
                version="1.0.0",
                description="The committed amount must equal the requested amount.",
                severity="high",
                left_pointer="/amount",
                right_pointer="/requested_amount",
            ),
        ),
    )
    return {
        "schema_version": "1.4.0",
        "interaction_id": "quickstart-payment",
        "original_input": "Pay AC-100.",
        "execution_plan": {
            "repetitions": 3,
            "max_target_calls": 6,
            "dataset_planned_target_calls": 6,
        },
        "limitations": "UL reports observations for human review.",
        "current_baseline": {
            "status": "ORIGINAL REPLAY STABLE (3/3 OBSERVED)",
            "observations": _observations(original_effect),
            "inconclusive_reasons": [],
        },
        "cases": [
            {
                "operator_id": "surface.disfluency_repeat",
                "operator_version": "1.0.0",
                "augmented_input": "Pay pay AC-100.",
                "status": "REPEATABLE DIFFERENCE — REVIEW",
                "variation_accepted": True,
                "variation_rejection_reasons": [],
                "observations": _observations(changed_effect),
                "findings": [
                    {
                        "finding_id": FINDING_ID,
                        "category": "changed_grounded_effect_argument",
                        "grounded_field_names": ["invoice_reference"],
                        "severity": "unrated",
                        "review_status": "needs_review",
                        "summary": "The variation changed the committed invoice.",
                        "reference_effects": [original_effect],
                        "observed_effects": [changed_effect],
                    }
                ],
                "inconclusive_reasons": [],
            }
        ],
        "invariant_evaluation": {
            "interaction_id": "quickstart-payment",
            "suite_sha256": suite.sha256,
            "observation_source": "target_output",
            "observation_authority": "agent_response",
            "baseline": {
                "arm": "baseline",
                "operator_id": None,
                "rules": [
                    _rule("satisfied", "AC-100", "AC-100"),
                    _amount_rule("satisfied"),
                ],
            },
            "variations": [
                {
                    "arm": "variation",
                    "operator_id": "surface.disfluency_repeat",
                    "rules": [
                        _rule("violated", "AC-101", "AC-100"),
                        _amount_rule("violated"),
                    ],
                }
            ],
        },
        "technical_details": _technical_details(),
    }


def _technical_details() -> dict[str, Any]:
    source_frame = SemanticFrame(interaction_id="quickstart-payment", extractor_version="test")
    candidate = DatasetAugmentationCandidate(
        source_interaction_id="quickstart-payment",
        operator_id="surface.disfluency_repeat",
        operator_version="1.0.0",
        augmented_input="Pay pay AC-100.",
        expected_input_frame=source_frame,
        reparsed_input_frame=source_frame,
        passed=True,
    )

    def trial_set(arm: str, invoice: str, amount: str) -> DatasetEvaluationTrialSet:
        trials = tuple(
            DatasetEvaluationTrial(
                repetition=repetition,
                target_output=ObservedAgentOutput(
                    raw_output={
                        "invoice_reference": invoice,
                        "requested_invoice_reference": "AC-100",
                        "amount": amount,
                        "requested_amount": "12500",
                    }
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
            raw_observed_output={},
        ),
        augmentation=DatasetAugmentationResult(
            source_frames=(source_frame,), candidates=(candidate,)
        ),
        baseline=DatasetEvaluationBaseline(
            verdict="no_divergence",
            trial_set=trial_set("current_baseline", "AC-100", "12500"),
        ),
        cases=(
            DatasetEvaluationCase(
                candidate=candidate,
                verdict="no_divergence",
                trial_set=trial_set("surface.disfluency_repeat", "AC-101", "12600"),
            ),
        ),
    )
    return result.model_dump(mode="json")


def _write_evidence(path: Path) -> None:
    path.write_text(
        json.dumps(_evidence_record(), ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _write_invariant_only_evidence(path: Path) -> str:
    record = _evidence_record()
    cast(list[dict[str, Any]], record["cases"])[0]["findings"] = []
    path.write_text(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    indexed_findings = dataset_review._index_findings(dataset_review._load_evidence(path))
    rule_finding_ids = [
        finding.finding_id
        for finding in indexed_findings.values()
        if finding.kind == "customer_invariant_violation"
        and finding.variation_rule is not None
        and finding.variation_rule.rule_id == RULE_ID
    ]
    assert len(rule_finding_ids) == 1
    return rule_finding_ids[0]


class _ReplayServer(ThreadingHTTPServer):
    fixed: bool
    unavailable: bool
    requests: list[dict[str, Any]]
    authorization_headers: list[str | None]


class _ReplayHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        replay_server = cast(_ReplayServer, self.server)
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length))
        replay_server.requests.append(cast(dict[str, Any], payload))
        replay_server.authorization_headers.append(self.headers.get("X-Test-Token"))
        if replay_server.unavailable:
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        invoice_reference = "AC-100" if replay_server.fixed else "AC-101"
        response = json.dumps(
            {
                "result": {
                    "action": "payment_committed",
                    "payment_id": "pay-0001",
                    "invoice_reference": invoice_reference,
                    "requested_invoice_reference": "AC-100",
                    "amount": "12500" if replay_server.fixed else "12600",
                    "requested_amount": "12500",
                }
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def _running_server() -> Generator[tuple[_ReplayServer, str]]:
    server = _ReplayServer(("127.0.0.1", 0), _ReplayHandler)
    server.fixed = False
    server.unavailable = False
    server.requests = []
    server.authorization_headers = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address)
    try:
        yield server, f"http://{host}:{port}/execute"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert not thread.is_alive()


def _write_target_config(path: Path, endpoint: str) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "url": endpoint,
                "headers_from_env": {"X-Test-Token": "UL_REGRESSION_TEST_SECRET"},
                "request_json_template": {
                    "request": {"message": "{{input}}"},
                    "settings": {"mode": "sandbox"},
                },
                "response_json_pointer": "/result",
            }
        ),
        encoding="utf-8",
    )


def _confirm_finding(evidence: Path, finding_id: str = FINDING_ID) -> None:
    report = runner.invoke(app, ["dataset", "report", str(evidence)])
    assert report.exit_code == 0, report.output
    assert f"Finding {finding_id}" in report.output
    review = runner.invoke(
        app,
        [
            "dataset",
            "review",
            str(evidence),
            finding_id,
            "--status",
            "confirmed",
            "--severity",
            "high",
            "--reviewer",
            "payments-risk",
            "--reason",
            "The sandbox committed payment for a different invoice.",
        ],
    )
    assert review.exit_code == 0, review.output
    reviewed_report = runner.invoke(app, ["dataset", "report", str(evidence)])
    assert reviewed_report.exit_code == 0, reviewed_report.output
    assert "confirmed=1" in reviewed_report.output


def _save_arguments(
    evidence: Path,
    target_config: Path,
    case_path: Path,
    *,
    finding_id: str = FINDING_ID,
    rule_id: str | None = RULE_ID,
) -> list[str]:
    arguments = [
        "regression",
        "save",
        str(evidence),
        finding_id,
        "--target-config",
        str(target_config),
        "--output",
        str(case_path),
        "--confirm-versioned-input",
    ]
    if rule_id is not None:
        arguments[4:4] = ["--rule", rule_id]
    return arguments


def _replay_arguments(case_path: Path, target_config: Path, result_path: Path) -> list[str]:
    return [
        "regression",
        "replay",
        str(case_path),
        "--target-config",
        str(target_config),
        "--allow-target-network",
        "--confirm-isolated-sandbox",
        "--confirm-fresh-state",
        "--allow-insecure-http",
        "--max-target-calls",
        "3",
        "--output",
        str(result_path),
    ]


def _run_arguments(
    cases_path: Path,
    target_config: Path,
    result_path: Path,
    *,
    max_target_calls: int = 100,
) -> list[str]:
    return [
        "regression",
        "run",
        str(cases_path),
        "--target-config",
        str(target_config),
        "--allow-target-network",
        "--confirm-isolated-sandbox",
        "--confirm-fresh-state",
        "--allow-insecure-http",
        "--max-target-calls",
        str(max_target_calls),
        "--output",
        str(result_path),
    ]


def test_confirmed_finding_save_and_replay_real_loopback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "evidence.jsonl"
    case_path = tmp_path / "wrong-invoice.regression.json"
    defective_result_path = tmp_path / "defective-replay.json"
    fixed_result_path = tmp_path / "fixed-replay.json"
    _write_evidence(evidence)
    monkeypatch.setenv("UL_REGRESSION_TEST_SECRET", TEST_SECRET)
    monkeypatch.delenv("OPEN_ROUTER_API_KEY", raising=False)

    with _running_server() as (server, endpoint):
        target_config = tmp_path / "target.json"
        _write_target_config(target_config, endpoint)
        _confirm_finding(evidence)

        saved = runner.invoke(app, _save_arguments(evidence, target_config, case_path))
        assert saved.exit_code == 0, saved.output
        assert "Saved regression case ulrc_v1_" in saved.output
        assert "--max-target-calls 3" in saved.output
        assert "--allow-insecure-http" in saved.output
        assert "not verified as the discovery target" in saved.output
        assert case_path.exists()
        assert TEST_SECRET not in saved.output
        assert TEST_SECRET not in case_path.read_text(encoding="utf-8")

        defective = runner.invoke(
            app, _replay_arguments(case_path, target_config, defective_result_path)
        )
        assert defective.exit_code == 1, defective.output
        assert ": failed" in defective.output
        assert len(server.requests) == 3
        assert all(
            request
            == {
                "request": {"message": "Pay pay AC-100."},
                "settings": {"mode": "sandbox"},
            }
            for request in server.requests
        )
        assert server.authorization_headers == [TEST_SECRET] * 3

        server.fixed = True
        fixed = runner.invoke(app, _replay_arguments(case_path, target_config, fixed_result_path))
        assert fixed.exit_code == 0, fixed.output
        assert ": passed" in fixed.output
        assert len(server.requests) == 6
        assert server.requests[:3] == server.requests[3:]

    for artifact in (case_path, defective_result_path, fixed_result_path):
        serialized = artifact.read_text(encoding="utf-8")
        assert TEST_SECRET not in serialized
        if os.name != "nt":
            assert stat.S_IMODE(artifact.stat().st_mode) == 0o600
    defective_result = json.loads(defective_result_path.read_text(encoding="utf-8"))
    fixed_result = json.loads(fixed_result_path.read_text(encoding="utf-8"))
    assert defective_result["status"] == "failed"
    assert fixed_result["status"] == "passed"
    assert len(defective_result["executions"]) == len(fixed_result["executions"]) == 3


def test_invariant_violation_without_semantic_finding_saves_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "evidence.jsonl"
    case_path = tmp_path / "wrong-invoice.regression.json"
    defective_result_path = tmp_path / "defective-replay.json"
    fixed_result_path = tmp_path / "fixed-replay.json"
    invariant_finding_id = _write_invariant_only_evidence(evidence)
    monkeypatch.setenv("UL_REGRESSION_TEST_SECRET", TEST_SECRET)
    _confirm_finding(evidence, invariant_finding_id)

    with _running_server() as (server, endpoint):
        target_config = tmp_path / "target.json"
        _write_target_config(target_config, endpoint)
        saved = runner.invoke(
            app,
            _save_arguments(
                evidence,
                target_config,
                case_path,
                finding_id=invariant_finding_id,
                rule_id=None,
            ),
        )
        assert saved.exit_code == 0, saved.output
        saved_case = json.loads(case_path.read_text(encoding="utf-8"))
        assert saved_case["lineage"]["finding_id"] == invariant_finding_id
        assert [rule["id"] for rule in saved_case["invariant_suite"]["rules"]] == [RULE_ID]

        defective = runner.invoke(
            app, _replay_arguments(case_path, target_config, defective_result_path)
        )
        assert defective.exit_code == 1, defective.output
        assert len(server.requests) == 3

        server.fixed = True
        fixed = runner.invoke(app, _replay_arguments(case_path, target_config, fixed_result_path))
        assert fixed.exit_code == 0, fixed.output
        assert len(server.requests) == 6


def test_invariant_finding_rejects_unrelated_explicit_rule(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.jsonl"
    target_config = tmp_path / "target.json"
    case_path = tmp_path / "case.json"
    invariant_finding_id = _write_invariant_only_evidence(evidence)
    _write_target_config(target_config, "https://sandbox.example.test/execute")
    _confirm_finding(evidence, invariant_finding_id)

    result = runner.invoke(
        app,
        _save_arguments(
            evidence,
            target_config,
            case_path,
            finding_id=invariant_finding_id,
            rule_id=SECOND_RULE_ID,
        ),
    )

    assert result.exit_code == 2
    assert "automatically selects" in result.output
    assert not case_path.exists()


def test_semantic_finding_still_requires_an_explicit_rule(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.jsonl"
    target_config = tmp_path / "target.json"
    case_path = tmp_path / "case.json"
    _write_evidence(evidence)
    _write_target_config(target_config, "https://sandbox.example.test/execute")
    _confirm_finding(evidence)

    result = runner.invoke(
        app,
        _save_arguments(evidence, target_config, case_path, rule_id=None),
    )

    assert result.exit_code == 2
    normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", result.output).split())
    assert "semantic findings require at least one --rule" in normalized_output
    assert not case_path.exists()


def test_regression_run_monitors_saved_cases_against_current_black_box_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "evidence.jsonl"
    cases_directory = tmp_path / "regressions"
    cases_directory.mkdir()
    invoice_case_path = cases_directory / "z-invoice.json"
    amount_case_path = cases_directory / "a-amount.json"
    defective_result_path = tmp_path / "defective-run.json"
    fixed_result_path = tmp_path / "fixed-run.json"
    inconclusive_result_path = tmp_path / "inconclusive-run.json"
    _write_evidence(evidence)
    monkeypatch.setenv("UL_REGRESSION_TEST_SECRET", TEST_SECRET)

    with _running_server() as (server, endpoint):
        target_config = tmp_path / "target.json"
        _write_target_config(target_config, endpoint)
        _confirm_finding(evidence)
        invoice_saved = runner.invoke(
            app,
            _save_arguments(evidence, target_config, invoice_case_path),
        )
        amount_arguments = _save_arguments(evidence, target_config, amount_case_path)
        amount_arguments[amount_arguments.index(RULE_ID)] = SECOND_RULE_ID
        amount_saved = runner.invoke(app, amount_arguments)
        assert invoice_saved.exit_code == amount_saved.exit_code == 0

        expected_case_ids = [
            json.loads(amount_case_path.read_text(encoding="utf-8"))["case_id"],
            json.loads(invoice_case_path.read_text(encoding="utf-8"))["case_id"],
        ]
        defective = runner.invoke(
            app,
            _run_arguments(
                cases_directory,
                target_config,
                defective_result_path,
                max_target_calls=6,
            ),
        )
        assert defective.exit_code == 1, defective.output
        assert "passed=0, failed=2, inconclusive=0" in defective.output
        assert len(server.requests) == 6
        defective_result = json.loads(defective_result_path.read_text(encoding="utf-8"))
        assert defective_result["status"] == "failed"
        assert [case["label"] for case in defective_result["cases"]] == [
            "a-amount.json",
            "z-invoice.json",
        ]
        assert [case["result"]["case_id"] for case in defective_result["cases"]] == (
            expected_case_ids
        )
        assert "a-amount.json: failed" in defective.output
        assert f"{SECOND_RULE_ID} (high)" in defective.output

        server.fixed = True
        fixed = runner.invoke(
            app,
            _run_arguments(
                cases_directory,
                target_config,
                fixed_result_path,
                max_target_calls=6,
            ),
        )
        assert fixed.exit_code == 0, fixed.output
        assert "passed=2, failed=0, inconclusive=0" in fixed.output
        assert len(server.requests) == 12

        server.unavailable = True
        inconclusive = runner.invoke(
            app,
            _run_arguments(
                invoice_case_path,
                target_config,
                inconclusive_result_path,
                max_target_calls=3,
            ),
        )
        assert inconclusive.exit_code == 2, inconclusive.output
        assert "passed=0, failed=0, inconclusive=1" in inconclusive.output
        assert len(server.requests) == 15

    for artifact in (
        defective_result_path,
        fixed_result_path,
        inconclusive_result_path,
    ):
        serialized = artifact.read_text(encoding="utf-8")
        assert TEST_SECRET not in serialized
        if os.name != "nt":
            assert stat.S_IMODE(artifact.stat().st_mode) == 0o600
    assert TEST_SECRET not in defective.output + fixed.output + inconclusive.output


def test_regression_run_preflights_total_budget_before_secrets_output_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "evidence.jsonl"
    cases_directory = tmp_path / "regressions"
    cases_directory.mkdir()
    first_case_path = cases_directory / "first.json"
    second_case_path = cases_directory / "second.json"
    result_path = tmp_path / "run.json"
    _write_evidence(evidence)

    with _running_server() as (server, endpoint):
        target_config = tmp_path / "target.json"
        _write_target_config(target_config, endpoint)
        _confirm_finding(evidence)
        monkeypatch.setenv("UL_REGRESSION_TEST_SECRET", TEST_SECRET)
        first_saved = runner.invoke(
            app,
            _save_arguments(evidence, target_config, first_case_path),
        )
        second_arguments = _save_arguments(evidence, target_config, second_case_path)
        second_arguments[second_arguments.index(RULE_ID)] = SECOND_RULE_ID
        second_saved = runner.invoke(app, second_arguments)
        assert first_saved.exit_code == second_saved.exit_code == 0
        monkeypatch.delenv("UL_REGRESSION_TEST_SECRET")

        run = runner.invoke(
            app,
            _run_arguments(
                cases_directory,
                target_config,
                result_path,
                max_target_calls=5,
            ),
        )

    assert run.exit_code == 2
    assert "6" in run.output and "5" in run.output
    assert server.requests == []
    assert TEST_SECRET not in run.output
    assert not result_path.exists()


def test_save_requires_explicit_sensitive_input_confirmation_and_confirmed_review(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.jsonl"
    target_config = tmp_path / "target.json"
    case_path = tmp_path / "case.json"
    _write_evidence(evidence)
    _write_target_config(target_config, "https://sandbox.example.test/execute")

    no_review = runner.invoke(app, _save_arguments(evidence, target_config, case_path))
    assert no_review.exit_code == 2
    assert not case_path.exists()

    _confirm_finding(evidence)
    arguments_without_confirmation = _save_arguments(evidence, target_config, case_path)[:-1]
    no_confirmation = runner.invoke(app, arguments_without_confirmation)
    assert no_confirmation.exit_code == 2
    assert "exact" in no_confirmation.output.casefold()
    assert "sensitive" in no_confirmation.output.casefold()
    assert "customer-rule definitions" in no_confirmation.output.replace("\n", " ")
    assert not case_path.exists()

    help_result = runner.invoke(app, ["regression", "save", "--help"], env={"COLUMNS": "300"})
    assert "selected customer-rule definitions" in help_result.output


def test_save_rejects_stale_review_lineage_and_never_overwrites(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.jsonl"
    target_config = tmp_path / "target.json"
    case_path = tmp_path / "case.json"
    _write_evidence(evidence)
    _write_target_config(target_config, "https://sandbox.example.test/execute")
    _confirm_finding(evidence)
    changed_evidence = _evidence_record()
    changed_evidence["technical_details"] = {"fixture": "changed-after-review"}
    evidence.write_text(json.dumps(changed_evidence) + "\n", encoding="utf-8")

    stale = runner.invoke(app, _save_arguments(evidence, target_config, case_path))
    assert stale.exit_code == 2
    assert "review" in stale.output.casefold()
    assert not case_path.exists()

    collision_evidence = tmp_path / "collision-evidence.jsonl"
    collision_case = tmp_path / "collision-case.json"
    _write_evidence(collision_evidence)
    _confirm_finding(collision_evidence)
    collision_case.write_text("keep", encoding="utf-8")
    collision = runner.invoke(
        app, _save_arguments(collision_evidence, target_config, collision_case)
    )
    assert collision.exit_code == 2
    assert collision_case.read_text(encoding="utf-8") == "keep"


def test_save_canonicalizes_selected_rule_order(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.jsonl"
    target_config = tmp_path / "target.json"
    first_case = tmp_path / "first.json"
    second_case = tmp_path / "second.json"
    _write_evidence(evidence)
    _write_target_config(target_config, "https://sandbox.example.test/execute")
    _confirm_finding(evidence)

    first_arguments = _save_arguments(evidence, target_config, first_case)
    first_arguments[6:6] = ["--rule", SECOND_RULE_ID]
    second_arguments = _save_arguments(evidence, target_config, second_case)
    second_arguments[4:4] = ["--rule", SECOND_RULE_ID]
    first = runner.invoke(app, first_arguments)
    second = runner.invoke(app, second_arguments)

    assert first.exit_code == second.exit_code == 0
    assert first_case.read_bytes() == second_case.read_bytes()


def test_save_prints_insecure_http_for_case_insensitive_scheme(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.jsonl"
    target_config = tmp_path / "target.json"
    case_path = tmp_path / "case.json"
    _write_evidence(evidence)
    _write_target_config(target_config, "HTTP://127.0.0.1:8765/execute")
    _confirm_finding(evidence)

    saved = runner.invoke(app, _save_arguments(evidence, target_config, case_path))

    assert saved.exit_code == 0, saved.output
    assert "--allow-insecure-http" in saved.output


def test_replay_rejects_untrusted_or_tampered_inputs_before_target_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "evidence.jsonl"
    case_path = tmp_path / "case.json"
    _write_evidence(evidence)
    monkeypatch.setenv("UL_REGRESSION_TEST_SECRET", TEST_SECRET)

    with _running_server() as (server, endpoint):
        target_config = tmp_path / "target.json"
        wrong_target_config = tmp_path / "wrong-target.json"
        result_path = tmp_path / "result.json"
        _write_target_config(target_config, endpoint)
        _write_target_config(wrong_target_config, f"{endpoint}/different")
        _confirm_finding(evidence)
        saved = runner.invoke(app, _save_arguments(evidence, target_config, case_path))
        assert saved.exit_code == 0, saved.output

        mismatch = runner.invoke(
            app, _replay_arguments(case_path, wrong_target_config, result_path)
        )
        assert mismatch.exit_code == 2
        assert "digest" in mismatch.output.casefold() or "sha" in mismatch.output.casefold()
        assert server.requests == []
        assert not result_path.exists()

        case = json.loads(case_path.read_text(encoding="utf-8"))
        cast(dict[str, Any], case["variation"])["variation_input"] = "Transfer everything."
        case_path.write_text(json.dumps(case), encoding="utf-8")
        tampered = runner.invoke(app, _replay_arguments(case_path, target_config, result_path))
        assert tampered.exit_code == 2
        assert server.requests == []
        assert not result_path.exists()


def test_replay_enforces_target_call_budget_before_secret_resolution_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "evidence.jsonl"
    case_path = tmp_path / "case.json"
    target_config = tmp_path / "target.json"
    result_path = tmp_path / "result.json"
    _write_evidence(evidence)
    _write_target_config(target_config, "https://sandbox.example.test/execute")
    _confirm_finding(evidence)
    saved = runner.invoke(app, _save_arguments(evidence, target_config, case_path))
    assert saved.exit_code == 0, saved.output
    monkeypatch.delenv("UL_REGRESSION_TEST_SECRET", raising=False)
    arguments = _replay_arguments(case_path, target_config, result_path)
    budget_index = arguments.index("--max-target-calls") + 1
    arguments[budget_index] = "2"

    replay = runner.invoke(app, arguments)

    assert replay.exit_code == 2
    assert "3" in replay.output and "2" in replay.output
    assert "target" in replay.output.casefold() and "call" in replay.output.casefold()
    assert TEST_SECRET not in replay.output
    assert not result_path.exists()


@pytest.mark.parametrize(
    "missing_option",
    ["--allow-target-network", "--confirm-isolated-sandbox", "--confirm-fresh-state"],
)
def test_replay_requires_every_target_safety_confirmation(
    tmp_path: Path, missing_option: str
) -> None:
    evidence = tmp_path / "evidence.jsonl"
    case_path = tmp_path / "case.json"
    target_config = tmp_path / "target.json"
    result_path = tmp_path / "result.json"
    _write_evidence(evidence)
    _write_target_config(target_config, "http://127.0.0.1:9/execute")
    _confirm_finding(evidence)
    saved = runner.invoke(app, _save_arguments(evidence, target_config, case_path))
    assert saved.exit_code == 0, saved.output
    arguments = _replay_arguments(case_path, target_config, result_path)
    arguments.remove(missing_option)

    replay = runner.invoke(app, arguments)

    assert replay.exit_code == 2
    assert not result_path.exists()


@pytest.mark.parametrize("contents", ["not json", "{}", "[]"])
def test_save_and_replay_reject_malformed_files_without_tracebacks_or_outputs(
    tmp_path: Path, contents: str
) -> None:
    evidence = tmp_path / "evidence.jsonl"
    target_config = tmp_path / "target.json"
    case_path = tmp_path / "case.json"
    result_path = tmp_path / "result.json"
    evidence.write_text(contents, encoding="utf-8")
    _write_target_config(target_config, "https://sandbox.example.test/execute")

    saved = runner.invoke(app, _save_arguments(evidence, target_config, case_path))
    replayed = runner.invoke(app, _replay_arguments(evidence, target_config, result_path))

    assert saved.exit_code == 2
    assert replayed.exit_code == 2
    assert "traceback" not in (saved.output + replayed.output).casefold()
    assert not case_path.exists()
    assert not result_path.exists()


def test_replay_escapes_terminal_controls_from_untrusted_case_errors(tmp_path: Path) -> None:
    case_path = tmp_path / "case.json"
    target_config = tmp_path / "target.json"
    result_path = tmp_path / "result.json"
    case_path.write_text(
        json.dumps({"schema_version": "1.0.0", "\u001b[31mPWN": True}),
        encoding="utf-8",
    )
    _write_target_config(target_config, "https://sandbox.example.test/execute")

    replayed = runner.invoke(
        app,
        _replay_arguments(case_path, target_config, result_path),
        color=True,
    )

    assert replayed.exit_code == 2
    assert "\x1b[31mPWN" not in replayed.output
    assert "\\u001b[31mPWN" in replayed.output
    assert not result_path.exists()
