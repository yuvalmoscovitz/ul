from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner
from ul_cli import dataset_review
from ul_cli.main import app

runner = CliRunner()
FINDING_ID = f"ulf_v1_{'a' * 64}"


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
                "operator_id": "surface.disfluency_repeat",
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
        "technical_details": {"fixture": "quickstart-like stable 3/3"},
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
                    "repetition": 1,
                    "status": status,
                    "reason_code": "values_equal" if status == "satisfied" else "values_differ",
                    "left_pointer": "/final_amount",
                    "right_pointer": "/corrected_amount",
                    "resolved_values": {"left": left, "right": right},
                }
            ],
        }

    return {
        "interaction_id": "quickstart-payment",
        "suite_sha256": "b" * 64,
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
                "operator_id": "surface.disfluency_repeat",
                "rules": [rule("violated", 200, 100)],
            }
        ],
    }


def _write_evidence(path: Path, records: list[dict[str, Any]] | None = None) -> bytes:
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


def test_report_schema_1_4_shows_customer_invariants_separately(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.jsonl"
    record = _evidence_record()
    record["schema_version"] = "1.4.0"
    record["invariant_evaluation"] = _invariant_evaluation()
    _write_evidence(evidence, [record])

    report = runner.invoke(app, ["dataset", "report", str(evidence)])

    assert report.exit_code == 0, report.output
    normalized_output = " ".join(report.output.split())
    assert "Dataset finding report: 1 finding(s)" in normalized_output
    assert "Customer invariant evaluation" in normalized_output
    assert "Declared observation authority: committed_state_snapshot" in normalized_output
    assert "severity=critical; arm=original; status=satisfied" in normalized_output
    assert (
        "severity=critical; arm=variation (surface.disfluency_repeat); status=violated"
        in normalized_output
    )
    assert "Description: Final amount equals the corrected amount." in normalized_output
    assert "reason=one_or_more_trials_violated" in normalized_output
    assert "satisfied=0, violated=1, not_evaluable=0" in normalized_output
    assert "selected_values=" not in normalized_output
    assert "Customer rule violated against declared committed_state_snapshot." in normalized_output
    assert "agent wrong" not in normalized_output.casefold()


@pytest.mark.parametrize(
    ("status_value", "severity"),
    [
        ("confirmed", "critical"),
        ("expected", None),
        ("unsupported", None),
        ("inconclusive", None),
    ],
)
def test_all_review_statuses_are_recorded(
    tmp_path: Path, status_value: str, severity: str | None
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

    monkeypatch.setattr("ul_cli.dataset.OpenRouterDatasetSettings", unexpected_call)
    monkeypatch.setattr("ul_cli.dataset.OpenRouterSemanticDeconstructor", unexpected_call)
    monkeypatch.setattr("ul_cli.dataset.JsonHttpDatasetTarget", unexpected_call)

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
