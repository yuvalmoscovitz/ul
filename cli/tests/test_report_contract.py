from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest
from pydantic import JsonValue, ValidationError
from typer.testing import CliRunner
from ul_cli import report as report_module
from ul_cli import report_contract as report_contract_module
from ul_cli.finding_reference import finding_public_reference, finding_reference_key_path
from ul_cli.main import app
from ul_cli.report_contract import (
    CapturedJson,
    CrossExaminationArm,
    DecisionReadyFinding,
    EvidenceArtifact,
    EvidencePointer,
    FindingCrossExamination,
    FindingDecisionReport,
    FindingEvidencePackage,
    FindingOccurrence,
    FindingPrivateReferences,
    FindingRepetition,
    LifecycleReceipt,
    ObservedDelta,
    ProbeChange,
    ProvenanceReceipt,
    ReceiptEvidenceValue,
    RedactionReceipt,
    RepetitionSummary,
    RunReceipt,
    RunReceiptContent,
    StateReceipt,
    UsageReceipt,
    VersionedReference,
    build_finding_decision,
    build_finding_decision_report,
    build_finding_occurrence,
    build_run_receipt,
    capture_json,
    parse_run_receipt,
    serialize_run_receipt,
)

_PRIVATE_CANARY = "customer@example.com:account-secret-canary"
_REFERENCE_KEY = b"0123456789abcdef0123456789abcdef"
_CAMPAIGN_ID = "private-campaign"
runner = CliRunner()


def test_report_module_imports_in_a_clean_process() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "from ul_cli import report"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _public_ref(namespace: str, *values: str, campaign_id: str = _CAMPAIGN_ID) -> str:
    return finding_public_reference(_REFERENCE_KEY, campaign_id, namespace, *values)


def _versioned_ref(namespace: str, identifier: str, version: str) -> VersionedReference:
    return VersionedReference(
        id=_public_ref(namespace, identifier),
        version=_public_ref(f"{namespace}-version", version),
    )


_RULE = _versioned_ref("rule", "private-rule", "1.0.0")


def _write_package_evidence(
    path: Path,
    *packages: FindingEvidencePackage,
    reference_key: bytes = _REFERENCE_KEY,
) -> None:
    path.write_text(
        "".join(package.model_dump_json() + "\n" for package in packages),
        encoding="utf-8",
    )
    key_path = finding_reference_key_path(path)
    key_path.write_text(
        reference_key.hex() + "\n2026-08-23T00:00:00+00:00\n",
        encoding="ascii",
    )
    if os.name != "nt":
        key_path.chmod(0o600)


def _rebind_occurrence(payload: dict[str, object]) -> None:
    occurrence = FindingOccurrence.model_validate_json(
        json.dumps(payload["occurrence"]),
        context={"building_occurrence": True},
    )
    values = occurrence.model_dump()
    values.pop("occurrence_id")
    payload["occurrence"] = build_finding_occurrence(**values).model_dump(mode="json")


def _pointer_id(value: str) -> str:
    return f"ulep_v1_{_sha(value)}"


def _pointer(
    label: str,
    kind: str,
    arm: Literal["source", "probe", "shared"],
    *,
    authority: str = "independent_observer",
    source_id: str = "observer",
) -> EvidencePointer:
    record_id = f"private-record:{label}"
    artifact = capture_json(_pointer_artifact_value(record_id, kind, arm))
    return EvidencePointer.model_validate(
        {
            "pointer_id": _pointer_id(label),
            "kind": kind,
            "artifact_sha256": artifact.sha256,
            "record_id": record_id,
            "json_pointer": "",
            "arm": arm,
            "authority": authority,
            "source_id": source_id,
        }
    )


def _pointer_artifact_value(record_id: str, kind: str, arm: str) -> JsonValue:
    return {
        "private": _PRIVATE_CANARY,
        "record_id": record_id,
        "kind": kind,
        "arm": arm,
    }


def _evidence(pointer: EvidencePointer, value: JsonValue) -> ReceiptEvidenceValue:
    del value
    assert pointer.record_id is not None
    return ReceiptEvidenceValue(
        evidence_pointer_id=pointer.pointer_id,
        value=capture_json(_pointer_artifact_value(pointer.record_id, pointer.kind, pointer.arm)),
    )


def _embedded_package(
    occurrence: FindingOccurrence,
    receipts: tuple[RunReceipt, ...],
) -> FindingEvidencePackage:
    artifacts: dict[str, EvidenceArtifact] = {}
    for receipt in receipts:
        for pointer in receipt.content.evidence_pointers:
            assert pointer.record_id is not None
            captured = capture_json(
                _pointer_artifact_value(pointer.record_id, pointer.kind, pointer.arm)
            )
            artifacts[captured.sha256] = EvidenceArtifact(
                artifact_sha256=captured.sha256,
                value=captured,
            )
    return FindingEvidencePackage(
        occurrence=occurrence,
        private_references=FindingPrivateReferences(
            campaign_id="private-campaign",
            case_id="private-case",
            source_interaction_id=(
                "private-source" if occurrence.source_interaction_ref is not None else None
            ),
            operator_id="private-operator",
            operator_version="1.0.0",
            rule_id="private-rule" if occurrence.violated_rule is not None else None,
            rule_version="1.0.0" if occurrence.violated_rule is not None else None,
        ),
        receipts=receipts,
        artifact_retention="embedded",
        artifacts=tuple(sorted(artifacts.values(), key=lambda artifact: artifact.artifact_sha256)),
    )


def _receipt(
    repetition: int,
    arm: Literal["source", "probe"],
    *,
    stateful: bool = False,
    rule_definition: bool = False,
    rule_violation: bool = False,
) -> RunReceipt:
    prefix = f"r{repetition}.{arm}"
    input_pointer = _pointer(f"{prefix}.input", "input", arm)
    response_pointer = _pointer(f"{prefix}.response", "response", arm)
    action_pointer = _pointer(f"{prefix}.action", "action", arm)
    lifecycle_pointer = _pointer(f"{prefix}.lifecycle", "lifecycle", arm)
    pointers = [input_pointer, response_pointer, action_pointer, lifecycle_pointer]
    historical_reference_pointer = None
    if arm == "source":
        historical_reference_pointer = _pointer(
            f"{prefix}.historical-reference",
            "response",
            "shared",
            authority="customer_declared",
            source_id="customer",
        )
        pointers.append(historical_reference_pointer)
    state_before = None
    state_after = None
    if stateful:
        state_before_pointer = _pointer(
            f"{prefix}.state.before",
            "state",
            arm,
            authority="environment_self_reported",
            source_id="environment",
        )
        state_after_pointer = _pointer(
            f"{prefix}.state.after",
            "state",
            arm,
            authority="environment_self_reported",
            source_id="environment",
        )
        pointers.extend((state_before_pointer, state_after_pointer))
        state_before = StateReceipt(evidence=_evidence(state_before_pointer, {"balance": 100}))
        state_after = StateReceipt(
            evidence=_evidence(
                state_after_pointer,
                {"balance": 200 if arm == "probe" else 100},
            )
        )
    if rule_definition:
        pointers.append(
            _pointer(
                f"{prefix}.rule.definition",
                "rule",
                "shared",
                authority="customer_declared",
                source_id="customer",
            )
        )
    if rule_violation:
        pointers.append(
            _pointer(
                f"{prefix}.rule.violation",
                "rule",
                "probe",
                authority="deterministic_evaluator",
                source_id="evaluator",
            )
        )
    provenance = [
        ProvenanceReceipt(role="model", id="agent-model", version="1.0.0"),
        ProvenanceReceipt(role="observer", id="observer", version="1.0.0"),
    ]
    if historical_reference_pointer is not None:
        provenance.append(ProvenanceReceipt(role="customer", id="customer"))
    if stateful:
        provenance.append(ProvenanceReceipt(role="environment", id="environment", version="1.0.0"))
    if rule_definition and historical_reference_pointer is None:
        provenance.append(ProvenanceReceipt(role="customer", id="customer"))
    if rule_violation:
        provenance.append(ProvenanceReceipt(role="evaluator", id="evaluator", version="1.0.0"))
    limitations = () if rule_violation else ("evaluator_provenance_unavailable",)
    content = RunReceiptContent(
        repetition=repetition,
        arm=arm,
        evidence_scope="response_and_state" if stateful else "response_only",
        input=_evidence(
            input_pointer,
            {"account": _PRIVATE_CANARY, "arm": arm, "repetition": repetition},
        ),
        historical_reference_response=(
            _evidence(historical_reference_pointer, {"status": "historical-reference"})
            if historical_reference_pointer is not None
            else None
        ),
        response=_evidence(response_pointer, {"status": "observed", "arm": arm}),
        state_before=state_before,
        state_after=state_after,
        lifecycle=(
            LifecycleReceipt(
                phase="execution",
                status="succeeded",
                evidence_pointer_ids=(lifecycle_pointer.pointer_id,),
            ),
        ),
        provenance=tuple(
            sorted(provenance, key=lambda item: (item.role, item.id, item.version or ""))
        ),
        usage=UsageReceipt(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cost=0.01,
            duration_ms=250.0,
        ),
        redaction=RedactionReceipt(
            policy_sha256=_sha("redaction-policy"),
            matched_value_count=1,
            redacted_value_count=0,
            retained_private_value_count=1,
        ),
        evidence_pointers=tuple(sorted(pointers, key=lambda pointer: pointer.pointer_id)),
        limitations=limitations,
        recorded_at=datetime(2026, 8, 23, repetition, tzinfo=UTC),
    )
    return build_run_receipt(content)


def _dataset_package() -> FindingEvidencePackage:
    receipts = tuple(
        sorted(
            (
                _receipt(1, "source"),
                _receipt(1, "probe"),
                _receipt(2, "source"),
                _receipt(2, "probe"),
            ),
            key=lambda receipt: receipt.receipt_id,
        )
    )
    pointers = {
        pointer.pointer_id: pointer
        for receipt in receipts
        for pointer in receipt.content.evidence_pointers
    }

    def ids(kind: str, arm: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                pointer_id
                for pointer_id, pointer in pointers.items()
                if pointer.kind == kind and pointer.arm == arm
            )
        )

    source_inputs = ids("input", "source")
    probe_inputs = ids("input", "probe")
    action_ids = tuple(sorted((*ids("action", "source"), *ids("action", "probe"))))
    response_ids = tuple(sorted((*ids("response", "source"), *ids("response", "probe"))))
    receipt_by_execution = {
        (receipt.content.repetition, receipt.content.arm): receipt for receipt in receipts
    }
    repetitions = tuple(
        FindingRepetition(
            repetition=repetition,
            outcome="finding_observed",
            source_receipt_id=receipt_by_execution[(repetition, "source")].receipt_id,
            probe_receipt_id=receipt_by_execution[(repetition, "probe")].receipt_id,
            evidence_pointer_ids=tuple(
                sorted(
                    pointer_id
                    for pointer_id, pointer in pointers.items()
                    if pointer.kind in {"action", "response"}
                    and pointer.record_id is not None
                    and f"r{repetition}." in pointer.record_id
                )
            ),
        )
        for repetition in (1, 2)
    )
    occurrence = build_finding_occurrence(
        kind="behavior_difference",
        category="changed_grounded_effect_argument",
        campaign_ref=_public_ref("campaign", _CAMPAIGN_ID),
        source_interaction_ref=_public_ref("source-interaction", "private-source"),
        fixture=_versioned_ref("fixture", "accounts-payable", "1.0.0"),
        case_ref=_public_ref("case", "private-case"),
        operator=_versioned_ref("operator", "private-operator", "1.0.0"),
        bundle=_versioned_ref("bundle", "surface-noise", "1.0.0"),
        probe_change=ProbeChange(
            kind="input",
            source_descriptor="recorded_input",
            probe_descriptor="augmented_input",
            source_evidence_pointer_ids=source_inputs,
            probe_evidence_pointer_ids=probe_inputs,
        ),
        cross_examination=FindingCrossExamination(
            historical_reference=CrossExaminationArm(
                role="historical_reference",
                response_evidence_pointer_ids=ids("response", "shared"),
                requested_repetitions=0,
                observed_repetitions=0,
                inconclusive_repetitions=0,
                stability="not_applicable",
            ),
            current_baseline=CrossExaminationArm(
                role="current_baseline",
                response_evidence_pointer_ids=ids("response", "source"),
                requested_repetitions=2,
                observed_repetitions=2,
                inconclusive_repetitions=0,
                stability="stable",
            ),
            variation=CrossExaminationArm(
                role="variation",
                response_evidence_pointer_ids=ids("response", "probe"),
                requested_repetitions=2,
                observed_repetitions=2,
                inconclusive_repetitions=0,
                stability="stable",
            ),
            augmentation_relation=_versioned_ref("input.surface.disfluency_repeat", "1.0.0"),
            baseline_drift="not_observed",
            augmentation_sensitivity="observed",
            intrinsic_instability="not_observed",
            material_delta_evidence_pointer_ids=action_ids,
            evidence_level="response_observed",
            limitations=(
                "causality_not_established",
                "correctness_not_verified",
                "historical_reference_not_an_oracle",
            ),
        ),
        observed_deltas=(
            ObservedDelta(
                kind="action",
                change="changed",
                subject_ref=_public_ref("subject", "payment.invoice_reference"),
                source_state="observed",
                probe_state="observed",
                evidence_pointer_ids=action_ids,
            ),
            ObservedDelta(
                kind="response",
                change="changed",
                subject_ref=_public_ref("subject", "agent-response"),
                source_state="observed",
                probe_state="observed",
                evidence_pointer_ids=response_ids,
            ),
        ),
        evidence_pointer_ids=tuple(
            sorted(
                {
                    *source_inputs,
                    *probe_inputs,
                    *action_ids,
                    *response_ids,
                    *ids("response", "shared"),
                }
            )
        ),
        repetitions=repetitions,
        repetition_summary=RepetitionSummary(
            requested=2,
            conclusive=2,
            observed=2,
            inconclusive=0,
            stability="stable",
            reproducibility="reproduced",
        ),
        required_capabilities=("response_observation",),
        limitations=("correctness_not_verified", "production_prevalence_not_measured"),
        next_action="review_dataset_finding",
    )
    return _embedded_package(occurrence, receipts)


def _stateful_package() -> FindingEvidencePackage:
    receipts = tuple(
        sorted(
            (
                _receipt(1, "source", stateful=True, rule_definition=True),
                _receipt(1, "probe", stateful=True, rule_violation=True),
                _receipt(2, "source", stateful=True),
                _receipt(2, "probe", stateful=True, rule_violation=True),
            ),
            key=lambda receipt: receipt.receipt_id,
        )
    )
    pointers = {
        pointer.pointer_id: pointer
        for receipt in receipts
        for pointer in receipt.content.evidence_pointers
    }

    def ids(kind: str, arm: str | None = None) -> tuple[str, ...]:
        return tuple(
            sorted(
                pointer_id
                for pointer_id, pointer in pointers.items()
                if pointer.kind == kind and (arm is None or pointer.arm == arm)
            )
        )

    source_inputs = ids("input", "source")
    probe_inputs = ids("input", "probe")
    state_ids = ids("state")
    rule_definition_ids = tuple(
        sorted(
            pointer_id
            for pointer_id, pointer in pointers.items()
            if pointer.kind == "rule" and pointer.authority == "customer_declared"
        )
    )
    rule_violation_ids = tuple(
        sorted(
            pointer_id
            for pointer_id, pointer in pointers.items()
            if pointer.kind == "rule" and pointer.authority == "deterministic_evaluator"
        )
    )
    receipt_by_execution = {
        (receipt.content.repetition, receipt.content.arm): receipt for receipt in receipts
    }
    repetitions = tuple(
        FindingRepetition(
            repetition=repetition,
            outcome="finding_observed",
            source_receipt_id=receipt_by_execution[(repetition, "source")].receipt_id,
            probe_receipt_id=receipt_by_execution[(repetition, "probe")].receipt_id,
            evidence_pointer_ids=tuple(
                sorted(
                    pointer_id
                    for pointer_id, pointer in pointers.items()
                    if pointer.kind in {"rule", "state"}
                    and pointer.record_id is not None
                    and f"r{repetition}." in pointer.record_id
                )
            ),
        )
        for repetition in (1, 2)
    )
    occurrence = build_finding_occurrence(
        kind="customer_invariant_violation",
        category="customer_invariant_violation",
        campaign_ref=_public_ref("campaign", _CAMPAIGN_ID),
        fixture=_versioned_ref("fixture", "accounts-payable", "1.0.0"),
        case_ref=_public_ref("case", "private-case"),
        operator=_versioned_ref(
            "operator",
            "private-operator",
            "1.0.0",
        ),
        probe_change=ProbeChange(
            kind="turn_sequence",
            source_descriptor="baseline_turn_sequence",
            probe_descriptor="augmented_turn_sequence",
            source_evidence_pointer_ids=source_inputs,
            probe_evidence_pointer_ids=probe_inputs,
        ),
        observed_deltas=(
            ObservedDelta(
                kind="rule",
                change="violated",
                subject_ref=_public_ref("subject", "customer-rule"),
                rule=_RULE,
                source_state="satisfied",
                probe_state="violated",
                evidence_pointer_ids=rule_violation_ids,
            ),
            ObservedDelta(
                kind="state",
                change="changed",
                subject_ref=_public_ref("subject", "final-amount"),
                source_state="observed",
                probe_state="observed",
                evidence_pointer_ids=state_ids,
            ),
        ),
        violated_rule=_RULE,
        rule_definition_evidence_pointer_ids=rule_definition_ids,
        evidence_pointer_ids=tuple(
            sorted(
                {
                    *source_inputs,
                    *probe_inputs,
                    *state_ids,
                    *rule_definition_ids,
                    *rule_violation_ids,
                }
            )
        ),
        repetitions=repetitions,
        repetition_summary=RepetitionSummary(
            requested=2,
            conclusive=2,
            observed=2,
            violated=2,
            inconclusive=0,
            stability="stable",
            reproducibility="reproduced",
        ),
        required_capabilities=("conversation_replay", "state_observation"),
        limitations=("production_prevalence_not_measured",),
        next_action="inspect_stateful_evidence",
    )
    return _embedded_package(occurrence, receipts)


def test_dataset_and_stateful_workflows_share_an_auditable_package_contract() -> None:
    for package in (_dataset_package(), _stateful_package()):
        round_trip = FindingEvidencePackage.model_validate_json(package.model_dump_json())
        assert round_trip == package
        assert package.schema_version == "1.1.0"
        assert len(package.occurrence.repetitions) == 2
        assert all(item.evidence_pointer_ids for item in package.occurrence.repetitions)


@pytest.mark.parametrize(("artifact_value", "receipt_value"), ((True, 1), (1, 1.0)))
def test_receipt_artifact_binding_preserves_exact_json_types(
    artifact_value: JsonValue,
    receipt_value: JsonValue,
) -> None:
    evidence = ReceiptEvidenceValue(
        evidence_pointer_id=_pointer_id("typed-value"),
        value=capture_json(receipt_value),
    )

    assert not report_contract_module._receipt_value_matches_artifact(artifact_value, evidence)


@pytest.mark.parametrize(
    ("package", "classification", "workflow", "evidence_level"),
    (
        (_dataset_package(), "observed_variance", "dataset_review", "response_observed"),
        (
            _stateful_package(),
            "customer_rule_violation",
            "external_review_required",
            "customer_rule_evaluated",
        ),
    ),
)
def test_dataset_and_stateful_packages_share_decision_ready_explanations(
    package: FindingEvidencePackage,
    classification: str,
    workflow: str,
    evidence_level: str,
) -> None:
    finding = build_finding_decision(package)
    known_pointer_ids = {
        pointer.pointer_id
        for receipt in package.receipts
        for pointer in receipt.content.evidence_pointers
    }

    assert isinstance(finding, DecisionReadyFinding)
    assert finding.classification == classification
    assert finding.review_workflow == workflow
    assert finding.evidence_level == evidence_level
    assert finding.campaign_ref == package.occurrence.campaign_ref
    assert finding.case_ref == package.occurrence.case_ref
    assert finding.operator == package.occurrence.operator
    assert finding.probe_change_kind == package.occurrence.probe_change.kind
    assert finding.violated_rule == package.occurrence.violated_rule
    assert tuple(claim.kind for claim in finding.claims) == (
        "tested_change",
        "agent_behavior",
        "observed_consequence",
        "flag_reason",
    )
    assert all(claim.evidence_pointer_ids for claim in finding.claims)
    assert all(set(claim.evidence_pointer_ids) <= known_pointer_ids for claim in finding.claims)
    category_pointer_ids = package.occurrence.observed_deltas[0].evidence_pointer_ids
    assert finding.claims[1].evidence_pointer_ids == category_pointer_ids
    assert finding.claims[2].evidence_pointer_ids == category_pointer_ids
    assert finding.receipt_ids == tuple(receipt.receipt_id for receipt in package.receipts)
    assert _PRIVATE_CANARY not in finding.model_dump_json()
    assert "private-record" not in finding.model_dump_json()

    report = build_finding_decision_report((package,))
    assert FindingDecisionReport.model_validate_json(report.model_dump_json()) == report


def test_finding_package_report_human_json_and_private_receipt_share_one_contract(
    tmp_path: Path,
) -> None:
    package = _dataset_package()
    evidence = tmp_path / "dataset-evidence.jsonl.findings.jsonl"
    _write_package_evidence(evidence, package)
    expected = build_finding_decision_report((package,))

    human = runner.invoke(app, ["report", str(evidence)])
    json_result = runner.invoke(app, ["report", str(evidence), "--json"])

    assert human.exit_code == 1, human.output
    assert json_result.exit_code == 1, json_result.output
    assert json.loads(json_result.output) == expected.model_dump(mode="json")
    for claim in expected.findings[0].claims:
        assert claim.summary in human.output
        assert all(pointer_id in human.output for pointer_id in claim.evidence_pointer_ids)
    assert expected.findings[0].next_action_summary in human.output
    assert expected.findings[0].case_ref in human.output
    assert expected.findings[0].operator.id in human.output
    assert expected.findings[0].operator.version in human.output
    assert _PRIVATE_CANARY not in human.output
    assert _PRIVATE_CANARY not in json_result.output
    assert "private-record" not in human.output
    assert "json_pointer" not in json_result.output

    private = runner.invoke(
        app,
        [
            "report",
            str(evidence),
            "--show-sensitive-values",
            "--finding",
            package.occurrence.occurrence_id,
        ],
    )
    assert private.exit_code == 1, private.output
    assert "WARNING: showing private normalized receipts" in private.output
    assert "Disclosure receipt:" in private.output
    assert _PRIVATE_CANARY in private.output
    assert "private-operator" in private.output
    assert all(receipt.receipt_id in private.output for receipt in package.receipts)


@pytest.mark.parametrize(
    ("package", "field"),
    (
        *(
            (_dataset_package(), field)
            for field in (
                "campaign_id",
                "case_id",
                "source_interaction_id",
                "operator_id",
                "operator_version",
            )
        ),
        *(
            (_stateful_package(), field)
            for field in (
                "campaign_id",
                "case_id",
                "operator_id",
                "operator_version",
                "rule_id",
                "rule_version",
            )
        ),
    ),
)
def test_finding_package_report_rejects_tampered_private_reference_resolution(
    tmp_path: Path,
    package: FindingEvidencePackage,
    field: str,
) -> None:
    private_values = package.private_references.model_dump()
    private_values[field] = f"tampered-{field}"
    tampered_package = package.model_copy(
        update={"private_references": FindingPrivateReferences.model_validate(private_values)}
    )
    evidence = tmp_path / f"tampered-{field}.findings.jsonl"
    _write_package_evidence(evidence, tampered_package)

    result = runner.invoke(app, ["report", str(evidence), "--json"])

    assert result.exit_code == 2
    assert "finding package evidence cannot be" in result.output
    assert "summarized" in result.output
    assert "safely" in result.output


def test_private_receipt_disclosure_fails_before_partial_output_when_over_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _dataset_package()
    evidence = tmp_path / "dataset-evidence.jsonl.findings.jsonl"
    _write_package_evidence(evidence, package)
    monkeypatch.setattr(report_module, "_MAXIMUM_PRIVATE_RECEIPT_BYTES", 100)

    result = runner.invoke(
        app,
        [
            "report",
            str(evidence),
            "--show-sensitive-values",
            "--finding",
            package.occurrence.occurrence_id,
        ],
    )

    assert result.exit_code == 2
    assert "disclosure cap" in result.output
    assert "WARNING: showing private" not in result.output
    assert _PRIVATE_CANARY not in result.output


def test_stateful_finding_package_uses_same_safe_offline_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _stateful_package()
    evidence = tmp_path / "stateful-evidence.json.findings.jsonl"
    _write_package_evidence(evidence, package)

    def unexpected_primary_report(*args: object, **kwargs: object) -> None:
        raise AssertionError("finding package report attempted a primary evidence workflow")

    monkeypatch.setattr(report_module, "load_unified_report", unexpected_primary_report)
    result = runner.invoke(app, ["report", str(evidence)])

    assert result.exit_code == 1, result.output
    assert "Classification: customer rule violation" in result.output
    assert "Evidence scope: response and state" in result.output
    assert "workflow=external review required" in result.output
    assert "Inspect the private normalized receipt" in result.output
    assert "Resolve private references and receipt" in result.output
    assert package.occurrence.violated_rule is not None
    assert package.occurrence.violated_rule.id in result.output
    assert package.occurrence.violated_rule.version in result.output
    assert _PRIVATE_CANARY not in result.output
    assert package.private_references.rule_id is not None

    private = runner.invoke(
        app,
        [
            "report",
            str(evidence),
            "--show-sensitive-values",
            "--finding",
            package.occurrence.occurrence_id,
        ],
    )
    assert private.exit_code == 1, private.output
    assert package.private_references.operator_id in private.output
    assert package.private_references.rule_id in private.output


def test_decision_report_supports_mixed_evidence_scopes_in_one_campaign() -> None:
    report = build_finding_decision_report((_dataset_package(), _stateful_package()))

    assert report.evidence_scope == "mixed"
    assert {finding.evidence_scope for finding in report.findings} == {
        "response_only",
        "response_and_state",
    }


def test_decision_report_rejects_packages_from_different_campaigns() -> None:
    payload = _stateful_package().model_dump(mode="json")
    payload["occurrence"]["campaign_ref"] = _public_ref(
        "campaign", "another-campaign", campaign_id="another-campaign"
    )
    payload["private_references"]["campaign_id"] = "another-campaign"
    _rebind_occurrence(payload)
    other_campaign = FindingEvidencePackage.model_validate_json(json.dumps(payload))

    with pytest.raises(ValueError, match="one campaign"):
        build_finding_decision_report((_dataset_package(), other_campaign))


def test_finding_package_report_detects_contract_without_filename_suffix(tmp_path: Path) -> None:
    package = _dataset_package()
    evidence = tmp_path / "renamed-evidence.jsonl"
    _write_package_evidence(evidence, package)

    result = runner.invoke(app, ["report", str(evidence), "--json"])

    assert result.exit_code == 1, result.output
    assert json.loads(result.output) == build_finding_decision_report((package,)).model_dump(
        mode="json"
    )
    assert "Primary review queue" not in runner.invoke(app, ["report", str(evidence)]).output


def test_finding_package_report_safely_rejects_excessive_json_depth(tmp_path: Path) -> None:
    evidence = tmp_path / "deep.findings.jsonl"
    evidence.write_text('{"occurrence":' + "[" * 2_000 + "0" + "]" * 2_000 + "}\n")

    result = runner.invoke(app, ["report", str(evidence)])

    assert result.exit_code == 2
    normalized_output = " ".join(result.output.split())
    assert "cannot be summarized" in normalized_output
    assert "safely" in normalized_output


def test_finding_package_report_rejects_primary_review_sidecar_option(tmp_path: Path) -> None:
    package = _dataset_package()
    evidence = tmp_path / "dataset-evidence.jsonl.findings.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    _write_package_evidence(evidence, package)
    reviews.write_text("", encoding="utf-8")

    result = runner.invoke(
        app,
        ["report", str(evidence), "--reviews", str(reviews)],
    )

    assert result.exit_code == 2
    normalized_output = " ".join(result.output.split())
    assert "available only for primary dataset" in normalized_output
    assert "evidence" in normalized_output


def test_decision_ready_report_rejects_unresolved_external_artifacts(tmp_path: Path) -> None:
    package = _dataset_package().model_copy(
        update={"artifact_retention": "external", "artifacts": ()}
    )
    evidence = tmp_path / "external.findings.jsonl"
    _write_package_evidence(evidence, package)

    result = runner.invoke(app, ["report", str(evidence), "--json"])

    assert result.exit_code == 2
    normalized_output = " ".join(result.output.split())
    assert "cannot be summarized" in normalized_output
    assert "safely" in normalized_output


def test_finding_package_report_rejects_duplicate_keys_and_symlinks(tmp_path: Path) -> None:
    package = _stateful_package()
    valid = package.model_dump_json()
    duplicate = tmp_path / "duplicate.findings.jsonl"
    duplicate.write_text(
        valid.replace(
            '"schema_version":"1.1.0"',
            '"schema_version":"1.1.0","schema_version":"1.1.0"',
            1,
        )
        + "\n",
        encoding="utf-8",
    )
    duplicate_result = runner.invoke(app, ["report", str(duplicate), "--json"])
    assert duplicate_result.exit_code == 2
    normalized_duplicate_output = " ".join(duplicate_result.output.split())
    assert "finding package evidence cannot be summarized" in normalized_duplicate_output
    assert "safely" in normalized_duplicate_output

    if hasattr(os, "symlink"):
        real = tmp_path / "real.findings.jsonl"
        real.write_text(valid + "\n", encoding="utf-8")
        linked = tmp_path / "linked.findings.jsonl"
        linked.symlink_to(real)
        linked_result = runner.invoke(app, ["report", str(linked)])
        assert linked_result.exit_code == 2
        normalized_linked_output = " ".join(linked_result.output.split())
        assert "cannot safely read finding packages" in normalized_linked_output


def test_public_occurrence_contains_only_privacy_safe_references() -> None:
    package = _dataset_package()
    public_json = package.occurrence.model_dump_json()

    assert _PRIVATE_CANARY in serialize_run_receipt(package.receipts[0])
    assert _PRIVATE_CANARY not in public_json
    assert "private-record" not in public_json
    assert "json_pointer" not in public_json
    assert "source_id" not in public_json

    payload = package.occurrence.model_dump(mode="json")
    payload["case_ref"] = "acct-secret-canary"
    with pytest.raises(ValidationError):
        FindingOccurrence.model_validate_json(json.dumps(payload))

    payload = package.occurrence.model_dump(mode="json")
    payload["operator"]["id"] = "patient_alice_secret"
    with pytest.raises(ValidationError):
        FindingOccurrence.model_validate_json(json.dumps(payload))


def test_repetition_summary_is_derived_from_exact_repetition_evidence() -> None:
    payload = _dataset_package().occurrence.model_dump(mode="json")
    payload["repetition_summary"]["requested"] = 100

    with pytest.raises(ValidationError, match="must match exact repetition evidence"):
        FindingOccurrence.model_validate_json(json.dumps(payload))


def test_package_binds_each_repetition_to_its_receipts() -> None:
    package = _dataset_package()
    payload = package.model_dump(mode="json")
    for repetition in payload["occurrence"]["repetitions"]:
        repetition["source_receipt_id"], repetition["probe_receipt_id"] = (
            repetition["probe_receipt_id"],
            repetition["source_receipt_id"],
        )
    _rebind_occurrence(payload)

    with pytest.raises(
        ValidationError,
        match=r"declared arm and repetition",
    ):
        FindingEvidencePackage.model_validate_json(json.dumps(payload))


def test_observed_repetitions_require_category_evidence_from_each_execution() -> None:
    package = _dataset_package()
    payload = package.model_dump(mode="json")
    pointers = {
        pointer.pointer_id: pointer
        for receipt in package.receipts
        for pointer in receipt.content.evidence_pointers
    }
    for repetition in payload["occurrence"]["repetitions"]:
        repetition_number = repetition["repetition"]
        repetition["evidence_pointer_ids"] = sorted(
            pointer_id
            for pointer_id, pointer in pointers.items()
            if pointer.kind == "input"
            and pointer.arm == "probe"
            and f"r{repetition_number}." in (pointer.record_id or "")
        )
    _rebind_occurrence(payload)

    with pytest.raises(ValidationError, match="category evidence from that execution"):
        FindingEvidencePackage.model_validate_json(json.dumps(payload))


def test_non_observed_repetitions_require_typed_counterevidence() -> None:
    package = _dataset_package()
    payload = package.model_dump(mode="json")
    pointers = {
        pointer.pointer_id: pointer
        for receipt in package.receipts
        for pointer in receipt.content.evidence_pointers
    }
    repetition = payload["occurrence"]["repetitions"][1]
    repetition["outcome"] = "finding_not_observed"
    repetition["evidence_pointer_ids"] = sorted(
        pointer_id
        for pointer_id, pointer in pointers.items()
        if pointer.kind == "input" and pointer.arm == "probe" and "r2." in (pointer.record_id or "")
    )
    payload["occurrence"]["repetition_summary"].update(
        observed=1,
        stability="unstable",
        reproducibility="intermittent",
    )
    _rebind_occurrence(payload)

    with pytest.raises(ValidationError, match="conclusive repetition requires category evidence"):
        FindingEvidencePackage.model_validate_json(json.dumps(payload))


def test_rule_violation_requires_customer_definition_and_evaluator_evidence() -> None:
    package = _stateful_package()
    payload = package.model_dump(mode="json")
    violation_pointer_id = package.occurrence.observed_deltas[0].evidence_pointer_ids[0]
    for receipt in payload["receipts"]:
        for pointer in receipt["content"]["evidence_pointers"]:
            if pointer["pointer_id"] == violation_pointer_id:
                pointer["authority"] = "source_self_reported"
                pointer["source_id"] = "target"
                receipt["content"]["provenance"].append(
                    {"role": "target", "id": "target", "version": None, "config_sha256": None}
                )
                receipt["content"]["provenance"] = sorted(
                    receipt["content"]["provenance"],
                    key=lambda item: (item["role"], item["id"], item["version"] or ""),
                )
                content = RunReceiptContent.model_validate_json(json.dumps(receipt["content"]))
                replacement = build_run_receipt(content)
                old_receipt_id = receipt["receipt_id"]
                receipt.update(replacement.model_dump(mode="json"))
                for repetition in payload["occurrence"]["repetitions"]:
                    if repetition["probe_receipt_id"] == old_receipt_id:
                        repetition["probe_receipt_id"] = replacement.receipt_id
    payload["receipts"] = sorted(payload["receipts"], key=lambda item: item["receipt_id"])
    _rebind_occurrence(payload)

    with pytest.raises(ValidationError, match="rule violations require evaluator"):
        FindingEvidencePackage.model_validate_json(json.dumps(payload))


def test_behavior_findings_reject_rule_violation_deltas() -> None:
    occurrence = _dataset_package().occurrence
    values = occurrence.model_dump()
    values.pop("occurrence_id")
    values["observed_deltas"] = (
        *occurrence.observed_deltas,
        ObservedDelta(
            kind="rule",
            change="violated",
            subject_ref=_public_ref("subject", "customer-rule"),
            rule=_RULE,
            source_state="satisfied",
            probe_state="violated",
            evidence_pointer_ids=occurrence.observed_deltas[0].evidence_pointer_ids,
        ),
    )
    values["evidence_pointer_ids"] = tuple(
        sorted(
            {
                *occurrence.evidence_pointer_ids,
                *values["observed_deltas"][-1].evidence_pointer_ids,
            }
        )
    )

    with pytest.raises(ValidationError, match="cannot contain rule-violation deltas"):
        build_finding_occurrence(**values)


def test_occurrence_id_is_bound_to_canonical_claims() -> None:
    payload = _dataset_package().occurrence.model_dump(mode="json")
    payload["observed_deltas"][0]["subject_ref"] = _public_ref("subject", "different-action")

    with pytest.raises(ValidationError, match="ID must match its canonical claims"):
        FindingOccurrence.model_validate_json(json.dumps(payload))


def test_receipt_values_are_finite_immutable_and_content_addressed() -> None:
    with pytest.raises(ValueError):
        capture_json({"invalid": float("nan")})

    captured = capture_json({"b": 2, "a": 1})
    assert captured == capture_json({"a": 1, "b": 2})
    with pytest.raises(ValidationError):
        CapturedJson(canonical_json='{"a":1}', sha256="0" * 64)
    with pytest.raises(ValidationError):
        captured.canonical_json = "null"

    receipt = _receipt(1, "source")
    payload = receipt.model_dump(mode="json")
    payload["content"]["recorded_at"] = "2026-08-24T00:00:00Z"
    with pytest.raises(ValidationError, match="ID must match"):
        RunReceipt.model_validate_json(json.dumps(payload))


def test_receipt_serialization_is_canonical_and_round_trips() -> None:
    first = _receipt(1, "source")
    serialized = serialize_run_receipt(first)

    assert serialized == serialize_run_receipt(parse_run_receipt(serialized))
    assert serialized == serialize_run_receipt(first)
    assert len(serialized.encode()) < 1_000_000


def test_receipt_values_must_point_to_matching_execution_evidence() -> None:
    receipt = _receipt(1, "source")
    payload = receipt.content.model_dump(mode="json")
    action_pointer_id = next(
        pointer.pointer_id
        for pointer in receipt.content.evidence_pointers
        if pointer.kind == "action"
    )
    payload["input"]["evidence_pointer_id"] = action_pointer_id

    with pytest.raises(ValidationError, match="incompatible evidence"):
        RunReceiptContent.model_validate_json(json.dumps(payload))


def test_missing_provenance_requires_an_explicit_limitation() -> None:
    receipt = _receipt(1, "source")
    payload = receipt.content.model_dump(mode="json")
    payload["provenance"] = [item for item in payload["provenance"] if item["role"] != "model"]

    with pytest.raises(ValidationError, match="model provenance"):
        RunReceiptContent.model_validate_json(json.dumps(payload))
