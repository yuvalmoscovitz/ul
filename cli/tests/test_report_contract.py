from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from ul_cli.report_contract import (
    EvidencePointer,
    FindingOccurrence,
    LifecycleReceipt,
    ObservedDelta,
    ProbeChange,
    ProvenanceReceipt,
    RedactionReceipt,
    RepetitionEvidence,
    RunReceipt,
    StateReceipt,
    ToolExchangeReceipt,
    UsageReceipt,
    VersionedReference,
    serialize_run_receipt,
)

_ARTIFACT_SHA256 = "a" * 64
_OCCURRENCE_ID = f"ulf_v1_{'b' * 64}"
_RECEIPT_ID = f"ulrr_v1_{'c' * 64}"
_POLICY_SHA256 = "d" * 64
_PRIVATE_CANARY = "account-secret-canary"


def _pointer(
    pointer_id: str,
    kind: str,
    json_pointer: str,
    *,
    authority: str = "independent_observer",
) -> EvidencePointer:
    return EvidencePointer.model_validate(
        {
            "pointer_id": pointer_id,
            "kind": kind,
            "artifact_sha256": _ARTIFACT_SHA256,
            "record_id": "case-record",
            "json_pointer": json_pointer,
            "arm": (
                "source"
                if pointer_id.endswith(".source")
                else "probe"
                if pointer_id.endswith(".probe")
                else "shared"
            ),
            "authority": authority,
            "source_id": "test-observer",
        }
    )


def _dataset_occurrence() -> FindingOccurrence:
    pointers = (
        _pointer("action.probe", "action", "/cases/0/findings/0/observed_effects"),
        _pointer("action.source", "action", "/cases/0/findings/0/expected_effects"),
        _pointer("input.probe", "input", "/cases/0/augmented_input"),
        _pointer("input.source", "input", "/original_input"),
        _pointer("response.probe", "response", "/technical_details/cases/0/trials/0"),
        _pointer("response.source", "response", "/technical_details/baseline/trials/0"),
    )
    return FindingOccurrence(
        occurrence_id=_OCCURRENCE_ID,
        kind="behavior_difference",
        category="changed_grounded_effect_argument",
        campaign_id="campaign-2026-08-23",
        source_interaction_id="invoice-payment",
        fixture=VersionedReference(id="accounts-payable", version="1.0.0"),
        case_id="invoice-payment:disfluency",
        operator=VersionedReference(id="input.surface.disfluency_repeat", version="1.0.0"),
        bundle=VersionedReference(id="surface-noise", version="1.0.0"),
        probe_change=ProbeChange(
            kind="input",
            source_evidence_pointer_ids=("input.source",),
            probe_evidence_pointer_ids=("input.probe",),
        ),
        observed_deltas=(
            ObservedDelta(
                kind="action",
                change="changed",
                subject="payment.invoice_reference",
                evidence_pointer_ids=("action.probe", "action.source"),
            ),
            ObservedDelta(
                kind="response",
                change="changed",
                subject="agent_response",
                evidence_pointer_ids=("response.probe", "response.source"),
            ),
        ),
        evidence_pointers=pointers,
        repetitions=RepetitionEvidence(
            requested=3,
            conclusive=3,
            violated=3,
            inconclusive=0,
            stability="stable",
            reproducibility="reproduced",
        ),
        required_capabilities=("response_observation",),
        limitations=("correctness_not_verified", "production_prevalence_not_measured"),
        run_receipt_id=_RECEIPT_ID,
        next_action="review_dataset_finding",
    )


def _stateful_occurrence() -> FindingOccurrence:
    pointers = (
        _pointer("input.probe", "input", "/case/conversation/1"),
        _pointer("input.source", "input", "/case/conversation/0"),
        _pointer("rule.probe", "rule", "/corrected_invariant_rules/0/trials/0"),
        _pointer("state.probe", "state", "/trials/0/variation/1/committed_state_snapshot"),
        _pointer("state.source", "state", "/trials/0/baseline/0/committed_state_snapshot"),
    )
    return FindingOccurrence(
        occurrence_id=_OCCURRENCE_ID,
        kind="customer_invariant_violation",
        category="customer_invariant_violation",
        campaign_id="campaign-2026-08-23",
        fixture=VersionedReference(id="accounts-payable", version="1.0.0"),
        case_id="correction-after-first-response",
        operator=VersionedReference(
            id="conversation.correction_after_first_response",
            version="1.0.0",
        ),
        probe_change=ProbeChange(
            kind="turn_sequence",
            source_evidence_pointer_ids=("input.source",),
            probe_evidence_pointer_ids=("input.probe",),
        ),
        observed_deltas=(
            ObservedDelta(
                kind="rule",
                change="violated",
                subject="final-amount-matches-corrected",
                evidence_pointer_ids=("rule.probe",),
            ),
            ObservedDelta(
                kind="state",
                change="changed",
                subject="final_amount",
                evidence_pointer_ids=("state.probe", "state.source"),
            ),
        ),
        violated_rule=VersionedReference(
            id="final-amount-matches-corrected",
            version="1.0.0",
        ),
        evidence_pointers=pointers,
        repetitions=RepetitionEvidence(
            requested=3,
            conclusive=2,
            violated=2,
            inconclusive=1,
            stability="stable",
            reproducibility="reproduced",
        ),
        required_capabilities=("conversation_replay", "state_observation"),
        limitations=("one_repetition_inconclusive",),
        inconclusive_reasons=("cleanup_reset_failed",),
        run_receipt_id=_RECEIPT_ID,
        next_action="inspect_stateful_evidence",
    )


def _private_receipt(*, evidence_scope: str = "response_and_state") -> RunReceipt:
    pointers = (
        _pointer("input.probe", "input", "/probe_input"),
        _pointer("input.source", "input", "/source_input"),
        _pointer("lifecycle.cleanup", "lifecycle", "/lifecycle/1"),
        _pointer("lifecycle.reset", "lifecycle", "/lifecycle/0"),
        _pointer("response.probe", "response", "/probe_response"),
        _pointer("response.source", "response", "/source_response"),
    )
    state = (
        StateReceipt(value={"balance": 100}, authority="environment_self_reported")
        if evidence_scope == "response_and_state"
        else None
    )
    return RunReceipt.model_validate(
        {
            "receipt_id": _RECEIPT_ID,
            "evidence_scope": evidence_scope,
            "source_input": {"account": _PRIVATE_CANARY, "amount": 100},
            "probe_input": {"account": _PRIVATE_CANARY, "amount": 200},
            "source_response": {"status": "paid"},
            "probe_response": {"status": "paid_twice"},
            "tool_exchanges": (
                ToolExchangeReceipt(
                    sequence=1,
                    call={"name": "pay_invoice", "arguments": {"amount": 200}},
                    result={"status": "completed"},
                    authority="source_self_reported",
                    source_id="target-adapter",
                ),
            ),
            "state_before": state,
            "state_after": state,
            "lifecycle": (
                LifecycleReceipt(
                    phase="initial_reset",
                    status="succeeded",
                    evidence_pointer_ids=("lifecycle.reset",),
                ),
                LifecycleReceipt(
                    phase="cleanup_reset",
                    status="succeeded",
                    evidence_pointer_ids=("lifecycle.cleanup",),
                ),
            ),
            "provenance": (
                ProvenanceReceipt(
                    role="environment",
                    id="accounts-payable",
                    version="1.0.0",
                    config_sha256="e" * 64,
                ),
                ProvenanceReceipt(role="evaluator", id="json-values-equal", version="1.0.0"),
                ProvenanceReceipt(role="observer", id="test-observer", version="1.0.0"),
            ),
            "trace_references": ("trace-01",),
            "usage": UsageReceipt(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                cost=0.01,
                duration_ms=250.0,
            ),
            "redaction": RedactionReceipt(
                policy_sha256=_POLICY_SHA256,
                matched_value_count=2,
                redacted_value_count=0,
                retained_private_value_count=2,
            ),
            "evidence_pointers": pointers,
            "recorded_at": datetime(2026, 8, 23, tzinfo=UTC),
        }
    )


def test_dataset_and_stateful_evidence_share_one_occurrence_contract() -> None:
    dataset_occurrence = _dataset_occurrence()
    stateful_occurrence = _stateful_occurrence()

    assert type(dataset_occurrence) is FindingOccurrence
    assert type(stateful_occurrence) is FindingOccurrence
    assert FindingOccurrence.model_validate_json(dataset_occurrence.model_dump_json()) == (
        dataset_occurrence
    )
    assert FindingOccurrence.model_validate_json(stateful_occurrence.model_dump_json()) == (
        stateful_occurrence
    )


def test_public_occurrence_references_private_receipt_without_copying_values() -> None:
    occurrence = _dataset_occurrence()
    receipt = _private_receipt()

    assert occurrence.run_receipt_id == receipt.receipt_id
    assert _PRIVATE_CANARY in receipt.model_dump_json()
    assert _PRIVATE_CANARY not in occurrence.model_dump_json()
    assert "source_input" not in occurrence.model_dump()
    assert "probe_response" not in occurrence.model_dump()


def test_occurrence_rejects_dangling_or_invalid_evidence_pointers() -> None:
    payload = _dataset_occurrence().model_dump(mode="json")
    payload["observed_deltas"][0]["evidence_pointer_ids"] = ["missing.pointer"]
    with pytest.raises(ValidationError, match="unknown evidence pointer"):
        FindingOccurrence.model_validate_json(json.dumps(payload))

    pointer_payload = _pointer("input.source", "input", "/source").model_dump(mode="json")
    pointer_payload["json_pointer"] = "/invalid~2escape"
    with pytest.raises(ValidationError, match="RFC 6901"):
        EvidencePointer.model_validate(pointer_payload)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"conclusive": 2}, "match requested repetitions"),
        ({"violated": 4}, "cannot exceed conclusive"),
        ({"violated": 1, "reproducibility": "reproduced"}, "match observed repetition counts"),
        (
            {"violated": 1, "reproducibility": "intermittent"},
            "intermittent reproduction is unstable",
        ),
        (
            {
                "conclusive": 0,
                "violated": 0,
                "inconclusive": 3,
                "stability": "stable",
                "reproducibility": "not_established",
            },
            "is inconclusive",
        ),
    ],
)
def test_repetition_evidence_rejects_contradictory_claims(
    update: dict[str, object],
    message: str,
) -> None:
    payload = RepetitionEvidence(
        requested=3,
        conclusive=3,
        violated=3,
        inconclusive=0,
        stability="stable",
        reproducibility="reproduced",
    ).model_dump(mode="json")
    payload.update(update)

    with pytest.raises(ValidationError, match=message):
        RepetitionEvidence.model_validate(payload)


def test_occurrence_requires_explicit_reasons_for_inconclusive_repetitions() -> None:
    payload = _stateful_occurrence().model_dump(mode="json")
    payload["inconclusive_reasons"] = []

    with pytest.raises(ValidationError, match="require at least one reason"):
        FindingOccurrence.model_validate_json(json.dumps(payload))


def test_occurrence_keeps_behavior_and_customer_rule_claims_distinct() -> None:
    behavior_payload = _dataset_occurrence().model_dump(mode="json")
    behavior_payload["violated_rule"] = {
        "id": "final-amount-matches-corrected",
        "version": "1.0.0",
    }
    with pytest.raises(ValidationError, match="cannot claim a customer rule violation"):
        FindingOccurrence.model_validate_json(json.dumps(behavior_payload))

    invariant_payload = _stateful_occurrence().model_dump(mode="json")
    invariant_payload["violated_rule"] = None
    with pytest.raises(ValidationError, match="require a violated customer rule"):
        FindingOccurrence.model_validate_json(json.dumps(invariant_payload))

    invariant_payload = _stateful_occurrence().model_dump(mode="json")
    invariant_payload["observed_deltas"][0]["subject"] = "different-customer-rule"
    with pytest.raises(ValidationError, match="rule identity must match"):
        FindingOccurrence.model_validate_json(json.dumps(invariant_payload))


def test_occurrence_binds_claims_to_compatible_source_and_probe_evidence() -> None:
    payload = _dataset_occurrence().model_dump(mode="json")
    for pointer in payload["evidence_pointers"]:
        if pointer["pointer_id"].startswith("action."):
            pointer["kind"] = "response"
    with pytest.raises(ValidationError, match="delta references incompatible evidence"):
        FindingOccurrence.model_validate_json(json.dumps(payload))

    payload = _dataset_occurrence().model_dump(mode="json")
    source_pointer = next(
        pointer
        for pointer in payload["evidence_pointers"]
        if pointer["pointer_id"] == "input.source"
    )
    source_pointer["arm"] = "probe"
    with pytest.raises(ValidationError, match="must reference the source arm"):
        FindingOccurrence.model_validate_json(json.dumps(payload))

    payload = _dataset_occurrence().model_dump(mode="json")
    action_source_pointer = next(
        pointer
        for pointer in payload["evidence_pointers"]
        if pointer["pointer_id"] == "action.source"
    )
    action_source_pointer["arm"] = "shared"
    with pytest.raises(ValidationError, match="require source and probe evidence"):
        FindingOccurrence.model_validate_json(json.dumps(payload))

    payload = _dataset_occurrence().model_dump(mode="json")
    payload["category"] = "missing_effect"
    with pytest.raises(ValidationError, match="category must match"):
        FindingOccurrence.model_validate_json(json.dumps(payload))


def test_response_only_receipt_rejects_state_evidence() -> None:
    payload = _private_receipt().model_dump(mode="json")
    payload["evidence_scope"] = "response_only"

    with pytest.raises(ValidationError, match="cannot contain state evidence"):
        RunReceipt.model_validate_json(json.dumps(payload))


def test_response_and_state_receipt_requires_both_snapshots() -> None:
    payload = _private_receipt().model_dump(mode="json")
    payload["state_before"] = None
    payload["state_after"] = None

    with pytest.raises(ValidationError, match="require before and after state"):
        RunReceipt.model_validate_json(json.dumps(payload))


def test_state_receipt_requires_independent_observer_identity() -> None:
    with pytest.raises(ValidationError, match="names an observer"):
        StateReceipt(value={}, authority="independent_observer")

    with pytest.raises(ValidationError, match="names an observer"):
        StateReceipt(
            value={},
            authority="environment_self_reported",
            observer_id="external-auditor",
        )


def test_independent_evidence_requires_matching_observer_provenance() -> None:
    payload = _private_receipt().model_dump(mode="json")
    payload["provenance"] = [item for item in payload["provenance"] if item["role"] != "observer"]

    with pytest.raises(ValidationError, match="matching observer provenance"):
        RunReceipt.model_validate_json(json.dumps(payload))


def test_receipt_rejects_noncontiguous_tool_order_and_unknown_lifecycle_pointer() -> None:
    payload = _private_receipt().model_dump(mode="json")
    payload["tool_exchanges"][0]["sequence"] = 2
    with pytest.raises(ValidationError, match="contiguous and ordered"):
        RunReceipt.model_validate_json(json.dumps(payload))

    payload = _private_receipt().model_dump(mode="json")
    payload["lifecycle"][0]["evidence_pointer_ids"] = ["missing.pointer"]
    with pytest.raises(ValidationError, match="unknown evidence pointer"):
        RunReceipt.model_validate_json(json.dumps(payload))

    payload = _private_receipt().model_dump(mode="json")
    payload["lifecycle"].reverse()
    with pytest.raises(ValidationError, match="preserve execution order"):
        RunReceipt.model_validate_json(json.dumps(payload))

    payload = _private_receipt().model_dump(mode="json")
    payload["lifecycle"][0]["evidence_pointer_ids"] = ["input.source"]
    with pytest.raises(ValidationError, match="require lifecycle evidence pointers"):
        RunReceipt.model_validate_json(json.dumps(payload))

    payload = _private_receipt().model_dump(mode="json")
    payload["evidence_pointers"] = [
        pointer
        for pointer in payload["evidence_pointers"]
        if pointer["pointer_id"] != "response.probe"
    ]
    with pytest.raises(ValidationError, match="source and probe input and response evidence"):
        RunReceipt.model_validate_json(json.dumps(payload))


def test_usage_and_redaction_accounting_reject_inconsistent_totals() -> None:
    with pytest.raises(ValidationError, match="total tokens"):
        UsageReceipt(input_tokens=10, output_tokens=5, total_tokens=14)

    with pytest.raises(ValidationError, match="cover every matched value"):
        RedactionReceipt(
            policy_sha256=_POLICY_SHA256,
            matched_value_count=3,
            redacted_value_count=1,
            retained_private_value_count=1,
        )


def test_receipt_requires_auditable_timestamp_and_bounded_private_values() -> None:
    payload = _private_receipt().model_dump(mode="json")
    payload["recorded_at"] = "2026-08-23T00:00:00"
    with pytest.raises(ValidationError, match="must include a UTC offset"):
        RunReceipt.model_validate_json(json.dumps(payload))


def test_receipt_serialization_revalidates_mutable_private_values() -> None:
    receipt = _private_receipt()
    assert isinstance(receipt.source_input, dict)
    receipt.source_input["oversized"] = "x" * 1_000_000

    with pytest.raises(ValidationError, match="exceeds the 1 MB JSON limit"):
        serialize_run_receipt(receipt)

    payload = _private_receipt().model_dump(mode="json")
    payload["source_input"] = {"oversized": "x" * 1_000_000}
    with pytest.raises(ValidationError, match="exceeds the 1 MB JSON limit"):
        RunReceipt.model_validate_json(json.dumps(payload))


def test_contracts_are_strict_and_reject_unknown_fields() -> None:
    payload = json.loads(_dataset_occurrence().model_dump_json())
    payload["headline"] = "This prose belongs in a later presentation contract."

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        FindingOccurrence.model_validate_json(json.dumps(payload))

    with pytest.raises(ValidationError):
        RepetitionEvidence.model_validate(
            {
                "requested": "3",
                "conclusive": 3,
                "violated": 3,
                "inconclusive": 0,
                "stability": "stable",
                "reproducibility": "reproduced",
            }
        )
