from pydantic import ValidationError
from pytest import raises
from ul_core import FindingEvidenceLevel, W3CTraceReference


def test_w3c_trace_reference_exposes_canonical_traceparent() -> None:
    reference = W3CTraceReference(trace_id="1" * 32, span_id="2" * 16)

    assert reference.traceparent == f"00-{'1' * 32}-{'2' * 16}-01"


def test_finding_evidence_level_requires_exact_fact_provenance() -> None:
    with raises(ValidationError, match="requires source and authority"):
        FindingEvidenceLevel(
            facts=("response_observed",),
            sources={"response_observed": "agent"},
            authorities={},
        )
