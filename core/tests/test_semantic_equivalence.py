from typing import Literal

import pytest
from pydantic import ValidationError
from ul_core.contracts import SemanticEquivalenceVerifier
from ul_core.dataset import (
    SemanticDelta,
    SemanticEquivalenceAssessment,
)


def changed_amount_delta() -> SemanticDelta:
    return SemanticDelta(
        category="value",
        operation="changed",
        source_quote="$125",
        candidate_quote="$150",
        description="The payment amount changed.",
    )


def test_equivalent_assessment_round_trips() -> None:
    assessment = SemanticEquivalenceAssessment(
        verdict="equivalent",
        explanation="The wording changed but the request did not.",
        verifier_version="test-verifier/1",
        metadata={"model": "test-model"},
    )

    assert (
        SemanticEquivalenceAssessment.model_validate_json(assessment.model_dump_json())
        == assessment
    )


def test_assessment_enforces_verdict_delta_invariants() -> None:
    with pytest.raises(ValidationError, match="equivalent assessments cannot contain deltas"):
        SemanticEquivalenceAssessment(
            verdict="equivalent",
            explanation="Contradictory result.",
            deltas=(changed_amount_delta(),),
            verifier_version="test-verifier/1",
        )

    with pytest.raises(ValidationError, match="require at least one delta"):
        SemanticEquivalenceAssessment(
            verdict="different",
            explanation="Missing evidence.",
            verifier_version="test-verifier/1",
        )


@pytest.mark.parametrize("operation", ("added", "removed", "changed", "reordered"))
def test_delta_requires_at_least_one_evidence_quote(
    operation: Literal["added", "removed", "changed", "reordered"],
) -> None:
    with pytest.raises(ValidationError, match="require source or candidate evidence"):
        SemanticDelta(
            category="request",
            operation=operation,
            description="A semantic difference.",
        )


def test_delta_accepts_exact_quotes_from_either_message() -> None:
    added = SemanticDelta(
        category="constraint",
        operation="added",
        candidate_quote="only after approval",
        description="The candidate adds an approval constraint.",
    )
    removed = SemanticDelta(
        category="negation",
        operation="removed",
        source_quote="do not send it",
        description="The candidate removes the prohibition.",
    )

    assert added.source_quote is None
    assert removed.candidate_quote is None


def test_uncertain_assessment_may_include_observed_deltas() -> None:
    assessment = SemanticEquivalenceAssessment(
        verdict="uncertain",
        explanation="The relationship between the requests is ambiguous.",
        deltas=(
            SemanticDelta(
                category="relationship",
                operation="changed",
                source_quote="if approved",
                candidate_quote="when approved",
                description="The condition may have become a timing statement.",
            ),
        ),
        verifier_version="test-verifier/1",
    )

    assert assessment.verdict == "uncertain"


def test_verifier_protocol_accepts_async_implementation() -> None:
    class DeterministicVerifier:
        async def verify(
            self, source_input: str, candidate_input: str
        ) -> SemanticEquivalenceAssessment:
            return SemanticEquivalenceAssessment(
                verdict="equivalent",
                explanation=f"Equivalent: {source_input!r} and {candidate_input!r}.",
                verifier_version="deterministic/1",
            )

    assert isinstance(DeterministicVerifier(), SemanticEquivalenceVerifier)
