from __future__ import annotations

import re

import pytest
from ul.dataset_augmentation import DatasetAugmentationEngine
from ul.dataset_evaluation import DatasetEvaluationRunner
from ul.deconstruction import OpenRouterDatasetSettings, OpenRouterSemanticDeconstructor
from ul_core.dataset import (
    CommunicationAct,
    EvidenceReference,
    InteractionRecord,
    ObservedAgentOutput,
    ObservedOutcome,
    RenderedUserInput,
    RequestUnit,
    SemanticFactor,
    SemanticFrame,
    SemanticRelation,
    UserInputRecord,
)
from ul_core.models import SafetyEnvelope

from examples.accounts_payable.dataset_target import (
    AMOUNT_SOURCE_INPUT,
    SELF_CORRECTED_PAYMENT_INPUT,
    SOURCE_INPUT,
    AccountsPayableDatasetTarget,
    SeededFirstValueWinsDefectAccountsPayableDatasetTarget,
    SeededIntentFanOutDefectAccountsPayableDatasetTarget,
)

_LIVE_SETTINGS = OpenRouterDatasetSettings()
_LIVE_TRANSFER_INPUT = "transfer 120$ to alice"


class _SeededFirstValueWinsTransferTarget:
    safety_envelope = SafetyEnvelope(
        description="Isolated synthetic transfer ledger.",
        isolated=True,
        allows_network_egress=False,
        allows_business_side_effects=False,
    )
    fresh_state_per_execution = True

    async def execute(self, raw_input: str) -> ObservedAgentOutput:
        amounts = re.findall(r"\b\d+\$", raw_input)
        if not amounts or "alice" not in raw_input.casefold():
            raise ValueError("target requires a transfer amount and recipient")
        return ObservedAgentOutput(
            raw_output={
                "actions": [
                    {
                        "action": "transfer",
                        "amount": amounts[0].removesuffix("$"),
                        "recipient": "alice",
                    }
                ]
            },
            metadata={"fixture_id": "first-value-wins-transfer", "isolated": True},
        )


def _input_evidence(quote: str) -> tuple[EvidenceReference, ...]:
    return (EvidenceReference(source="input", json_pointer="/raw_input", text_quote=quote),)


def _payment_frame(
    interaction_id: str,
    *,
    candidate: bool = False,
    observed_amount: str | None = None,
) -> SemanticFrame:
    invoice = SemanticFactor(
        id="invoice",
        evidence=_input_evidence("AC-100"),
        confidence=1,
        status="explicit",
        kind="identifier",
        role="invoice_reference",
        value="AC-100",
    )
    final_amount = SemanticFactor(
        id="final_amount",
        evidence=_input_evidence("12500$"),
        confidence=1,
        status="explicit",
        kind="money",
        role="amount",
        value="12500",
    )
    request = RequestUnit(
        id="pay_invoice",
        evidence=_input_evidence("pay"),
        confidence=1,
        status="explicit",
        mode="act",
        predicate="payment_committed",
        factor_ids=(invoice.id, final_amount.id),
    )
    factors: tuple[SemanticFactor, ...] = (invoice, final_amount)
    communication_acts: tuple[CommunicationAct, ...] = ()
    relations: tuple[SemanticRelation, ...] = ()
    if candidate:
        provisional_amount = SemanticFactor(
            id="provisional_amount",
            evidence=_input_evidence("13500$"),
            confidence=1,
            status="superseded",
            kind="money",
            role="amount",
            value="13500",
        )
        correction_evidence = _input_evidence("13500$, sorry 12500$")
        factors = (invoice, provisional_amount, final_amount)
        communication_acts = (
            CommunicationAct(
                id="correction",
                evidence=correction_evidence,
                confidence=1,
                status="explicit",
                kind="self_correction",
                factor_ids=(provisional_amount.id, final_amount.id),
            ),
        )
        relations = (
            SemanticRelation(
                id="superseded_amount",
                evidence=correction_evidence,
                confidence=1,
                status="explicit",
                kind="superseded_by",
                source_ids=(provisional_amount.id,),
                target_ids=(final_amount.id,),
            ),
        )
    outcomes: tuple[ObservedOutcome, ...] = ()
    if observed_amount is not None:
        outcomes = (
            ObservedOutcome(
                id="payment",
                evidence=tuple(
                    EvidenceReference(
                        source="output",
                        json_pointer=f"/raw_observed_output/actions/0/{field}",
                        text_quote=None,
                    )
                    for field in ("action", "invoice_reference", "amount")
                ),
                confidence=1,
                status="observed",
                request_unit_ids=(request.id,),
                position=0,
                kind="action",
                predicate="payment_committed",
                fields={"invoice_reference": "AC-100", "amount": observed_amount},
            ),
        )
    return SemanticFrame(
        interaction_id=interaction_id,
        request_units=(request,),
        factors=factors,
        relations=relations,
        communication_acts=communication_acts,
        outcomes=outcomes,
        extractor_version="deterministic-self-correction-proof",
    )


class _DeterministicSelfCorrectionPipeline:
    async def render(
        self,
        raw_input: str,
        instruction: str,
        *,
        allow_temporary_value: bool = False,
    ) -> RenderedUserInput:
        assert raw_input == AMOUNT_SOURCE_INPUT
        assert "temporary value" in instruction
        assert allow_temporary_value is True
        return RenderedUserInput(
            text=SELF_CORRECTED_PAYMENT_INPUT,
            metadata={"renderer": "deterministic-correction-proof"},
        )

    async def deconstruct(
        self,
        record: InteractionRecord | UserInputRecord,
        reference_frame: SemanticFrame | None = None,
    ) -> SemanticFrame:
        if record.id == "ap-self-correction":
            assert reference_frame is None
            return _payment_frame(record.id, observed_amount="12500")
        if not isinstance(record, InteractionRecord):
            assert reference_frame is not None
            return _payment_frame(record.id, candidate=True)
        assert isinstance(record, InteractionRecord)
        assert reference_frame is not None
        assert isinstance(record.raw_observed_output, dict)
        actions = record.raw_observed_output["actions"]
        assert isinstance(actions, list)
        action = actions[0]
        assert isinstance(action, dict)
        amount = action["amount"]
        assert isinstance(amount, str)
        return _payment_frame(
            record.id,
            candidate=record.raw_input == SELF_CORRECTED_PAYMENT_INPUT,
            observed_amount=amount,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "expected_verdict", "expected_finding"),
    [
        (AccountsPayableDatasetTarget(), "no_divergence", None),
        (
            SeededFirstValueWinsDefectAccountsPayableDatasetTarget(),
            "divergence_needs_review",
            "changed_grounded_effect_argument",
        ),
    ],
)
async def test_self_correction_e2e_compares_real_isolated_payment_actions(
    target: AccountsPayableDatasetTarget | SeededFirstValueWinsDefectAccountsPayableDatasetTarget,
    expected_verdict: str,
    expected_finding: str | None,
) -> None:
    source_output = await AccountsPayableDatasetTarget().execute(AMOUNT_SOURCE_INPUT)
    source = InteractionRecord(
        id="ap-self-correction",
        raw_input=AMOUNT_SOURCE_INPUT,
        raw_observed_output=source_output.raw_output,
    )
    semantic_pipeline = _DeterministicSelfCorrectionPipeline()

    result = await DatasetEvaluationRunner(
        DatasetAugmentationEngine(semantic_pipeline, semantic_pipeline),
        semantic_pipeline,
        target,
    ).run(source, operator_ids=("intent.self_correction",))

    assert result.baseline.verdict == "no_divergence"
    assert result.cases[0].candidate.augmented_input == SELF_CORRECTED_PAYMENT_INPUT
    assert result.cases[0].verdict == expected_verdict
    assert [finding.category for finding in result.cases[0].findings] == (
        [] if expected_finding is None else [expected_finding]
    )


@pytest.mark.asyncio
@pytest.mark.skipif(
    not (
        _LIVE_SETTINGS.live_calls
        and _LIVE_SETTINGS.allow_external_data_processing
        and _LIVE_SETTINGS.api_key is not None
    ),
    reason="requires explicit live dataset call and external processing opt-ins",
)
async def test_live_deconstructor_discovers_seeded_duplicate_payment() -> None:
    source_output = await AccountsPayableDatasetTarget().execute(SOURCE_INPUT)
    source = InteractionRecord(
        id="ap-single-approved-payment",
        raw_input=SOURCE_INPUT,
        raw_observed_output=source_output.raw_output,
    )
    async with OpenRouterSemanticDeconstructor(_LIVE_SETTINGS) as semantic_model:
        result = await DatasetEvaluationRunner(
            DatasetAugmentationEngine(semantic_model, semantic_model),
            semantic_model,
            SeededIntentFanOutDefectAccountsPayableDatasetTarget(),
        ).run(source, operator_ids=("surface.disfluency_repeat",))

    case = result.cases[0]
    assert case.candidate.augmented_input.casefold() == "pay pay ac-100."
    assert case.verdict == "divergence_needs_review", case.candidate.failure_reasons
    assert [finding.category for finding in case.findings] == ["duplicate_effect"]


@pytest.mark.asyncio
@pytest.mark.skipif(
    not (
        _LIVE_SETTINGS.live_calls
        and _LIVE_SETTINGS.allow_external_data_processing
        and _LIVE_SETTINGS.api_key is not None
    ),
    reason="requires explicit live dataset call and external processing opt-ins",
)
async def test_live_pipeline_discovers_seeded_first_value_wins_defect() -> None:
    source = InteractionRecord(
        id="live-transfer-self-correction",
        raw_input=_LIVE_TRANSFER_INPUT,
        raw_observed_output={
            "actions": [{"action": "transfer", "amount": "120", "recipient": "alice"}]
        },
    )
    async with OpenRouterSemanticDeconstructor(_LIVE_SETTINGS) as semantic_model:
        result = await DatasetEvaluationRunner(
            DatasetAugmentationEngine(semantic_model, semantic_model),
            semantic_model,
            _SeededFirstValueWinsTransferTarget(),
        ).run(source, operator_ids=("intent.self_correction",))

    case = result.cases[0]
    assert case.candidate.passed, case.candidate.failure_reasons
    assert case.verdict == "divergence_needs_review"
    assert [finding.category for finding in case.findings] == ["changed_grounded_effect_argument"]
