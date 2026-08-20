from __future__ import annotations

import re
from typing import Protocol

import pytest
from ul.dataset_augmentation import DatasetAugmentationEngine
from ul.dataset_evaluation import DatasetEvaluationRunner
from ul.deconstruction import OpenRouterDatasetSettings, create_semantic_model_deconstructor
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
from ul_core.evaluation import (
    EnvironmentCapabilities,
    EnvironmentLifecycleEvidence,
    EnvironmentResetEvidence,
    EnvironmentStateEvidence,
    EnvironmentTurnEvidence,
    EvaluationCase,
    ExecutionEvidence,
)

from examples.accounts_payable.dataset_target import (
    AMOUNT_SOURCE_INPUT,
    REPEATED_PAYMENT_INPUT,
    SELF_CORRECTED_PAYMENT_INPUT,
    SOURCE_INPUT,
    AccountsPayableDatasetTarget,
    SeededFirstValueWinsDefectAccountsPayableDatasetTarget,
    SeededFlakyIntentFanOutDefectAccountsPayableDatasetTarget,
    SeededIntentFanOutDefectAccountsPayableDatasetTarget,
)

_LIVE_SETTINGS = OpenRouterDatasetSettings()
_LIVE_TRANSFER_INPUT = "transfer 120$ to alice"


class _SingleTurnTestAgent(Protocol):
    async def execute(self, raw_input: str) -> ObservedAgentOutput: ...


class _RecordingDatasetTarget:
    environment_id = "accounts-payable-test-environment"
    config_sha256 = "0" * 64
    capabilities = EnvironmentCapabilities(
        supports_conversations=True,
        supports_state_observation=True,
        state_observation_authority="environment_self_reported",
        cancellation_guarantee="guaranteed",
    )

    def __init__(self, target: _SingleTurnTestAgent) -> None:
        self._target = target
        self.raw_inputs: list[str] = []

    def api_calls_for_case(self, case: EvaluationCase) -> int:
        return len(case.turns)

    async def execute(self, case: EvaluationCase) -> ExecutionEvidence:
        turns: list[EnvironmentTurnEvidence] = []
        for turn in case.turns:
            self.raw_inputs.append(turn.content)
            output = await self._target.execute(turn.content)
            turns.append(
                EnvironmentTurnEvidence(
                    turn_id=turn.id,
                    response=output.raw_output,
                    state_snapshot=output.raw_output,
                    state_observation_authority="environment_self_reported",
                )
            )
        return ExecutionEvidence(
            case_id=case.id,
            environment_id=self.environment_id,
            environment_config_sha256=self.config_sha256,
            initial_state=EnvironmentStateEvidence(value={}, authority="environment_self_reported"),
            turns=tuple(turns),
            final_response=turns[-1].response,
            final_state=EnvironmentStateEvidence(
                value=turns[-1].state_snapshot,
                authority="environment_self_reported",
            ),
            lifecycle=EnvironmentLifecycleEvidence(
                initial_reset=EnvironmentResetEvidence(
                    reset_session_requested=True,
                    reset_session_acknowledged=True,
                    reset_env_requested=True,
                    reset_env_acknowledged=True,
                ),
                cleanup_reset=EnvironmentResetEvidence(
                    reset_session_requested=True,
                    reset_session_acknowledged=True,
                    reset_env_requested=True,
                    reset_env_acknowledged=True,
                ),
                terminal_status="succeeded",
                completed_phases=("execute", "cleanup"),
                delivery="certain",
                cleanup="succeeded",
                environment_state_uncertain=False,
            ),
        )


class _SeededFirstValueWinsTransferTarget:
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


def _repeated_payment_frame(
    interaction_id: str,
    *,
    candidate: bool = False,
    observed_payment_count: int = 0,
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
    request = RequestUnit(
        id="pay_invoice",
        evidence=_input_evidence("Pay"),
        confidence=1,
        status="explicit",
        mode="act",
        predicate="payment_committed",
        factor_ids=(invoice.id,),
    )
    communication_acts: tuple[CommunicationAct, ...] = ()
    if candidate:
        communication_acts = (
            CommunicationAct(
                id="repeated_pay",
                evidence=_input_evidence("Pay pay"),
                confidence=1,
                status="explicit",
                kind="repetition",
            ),
        )
    outcomes = tuple(
        ObservedOutcome(
            id=f"payment-{position}",
            evidence=tuple(
                EvidenceReference(
                    source="output",
                    json_pointer=f"/raw_observed_output/actions/{position}/{field}",
                    text_quote=None,
                )
                for field in ("action", "invoice_reference")
            ),
            confidence=1,
            status="observed",
            request_unit_ids=(request.id,),
            position=position,
            kind="action",
            predicate="payment_committed",
            fields={"invoice_reference": "AC-100"},
        )
        for position in range(observed_payment_count)
    )
    return SemanticFrame(
        interaction_id=interaction_id,
        request_units=(request,),
        factors=(invoice,),
        communication_acts=communication_acts,
        outcomes=outcomes,
        extractor_version="deterministic-repetition-proof",
    )


class _DeterministicRepetitionPipeline:
    async def render(
        self,
        raw_input: str,
        instruction: str,
        *,
        allow_temporary_value: bool = False,
    ) -> RenderedUserInput:
        del raw_input, instruction, allow_temporary_value
        raise AssertionError("word repetition uses the deterministic renderer")

    async def deconstruct(
        self,
        record: InteractionRecord | UserInputRecord,
        reference_frame: SemanticFrame | None = None,
    ) -> SemanticFrame:
        candidate = record.raw_input.casefold() == REPEATED_PAYMENT_INPUT.casefold()
        if record.id == "ap-single-approved-payment":
            assert reference_frame is None
            return _repeated_payment_frame(record.id, observed_payment_count=1)
        if not isinstance(record, InteractionRecord):
            assert reference_frame is not None
            return _repeated_payment_frame(record.id, candidate=candidate)
        assert reference_frame is not None
        assert isinstance(record.raw_observed_output, dict)
        actions = record.raw_observed_output["actions"]
        assert isinstance(actions, list)
        return _repeated_payment_frame(
            record.id,
            candidate=candidate,
            observed_payment_count=len(actions),
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
    recording_target = _RecordingDatasetTarget(target)

    result = await DatasetEvaluationRunner(
        DatasetAugmentationEngine(semantic_pipeline, semantic_pipeline),
        semantic_pipeline,
        recording_target,
        allow_network_egress=True,
    ).run(source, operator_ids=("input.intent.self_correction",))

    assert result.baseline.verdict == "no_divergence"
    assert result.baseline.trial_set.requested_repetitions == 3
    assert result.baseline.trial_set.stability == "stable"
    assert result.cases[0].candidate.augmented_input == SELF_CORRECTED_PAYMENT_INPUT
    assert result.cases[0].verdict == expected_verdict
    assert result.cases[0].trial_set is not None
    assert result.cases[0].trial_set.requested_repetitions == 3
    assert result.cases[0].trial_set.stability == "stable"
    assert [finding.category for finding in result.cases[0].findings] == (
        [] if expected_finding is None else [expected_finding]
    )
    assert (
        recording_target.raw_inputs
        == [
            AMOUNT_SOURCE_INPUT,
            SELF_CORRECTED_PAYMENT_INPUT,
        ]
        * 3
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "expected_verdict", "expected_finding"),
    [
        (AccountsPayableDatasetTarget(), "no_divergence", None),
        (
            SeededIntentFanOutDefectAccountsPayableDatasetTarget(),
            "divergence_needs_review",
            "duplicate_effect",
        ),
    ],
)
async def test_repetition_e2e_compares_two_fresh_runs_per_input(
    target: AccountsPayableDatasetTarget | SeededIntentFanOutDefectAccountsPayableDatasetTarget,
    expected_verdict: str,
    expected_finding: str | None,
) -> None:
    source_output = await AccountsPayableDatasetTarget().execute(SOURCE_INPUT)
    source = InteractionRecord(
        id="ap-single-approved-payment",
        raw_input=SOURCE_INPUT,
        raw_observed_output=source_output.raw_output,
    )
    semantic_pipeline = _DeterministicRepetitionPipeline()
    recording_target = _RecordingDatasetTarget(target)

    result = await DatasetEvaluationRunner(
        DatasetAugmentationEngine(semantic_pipeline, semantic_pipeline),
        semantic_pipeline,
        recording_target,
        allow_network_egress=True,
    ).run(
        source,
        operator_ids=("input.surface.disfluency_repeat",),
        repetitions=2,
    )

    case = result.cases[0]
    assert result.baseline.trial_set.requested_repetitions == 2
    assert result.baseline.trial_set.stability == "stable"
    assert case.candidate.augmented_input == REPEATED_PAYMENT_INPUT
    assert case.verdict == expected_verdict
    assert case.trial_set is not None
    assert case.trial_set.requested_repetitions == 2
    assert case.trial_set.stability == "stable"
    assert [finding.category for finding in case.findings] == (
        [] if expected_finding is None else [expected_finding]
    )
    assert recording_target.raw_inputs == [SOURCE_INPUT, REPEATED_PAYMENT_INPUT] * 2
    for trial in (*result.baseline.trial_set.trials, *case.trial_set.trials):
        assert trial.target_output is not None
        raw_output = trial.target_output.raw_output
        assert isinstance(raw_output, dict)
        actions = raw_output["actions"]
        assert isinstance(actions, list)
        first_action = actions[0]
        assert isinstance(first_action, dict)
        assert first_action["payment_id"] == "pay-0001"


@pytest.mark.asyncio
async def test_repetition_e2e_reports_seeded_variation_instability() -> None:
    source_output = await AccountsPayableDatasetTarget().execute(SOURCE_INPUT)
    source = InteractionRecord(
        id="ap-single-approved-payment",
        raw_input=SOURCE_INPUT,
        raw_observed_output=source_output.raw_output,
    )
    semantic_pipeline = _DeterministicRepetitionPipeline()
    recording_target = _RecordingDatasetTarget(
        SeededFlakyIntentFanOutDefectAccountsPayableDatasetTarget(seed=4)
    )

    result = await DatasetEvaluationRunner(
        DatasetAugmentationEngine(semantic_pipeline, semantic_pipeline),
        semantic_pipeline,
        recording_target,
        allow_network_egress=True,
    ).run(source, operator_ids=("input.surface.disfluency_repeat",))

    case = result.cases[0]
    assert result.baseline.trial_set.stability == "stable"
    assert case.verdict == "divergence_needs_review"
    assert case.trial_set is not None
    assert case.trial_set.stability == "unstable"
    assert case.findings == ()
    assert {
        len(group.representative_effects): group.repetitions
        for group in case.trial_set.outcome_groups
    } == {1: (1, 3), 2: (2,)}
    assert recording_target.raw_inputs == [SOURCE_INPUT, REPEATED_PAYMENT_INPUT] * 3


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
    async with create_semantic_model_deconstructor(_LIVE_SETTINGS) as semantic_model:
        result = await DatasetEvaluationRunner(
            DatasetAugmentationEngine(semantic_model, semantic_model),
            semantic_model,
            _RecordingDatasetTarget(SeededIntentFanOutDefectAccountsPayableDatasetTarget()),
            allow_network_egress=True,
        ).run(source, operator_ids=("input.surface.disfluency_repeat",))

    case = result.cases[0]
    assert case.candidate.augmented_input.casefold() == "pay pay ac-100."
    if case.verdict == "inconclusive":
        assert case.trial_set is not None
        assert case.trial_set.stability == "inconclusive"
        return
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
    async with create_semantic_model_deconstructor(_LIVE_SETTINGS) as semantic_model:
        result = await DatasetEvaluationRunner(
            DatasetAugmentationEngine(semantic_model, semantic_model),
            semantic_model,
            _RecordingDatasetTarget(_SeededFirstValueWinsTransferTarget()),
            allow_network_egress=True,
        ).run(source, operator_ids=("input.intent.self_correction",))

    case = result.cases[0]
    assert case.candidate.passed, case.candidate.failure_reasons
    if case.verdict == "inconclusive":
        assert case.trial_set is not None
        assert case.trial_set.stability == "inconclusive"
        return
    assert case.verdict == "divergence_needs_review"
    assert [finding.category for finding in case.findings] == ["changed_grounded_effect_argument"]
