from __future__ import annotations

import pytest
from ul.dataset_augmentation import DatasetAugmentationEngine
from ul.dataset_evaluation import DatasetEvaluationRunner
from ul.deconstruction import OpenRouterDatasetSettings, OpenRouterSemanticDeconstructor
from ul_core.dataset import InteractionRecord

from examples.accounts_payable.dataset_target import (
    SOURCE_INPUT,
    AccountsPayableDatasetTarget,
    SeededIntentFanOutDefectAccountsPayableDatasetTarget,
)

_LIVE_SETTINGS = OpenRouterDatasetSettings()


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
    assert case.candidate.augmented_input.casefold() in {
        "pay pay invoice ac-100.",
        "pay invoice invoice ac-100.",
    }
    assert case.verdict == "divergence_needs_review", case.candidate.failure_reasons
    assert [finding.category for finding in case.findings] == ["duplicate_effect"]
