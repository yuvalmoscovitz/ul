from __future__ import annotations

from datetime import UTC, datetime
from typing import TextIO, cast

import pytest
from ul_cli.dataset.evaluation.campaign_persistence import CampaignPersistence
from ul_cli.dataset_augmentation_ledger import DatasetAugmentationLedger
from ul_cli.finding_reference import FindingReferenceContext


class _FailingOutputStream:
    closed = False

    def close(self) -> None:
        raise RuntimeError("output close failed")


class _RecordingLedger:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_campaign_persistence_closes_ledger_when_output_close_fails() -> None:
    ledger = _RecordingLedger()
    persistence = CampaignPersistence(
        augmentation_ledger=cast(DatasetAugmentationLedger, ledger),
        output_stream=cast(TextIO, _FailingOutputStream()),
        finding_reference_context=FindingReferenceContext(
            key=b"test-key",
            recorded_at=datetime.now(UTC),
        ),
    )

    with pytest.raises(RuntimeError, match="output close failed"):
        persistence.close()

    assert ledger.closed
