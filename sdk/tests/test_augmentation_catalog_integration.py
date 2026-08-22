from ul.dataset_augmentation import builtin_dataset_augmentation_operators
from ul.event_stress import (
    CorrectionAfterFirstResponseCase,
    RetryAfterSuccessfulCommitCase,
)
from ul.timeout_after_commit import TimeoutAfterCommitCase
from ul_core.augmentation import builtin_augmentation_registry
from ul_core.augmentation_catalog import builtin_augmentation_catalog


def test_catalog_covers_every_current_augmentation_identity() -> None:
    dataset_references = {
        (operator.id, operator.version) for operator in builtin_dataset_augmentation_operators()
    }
    scenario_references = {
        (augmentation.metadata.id, augmentation.metadata.version)
        for augmentation in builtin_augmentation_registry().list(latest_only=False)
    }
    stress_references = {
        (
            CorrectionAfterFirstResponseCase.model_fields["operator_id"].default,
            CorrectionAfterFirstResponseCase.model_fields["operator_version"].default,
        ),
        (
            RetryAfterSuccessfulCommitCase.model_fields["operator_id"].default,
            RetryAfterSuccessfulCommitCase.model_fields["operator_version"].default,
        ),
        (
            TimeoutAfterCommitCase.model_fields["operator_id"].default,
            TimeoutAfterCommitCase.model_fields["operator_version"].default,
        ),
    }
    catalog_references = {
        (augmentation.ref.id, augmentation.ref.version)
        for augmentation in builtin_augmentation_catalog().list(latest_only=False)
    }

    assert catalog_references == dataset_references | scenario_references | stress_references
    assert len(catalog_references) == 21


def test_catalog_discovery_does_not_change_runtime_metadata() -> None:
    before = tuple(
        augmentation.metadata.model_dump_json()
        for augmentation in builtin_augmentation_registry().list(latest_only=False)
    )

    builtin_augmentation_catalog().list()

    after = tuple(
        augmentation.metadata.model_dump_json()
        for augmentation in builtin_augmentation_registry().list(latest_only=False)
    )
    assert after == before
