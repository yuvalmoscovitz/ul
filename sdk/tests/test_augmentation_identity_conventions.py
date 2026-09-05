from ul.augmentations.conversation import (
    CorrectionAfterFirstResponseCase,
    RetryAfterSuccessfulCommitCase,
)
from ul.augmentations.dataset import builtin_dataset_augmentation_operators
from ul.augmentations.environment_fault import TimeoutAfterCommitCase
from ul_core.augmentations.registry import builtin_augmentation_registry


def test_every_runtime_augmentation_uses_one_canonical_namespace() -> None:
    dataset_references = tuple(
        (operator.id, operator.version) for operator in builtin_dataset_augmentation_operators()
    )
    scenario_references = tuple(
        (augmentation.metadata.id, augmentation.metadata.version)
        for augmentation in builtin_augmentation_registry().list(latest_only=False)
    )
    stress_references = (
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
    )
    references = (*dataset_references, *scenario_references, *stress_references)

    assert all(
        augmentation_id.startswith(("input.", "conversation.", "environment."))
        for augmentation_id, _ in references
    )
    assert len(set(references)) == 22
    assert references.count(("conversation.correction_after_first_response", "1.0.0")) == 2
    assert references.count(("environment.tool.timeout_after_commit", "1.0.0")) == 2
