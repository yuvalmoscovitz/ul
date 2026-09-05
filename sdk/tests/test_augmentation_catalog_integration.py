from importlib import import_module
from pathlib import Path

from ul.augmentations.conversation import (
    CorrectionAfterFirstResponseCase,
    RetryAfterSuccessfulCommitCase,
)
from ul.augmentations.dataset import builtin_dataset_augmentation_operators
from ul.augmentations.environment_fault import TimeoutAfterCommitCase
from ul_core.augmentations.definitions import builtin_augmentation_catalog
from ul_core.augmentations.registry import builtin_augmentation_registry


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


def test_dataset_catalog_and_runtime_share_applicability_contracts() -> None:
    catalog = builtin_augmentation_catalog()

    for operator in builtin_dataset_augmentation_operators():
        specification = catalog.get(operator.id, operator.version)
        assert specification.applicability_profile == operator.applicability_profile
        assert specification.applicability_rule == operator.applicability_rule


def test_every_catalog_binding_points_to_an_importable_runtime() -> None:
    stress_case_by_runtime = {
        "ul.augmentations.conversation:run_correction_stress_test": (
            CorrectionAfterFirstResponseCase
        ),
        "ul.augmentations.conversation:run_retry_after_successful_commit_stress_test": (
            RetryAfterSuccessfulCommitCase
        ),
        "ul.augmentations.environment_fault:run_timeout_after_commit_stress_test": (
            TimeoutAfterCommitCase
        ),
    }
    scenario_registry = builtin_augmentation_registry()
    for specification in builtin_augmentation_catalog().list(latest_only=False):
        for binding in specification.bindings:
            module_name, separator, attribute_name = binding.runtime.partition(":")
            assert separator == ":"
            runtime = getattr(import_module(module_name), attribute_name)
            reference = f"{specification.ref.id}@{specification.ref.version}"
            if binding.mode == "dataset_variation":
                operator = runtime(reference)
                assert (operator.id, operator.version) == (
                    specification.ref.id,
                    specification.ref.version,
                )
            elif binding.mode == "scenario_materialization":
                registered = scenario_registry.get(specification.ref.id, specification.ref.version)
                assert runtime is type(registered)
            else:
                case_model = stress_case_by_runtime[binding.runtime]
                assert case_model.model_fields["operator_id"].default == specification.ref.id
                assert (
                    case_model.model_fields["operator_version"].default == specification.ref.version
                )


def test_readme_points_to_the_authoritative_catalog_and_public_cli() -> None:
    readme = (Path(__file__).parents[2] / "core/src/ul_core/augmentations/README.md").read_text()

    assert "`definitions.py` is the single source of truth" in readme
    assert "ul augmentations list" in readme
    assert "ul augmentations show ID[@VERSION]" in readme
    assert "ul augmentations list --json" in readme
    assert "| ID | Controlled change | Expected relation |" not in readme
