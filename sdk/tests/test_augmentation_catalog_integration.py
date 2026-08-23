from importlib import import_module
from pathlib import Path

from ul.augmentations.dataset import builtin_dataset_augmentation_operators
from ul.event_stress import (
    CorrectionAfterFirstResponseCase,
    RetryAfterSuccessfulCommitCase,
)
from ul.timeout_after_commit import TimeoutAfterCommitCase
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
        "ul.event_stress:run_correction_stress_test": CorrectionAfterFirstResponseCase,
        "ul.event_stress:run_retry_after_successful_commit_stress_test": (
            RetryAfterSuccessfulCommitCase
        ),
        "ul.timeout_after_commit:run_timeout_after_commit_stress_test": TimeoutAfterCommitCase,
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


def test_every_catalog_entry_is_documented_with_authoritative_fields() -> None:
    readme = (Path(__file__).parents[2] / "core/src/ul_core/augmentations/README.md").read_text()
    section_by_surface = {
        "human_behavior": "Human behavior",
        "task_semantics": "Task semantics",
        "conversation_workflow": "Conversation and workflow",
        "world_business_state": "World and business state",
        "tool_execution": "Tool and execution",
        "trust_policy_authorization": "Trust, policy, and authorization",
    }

    for specification in builtin_augmentation_catalog().list(latest_only=False):
        expected_row = (
            f"| `{specification.ref.id}` | {specification.summary} | "
            f"{specification.expected_relation} |"
        )
        assert readme.count(expected_row) == 1
        section_start = readme.index(f"### {section_by_surface[specification.surface]}")
        next_section = readme.find("\n### ", section_start + 1)
        section = readme[section_start : next_section if next_section >= 0 else len(readme)]
        assert expected_row in section
