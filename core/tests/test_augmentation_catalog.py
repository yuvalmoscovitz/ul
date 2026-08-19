import pytest
from pydantic import ValidationError
from ul_core.augmentation_catalog import (
    AugmentationBinding,
    AugmentationRef,
    AugmentationRequirements,
    BuiltinAugmentationCatalog,
    BuiltinAugmentationSpec,
    builtin_augmentation_catalog,
)


def test_builtin_catalog_is_unique_sorted_and_immutable() -> None:
    catalog = builtin_augmentation_catalog()
    references = tuple((item.ref.id, item.ref.version) for item in catalog.list())

    assert len(references) == 18
    assert references == tuple(sorted(references))
    assert len(references) == len(set(references))
    with pytest.raises(ValidationError, match="frozen"):
        catalog.augmentations = ()
    with pytest.raises(ValidationError, match="frozen"):
        catalog.augmentations[0].summary = "changed"


def test_exported_catalog_models_round_trip_through_json() -> None:
    catalog = builtin_augmentation_catalog()
    spec = catalog.get("conversation.correction_after_first_response")

    assert BuiltinAugmentationSpec.model_validate_json(spec.model_dump_json()) == spec
    assert BuiltinAugmentationCatalog.model_validate_json(catalog.model_dump_json()) == catalog


def test_timeout_is_one_catalog_entry_with_two_typed_bindings() -> None:
    timeout = builtin_augmentation_catalog().get("environment.tool.timeout_after_commit", "1.0.0")

    assert timeout.scope == "environment"
    assert tuple(binding.mode for binding in timeout.bindings) == (
        "scenario_materialization",
        "sandbox_fault",
    )
    assert timeout.bindings[0].cli_available is False
    assert timeout.bindings[0].execution_owner == "augmentation_registry"
    assert timeout.bindings[1].command == "ul stress timeout-after-commit"
    assert timeout.bindings[1].requirements.sandbox_capabilities == (
        "environment.tool.timeout_after_commit@1.0.0",
    )


def test_correction_is_one_catalog_entry_with_two_typed_bindings() -> None:
    correction = builtin_augmentation_catalog().get(
        "conversation.correction_after_first_response", "1.0.0"
    )

    assert tuple(binding.mode for binding in correction.bindings) == (
        "scenario_materialization",
        "conversation_stress",
    )
    assert correction.bindings[0].execution_owner == "augmentation_registry"
    assert correction.bindings[1].command == "ul stress correction"


def test_dataset_augmentation_declares_actual_execution_requirements() -> None:
    requirements = (
        builtin_augmentation_catalog().get("input.surface.rephrase").bindings[0].requirements
    )

    assert requirements.semantic_model is True
    assert requirements.sandbox is True
    assert requirements.state_observation is True
    assert requirements.customer_evaluator is False


def test_catalog_rejects_duplicate_references() -> None:
    spec = BuiltinAugmentationSpec(
        ref=AugmentationRef(id="input.example", version="1.0.0"),
        scope="input",
        summary="Example augmentation.",
        bindings=(
            AugmentationBinding(
                mode="scenario_materialization",
                stages=("materialization",),
                execution_owner="augmentation_registry",
            ),
        ),
    )

    with pytest.raises(ValidationError, match="duplicate ID and version"):
        BuiltinAugmentationCatalog(augmentations=(spec, spec))


@pytest.mark.parametrize(
    "reference",
    (
        {"id": "single", "version": "1.0.0"},
        {"id": "Input.example", "version": "1.0.0"},
        {"id": "input..example", "version": "1.0.0"},
        {"id": "input.example", "version": "01.0.0"},
        {"id": "input.example", "version": "1.0"},
    ),
)
def test_augmentation_reference_requires_canonical_bounded_values(
    reference: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        AugmentationRef.model_validate(reference)


def test_requirements_reject_capabilities_without_a_sandbox() -> None:
    with pytest.raises(ValidationError, match="capabilities require a sandbox"):
        AugmentationRequirements(sandbox_capabilities=("tool.timeout@1.0.0",))


def test_binding_rejects_owner_mode_mismatch_and_unsafe_sandbox_fault() -> None:
    with pytest.raises(ValidationError, match="mode and execution owner"):
        AugmentationBinding(
            mode="sandbox_fault",
            stages=("execution",),
            execution_owner="dataset_cli",
            command="ul dataset evaluate",
        )
    with pytest.raises(ValidationError, match="require a sandbox capability"):
        AugmentationBinding(
            mode="sandbox_fault",
            stages=("execution",),
            execution_owner="stress_cli",
            command="ul stress timeout-after-commit",
        )


def test_catalog_resolves_latest_and_rejects_unknown_versions() -> None:
    catalog = builtin_augmentation_catalog()

    assert catalog.get("input.surface.rephrase").ref.version == "1.0.0"
    with pytest.raises(KeyError):
        catalog.get("input.surface.rephrase", "2.0.0")


def test_mode_and_cli_filters_apply_to_the_same_binding() -> None:
    catalog = builtin_augmentation_catalog()

    assert catalog.list(mode="scenario_materialization", cli_only=True) == ()
    assert tuple(
        augmentation.ref.id for augmentation in catalog.list(mode="sandbox_fault", cli_only=True)
    ) == ("environment.tool.timeout_after_commit",)


def test_latest_version_is_selected_before_cli_and_mode_filters() -> None:
    version_one = BuiltinAugmentationSpec(
        ref=AugmentationRef(id="input.example", version="1.0.0"),
        scope="input",
        summary="Old CLI-backed implementation.",
        bindings=(
            AugmentationBinding(
                mode="dataset_variation",
                stages=("materialization",),
                execution_owner="dataset_cli",
                command="ul dataset evaluate --operator input.example",
            ),
        ),
    )
    version_two = BuiltinAugmentationSpec(
        ref=AugmentationRef(id="input.example", version="2.0.0"),
        scope="input",
        summary="Current SDK-only implementation.",
        bindings=(
            AugmentationBinding(
                mode="scenario_materialization",
                stages=("materialization",),
                execution_owner="augmentation_registry",
            ),
        ),
    )
    catalog = BuiltinAugmentationCatalog(augmentations=(version_one, version_two))

    assert catalog.list(mode="dataset_variation", cli_only=True) == ()
    assert catalog.list(mode="dataset_variation", cli_only=True, latest_only=False) == (
        version_one,
    )
