import pytest
from pydantic import ValidationError
from ul_core.augmentations.definitions import (
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

    assert len(references) == 21
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
        "environment_fault",
    )
    assert timeout.bindings[0].cli_available is False
    assert timeout.bindings[0].execution_owner == "augmentation_registry"
    assert timeout.bindings[1].command == "ul stress timeout-after-commit"
    assert timeout.bindings[1].requirements.environment_capabilities == (
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
    assert requirements.environment is True
    assert requirements.state_observation is True
    assert requirements.customer_evaluator is False


def test_dataset_applicability_contracts_are_discoverable_before_execution() -> None:
    catalog = builtin_augmentation_catalog()
    broad = catalog.get("input.surface.grammar_error")
    case_variation = catalog.get("input.surface.case_variation")
    punctuation = catalog.get("input.surface.punctuation_noise")
    conditional = catalog.get("input.intent.self_correction")

    assert broad.applicability_profile == "broad"
    assert "nonempty user input" in broad.applicability_rule
    assert case_variation.applicability_profile == "conditional"
    assert "unprotected Unicode letter" in case_variation.applicability_rule
    assert punctuation.applicability_profile == "conditional"
    assert "protected semantic value" in punctuation.applicability_rule
    assert conditional.applicability_profile == "conditional"
    assert "numeric, monetary, date, or duration" in conditional.applicability_rule


def test_catalog_rejects_duplicate_references() -> None:
    spec = BuiltinAugmentationSpec(
        ref=AugmentationRef(id="input.example", version="1.0.0"),
        surface="task_semantics",
        scope="input",
        summary="Example augmentation.",
        expected_relation="Only the declared example change may differ.",
        applicability_profile="conditional",
        applicability_rule="Applies to examples.",
        bindings=(
            AugmentationBinding(
                mode="scenario_materialization",
                stages=("materialization",),
                execution_owner="augmentation_registry",
                runtime="example.module:ExampleAugmentation",
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


def test_requirements_reject_capabilities_without_an_environment() -> None:
    with pytest.raises(ValidationError, match="capabilities require an environment"):
        AugmentationRequirements(environment_capabilities=("tool.timeout@1.0.0",))


def test_binding_rejects_owner_mode_mismatch_and_unsafe_environment_fault() -> None:
    with pytest.raises(ValidationError, match="mode and execution owner"):
        AugmentationBinding(
            mode="environment_fault",
            stages=("execution",),
            execution_owner="dataset_cli",
            runtime="example.module:run",
            command="ul dataset evaluate",
        )
    with pytest.raises(ValidationError, match="require an environment capability"):
        AugmentationBinding(
            mode="environment_fault",
            stages=("execution",),
            execution_owner="stress_cli",
            runtime="example.module:run",
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
        augmentation.ref.id
        for augmentation in catalog.list(mode="environment_fault", cli_only=True)
    ) == ("environment.tool.timeout_after_commit",)


def test_latest_version_is_selected_before_cli_and_mode_filters() -> None:
    version_one = BuiltinAugmentationSpec(
        ref=AugmentationRef(id="input.example", version="1.0.0"),
        surface="task_semantics",
        scope="input",
        summary="Old CLI-backed implementation.",
        expected_relation="Only the declared example change may differ.",
        applicability_profile="broad",
        applicability_rule="Applies to nonempty inputs.",
        bindings=(
            AugmentationBinding(
                mode="dataset_variation",
                stages=("materialization",),
                execution_owner="dataset_cli",
                runtime="example.module:resolve",
                command="ul dataset evaluate --operator input.example",
            ),
        ),
    )
    version_two = BuiltinAugmentationSpec(
        ref=AugmentationRef(id="input.example", version="2.0.0"),
        surface="task_semantics",
        scope="input",
        summary="Current SDK-only implementation.",
        expected_relation="Only the declared example change may differ.",
        applicability_profile="conditional",
        applicability_rule="Applies to example scenarios.",
        bindings=(
            AugmentationBinding(
                mode="scenario_materialization",
                stages=("materialization",),
                execution_owner="augmentation_registry",
                runtime="example.module:ExampleAugmentation",
            ),
        ),
    )
    catalog = BuiltinAugmentationCatalog(augmentations=(version_one, version_two))

    assert catalog.list(mode="dataset_variation", cli_only=True) == ()
    assert catalog.list(mode="dataset_variation", cli_only=True, latest_only=False) == (
        version_one,
    )
