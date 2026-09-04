import pytest
from ul.augmentations._dataset_operators import (
    _operators_from_specifications,
    dataset_operator_prompt_name,
    dataset_operator_runtime,
)
from ul.augmentations.dataset import (
    builtin_dataset_augmentation_operators,
    resolve_dataset_augmentation_operator,
)
from ul_core.augmentations.definitions import (
    AugmentationBinding,
    BuiltinAugmentationSpec,
    builtin_augmentation_catalog,
)
from ul_core.prompts import PromptManager


def _dataset_catalog_entries() -> tuple[tuple[BuiltinAugmentationSpec, AugmentationBinding], ...]:
    entries = tuple(
        (specification, binding)
        for specification in builtin_augmentation_catalog().list(mode="dataset_variation")
        for binding in specification.bindings
        if binding.mode == "dataset_variation"
    )

    def order(entry: tuple[BuiltinAugmentationSpec, AugmentationBinding]) -> tuple[int, str, str]:
        specification, binding = entry
        assert binding.dataset_runtime is not None
        return binding.dataset_runtime.order, specification.ref.id, specification.ref.version

    return tuple(sorted(entries, key=order))


def test_dataset_operators_are_derived_from_catalog_in_order() -> None:
    catalog_entries = _dataset_catalog_entries()
    operators = builtin_dataset_augmentation_operators()

    assert tuple((operator.id, operator.version) for operator in operators) == tuple(
        (specification.ref.id, specification.ref.version)
        for specification, _binding in catalog_entries
    )
    for operator, (specification, binding) in zip(operators, catalog_entries, strict=True):
        runtime = binding.dataset_runtime
        assert runtime is not None
        assert dataset_operator_runtime(operator) == runtime
        assert operator.model_dump(exclude={"instruction"}) == {
            "id": specification.ref.id,
            "version": specification.ref.version,
            "applicability_profile": specification.applicability_profile,
            "applicability_rule": specification.applicability_rule,
            "generation_mechanism": runtime.generation_mechanism,
            "allowed_change": runtime.allowed_change,
            "target_communication_kind": runtime.target_communication_kind,
            "human_review_required": binding.requirements.human_review,
        }


def test_every_dataset_operator_uses_its_catalog_named_prompt() -> None:
    prompts = PromptManager.instance()

    for operator in builtin_dataset_augmentation_operators():
        prompt_name = dataset_operator_prompt_name(operator)
        assert prompts.get_template_info(prompt_name).name == prompt_name
        assert operator.instruction == prompts.get_prompt(prompt_name)


def _versioned_rephrase_specification(
    version: str, prompt_name: str | None
) -> BuiltinAugmentationSpec:
    specification = builtin_augmentation_catalog().get("input.surface.rephrase")
    binding = next(
        binding for binding in specification.bindings if binding.mode == "dataset_variation"
    )
    assert binding.dataset_runtime is not None
    versioned_binding = binding.model_copy(
        update={
            "dataset_runtime": binding.dataset_runtime.model_copy(
                update={"prompt_name": prompt_name}
            )
        }
    )
    return specification.model_copy(
        update={
            "ref": specification.ref.model_copy(update={"version": version}),
            "bindings": (versioned_binding,),
        }
    )


def test_default_operators_select_only_latest_version_with_its_prompt() -> None:
    version_one = _versioned_rephrase_specification("1.0.0", "augmentation.input.surface.rephrase")
    version_two = _versioned_rephrase_specification("2.0.0", "augmentation.input.style.terse")

    default_operators = _operators_from_specifications((version_one, version_two), latest_only=True)
    historical_operators = _operators_from_specifications(
        (version_one, version_two), latest_only=False
    )

    assert tuple(operator.version for operator in default_operators) == ("2.0.0",)
    assert tuple(operator.version for operator in historical_operators) == ("1.0.0", "2.0.0")
    assert default_operators[0].instruction == PromptManager.instance().get_prompt(
        "augmentation.input.style.terse"
    )
    assert historical_operators[0].instruction != historical_operators[1].instruction


def test_versioned_operators_require_explicit_prompt_identities() -> None:
    version_one = _versioned_rephrase_specification("1.0.0", None)
    version_two = _versioned_rephrase_specification("2.0.0", None)

    with pytest.raises(
        ValueError,
        match=r"^versioned dataset augmentations require explicit prompt identities$",
    ):
        _operators_from_specifications((version_one, version_two), latest_only=True)


@pytest.mark.parametrize(
    "reference",
    (
        "",
        "unknown.dataset.operator",
        "input.surface.rephrase@",
        "@1.0.0",
        "input.surface.rephrase@1.0.0@2.0.0",
    ),
)
def test_dataset_operator_resolution_rejects_invalid_references(reference: str) -> None:
    with pytest.raises(ValueError, match=r"^unknown dataset augmentation reference$"):
        resolve_dataset_augmentation_operator(reference)


def test_dataset_operator_resolution_rejects_an_unknown_version() -> None:
    operator = builtin_dataset_augmentation_operators()[0]

    with pytest.raises(ValueError, match=r"^unknown dataset augmentation reference$"):
        resolve_dataset_augmentation_operator(f"{operator.id}@999.0.0")
