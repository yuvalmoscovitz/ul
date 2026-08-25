import pytest
from ul_core.augmentations.bundles import (
    AugmentationBundle,
    AugmentationBundleBudget,
    BundleOperatorPolicy,
    BundleSourceCase,
    builtin_augmentation_bundle_catalog,
    plan_augmentation_bundle,
)
from ul_core.augmentations.definitions import AugmentationRef


def test_builtin_bundles_are_versioned_selection_and_budget_policies() -> None:
    catalog = builtin_augmentation_bundle_catalog()

    assert [bundle.id for bundle in catalog.list()] == [
        "everyday-customers",
        "retries-interrupted-work",
        "unclear-changing-requests",
    ]
    everyday = catalog.get("everyday-customers")
    assert everyday.version == "1.0.0"
    assert everyday.composition == "independent_only"
    assert [policy.ref.id for policy in everyday.operators] == [
        "input.surface.typing_noise",
        "input.style.terse",
        "input.style.verbose",
    ]
    assert everyday.budget.maximum_cases == 10
    assert everyday.budget.maximum_fan_out_per_case == 3


def test_bundle_plan_fans_out_independently_and_exposes_bounded_work() -> None:
    bundle = builtin_augmentation_bundle_catalog().get("everyday-customers")

    plan = plan_augmentation_bundle(
        bundle,
        (
            BundleSourceCase(
                id="case-1",
                available_features=frozenset(("production interaction",)),
            ),
            BundleSourceCase(
                id="case-2",
                available_features=frozenset(("production interaction",)),
            ),
        ),
    )

    assert plan.composition == "independent_only"
    assert len(plan.probes) == 6
    assert all(probe.source_parent_id == probe.source_case_id for probe in plan.probes)
    assert all(probe.status == "planned" for probe in plan.probes)
    assert all(
        probe.changed_surfaces == ("structured_input", "conversation") for probe in plan.probes
    )
    assert len({probe.canonical_id for probe in plan.probes}) == 6
    assert plan.totals.model_calls == 22
    assert plan.totals.target_calls == 6
    assert plan.totals.maximum_duration_seconds == 540
    assert plan.totals.maximum_cost_usd == 1.5
    assert plan.totals.mutating_probes == 6
    assert all(probe.reset_required for probe in plan.probes)
    assert plan.inspection_model_calls == 0
    assert plan.inspection_target_calls == 0


def test_bundle_plan_reports_skips_and_deduplicates_canonical_probes() -> None:
    policy = BundleOperatorPolicy(
        ref=AugmentationRef(id="input.intent.self_correction", version="1.0.0"),
        mode="dataset_variation",
        model_calls=4,
        target_calls=1,
        maximum_duration_seconds=90,
        maximum_cost_usd=0.25,
    )
    bundle = AugmentationBundle(
        id="duplicate-selection",
        version="1.0.0",
        summary="A test bundle.",
        operators=(policy, policy),
        budget=_budget(maximum_fan_out_per_case=1),
    )

    plan = plan_augmentation_bundle(bundle, (BundleSourceCase(id="case-1"),))

    assert len(plan.probes) == 1
    assert plan.probes[0].status == "skipped"
    assert plan.probes[0].applicability == "ineligible"
    assert "production interaction" in plan.probes[0].reasons[0]
    assert plan.totals.planned_probes == 0
    assert plan.totals.skipped_probes == 1


def test_bundle_plan_enforces_every_hard_bound() -> None:
    policy = BundleOperatorPolicy(
        ref=AugmentationRef(id="input.surface.typing_noise", version="1.0.0"),
        mode="dataset_variation",
        model_calls=3,
        target_calls=1,
        maximum_duration_seconds=90,
        maximum_cost_usd=0.25,
    )
    source = BundleSourceCase(
        id="case-1",
        available_features=frozenset(("production interaction",)),
    )
    overrides = (
        ("maximum_model_calls", 2, "model calls"),
        ("maximum_target_calls", 0, "target calls"),
        ("maximum_duration_seconds", 89, "duration"),
        ("maximum_cost_usd", 0.24, "cost"),
    )
    for field, limit, expected in overrides:
        budget = _budget().model_copy(update={field: limit})
        bundle = AugmentationBundle(
            id="bounded-bundle",
            version="1.0.0",
            summary="A test bundle.",
            operators=(policy,),
            budget=budget,
        )

        with pytest.raises(ValueError, match=expected):
            plan_augmentation_bundle(bundle, (source,))


def test_bundle_plan_rejects_chained_composition_and_mutation_without_reset() -> None:
    with pytest.raises(ValueError, match="independent_only"):
        AugmentationBundle.model_validate(
            {
                "id": "chained-bundle",
                "version": "1.0.0",
                "summary": "Invalid composition.",
                "composition": "chained",
                "operators": [],
                "budget": _budget().model_dump(mode="json"),
            }
        )
    with pytest.raises(ValueError, match="require reset"):
        BundleOperatorPolicy(
            ref=AugmentationRef(id="input.surface.typing_noise", version="1.0.0"),
            mode="dataset_variation",
            model_calls=0,
            target_calls=1,
            maximum_duration_seconds=10,
            maximum_cost_usd=0,
            mutation_risk="state",
        )


def test_bundle_plan_enforces_fan_out_and_mutation_bounds() -> None:
    source = BundleSourceCase(
        id="case-1",
        available_features=frozenset(("production interaction",)),
    )
    typing = BundleOperatorPolicy(
        ref=AugmentationRef(id="input.surface.typing_noise", version="1.0.0"),
        mode="dataset_variation",
        model_calls=3,
        target_calls=1,
        maximum_duration_seconds=90,
        maximum_cost_usd=0.25,
        mutation_risk="state",
        reset_required=True,
    )
    terse = typing.model_copy(
        update={"ref": AugmentationRef(id="input.style.terse", version="1.0.0")}
    )
    fan_out_bundle = AugmentationBundle(
        id="fan-out-bound",
        version="1.0.0",
        summary="A test bundle.",
        operators=(typing, terse),
        budget=_budget(maximum_fan_out_per_case=1),
    )
    mutation_bundle = fan_out_bundle.model_copy(
        update={
            "id": "mutation-bound",
            "operators": (typing,),
            "budget": _budget().model_copy(update={"maximum_mutating_probes": 0}),
        }
    )

    with pytest.raises(ValueError, match="fan-out"):
        plan_augmentation_bundle(fan_out_bundle, (source,))
    with pytest.raises(ValueError, match="mutating probes"):
        plan_augmentation_bundle(mutation_bundle, (source,))


def _budget(*, maximum_fan_out_per_case: int = 3) -> AugmentationBundleBudget:
    return AugmentationBundleBudget(
        maximum_cases=10,
        maximum_fan_out_per_case=maximum_fan_out_per_case,
        maximum_model_calls=100,
        maximum_target_calls=100,
        maximum_duration_seconds=1_000,
        maximum_cost_usd=100,
        maximum_mutating_probes=100,
    )
