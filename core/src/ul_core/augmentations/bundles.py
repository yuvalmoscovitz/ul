from __future__ import annotations

import hashlib
import json
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from ul_core.augmentations.definitions import (
    AugmentationMode,
    AugmentationRef,
    AugmentationRequirements,
    BuiltinAugmentationCatalog,
    builtin_augmentation_catalog,
)
from ul_core.models import ULModel

BundleMutationRisk = Literal["none", "state", "fault"]
BundlePlanStatus = Literal["planned", "skipped"]

_BUNDLE_ID_PATTERN = r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$"


class AugmentationBundleBudget(ULModel):
    maximum_cases: int = Field(ge=1, le=100)
    maximum_fan_out_per_case: int = Field(ge=1, le=100)
    maximum_model_calls: int = Field(ge=0)
    maximum_target_calls: int = Field(ge=0)
    maximum_duration_seconds: int = Field(ge=1)
    maximum_cost_usd: float = Field(ge=0)
    maximum_mutating_probes: int = Field(ge=0)


class BundleOperatorPolicy(ULModel):
    ref: AugmentationRef
    mode: AugmentationMode
    model_calls: int = Field(ge=0)
    target_calls: int = Field(ge=0)
    maximum_duration_seconds: int = Field(ge=1)
    maximum_cost_usd: float = Field(ge=0)
    mutation_risk: BundleMutationRisk = "none"
    reset_required: bool = False

    @model_validator(mode="after")
    def validate_mutation_controls(self) -> Self:
        if self.mutation_risk != "none" and not self.reset_required:
            raise ValueError("mutating bundle operators require reset")
        return self


class AugmentationBundle(ULModel):
    id: str = Field(min_length=3, max_length=100, pattern=_BUNDLE_ID_PATTERN)
    version: str = Field(pattern=r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
    summary: str = Field(min_length=1, max_length=500)
    composition: Literal["independent_only"] = "independent_only"
    operators: tuple[BundleOperatorPolicy, ...] = Field(min_length=1)
    budget: AugmentationBundleBudget


class BundleSourceCase(ULModel):
    id: str = Field(min_length=1, max_length=500)
    available_features: frozenset[str] = frozenset()


class BundleProbePlan(ULModel):
    source_case_id: str
    source_parent_id: str
    operator: AugmentationRef
    mode: AugmentationMode
    status: BundlePlanStatus
    applicability: Literal["broad", "conditional", "ineligible"]
    reasons: tuple[str, ...]
    changed_surfaces: tuple[str, ...]
    expected_relation: str
    required_evidence: tuple[str, ...]
    model_calls: int = Field(ge=0)
    target_calls: int = Field(ge=0)
    maximum_duration_seconds: int = Field(ge=0)
    maximum_cost_usd: float = Field(ge=0)
    mutation_risk: BundleMutationRisk
    reset_required: bool
    canonical_id: str = Field(pattern=r"^[0-9a-f]{64}$")


class BundlePlanTotals(ULModel):
    cases: int = Field(ge=1)
    planned_probes: int = Field(ge=0)
    skipped_probes: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    target_calls: int = Field(ge=0)
    maximum_duration_seconds: int = Field(ge=0)
    maximum_cost_usd: float = Field(ge=0)
    mutating_probes: int = Field(ge=0)


class AugmentationBundlePlan(ULModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    bundle_id: str
    bundle_version: str
    composition: Literal["independent_only"] = "independent_only"
    probes: tuple[BundleProbePlan, ...]
    totals: BundlePlanTotals
    inspection_model_calls: Literal[0] = 0
    inspection_target_calls: Literal[0] = 0
    inspection_network_requests: Literal[0] = 0


class AugmentationBundleCatalog(ULModel):
    bundles: tuple[AugmentationBundle, ...]

    @field_validator("bundles")
    @classmethod
    def sort_bundles(
        cls, bundles: tuple[AugmentationBundle, ...]
    ) -> tuple[AugmentationBundle, ...]:
        return tuple(sorted(bundles, key=lambda bundle: (bundle.id, bundle.version)))

    @model_validator(mode="after")
    def validate_unique_bundles(self) -> Self:
        identities = tuple((bundle.id, bundle.version) for bundle in self.bundles)
        if len(identities) != len(set(identities)):
            raise ValueError("bundle catalog contains a duplicate ID and version")
        return self

    def list(self) -> tuple[AugmentationBundle, ...]:
        latest: dict[str, AugmentationBundle] = {}
        for bundle in self.bundles:
            previous = latest.get(bundle.id)
            if previous is None or _version_tuple(bundle.version) > _version_tuple(
                previous.version
            ):
                latest[bundle.id] = bundle
        return tuple(latest[bundle_id] for bundle_id in sorted(latest))

    def get(self, bundle_id: str, version: str | None = None) -> AugmentationBundle:
        matches = tuple(bundle for bundle in self.bundles if bundle.id == bundle_id)
        if not matches:
            raise KeyError(bundle_id)
        if version is None:
            return max(
                matches,
                key=lambda bundle: tuple(int(part) for part in bundle.version.split(".")),
            )
        for bundle in matches:
            if bundle.version == version:
                return bundle
        raise KeyError(f"{bundle_id}@{version}")


def plan_augmentation_bundle(
    bundle: AugmentationBundle,
    source_cases: tuple[BundleSourceCase, ...],
    *,
    catalog: BuiltinAugmentationCatalog | None = None,
) -> AugmentationBundlePlan:
    if not source_cases:
        raise ValueError("bundle planning requires at least one source case")
    if len(source_cases) > bundle.budget.maximum_cases:
        raise ValueError("bundle case count exceeds its hard maximum")
    source_ids = tuple(source.id for source in source_cases)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("bundle source case identifiers must be unique")

    definitions = catalog or builtin_augmentation_catalog()
    probes: list[BundleProbePlan] = []
    canonical_ids: set[str] = set()
    for source in source_cases:
        fan_out = 0
        for policy in bundle.operators:
            try:
                definition = definitions.get(policy.ref.id, policy.ref.version)
            except KeyError:
                raise ValueError("bundle references an unknown augmentation") from None
            binding = next(
                (binding for binding in definition.bindings if binding.mode == policy.mode),
                None,
            )
            if binding is None:
                raise ValueError("bundle references an unavailable augmentation binding")
            missing_features = tuple(
                feature
                for feature in binding.requirements.required_source_features
                if feature not in source.available_features
            )
            status: BundlePlanStatus = "skipped" if missing_features else "planned"
            canonical_id = _canonical_probe_id(
                source.id,
                policy.ref,
                policy.mode,
                binding.projection.writes,
            )
            if canonical_id in canonical_ids:
                continue
            canonical_ids.add(canonical_id)
            if status == "planned":
                fan_out += 1
                if fan_out > bundle.budget.maximum_fan_out_per_case:
                    raise ValueError("bundle fan-out exceeds its hard per-case maximum")
            reasons = (
                ("missing source features: " + ", ".join(missing_features),)
                if missing_features
                else (definition.applicability_rule,)
            )
            probes.append(
                BundleProbePlan(
                    source_case_id=source.id,
                    source_parent_id=source.id,
                    operator=policy.ref,
                    mode=policy.mode,
                    status=status,
                    applicability=(
                        "ineligible" if missing_features else definition.applicability_profile
                    ),
                    reasons=reasons,
                    changed_surfaces=tuple(binding.projection.writes),
                    expected_relation=definition.expected_relation,
                    required_evidence=_required_evidence(binding.requirements),
                    model_calls=policy.model_calls if status == "planned" else 0,
                    target_calls=policy.target_calls if status == "planned" else 0,
                    maximum_duration_seconds=(
                        policy.maximum_duration_seconds if status == "planned" else 0
                    ),
                    maximum_cost_usd=policy.maximum_cost_usd if status == "planned" else 0,
                    mutation_risk=policy.mutation_risk,
                    reset_required=policy.reset_required,
                    canonical_id=canonical_id,
                )
            )

    planned = tuple(probe for probe in probes if probe.status == "planned")
    totals = BundlePlanTotals(
        cases=len(source_cases),
        planned_probes=len(planned),
        skipped_probes=len(probes) - len(planned),
        model_calls=sum(probe.model_calls for probe in planned),
        target_calls=sum(probe.target_calls for probe in planned),
        maximum_duration_seconds=sum(probe.maximum_duration_seconds for probe in planned),
        maximum_cost_usd=round(sum(probe.maximum_cost_usd for probe in planned), 6),
        mutating_probes=sum(probe.mutation_risk != "none" for probe in planned),
    )
    _enforce_budget(bundle.budget, totals)
    return AugmentationBundlePlan(
        bundle_id=bundle.id,
        bundle_version=bundle.version,
        probes=tuple(probes),
        totals=totals,
    )


def builtin_augmentation_bundle_catalog() -> AugmentationBundleCatalog:
    return AugmentationBundleCatalog(bundles=_BUILTIN_BUNDLES)


def _required_evidence(requirements: AugmentationRequirements) -> tuple[str, ...]:
    evidence: list[str] = ["response"]
    if requirements.state_observation:
        evidence.append("authoritative_state")
    if requirements.customer_evaluator:
        evidence.append("customer_evaluator")
    if requirements.human_review:
        evidence.append("human_review")
    return tuple(evidence)


def _canonical_probe_id(
    source_case_id: str,
    ref: AugmentationRef,
    mode: AugmentationMode,
    changed_surfaces: tuple[str, ...],
) -> str:
    payload = json.dumps(
        {
            "source_case_id": source_case_id,
            "operator": ref.model_dump(mode="json"),
            "mode": mode,
            "changed_surfaces": changed_surfaces,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _enforce_budget(budget: AugmentationBundleBudget, totals: BundlePlanTotals) -> None:
    limits = (
        (totals.model_calls, budget.maximum_model_calls, "model calls"),
        (totals.target_calls, budget.maximum_target_calls, "target calls"),
        (
            totals.maximum_duration_seconds,
            budget.maximum_duration_seconds,
            "duration",
        ),
        (totals.maximum_cost_usd, budget.maximum_cost_usd, "cost"),
        (totals.mutating_probes, budget.maximum_mutating_probes, "mutating probes"),
    )
    for actual, maximum, name in limits:
        if actual > maximum:
            raise ValueError(f"bundle {name} exceed its hard maximum")


def _policy(
    augmentation_id: str,
    *,
    mode: AugmentationMode,
    model_calls: int,
    target_calls: int = 1,
    maximum_duration_seconds: int = 90,
    maximum_cost_usd: float = 0.25,
    mutation_risk: BundleMutationRisk = "none",
    reset_required: bool = False,
) -> BundleOperatorPolicy:
    return BundleOperatorPolicy(
        ref=AugmentationRef(id=augmentation_id, version="1.0.0"),
        mode=mode,
        model_calls=model_calls,
        target_calls=target_calls,
        maximum_duration_seconds=maximum_duration_seconds,
        maximum_cost_usd=maximum_cost_usd,
        mutation_risk=mutation_risk,
        reset_required=reset_required,
    )


_DEFAULT_BUDGET = AugmentationBundleBudget(
    maximum_cases=10,
    maximum_fan_out_per_case=3,
    maximum_model_calls=120,
    maximum_target_calls=30,
    maximum_duration_seconds=2_700,
    maximum_cost_usd=7.5,
    maximum_mutating_probes=30,
)

_BUILTIN_BUNDLES = (
    AugmentationBundle(
        id="everyday-customers",
        version="1.0.0",
        summary="Probe ordinary wording, typing, and information-density variation.",
        operators=(
            _policy(
                "input.surface.typing_noise",
                mode="dataset_variation",
                model_calls=3,
                mutation_risk="state",
                reset_required=True,
            ),
            _policy(
                "input.style.terse",
                mode="dataset_variation",
                model_calls=4,
                mutation_risk="state",
                reset_required=True,
            ),
            _policy(
                "input.style.verbose",
                mode="dataset_variation",
                model_calls=4,
                mutation_risk="state",
                reset_required=True,
            ),
        ),
        budget=_DEFAULT_BUDGET,
    ),
    AugmentationBundle(
        id="unclear-changing-requests",
        version="1.0.0",
        summary="Probe ambiguous requests and corrections without chaining transformations.",
        operators=(
            _policy(
                "input.intent.self_correction",
                mode="dataset_variation",
                model_calls=4,
                mutation_risk="state",
                reset_required=True,
            ),
            _policy(
                "conversation.ambiguity",
                mode="scenario_materialization",
                model_calls=0,
                target_calls=0,
            ),
            _policy(
                "conversation.correction_after_first_response",
                mode="conversation_stress",
                model_calls=0,
                mutation_risk="state",
                reset_required=True,
            ),
        ),
        budget=_DEFAULT_BUDGET,
    ),
    AugmentationBundle(
        id="retries-interrupted-work",
        version="1.0.0",
        summary="Probe retries and uncertain commits with explicit reset requirements.",
        operators=(
            _policy(
                "conversation.retry_after_successful_commit",
                mode="conversation_stress",
                model_calls=0,
                mutation_risk="state",
                reset_required=True,
            ),
            _policy(
                "environment.tool.timeout_before_commit",
                mode="scenario_materialization",
                model_calls=0,
                target_calls=0,
            ),
            _policy(
                "environment.tool.timeout_after_commit",
                mode="environment_fault",
                model_calls=0,
                mutation_risk="fault",
                reset_required=True,
            ),
        ),
        budget=_DEFAULT_BUDGET,
    ),
)


def _version_tuple(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)
