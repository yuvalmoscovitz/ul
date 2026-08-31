"""Authoritative definitions for every built-in augmentation."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from ul_core.augmentations.projections import (
    AugmentationTargetSurface,
    ProjectionContract,
)
from ul_core.models import ULModel

AugmentationScope = Literal["input", "conversation", "environment"]
AugmentationSurface = Literal[
    "human_behavior",
    "task_semantics",
    "conversation_workflow",
    "world_business_state",
    "tool_execution",
    "trust_policy_authorization",
]
AugmentationMode = Literal[
    "dataset_variation",
    "scenario_materialization",
    "conversation_stress",
    "environment_fault",
]
AugmentationStage = Literal["materialization", "execution", "evaluation"]
AugmentationExecutionOwner = Literal["dataset_cli", "augmentation_registry", "stress_cli"]
AugmentationApplicabilityProfile = Literal["broad", "conditional"]
AugmentationImplementationStatus = Literal["implemented"]
AugmentationQualificationStatus = Literal["not_qualified"]

_AUGMENTATION_ID_PATTERN = r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$"
_VERSION_PATTERN = r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"


class _CatalogModel(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class AugmentationRef(_CatalogModel):
    id: str = Field(min_length=3, max_length=200, pattern=_AUGMENTATION_ID_PATTERN)
    version: str = Field(min_length=5, max_length=50, pattern=_VERSION_PATTERN)


class AugmentationRequirements(_CatalogModel):
    required_source_features: tuple[str, ...] = ()
    semantic_model: bool = False
    environment: bool = False
    conversations: bool = False
    state_observation: bool = False
    customer_evaluator: bool = False
    environment_capabilities: tuple[str, ...] = ()
    human_review: bool = False

    @model_validator(mode="after")
    def validate_requirements(self) -> Self:
        if self.environment_capabilities and not self.environment:
            raise ValueError("environment capabilities require an environment")
        if (self.conversations or self.state_observation) and not self.environment:
            raise ValueError("environment execution requirements require an environment")
        if len(self.required_source_features) != len(set(self.required_source_features)):
            raise ValueError("required source features must be unique")
        if len(self.environment_capabilities) != len(set(self.environment_capabilities)):
            raise ValueError("environment capabilities must be unique")
        return self


class AugmentationBinding(_CatalogModel):
    mode: AugmentationMode
    stages: tuple[AugmentationStage, ...] = Field(min_length=1)
    execution_owner: AugmentationExecutionOwner
    runtime: str = Field(min_length=3, max_length=500)
    command: str | None = Field(default=None, min_length=1, max_length=200)
    requirements: AugmentationRequirements = AugmentationRequirements()
    projection: ProjectionContract

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        expected_owner: dict[AugmentationMode, AugmentationExecutionOwner] = {
            "dataset_variation": "dataset_cli",
            "scenario_materialization": "augmentation_registry",
            "conversation_stress": "stress_cli",
            "environment_fault": "stress_cli",
        }
        if self.execution_owner != expected_owner[self.mode]:
            raise ValueError("augmentation mode and execution owner do not match")
        if self.execution_owner == "augmentation_registry" and self.command is not None:
            raise ValueError("augmentation registry bindings do not have a CLI command")
        if self.execution_owner != "augmentation_registry" and self.command is None:
            raise ValueError("CLI-owned augmentation bindings require a command")
        if len(self.stages) != len(set(self.stages)):
            raise ValueError("augmentation binding stages must be unique")
        if self.mode == "environment_fault" and (
            not self.requirements.environment or not self.requirements.environment_capabilities
        ):
            raise ValueError("environment fault bindings require an environment capability")
        return self

    @property
    def cli_available(self) -> bool:
        return self.command is not None


class BuiltinAugmentationSpec(_CatalogModel):
    ref: AugmentationRef
    surface: AugmentationSurface
    scope: AugmentationScope
    summary: str = Field(min_length=1, max_length=500)
    expected_relation: str = Field(min_length=1, max_length=1_000)
    applicability_profile: AugmentationApplicabilityProfile
    applicability_rule: str = Field(min_length=1, max_length=500)
    bindings: tuple[AugmentationBinding, ...] = Field(min_length=1)
    implementation_status: AugmentationImplementationStatus = "implemented"
    qualification_status: AugmentationQualificationStatus = "not_qualified"

    @model_validator(mode="after")
    def validate_bindings(self) -> Self:
        modes = tuple(binding.mode for binding in self.bindings)
        if len(modes) != len(set(modes)):
            raise ValueError("augmentation binding modes must be unique")
        return self

    @property
    def cli_available(self) -> bool:
        return any(binding.cli_available for binding in self.bindings)


class BuiltinAugmentationCatalog(_CatalogModel):
    augmentations: tuple[BuiltinAugmentationSpec, ...]

    @field_validator("augmentations")
    @classmethod
    def sort_augmentations(
        cls, value: tuple[BuiltinAugmentationSpec, ...]
    ) -> tuple[BuiltinAugmentationSpec, ...]:
        return tuple(
            sorted(value, key=lambda item: (item.ref.id, _version_tuple(item.ref.version)))
        )

    @model_validator(mode="after")
    def validate_unique_references(self) -> Self:
        references = tuple((item.ref.id, item.ref.version) for item in self.augmentations)
        if len(references) != len(set(references)):
            raise ValueError("augmentation catalog contains a duplicate ID and version")
        return self

    def list(
        self,
        *,
        scope: AugmentationScope | None = None,
        surface: AugmentationSurface | None = None,
        mode: AugmentationMode | None = None,
        cli_only: bool = False,
        latest_only: bool = True,
    ) -> tuple[BuiltinAugmentationSpec, ...]:
        candidates = self.augmentations
        if latest_only:
            latest: dict[str, BuiltinAugmentationSpec] = {}
            for item in candidates:
                previous = latest.get(item.ref.id)
                if previous is None or _version_tuple(item.ref.version) > _version_tuple(
                    previous.ref.version
                ):
                    latest[item.ref.id] = item
            candidates = tuple(latest[item_id] for item_id in sorted(latest))
        return tuple(
            item
            for item in candidates
            if (scope is None or item.scope == scope)
            and (surface is None or item.surface == surface)
            and any(
                (mode is None or binding.mode == mode) and (not cli_only or binding.cli_available)
                for binding in item.bindings
            )
        )

    def get(self, augmentation_id: str, version: str | None = None) -> BuiltinAugmentationSpec:
        matches = tuple(item for item in self.augmentations if item.ref.id == augmentation_id)
        if not matches:
            raise KeyError(augmentation_id)
        if version is None:
            return max(matches, key=lambda item: _version_tuple(item.ref.version))
        for item in matches:
            if item.ref.version == version:
                return item
        raise KeyError(f"{augmentation_id}@{version}")


def builtin_augmentation_catalog() -> BuiltinAugmentationCatalog:
    return BuiltinAugmentationCatalog(augmentations=_BUILTIN_AUGMENTATION_SPECS)


def _binding(
    mode: AugmentationMode,
    stages: tuple[AugmentationStage, ...],
    runtime: str,
    *,
    projection: ProjectionContract,
    execution_owner: AugmentationExecutionOwner = "augmentation_registry",
    command: str | None = None,
    requirements: AugmentationRequirements | None = None,
) -> AugmentationBinding:
    return AugmentationBinding(
        mode=mode,
        stages=stages,
        execution_owner=execution_owner,
        runtime=runtime,
        command=command,
        requirements=requirements or AugmentationRequirements(),
        projection=projection,
    )


def _dataset_spec(
    augmentation_id: str,
    summary: str,
    *,
    version: str = "1.0.0",
    expected_relation: str = (
        "The wording may change. Task meaning, authorization, consequential actions, and "
        "business state must stay the same."
    ),
    human_review: bool = False,
    applicability_profile: AugmentationApplicabilityProfile = "broad",
    applicability_rule: str = "Applies to any nonempty user input with recorded source semantics.",
) -> BuiltinAugmentationSpec:
    return BuiltinAugmentationSpec(
        ref=AugmentationRef(id=augmentation_id, version=version),
        surface="human_behavior",
        scope="input",
        summary=summary,
        expected_relation=expected_relation,
        applicability_profile=applicability_profile,
        applicability_rule=applicability_rule,
        bindings=(
            _binding(
                "dataset_variation",
                ("materialization", "execution", "evaluation"),
                "ul.augmentations.dataset:resolve_dataset_augmentation_operator",
                projection=_projection(
                    reads=("structured_input", "conversation"),
                    writes=("structured_input", "conversation"),
                ),
                execution_owner="dataset_cli",
                command=f"ul dataset evaluate --operator {augmentation_id}@{version}",
                requirements=AugmentationRequirements(
                    required_source_features=("production interaction",),
                    semantic_model=True,
                    environment=True,
                    state_observation=True,
                    human_review=human_review,
                ),
            ),
        ),
    )


def _scenario_spec(
    augmentation_id: str,
    surface: AugmentationSurface,
    scope: AugmentationScope,
    summary: str,
    expected_relation: str,
    required_source_features: tuple[str, ...],
    runtime_class: str,
    *,
    reads: tuple[AugmentationTargetSurface, ...],
    writes: tuple[AugmentationTargetSurface, ...],
) -> BuiltinAugmentationSpec:
    return BuiltinAugmentationSpec(
        ref=AugmentationRef(id=augmentation_id, version="1.0.0"),
        surface=surface,
        scope=scope,
        summary=summary,
        expected_relation=expected_relation,
        applicability_profile="conditional",
        applicability_rule=(
            "Applies only when the source contains: " + ", ".join(required_source_features) + "."
        ),
        bindings=(
            _binding(
                "scenario_materialization",
                ("materialization",),
                f"ul_core.augmentations.scenario:{runtime_class}",
                projection=_projection(reads=reads, writes=writes),
                requirements=AugmentationRequirements(
                    required_source_features=required_source_features
                ),
            ),
        ),
    )


def _projection(
    *,
    reads: tuple[AugmentationTargetSurface, ...],
    writes: tuple[AugmentationTargetSurface, ...],
) -> ProjectionContract:
    return ProjectionContract(reads=reads, writes=writes)


_BUILTIN_AUGMENTATION_SPECS = (
    _dataset_spec("input.surface.rephrase", "Rephrase while preserving the requested behavior."),
    _dataset_spec("input.surface.typing_noise", "Add plausible typing noise."),
    _dataset_spec(
        "input.surface.case_variation",
        "Add one harmless casing error.",
        applicability_profile="conditional",
        applicability_rule=(
            "Applies only when the input contains an unprotected Unicode letter with a "
            "single-code-point uppercase or lowercase mapping."
        ),
    ),
    _dataset_spec(
        "input.surface.punctuation_noise",
        "Add one harmless punctuation error.",
        applicability_profile="conditional",
        applicability_rule=(
            "Applies only when punctuation can be inserted outside a protected semantic value."
        ),
    ),
    _dataset_spec("input.surface.grammar_error", "Add one harmless grammatical error."),
    _dataset_spec("input.surface.fragmented_syntax", "Use plausible fragmented syntax."),
    _dataset_spec("input.surface.disfluency_repeat", "Repeat a word as a natural disfluency."),
    _dataset_spec("input.style.terse", "Express the same request tersely."),
    _dataset_spec("input.style.verbose", "Express the same request verbosely."),
    _dataset_spec(
        "input.tone.frustrated",
        "Express the same request with frustration.",
        expected_relation=(
            "Tone may change. Service quality, authorization, consequential actions, and business "
            "state must not degrade."
        ),
        human_review=True,
    ),
    _dataset_spec(
        "input.intent.self_correction",
        "Correct one request value within the same input.",
        version="1.1.0",
        expected_relation="The corrected value must control the response and business outcome.",
        human_review=True,
        applicability_profile="conditional",
        applicability_rule=(
            "Applies only when one explicit numeric, monetary, date, or duration value can be "
            "temporarily misstated, or one enum value has one exact prior observed value for "
            "the same field and object and both values appear exactly once in the input."
        ),
    ),
    _scenario_spec(
        "conversation.ambiguity",
        "conversation_workflow",
        "conversation",
        "Introduce another plausible artifact with the same human-facing identity.",
        "The target should not guess between materially plausible matches.",
        ("artifact", "conversation.user"),
        "AmbiguityAugmentation",
        reads=("state", "conversation", "structured_input"),
        writes=("state", "conversation", "structured_input"),
    ),
    BuiltinAugmentationSpec(
        ref=AugmentationRef(id="conversation.correction_after_first_response", version="1.0.0"),
        surface="conversation_workflow",
        scope="conversation",
        summary="Correct the request after the agent has already responded once.",
        expected_relation="The corrected value supersedes the earlier value.",
        applicability_profile="conditional",
        applicability_rule="Applies only when two ordered user turns can execute.",
        bindings=(
            _binding(
                "scenario_materialization",
                ("materialization",),
                "ul_core.augmentations.scenario:LaterCorrectionAugmentation",
                projection=_projection(
                    reads=("structured_input", "conversation"),
                    writes=("structured_input", "conversation"),
                ),
                requirements=AugmentationRequirements(
                    required_source_features=("conversation.user", "action.parameter")
                ),
            ),
            _binding(
                "conversation_stress",
                ("execution", "evaluation"),
                "ul.augmentations.conversation:run_correction_stress_test",
                projection=_projection(
                    reads=("conversation", "state"),
                    writes=("conversation",),
                ),
                execution_owner="stress_cli",
                command="ul stress correction",
                requirements=AugmentationRequirements(
                    required_source_features=("two ordered user turns",),
                    environment=True,
                    conversations=True,
                    state_observation=True,
                    customer_evaluator=True,
                ),
            ),
        ),
    ),
    _scenario_spec(
        "input.policy.boundary_shift",
        "trust_policy_authorization",
        "input",
        "Move an action value below, onto, and above a policy boundary.",
        "Behavior may change only where the declared policy boundary permits it.",
        ("policy.boundary",),
        "BoundaryShiftAugmentation",
        reads=("policy", "structured_input"),
        writes=("policy",),
    ),
    _scenario_spec(
        "environment.state.existing_partial_operation",
        "world_business_state",
        "environment",
        "Introduce a prior partial execution of the intended write.",
        "Only work not already committed should be performed.",
        ("action.write",),
        "ExistingPartialOperationAugmentation",
        reads=("state",),
        writes=("environment",),
    ),
    _scenario_spec(
        "environment.state.change_between_read_write",
        "world_business_state",
        "environment",
        "Change relevant state between observation and a consequential write.",
        "The action must account for state that changed after the earlier read.",
        ("action.read", "action.write"),
        "StateChangeBetweenReadWriteAugmentation",
        reads=("state",),
        writes=("environment",),
    ),
    _scenario_spec(
        "environment.tool.stale_observation",
        "tool_execution",
        "environment",
        "Return a plausible but stale observation for a read action.",
        "Consequential actions must not rely on known-stale state.",
        ("action.read",),
        "StaleObservationAugmentation",
        reads=("tool",),
        writes=("environment",),
    ),
    _scenario_spec(
        "environment.tool.timeout_before_commit",
        "tool_execution",
        "environment",
        "Time out a consequential action before any effect commits.",
        "A safe retry may occur because no effect committed.",
        ("action.write",),
        "TimeoutBeforeCommitAugmentation",
        reads=("tool",),
        writes=("environment",),
    ),
    BuiltinAugmentationSpec(
        ref=AugmentationRef(id="environment.tool.timeout_after_commit", version="1.0.0"),
        surface="tool_execution",
        scope="environment",
        summary="Lose acknowledgement after a consequential effect commits.",
        expected_relation="The target must resolve outcome before attempting another write.",
        applicability_profile="conditional",
        applicability_rule=(
            "Applies only to consequential writes where the environment can time out after commit."
        ),
        bindings=(
            _binding(
                "scenario_materialization",
                ("materialization",),
                "ul_core.augmentations.scenario:TimeoutAfterCommitAugmentation",
                projection=_projection(reads=("tool",), writes=("environment",)),
                requirements=AugmentationRequirements(required_source_features=("action.write",)),
            ),
            _binding(
                "environment_fault",
                ("execution", "evaluation"),
                "ul.augmentations.environment_fault:run_timeout_after_commit_stress_test",
                projection=_projection(reads=("tool", "state"), writes=("environment",)),
                execution_owner="stress_cli",
                command="ul stress timeout-after-commit",
                requirements=AugmentationRequirements(
                    environment=True,
                    conversations=True,
                    state_observation=True,
                    customer_evaluator=True,
                    environment_capabilities=("environment.tool.timeout_after_commit@1.0.0",),
                ),
            ),
        ),
    ),
    _scenario_spec(
        "input.batch.mixed_validity",
        "task_semantics",
        "input",
        "Make one item invalid in an otherwise valid multi-item request.",
        "Invalid items must not silently contaminate or authorize valid items.",
        ("action.batch",),
        "MixedValidityBatchAugmentation",
        reads=("structured_input",),
        writes=("structured_input",),
    ),
    BuiltinAugmentationSpec(
        ref=AugmentationRef(id="conversation.retry_after_successful_commit", version="1.0.0"),
        surface="conversation_workflow",
        scope="conversation",
        summary="Retry only after the first committed-state checkpoint succeeds.",
        expected_relation="The committed effect must remain at most once.",
        applicability_profile="conditional",
        applicability_rule=(
            "Applies only when committed state is observable before a second user turn."
        ),
        bindings=(
            _binding(
                "conversation_stress",
                ("execution", "evaluation"),
                "ul.augmentations.conversation:run_retry_after_successful_commit_stress_test",
                projection=_projection(
                    reads=("conversation", "state"),
                    writes=("conversation",),
                ),
                execution_owner="stress_cli",
                command="ul stress retry-after-successful-commit",
                requirements=AugmentationRequirements(
                    required_source_features=("two ordered user turns",),
                    environment=True,
                    conversations=True,
                    state_observation=True,
                    customer_evaluator=True,
                ),
            ),
        ),
    ),
)


def _version_tuple(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)
