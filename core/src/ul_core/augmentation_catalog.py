from __future__ import annotations

from typing import Literal, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from ul_core.models import ULModel

AugmentationScope = Literal["input", "conversation", "environment"]
AugmentationMode = Literal[
    "dataset_variation",
    "scenario_materialization",
    "conversation_stress",
    "sandbox_fault",
]
AugmentationStage = Literal["materialization", "execution", "evaluation"]
AugmentationExecutionOwner = Literal["dataset_cli", "augmentation_registry", "stress_cli"]

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
    sandbox: bool = False
    conversations: bool = False
    state_observation: bool = False
    customer_evaluator: bool = False
    sandbox_capabilities: tuple[str, ...] = ()
    human_review: bool = False

    @model_validator(mode="after")
    def validate_requirements(self) -> Self:
        if self.sandbox_capabilities and not self.sandbox:
            raise ValueError("sandbox capabilities require a sandbox")
        if (self.conversations or self.state_observation) and not self.sandbox:
            raise ValueError("sandbox execution requirements require a sandbox")
        if len(self.required_source_features) != len(set(self.required_source_features)):
            raise ValueError("required source features must be unique")
        if len(self.sandbox_capabilities) != len(set(self.sandbox_capabilities)):
            raise ValueError("sandbox capabilities must be unique")
        return self


class AugmentationBinding(_CatalogModel):
    mode: AugmentationMode
    stages: tuple[AugmentationStage, ...] = Field(min_length=1)
    execution_owner: AugmentationExecutionOwner
    command: str | None = Field(default=None, min_length=1, max_length=200)
    requirements: AugmentationRequirements = AugmentationRequirements()

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        expected_owner: dict[AugmentationMode, AugmentationExecutionOwner] = {
            "dataset_variation": "dataset_cli",
            "scenario_materialization": "augmentation_registry",
            "conversation_stress": "stress_cli",
            "sandbox_fault": "stress_cli",
        }
        if self.execution_owner != expected_owner[self.mode]:
            raise ValueError("augmentation mode and execution owner do not match")
        if self.execution_owner == "augmentation_registry" and self.command is not None:
            raise ValueError("augmentation registry bindings do not have a CLI command")
        if self.execution_owner != "augmentation_registry" and self.command is None:
            raise ValueError("CLI-owned augmentation bindings require a command")
        if len(self.stages) != len(set(self.stages)):
            raise ValueError("augmentation binding stages must be unique")
        if self.mode == "sandbox_fault" and (
            not self.requirements.sandbox or not self.requirements.sandbox_capabilities
        ):
            raise ValueError("sandbox fault bindings require a sandbox capability")
        return self

    @property
    def cli_available(self) -> bool:
        return self.command is not None


class BuiltinAugmentationSpec(_CatalogModel):
    ref: AugmentationRef
    scope: AugmentationScope
    summary: str = Field(min_length=1, max_length=500)
    bindings: tuple[AugmentationBinding, ...] = Field(min_length=1)

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
    *,
    execution_owner: AugmentationExecutionOwner = "augmentation_registry",
    command: str | None = None,
    requirements: AugmentationRequirements | None = None,
) -> AugmentationBinding:
    return AugmentationBinding(
        mode=mode,
        stages=stages,
        execution_owner=execution_owner,
        command=command,
        requirements=requirements or AugmentationRequirements(),
    )


def _dataset_spec(
    augmentation_id: str,
    summary: str,
    *,
    human_review: bool = False,
) -> BuiltinAugmentationSpec:
    version = "1.0.0"
    return BuiltinAugmentationSpec(
        ref=AugmentationRef(id=augmentation_id, version=version),
        scope="input",
        summary=summary,
        bindings=(
            _binding(
                "dataset_variation",
                ("materialization", "execution", "evaluation"),
                execution_owner="dataset_cli",
                command=f"ul dataset evaluate --operator {augmentation_id}@{version}",
                requirements=AugmentationRequirements(
                    required_source_features=("production interaction",),
                    semantic_model=True,
                    sandbox=True,
                    state_observation=True,
                    human_review=human_review,
                ),
            ),
        ),
    )


def _scenario_spec(
    augmentation_id: str,
    scope: AugmentationScope,
    summary: str,
    required_source_features: tuple[str, ...],
) -> BuiltinAugmentationSpec:
    return BuiltinAugmentationSpec(
        ref=AugmentationRef(id=augmentation_id, version="1.0.0"),
        scope=scope,
        summary=summary,
        bindings=(
            _binding(
                "scenario_materialization",
                ("materialization",),
                requirements=AugmentationRequirements(
                    required_source_features=required_source_features
                ),
            ),
        ),
    )


_BUILTIN_AUGMENTATION_SPECS = (
    _dataset_spec("input.surface.rephrase", "Rephrase while preserving the requested behavior."),
    _dataset_spec("input.surface.typing_noise", "Add plausible typing noise."),
    _dataset_spec("input.surface.fragmented_syntax", "Use plausible fragmented syntax."),
    _dataset_spec("input.surface.disfluency_repeat", "Repeat a word as a natural disfluency."),
    _dataset_spec("input.style.terse", "Express the same request tersely."),
    _dataset_spec("input.style.verbose", "Express the same request verbosely."),
    _dataset_spec(
        "input.tone.frustrated", "Express the same request with frustration.", human_review=True
    ),
    _dataset_spec(
        "input.intent.self_correction",
        "Correct one request value within the same input.",
        human_review=True,
    ),
    _scenario_spec(
        "conversation.ambiguity",
        "conversation",
        "Introduce another plausible artifact with the same human-facing identity.",
        ("artifact", "conversation.user"),
    ),
    BuiltinAugmentationSpec(
        ref=AugmentationRef(id="conversation.correction_after_first_response", version="1.0.0"),
        scope="conversation",
        summary="Correct the request after the agent has already responded once.",
        bindings=(
            _binding(
                "scenario_materialization",
                ("materialization",),
                requirements=AugmentationRequirements(
                    required_source_features=("conversation.user", "action.parameter")
                ),
            ),
            _binding(
                "conversation_stress",
                ("execution", "evaluation"),
                execution_owner="stress_cli",
                command="ul stress correction",
                requirements=AugmentationRequirements(
                    required_source_features=("two ordered user turns",),
                    sandbox=True,
                    conversations=True,
                    state_observation=True,
                    customer_evaluator=True,
                ),
            ),
        ),
    ),
    _scenario_spec(
        "input.policy.boundary_shift",
        "input",
        "Move an action value below, onto, and above a policy boundary.",
        ("policy.boundary",),
    ),
    _scenario_spec(
        "environment.state.existing_partial_operation",
        "environment",
        "Introduce a prior partial execution of the intended write.",
        ("action.write",),
    ),
    _scenario_spec(
        "environment.state.change_between_read_write",
        "environment",
        "Change relevant state between observation and a consequential write.",
        ("action.read", "action.write"),
    ),
    _scenario_spec(
        "environment.tool.stale_observation",
        "environment",
        "Return a plausible but stale observation for a read action.",
        ("action.read",),
    ),
    _scenario_spec(
        "environment.tool.timeout_before_commit",
        "environment",
        "Time out a consequential action before any effect commits.",
        ("action.write",),
    ),
    BuiltinAugmentationSpec(
        ref=AugmentationRef(id="environment.tool.timeout_after_commit", version="1.0.0"),
        scope="environment",
        summary="Lose acknowledgement after a consequential effect commits.",
        bindings=(
            _binding(
                "scenario_materialization",
                ("materialization",),
                requirements=AugmentationRequirements(required_source_features=("action.write",)),
            ),
            _binding(
                "sandbox_fault",
                ("execution", "evaluation"),
                execution_owner="stress_cli",
                command="ul stress timeout-after-commit",
                requirements=AugmentationRequirements(
                    sandbox=True,
                    conversations=True,
                    state_observation=True,
                    customer_evaluator=True,
                    sandbox_capabilities=("environment.tool.timeout_after_commit@1.0.0",),
                ),
            ),
        ),
    ),
    _scenario_spec(
        "input.batch.mixed_validity",
        "input",
        "Make one item invalid in an otherwise valid multi-item request.",
        ("action.batch",),
    ),
    BuiltinAugmentationSpec(
        ref=AugmentationRef(id="conversation.retry_after_successful_commit", version="1.0.0"),
        scope="conversation",
        summary="Retry only after the first committed-state checkpoint succeeds.",
        bindings=(
            _binding(
                "conversation_stress",
                ("execution", "evaluation"),
                execution_owner="stress_cli",
                command="ul stress retry-after-successful-commit",
                requirements=AugmentationRequirements(
                    required_source_features=("two ordered user turns",),
                    sandbox=True,
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
