from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class ULModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Actor(ULModel):
    id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    name: str | None = None
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class Artifact(ULModel):
    id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    label: str | None = None
    state: str | None = None
    version: str | None = None
    supersedes_id: str | None = None
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class Resource(ULModel):
    id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    state: str | None = None
    owner_actor_id: str | None = None
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class ActionEffect(StrEnum):
    READ = "read"
    WRITE = "write"
    COMMUNICATE = "communicate"
    DECIDE = "decide"


class ActionStatus(StrEnum):
    PLANNED = "planned"
    PENDING = "pending"
    PARTIALLY_COMPLETED = "partially_completed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ItemValidity(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"


class ActionItem(ULModel):
    id: str = Field(min_length=1)
    target_ids: tuple[str, ...] = ()
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    validity: ItemValidity = ItemValidity.VALID
    status: ActionStatus = ActionStatus.PLANNED


class Action(ULModel):
    id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    effect: ActionEffect
    actor_id: str | None = None
    artifact_ids: tuple[str, ...] = ()
    resource_ids: tuple[str, ...] = ()
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    batch_items: tuple[ActionItem, ...] = ()
    status: ActionStatus = ActionStatus.PLANNED


class PolicyBoundary(ULModel):
    action_id: str = Field(min_length=1)
    parameter: str = Field(min_length=1)
    threshold: int | float
    increment: int | float = Field(gt=0)


class Policy(ULModel):
    id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    description: str = Field(min_length=1)
    state: str = "active"
    boundaries: tuple[PolicyBoundary, ...] = ()
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class EventTiming(StrEnum):
    BEFORE_ACTION = "before_action"
    AFTER_ACTION = "after_action"
    BETWEEN_ACTIONS = "between_actions"
    ON_OBSERVATION = "on_observation"


class EnvironmentEvent(ULModel):
    id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    timing: EventTiming
    action_id: str | None = None
    after_action_id: str | None = None
    before_action_id: str | None = None
    target_ids: tuple[str, ...] = ()
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class ConversationRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ConversationTurn(ULModel):
    id: str = Field(min_length=1)
    role: ConversationRole
    content: str
    actor_id: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class AugmentationApplication(ULModel):
    augmentation_id: str = Field(min_length=1)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    changed_paths: tuple[str, ...]
    changed_events: tuple[str, ...] = ()


class OracleRelation(ULModel):
    kind: str = Field(min_length=1)
    description: str = Field(min_length=1)
    invariants: tuple[str, ...] = ()
    permitted_changes: tuple[str, ...] = ()


class ShrinkMetadata(ULModel):
    removable_paths: tuple[str, ...] = ()
    simplifications: tuple[str, ...] = ()


class ScenarioProvenance(ULModel):
    source: str = Field(min_length=1)
    source_reference: str | None = None
    parent_scenario_id: str | None = None
    lineage: tuple[AugmentationApplication, ...] = ()


class LaterCorrectionHint(ULModel):
    action_id: str = Field(min_length=1)
    parameter: str = Field(min_length=1)
    corrected_value: JsonValue
    message: str = Field(min_length=1)


class AmbiguityHint(ULModel):
    source_artifact_id: str = Field(min_length=1)
    alternative_id: str | None = Field(default=None, min_length=1)
    label: str | None = None
    state: str | None = None
    version: str | None = None
    attributes: dict[str, JsonValue] = Field(default_factory=dict)
    replacement_text: str | None = None


class AugmentationHints(ULModel):
    later_correction: LaterCorrectionHint | None = None
    ambiguity: AmbiguityHint | None = None


class Scenario(ULModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    actors: tuple[Actor, ...] = ()
    artifacts: tuple[Artifact, ...] = ()
    resources: tuple[Resource, ...] = ()
    actions: tuple[Action, ...] = ()
    policies: tuple[Policy, ...] = ()
    environment_events: tuple[EnvironmentEvent, ...] = ()
    conversation: tuple[ConversationTurn, ...] = ()
    provenance: ScenarioProvenance
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        collections = {
            "actors": tuple(actor.id for actor in self.actors),
            "artifacts": tuple(artifact.id for artifact in self.artifacts),
            "resources": tuple(resource.id for resource in self.resources),
            "actions": tuple(action.id for action in self.actions),
            "policies": tuple(policy.id for policy in self.policies),
            "events": tuple(event.id for event in self.environment_events),
            "turns": tuple(turn.id for turn in self.conversation),
        }
        for collection_name, identifiers in collections.items():
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"duplicate identifier in {collection_name}")

        all_identifiers = tuple(
            identifier for identifiers in collections.values() for identifier in identifiers
        )
        if len(all_identifiers) != len(set(all_identifiers)):
            raise ValueError("scenario identifiers must be globally unique")

        actor_ids = set(collections["actors"])
        artifact_ids = set(collections["artifacts"])
        resource_ids = set(collections["resources"])
        action_ids = set(collections["actions"])
        known_target_ids = actor_ids | artifact_ids | resource_ids

        raw_hints = self.metadata.get("augmentation_hints")
        if raw_hints is not None:
            hints = AugmentationHints.model_validate(raw_hints)
            if (
                hints.later_correction is not None
                and hints.later_correction.action_id not in action_ids
            ):
                raise ValueError(f"unknown correction action {hints.later_correction.action_id}")
            if hints.later_correction is not None:
                correction_action = next(
                    action
                    for action in self.actions
                    if action.id == hints.later_correction.action_id
                )
                if hints.later_correction.parameter not in correction_action.parameters:
                    raise ValueError(
                        f"unknown correction parameter {hints.later_correction.parameter}"
                    )
            if (
                hints.ambiguity is not None
                and hints.ambiguity.source_artifact_id not in artifact_ids
            ):
                raise ValueError(f"unknown ambiguity artifact {hints.ambiguity.source_artifact_id}")
            if hints.ambiguity is not None and hints.ambiguity.alternative_id in set(
                all_identifiers
            ):
                raise ValueError(
                    f"ambiguity alternative identifier already exists: "
                    f"{hints.ambiguity.alternative_id}"
                )

        for artifact in self.artifacts:
            if artifact.supersedes_id is not None and artifact.supersedes_id not in artifact_ids:
                raise ValueError(f"unknown superseded artifact {artifact.supersedes_id}")
        for resource in self.resources:
            if resource.owner_actor_id is not None and resource.owner_actor_id not in actor_ids:
                raise ValueError(f"unknown owner actor {resource.owner_actor_id}")
        for action in self.actions:
            if action.actor_id is not None and action.actor_id not in actor_ids:
                raise ValueError(f"unknown action actor {action.actor_id}")
            if not set(action.artifact_ids) <= artifact_ids:
                raise ValueError(f"unknown artifact referenced by action {action.id}")
            if not set(action.resource_ids) <= resource_ids:
                raise ValueError(f"unknown resource referenced by action {action.id}")
            if any(not set(item.target_ids) <= known_target_ids for item in action.batch_items):
                raise ValueError(f"unknown batch item target in action {action.id}")
            item_ids = tuple(item.id for item in action.batch_items)
            if len(item_ids) != len(set(item_ids)):
                raise ValueError(f"duplicate batch item identifier in action {action.id}")
        for policy in self.policies:
            if any(boundary.action_id not in action_ids for boundary in policy.boundaries):
                raise ValueError(f"unknown action referenced by policy {policy.id}")
        for event in self.environment_events:
            referenced_actions = (event.action_id, event.after_action_id, event.before_action_id)
            if any(value is not None and value not in action_ids for value in referenced_actions):
                raise ValueError(f"unknown action referenced by event {event.id}")
            if not set(event.target_ids) <= known_target_ids:
                raise ValueError(f"unknown target referenced by event {event.id}")
        for turn in self.conversation:
            if turn.actor_id is not None and turn.actor_id not in actor_ids:
                raise ValueError(f"unknown conversation actor {turn.actor_id}")
        return self


class ExecutionMode(StrEnum):
    CONTROLLED = "controlled"
    LIVE = "live"


class SafetyEnvelope(ULModel):
    description: str = Field(min_length=1)
    isolated: bool
    allows_network_egress: bool
    allows_business_side_effects: bool


class MaterializedScenario(ULModel):
    scenario_id: str = Field(min_length=1)
    target_input: JsonValue
    environment: JsonValue
    expectations: JsonValue = None
    execution_mode: ExecutionMode = ExecutionMode.CONTROLLED
    safety_envelope: SafetyEnvelope

    @model_validator(mode="after")
    def validate_safety_envelope(self) -> Self:
        if self.execution_mode == ExecutionMode.CONTROLLED and (
            not self.safety_envelope.isolated or self.safety_envelope.allows_business_side_effects
        ):
            raise ValueError(
                "controlled execution requires an isolated, business-side-effect-free envelope"
            )
        if (
            self.safety_envelope.allows_business_side_effects
            and self.execution_mode != ExecutionMode.LIVE
        ):
            raise ValueError("business side effects require live execution mode")
        return self


class ExecutionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    ERROR = "error"


class ToolCall(ULModel):
    name: str = Field(min_length=1)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    result: JsonValue = None
    error: str | None = None


class ExecutionResult(ULModel):
    scenario_id: str = Field(min_length=1)
    status: ExecutionStatus
    tool_calls: tuple[ToolCall, ...] = ()
    final_output: JsonValue = None
    state_before: JsonValue = None
    state_after: JsonValue = None
    error: str | None = None
    cost_usd: float = Field(default=0, ge=0)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class FindingSeverity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OracleFinding(ULModel):
    oracle_id: str = Field(min_length=1)
    passed: bool
    category: str = Field(min_length=1)
    message: str = Field(min_length=1)
    severity: FindingSeverity = FindingSeverity.INFO
    evidence: JsonValue = None


class SemanticCoverageFeatures(ULModel):
    action_kinds: tuple[str, ...] = ()
    action_statuses: tuple[str, ...] = ()
    policy_states: tuple[str, ...] = ()
    environment_event_kinds: tuple[str, ...] = ()
    environment_event_semantics: tuple[str, ...] = ()
    tool_sequence: tuple[str, ...] = ()
    execution_outcome: str | None = None
    oracle_categories: tuple[str, ...] = ()
    semantic_tags: tuple[str, ...] = ()


class CampaignCaseResult(ULModel):
    scenario_id: str = Field(min_length=1)
    source_scenario_id: str | None = None
    scenario: Scenario
    augmentation_ids: tuple[str, ...] = ()
    augmentation_applications: tuple[AugmentationApplication, ...] = ()
    oracle_relations: tuple[OracleRelation, ...] = ()
    execution: ExecutionResult
    findings: tuple[OracleFinding, ...] = ()
    coverage: SemanticCoverageFeatures


class CampaignResult(ULModel):
    campaign_id: str = Field(min_length=1)
    cases: tuple[CampaignCaseResult, ...]

    @property
    def failed_case_count(self) -> int:
        return sum(any(not finding.passed for finding in case.findings) for case in self.cases)
