from __future__ import annotations

import asyncio
import math

from ul_core import (
    AugmentationApplication,
    AugmentationRegistry,
    CampaignCaseResult,
    CampaignResult,
    CoverageArchive,
    DeterministicAugmentationSelector,
    ExecutionResult,
    ExecutionStatus,
    FindingSeverity,
    OracleEvaluator,
    OracleFinding,
    OracleRelation,
    Scenario,
    ScenarioMaterializer,
    TargetExecutor,
    builtin_augmentation_registry,
    extract_semantic_coverage,
)


class CampaignRunner:
    def __init__(
        self,
        materializer: ScenarioMaterializer,
        executor: TargetExecutor,
        oracle: OracleEvaluator,
        *,
        registry: AugmentationRegistry | None = None,
        archive: CoverageArchive | None = None,
        allow_network_egress: bool = False,
        allow_business_side_effects: bool = False,
        case_timeout_seconds: float = 60,
    ) -> None:
        if not math.isfinite(case_timeout_seconds) or case_timeout_seconds <= 0:
            raise ValueError("case_timeout_seconds must be a positive finite number")
        self._materializer = materializer
        self._executor = executor
        self._oracle = oracle
        self._allow_network_egress = allow_network_egress
        self._allow_business_side_effects = allow_business_side_effects
        self._case_timeout_seconds = case_timeout_seconds
        self.registry = registry if registry is not None else builtin_augmentation_registry()
        self.archive = archive if archive is not None else CoverageArchive()
        self._selector = DeterministicAugmentationSelector(self.registry, self.archive)

    async def run(
        self,
        campaign_id: str,
        seed_scenario: Scenario,
        *,
        augmentation_limit: int | None = None,
        max_cases: int = 100,
    ) -> CampaignResult:
        if not campaign_id:
            raise ValueError("campaign_id must not be empty")
        if max_cases < 1:
            raise ValueError("max_cases must allow at least the baseline case")

        cases = [await self._run_case(seed_scenario)]
        selected = self._selector.select(seed_scenario, limit=augmentation_limit)
        for augmentation in selected:
            for augmented in augmentation.apply(seed_scenario):
                if len(cases) >= max_cases:
                    return CampaignResult(campaign_id=campaign_id, cases=tuple(cases))
                validation = augmentation.validate(seed_scenario, augmented.scenario)
                if not validation.valid:
                    issues = "; ".join(validation.issues)
                    raise ValueError(f"invalid candidate from {augmentation.metadata.id}: {issues}")
                cases.append(
                    await self._run_case(
                        augmented.scenario,
                        source_scenario_id=seed_scenario.id,
                        augmentation_applications=(augmented.scenario.provenance.lineage[-1],),
                        oracle_relations=(augmented.oracle_relation,),
                    )
                )
        return CampaignResult(campaign_id=campaign_id, cases=tuple(cases))

    async def _run_case(
        self,
        scenario: Scenario,
        *,
        source_scenario_id: str | None = None,
        augmentation_applications: tuple[AugmentationApplication, ...] = (),
        oracle_relations: tuple[OracleRelation, ...] = (),
    ) -> CampaignCaseResult:
        augmentation_ids = tuple(
            application.augmentation_id for application in augmentation_applications
        )
        try:
            materialized = self._materializer.materialize(scenario)
        except Exception as error:
            return self._framework_failure_case(
                scenario,
                source_scenario_id,
                augmentation_applications,
                oracle_relations,
                category="materialization_error",
                message="Scenario materialization failed.",
                error=error,
            )

        if materialized.safety_envelope.allows_network_egress and not self._allow_network_egress:
            return self._framework_failure_case(
                scenario,
                source_scenario_id,
                augmentation_applications,
                oracle_relations,
                category="network_egress_blocked",
                message="Network egress requires explicit opt-in.",
            )

        if (
            materialized.safety_envelope.allows_business_side_effects
            and not self._allow_business_side_effects
        ):
            return self._framework_failure_case(
                scenario,
                source_scenario_id,
                augmentation_applications,
                oracle_relations,
                category="business_side_effects_blocked",
                message="Consequential business side effects require explicit opt-in.",
            )

        try:
            async with asyncio.timeout(self._case_timeout_seconds):
                try:
                    execution = await self._executor.execute(materialized)
                except Exception as error:
                    raise _CaseComponentError(type(error).__name__) from error
                if execution.scenario_id != scenario.id:
                    raise ValueError("executor returned a result for a different scenario")
                try:
                    findings = await self._oracle.evaluate(scenario, materialized, execution)
                except Exception as error:
                    raise _CaseComponentError(type(error).__name__) from error
                if execution.status != ExecutionStatus.SUCCEEDED:
                    findings = (
                        *findings,
                        _framework_finding(
                            "target_execution_failed",
                            f"Target execution ended with status {execution.status.value}.",
                        ),
                    )
        except TimeoutError:
            execution = ExecutionResult(
                scenario_id=scenario.id,
                status=ExecutionStatus.TIMED_OUT,
                error="case deadline exceeded",
            )
            findings = (
                _framework_finding(
                    "case_timeout",
                    "Target execution or oracle evaluation exceeded the case deadline.",
                ),
            )
        except Exception as error:
            execution = ExecutionResult(
                scenario_id=scenario.id,
                status=ExecutionStatus.ERROR,
                error=(
                    error.component_error_type
                    if isinstance(error, _CaseComponentError)
                    else type(error).__name__
                ),
            )
            findings = (
                _framework_finding(
                    "case_execution_error",
                    "Target execution or oracle evaluation failed.",
                ),
            )

        coverage = extract_semantic_coverage(scenario, execution, findings)
        self.archive.record(coverage, augmentation_ids)
        return CampaignCaseResult(
            scenario_id=scenario.id,
            source_scenario_id=source_scenario_id,
            scenario=scenario,
            augmentation_ids=augmentation_ids,
            augmentation_applications=augmentation_applications,
            oracle_relations=oracle_relations,
            execution=execution,
            findings=findings,
            coverage=coverage,
        )

    def _framework_failure_case(
        self,
        scenario: Scenario,
        source_scenario_id: str | None,
        augmentation_applications: tuple[AugmentationApplication, ...],
        oracle_relations: tuple[OracleRelation, ...],
        *,
        category: str,
        message: str,
        error: Exception | None = None,
    ) -> CampaignCaseResult:
        augmentation_ids = tuple(
            application.augmentation_id for application in augmentation_applications
        )
        execution = ExecutionResult(
            scenario_id=scenario.id,
            status=ExecutionStatus.ERROR,
            error=None if error is None else type(error).__name__,
        )
        findings = (_framework_finding(category, message),)
        coverage = extract_semantic_coverage(scenario, execution, findings)
        self.archive.record(coverage, augmentation_ids)
        return CampaignCaseResult(
            scenario_id=scenario.id,
            source_scenario_id=source_scenario_id,
            scenario=scenario,
            augmentation_ids=augmentation_ids,
            augmentation_applications=augmentation_applications,
            oracle_relations=oracle_relations,
            execution=execution,
            findings=findings,
            coverage=coverage,
        )


def _framework_finding(category: str, message: str) -> OracleFinding:
    return OracleFinding(
        oracle_id="ul.framework",
        passed=False,
        category=category,
        message=message,
        severity=FindingSeverity.HIGH,
    )


class _CaseComponentError(Exception):
    def __init__(self, component_error_type: str) -> None:
        super().__init__(component_error_type)
        self.component_error_type = component_error_type
