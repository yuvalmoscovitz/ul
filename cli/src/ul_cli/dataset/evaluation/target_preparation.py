from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError
from ul import InteractionRecord
from ul.dataset_invariants import DatasetInvariantSuite
from ul.http_environment import (
    JsonHttpTargetConfig,
    json_http_environment_calls_per_execution,
    json_http_environment_capabilities,
    json_http_environment_config_sha256,
    json_http_environment_origin,
    load_json_http_environment_config,
    validate_json_http_environment_configuration,
)

from ul_cli.dataset_run_config import DatasetRunConfig, TargetExecutionConfig
from ul_cli.dataset_trial_journal import DatasetRunManifest
from ul_cli.environment import TEST_ENVIRONMENT_CONFIRMATION_MESSAGE
from ul_cli.http_target_resolution import (
    ResolvedHttpTarget,
    resolve_http_target,
    resolve_http_target_config,
)
from ul_cli.local_target_resolution import ResolvedLocalTarget, resolve_local_target

from ..presentation.runtime import print_dataset_plain
from .request import DatasetRequestError, NormalizedDatasetEvaluationRequest


@dataclass(frozen=True)
class PreparedEvaluationTarget:
    local_target: ResolvedLocalTarget | None
    http_target: ResolvedHttpTarget | None
    config: JsonHttpTargetConfig | None
    run_config: DatasetRunConfig


def prepare_evaluation_target(
    request: NormalizedDatasetEvaluationRequest,
    *,
    selected_records: tuple[InteractionRecord, ...],
    selected_operator_ids: tuple[str, ...],
    invariant_suite: DatasetInvariantSuite | None,
) -> PreparedEvaluationTarget:
    requested = request.requested
    if not requested.dry_run and requested.resume is None:
        if requested.environment_config is None and requested.target is None:
            raise DatasetRequestError(
                "execution requires --target or --environment-config", parameter="--target"
            )
        if requested.environment_config is not None and not request.allow_environment_network:
            raise DatasetRequestError(
                "execution requires --allow-environment-network",
                parameter="--allow-environment-network",
            )
        if not request.confirm_test_environment:
            raise DatasetRequestError(
                TEST_ENVIRONMENT_CONFIRMATION_MESSAGE,
                parameter="--confirm-test-environment",
            )

    local_target: ResolvedLocalTarget | None = None
    http_target: ResolvedHttpTarget | None = None
    direct_http_options_used = (
        requested.http_preset is not None
        or requested.request_json_template is not None
        or requested.response_json_pointer is not None
        or requested.agent_model is not None
        or bool(requested.headers_from_env)
    )
    if requested.target is not None:
        direct_http_target = requested.target.casefold().startswith(("https://", "http://"))
        if direct_http_target and not requested.confirm_request_isolation:
            raise DatasetRequestError(
                "direct HTTP targets require --confirm-request-isolation",
                parameter="--confirm-request-isolation",
            )
        if direct_http_target and not requested.confirm_safe_test_target:
            raise DatasetRequestError(
                "direct HTTP targets require --confirm-safe-test-target",
                parameter="--confirm-safe-test-target",
            )
        try:
            http_target = resolve_http_target(
                requested.target,
                allow_insecure_http=request.allow_insecure_http,
                http_preset=requested.http_preset,
                request_json_template=requested.request_json_template,
                response_json_pointer=requested.response_json_pointer,
                agent_model=requested.agent_model,
                header_from_env=list(requested.headers_from_env),
                request_isolation_attested=requested.confirm_request_isolation,
                safe_test_target_attested=requested.confirm_safe_test_target,
            )
        except (OSError, ValidationError, ValueError):
            if direct_http_target or direct_http_options_used:
                raise
            local_target = resolve_local_target(
                requested.target,
                explicit_artifacts=requested.target_artifacts,
            )
        if http_target is not None and requested.target_artifacts:
            raise ValueError("--target-artifact applies only to local targets")

    target_config = (
        http_target.config
        if http_target is not None
        else (
            load_json_http_environment_config(requested.environment_config)
            if requested.environment_config is not None
            else _recorded_http_target_config(request.recorded_manifest)
        )
    )
    _validate_recorded_target_identity(request, local_target, http_target, target_config)
    _validate_initialized_environment(request, target_config)

    request_isolation: str | None = None
    if target_config is not None:
        validate_json_http_environment_configuration(
            target_config,
            test_environment_confirmed=(
                request.confirm_test_environment
                or requested.dry_run
                or requested.resume is not None
            ),
            allow_insecure_http=request.allow_insecure_http,
        )
        capabilities = json_http_environment_capabilities(target_config)
        request_isolation = capabilities.request_isolation
        if (
            invariant_suite is not None
            and invariant_suite.observation_authority == "committed_state_snapshot"
            and not capabilities.supports_state_observation
        ):
            raise ValueError(
                "committed-state invariants require the stateful-lifecycle adapter tier; "
                "isolated-response targets provide response evidence only"
            )

    if (
        http_target is not None
        and not requested.dry_run
        and requested.resume is None
        and not request.allow_environment_network
    ):
        raise ValueError("HTTP target execution requires --allow-environment-network")
    if (
        http_target is not None
        and not requested.dry_run
        and requested.resume is None
        and requested.confirm_target != http_target.confirmation_sha256
    ):
        raise ValueError("HTTP execution requires --confirm-target with the exact displayed digest")
    if (
        local_target is not None
        and invariant_suite is not None
        and invariant_suite.observation_authority == "committed_state_snapshot"
    ):
        raise ValueError(
            "committed-state invariants require the stateful-lifecycle adapter tier; "
            "local targets provide response evidence only"
        )
    _validate_target_concurrency(
        request.concurrency,
        request_isolation=request_isolation,
        local_target=local_target is not None,
    )
    if requested.resume is not None and target_config is None and local_target is None:
        raise ValueError("--resume requires a recorded or explicit environment configuration")

    calls_per_execution = (
        json_http_environment_calls_per_execution(target_config) if target_config is not None else 1
    )
    initial_target_calls = (
        len(selected_records)
        * request.repetitions
        * (1 + len(selected_operator_ids))
        * calls_per_execution
    )
    if requested.resume is None and initial_target_calls > request.max_environment_api_calls:
        raise ValueError(
            f"selection would make up to {initial_target_calls} environment API calls, "
            f"exceeding --max-environment-api-calls {request.max_environment_api_calls}; reduce "
            "--limit, --operator, or --repetitions, or explicitly raise the call budget"
        )
    run_config = DatasetRunConfig(
        evaluation_mode=request.evaluation_mode,
        repetitions=request.repetitions,
        concurrency=request.concurrency,
        target=TargetExecutionConfig(
            trial_timeout_seconds=request.target_timeout_seconds,
            max_environment_api_calls=request.max_environment_api_calls,
            environment_api_calls_per_trial=calls_per_execution,
            planned_environment_api_calls=initial_target_calls,
            allow_network_egress=(request.allow_environment_network or local_target is not None),
            test_environment_confirmed=request.confirm_test_environment,
            allow_insecure_http=request.allow_insecure_http,
        ),
    )
    return PreparedEvaluationTarget(
        local_target=local_target,
        http_target=http_target,
        config=target_config,
        run_config=run_config,
    )


def _validate_recorded_target_identity(
    request: NormalizedDatasetEvaluationRequest,
    local_target: ResolvedLocalTarget | None,
    http_target: ResolvedHttpTarget | None,
    target_config: JsonHttpTargetConfig | None,
) -> None:
    manifest = request.recorded_manifest
    recorded_confirmation = (
        manifest.effective_command.http_target_confirmation if manifest is not None else None
    )
    if recorded_confirmation is not None and http_target is None and target_config is not None:
        current_confirmation = resolve_http_target_config(
            recorded_confirmation.reference,
            target_config,
            allow_insecure_http=request.allow_insecure_http,
        ).confirmation
        if current_confirmation != recorded_confirmation:
            raise ValueError(
                "HTTP target credential identity changed since this run was confirmed; start a "
                "new evaluation and confirm the new target digest"
            )
    if (
        request.requested.resume is not None
        and manifest is not None
        and manifest.run_context.target.kind == "probe_target"
        and manifest.effective_command.http_target_config is None
        and local_target is None
    ):
        raise ValueError("local target resume requires the same explicit --target")


def _validate_initialized_environment(
    request: NormalizedDatasetEvaluationRequest,
    target_config: JsonHttpTargetConfig | None,
) -> None:
    requested = request.requested
    if requested.expected_environment_origin is not None:
        if target_config is None:
            raise ValueError("saved environment origin requires --environment-config")
        if json_http_environment_origin(target_config) != requested.expected_environment_origin:
            raise ValueError(
                "environment origin changed since 'ul init'; reinitialize the project and repeat "
                "the environment safety acknowledgements"
            )
    if requested.expected_environment_config_sha256 is not None:
        if target_config is None:
            raise ValueError("saved environment configuration requires --environment-config")
        if (
            json_http_environment_config_sha256(target_config)
            != requested.expected_environment_config_sha256
        ):
            raise ValueError(
                "environment configuration changed since 'ul init'; reinitialize the project and "
                "repeat the environment safety acknowledgements"
            )


def _recorded_http_target_config(
    recorded_manifest: DatasetRunManifest | None,
) -> JsonHttpTargetConfig | None:
    if recorded_manifest is None:
        return None
    direct_http_config = recorded_manifest.effective_command.http_target_config
    if direct_http_config is not None:
        return direct_http_config
    if recorded_manifest.run_context.target.kind == "environment_http":
        return recorded_manifest.run_context.target.config
    return None


def _validate_target_concurrency(
    concurrency: int,
    *,
    request_isolation: str | None,
    local_target: bool,
) -> None:
    if concurrency == 1:
        return
    if local_target:
        raise DatasetRequestError(
            "--concurrency above 1 requires an isolated-response HTTP target; local targets "
            "remain sequential",
            parameter="--concurrency",
        )
    if request_isolation != "per_request_attested":
        raise DatasetRequestError(
            "--concurrency above 1 requires an isolated-response target and "
            "--confirm-request-isolation; stateful lifecycle targets remain sequential",
            parameter="--concurrency",
        )


def print_local_target_identity(target: ResolvedLocalTarget) -> None:
    confirmation = target.confirmation
    print_dataset_plain("UL active-probe target")
    print_dataset_plain(f"  Kind: {target.kind}")
    print_dataset_plain(f"  Config sha256: {target.config_sha256}")
    print_dataset_plain(f"  Confirmation sha256: {target.confirmation_sha256}")
    print_dataset_plain(f"  Selected executable: {confirmation.selected_executable}")
    print_dataset_plain(
        f"  Executable identity: {confirmation.executable.path} ({confirmation.executable.sha256})"
    )
    for artifact in confirmation.artifacts:
        print_dataset_plain(f"  Artifact: {artifact.path} ({artifact.sha256})")
    for environment in confirmation.environment:
        print_dataset_plain(
            f"  Environment: {environment.name} value sha256 {environment.value_sha256}"
        )
    if confirmation.callable is not None:
        print_dataset_plain(f"  Callable: {confirmation.callable}")
    print_dataset_plain("Use only a dedicated test target that cannot cause real-world effects.")


def print_http_target_identity(target: ResolvedHttpTarget) -> None:
    print_dataset_plain("UL active-probe target")
    print_dataset_plain("  Kind: http")
    print_dataset_plain(f"  Config sha256: {target.config_sha256}")
    print_dataset_plain(f"  Confirmation sha256: {target.confirmation_sha256}")
    print_dataset_plain("Use only a dedicated test target that cannot cause real-world effects.")
