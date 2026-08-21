from __future__ import annotations

import asyncio
import json
import os
import shlex
import stat
from pathlib import Path
from typing import Annotated, TextIO

import typer
from pydantic import ValidationError
from ul.dataset_invariants import (
    DatasetInvariantRule,
    ObservationAuthority,
    load_dataset_invariant_suite,
)
from ul.dataset_regression import dataset_regression_target_config_sha256
from ul.event_stress import (
    CorrectionAfterFirstResponseCase,
    CorrectionStressResult,
    MultiTurnRegressionCase,
    RetryAfterSuccessfulCommitCase,
    RetryAfterSuccessfulCommitStressResult,
    create_multi_turn_regression_case,
    load_correction_after_first_response_case,
    load_multi_turn_regression_case,
    load_retry_after_successful_commit_case,
    plan_correction_stress_test,
    plan_retry_after_successful_commit_stress_test,
    replay_multi_turn_regression,
    run_correction_stress_test,
    run_retry_after_successful_commit_stress_test,
)
from ul.http_environment import (
    JsonHttpEnvironmentConnection,
    load_json_http_environment_config,
    require_stateful_json_http_environment,
    validate_json_http_environment_configuration,
)
from ul.timeout_after_commit import (
    TimeoutAfterCommitCase,
    TimeoutAfterCommitStressResult,
    load_timeout_after_commit_case,
    plan_timeout_after_commit_stress_test,
    run_timeout_after_commit_stress_test,
)
from ul.trace_replay import (
    TraceReplayBundle,
    TraceReplayCampaignPlan,
    TraceReplayCampaignResult,
    TraceReplayCase,
    TraceReplayDifferenceGrouping,
    TraceReplayResult,
    TraceStressPlan,
    derive_trace_stress_plan,
    group_trace_replay_differences,
    load_trace_replay_bundle,
    load_trace_replay_results,
    plan_trace_replay,
    plan_trace_replay_campaign,
    run_trace_replay,
    run_trace_replay_campaign,
    select_trace_replay_case,
)

from ul_cli.environment import TEST_ENVIRONMENT_CONFIRMATION_MESSAGE

app = typer.Typer(help="Stress stateful agents with ordered conversation events.")


@app.command("timeout-after-commit")
def run_timeout_after_commit(
    case_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    target_config_path: Annotated[
        Path, typer.Option("--environment-config", exists=True, dir_okay=False, readable=True)
    ],
    invariants_path: Annotated[
        Path, typer.Option("--invariants", exists=True, dir_okay=False, readable=True)
    ],
    output: Annotated[Path | None, typer.Option(help="New private JSON evidence file.")] = None,
    repetitions: Annotated[int, typer.Option(min=1)] = 3,
    max_target_calls: Annotated[int, typer.Option("--max-environment-api-calls", min=1)] = 100,
    allow_target_network: Annotated[bool, typer.Option("--allow-environment-network")] = False,
    confirm_test_environment: Annotated[
        bool,
        typer.Option(help=("Confirm the environment is intended for testing and can be reset.")),
    ] = False,
    allow_insecure_http: Annotated[
        bool, typer.Option(help="Allow an HTTP environment API. Intended for local environments.")
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(help="Validate and show the complete plan without environment API calls."),
    ] = False,
) -> None:
    """Inject one versioned lost acknowledgement after a committed tool write."""
    try:
        case = load_timeout_after_commit_case(case_path)
        target_config = require_stateful_json_http_environment(
            load_json_http_environment_config(target_config_path)
        )
        invariant_suite = load_dataset_invariant_suite(invariants_path)
        if invariant_suite.observation_authority != "committed_state_snapshot":
            raise ValueError("timeout-after-commit testing requires committed-state invariants")
        validate_json_http_environment_configuration(
            target_config,
            test_environment_confirmed=confirm_test_environment or dry_run,
            allow_insecure_http=allow_insecure_http,
            resolve_header_values=not dry_run,
        )
        plan = plan_timeout_after_commit_stress_test(
            case,
            target_config,
            repetitions=repetitions,
            max_environment_api_calls=max_target_calls,
        )
    except (ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from None

    if dry_run:
        typer.echo(f"Operator: {plan.operator_id}@{plan.operator_version}")
        typer.echo(f"Repetitions: {plan.repetitions}")
        typer.echo(f"Target calls per repetition: {plan.target_calls_per_repetition}")
        typer.echo(f"Potential environment API calls: {plan.required_target_calls}")
        typer.echo("External calls: none")
        return
    if not allow_target_network:
        raise typer.BadParameter(
            "execution requires --allow-environment-network",
            param_hint="--allow-environment-network",
        )
    if not confirm_test_environment:
        raise typer.BadParameter(
            TEST_ENVIRONMENT_CONFIRMATION_MESSAGE,
            param_hint="--confirm-test-environment",
        )
    if output is None:
        raise typer.BadParameter("execution requires --output", param_hint="--output")
    if output.exists():
        raise typer.BadParameter("output already exists; UL will not overwrite it")
    try:
        output_stream = _create_private_output(output)
    except OSError:
        raise typer.BadParameter("output could not be created", param_hint="--output") from None
    try:
        with output_stream:
            target = JsonHttpEnvironmentConnection.from_config(
                target_config,
                test_environment_confirmed=True,
                allow_insecure_http=allow_insecure_http,
                max_environment_api_calls=max_target_calls,
            )
            result = asyncio.run(
                _run_timeout_after_commit_and_close(
                    case,
                    target,
                    invariant_rules=invariant_suite.rules,
                    observation_authority=invariant_suite.observation_authority,
                    repetitions=repetitions,
                    max_target_calls=max_target_calls,
                )
            )
            json.dump(result.model_dump(mode="json"), output_stream, ensure_ascii=False, indent=2)
            output_stream.write("\n")
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    _print_timeout_after_commit_result(result, output)


@app.command("trace")
def replay_production_trace(
    bundle_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    target_config_path: Annotated[
        Path, typer.Option("--environment-config", exists=True, dir_okay=False, readable=True)
    ],
    case_id: Annotated[
        str | None,
        typer.Option(help="Replay case ID; optional only when the bundle contains one case."),
    ] = None,
    output: Annotated[
        Path | None, typer.Option(help="New private JSON replay evidence file.")
    ] = None,
    repetitions: Annotated[int, typer.Option(min=1)] = 3,
    max_target_calls: Annotated[int, typer.Option("--max-environment-api-calls", min=1)] = 100,
    allow_target_network: Annotated[bool, typer.Option("--allow-environment-network")] = False,
    confirm_test_environment: Annotated[
        bool,
        typer.Option(help=("Confirm the environment is intended for testing and can be reset.")),
    ] = False,
    allow_insecure_http: Annotated[
        bool, typer.Option(help="Allow an HTTP environment API. Intended for local environments.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option(help="Validate and show the replay plan without environment API calls.")
    ] = False,
) -> None:
    """Replay a production trace conversation prefix in a clean environment."""
    try:
        bundle = load_trace_replay_bundle(bundle_path)
        case = select_trace_replay_case(bundle, case_id)
        target_config = load_json_http_environment_config(target_config_path)
        validate_json_http_environment_configuration(
            target_config,
            test_environment_confirmed=confirm_test_environment or dry_run,
            allow_insecure_http=allow_insecure_http,
            resolve_header_values=not dry_run,
        )
        plan = plan_trace_replay(
            case,
            target_config,
            repetitions=repetitions,
            max_target_calls=max_target_calls,
        )
    except (ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from None

    if dry_run:
        typer.echo(f"Replay case: {plan.case_id}")
        typer.echo(f"Ordered user turns: {plan.replay_turn_count}")
        typer.echo(f"Repetitions: {plan.repetitions}")
        typer.echo(f"Target calls per repetition: {plan.target_calls_per_repetition}")
        typer.echo(f"Potential environment API calls: {plan.required_target_calls}")
        typer.echo("Recorded content: not printed")
        typer.echo("External calls: none")
        return
    if not allow_target_network:
        raise typer.BadParameter(
            "execution requires --allow-environment-network",
            param_hint="--allow-environment-network",
        )
    if not confirm_test_environment:
        raise typer.BadParameter(
            TEST_ENVIRONMENT_CONFIRMATION_MESSAGE,
            param_hint="--confirm-test-environment",
        )
    if output is None:
        raise typer.BadParameter("execution requires --output", param_hint="--output")
    if output.exists():
        raise typer.BadParameter("output already exists; UL will not overwrite it")
    try:
        output_stream = _create_private_output(output)
    except OSError:
        raise typer.BadParameter("output could not be created", param_hint="--output") from None
    try:
        with output_stream:
            target = JsonHttpEnvironmentConnection.from_config(
                target_config,
                test_environment_confirmed=True,
                allow_insecure_http=allow_insecure_http,
                max_environment_api_calls=max_target_calls,
            )
            result = asyncio.run(
                _run_trace_replay_and_close(
                    case,
                    target,
                    repetitions=repetitions,
                    max_target_calls=max_target_calls,
                )
            )
            json.dump(result.model_dump(mode="json"), output_stream, ensure_ascii=False, indent=2)
            output_stream.write("\n")
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    _print_trace_replay_result(result, output)


@app.command("trace-plan")
def plan_production_trace_stress(
    bundle_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit stable machine-readable JSON.")
    ] = False,
) -> None:
    """Turn imported trace evidence into a prioritized stress plan."""
    try:
        plan = derive_trace_stress_plan(load_trace_replay_bundle(bundle_path))
    except (ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from None
    if json_output:
        typer.echo(json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return
    _print_trace_stress_plan(plan, bundle_path)


@app.command("trace-replay-campaign")
def run_production_trace_campaign(
    bundle_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    target_config_path: Annotated[
        Path,
        typer.Option(
            "-e",
            "--environment-config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSON connection config for the resettable test environment.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="New private JSON campaign evidence file."),
    ] = None,
    limit: Annotated[
        int, typer.Option(min=1, max=100, help="Run at most this many prioritized cases.")
    ] = 10,
    repetitions: Annotated[
        int, typer.Option(min=1, help="Fresh replay attempts for each selected case.")
    ] = 3,
    max_target_calls: Annotated[
        int,
        typer.Option(
            "-b",
            "--max-environment-api-calls",
            min=1,
            help="Hard cumulative environment API call budget for the campaign.",
        ),
    ] = 100,
    allow_target_network: Annotated[
        bool,
        typer.Option(
            "-n",
            "--allow-environment-network",
            help="Authorize calls to the configured test environment.",
        ),
    ] = False,
    confirm_test_environment: Annotated[
        bool,
        typer.Option(help="Confirm the environment is intended for testing and can be reset."),
    ] = False,
    allow_insecure_http: Annotated[
        bool, typer.Option(help="Allow an HTTP environment API. Intended for local environments.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option(help="Validate and show the campaign without environment API calls.")
    ] = False,
) -> None:
    """Replay the highest-priority imported trace cases as one campaign."""
    try:
        bundle = load_trace_replay_bundle(bundle_path)
        target_config = load_json_http_environment_config(target_config_path)
        validate_json_http_environment_configuration(
            target_config,
            test_environment_confirmed=confirm_test_environment or dry_run,
            allow_insecure_http=allow_insecure_http,
            resolve_header_values=not dry_run,
        )
        plan = plan_trace_replay_campaign(
            bundle,
            target_config,
            limit=limit,
            repetitions=repetitions,
            max_target_calls=max_target_calls,
        )
    except (ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from None

    if dry_run:
        _print_trace_replay_campaign_plan(
            plan,
            bundle_path=bundle_path,
            target_config_path=target_config_path,
            allow_insecure_http=allow_insecure_http,
        )
        return
    if not allow_target_network:
        raise typer.BadParameter(
            "execution requires --allow-environment-network",
            param_hint="--allow-environment-network",
        )
    if not confirm_test_environment:
        raise typer.BadParameter(
            TEST_ENVIRONMENT_CONFIRMATION_MESSAGE,
            param_hint="--confirm-test-environment",
        )
    if output is None:
        raise typer.BadParameter("execution requires --output", param_hint="--output")
    if output.exists():
        raise typer.BadParameter("output already exists; UL will not overwrite it")
    try:
        output_stream = _create_private_output(output)
    except OSError:
        raise typer.BadParameter("output could not be created", param_hint="--output") from None
    try:
        with output_stream:
            target = JsonHttpEnvironmentConnection.from_config(
                target_config,
                test_environment_confirmed=True,
                allow_insecure_http=allow_insecure_http,
                max_environment_api_calls=max_target_calls,
            )
            result = asyncio.run(_run_trace_replay_campaign_and_close(bundle, target, plan=plan))
            json.dump(result.model_dump(mode="json"), output_stream, ensure_ascii=False, indent=2)
            output_stream.write("\n")
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    _print_trace_replay_campaign_result(result, output)


@app.command("trace-group")
def group_production_trace_differences(
    result_paths: Annotated[
        list[Path],
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit stable machine-readable JSON.")
    ] = False,
) -> None:
    """Group trace replay differences by stable evidence signatures."""
    try:
        grouping = group_trace_replay_differences(load_trace_replay_results(result_paths))
    except (ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from None
    if json_output:
        typer.echo(json.dumps(grouping.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return
    _print_trace_replay_difference_grouping(grouping)


@app.command("save")
def save_multi_turn_regression(
    case_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    target_config_path: Annotated[
        Path, typer.Option("--environment-config", exists=True, dir_okay=False, readable=True)
    ],
    invariants_path: Annotated[
        Path, typer.Option("--invariants", exists=True, dir_okay=False, readable=True)
    ],
    output: Annotated[Path, typer.Option(help="New private regression case file.")],
    repetitions: Annotated[int, typer.Option(min=1)] = 3,
    confirm_versioned_input: Annotated[
        bool,
        typer.Option(
            help="Confirm the exact conversation and invariant literals are safe to version."
        ),
    ] = False,
) -> None:
    """Save an exact correction conversation as a content-addressed regression."""
    if not confirm_versioned_input:
        raise typer.BadParameter(
            "saving requires --confirm-versioned-input because conversation turns may be sensitive"
        )
    if output.exists():
        raise typer.BadParameter("output already exists; UL will not overwrite it")
    try:
        case = load_correction_after_first_response_case(case_path)
        target_config = require_stateful_json_http_environment(
            load_json_http_environment_config(target_config_path)
        )
        invariant_suite = load_dataset_invariant_suite(invariants_path)
        regression = create_multi_turn_regression_case(
            stress_case=case,
            target_config=target_config,
            source_suite_sha256=invariant_suite.sha256,
            observation_authority=invariant_suite.observation_authority,
            invariant_rules=invariant_suite.rules,
            repetitions=repetitions,
        )
    except (ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from None
    with _create_private_output(output) as output_stream:
        json.dump(regression.model_dump(mode="json"), output_stream, ensure_ascii=False, indent=2)
        output_stream.write("\n")
        output_stream.flush()
        os.fsync(output_stream.fileno())
    typer.echo(f"Saved multi-turn regression {regression.case_id}: {output}")


@app.command("correction")
def run_correction_after_first_response(
    case_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    target_config_path: Annotated[
        Path, typer.Option("--environment-config", exists=True, dir_okay=False, readable=True)
    ],
    invariants_path: Annotated[
        Path, typer.Option("--invariants", exists=True, dir_okay=False, readable=True)
    ],
    output: Annotated[Path | None, typer.Option(help="New private JSON evidence file.")] = None,
    repetitions: Annotated[int, typer.Option(min=1)] = 3,
    max_target_calls: Annotated[int, typer.Option("--max-environment-api-calls", min=1)] = 100,
    allow_target_network: Annotated[bool, typer.Option("--allow-environment-network")] = False,
    confirm_test_environment: Annotated[
        bool,
        typer.Option(help=("Confirm the environment is intended for testing and can be reset.")),
    ] = False,
    allow_insecure_http: Annotated[
        bool, typer.Option(help="Allow an HTTP environment API. Intended for local environments.")
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(help="Validate and show the complete plan without environment API calls."),
    ] = False,
) -> None:
    """Run the fixed correction-after-first-response event operator."""
    try:
        case = load_correction_after_first_response_case(case_path)
        target_config = require_stateful_json_http_environment(
            load_json_http_environment_config(target_config_path)
        )
        invariant_suite = load_dataset_invariant_suite(invariants_path)
        validate_json_http_environment_configuration(
            target_config,
            test_environment_confirmed=confirm_test_environment or dry_run,
            allow_insecure_http=allow_insecure_http,
        )
        plan = plan_correction_stress_test(
            case,
            target_config,
            repetitions=repetitions,
            max_environment_api_calls=max_target_calls,
        )
    except (ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from None

    if dry_run:
        typer.echo(f"Operator: {plan.operator_id}@{plan.operator_version}")
        typer.echo("Ordered turns: initial request -> correction after first response")
        typer.echo(f"Repetitions: {plan.repetitions}")
        typer.echo(f"Target calls per paired repetition: {plan.target_calls_per_pair}")
        typer.echo(f"Potential environment API calls: {plan.required_target_calls}")
        typer.echo("External calls: none")
        return
    if not allow_target_network:
        raise typer.BadParameter(
            "execution requires --allow-environment-network",
            param_hint="--allow-environment-network",
        )
    if not confirm_test_environment:
        raise typer.BadParameter(
            TEST_ENVIRONMENT_CONFIRMATION_MESSAGE,
            param_hint="--confirm-test-environment",
        )
    if output is None:
        raise typer.BadParameter("execution requires --output", param_hint="--output")
    if output.exists():
        raise typer.BadParameter("output already exists; UL will not overwrite it")
    target = JsonHttpEnvironmentConnection.from_config(
        target_config,
        test_environment_confirmed=True,
        allow_insecure_http=allow_insecure_http,
        max_environment_api_calls=max_target_calls,
    )
    result = asyncio.run(
        _run_and_close(
            case,
            target,
            invariant_rules=invariant_suite.rules,
            observation_authority=invariant_suite.observation_authority,
            repetitions=repetitions,
            max_target_calls=max_target_calls,
        )
    )
    with _create_private_output(output) as output_stream:
        json.dump(result.model_dump(mode="json"), output_stream, ensure_ascii=False, indent=2)
        output_stream.write("\n")
        output_stream.flush()
        os.fsync(output_stream.fileno())
    _print_result(result, output)


@app.command("retry-after-successful-commit")
def run_retry_after_successful_commit(
    case_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    target_config_path: Annotated[
        Path, typer.Option("--environment-config", exists=True, dir_okay=False, readable=True)
    ],
    invariants_path: Annotated[
        Path, typer.Option("--invariants", exists=True, dir_okay=False, readable=True)
    ],
    output: Annotated[Path | None, typer.Option(help="New private JSON evidence file.")] = None,
    repetitions: Annotated[int, typer.Option(min=1)] = 3,
    max_target_calls: Annotated[int, typer.Option("--max-environment-api-calls", min=1)] = 100,
    allow_target_network: Annotated[bool, typer.Option("--allow-environment-network")] = False,
    confirm_test_environment: Annotated[
        bool,
        typer.Option(help=("Confirm the environment is intended for testing and can be reset.")),
    ] = False,
    allow_insecure_http: Annotated[
        bool, typer.Option(help="Allow an HTTP environment API. Intended for local environments.")
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(help="Validate and show the complete plan without environment API calls."),
    ] = False,
) -> None:
    """Retry an operation only after its first committed-state checkpoint succeeds."""
    try:
        case = load_retry_after_successful_commit_case(case_path)
        target_config = require_stateful_json_http_environment(
            load_json_http_environment_config(target_config_path)
        )
        invariant_suite = load_dataset_invariant_suite(invariants_path)
        if invariant_suite.observation_authority != "committed_state_snapshot":
            raise ValueError("retry stress testing requires committed-state invariant observation")
        validate_json_http_environment_configuration(
            target_config,
            test_environment_confirmed=confirm_test_environment or dry_run,
            allow_insecure_http=allow_insecure_http,
        )
        plan = plan_retry_after_successful_commit_stress_test(
            case,
            target_config,
            repetitions=repetitions,
            max_environment_api_calls=max_target_calls,
        )
    except (ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from None

    if dry_run:
        typer.echo(f"Operator: {plan.operator_id}@{plan.operator_version}")
        typer.echo("Ordered turns: initial committed operation -> explicit retry")
        typer.echo(f"Repetitions: {plan.repetitions}")
        typer.echo(f"Target calls per paired repetition: {plan.target_calls_per_pair}")
        typer.echo(f"Potential environment API calls: {plan.required_target_calls}")
        typer.echo("External calls: none")
        return
    if not allow_target_network:
        raise typer.BadParameter(
            "execution requires --allow-environment-network",
            param_hint="--allow-environment-network",
        )
    if not confirm_test_environment:
        raise typer.BadParameter(
            TEST_ENVIRONMENT_CONFIRMATION_MESSAGE,
            param_hint="--confirm-test-environment",
        )
    if output is None:
        raise typer.BadParameter("execution requires --output", param_hint="--output")
    if output.exists():
        raise typer.BadParameter("output already exists; UL will not overwrite it")
    target = JsonHttpEnvironmentConnection.from_config(
        target_config,
        test_environment_confirmed=True,
        allow_insecure_http=allow_insecure_http,
        max_environment_api_calls=max_target_calls,
    )
    result = asyncio.run(
        _run_retry_and_close(
            case,
            target,
            invariant_rules=invariant_suite.rules,
            observation_authority=invariant_suite.observation_authority,
            repetitions=repetitions,
            max_target_calls=max_target_calls,
        )
    )
    with _create_private_output(output) as output_stream:
        json.dump(result.model_dump(mode="json"), output_stream, ensure_ascii=False, indent=2)
        output_stream.write("\n")
        output_stream.flush()
        os.fsync(output_stream.fileno())
    _print_retry_result(result, output)


@app.command("replay")
def replay_saved_multi_turn_case(
    case_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    target_config_path: Annotated[
        Path, typer.Option("--environment-config", exists=True, dir_okay=False, readable=True)
    ],
    output: Annotated[Path, typer.Option(help="New private JSON replay evidence file.")],
    max_target_calls: Annotated[int, typer.Option("--max-environment-api-calls", min=1)] = 100,
    allow_target_network: Annotated[bool, typer.Option("--allow-environment-network")] = False,
    confirm_test_environment: Annotated[
        bool,
        typer.Option(help=("Confirm the environment is intended for testing and can be reset.")),
    ] = False,
    allow_insecure_http: Annotated[
        bool, typer.Option(help="Allow an HTTP environment API. Intended for local environments.")
    ] = False,
) -> None:
    """Replay a content-addressed multi-turn correction regression."""
    try:
        case = load_multi_turn_regression_case(case_path)
        target_config = require_stateful_json_http_environment(
            load_json_http_environment_config(target_config_path)
        )
        if dataset_regression_target_config_sha256(target_config) != case.target.config_sha256:
            raise ValueError("trusted environment config digest does not match the regression case")
        plan_correction_stress_test(
            case.stress_case,
            target_config,
            repetitions=case.repetitions,
            max_environment_api_calls=max_target_calls,
        )
    except (ValidationError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from None
    if not allow_target_network or not confirm_test_environment:
        raise typer.BadParameter(
            "replay requires --allow-environment-network and --confirm-test-environment"
        )
    if output.exists():
        raise typer.BadParameter("output already exists; UL will not overwrite it")
    target = JsonHttpEnvironmentConnection.from_config(
        target_config,
        test_environment_confirmed=True,
        allow_insecure_http=allow_insecure_http,
        max_environment_api_calls=max_target_calls,
    )
    result = asyncio.run(
        _replay_and_close(
            case,
            target,
            max_target_calls=max_target_calls,
        )
    )
    with _create_private_output(output) as output_stream:
        json.dump(result.model_dump(mode="json"), output_stream, ensure_ascii=False, indent=2)
        output_stream.write("\n")
    _print_result(result, output)


async def _run_and_close(
    case: CorrectionAfterFirstResponseCase,
    target: JsonHttpEnvironmentConnection,
    *,
    invariant_rules: tuple[DatasetInvariantRule, ...],
    observation_authority: ObservationAuthority,
    repetitions: int,
    max_target_calls: int,
) -> CorrectionStressResult:
    try:
        return await run_correction_stress_test(
            case,
            target,
            invariant_rules=invariant_rules,
            observation_authority=observation_authority,
            repetitions=repetitions,
            max_environment_api_calls=max_target_calls,
            allow_network_egress=True,
        )
    finally:
        await target.aclose()


async def _run_retry_and_close(
    case: RetryAfterSuccessfulCommitCase,
    target: JsonHttpEnvironmentConnection,
    *,
    invariant_rules: tuple[DatasetInvariantRule, ...],
    observation_authority: ObservationAuthority,
    repetitions: int,
    max_target_calls: int,
) -> RetryAfterSuccessfulCommitStressResult:
    try:
        return await run_retry_after_successful_commit_stress_test(
            case,
            target,
            invariant_rules=invariant_rules,
            observation_authority=observation_authority,
            repetitions=repetitions,
            max_environment_api_calls=max_target_calls,
            allow_network_egress=True,
        )
    finally:
        await target.aclose()


async def _run_timeout_after_commit_and_close(
    case: TimeoutAfterCommitCase,
    target: JsonHttpEnvironmentConnection,
    *,
    invariant_rules: tuple[DatasetInvariantRule, ...],
    observation_authority: ObservationAuthority,
    repetitions: int,
    max_target_calls: int,
) -> TimeoutAfterCommitStressResult:
    try:
        return await run_timeout_after_commit_stress_test(
            case,
            target,
            invariant_rules=invariant_rules,
            observation_authority=observation_authority,
            repetitions=repetitions,
            max_environment_api_calls=max_target_calls,
            allow_network_egress=True,
        )
    finally:
        await target.aclose()


async def _run_trace_replay_and_close(
    case: TraceReplayCase,
    target: JsonHttpEnvironmentConnection,
    *,
    repetitions: int,
    max_target_calls: int,
) -> TraceReplayResult:
    try:
        return await run_trace_replay(
            case,
            target,
            repetitions=repetitions,
            max_target_calls=max_target_calls,
            allow_network_egress=True,
        )
    finally:
        await target.aclose()


async def _run_trace_replay_campaign_and_close(
    bundle: TraceReplayBundle,
    target: JsonHttpEnvironmentConnection,
    *,
    plan: TraceReplayCampaignPlan,
) -> TraceReplayCampaignResult:
    try:
        return await run_trace_replay_campaign(
            bundle,
            target,
            plan=plan,
            allow_network_egress=True,
        )
    finally:
        await target.aclose()


async def _replay_and_close(
    case: MultiTurnRegressionCase,
    target: JsonHttpEnvironmentConnection,
    *,
    max_target_calls: int,
) -> CorrectionStressResult:
    try:
        return await replay_multi_turn_regression(
            case, target, max_environment_api_calls=max_target_calls, allow_network_egress=True
        )
    finally:
        await target.aclose()


def _print_result(result: CorrectionStressResult, output: Path) -> None:
    typer.echo(f"Correction stress result: {result.status}")
    typer.echo("Exact baseline/variation responses and state snapshots are in private evidence.")
    typer.echo(f"First response divergence: {result.first_response_divergence_turn_id or 'none'}")
    typer.echo(
        "First committed-state divergence: "
        f"{result.first_committed_state_divergence_turn_id or 'none'}"
    )
    typer.echo(
        "Response divergence stability: "
        f"{result.response_divergence_stability}; "
        f"counts={_format_divergence_counts(result.response_divergence_counts)}"
    )
    typer.echo(
        "Committed-state divergence stability: "
        f"{result.committed_state_divergence_stability}; "
        f"counts={_format_divergence_counts(result.committed_state_divergence_counts)}"
    )
    if result.baseline_drift_observed:
        typer.echo(
            "Causal warning: the variation diverged before the correction; do not attribute "
            "the corrected-arm failure to the correction alone."
        )
    for trial in result.trials:
        typer.echo(f"Repetition {trial.repetition}")
        arms = (("baseline", trial.baseline), ("variation", trial.variation))
        for arm_name, observations in arms:
            for index, observation in enumerate(observations, start=1):
                typer.echo(f"  {arm_name} turn {index}: {observation.turn.id}")
    for arm, rules in (
        ("baseline", result.baseline_invariant_rules),
        ("corrected", result.corrected_invariant_rules),
    ):
        for rule in rules:
            typer.echo(
                f"Invariant {rule.rule_id}: {rule.status}; arm={arm}; severity={rule.severity}"
            )
    typer.echo(f"Complete evidence: {output}")
    if result.status == "failed":
        raise typer.Exit(code=1)
    if result.status == "inconclusive":
        raise typer.Exit(code=2)


def _print_retry_result(result: RetryAfterSuccessfulCommitStressResult, output: Path) -> None:
    typer.echo(f"Retry-after-successful-commit result: {result.status}")
    typer.echo(f"Operator: {result.case.operator_id}@{result.case.operator_version}")
    typer.echo("Exact baseline, successful-commit, and retried state are in private evidence.")
    if result.baseline_drift_observed:
        typer.echo(
            "Causal warning: the variation diverged before the retry; do not attribute the "
            "result to the retry alone."
        )
    for baseline_rule, successful_commit_rule, retried_rule in zip(
        result.baseline_invariant_rules,
        result.successful_commit_invariant_rules,
        result.retried_invariant_rules,
        strict=True,
    ):
        typer.echo(
            f"Invariant {baseline_rule.rule_id}: baseline={baseline_rule.status}; "
            f"first_commit={successful_commit_rule.status}; after_retry={retried_rule.status}; "
            f"severity={baseline_rule.severity}"
        )
    typer.echo(f"Complete evidence: {output}")
    if result.status == "failed":
        raise typer.Exit(code=1)
    if result.status == "inconclusive":
        raise typer.Exit(code=2)


def _print_timeout_after_commit_result(
    result: TimeoutAfterCommitStressResult,
    output: Path,
) -> None:
    typer.echo(f"Timeout-after-commit result: {result.status}")
    fired_count = sum(
        trial.execution_evidence is not None
        and trial.execution_evidence.timeout_after_commit_event is not None
        and trial.execution_evidence.timeout_after_commit_event.trigger_status == "fired"
        for trial in result.trials
    )
    typer.echo(f"Triggered events: {fired_count}/{result.requested_repetitions}")
    for rule in result.invariant_rules:
        typer.echo(f"Invariant {rule.rule_id}: {rule.status}; severity={rule.severity}")
    typer.echo("Exact responses, committed state, and event receipts are in private evidence.")
    typer.echo(f"Complete evidence: {output}")
    if result.status == "failed":
        raise typer.Exit(code=1)
    if result.status == "inconclusive":
        raise typer.Exit(code=2)


def _print_trace_replay_result(result: TraceReplayResult, output: Path) -> None:
    typer.echo(f"Trace replay result: {result.status}")
    typer.echo(f"Replay case: {result.case.case_id}")
    typer.echo(
        f"Recorded response matches: {result.response_match_count}/{result.requested_repetitions}"
    )
    if result.state_match_count is None:
        typer.echo("Recorded committed state: unavailable in the imported trace")
    else:
        typer.echo(
            f"Recorded state matches: {result.state_match_count}/{result.requested_repetitions}"
        )
    typer.echo(
        "Interpretation: replay establishes reproducibility only; response or state drift is not "
        "automatically a correctness failure."
    )
    typer.echo(f"Complete evidence: {output}")
    if result.status == "drifted":
        raise typer.Exit(code=1)
    if result.status == "inconclusive":
        raise typer.Exit(code=2)


def _print_trace_stress_plan(plan: TraceStressPlan, bundle_path: Path) -> None:
    typer.echo(f"Trace-derived stress plan: {plan.case_count} case(s)")
    for planned_case in plan.cases:
        typer.echo(
            f"{planned_case.priority_rank}. {planned_case.case_id} "
            f"(score={planned_case.priority_score})"
        )
        typer.echo(f"   Trace: {planned_case.source_trace_id}")
        if planned_case.signals:
            typer.echo("   Why this case is prioritized:")
            for signal in planned_case.signals:
                typer.echo(f"   - {signal.description} [{signal.code}]")
        else:
            typer.echo("   Why this case is prioritized: no elevated trace signals")
        focuses = ", ".join(focus.replace("_", " ") for focus in planned_case.recommended_focuses)
        typer.echo(f"   Suggested stress focus: {focuses}")
        if planned_case.source_span_ids:
            typer.echo(f"   Case spans: {', '.join(planned_case.source_span_ids)}")
        signal_span_ids = tuple(
            dict.fromkeys(
                span_id for signal in planned_case.signals for span_id in signal.source_span_ids
            )
        )
        if signal_span_ids:
            typer.echo(f"   Signal spans: {', '.join(signal_span_ids)}")
        dry_run_command = " ".join(
            (
                "ul stress trace",
                shlex.quote(str(bundle_path)),
                "--case-id",
                planned_case.case_id,
                "--environment-config .ul/environment.json --dry-run",
            )
        )
        typer.echo(f"   Next replay check: {dry_run_command}")
    typer.echo("Priority score schedules evidence-rich cases; it is not risk or causality.")
    typer.echo(
        "Suggested focus labels are review directions, not automatically runnable augmentations."
    )
    typer.echo("Recorded message and state content: not printed")


def _print_trace_replay_campaign_plan(
    plan: TraceReplayCampaignPlan,
    *,
    bundle_path: Path,
    target_config_path: Path,
    allow_insecure_http: bool,
) -> None:
    typer.echo(
        "Trace replay campaign: "
        f"{plan.selected_case_count}/{plan.total_case_count} prioritized case(s)"
    )
    typer.echo(f"Repetitions per case: {plan.repetitions}")
    typer.echo(
        "Potential environment API calls: "
        f"{plan.required_target_calls} / {plan.authorized_target_calls} authorized"
    )
    typer.echo("Execution order:")
    for case in plan.cases:
        focuses = ", ".join(focus.replace("_", " ") for focus in case.stress.recommended_focuses)
        signals = ", ".join(signal.code for signal in case.stress.signals) or "baseline replay"
        typer.echo(
            f"  {case.stress.priority_rank}. {case.replay.case_id} "
            f"(priority score {case.stress.priority_score})"
        )
        typer.echo(f"     Why: {signals}")
        typer.echo(f"     Review focus: {focuses}")
        typer.echo(
            f"     Calls: {case.replay.target_calls_per_repetition} per replay x "
            f"{case.replay.repetitions} = {case.replay.required_target_calls}"
        )
    typer.echo("Recorded message and state content: not printed")
    typer.echo("External calls: none")
    command_parts = [
        "ul",
        "stress",
        "trace-replay-campaign",
        str(bundle_path),
        "--environment-config",
        str(target_config_path),
        "--output",
        "trace-replay-campaign.json",
        "--limit",
        str(plan.selected_case_count),
        "--repetitions",
        str(plan.repetitions),
        "--max-environment-api-calls",
        str(plan.authorized_target_calls),
        "--allow-environment-network",
        "--confirm-test-environment",
    ]
    if allow_insecure_http:
        command_parts.append("--allow-insecure-http")
    typer.echo(f"Next: {' '.join(shlex.quote(part) for part in command_parts)}")


def _print_trace_replay_campaign_result(result: TraceReplayCampaignResult, output: Path) -> None:
    reproduced = result.grouping.reproduced_count
    drifted = sum(case.status == "drifted" for case in result.results)
    inconclusive = sum(case.status == "inconclusive" for case in result.results)
    typer.echo(f"Trace replay campaign complete: {len(result.results)} case(s)")
    typer.echo(f"Reproduced: {reproduced}  Different: {drifted}  Inconclusive: {inconclusive}")
    typer.echo(f"Difference patterns: {len(result.grouping.groups)}")
    for group in result.grouping.groups:
        typer.echo(f"  {group.signature}: {group.occurrence_count} case(s)")
        for reason_code in group.reason_codes:
            typer.echo(f"    {_trace_replay_reason_explanation(reason_code)}")
        typer.echo(f"    Cases: {', '.join(member.case_id for member in group.members)}")
    typer.echo(f"Complete private evidence: {output}")
    typer.echo("Interpretation: differences show replay drift, not correctness or cause.")
    if inconclusive:
        raise typer.Exit(code=2)


def _print_trace_replay_difference_grouping(
    grouping: TraceReplayDifferenceGrouping,
) -> None:
    typer.echo(
        f"Trace replay patterns: {grouping.difference_count} difference(s), "
        f"{grouping.reproduced_count} reproduced"
    )
    if not grouping.groups:
        typer.echo("No drifted or inconclusive replay results to group.")
        return
    for group in grouping.groups:
        typer.echo(f"{group.signature}: {group.occurrence_count} occurrence(s)")
        for reason_code in group.reason_codes:
            typer.echo(f"  - {_trace_replay_reason_explanation(reason_code)}")
        for member in group.members:
            typer.echo(f"  Case {member.case_id} (trace {member.source_trace_id})")
            if member.source_span_ids:
                typer.echo(f"    Spans: {', '.join(member.source_span_ids)}")
    typer.echo(
        "Groups describe technical replay differences only. They do not identify a cause or "
        "a semantic agent failure."
    )


def _trace_replay_reason_explanation(reason_code: str) -> str:
    explanations = {
        "response_mismatch": "At least one replay response differed from the recorded response.",
        "state_mismatch": "At least one observed replay state differed from recorded state.",
        "environment_lifecycle_failed": "The test environment did not complete its lifecycle.",
        "environment_execution_timeout": "The test environment did not answer before timeout.",
        "environment_execution_failed": "The test environment could not complete execution.",
        "environment_state_uncertain": "A prior run left the test environment state uncertain.",
        "other_inconclusive": "The replay was inconclusive for another recorded reason.",
    }
    return explanations[reason_code]


def _format_divergence_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{turn_id}={count}" for turn_id, count in counts.items())


def _create_private_output(path: Path) -> TextIO:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("output is not a regular file")
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "w", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise
