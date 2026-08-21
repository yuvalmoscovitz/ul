from __future__ import annotations

import asyncio
import json
import unicodedata
import uuid
from pathlib import Path
from typing import Annotated, Never

import typer
from ul.environment import evaluation_case_from_inputs, validate_execution_evidence
from ul.http_environment import (
    ENVIRONMENT_ID_PLACEHOLDER,
    JsonHttpEnvironmentConnection,
    json_http_environment_calls_per_execution,
    json_http_environment_capabilities,
    json_http_environment_config_sha256,
    load_json_http_environment_config,
)
from ul_core.evaluation import EnvironmentResetEvidence, EvaluationCase, ExecutionEvidence

app = typer.Typer(help="Validate a customer-managed environment connection.")
TEST_ENVIRONMENT_CONFIRMATION_MESSAGE = (
    "UL will call this environment, and the agent may change external state. Use an environment "
    "intended for testing that you can reset. Re-run with --confirm-test-environment to continue."
)


@app.command("check")
def check_environment(
    environment_config: Annotated[
        Path,
        typer.Argument(),
    ],
    probe: Annotated[
        str | None,
        typer.Option(
            help="Non-sensitive input that the customer has verified cannot cause a real effect."
        ),
    ] = None,
    allow_environment_network: Annotated[
        bool,
        typer.Option(
            "--allow-environment-network",
            help="Authorize configured environment API calls.",
            rich_help_panel="Required safety flags",
        ),
    ] = False,
    confirm_test_environment: Annotated[
        bool,
        typer.Option(
            "--confirm-test-environment",
            help="Confirm this is a dedicated test environment, not a production target.",
            rich_help_panel="Required safety flags",
        ),
    ] = False,
    confirm_harmless_probe: Annotated[
        bool,
        typer.Option(
            "--confirm-harmless-probe",
            help="Attest the probe cannot cause a real business effect.",
            rich_help_panel="Required safety flags",
        ),
    ] = False,
    allow_insecure_http: Annotated[
        bool,
        typer.Option(
            "--allow-insecure-http",
            help="Allow an HTTP environment API. Intended for local environments.",
        ),
    ] = False,
    request_timeout_seconds: Annotated[
        float,
        typer.Option(
            "--timeout-seconds",
            min=0.1,
            max=60,
            help="Timeout for each environment API request.",
        ),
    ] = 10,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Print a safe machine-readable result."),
    ] = False,
) -> None:
    """Run one probe through the complete lifecycle or response-only adapter.

    Requires --allow-environment-network, --confirm-test-environment, and
    --confirm-harmless-probe.
    """
    if probe is None or not probe.strip():
        _preflight_failure(
            output_json=output_json,
            code="probe_missing",
            reason="probe must contain non-whitespace text",
            remediation="Provide a non-sensitive input with --probe.",
            param_hint="--probe",
        )
    try:
        encoded_probe = probe.encode("utf-8")
    except UnicodeEncodeError:
        _preflight_failure(
            output_json=output_json,
            code="probe_invalid",
            reason="probe must be valid UTF-8 text",
            remediation="Provide a valid non-sensitive UTF-8 probe.",
            param_hint="--probe",
        )
    if len(encoded_probe) > 100_000:
        _preflight_failure(
            output_json=output_json,
            code="probe_too_large",
            reason="probe exceeds the 100 KB limit",
            remediation="Use a smaller non-sensitive probe.",
            param_hint="--probe",
        )
    try:
        config = load_json_http_environment_config(environment_config)
    except (ValueError, RuntimeError) as error:
        reason = _terminal_safe(str(error))
        _preflight_failure(
            output_json=output_json,
            code="environment_config_invalid",
            reason=reason,
            remediation="Correct the environment configuration and try again.",
            param_hint="ENVIRONMENT_CONFIG",
        )
    if config.environment_id == ENVIRONMENT_ID_PLACEHOLDER:
        _preflight_failure(
            output_json=output_json,
            code="environment_config_invalid",
            reason="environment_id is still the generated placeholder",
            remediation="Replace environment_id with a stable name for this test environment.",
            param_hint="ENVIRONMENT_CONFIG",
        )
    if not allow_environment_network:
        _preflight_failure(
            output_json=output_json,
            code="environment_network_not_allowed",
            reason="check requires --allow-environment-network",
            remediation="Explicitly authorize environment API calls for this check.",
            param_hint="--allow-environment-network",
        )
    if not confirm_test_environment:
        capabilities = json_http_environment_capabilities(config)
        _preflight_failure(
            output_json=output_json,
            code="test_environment_not_confirmed",
            reason=(
                "UL will send an isolated request to this target. Use a dedicated test target, "
                "not a production system. Re-run with --confirm-test-environment to continue."
                if capabilities.request_isolation == "per_request_attested"
                else TEST_ENVIRONMENT_CONFIRMATION_MESSAGE
            ),
            remediation="Confirm the environment before UL calls it.",
            param_hint="--confirm-test-environment",
        )
    if not confirm_harmless_probe:
        _preflight_failure(
            output_json=output_json,
            code="probe_safety_not_attested",
            reason="check requires --confirm-harmless-probe",
            remediation="Verify the probe cannot cause a real effect, then attest it.",
            param_hint="--confirm-harmless-probe",
        )

    environment_api_calls = json_http_environment_calls_per_execution(config)
    try:
        environment = JsonHttpEnvironmentConnection.from_config(
            config,
            test_environment_confirmed=True,
            allow_insecure_http=allow_insecure_http,
            timeout_seconds=request_timeout_seconds,
            max_environment_api_calls=environment_api_calls,
        )
    except (ValueError, RuntimeError) as error:
        reason = _terminal_safe(str(error))
        if "header environment variable" in reason:
            code = "credential_configuration"
            remediation = (
                "Set the configured credential environment variable to a valid non-empty "
                "ASCII value."
            )
            param_hint = "ENVIRONMENT_CONFIG"
        elif reason == "HTTP endpoints require explicit insecure transport opt-in":
            code = "insecure_http_not_allowed"
            remediation = (
                "For a local environment, pass --allow-insecure-http; otherwise use HTTPS."
            )
            param_hint = "--allow-insecure-http"
        else:
            code = "environment_config_invalid"
            remediation = "Correct the environment configuration and try again."
            param_hint = "ENVIRONMENT_CONFIG"
        _preflight_failure(
            output_json=output_json,
            code=code,
            reason=reason,
            remediation=remediation,
            param_hint=param_hint,
        )
    case = evaluation_case_from_inputs(
        case_id=f"ul-case-{uuid.uuid4().hex}",
        raw_inputs=(probe,),
        max_environment_api_calls=environment_api_calls,
        timeout_seconds=(environment_api_calls + 1) * request_timeout_seconds,
        required_state_observation_authority=(
            "environment_self_reported"
            if json_http_environment_capabilities(config).supports_state_observation
            else None
        ),
    )
    try:
        evidence = asyncio.run(_execute_and_close(environment, case))
    except TimeoutError:
        _print_timeout_result(
            config.environment_id,
            json_http_environment_config_sha256(config),
            environment_api_calls,
            output_json=output_json,
        )
        raise typer.Exit(code=2) from None
    except Exception:
        _print_unexpected_execution_result(
            config.environment_id,
            json_http_environment_config_sha256(config),
            environment_api_calls,
            output_json=output_json,
        )
        raise typer.Exit(code=2) from None
    try:
        validate_execution_evidence(case, environment, evidence)
    except ValueError:
        _print_invalid_evidence_result(
            config.environment_id,
            json_http_environment_config_sha256(config),
            environment_api_calls,
            output_json=output_json,
        )
        raise typer.Exit(code=2) from None

    if evidence.lifecycle.terminal_status == "succeeded":
        _print_success(evidence, environment_api_calls, output_json=output_json)
        return
    _print_failure(evidence, environment_api_calls, output_json=output_json)
    raise typer.Exit(code=2)


async def _execute_and_close(
    environment: JsonHttpEnvironmentConnection, case: EvaluationCase
) -> ExecutionEvidence:
    try:
        return await environment.execute(case)
    finally:
        await environment.aclose()


def _print_success(
    evidence: ExecutionEvidence, environment_api_calls: int, *, output_json: bool
) -> None:
    result = _environment_check_result(
        status="ready",
        environment_id=evidence.environment_id,
        environment_config_sha256=evidence.environment_config_sha256,
        environment_api_call_budget=environment_api_calls,
        completed_phases=evidence.lifecycle.completed_phases,
        state_observation_authority=evidence.final_state.authority
        if evidence.final_state is not None
        else None,
        delivery=evidence.lifecycle.delivery,
        cleanup=evidence.lifecycle.cleanup,
        initial_reset=(
            evidence.lifecycle.initial_reset.model_dump(mode="json")
            if evidence.lifecycle.initial_reset is not None
            else None
        ),
        cleanup_reset=(
            evidence.lifecycle.cleanup_reset.model_dump(mode="json")
            if evidence.lifecycle.cleanup_reset is not None
            else None
        ),
        environment_state_uncertain=evidence.lifecycle.environment_state_uncertain,
    )
    if output_json:
        _print_safe(json.dumps(result, sort_keys=True))
        return
    _print_safe("Environment check: READY")
    _print_safe(f"Environment: {evidence.environment_id}")
    _print_safe(f"Lifecycle: {' -> '.join(evidence.lifecycle.completed_phases)}")
    _print_safe(f"Environment API call budget: {environment_api_calls}")
    if evidence.lifecycle.initial_reset is not None:
        _print_reset_receipt("Initial reset", evidence.lifecycle.initial_reset)
    if evidence.lifecycle.cleanup_reset is not None:
        _print_reset_receipt("Cleanup reset", evidence.lifecycle.cleanup_reset)
    _print_safe(
        "Probe and response: not printed"
        if evidence.evidence_scope == "response_only"
        else "Probe, response, and state: not printed"
    )
    _print_safe("UL semantic-model calls: 0")
    _print_safe("Isolation and probe safety: customer-attested, not verified by UL")


def _print_failure(
    evidence: ExecutionEvidence, environment_api_calls: int, *, output_json: bool
) -> None:
    lifecycle = evidence.lifecycle
    reason = lifecycle.failure_reason or "environment lifecycle failed without a diagnostic reason"
    code = lifecycle.failure_code
    if code is None:
        raise AssertionError("failed environment lifecycle requires a diagnostic code")
    remediation = _remediation_for_code(code)
    result = _environment_check_result(
        status="not_ready",
        environment_id=evidence.environment_id,
        environment_config_sha256=evidence.environment_config_sha256,
        environment_api_call_budget=environment_api_calls,
        completed_phases=lifecycle.completed_phases,
        failed_phase=lifecycle.failed_phase,
        error_code=code,
        reason=reason,
        remediation=remediation,
        delivery=lifecycle.delivery,
        cleanup=lifecycle.cleanup,
        initial_reset=(
            lifecycle.initial_reset.model_dump(mode="json")
            if lifecycle.initial_reset is not None
            else None
        ),
        cleanup_reset=(
            lifecycle.cleanup_reset.model_dump(mode="json")
            if lifecycle.cleanup_reset is not None
            else None
        ),
        cleanup_failure_code=lifecycle.cleanup_failure_code,
        cleanup_failure_reason=lifecycle.cleanup_failure_reason,
        environment_state_uncertain=lifecycle.environment_state_uncertain,
    )
    if output_json:
        _print_safe(json.dumps(result, sort_keys=True))
        return
    _print_safe("Environment check: NOT READY")
    _print_safe(f"Failed phase: {lifecycle.failed_phase or 'unknown'}")
    _print_safe(f"Error: {code}")
    _print_safe(f"Reason: {reason}")
    _print_safe(f"Remediation: {remediation}")
    _print_safe(f"Delivery: {lifecycle.delivery}")
    _print_safe(f"Cleanup: {lifecycle.cleanup}")
    if lifecycle.initial_reset is not None:
        _print_reset_receipt("Initial reset", lifecycle.initial_reset)
    if lifecycle.cleanup_reset is not None:
        _print_reset_receipt("Cleanup reset", lifecycle.cleanup_reset)
    if lifecycle.cleanup_failure_reason is not None:
        _print_safe(f"Cleanup reason: {lifecycle.cleanup_failure_reason}")
    _print_safe(
        "Environment state: "
        + (
            "UNCERTAIN — quarantine before reuse"
            if lifecycle.environment_state_uncertain
            else "clean"
        )
    )
    _print_safe("Probe, response, and state: not printed")
    _print_safe("UL semantic-model calls: 0")


def _print_timeout_result(
    environment_id: str,
    environment_config_sha256: str,
    environment_api_calls: int,
    *,
    output_json: bool,
) -> None:
    result = _environment_check_result(
        status="not_ready",
        environment_id=environment_id,
        environment_config_sha256=environment_config_sha256,
        environment_api_call_budget=environment_api_calls,
        failed_phase="lifecycle_deadline",
        error_code="lifecycle_timeout",
        reason="complete environment lifecycle exceeded its deadline",
        remediation="Inspect environment availability and quarantine it before reuse.",
        delivery="uncertain",
        cleanup="unknown",
        environment_state_uncertain=True,
    )
    if output_json:
        _print_safe(json.dumps(result, sort_keys=True))
        return
    _print_safe("Environment check: NOT READY")
    _print_safe("Failed phase: lifecycle_deadline")
    _print_safe("Error: lifecycle_timeout")
    _print_safe("Reason: complete environment lifecycle exceeded its deadline")
    _print_safe("Delivery: uncertain")
    _print_safe("Cleanup: unknown")
    _print_safe("Environment state: UNCERTAIN — quarantine before reuse")
    _print_safe("Probe, response, and state: not printed")
    _print_safe("UL semantic-model calls: 0")


def _print_invalid_evidence_result(
    environment_id: str,
    environment_config_sha256: str,
    environment_api_calls: int,
    *,
    output_json: bool,
) -> None:
    result = _environment_check_result(
        status="not_ready",
        environment_id=environment_id,
        environment_config_sha256=environment_config_sha256,
        environment_api_call_budget=environment_api_calls,
        failed_phase="evidence_validation",
        error_code="invalid_lifecycle_evidence",
        reason="environment lifecycle evidence did not match the requested probe",
        remediation="Verify environment, case, turn, and state-observer identity handling.",
        delivery="uncertain",
        cleanup="unknown",
        environment_state_uncertain=True,
    )
    if output_json:
        _print_safe(json.dumps(result, sort_keys=True))
        return
    _print_safe("Environment check: NOT READY")
    _print_safe("Failed phase: evidence_validation")
    _print_safe("Error: invalid_lifecycle_evidence")
    _print_safe("Reason: environment lifecycle evidence did not match the requested probe")
    _print_safe(
        "Remediation: Verify environment, case, turn, and state-observer identity handling."
    )
    _print_safe("Probe, response, and state: not printed")
    _print_safe("UL semantic-model calls: 0")


def _print_unexpected_execution_result(
    environment_id: str,
    environment_config_sha256: str,
    environment_api_calls: int,
    *,
    output_json: bool,
) -> None:
    reason = "environment check failed before safe lifecycle evidence was available"
    remediation = (
        "Quarantine the environment before reuse and inspect its lifecycle implementation."
    )
    result = _environment_check_result(
        status="not_ready",
        environment_id=environment_id,
        environment_config_sha256=environment_config_sha256,
        environment_api_call_budget=environment_api_calls,
        failed_phase="execution_boundary",
        error_code="unexpected_execution_error",
        reason=reason,
        remediation=remediation,
        delivery="uncertain",
        cleanup="unknown",
        environment_state_uncertain=True,
    )
    if output_json:
        _print_safe(json.dumps(result, sort_keys=True))
        return
    _print_safe("Environment check: NOT READY")
    _print_safe("Failed phase: execution_boundary")
    _print_safe("Error: unexpected_execution_error")
    _print_safe(f"Reason: {reason}")
    _print_safe(f"Remediation: {remediation}")
    _print_safe("Delivery: uncertain")
    _print_safe("Cleanup: unknown")
    _print_safe("Environment state: UNCERTAIN — quarantine before reuse")
    _print_safe("Probe, response, and state: not printed")
    _print_safe("UL semantic-model calls: 0")


def _environment_check_result(
    *,
    status: str,
    environment_id: str | None = None,
    environment_config_sha256: str | None = None,
    environment_api_call_budget: int | None = None,
    completed_phases: tuple[str, ...] = (),
    state_observation_authority: str | None = None,
    failed_phase: str | None = None,
    error_code: str | None = None,
    reason: str | None = None,
    remediation: str | None = None,
    delivery: str | None = None,
    cleanup: str | None = None,
    cleanup_failure_code: str | None = None,
    cleanup_failure_reason: str | None = None,
    environment_state_uncertain: bool | None = None,
    initial_reset: dict[str, object] | None = None,
    cleanup_reset: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "status": status,
        "environment_id": environment_id,
        "environment_config_sha256": environment_config_sha256,
        "environment_api_call_budget": environment_api_call_budget,
        "completed_phases": list(completed_phases),
        "state_observation_authority": state_observation_authority,
        "failed_phase": failed_phase,
        "error_code": error_code,
        "reason": reason,
        "remediation": remediation,
        "delivery": delivery,
        "cleanup": cleanup,
        "cleanup_failure_code": cleanup_failure_code,
        "cleanup_failure_reason": cleanup_failure_reason,
        "environment_state_uncertain": environment_state_uncertain,
        "initial_reset": initial_reset,
        "cleanup_reset": cleanup_reset,
        "probe_and_observations": "not_printed",
        "ul_semantic_model_calls": 0,
    }


def _print_reset_receipt(label: str, reset: EnvironmentResetEvidence) -> None:
    _print_safe(
        f"{label}: session requested={reset.reset_session_requested}, "
        f"acknowledged={reset.reset_session_acknowledged}; environment "
        f"requested={reset.reset_env_requested}, acknowledged={reset.reset_env_acknowledged}"
    )


def _remediation_for_code(code: str) -> str:
    return {
        "authentication_rejected": "Verify environment credentials and permissions.",
        "rate_limited": "Wait for environment capacity or adjust its request quota.",
        "http_status": "Verify endpoint routing, request identity, and environment service logs.",
        "response_content_type": "Return application/json from this lifecycle endpoint.",
        "response_content_encoding": "Disable compressed lifecycle responses.",
        "invalid_json": "Return bounded, standards-compliant JSON without duplicate keys.",
        "null_json": "Return the configured lifecycle response object instead of null.",
        "response_mapping": "Correct the response JSON pointer for this lifecycle phase.",
        "environment_identity": (
            "Return the configured environment_id and the request case identity unchanged."
        ),
        "case_identity": "Echo the request case_id unchanged in the response.",
        "turn_identity": "Echo the request turn_id unchanged in the response.",
        "reset_generation": "Return a non-empty string or integer reset generation.",
        "reset_generation_reused": "Return a new generation after every reset.",
        "reset_session_not_acknowledged": (
            "Return reset_session=true after clearing the agent conversation/session."
        ),
        "reset_env_not_acknowledged": (
            "Return reset_env=true after restoring external environment state."
        ),
        "reset_not_clean": (
            "Return the configured clean-state acknowledgement only after reset completes."
        ),
        "request_too_large": "Reduce the probe or configured request template size.",
        "response_too_large": "Reduce the lifecycle response size.",
        "call_budget": "Increase the explicit environment API call budget.",
        "request_timeout": (
            "Inspect environment availability and state; do not retry an ambiguous mutation "
            "blindly."
        ),
        "write_timeout": (
            "Inspect environment availability and state; do not retry an ambiguous mutation "
            "blindly."
        ),
        "response_timeout": (
            "Inspect environment availability and state; do not retry an ambiguous mutation "
            "blindly."
        ),
        "connect_timeout": "Verify the environment address, firewall, and service availability.",
        "pool_timeout": "Reduce local concurrency or increase environment connection capacity.",
        "dns_resolution": "Verify the configured environment hostname and DNS availability.",
        "tls_connection": "Verify the environment certificate, hostname, and trust configuration.",
        "connect_failed": (
            "Verify the environment address, DNS, TLS, firewall, and service availability."
        ),
        "transport_protocol": (
            "Inspect the environment HTTP server and intermediary protocol handling."
        ),
        "transport_failed": (
            "Inspect transport health and environment state; do not retry an ambiguous execute "
            "blindly."
        ),
        "environment_state_uncertain": (
            "Quarantine and independently reset the environment before reuse."
        ),
        "environment_lifecycle_error": (
            "Quarantine the environment before reuse and inspect its lifecycle implementation."
        ),
        "environment_cleanup_error": (
            "Quarantine the environment and independently restore clean state before reuse."
        ),
    }.get(code, "Inspect the configured lifecycle contract for the failed phase.")


def _print_safe(message: str) -> None:
    typer.echo(_terminal_safe(message))


def _terminal_safe(message: str) -> str:
    return "".join(
        character
        if (ord(character) >= 32 and not 0x7F <= ord(character) <= 0x9F)
        and unicodedata.category(character) not in {"Cf", "Cs"}
        else f"\\u{ord(character):04x}"
        for character in message
    )


def _preflight_failure(
    *,
    output_json: bool,
    code: str,
    reason: str,
    remediation: str,
    param_hint: str,
) -> Never:
    if output_json:
        _print_safe(
            json.dumps(
                _environment_check_result(
                    status="not_ready",
                    failed_phase="preflight",
                    error_code=code,
                    reason=reason,
                    remediation=remediation,
                ),
                sort_keys=True,
            )
        )
        raise typer.Exit(code=2)
    raise typer.BadParameter(reason, param_hint=param_hint)
