from __future__ import annotations

import json
import re
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner
from ul import (
    InteractionRecord,
    JsonHttpEnvironmentConfig,
    ProviderDiagnostic,
    ProviderDiagnosticError,
)
from ul.environment import evaluation_case_from_inputs
from ul_cli.dataset.evaluation import command as command_module
from ul_cli.dataset.evaluation import runner as runner_module
from ul_cli.dataset.evidence import persistence as persistence_module
from ul_cli.dataset_trial_journal import manifest_path, read_dataset_run_manifest
from ul_cli.main import app as root_app

from ._factories import (
    _evaluator_preflight,
    _settings,
)
from ._files import (
    _record,
    _write_dataset,
    _write_target_config,
)

runner = CliRunner()
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def test_execution_requires_config_network_confirmation_environment_and_output(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config)

    option_stages = [
        ([], "--environment-config"),
        (["--environment-config", str(target_config)], "--allow-environment-network"),
        (
            ["--environment-config", str(target_config), "--allow-environment-network"],
            "--confirm-test-environment",
        ),
        (
            [
                "--environment-config",
                str(target_config),
                "--allow-environment-network",
                "--confirm-test-environment",
            ],
            "execution requires --output",
        ),
    ]
    for options, expected_error in option_stages:
        result = runner.invoke(root_app, ["dataset", "evaluate", str(dataset), *options])
        assert result.exit_code != 0
        normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", result.output).split())
        assert expected_error in normalized_output


def test_execution_refuses_to_overwrite_output_before_model_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config)
    output.write_text("keep me", encoding="utf-8")

    def unexpected_settings() -> None:
        raise AssertionError("output collision reached model setup")

    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", unexpected_settings)
    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--allow-environment-network",
            "--confirm-test-environment",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert "will not overwrite" in result.output
    assert output.read_text(encoding="utf-8") == "keep me"


def test_execution_refuses_default_augmentations_collision_before_model_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "results.jsonl"
    augmentations = tmp_path / "results.augmentations.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config)
    augmentations.write_text("keep me\n", encoding="utf-8")

    def unexpected_settings() -> None:
        raise AssertionError("augmentations collision reached model setup")

    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", unexpected_settings)
    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--allow-environment-network",
            "--confirm-test-environment",
            "--output",
            str(evidence),
        ],
    )

    assert result.exit_code != 0
    normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", result.output).split())
    assert "augmentations output already" in normalized_output
    assert "exists; UL will not overwrite it" in normalized_output
    assert not evidence.exists()
    assert augmentations.read_text(encoding="utf-8") == "keep me\n"


def test_invalid_custom_augmentations_path_does_not_strand_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "results.jsonl"
    augmentations = tmp_path / "missing" / "augmentations.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config)

    class FakeTarget:
        @classmethod
        def from_config(cls, *_args: object, **_kwargs: object) -> FakeTarget:
            return cls()

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", _settings)
    monkeypatch.setattr(command_module, "JsonHttpEnvironmentConnection", FakeTarget)

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--allow-environment-network",
            "--confirm-test-environment",
            "--output",
            str(evidence),
            "--augmentations-output",
            str(augmentations),
        ],
    )

    assert result.exit_code != 0
    normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", result.output).split())
    assert "--augmentations-output" in normalized_output
    assert "cannot safely open" in normalized_output
    assert "FileNotFoundError" in normalized_output
    assert not evidence.exists()
    assert not augmentations.exists()


def test_execution_rejects_missing_header_secret_before_model_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(
        target_config,
        headers_from_env={"Authorization": "UL_ENVIRONMENT_MISSING_TOKEN"},
    )
    monkeypatch.delenv("UL_ENVIRONMENT_MISSING_TOKEN", raising=False)
    monkeypatch.setattr(
        command_module,
        "load_dataset_semantic_settings",
        _settings,
    )

    def unexpected_deconstructor(*args: object, **kwargs: object) -> None:
        raise AssertionError("missing target auth reached semantic model setup")

    monkeypatch.setattr(
        runner_module, "create_semantic_model_deconstructor", unexpected_deconstructor
    )
    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--allow-environment-network",
            "--confirm-test-environment",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert not output.exists()


def test_execution_creates_private_explicit_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config, url="http://127.0.0.1:8765/execute")
    captured_records: list[str] = []

    class FakeTarget:
        @classmethod
        def from_config(cls, config: JsonHttpEnvironmentConfig, **options: object) -> FakeTarget:
            assert config.execute_turn.url == "http://127.0.0.1:8765/execute"
            assert options["test_environment_confirmed"] is True
            return cls()

    async def fake_evaluate(
        records: tuple[Any, ...],
        operator_ids: tuple[str, ...],
        settings: object,
        target: object,
        output_stream: Any,
        *,
        repetitions: int,
        max_environment_api_calls: int,
        planned_target_calls: int,
        run_context: object,
        augmentation_ledger: object,
        saved_augmentations: object,
        redaction_engine: object,
        evaluator_preflight: object,
        trial_journal: object,
    ) -> tuple[object, ...]:
        del settings, target, run_context, augmentation_ledger, saved_augmentations
        assert evaluator_preflight == _evaluator_preflight()
        assert redaction_engine is None
        captured_records.extend(record.id for record in records)
        assert operator_ids == ("input.surface.disfluency_repeat",)
        assert repetitions == 3
        assert max_environment_api_calls == 100
        assert planned_target_calls == 30
        output_stream.write('{"saved":true}\n')
        output_stream.flush()
        return ()

    monkeypatch.setattr(
        command_module,
        "load_dataset_semantic_settings",
        _settings,
    )
    monkeypatch.setattr(command_module, "JsonHttpEnvironmentConnection", FakeTarget)
    monkeypatch.setattr(command_module, "evaluate_interaction_records", fake_evaluate)
    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--operator",
            "input.surface.disfluency_repeat",
            "--environment-config",
            str(target_config),
            "--allow-insecure-http",
            "--allow-environment-network",
            "--confirm-test-environment",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured_records == ["interaction-1"]
    output_lines = output.read_text(encoding="utf-8").splitlines()
    assert json.loads(output_lines[0])["record_type"] == "dataset_durable_run"
    assert output_lines[1] == '{"saved":true}'
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert "Complete evidence" in result.output
    assert "Next: ul dataset report" in result.output
    assert "Transfer 100" not in result.output


def test_provider_failure_has_concise_output_and_private_sanitized_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    target_config = tmp_path / "target.json"
    secret = "private-provider-response"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config, url="http://127.0.0.1:8765/execute")

    class FakeTarget:
        @classmethod
        def from_config(cls, *_args: object, **_kwargs: object) -> FakeTarget:
            return cls()

    async def fail_evaluation(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        error = ProviderDiagnosticError(
            ProviderDiagnostic(
                provider="customer-gateway",
                operation="verify",
                category="provider_unavailable",
                retryable=True,
                suggested_action="check provider status, then resume the run.",
                endpoint_sha256="a" * 64,
                http_status=503,
            )
        )
        error.add_note(secret)
        raise error

    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", _settings)
    monkeypatch.setattr(command_module, "JsonHttpEnvironmentConnection", FakeTarget)
    monkeypatch.setattr(command_module, "evaluate_interaction_records", fail_evaluation)

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--allow-insecure-http",
            "--allow-environment-network",
            "--confirm-test-environment",
            "--output",
            str(output),
            "--no-save-augmentations",
        ],
    )

    diagnostics = tmp_path / "results.jsonl.debug.json"
    normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", result.output).split())
    assert result.exit_code == 2
    assert "customer-gateway failed during verify" in normalized_output
    assert "provider_unavailable; retryable: yes" in normalized_output
    assert "Next: check provider status" in normalized_output
    assert secret not in normalized_output
    assert diagnostics.exists()
    assert stat.S_IMODE(diagnostics.stat().st_mode) == 0o600
    serialized_diagnostics = diagnostics.read_text(encoding="utf-8")
    assert secret not in serialized_diagnostics
    assert json.loads(serialized_diagnostics) == {
        "schema_version": "1.0.0",
        "record_type": "provider_diagnostic",
        "diagnostic": {
            "provider": "customer-gateway",
            "operation": "verify",
            "category": "provider_unavailable",
            "retryable": True,
            "retry_status": "not_retried",
            "suggested_action": "check provider status, then resume the run.",
            "endpoint_sha256": "a" * 64,
            "http_status": 503,
        },
    }

    def fail_diagnostic_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("private filesystem detail")

    failed_output = tmp_path / "failed-receipt-results.jsonl"
    monkeypatch.setattr(command_module, "write_provider_diagnostic", fail_diagnostic_write)
    failed_receipt_result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--allow-insecure-http",
            "--allow-environment-network",
            "--confirm-test-environment",
            "--output",
            str(failed_output),
            "--no-save-augmentations",
        ],
    )
    failed_receipt_output = " ".join(
        _ANSI_ESCAPE_PATTERN.sub("", failed_receipt_result.output).split()
    )
    assert failed_receipt_result.exit_code == 2
    assert "customer-gateway failed during verify" in failed_receipt_output
    assert "diagnostics could not be written (OSError)" in failed_receipt_output
    assert "private filesystem detail" not in failed_receipt_output


def test_provider_diagnostic_receipts_preserve_collisions_and_reject_symlinks(
    tmp_path: Path,
) -> None:
    output = tmp_path / "results.jsonl"
    diagnostic_error = ProviderDiagnosticError(
        ProviderDiagnostic(
            provider="customer-gateway",
            operation="render",
            category="rate_limit",
            retryable=True,
            suggested_action="wait, then resume the run.",
            endpoint_sha256="b" * 64,
            http_status=429,
        )
    )

    first = persistence_module.write_provider_diagnostic(output, diagnostic_error)
    second = persistence_module.write_provider_diagnostic(output, diagnostic_error)

    assert first == tmp_path / "results.jsonl.debug.json"
    assert second == tmp_path / "results.jsonl.debug.2.json"
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    assert stat.S_IMODE(first.stat().st_mode) == 0o600
    assert stat.S_IMODE(second.stat().st_mode) == 0o600

    if sys.platform == "win32":
        return
    symlink_output = tmp_path / "symlink-results.jsonl"
    protected_file = tmp_path / "protected.txt"
    protected_file.write_text("unchanged", encoding="utf-8")
    symlink_receipt = tmp_path / "symlink-results.jsonl.debug.json"
    symlink_receipt.symlink_to(protected_file)

    collision_receipt = persistence_module.write_provider_diagnostic(
        symlink_output, diagnostic_error
    )

    assert collision_receipt == tmp_path / "symlink-results.jsonl.debug.2.json"
    assert protected_file.read_text(encoding="utf-8") == "unchanged"


def test_execution_wires_redaction_into_records_pipeline_and_run_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    target_config = tmp_path / "target.json"
    policy_path = tmp_path / "redaction.json"
    state_path = tmp_path / "private" / "pseudonyms.json"
    secret = "customer-secret-value"
    key = "customer-key-with-at-least-thirty-two-bytes"
    _write_dataset(
        dataset,
        [{"id": "private", "input": f"Use {secret}", "output": {"private": secret}}],
    )
    _write_target_config(target_config, url="http://127.0.0.1:8765/execute")
    policy_path.write_text(
        json.dumps(
            {
                "version": 1,
                "rules": [
                    {
                        "name": "customer_secret",
                        "locations": ["input", "output"],
                        "selector": "$text",
                        "literal": secret,
                        "action": "pseudonymize",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("UL_DATASET_REDACTION_KEY", key)

    class FakeTarget:
        @classmethod
        def from_config(cls, *_args: object, **_kwargs: object) -> FakeTarget:
            return cls()

    async def fake_evaluate(
        records: tuple[InteractionRecord, ...],
        _operator_ids: tuple[str, ...],
        _settings: object,
        _target: object,
        output_stream: Any,
        *,
        repetitions: int,
        max_environment_api_calls: int,
        planned_target_calls: int,
        run_context: object,
        augmentation_ledger: object,
        saved_augmentations: object,
        redaction_engine: object,
        evaluator_preflight: object,
        trial_journal: object,
    ) -> tuple[object, ...]:
        del (
            repetitions,
            max_environment_api_calls,
            planned_target_calls,
            augmentation_ledger,
            saved_augmentations,
        )
        assert redaction_engine is not None
        assert evaluator_preflight == _evaluator_preflight()
        assert secret not in records[0].model_dump_json()
        serialized_context = cast(Any, run_context).model_dump_json()
        assert secret not in serialized_context
        assert key not in serialized_context
        assert '"matched_values":1' in serialized_context
        output_stream.write(serialized_context + "\n")
        return ()

    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", _settings)
    monkeypatch.setattr(command_module, "JsonHttpEnvironmentConnection", FakeTarget)
    monkeypatch.setattr(command_module, "evaluate_interaction_records", fake_evaluate)

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--allow-insecure-http",
            "--allow-environment-network",
            "--confirm-test-environment",
            "--output",
            str(output),
            "--redaction-policy",
            str(policy_path),
            "--redaction-state",
            str(state_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert secret not in output.read_text()
    assert key not in output.read_text()
    assert secret not in state_path.read_text()
    assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
    manifest = read_dataset_run_manifest(manifest_path(output))
    assert manifest.effective_command.redaction_policy_snapshot is not None
    assert manifest.effective_command.redaction_policy_source == str(policy_path.resolve())
    assert manifest.effective_command.redaction_state_path == str(state_path.resolve())
    assert manifest.effective_command.redaction_state_sha256 is not None


def test_target_config_runs_nested_request_and_response_against_loopback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received_requests: list[object] = []
    generation = 0
    committed_state: object = None

    class TargetHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            nonlocal committed_state, generation
            content_length = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(content_length))
            if self.path == "/reset":
                generation += 1
                committed_state = {"envelope": {"agent": {"actions": []}}}
                response_value: object = {
                    "environment_id": "test-environment",
                    "case_id": request["case_id"],
                    "generation": generation,
                    "clean": True,
                    "reset_session": True,
                    "reset_env": True,
                }
            elif self.path == "/execute":
                received_requests.append(request)
                committed_state = {
                    "envelope": {
                        "agent": {
                            "actions": [{"action": "transfer", "amount": 100, "recipient": "Alice"}]
                        }
                    }
                }
                response_value = {
                    "environment_id": "test-environment",
                    "case_id": request["case_id"],
                    "turn_id": request["turn_id"],
                    **cast(dict[str, object], committed_state),
                }
            elif self.path == "/snapshot":
                response_value = {
                    "environment_id": "test-environment",
                    "case_id": request["case_id"],
                    "turn_id": request["turn_id"],
                    "state": committed_state,
                }
            else:
                self.send_response(404)
                self.end_headers()
                return
            response = json.dumps(response_value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            pass

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
    except PermissionError:
        pytest.skip("the test environment does not allow binding a loopback server")
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    try:
        dataset = tmp_path / "interactions.jsonl"
        target_config = tmp_path / "target.json"
        output = tmp_path / "results.jsonl"
        _write_dataset(dataset, [_record()])
        _write_target_config(
            target_config,
            url=f"http://127.0.0.1:{server.server_port}/execute",
            request_json_template={
                "payload": {
                    "messages": [{"role": "user", "content": "{{input}}"}],
                }
            },
            response_json_pointer="/envelope/agent",
        )
        target_payload = json.loads(target_config.read_text(encoding="utf-8"))
        target_payload["snapshot"]["response_json_pointer"] = "/state/envelope/agent"
        target_config.write_text(json.dumps(target_payload), encoding="utf-8")
        observed_outputs: list[object] = []

        async def evaluate_once(
            records: tuple[Any, ...],
            operator_ids: tuple[str, ...],
            settings: object,
            target: Any,
            output_stream: Any,
            *,
            repetitions: int,
            max_environment_api_calls: int,
            planned_target_calls: int,
            run_context: object,
            augmentation_ledger: object,
            saved_augmentations: object,
            redaction_engine: object,
            evaluator_preflight: object,
            trial_journal: object,
        ) -> tuple[object, ...]:
            del (
                operator_ids,
                settings,
                repetitions,
                max_environment_api_calls,
                planned_target_calls,
                run_context,
                augmentation_ledger,
                saved_augmentations,
            )
            assert redaction_engine is None
            assert evaluator_preflight == _evaluator_preflight()
            async with target:
                case = evaluation_case_from_inputs(
                    case_id="ul-case-00000000000000000000000000000000",
                    raw_inputs=(records[0].raw_input,),
                    max_environment_api_calls=5,
                    timeout_seconds=30,
                )
                evidence = await target.execute(case)
                observed_outputs.append(evidence.turns[0].response)
            output_stream.write('{"saved":true}\n')
            return ()

        monkeypatch.setattr(
            command_module,
            "load_dataset_semantic_settings",
            _settings,
        )
        monkeypatch.setattr(command_module, "evaluate_interaction_records", evaluate_once)

        result = runner.invoke(
            root_app,
            [
                "dataset",
                "evaluate",
                str(dataset),
                "--environment-config",
                str(target_config),
                "--allow-insecure-http",
                "--allow-environment-network",
                "--confirm-test-environment",
                "--output",
                str(output),
            ],
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()

    assert result.exit_code == 0, result.output
    assert received_requests == [
        {
            "case_id": "ul-case-00000000000000000000000000000000",
            "turn_id": "ul-case-00000000000000000000000000000000:turn-1",
            "payload": {
                "messages": [{"role": "user", "content": "Transfer 100 to Alice."}],
            },
        }
    ]
    assert observed_outputs == [
        {"actions": [{"action": "transfer", "amount": 100, "recipient": "Alice"}]}
    ]
    output_lines = output.read_text(encoding="utf-8").splitlines()
    assert json.loads(output_lines[0])["record_type"] == "dataset_durable_run"
    assert output_lines[1] == '{"saved":true}'
