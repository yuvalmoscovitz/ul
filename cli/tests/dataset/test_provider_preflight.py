from __future__ import annotations

import re
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner
from ul import (
    EvaluatorModelCompatibilityError,
    EvaluatorModelPreflight,
)
from ul_cli.dataset.evaluation import execution as execution_module
from ul_cli.dataset.evaluation import runner as runner_module
from ul_cli.dataset.evidence.persistence import durable_evidence_marker_manifest_sha256
from ul_cli.main import app as root_app

from ._factories import (
    _evaluator_preflight,
)
from ._files import (
    _record,
    _write_dataset,
    _write_target_config,
)

runner = CliRunner()
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def test_openai_compatible_dry_run_reports_provider_without_making_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record()])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UL_DATASET_SEMANTIC_PROVIDER", "openai-compatible")
    monkeypatch.setenv("UL_DATASET_OPENAI_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("UL_DATASET_MODEL", "local-semantic-model")

    def unexpected_deconstructor(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run constructed a semantic model client")

    monkeypatch.setattr(
        runner_module, "create_semantic_model_deconstructor", unexpected_deconstructor
    )
    result = runner.invoke(root_app, ["dataset", "evaluate", str(dataset), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Semantic provider: openai-compatible (endpoint sha256: cbff7260a780)" in result.output
    assert "http://localhost:8000/v1" not in result.output
    assert "No model or environment API requests sent." in result.output


def test_openai_compatible_cli_hides_rejected_base_url_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record()])
    credential_sentinel = "credential-sentinel"
    query_sentinel = "query-sentinel"
    rejected_url = (
        f"https://user:{credential_sentinel}@models.example.test/v1?token={query_sentinel}"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UL_DATASET_SEMANTIC_PROVIDER", "openai-compatible")
    monkeypatch.setenv("UL_DATASET_OPENAI_BASE_URL", rejected_url)
    monkeypatch.setenv("UL_DATASET_MODEL", "customer/model")

    result = runner.invoke(root_app, ["dataset", "evaluate", str(dataset), "--dry-run"])

    normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", result.output).split())
    assert result.exit_code != 0
    assert "UL_DATASET_OPENAI_BASE_URL must be an HTTPS API root" in normalized_output
    assert credential_sentinel not in normalized_output
    assert query_sentinel not in normalized_output
    assert rejected_url not in normalized_output


def test_openai_compatible_execution_allows_an_unauthenticated_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    _write_dataset(dataset, [_record()])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UL_DATASET_SEMANTIC_PROVIDER", "openai-compatible")
    monkeypatch.setenv("UL_DATASET_OPENAI_BASE_URL", "https://models.example.test/v1")
    monkeypatch.setenv("UL_DATASET_MODEL", "customer/model")
    monkeypatch.setenv("UL_LIVE", "true")
    monkeypatch.delenv("UL_DATASET_OPENAI_API_KEY", raising=False)
    target_config = tmp_path / "target.json"
    _write_target_config(target_config)

    class FakeTarget:
        @classmethod
        def from_config(cls, *args: object, **kwargs: object) -> FakeTarget:
            return cls()

        async def aclose(self) -> None:
            return None

    async def fake_evaluate(*args: object, **kwargs: object) -> tuple[object, ...]:
        return ()

    monkeypatch.setattr(execution_module, "JsonHttpEnvironmentConnection", FakeTarget)
    monkeypatch.setattr(execution_module, "evaluate_interaction_records", fake_evaluate)

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

    assert result.exit_code == 0, result.output
    assert output.exists()


def test_evaluator_preflight_failure_surfaces_capability_and_safe_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config)
    monkeypatch.setenv("UL_LIVE", "true")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")

    class FakeTarget:
        @classmethod
        def from_config(cls, *args: object, **kwargs: object) -> FakeTarget:
            return cls()

        async def aclose(self) -> None:
            return None

    async def fail_preflight(*args: object, **kwargs: object) -> EvaluatorModelPreflight:
        raise EvaluatorModelCompatibilityError(
            "evaluator model is incompatible with required seed capability; "
            "choose another configured evaluator model or verify the configured route and retry"
        )

    monkeypatch.setattr(execution_module, "preflight_evaluator", fail_preflight)
    monkeypatch.setattr(execution_module, "JsonHttpEnvironmentConnection", FakeTarget)

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

    assert result.exit_code == 2
    assert "required seed capability" in result.output
    assert "choose another configured evaluator model" in result.output
    assert "before campaign execution" in result.output
    assert output.exists()
    assert durable_evidence_marker_manifest_sha256(output) is not None
    assert len(output.read_bytes().splitlines()) == 1
    assert (tmp_path / "results.jsonl.manifest.json").exists()
    assert (tmp_path / "results.jsonl.trials.jsonl").exists()
    assert not (tmp_path / "results.augmentations.jsonl").exists()
    assert not (tmp_path / "results.jsonl.preflight.json").exists()


def test_local_environment_gate_fails_before_evaluator_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config)
    monkeypatch.setenv("UL_LIVE", "true")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    preflight_calls = 0

    async def unexpected_preflight(*args: object, **kwargs: object) -> EvaluatorModelPreflight:
        nonlocal preflight_calls
        preflight_calls += 1
        raise AssertionError("local validation must run before evaluator preflight")

    monkeypatch.setattr(execution_module, "preflight_evaluator", unexpected_preflight)

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--confirm-test-environment",
            "--output",
            str(output),
        ],
        terminal_width=240,
    )

    assert result.exit_code == 2
    assert preflight_calls == 0
    assert not output.exists()
    assert not (tmp_path / "results.jsonl.preflight.json").exists()


def test_evaluator_preflight_receipt_survives_later_semantic_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    receipt = tmp_path / "results.jsonl.preflight.json"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config)
    monkeypatch.setenv("UL_LIVE", "true")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    expected_preflight = _evaluator_preflight()
    preflight_calls = 0

    async def successful_preflight(settings: object) -> EvaluatorModelPreflight:
        nonlocal preflight_calls
        del settings
        preflight_calls += 1
        return expected_preflight

    class FakeTarget:
        @classmethod
        def from_config(cls, *args: object, **kwargs: object) -> FakeTarget:
            return cls()

        async def aclose(self) -> None:
            return None

    async def fail_after_preflight(*args: object, **kwargs: object) -> tuple[object, ...]:
        assert kwargs["evaluator_preflight"] is expected_preflight
        raise ValueError("later semantic failure")

    monkeypatch.setattr(execution_module, "preflight_evaluator", successful_preflight)
    monkeypatch.setattr(execution_module, "JsonHttpEnvironmentConnection", FakeTarget)
    monkeypatch.setattr(execution_module, "evaluate_interaction_records", fail_after_preflight)

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

    assert result.exit_code == 2
    assert preflight_calls == 1
    assert EvaluatorModelPreflight.model_validate_json(receipt.read_text()) == expected_preflight
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600
