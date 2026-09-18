from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner
from ul import OpenRouterDecisionClient, OpenRouterDecisionSettings
from ul_cli.dataset.evaluation import preparation as preparation_module
from ul_cli.dataset.evaluation import runner as runner_module
from ul_cli.dataset_trial_journal import manifest_path, read_dataset_run_manifest
from ul_cli.local_target_resolution import resolve_local_target
from ul_cli.main import app

from ._factories import _settings
from ._files import _record, _write_dataset, _write_target_config
from .test_execution import _MaterialVarianceSemanticModel

runner = CliRunner()


def test_jev_dry_run_discloses_provider_without_constructing_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UL_DATASET_MATERIALITY_JUDGE", "jev")
    dataset = tmp_path / "data.jsonl"
    target = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target)
    monkeypatch.setattr(preparation_module, "load_dataset_semantic_settings", _settings)

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run created a decision client")

    monkeypatch.setattr(runner_module, "OpenRouterDecisionClient", unexpected)
    args = [
        "dataset",
        "evaluate",
        str(dataset),
        "--environment-config",
        str(target),
        "--dry-run",
    ]
    human = runner.invoke(app, args)
    assert human.exit_code == 0, human.output
    assert "OpenRouter/TypeSafe" in human.output
    result = runner.invoke(app, [*args, "--json"])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["materiality_judge"] == "jev"
    assert plan["materiality_model"] == "typesafe/jev-1.13"


def test_jev_cli_evidence_report_and_resume_bind_selected_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UL_DATASET_MATERIALITY_JUDGE", "jev")
    dataset = tmp_path / "data.jsonl"
    output = tmp_path / "results.jsonl"
    dataset.write_text(
        '{"id":"case-1","input":"Return status for ticket 42.",'
        '"output":{"action":"lookup","ticket":42}}\n'
    )
    (tmp_path / "customer_agent.py").write_text(
        "def run(value):\n"
        "    return {'action': 'lookup', 'ticket': 43 if value.startswith('check') else 42}\n"
    )
    monkeypatch.setattr(preparation_module, "load_dataset_semantic_settings", _settings)
    monkeypatch.setattr(
        runner_module, "create_semantic_model_deconstructor", _MaterialVarianceSemanticModel
    )
    requests: list[dict[str, object]] = []
    clients: list[httpx.AsyncClient] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        assert body["provider"]["only"] == ["typesafe"]
        return httpx.Response(
            200,
            json={
                "id": "test",
                "model": "typesafe/jev-1.13",
                "provider": "TypeSafe",
                "usage": {"input_tokens": 100, "output_tokens": 10},
                "answers": {
                    "0": {"type": "choice", "choice": "material_variance", "confidence": 0.99}
                },
            },
        )

    def decision_client(settings: OpenRouterDecisionSettings) -> OpenRouterDecisionClient:
        http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        clients.append(http)
        return OpenRouterDecisionClient(settings, client=http)

    monkeypatch.setattr(runner_module, "OpenRouterDecisionClient", decision_client)
    target = resolve_local_target("customer_agent:run")
    binding = ["--target", "customer_agent:run", "--confirm-target", target.confirmation_sha256]
    try:
        result = runner.invoke(
            app,
            [
                "dataset",
                "evaluate",
                str(dataset),
                *binding,
                "--confirm-test-environment",
                "--repetitions",
                "1",
                "--operator",
                "input.surface.rephrase",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 1, result.output
        saved = json.loads(output.read_text().splitlines()[1])
        assessment = saved["cases"][0]["material_variance"]
        assert assessment["decision"] == "material_variance"
        assert (
            assessment["evaluator_version_id"]
            == saved["run_context"]["semantic_settings"]["materiality_evaluator_version_id"]
        )
        assert len(requests) == 1
        manifest = read_dataset_run_manifest(manifest_path(output))
        assert manifest.effective_command.run_config.materiality_judge == "jev"
        report = runner.invoke(app, ["dataset", "report", str(output), "--all-findings"])
        assert report.exit_code == 0, report.output
        assert "consequential=1" in report.output
        resume = ["dataset", "evaluate", "--resume", str(output), *binding, "--dry-run"]
        monkeypatch.delenv("UL_DATASET_MATERIALITY_JUDGE")
        compatible = runner.invoke(app, resume)
        assert compatible.exit_code == 0, compatible.output
        assert "Outcome comparison provider: OpenRouter/TypeSafe" in compatible.output
        monkeypatch.setenv("UL_DATASET_MATERIALITY_JUDGE", "llm")
        incompatible = runner.invoke(app, resume)
        assert incompatible.exit_code == 2
        assert "incompatible" in incompatible.output
        monkeypatch.delenv("UL_DATASET_MATERIALITY_JUDGE")
        monkeypatch.setenv("UL_DECISION_MODEL", "typesafe/jev-1.14")
        changed_model = runner.invoke(app, resume)
        assert changed_model.exit_code == 2
        assert "incompatible" in changed_model.output
        assert len(requests) == 1
    finally:
        for client in clients:
            asyncio.run(client.aclose())


def test_jev_requires_its_own_key_before_target_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPEN_ROUTER_API_KEY", raising=False)
    monkeypatch.setenv("UL_DATASET_MATERIALITY_JUDGE", "jev")
    dataset = tmp_path / "data.jsonl"
    target = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target)
    monkeypatch.setattr(
        preparation_module,
        "load_dataset_semantic_settings",
        lambda: _settings(
            semantic_provider_type="openai-compatible",
            semantic_provider_id="customer",
            semantic_base_url="https://customer.example/v1",
            upstream_provider=None,
            api_key_environment_variable="UL_DATASET_OPENAI_API_KEY",
        ),
    )

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("missing key reached model execution")

    monkeypatch.setattr(runner_module, "create_semantic_model_deconstructor", unexpected)
    result = runner.invoke(
        app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target),
            "--allow-environment-network",
            "--confirm-test-environment",
            "--output",
            str(tmp_path / "result.jsonl"),
        ],
    )
    assert result.exit_code == 2
    assert "OPEN_ROUTER_API_KEY" in result.output
    assert "test-key" not in result.output


def test_normal_cli_does_not_expose_judge_selection() -> None:
    result = runner.invoke(app, ["dataset", "evaluate", "--help"])
    assert result.exit_code == 0
    assert "materiality-judge" not in result.output
    assert "Jev" not in result.output


def test_advanced_comparison_configuration_loads_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ul_cli.dataset_materiality import DatasetOutcomeComparisonSettings

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UL_DATASET_MATERIALITY_JUDGE", raising=False)
    assert DatasetOutcomeComparisonSettings().materiality_judge is None
    (tmp_path / ".env").write_text("UL_DATASET_MATERIALITY_JUDGE=jev\n")
    assert DatasetOutcomeComparisonSettings().materiality_judge == "jev"
