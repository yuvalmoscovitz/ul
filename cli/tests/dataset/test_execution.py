# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import re
import stat
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner
from ul import (
    DatasetSemanticPreparationError,
    EvaluatorDecision,
    EvaluatorEvidence,
    InteractionRecord,
    JsonHttpEnvironmentConfig,
    ProviderDiagnostic,
    ProviderDiagnosticError,
)
from ul.environment import evaluation_case_from_inputs
from ul.evaluators import evaluator_judge_version_from_llm_config
from ul.llm import LLMClient, llm_client_config_from_dataset_settings
from ul_cli import dataset_augmentation_ledger as augmentation_ledger_module
from ul_cli import progress_action as progress_action_module
from ul_cli.dataset.evaluation import command as command_module
from ul_cli.dataset.evaluation import runner as runner_module
from ul_cli.dataset.evidence import customer as customer_module
from ul_cli.dataset.evidence import persistence as persistence_module
from ul_cli.dataset_trial_journal import (
    journal_path,
    manifest_path,
    open_dataset_trial_journal,
    read_dataset_run_manifest,
)
from ul_cli.http_target_resolution import (
    create_isolated_response_target_config,
    http_target_evidence_receipt,
    resolve_http_target,
)
from ul_cli.local_target_resolution import resolve_local_target
from ul_cli.main import app as root_app
from ul_core.dataset import (
    EvidenceReference,
    ObservedOutcome,
    RenderedUserInput,
    RequestUnit,
    SemanticEquivalenceAssessment,
    SemanticFrame,
    UserInputRecord,
)

from ._factories import (
    _evaluation_result,
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


def test_execution_reuses_complete_augmentation_input_without_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation_result = _evaluation_result("interaction-1", has_review_finding=True)
    dataset = tmp_path / "interactions.jsonl"
    augmentation_input = tmp_path / "accepted.augmentations.jsonl"
    output = tmp_path / "fresh-target-results.jsonl"
    target_config = tmp_path / "fresh-target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config, url="http://127.0.0.1:8765/execute")
    settings = _settings()
    generation_context = augmentation_ledger_module.create_dataset_augmentation_generation_context(
        selected_records=(evaluation_result.source,),
        operators=(("input.surface.rephrase", "1.0.0"),),
        semantic_settings=(
            augmentation_ledger_module.dataset_augmentation_ledger_semantic_settings(settings)
        ),
    )
    with augmentation_ledger_module.create_private_augmentation_ledger(
        augmentation_input,
        generation_context=generation_context,
        selected_records=(evaluation_result.source,),
    ) as ledger:
        ledger.append(
            source=evaluation_result.source,
            augmentation=evaluation_result.augmentation,
        )
    accepted_bytes = augmentation_input.read_bytes()
    captured_saved_augmentations: dict[str, object] = {}

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
        **arguments: object,
    ) -> tuple[object, ...]:
        assert records == (evaluation_result.source,)
        assert arguments["augmentation_ledger"] is None
        captured_saved_augmentations.update(
            cast(dict[str, object], arguments["saved_augmentations"])
        )
        output_stream.write('{"saved":true}\n')
        output_stream.flush()
        return ()

    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", lambda: settings)
    monkeypatch.setattr(command_module, "JsonHttpEnvironmentConnection", FakeTarget)
    monkeypatch.setattr(command_module, "evaluate_interaction_records", fake_evaluate)

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--operator",
            "input.surface.rephrase",
            "--environment-config",
            str(target_config),
            "--allow-insecure-http",
            "--allow-environment-network",
            "--confirm-test-environment",
            "--repetitions",
            "1",
            "--augmentations-input",
            str(augmentation_input),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured_saved_augmentations == {"interaction-1": evaluation_result.augmentation}
    assert augmentation_input.read_bytes() == accepted_bytes
    assert not (tmp_path / "fresh-target-results.augmentations.jsonl").exists()
    manifest = read_dataset_run_manifest(manifest_path(output))
    assert manifest.effective_command.augmentations_input_path == str(augmentation_input.resolve())
    assert (
        manifest.effective_command.augmentations_input_sha256
        == hashlib.sha256(accepted_bytes).hexdigest()
    )


class _LocalEvaluationSemanticModel:
    def __init__(self, semantic_settings: Any | None = None) -> None:
        self.llm_client = LLMClient(
            llm_client_config_from_dataset_settings(semantic_settings or _settings())
        )

    async def __aenter__(self) -> _LocalEvaluationSemanticModel:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.llm_client.aclose()

    def reuse_preflight(self, result: object) -> None:
        del result

    async def deconstruct(
        self,
        record: InteractionRecord | UserInputRecord,
        reference_frame: SemanticFrame | None = None,
    ) -> SemanticFrame:
        if not isinstance(record, InteractionRecord):
            if reference_frame is not None:
                return reference_frame.model_copy(update={"interaction_id": record.id})
            return SemanticFrame(
                interaction_id=record.id,
                request_units=(
                    RequestUnit(
                        id="lookup-request",
                        evidence=(
                            EvidenceReference(
                                source="input",
                                json_pointer="/raw_input",
                                text_quote=None,
                            ),
                        ),
                        confidence=1,
                        status="explicit",
                        mode="ask",
                        predicate="lookup",
                    ),
                ),
                extractor_version="local-evaluation-test",
            )
        return SemanticFrame(
            interaction_id=record.id,
            request_units=(
                RequestUnit(
                    id="lookup-request",
                    evidence=(
                        EvidenceReference(
                            source="input",
                            json_pointer="/raw_input",
                            text_quote=None,
                        ),
                    ),
                    confidence=1,
                    status="explicit",
                    mode="ask",
                    predicate="lookup",
                ),
            ),
            outcomes=(
                ObservedOutcome(
                    id="lookup-outcome",
                    evidence=(
                        EvidenceReference(
                            source="output",
                            json_pointer="/raw_observed_output/action",
                            text_quote=None,
                        ),
                        EvidenceReference(
                            source="output",
                            json_pointer="/raw_observed_output/ticket",
                            text_quote=None,
                        ),
                    ),
                    confidence=1,
                    status="observed",
                    request_unit_ids=("lookup-request",),
                    position=0,
                    kind="action",
                    predicate="lookup",
                    fields={"ticket": 42},
                ),
            ),
            extractor_version="local-evaluation-test",
        )

    async def render(
        self,
        raw_input: str,
        instruction: str,
        *,
        allow_temporary_value: bool = False,
    ) -> RenderedUserInput:
        del instruction, allow_temporary_value
        return RenderedUserInput(text=raw_input)

    async def verify(
        self,
        source_input: str,
        candidate_input: str,
    ) -> SemanticEquivalenceAssessment:
        del source_input, candidate_input
        return SemanticEquivalenceAssessment(
            verdict="equivalent",
            explanation="The requests are equivalent.",
            verifier_version="local-evaluation-test",
        )


class _WrappedActionFieldSemanticModel(_LocalEvaluationSemanticModel):
    async def deconstruct(
        self,
        record: InteractionRecord | UserInputRecord,
        reference_frame: SemanticFrame | None = None,
    ) -> SemanticFrame:
        frame = await super().deconstruct(record, reference_frame)
        if not isinstance(record, InteractionRecord):
            return frame
        wrapped_outcomes = tuple(
            outcome.model_copy(
                update={
                    "evidence": (
                        EvidenceReference(
                            source="output",
                            json_pointer="/raw_observed_output/actions/0",
                            text_quote=None,
                        ),
                    ),
                    "fields": {
                        name: {
                            "value": value,
                            "evidence": [
                                {
                                    "source": "output",
                                    "json_pointer": f"/raw_observed_output/actions/0/{name}",
                                }
                            ],
                        }
                        for name, value in {
                            "ticket": 42,
                            "body.intent": "order",
                            "body.note.text": "Return status for ticket 42.",
                            "authoredOn": "stale-reference-only-value",
                        }.items()
                    },
                }
            )
            for outcome in frame.outcomes
        )
        return frame.model_copy(update={"outcomes": wrapped_outcomes})


class _MaterialVarianceSemanticModel(_LocalEvaluationSemanticModel):
    async def render(
        self,
        raw_input: str,
        instruction: str,
        *,
        allow_temporary_value: bool = False,
    ) -> RenderedUserInput:
        del raw_input, instruction, allow_temporary_value
        return RenderedUserInput(text="Please return status for ticket 42.")

    async def deconstruct(
        self,
        record: InteractionRecord | UserInputRecord,
        reference_frame: SemanticFrame | None = None,
    ) -> SemanticFrame:
        frame = await super().deconstruct(record, reference_frame)
        if not isinstance(record, InteractionRecord):
            return frame
        raw_output = cast(dict[str, Any], record.raw_observed_output)
        ticket = cast(int, raw_output["ticket"])
        return frame.model_copy(
            update={
                "outcomes": tuple(
                    outcome.model_copy(update={"fields": {"ticket": ticket}})
                    for outcome in frame.outcomes
                )
            }
        )


class _ResponseMaterialVarianceSemanticModel(_MaterialVarianceSemanticModel):
    async def deconstruct(
        self,
        record: InteractionRecord | UserInputRecord,
        reference_frame: SemanticFrame | None = None,
    ) -> SemanticFrame:
        frame = await _LocalEvaluationSemanticModel.deconstruct(self, record, reference_frame)
        if not isinstance(record, InteractionRecord):
            return frame
        return frame.model_copy(
            update={
                "outcomes": (
                    ObservedOutcome(
                        id="returned-response",
                        evidence=(
                            EvidenceReference(
                                source="output",
                                json_pointer="/raw_observed_output",
                                text_quote=None,
                            ),
                        ),
                        confidence=1,
                        status="observed",
                        position=0,
                        kind="answer",
                        predicate="returned_response",
                        fields={"value": record.raw_observed_output},
                    ),
                )
            }
        )


class _MaterialVarianceJudge:
    label = "material_variance:grounded_argument_changed"
    score = 1
    expected_token_parameter = "max_tokens"

    def __init__(self, *, llm_client: LLMClient) -> None:
        config = llm_client.config
        assert config.role_config("materiality").token_parameter == self.expected_token_parameter
        if config.provider_type == "openrouter":
            assert config.upstream_provider == "test-provider"
        else:
            assert config.upstream_provider is None
        self.version = evaluator_judge_version_from_llm_config(config)

    async def __aenter__(self) -> _MaterialVarianceJudge:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    async def evaluate(self, request: object) -> EvaluatorDecision:
        del request
        return EvaluatorDecision(
            score=self.score,
            label=self.label,
            explanation="Changed ticket.",
            evidence=(
                EvaluatorEvidence(
                    source="judge_payload",
                    json_pointer="/payload/answer/findings/0/baseline_effects/0",
                    description="baseline",
                ),
                EvaluatorEvidence(
                    source="judge_payload",
                    json_pointer="/payload/answer/findings/0/variation_effects/0",
                    description="variation",
                ),
            ),
        )


@pytest.fixture(autouse=True)
def isolate_progress_action_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        progress_action_module,
        "_action_receipt_directory",
        lambda: tmp_path / "action-state",
    )


def test_safe_boundary_pause_flushes_a_resumable_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config)

    class FakeTarget:
        @classmethod
        def from_config(cls, *_args: object, **_kwargs: object) -> FakeTarget:
            return cls()

        async def aclose(self) -> None:
            pass

    async def successful_preflight(_settings: object) -> object:
        return _evaluator_preflight()

    create_runtime = command_module.create_campaign_progress_runtime

    def create_paused_runtime(**arguments: object) -> object:
        runtime = create_runtime(**arguments)
        runtime.control.request_pause()
        return runtime

    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", _settings)
    monkeypatch.setattr(command_module, "JsonHttpEnvironmentConnection", FakeTarget)
    monkeypatch.setattr(command_module, "preflight_evaluator", successful_preflight)
    monkeypatch.setattr(
        command_module,
        "create_campaign_progress_runtime",
        create_paused_runtime,
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

    assert result.exit_code == 130, result.output
    assert result.output.count(" next_action=") == 1
    assert "next_action=resume" in result.output
    assert re.search(
        r'next_argv=\["ul","action","[0-9a-f]{64}"\]',
        result.output,
    )
    assert str(output) not in result.output.split("next_action=", 1)[1]
    manifest = read_dataset_run_manifest(manifest_path(output))
    journal = open_dataset_trial_journal(journal_path(output), manifest)
    journal.close()
    resume_check = runner.invoke(
        root_app,
        ["dataset", "evaluate", "--resume", str(output), "--dry-run"],
    )
    assert resume_check.exit_code == 0, resume_check.output


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


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("wrapped_action_fields", [False, True])
def test_full_dataset_evaluation_runs_local_callable_through_worker_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
    wrapped_action_fields: bool,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    canonical_action = {
        "action": "lookup",
        "ticket": 42,
        "body.intent": "order",
        "body.note.text": "Return status for ticket 42.",
    }
    recorded_output = (
        {"actions": [canonical_action]}
        if wrapped_action_fields
        else {"action": "lookup", "ticket": 42}
    )
    dataset.write_text(
        json.dumps(
            {
                "id": "case-1",
                "input": "Return status for ticket 42.",
                "output": recorded_output,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    function_prefix = "async " if asynchronous else ""
    await_statement = "    await asyncio.sleep(0)\n" if asynchronous else ""
    returned_output = (
        repr({"actions": [canonical_action]})
        if wrapped_action_fields
        else "{'action': 'lookup', 'ticket': 42, 'received': value}"
    )
    (tmp_path / "customer_agent.py").write_text(
        "import asyncio\n\n"
        f"{function_prefix}def run(value):\n"
        f"{await_statement}"
        f"    return {returned_output}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UL_LIVE", "true")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    semantic_model = (
        _WrappedActionFieldSemanticModel()
        if wrapped_action_fields
        else _LocalEvaluationSemanticModel()
    )

    async def successful_preflight(_settings: object) -> object:
        return _evaluator_preflight()

    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", _settings)
    monkeypatch.setattr(command_module, "preflight_evaluator", successful_preflight)
    monkeypatch.setattr(
        runner_module,
        "create_semantic_model_deconstructor",
        lambda _settings: semantic_model,
    )
    target = resolve_local_target("customer_agent:run")

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target",
            "customer_agent:run",
            "--confirm-target",
            target.confirmation_sha256,
            "--confirm-test-environment",
            "--repetitions",
            "2",
            "--target-timeout-seconds",
            "90",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    saved = json.loads(output.read_text(encoding="utf-8").splitlines()[1])
    assert saved["run_context"]["target"]["kind"] == "probe_target"
    assert saved["run_context"]["fixture"]["status"] == "not_required"
    assert saved["run_context"]["target"]["receipt"]["supports_state_observation"] is False
    assert saved["run_context"]["target_timeout_seconds"] == 90.0
    manifest = read_dataset_run_manifest(manifest_path(output))
    assert manifest.effective_command.run_config.target.trial_timeout_seconds == 90.0
    assert len(saved["technical_details"]["baseline"]["trial_set"]["trials"]) == 2
    assert saved["technical_details"]["baseline"]["verdict"] == "no_divergence"
    assert saved["technical_details"]["cases"][0]["verdict"] == "no_divergence"
    observed_fields = saved["technical_details"]["baseline"]["trial_set"]["trials"][0][
        "observed_frame"
    ]["outcomes"][0]["fields"]
    assert observed_fields["ticket"] == 42
    if wrapped_action_fields:
        assert observed_fields == {
            "ticket": 42,
            "body.intent": "order",
            "body.note.text": "Return status for ticket 42.",
            "authoredOn": {
                "value": "stale-reference-only-value",
                "evidence": [
                    {
                        "source": "output",
                        "json_pointer": "/raw_observed_output/actions/0/authoredOn",
                    }
                ],
            },
        }
    final_response = saved["technical_details"]["baseline"]["trial_set"]["trials"][0][
        "execution_evidence"
    ]["final_response"]
    assert (
        final_response["actions"][0]["ticket"]
        if wrapped_action_fields
        else final_response["ticket"]
    ) == 42
    if not asynchronous:
        missing_target_resume = runner.invoke(
            root_app,
            ["dataset", "evaluate", "--resume", str(output), "--dry-run"],
        )
        assert missing_target_resume.exit_code == 2
        normalized_resume_error = " ".join(
            _ANSI_ESCAPE_PATTERN.sub("", missing_target_resume.output).split()
        )
        assert "local target resume requires the same explicit --target" in normalized_resume_error
        resumed = runner.invoke(
            root_app,
            [
                "dataset",
                "evaluate",
                "--resume",
                str(output),
                "--target",
                "customer_agent:run",
                "--confirm-target",
                target.confirmation_sha256,
                "--dry-run",
            ],
        )
        assert resumed.exit_code == 0, resumed.output
        assert "Resume compatible: 1 complete interaction(s) skipped; 0 remaining" in (
            resumed.output
        )
        assert "Target trial timeout: 90 seconds" in resumed.output
        incompatible_resume = runner.invoke(
            root_app,
            [
                "dataset",
                "evaluate",
                "--resume",
                str(output),
                "--target",
                "customer_agent:run",
                "--confirm-target",
                target.confirmation_sha256,
                "--target-timeout-seconds",
                "91",
                "--dry-run",
            ],
        )
        assert incompatible_resume.exit_code == 2
        normalized_incompatible_error = " ".join(
            _ANSI_ESCAPE_PATTERN.sub("", incompatible_resume.output).split()
        )
        assert "incompatible with the current evaluation plan" in normalized_incompatible_error


@pytest.mark.parametrize(
    (
        "label",
        "score",
        "decision",
        "reason_code",
        "expected_exit_code",
        "settings_overrides",
    ),
    (
        (
            "material_variance:grounded_argument_changed",
            1,
            "material_variance",
            "grounded_argument_changed",
            1,
            {},
        ),
        (
            "operationally_equivalent:same_real_world_effect",
            0,
            "operationally_equivalent",
            "same_real_world_effect",
            0,
            {},
        ),
        (
            "insufficient_evidence:missing_comparison_evidence",
            0.5,
            "insufficient_evidence",
            "missing_comparison_evidence",
            2,
            {},
        ),
        (
            "material_variance:grounded_argument_changed",
            1,
            "material_variance",
            "grounded_argument_changed",
            1,
            {
                "semantic_provider_id": "generic-test",
                "semantic_provider_type": "openai-compatible",
                "semantic_base_url": "https://evaluator.example/v1",
                "semantic_endpoint_sha256": "f" * 64,
                "api_key_required": False,
                "api_key_environment_variable": "UL_DATASET_OPENAI_API_KEY",
            },
        ),
    ),
)
def test_public_cli_persists_and_applies_automatic_materiality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    score: float,
    decision: str,
    reason_code: str,
    expected_exit_code: int,
    settings_overrides: dict[str, object],
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    dataset.write_text(
        '{"id":"case-1","input":"Return status for ticket 42.",'
        '"output":{"action":"lookup","ticket":42}}\n',
        encoding="utf-8",
    )
    (tmp_path / "customer_agent.py").write_text(
        "def run(value):\n"
        "    ticket = 43 if value.startswith('Please') else 42\n"
        "    return {'action': 'lookup', 'ticket': ticket}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UL_LIVE", "true")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")

    async def successful_preflight(_settings: object) -> object:
        return _evaluator_preflight()

    monkeypatch.setattr(
        command_module,
        "load_dataset_semantic_settings",
        lambda: _settings(**settings_overrides),
    )
    monkeypatch.setattr(command_module, "preflight_evaluator", successful_preflight)
    monkeypatch.setattr(
        runner_module,
        "create_semantic_model_deconstructor",
        lambda semantic_settings: _MaterialVarianceSemanticModel(semantic_settings),
    )
    monkeypatch.setattr(
        runner_module,
        "OpenAICompatibleEvaluatorJudge",
        _MaterialVarianceJudge,
    )
    monkeypatch.setattr(_MaterialVarianceJudge, "label", label)
    monkeypatch.setattr(_MaterialVarianceJudge, "score", score)
    target = resolve_local_target("customer_agent:run")

    evaluated = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--operator",
            "input.surface.rephrase",
            "--target",
            "customer_agent:run",
            "--confirm-target",
            target.confirmation_sha256,
            "--confirm-test-environment",
            "--repetitions",
            "1",
            "--output",
            str(output),
        ],
    )

    assert evaluated.exit_code == expected_exit_code, evaluated.output
    saved = json.loads(output.read_text(encoding="utf-8").splitlines()[1])
    comparison = saved["cases"][0]
    assert comparison["material_variance"]["decision"] == decision
    assert comparison["material_variance"]["reason_code"] == reason_code
    assert saved["technical_details"]["semantic_calls"]["actual_calls"] == 1
    normalized_output = " ".join(evaluated.output.split())
    assert "Next: ul dataset report" in normalized_output

    report = runner.invoke(root_app, ["dataset", "report", str(output), "--all-findings"])

    assert report.exit_code == 0, report.output
    report_decision = {
        "material_variance": "consequential",
        "operationally_equivalent": "equivalent",
        "insufficient_evidence": "inconclusive",
    }[decision]
    assert f"{report_decision}=1" in report.output
    assert f"Reason: {reason_code.replace('_', ' ')}" in report.output
    finding_output = output.with_name(f"{output.name}.findings.jsonl")
    if decision == "material_variance":
        assert finding_output.stat().st_size > 0
        assert "Actionable finding export: ul report" in normalized_output
    else:
        assert finding_output.stat().st_size == 0
        assert "Actionable finding export: ul report" not in normalized_output


def test_public_cli_does_not_let_judge_suppress_removed_committed_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    dataset.write_text(
        '{"id":"case-1","input":"Create record 42.",'
        '"output":{"answer":[],"actions":[{"action":"CREATE_RECORD","id":42}]}}\n',
        encoding="utf-8",
    )
    (tmp_path / "customer_agent.py").write_text(
        "def run(value):\n"
        "    actions = [] if value.startswith('Please') else "
        "[{'action': 'CREATE_RECORD', 'id': 42}]\n"
        "    return {'answer': [], 'actions': actions}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UL_LIVE", "true")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")

    async def successful_preflight(_settings: object) -> object:
        return _evaluator_preflight()

    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", _settings)
    monkeypatch.setattr(command_module, "preflight_evaluator", successful_preflight)
    monkeypatch.setattr(
        runner_module,
        "create_semantic_model_deconstructor",
        lambda _settings: _ResponseMaterialVarianceSemanticModel(),
    )
    monkeypatch.setattr(runner_module, "OpenAICompatibleEvaluatorJudge", _MaterialVarianceJudge)
    monkeypatch.setattr(
        _MaterialVarianceJudge,
        "label",
        "operationally_equivalent:same_real_world_effect",
    )
    monkeypatch.setattr(_MaterialVarianceJudge, "score", 0)
    target = resolve_local_target("customer_agent:run")

    evaluated = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--operator",
            "input.surface.rephrase",
            "--target",
            "customer_agent:run",
            "--confirm-target",
            target.confirmation_sha256,
            "--confirm-test-environment",
            "--repetitions",
            "1",
            "--output",
            str(output),
        ],
    )

    assert evaluated.exit_code == 1, evaluated.output
    saved = json.loads(output.read_text(encoding="utf-8").splitlines()[1])
    comparison = saved["cases"][0]
    assert comparison["material_variance"]["decision"] == "material_variance"
    assert comparison["material_variance"]["reason_code"] == "action_removed"
    assert saved["technical_details"]["semantic_calls"]["actual_calls"] == 0

    report = runner.invoke(root_app, ["dataset", "report", str(output)])

    assert report.exit_code == 0, report.output
    assert "ACTION REQUIRED" in report.output
    assert "consequential=1" in report.output


def test_dataset_evaluation_allows_http_target_response_after_thirty_seconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_count = 0

    class SlowTargetHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            nonlocal request_count
            content_length = int(self.headers["Content-Length"])
            json.loads(self.rfile.read(content_length))
            request_count += 1
            if request_count == 1:
                time.sleep(31)
            response = json.dumps({"result": {"action": "lookup", "ticket": 42}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            pass

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), SlowTargetHandler)
    except PermissionError:
        pytest.skip("the test environment does not allow binding a loopback server")
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    dataset.write_text(
        '{"id":"case-1","input":"Return status for ticket 42.",'
        '"output":{"action":"lookup","ticket":42}}\n',
        encoding="utf-8",
    )
    url = f"http://127.0.0.1:{server.server_port}/invoke"
    direct_options = {
        "allow_insecure_http": True,
        "request_json_template": '{"input":"{{input}}"}',
        "response_json_pointer": "/result",
        "request_isolation_attested": True,
        "safe_test_target_attested": True,
    }
    resolved_target = resolve_http_target(url, **direct_options)
    semantic_model = _LocalEvaluationSemanticModel()

    async def successful_preflight(_settings: object) -> object:
        return _evaluator_preflight()

    monkeypatch.setenv("UL_LIVE", "true")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", _settings)
    monkeypatch.setattr(command_module, "preflight_evaluator", successful_preflight)
    monkeypatch.setattr(
        runner_module,
        "create_semantic_model_deconstructor",
        lambda _settings: semantic_model,
    )
    try:
        result = runner.invoke(
            root_app,
            [
                "dataset",
                "evaluate",
                str(dataset),
                "--target",
                url,
                "--request-json-template",
                cast(str, direct_options["request_json_template"]),
                "--response-json-pointer",
                "/result",
                "--confirm-request-isolation",
                "--confirm-safe-test-target",
                "--confirm-target",
                resolved_target.confirmation_sha256,
                "--allow-insecure-http",
                "--allow-environment-network",
                "--confirm-test-environment",
                "--repetitions",
                "1",
                "--target-timeout-seconds",
                "35",
                "--output",
                str(output),
            ],
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()

    assert result.exit_code == 0, result.output
    assert request_count == 2
    saved = json.loads(output.read_text(encoding="utf-8").splitlines()[1])
    baseline_trial = saved["technical_details"]["baseline"]["trial_set"]["trials"][0]
    variation_trial = saved["technical_details"]["cases"][0]["trial_set"]["trials"][0]
    assert baseline_trial["execution_evidence"]["final_response"]["ticket"] == 42
    assert variation_trial["execution_evidence"]["final_response"]["ticket"] == 42


def test_declared_projection_compares_raw_recorded_tool_calls_through_public_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    dataset.write_text(
        '{"id":"case-1","input":"Return status for ticket 42.",'
        '"output":{"tool_calls":[{"name":"lookup",'
        '"arguments":"{\\"ticket\\":42}"}]}}\n',
        encoding="utf-8",
    )
    (tmp_path / "customer_agent.py").write_text(
        "import json\n\n"
        "def run(value):\n"
        "    return {'tool_calls': [{'name': 'lookup', "
        "'arguments': json.dumps({'ticket': 42})}]}\n",
        encoding="utf-8",
    )
    target_config = tmp_path / "target.json"
    target_config.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "python_callable",
                "target_id": "projected-tool-agent",
                "working_directory": str(tmp_path),
                "interpreter": str(Path(sys.executable).resolve()),
                "target": "customer_agent:run",
                "outcome": {
                    "compose": {
                        "fields": {"action": "/tool_calls/0/name"},
                        "spread": {
                            "selector": "/tool_calls/0/arguments",
                            "decode": "json_string",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UL_LIVE", "true")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")

    class ProjectedResponseSemanticModel(_LocalEvaluationSemanticModel):
        async def render(
            self,
            raw_input: str,
            instruction: str,
            *,
            allow_temporary_value: bool = False,
        ) -> RenderedUserInput:
            del raw_input, instruction, allow_temporary_value
            return RenderedUserInput(text="Please return status for ticket 42.")

    semantic_model = ProjectedResponseSemanticModel()

    async def successful_preflight(_settings: object) -> object:
        return _evaluator_preflight()

    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", _settings)
    monkeypatch.setattr(command_module, "preflight_evaluator", successful_preflight)
    monkeypatch.setattr(
        runner_module,
        "create_semantic_model_deconstructor",
        lambda _settings: semantic_model,
    )
    target = resolve_local_target(str(target_config))

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target",
            str(target_config),
            "--confirm-target",
            target.confirmation_sha256,
            "--confirm-test-environment",
            "--repetitions",
            "1",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    evidence = json.loads(output.read_text(encoding="utf-8").splitlines()[1])
    details = evidence["technical_details"]
    assert details["comparison_surface"] == "response"
    assert details["baseline"]["verdict"] == "no_divergence"
    assert details["cases"][0]["verdict"] == "no_divergence"
    assert details["baseline"]["trial_set"]["trials"][0]["target_output"]["raw_output"] == {
        "action": "lookup",
        "ticket": 42,
    }


def test_local_target_pause_action_preserves_binding_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    artifact = tmp_path / "customer-policy.json"
    dataset.write_text(
        '{"id":"case-1","input":"Return status for ticket 42.",'
        '"output":{"action":"lookup","ticket":42}}\n',
        encoding="utf-8",
    )
    (tmp_path / "customer_agent.py").write_text(
        "def run(value):\n    return {'action': 'lookup', 'ticket': 42, 'received': value}\n",
        encoding="utf-8",
    )
    artifact.write_text('{"policy":"test-only"}\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UL_LIVE", "true")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")

    async def successful_preflight(_settings: object) -> object:
        return _evaluator_preflight()

    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", _settings)
    monkeypatch.setattr(command_module, "preflight_evaluator", successful_preflight)
    monkeypatch.setattr(
        runner_module,
        "create_semantic_model_deconstructor",
        lambda _settings: _LocalEvaluationSemanticModel(),
    )
    create_runtime = command_module.create_campaign_progress_runtime

    def create_paused_runtime(**arguments: object) -> object:
        runtime = create_runtime(**arguments)
        runtime.control.request_pause()
        return runtime

    monkeypatch.setattr(
        command_module,
        "create_campaign_progress_runtime",
        create_paused_runtime,
    )
    target = resolve_local_target(
        "customer_agent:run",
        explicit_artifacts=(artifact,),
    )

    paused = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target",
            "customer_agent:run",
            "--target-artifact",
            str(artifact),
            "--confirm-target",
            target.confirmation_sha256,
            "--confirm-test-environment",
            "--repetitions",
            "1",
            "--output",
            str(output),
        ],
    )

    assert paused.exit_code == 130, paused.output
    action_id_match = re.search(r'next_argv=\["ul","action","([0-9a-f]{64})"\]', paused.output)
    assert action_id_match is not None
    receipt = json.loads(
        (tmp_path / "action-state" / f"{action_id_match.group(1)}.json").read_text(encoding="utf-8")
    )
    assert receipt["argv"] == [
        "ul",
        "dataset",
        "evaluate",
        "--resume",
        str(output.resolve()),
        "--target",
        "customer_agent:run",
        "--confirm-target",
        target.confirmation_sha256,
        "--target-artifact",
        str(artifact.resolve()),
    ]

    monkeypatch.setattr(
        command_module,
        "create_campaign_progress_runtime",
        create_runtime,
    )
    resumed = runner.invoke(root_app, receipt["argv"][1:])

    assert resumed.exit_code == 0, resumed.output
    saved = json.loads(output.read_text(encoding="utf-8").splitlines()[1])
    assert (
        saved["run_context"]["target"]["receipt"]["confirmation_sha256"]
        == target.confirmation_sha256
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable-script test")
def test_full_dataset_evaluation_runs_command_target_through_worker_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    dataset.write_text(
        '{"id":"case-1","input":"Return status for ticket 42.",'
        '"output":{"action":"lookup","ticket":42}}\n',
        encoding="utf-8",
    )
    command = tmp_path / "command-worker"
    command.write_text(
        r"""#!/bin/sh
read -r line
printf '%s\n' '{"protocol_version":"1.0.0","type":"ready","request_id":"startup","runtime":{"name":"sh","version":"1"}}'
while read -r line; do
  request_id=$(printf '%s' "$line" | /usr/bin/sed -n 's/.*"request_id":"\([^"]*\)".*/\1/p')
  case "$line" in
    *'"type":"session_start"'*)
      session_id=$(printf '%s' "$line" | /usr/bin/sed -n 's/.*"session_id":"\([^"]*\)".*/\1/p')
      printf '{"protocol_version":"1.0.0","type":"session_ready","request_id":"%s","session_id":"%s"}\n' "$request_id" "$session_id"
      ;;
    *'"type":"invoke"'*)
      printf '{"protocol_version":"1.0.0","type":"result","request_id":"%s","response":{"action":"lookup","ticket":42},"execution_events":[]}\n' "$request_id"
      ;;
    *'"type":"shutdown"'*)
      printf '%s\n' '{"protocol_version":"1.0.0","type":"shutdown_complete","request_id":"shutdown"}'
      exit 0
      ;;
  esac
done
""",
        encoding="utf-8",
    )
    command.chmod(0o700)
    config = tmp_path / "target.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "command",
                "target_id": "dataset-command-agent",
                "working_directory": str(tmp_path),
                "argv": [str(command)],
                "environment_allowlist": [],
                "limits": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UL_LIVE", "true")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")

    async def successful_preflight(_settings: object) -> object:
        return _evaluator_preflight()

    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", _settings)
    monkeypatch.setattr(command_module, "preflight_evaluator", successful_preflight)
    monkeypatch.setattr(
        runner_module,
        "create_semantic_model_deconstructor",
        lambda _settings: _LocalEvaluationSemanticModel(),
    )
    resolved_target = resolve_local_target(str(config))
    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target",
            str(config),
            "--confirm-target",
            resolved_target.confirmation_sha256,
            "--confirm-test-environment",
            "--repetitions",
            "1",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    saved = json.loads(output.read_text(encoding="utf-8").splitlines()[1])
    receipt = saved["run_context"]["target"]["receipt"]
    assert receipt["kind"] == "command"
    assert receipt["supports_state_observation"] is False
    assert saved["technical_details"]["baseline"]["trial_set"]["trials"][0]["execution_evidence"][
        "final_response"
    ] == {"action": "lookup", "ticket": 42}


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


@pytest.mark.parametrize("has_source_preparation_failure", (False, True))
def test_execution_creates_private_explicit_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    has_source_preparation_failure: bool,
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
            assert options["timeout_seconds"] == 75.0
            return cls()

    async def fake_evaluate(
        records: tuple[Any, ...],
        operator_ids: tuple[str, ...],
        settings: object,
        target: object,
        output_stream: Any,
        *,
        run_config: Any,
        run_context: object,
        augmentation_ledger: object,
        saved_augmentations: object,
        redaction_engine: object,
        evaluator_preflight: object,
        trial_journal: object,
        progress_plan: Any,
        source_preparation_failures: list[Any],
    ) -> tuple[object, ...]:
        del settings, target, augmentation_ledger, saved_augmentations
        assert evaluator_preflight == _evaluator_preflight()
        assert redaction_engine is None
        captured_records.extend(record.id for record in records)
        assert operator_ids == ("input.surface.disfluency_repeat",)
        assert run_config.repetitions == 3
        assert run_config.target.max_environment_api_calls == 100
        assert run_config.target.planned_environment_api_calls == 30
        assert run_config.target.trial_timeout_seconds == 75.0
        assert progress_plan.calls.total_environment_api == 30
        if has_source_preparation_failure:
            failure = customer_module.build_source_preparation_failure_evidence(
                records[0],
                DatasetSemanticPreparationError(),
                repetitions=run_config.repetitions,
                max_environment_api_calls=run_config.target.max_environment_api_calls,
                planned_target_calls=run_config.target.planned_environment_api_calls,
                run_context=cast(Any, run_context),
            )
            source_preparation_failures.append(failure)
            output_stream.write(failure.model_dump_json(exclude_none=True) + "\n")
        else:
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
            "--target-timeout-seconds",
            "75",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == (2 if has_source_preparation_failure else 0), result.output
    assert captured_records == ["interaction-1"]
    output_lines = output.read_text(encoding="utf-8").splitlines()
    assert json.loads(output_lines[0])["record_type"] == "dataset_durable_run"
    if has_source_preparation_failure:
        assert json.loads(output_lines[1])["record_type"] == "source_preparation_failure"
        assert "Source preparation failures: 1" in result.output
        assert "failed stage=terminal" in result.output
    else:
        assert output_lines[1] == '{"saved":true}'
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert "Complete evidence" in result.output
    assert "Next: ul dataset report" in result.output
    assert "Transfer 100" not in result.output
    report_position = result.output.index("stage=report")
    completion_position = result.output.index(
        "failed stage=terminal" if has_source_preparation_failure else "completed stage=terminal"
    )
    assert report_position < completion_position
    assert result.output.count("next_action=") == 1

    def fail_presentation(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("private-presentation-canary")

    failed_output = tmp_path / "failed-results.jsonl"
    monkeypatch.setattr(command_module, "print_dataset_results", fail_presentation)
    failed = runner.invoke(
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
            "--target-timeout-seconds",
            "75",
            "--output",
            str(failed_output),
        ],
    )

    assert failed.exit_code == 1
    assert "completed stage=terminal" not in failed.output
    assert "failed stage=terminal" in failed.output
    assert "private-presentation-canary" not in failed.output


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
        run_config: object,
        run_context: object,
        augmentation_ledger: object,
        saved_augmentations: object,
        redaction_engine: object,
        evaluator_preflight: object,
        trial_journal: object,
        progress_plan: object,
    ) -> tuple[object, ...]:
        del (
            run_config,
            augmentation_ledger,
            saved_augmentations,
            progress_plan,
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
            run_config: object,
            run_context: object,
            augmentation_ledger: object,
            saved_augmentations: object,
            redaction_engine: object,
            evaluator_preflight: object,
            trial_journal: object,
            progress_plan: object,
        ) -> tuple[object, ...]:
            del (
                operator_ids,
                settings,
                run_config,
                run_context,
                augmentation_ledger,
                saved_augmentations,
                progress_plan,
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


@pytest.mark.parametrize("target_mode", ("direct", "saved"))
def test_http_target_contract_runs_authenticated_loopback_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_mode: str,
) -> None:
    secret_canary = "UL59-HTTP-SECRET-CANARY"
    received_requests: list[object] = []

    class TargetHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.headers.get("Authorization") != secret_canary:
                self.send_response(401)
                self.end_headers()
                return
            content_length = int(self.headers["Content-Length"])
            received_requests.append(json.loads(self.rfile.read(content_length)))
            response = json.dumps({"result": {"action": "lookup", "ticket": 42}}).encode()
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
        output = tmp_path / "results.jsonl"
        target_config = tmp_path / "target.json"
        _write_dataset(dataset, [_record()])
        monkeypatch.setenv("UL_ENVIRONMENT_CUSTOMER_TOKEN", secret_canary)
        monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
        url = f"http://127.0.0.1:{server.server_port}/invoke"
        direct_options = {
            "allow_insecure_http": True,
            "request_json_template": '{"message":"{{input}}"}',
            "response_json_pointer": "/result",
            "header_from_env": ["Authorization=UL_ENVIRONMENT_CUSTOMER_TOKEN"],
        }
        if target_mode == "saved":
            config = create_isolated_response_target_config(
                url,
                request_json_template=cast(str, direct_options["request_json_template"]),
                response_json_pointer=cast(str, direct_options["response_json_pointer"]),
                header_from_env=cast(list[str], direct_options["header_from_env"]),
                request_isolation_attested=True,
                safe_test_target_attested=True,
            )
            target_config.write_text(
                json.dumps(config.model_dump(mode="json", exclude_none=True)),
                encoding="utf-8",
            )
            target_reference = str(target_config)
            target_arguments = ["--target", target_reference]
            resolved_target = resolve_http_target(target_reference, allow_insecure_http=True)
        else:
            target_reference = url
            target_arguments = [
                "--target",
                target_reference,
                "--request-json-template",
                cast(str, direct_options["request_json_template"]),
                "--response-json-pointer",
                cast(str, direct_options["response_json_pointer"]),
                "--header-from-env",
                cast(list[str], direct_options["header_from_env"])[0],
            ]
            resolved_target = resolve_http_target(
                target_reference,
                request_isolation_attested=True,
                safe_test_target_attested=True,
                **direct_options,
            )

        observed_outputs: list[object] = []

        async def evaluate_once(
            records: tuple[Any, ...],
            operator_ids: tuple[str, ...],
            settings: object,
            target: Any,
            output_stream: Any,
            **options: object,
        ) -> tuple[object, ...]:
            del operator_ids, settings, options
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

        monkeypatch.setattr(command_module, "load_dataset_semantic_settings", _settings)
        monkeypatch.setattr(command_module, "evaluate_interaction_records", evaluate_once)
        arguments = [
            "dataset",
            "evaluate",
            str(dataset),
            *target_arguments,
            "--allow-insecure-http",
            "--allow-environment-network",
            "--confirm-test-environment",
            "--confirm-target",
            resolved_target.confirmation_sha256,
            "--operator",
            "input.surface.case_variation",
            "--repetitions",
            "2",
            "--output",
            str(output),
        ]
        if target_mode == "direct":
            arguments.extend(["--confirm-request-isolation", "--confirm-safe-test-target"])
            create_runtime = command_module.create_campaign_progress_runtime

            def create_paused_runtime(**options: object) -> object:
                runtime = create_runtime(**options)
                runtime.control.request_pause()
                return runtime

            monkeypatch.setattr(
                command_module,
                "create_campaign_progress_runtime",
                create_paused_runtime,
            )
            paused = runner.invoke(root_app, arguments)
            assert paused.exit_code == 130, paused.output
            assert received_requests == []
            action_id_match = re.search(
                r'next_argv=\["ul","action","([0-9a-f]{64})"\]', paused.output
            )
            assert action_id_match is not None
            receipt = json.loads(
                (tmp_path / "action-state" / f"{action_id_match.group(1)}.json").read_text(
                    encoding="utf-8"
                )
            )
            assert receipt["argv"] == [
                "ul",
                "dataset",
                "evaluate",
                "--resume",
                str(output.resolve()),
            ]
            changed_target = resolve_http_target(
                target_reference,
                allow_insecure_http=True,
                request_json_template=cast(str, direct_options["request_json_template"]),
                response_json_pointer="/changed",
                header_from_env=cast(list[str], direct_options["header_from_env"]),
                request_isolation_attested=True,
                safe_test_target_attested=True,
            )
            changed = runner.invoke(
                root_app,
                [
                    "dataset",
                    "evaluate",
                    "--resume",
                    str(output),
                    "--target",
                    target_reference,
                    "--request-json-template",
                    cast(str, direct_options["request_json_template"]),
                    "--response-json-pointer",
                    "/changed",
                    "--header-from-env",
                    cast(list[str], direct_options["header_from_env"])[0],
                    "--confirm-request-isolation",
                    "--confirm-safe-test-target",
                    "--confirm-target",
                    changed_target.confirmation_sha256,
                ],
            )
            assert changed.exit_code != 0
            assert "resume_incompatible:target" in " ".join(changed.output.split())
            assert received_requests == []
            monkeypatch.setattr(
                command_module,
                "create_campaign_progress_runtime",
                create_runtime,
            )
            monkeypatch.setenv("UL_ENVIRONMENT_CUSTOMER_TOKEN", "changed-secret")
            changed_credential = runner.invoke(root_app, receipt["argv"][1:])
            assert changed_credential.exit_code != 0
            assert "HTTP target credential identity changed" in " ".join(
                changed_credential.output.replace("│", "").split()
            )
            assert received_requests == []
            monkeypatch.setenv("UL_ENVIRONMENT_CUSTOMER_TOKEN", secret_canary)
            result = runner.invoke(root_app, receipt["argv"][1:])
        else:
            result = runner.invoke(root_app, arguments)
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()

    assert result.exit_code == 0, result.output
    assert received_requests == [{"message": "Transfer 100 to Alice."}]
    assert observed_outputs == [{"action": "lookup", "ticket": 42}]
    manifest = read_dataset_run_manifest(manifest_path(output))
    assert manifest.run_context.target.kind == "probe_target"
    assert manifest.run_context.target.receipt == http_target_evidence_receipt(resolved_target)
    assert manifest.effective_command.http_target_config is not None
    assert manifest.effective_command.http_target_config.headers_from_env == {
        "Authorization": "UL_ENVIRONMENT_CUSTOMER_TOKEN"
    }
    persisted_text = "\n".join(
        path.read_text(encoding="utf-8") for path in tmp_path.iterdir() if path.is_file()
    )
    assert secret_canary not in persisted_text
