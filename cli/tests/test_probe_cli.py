# ruff: noqa: E501

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner
from ul import (
    DatasetEvaluationResult,
    InteractionRecord,
)
from ul_cli import probe as probe_module
from ul_cli.dataset.evaluation import runner as campaign_runner_module
from ul_cli.main import app
from ul_core.dataset import (
    EvidenceReference,
    ObservedOutcome,
    RenderedUserInput,
    RequestUnit,
    SemanticEquivalenceAssessment,
    SemanticFrame,
    UserInputRecord,
)

runner = CliRunner()


def _write_dataset(path: Path, count: int = 1) -> Path:
    dataset = path / "examples.jsonl"
    dataset.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"case-{index}",
                    "input": f"Echo grounded example {index}.",
                    "output": {"echo": f"Echo grounded example {index}."},
                }
            )
            + "\n"
            for index in range(1, count + 1)
        ),
        encoding="utf-8",
    )
    return dataset


def _write_callable(path: Path, *, failing: bool = False) -> None:
    body = (
        "def run(value):\n    raise RuntimeError('private target detail')\n"
        if failing
        else (
            "import json\n"
            "from pathlib import Path\n\n"
            "def run(value):\n"
            "    counter = Path('target-invocations.jsonl')\n"
            "    with counter.open('a', encoding='utf-8') as stream:\n"
            "        stream.write(json.dumps(value) + '\\n')\n"
            "    return {'echo': value, 'source': 'live-call'}\n"
        )
    )
    (path / "customer_agent.py").write_text(body, encoding="utf-8")


class _CleanRoomSemanticModel:
    async def __aenter__(self) -> _CleanRoomSemanticModel:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def reuse_preflight(self, result: object) -> None:
        del result

    async def deconstruct(
        self,
        record: InteractionRecord | UserInputRecord,
        reference_frame: SemanticFrame | None = None,
    ) -> SemanticFrame:
        if not isinstance(record, InteractionRecord):
            assert reference_frame is not None
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
            extractor_version="clean-room-test",
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
            explanation="The clean-room inputs have the same request.",
            verifier_version="clean-room-test",
        )


def test_declining_target_confirmation_imports_nothing_and_calls_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "customer_agent.py").write_text(
        "from pathlib import Path\n"
        "Path('module-imported').write_text('yes')\n"
        "def run(value): return value\n",
        encoding="utf-8",
    )
    dataset = _write_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    async def unexpected_preflight(*args: object, **kwargs: object) -> None:
        raise AssertionError("semantic preflight must not run")

    monkeypatch.setattr(probe_module, "preflight_evaluator", unexpected_preflight)
    result = runner.invoke(
        app,
        ["probe", str(dataset), "--target", "customer_agent:run"],
        input="n\n",
    )

    assert result.exit_code == 2
    assert "Reason: PROBE_TARGET_NOT_CONFIRMED" in result.output
    assert "no target or semantic-model calls" in result.output
    assert not (tmp_path / "module-imported").exists()
    assert not (tmp_path / ".ul").exists()


def test_callable_smoke_proves_target_call_and_decline_makes_zero_semantic_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_callable(tmp_path)
    dataset = _write_dataset(tmp_path, count=12)
    monkeypatch.chdir(tmp_path)

    async def unexpected_preflight(*args: object, **kwargs: object) -> None:
        raise AssertionError("semantic preflight must not run before paid confirmation")

    monkeypatch.setattr(probe_module, "preflight_evaluator", unexpected_preflight)
    result = runner.invoke(
        app,
        ["probe", str(dataset), "--target", "customer_agent:run", "--confirm-target"],
        input="n\n",
    )

    assert result.exit_code == 0, result.output
    assert "Smoke target invocation succeeded" in result.output
    assert 'Normalized response: {"echo":"Echo grounded example 1."' in result.output
    assert "Evidence level: response only" in result.output
    assert "Source interactions: 10 (maximum 10)" in result.output
    assert "Original agent invocations: 10" in result.output
    assert "Probe agent invocations: 10" in result.output
    assert "Repetitions: 1" in result.output
    assert "No semantic-model calls were made" in result.output
    assert len((tmp_path / "target-invocations.jsonl").read_text().splitlines()) == 1
    saved = json.loads((tmp_path / ".ul" / "probe.json").read_text())
    assert saved["target_kind"] == "python_callable"
    assert saved["limit"] == 10
    assert saved["repetitions"] == 1


def test_paid_precondition_failure_occurs_after_budget_without_semantic_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_callable(tmp_path)
    dataset = _write_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UL_LIVE", raising=False)
    calls = 0

    async def unexpected_preflight(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(probe_module, "preflight_evaluator", unexpected_preflight)
    result = runner.invoke(
        app,
        [
            "probe",
            str(dataset),
            "--target",
            "customer_agent:run",
            "--confirm-target",
            "--confirm-paid-execution",
        ],
    )

    assert result.exit_code == 2
    assert "Semantic-model calls: up to" in result.output
    assert "Maximum active wall time:" in result.output
    assert "Stage: augmentation preparation" in result.output
    assert "Reason: PROBE_SEMANTIC_CALLS_DISABLED" in result.output
    assert "Target safe to reuse: yes" in result.output
    assert calls == 0


def test_copy_ready_confirmation_run_reuses_bound_config_and_rebudgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_callable(tmp_path)
    dataset = _write_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    base_arguments = [
        "probe",
        str(dataset),
        "--target",
        "customer_agent:run",
        "--confirm-target",
    ]

    pilot = runner.invoke(app, base_arguments, input="n\n")
    confirmation = runner.invoke(app, [*base_arguments, "--confirmation-run"], input="n\n")

    assert pilot.exit_code == 0, pilot.output
    assert confirmation.exit_code == 0, confirmation.output
    assert "Using saved project config:" in confirmation.output
    assert "Repetitions: 3" in confirmation.output
    assert "Original agent invocations: 3" in confirmation.output
    assert "Probe agent invocations: 3" in confirmation.output
    assert "No semantic-model calls were made" in confirmation.output


def test_failed_smoke_has_staged_safe_diagnostic_and_does_not_save_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_callable(tmp_path, failing=True)
    dataset = _write_dataset(tmp_path)
    diagnostic = tmp_path / "private" / "diagnostic.json"
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "probe",
            str(dataset),
            "--target",
            "customer_agent:run",
            "--confirm-target",
            "--diagnostic-artifact",
            str(diagnostic),
        ],
    )

    assert result.exit_code == 2
    assert "Stage: smoke invocation" in result.output
    assert "Reason: PROBE_SMOKE_INCONCLUSIVE" in result.output
    assert "private target detail" not in result.output
    assert "Target safe to reuse: no" in result.output
    assert not (tmp_path / ".ul" / "probe.json").exists()
    assert json.loads(diagnostic.read_text())["reason_code"] == "PROBE_SMOKE_INCONCLUSIVE"
    if os.name != "nt":
        assert stat.S_IMODE(diagnostic.stat().st_mode) == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation requires privileges")
def test_successful_smoke_refuses_symlinked_project_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_callable(tmp_path)
    dataset = _write_dataset(tmp_path)
    redirected = tmp_path / "redirected"
    redirected.mkdir()
    (tmp_path / ".ul").symlink_to(redirected, target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        ["probe", str(dataset), "--target", "customer_agent:run", "--confirm-target"],
    )

    assert result.exit_code == 2
    assert "Reason: PROBE_CONFIG_WRITE_FAILED" in result.output
    assert not (redirected / "probe.json").exists()


def test_invalid_observations_fail_before_target_or_project_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_callable(tmp_path)
    dataset = tmp_path / "invalid.jsonl"
    dataset.write_text("not-json\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        ["probe", str(dataset), "--target", "customer_agent:run", "--confirm-target"],
    )

    assert result.exit_code == 2
    assert "Stage: observation import" in result.output
    assert "Reason: PROBE_OBSERVATION_INVALID" in result.output
    assert not (tmp_path / "target-invocations.jsonl").exists()
    assert not (tmp_path / ".ul").exists()


def test_confirmed_flow_composes_campaign_and_normal_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_callable(tmp_path)
    dataset = _write_dataset(tmp_path)
    output = tmp_path / ".ul" / "runs" / "evidence.jsonl"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UL_LIVE", "true")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    called: dict[str, object] = {}

    def fake_campaign(**kwargs: object) -> tuple[DatasetEvaluationResult, ...]:
        called["plan"] = kwargs["plan"]
        created_output = kwargs["output"]
        assert isinstance(created_output, Path)
        created_output.parent.mkdir(parents=True, exist_ok=True)
        created_output.write_text("normal evidence\n", encoding="utf-8")
        return ()

    def fake_results(
        results: tuple[DatasetEvaluationResult, ...],
        evidence: Path,
        **kwargs: object,
    ) -> None:
        called["results"] = (results, evidence, kwargs)

    def fake_report(evidence: Path, **kwargs: object) -> None:
        called["report"] = (evidence, kwargs)

    monkeypatch.setattr(probe_module, "_run_campaign", fake_campaign)
    monkeypatch.setattr(probe_module, "print_dataset_results", fake_results)
    monkeypatch.setattr(probe_module, "report_evidence", fake_report)

    result = runner.invoke(
        app,
        [
            "probe",
            str(dataset),
            "--target",
            "customer_agent:run",
            "--output",
            str(output),
            "--confirm-target",
            "--confirm-paid-execution",
        ],
    )

    assert result.exit_code == 0, result.output
    plan = called["plan"]
    assert isinstance(plan, probe_module.DatasetCampaignPlan)
    assert plan.calls.baseline == 1
    assert plan.calls.variation == 1
    assert plan.calls.repetitions == 1
    assert called["report"] == (output, {})
    assert "Stronger confirmation:" in result.output


def test_public_documentation_flow_runs_real_callable_campaign_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "agent.py").write_text(
        "def run(value):\n    return {'action': 'lookup', 'ticket': 42}\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "interactions.jsonl"
    dataset.write_text(
        '{"id":"case-1","input":"Return the status for ticket 42.",'
        '"output":{"action":"lookup","ticket":42}}\n',
        encoding="utf-8",
    )
    output = tmp_path / ".ul" / "runs" / "probe-evidence.jsonl"
    semantic_model = _CleanRoomSemanticModel()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UL_LIVE", "true")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")

    async def clean_room_preflight(settings: object) -> object:
        del settings
        return object()

    def clean_room_model(settings: object) -> _CleanRoomSemanticModel:
        del settings
        return semantic_model

    monkeypatch.setattr(probe_module, "preflight_evaluator", clean_room_preflight)
    monkeypatch.setattr(
        campaign_runner_module,
        "create_semantic_model_deconstructor",
        clean_room_model,
    )
    result = runner.invoke(
        app,
        [
            "probe",
            str(dataset),
            "--target",
            "agent:run",
            "--output",
            str(output),
            "--confirm-target",
            "--confirm-paid-execution",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Smoke target invocation succeeded" in result.output
    assert "Dataset evaluation" in result.output
    assert "UL run report" in result.output
    assert "Evidence type: dataset evaluation" in result.output
    assert "Stronger confirmation:" in result.output
    evidence = json.loads(output.read_text().splitlines()[0])
    assert evidence["execution_plan"]["repetitions"] == 1
    assert evidence["execution_plan"]["dataset_planned_target_calls"] == 2
    assert evidence["technical_details"]["baseline"]["trial_set"]["trials"][0][
        "execution_evidence"
    ]["environment_id"].startswith("probe-")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable-script test")
def test_command_config_uses_real_subprocess_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = tmp_path / "command-worker"
    command.write_text(
        """#!/bin/sh
read -r line
printf '%s\n' '{"protocol_version":"1.0.0","type":"ready","request_id":"startup","runtime":{"name":"sh","version":"1"}}'
while read -r line; do
  request_id=$(printf '%s' "$line" | /usr/bin/sed -n 's/.*"request_id":"\\([^"]*\\)".*/\\1/p')
  case "$line" in
    *'"type":"session_start"'*)
      session_id=$(printf '%s' "$line" | /usr/bin/sed -n 's/.*"session_id":"\\([^"]*\\)".*/\\1/p')
      printf '{"protocol_version":"1.0.0","type":"session_ready","request_id":"%s","session_id":"%s"}\n' "$request_id" "$session_id"
      ;;
    *'"type":"invoke"'*)
      printf '{"protocol_version":"1.0.0","type":"result","request_id":"%s","response":{"transport":"command"},"execution_events":[]}\n' "$request_id"
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
                "target_id": "documented-command-agent",
                "working_directory": str(tmp_path),
                "argv": [str(command)],
                "environment_allowlist": [],
                "limits": {},
            }
        ),
        encoding="utf-8",
    )
    dataset = _write_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        ["probe", str(dataset), "--target", str(config), "--confirm-target"],
        input="n\n",
    )

    assert result.exit_code == 0, result.output
    assert 'Normalized response: {"transport":"command"}' in result.output
    saved = json.loads((tmp_path / ".ul" / "probe.json").read_text())
    assert saved["target_kind"] == "command"
