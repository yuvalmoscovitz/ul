# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import typer
from dataset._factories import _evaluator_preflight
from typer.testing import CliRunner
from ul import (
    DatasetEvaluationResult,
    InteractionRecord,
)
from ul_cli import probe as probe_module
from ul_cli import progress_action as progress_action_module
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


@pytest.fixture(autouse=True)
def _private_progress_action_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        progress_action_module,
        "_action_receipt_directory",
        lambda: tmp_path / "action-state",
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
        ["probe", str(dataset), "--target", "customer_agent:run"],
        input="y\nn\n",
    )

    assert result.exit_code == 0, result.output
    assert "Smoke target invocation succeeded" in result.output
    assert "Response structure: dict;" in result.output
    assert "Response sha256:" in result.output
    assert "Request sha256:" in result.output
    assert "Case: case-1:smoke" in result.output
    assert "Echo grounded example 1." not in result.output
    assert "Evidence level: response only" in result.output
    assert "Source interactions: 10 (maximum 10)" in result.output
    assert "Original agent invocations: 10" in result.output
    assert "Probe agent invocations: 10" in result.output
    assert "Repetitions: 1" in result.output
    assert "No semantic-model calls were made" in result.output
    assert len((tmp_path / "target-invocations.jsonl").read_text().splitlines()) == 1
    assert not (tmp_path / "__pycache__").exists()
    saved = json.loads((tmp_path / ".ul" / "probe.json").read_text())
    assert len((tmp_path / ".ul" / "review-history.key").read_bytes()) == 32
    assert saved["target_kind"] == "python_callable"
    assert len(saved["target_confirmation_sha256"]) == 64
    assert saved["limit"] == 10
    assert saved["repetitions"] == 1


def test_direct_authenticated_http_smoke_maps_request_and_response_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_requests: list[object] = []
    received_authorization: list[str | None] = []

    class TargetHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers["Content-Length"])
            received_requests.append(json.loads(self.rfile.read(content_length)))
            received_authorization.append(self.headers.get("Authorization"))
            response = json.dumps({"result": "mapped live response"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
    except PermissionError:
        pytest.skip("the test environment does not allow binding a loopback server")
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    secret = "Bearer private-test-secret"
    try:
        dataset = _write_dataset(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("UL_ENVIRONMENT_AGENT_TOKEN", secret)

        async def unexpected_preflight(*args: object, **kwargs: object) -> None:
            raise AssertionError("semantic preflight must not run before paid confirmation")

        monkeypatch.setattr(probe_module, "preflight_evaluator", unexpected_preflight)
        result = runner.invoke(
            app,
            [
                "probe",
                str(dataset),
                "--target",
                f"http://127.0.0.1:{server.server_port}/invoke",
                "--request-json-template",
                '{"payload":{"prompt":"{{input}}"}}',
                "--response-json-pointer",
                "/result",
                "--header-from-env",
                "Authorization=UL_ENVIRONMENT_AGENT_TOKEN",
                "--allow-insecure-http",
            ],
            input="y\nn\n",
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()

    assert result.exit_code == 0, result.output
    assert received_requests == [{"payload": {"prompt": "Echo grounded example 1."}}]
    assert received_authorization == [secret]
    assert "Response structure: str;" in result.output
    assert "Evidence level: response only" in result.output
    assert "No semantic-model calls were made" in result.output
    assert secret not in result.output
    assert secret not in (tmp_path / ".ul" / "probe.json").read_text()


def test_direct_http_rejects_echoed_nonstandard_header_before_semantic_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_header: list[str | None] = []
    secret = "private-workspace-value"

    class TargetHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            received_header.append(self.headers.get("X-Workspace"))
            content_length = int(self.headers["Content-Length"])
            self.rfile.read(content_length)
            response = json.dumps({"response": secret}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
    except PermissionError:
        pytest.skip("the test environment does not allow binding a loopback server")
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    semantic_calls = 0
    try:
        dataset = _write_dataset(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("UL_ENVIRONMENT_WORKSPACE", secret)

        async def unexpected_preflight(*args: object, **kwargs: object) -> None:
            nonlocal semantic_calls
            semantic_calls += 1

        monkeypatch.setattr(probe_module, "preflight_evaluator", unexpected_preflight)
        result = runner.invoke(
            app,
            [
                "probe",
                str(dataset),
                "--target",
                f"http://127.0.0.1:{server.server_port}/invoke",
                "--header-from-env",
                "X-Workspace=UL_ENVIRONMENT_WORKSPACE",
                "--allow-insecure-http",
            ],
            input="y\n",
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()

    assert result.exit_code == 2
    assert received_header == [secret]
    assert "Reason: PROBE_SMOKE_INCONCLUSIVE" in result.output
    assert semantic_calls == 0
    assert secret not in result.output
    assert not (tmp_path / ".ul" / "probe.json").exists()
    assert all(secret.encode() not in path.read_bytes() for path in (tmp_path / ".ul").iterdir())


def test_direct_openai_http_preset_builds_existing_response_only_target(tmp_path: Path) -> None:
    resolved = probe_module._resolve_target(
        "https://agent.example.test/v1/chat/completions",
        allow_insecure_http=False,
        http_preset="openai-chat",
        agent_model="test-agent-model",
        header_from_env=[],
    )

    assert resolved.kind == "http"
    config = cast(probe_module.JsonHttpTargetConfig, resolved.config)
    assert config.model_dump(mode="json")["execute"] == {
        "url": "https://agent.example.test/v1/chat/completions",
        "request_json_template": {
            "model": "test-agent-model",
            "messages": [{"role": "user", "content": "{{input}}"}],
        },
        "response_json_pointer": "/choices/0/message/content",
    }
    assert resolved.calls_per_execution == 1
    assert resolved.supports_state_observation is False


def test_direct_http_target_rejects_remote_plaintext_even_with_opt_in() -> None:
    with pytest.raises(probe_module.ProbeFailure, match="exact loopback"):
        probe_module._resolve_target(
            "http://agent.example.test/invoke",
            allow_insecure_http=True,
        )


@pytest.mark.parametrize(
    "direct_http_arguments",
    (
        ("--http-preset", "generic-json"),
        ("--http-preset", "openai-chat"),
        ("--request-json-template", '{"input":"{{input}}"}'),
        ("--response-json-pointer", "/response"),
        ("--agent-model", "test-model"),
        ("--header-from-env", "X-Test=UL_ENVIRONMENT_TEST"),
        ("--allow-insecure-http",),
    ),
)
def test_callable_rejects_every_direct_http_mapping_option(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    direct_http_arguments: tuple[str, ...],
) -> None:
    _write_callable(tmp_path)
    dataset = _write_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UL_ENVIRONMENT_TEST", "private-test-value")

    result = runner.invoke(
        app,
        [
            "probe",
            str(dataset),
            "--target",
            "customer_agent:run",
            *direct_http_arguments,
        ],
    )

    assert result.exit_code == 2
    assert "Reason: PROBE_TARGET_INVALID" in result.output
    assert "HTTP" in result.output
    assert not (tmp_path / "target-invocations.jsonl").exists()
    assert not (tmp_path / ".ul").exists()


def test_private_smoke_output_requires_explicit_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_callable(tmp_path)
    dataset = _write_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "probe",
            str(dataset),
            "--target",
            "customer_agent:run",
            "--show-smoke-response",
        ],
        input="y\nn\n",
    )

    assert result.exit_code == 0, result.output
    assert "Private raw target response:" in result.output


def _write_projected_callable_config(
    path: Path, outcome: dict[str, object], *, target: str = "customer_agent:run"
) -> Path:
    config = path / "projected-target.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "python_callable",
                "target_id": "projected-agent",
                "working_directory": str(path),
                "interpreter": str(Path(sys.executable).resolve()),
                "target": target,
                "outcome": outcome,
            }
        ),
        encoding="utf-8",
    )
    return config


def test_smoke_previews_projection_before_any_semantic_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_callable(tmp_path)
    dataset = _write_dataset(tmp_path)
    config = _write_projected_callable_config(
        tmp_path,
        {
            "schema_version": "1.0.0",
            "complete_result": "",
            "private_json_pointers": ["/source"],
        },
    )
    monkeypatch.chdir(tmp_path)

    async def unexpected_preflight(*args: object, **kwargs: object) -> None:
        raise AssertionError("semantic preflight must not run before paid confirmation")

    monkeypatch.setattr(probe_module, "preflight_evaluator", unexpected_preflight)
    result = runner.invoke(app, ["probe", str(dataset), "--target", str(config)], input="y\nn\n")

    assert result.exit_code == 0, result.output
    assert "Target-reported normalized result preview:" in result.output
    assert "State summary: unverified (no state observation configured)" in result.output
    assert '"source":"[PRIVATE]"' in result.output
    assert "live-call" not in result.output
    saved = json.loads((tmp_path / ".ul" / "probe.json").read_text())
    assert len(saved["outcome_projection_sha256"]) == 64


def test_invalid_smoke_projection_names_selector_and_makes_zero_semantic_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_callable(tmp_path)
    dataset = _write_dataset(tmp_path)
    config = _write_projected_callable_config(
        tmp_path,
        {"schema_version": "1.0.0", "action": "/missing/action"},
    )
    monkeypatch.chdir(tmp_path)
    semantic_calls = 0

    async def unexpected_preflight(*args: object, **kwargs: object) -> None:
        nonlocal semantic_calls
        semantic_calls += 1

    monkeypatch.setattr(probe_module, "preflight_evaluator", unexpected_preflight)
    result = runner.invoke(app, ["probe", str(dataset), "--target", str(config)], input="y\n")

    assert result.exit_code == 2
    assert "Reason: PROBE_OUTCOME_PROJECTION_INVALID" in result.output
    assert "'action' at selector '/missing/action' does not resolve" in result.output
    assert "Restore a known-safe fixture before retrying" in result.output
    assert "Target safe to reuse: no" in result.output
    assert semantic_calls == 0
    safety_state = next((tmp_path / ".ul").glob("probe-quarantine-*.json"))
    assert json.loads(safety_state.read_text())["status"] == "quarantined"


def test_saved_probe_binding_rejects_altered_projection_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_callable(tmp_path)
    dataset = _write_dataset(tmp_path)
    config = _write_projected_callable_config(
        tmp_path,
        {"schema_version": "1.0.0", "complete_result": ""},
    )
    monkeypatch.chdir(tmp_path)
    first = runner.invoke(app, ["probe", str(dataset), "--target", str(config)], input="y\nn\n")
    assert first.exit_code == 0, first.output
    saved_path = tmp_path / ".ul" / "probe.json"
    saved = json.loads(saved_path.read_text())
    saved["outcome_projection_sha256"] = "0" * 64
    saved_path.write_text(json.dumps(saved), encoding="utf-8")
    saved_path.chmod(0o600)

    second = runner.invoke(app, ["probe", str(dataset), "--target", str(config)])

    assert second.exit_code == 2
    assert "Reason: PROBE_CONFIG_EXISTS" in second.output
    assert len((tmp_path / "target-invocations.jsonl").read_text().splitlines()) == 1


def test_run_target_receipt_records_projection_definition_and_digest(tmp_path: Path) -> None:
    _write_callable(tmp_path)
    config = _write_projected_callable_config(
        tmp_path,
        {
            "schema_version": "1.0.0",
            "action": "/result/action",
            "status": "/result/status",
        },
    )
    resolved = probe_module._resolve_target(str(config), allow_insecure_http=False)
    assert resolved.config.outcome is not None

    receipt = probe_module._target_evidence_receipt(resolved)

    assert receipt["outcome_projection"] == {
        "schema_version": "1.0.0",
        "action": "/result/action",
        "status": "/result/status",
        "resource_id": None,
        "decision": None,
        "amount": None,
        "effects": None,
        "complete_result": None,
        "private_json_pointers": [],
    }
    assert receipt["outcome_projection_sha256"] == resolved.config.outcome.digest


def test_changed_target_artifact_is_rejected_immediately_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_callable(tmp_path)
    dataset = _write_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    def mutate_then_confirm(*args: object, **kwargs: object) -> bool:
        del args, kwargs
        (tmp_path / "customer_agent.py").write_text(
            "def run(value): return {'changed': value}\n", encoding="utf-8"
        )
        return True

    monkeypatch.setattr(typer, "confirm", mutate_then_confirm)
    result = runner.invoke(app, ["probe", str(dataset), "--target", "customer_agent:run"])

    assert result.exit_code == 2
    assert "Reason: PROBE_TARGET_IDENTITY_CHANGED" in result.output
    assert not (tmp_path / "target-invocations.jsonl").exists()


def test_python_declared_helpers_and_allowlisted_environment_are_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_callable(tmp_path)
    helper = tmp_path / "customer_helper.py"
    helper.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUSTOMER_MODE", "test-a")
    config = probe_module.PythonCallableTargetConfig(
        target_id="bound-agent",
        working_directory=tmp_path,
        interpreter=Path(sys.executable).resolve(),
        target="customer_agent:run",
        environment_allowlist=("CUSTOMER_MODE",),
    )
    plan = probe_module.create_local_target_dry_run_plan(config)
    resolved = probe_module._local_target(
        "customer_agent:run", config, plan.config_sha256, (helper,)
    )

    assert str(helper.resolve()) in {item.path for item in resolved.confirmation.artifacts}
    assert resolved.confirmation.environment[0].name == "CUSTOMER_MODE"
    first_digest = resolved.confirmation.environment[0].value_sha256
    monkeypatch.setenv("CUSTOMER_MODE", "test-b")
    with pytest.raises(probe_module.ProbeFailure, match="artifact changed"):
        resolved.revalidate_identity()
    assert first_digest != hashlib.sha256(b"test-b").hexdigest()


def test_existing_binding_is_checked_before_target_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_callable(tmp_path)
    dataset = _write_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    first = runner.invoke(
        app,
        ["probe", str(dataset), "--target", "customer_agent:run"],
        input="y\nn\n",
    )
    assert first.exit_code == 0, first.output
    saved = json.loads((tmp_path / ".ul" / "probe.json").read_text())
    saved["dataset"] = str(tmp_path / "different.jsonl")
    (tmp_path / ".ul" / "probe.json").write_text(json.dumps(saved), encoding="utf-8")
    (tmp_path / ".ul" / "probe.json").chmod(0o600)

    second = runner.invoke(
        app,
        ["probe", str(dataset), "--target", "customer_agent:run"],
        input="y\n",
    )

    assert second.exit_code == 2
    assert "Reason: PROBE_CONFIG_EXISTS" in second.output
    assert len((tmp_path / "target-invocations.jsonl").read_text().splitlines()) == 1


def test_post_smoke_preparation_failure_is_safe_and_staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_callable(tmp_path)
    dataset = _write_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)

    def fail_settings() -> None:
        raise ValueError("private-provider-secret")

    monkeypatch.setattr(probe_module, "load_dataset_semantic_settings", fail_settings)
    result = runner.invoke(
        app,
        ["probe", str(dataset), "--target", "customer_agent:run"],
        input="y\n",
    )

    assert result.exit_code == 2
    assert "Stage: augmentation preparation" in result.output
    assert "Reason: PROBE_AUGMENTATION_PREPARATION_FAILED" in result.output
    assert "private-provider-secret" not in result.output


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
        ],
        input="y\ny\n",
    )

    assert result.exit_code == 2
    assert "Semantic-model calls: up to" in result.output
    assert "Maximum active wall time:" in result.output
    assert "Stage: augmentation preparation" in result.output
    assert "Reason: PROBE_SEMANTIC_CALLS_DISABLED" in result.output
    assert "Target safe to reuse: yes" in result.output
    assert calls == 0


def test_command_wide_local_execution_limit_includes_smoke_and_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_callable(tmp_path)
    dataset = _write_dataset(tmp_path)
    config = tmp_path / "target.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "python_callable",
                "target_id": "bounded-agent",
                "working_directory": str(tmp_path),
                "interpreter": sys.executable,
                "target": "customer_agent:run",
                "environment_allowlist": [],
                "limits": {"max_executions": 2},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["probe", str(dataset), "--target", str(config)], input="y\n")

    assert result.exit_code == 2
    assert "Reason: PROBE_TARGET_CALL_LIMIT_TOO_LOW" in result.output
    assert len((tmp_path / "target-invocations.jsonl").read_text().splitlines()) == 1


def test_stale_campaign_digest_cannot_authorize_changed_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_callable(tmp_path)
    dataset = _write_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    stale_digest = "0" * 64

    result = runner.invoke(
        app,
        [
            "probe",
            str(dataset),
            "--target",
            "customer_agent:run",
            "--confirm-paid-execution",
            stale_digest,
        ],
        input="y\n",
    )

    assert result.exit_code == 2
    assert "Semantic provider:" in result.output
    assert "Semantic endpoint sha256:" in result.output
    assert "Semantic settings sha256:" in result.output
    assert "Data policy:" in result.output
    assert "UNKNOWN AND UNBOUNDED" in result.output
    assert "Reason: PROBE_CAMPAIGN_CONFIRMATION_CHANGED" in result.output


def test_campaign_receipt_binds_models_bounds_and_command_wide_smoke_wall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_callable(tmp_path)
    records = probe_module._load_pilot_records(_write_dataset(tmp_path))
    monkeypatch.chdir(tmp_path)
    resolved = probe_module._resolve_target("customer_agent:run", allow_insecure_http=False)
    settings = probe_module.load_dataset_semantic_settings()
    operator_ids = probe_module.validate_operator_ids(["input.surface.typing_noise"])
    plan = probe_module.create_dataset_campaign_plan(
        records=records,
        selected_operator_ids=operator_ids,
        repetitions=1,
        target_calls_per_execution=1,
        settings=settings,
    )
    original = probe_module._campaign_confirmation(plan, settings, resolved)
    changed = probe_module._campaign_confirmation(
        plan,
        settings.model_copy(update={"model": "different-model", "max_output_tokens": 1234}),
        resolved,
    )

    assert original.semantic_settings_sha256 != changed.semantic_settings_sha256
    assert probe_module._model_sha256(original) != probe_module._model_sha256(changed)
    assert original.maximum_wall_seconds >= resolved.maximum_active_target_seconds
    assert original.monetary_cost_status == "unknown_unbounded"

    http_config = tmp_path / "http-target.json"
    http_config.write_text(
        json.dumps(
            {
                "version": 1,
                "adapter_tier": "isolated_response",
                "environment_id": "http-smoke-agent",
                "request_isolation_attested": True,
                "safe_test_target_attested": True,
                "execute": {
                    "url": "https://agent.example.test/execute",
                    "request_json_template": {"input": "{{input}}"},
                    "response_json_pointer": "/response",
                },
            }
        ),
        encoding="utf-8",
    )
    http_target = probe_module._resolve_target(str(http_config), allow_insecure_http=False)
    http_confirmation = probe_module._campaign_confirmation(plan, settings, http_target)
    expected_http_wall = (
        (1 + plan.calls.repetition_executions) * probe_module._TARGET_TIMEOUT_SECONDS
        + plan.calls.total_semantic_model * settings.timeout_seconds
    )
    assert http_confirmation.maximum_wall_seconds == expected_http_wall


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
    ]

    pilot = runner.invoke(app, base_arguments, input="y\nn\n")
    (tmp_path / ".ul" / "review-history.key").unlink()
    confirmation = runner.invoke(app, [*base_arguments, "--confirmation-run"], input="y\nn\n")

    assert pilot.exit_code == 0, pilot.output
    assert confirmation.exit_code == 0, confirmation.output
    assert "Using saved project config:" in confirmation.output
    assert "Repetitions: 3" in confirmation.output
    assert "Original agent invocations: 3" in confirmation.output
    assert "Probe agent invocations: 3" in confirmation.output
    assert "No semantic-model calls were made" in confirmation.output
    assert len((tmp_path / ".ul" / "review-history.key").read_bytes()) == 32


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
            "--diagnostic-artifact",
            str(diagnostic),
        ],
        input="y\n",
    )

    assert result.exit_code == 2
    assert "Stage: smoke invocation" in result.output
    assert "Reason: PROBE_SMOKE_INCONCLUSIVE" in result.output
    assert "private target detail" not in result.output
    assert "Target safe to reuse: no" in result.output
    assert "target_calls=1" in result.output
    assert "environment_calls=1" in result.output
    assert not (tmp_path / ".ul" / "probe.json").exists()
    assert json.loads(diagnostic.read_text())["reason_code"] == "PROBE_SMOKE_INCONCLUSIVE"
    if os.name != "nt":
        assert stat.S_IMODE(diagnostic.stat().st_mode) == 0o600
    quarantine_receipts = list((tmp_path / ".ul").glob("probe-quarantine-*.json"))
    assert len(quarantine_receipts) == 1
    if os.name != "nt":
        assert stat.S_IMODE(quarantine_receipts[0].stat().st_mode) == 0o600
    blocked = runner.invoke(
        app,
        ["probe", str(dataset), "--target", "customer_agent:run"],
    )

    assert blocked.exit_code == 2
    assert "Reason: PROBE_TARGET_QUARANTINED" in blocked.output

    resolved = runner.invoke(
        app,
        [
            "probe",
            str(dataset),
            "--target",
            "customer_agent:run",
            "--resolve-quarantine-after",
            "environment-reset",
        ],
        input="y\n",
    )

    assert resolved.exit_code == 2
    assert "Reason: PROBE_SMOKE_INCONCLUSIVE" in resolved.output


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation requires privileges")
def test_probe_safety_state_deletion_and_dangling_symlink_fail_closed_before_target_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_callable(tmp_path)
    dataset = _write_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    arguments = ["probe", str(dataset), "--target", "customer_agent:run"]

    first = runner.invoke(app, arguments, input="y\nn\n")

    assert first.exit_code == 0, first.output
    invocations = tmp_path / "target-invocations.jsonl"
    assert len(invocations.read_text().splitlines()) == 1
    safety_state = next((tmp_path / ".ul").glob("probe-quarantine-*.json"))
    safety_state.unlink()

    missing = runner.invoke(app, arguments)

    assert missing.exit_code == 2
    assert "Reason: PROBE_QUARANTINE_RECEIPT_MISSING" in missing.output
    assert len(invocations.read_text().splitlines()) == 1
    safety_state.unlink()
    safety_state.symlink_to(tmp_path / "missing-safety-state.json")

    dangling = runner.invoke(app, arguments)

    assert dangling.exit_code == 2
    assert "Reason: PROBE_QUARANTINE_RECEIPT_INVALID" in dangling.output
    assert len(invocations.read_text().splitlines()) == 1


def test_target_lock_serializes_probe_safety_check_through_target_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_callable(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    resolved_target = probe_module._resolve_target(
        "customer_agent:run",
        allow_insecure_http=False,
    )
    first_descriptor, _ = probe_module._open_probe_target_lock(resolved_target)
    second_acquired = threading.Event()
    second_descriptor: list[int] = []

    def acquire_same_target() -> None:
        descriptor, _ = probe_module._open_probe_target_lock(resolved_target)
        second_descriptor.append(descriptor)
        second_acquired.set()

    thread = threading.Thread(target=acquire_same_target)
    thread.start()
    try:
        assert not second_acquired.wait(0.1)
    finally:
        probe_module._close_probe_target_lock(first_descriptor)
    assert second_acquired.wait(2)
    probe_module._close_probe_target_lock(second_descriptor.pop())
    thread.join(timeout=2)
    assert not thread.is_alive()


@pytest.mark.skipif(sys.platform == "win32", reason="directory fsync is POSIX-specific")
def test_probe_quarantine_fsyncs_private_state_and_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_callable(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    resolved_target = probe_module._resolve_target(
        "customer_agent:run",
        allow_insecure_http=False,
    )
    synced_file_types: list[int] = []

    def record_fsync(descriptor: int) -> None:
        synced_file_types.append(stat.S_IFMT(os.fstat(descriptor).st_mode))

    monkeypatch.setattr(probe_module.os, "fsync", record_fsync)

    probe_module._persist_probe_quarantine(resolved_target, "delivery_uncertain")

    assert stat.S_IFREG in synced_file_types
    assert stat.S_IFDIR in synced_file_types
    safety_state = next((tmp_path / ".ul").glob("probe-quarantine-*.json"))
    assert json.loads(safety_state.read_text())["status"] == "quarantined"


@pytest.mark.parametrize(
    ("mapping_arguments", "expected_request", "response_value"),
    (
        (
            ("--http-preset", "openai-chat", "--agent-model", "test-agent-model"),
            {
                "model": "test-agent-model",
                "messages": [{"role": "user", "content": "Return ticket 42."}],
            },
            {"choices": [{"message": {"content": {"action": "lookup", "ticket": 42}}}]},
        ),
        (
            (
                "--request-json-template",
                '{"custom":{"prompt":"{{input}}"}}',
                "--response-json-pointer",
                "/custom/result",
            ),
            {"custom": {"prompt": "Return ticket 42."}},
            {"custom": {"result": {"action": "lookup", "ticket": 42}}},
        ),
    ),
)
def test_authenticated_direct_http_pause_resume_preserves_mapping_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mapping_arguments: tuple[str, ...],
    expected_request: dict[str, object],
    response_value: dict[str, object],
) -> None:
    received_requests: list[object] = []
    received_headers: list[str | None] = []

    class TargetHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers["Content-Length"])
            received_requests.append(json.loads(self.rfile.read(content_length)))
            received_headers.append(self.headers.get("X-Agent-Key"))
            response = json.dumps(response_value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
    except PermissionError:
        pytest.skip("the test environment does not allow binding a loopback server")
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    output = tmp_path / "evidence.jsonl"
    dataset = tmp_path / "examples.jsonl"
    dataset.write_text(
        '{"id":"case-1","input":"Return ticket 42.","output":{"action":"lookup","ticket":42}}\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UL_ENVIRONMENT_AGENT_KEY", "private-agent-key")
    monkeypatch.setenv("UL_LIVE", "true")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    preflight_calls = 0

    async def clean_room_preflight(_settings: object) -> object:
        nonlocal preflight_calls
        preflight_calls += 1
        return _evaluator_preflight()

    async def load_clean_room_preflight(_output: Path, _settings: object) -> tuple[object, Path]:
        return _evaluator_preflight(), _output.with_name(f"{_output.name}.preflight.json")

    create_runtime = probe_module.create_campaign_progress_runtime
    runtime_calls = 0

    def create_paused_runtime(**arguments: object) -> object:
        nonlocal runtime_calls
        runtime_calls += 1
        runtime = create_runtime(**arguments)
        if runtime_calls == 1:
            runtime.control.request_pause()
        return runtime

    monkeypatch.setattr(probe_module, "preflight_evaluator", clean_room_preflight)
    monkeypatch.setattr(probe_module, "load_evaluator_preflight", load_clean_room_preflight)
    monkeypatch.setattr(probe_module, "create_campaign_progress_runtime", create_paused_runtime)
    monkeypatch.setattr(
        campaign_runner_module,
        "create_semantic_model_deconstructor",
        lambda _settings: _CleanRoomSemanticModel(),
    )
    endpoint = f"http://127.0.0.1:{server.server_port}/invoke"
    try:
        result = runner.invoke(
            app,
            [
                "probe",
                str(dataset),
                "--target",
                endpoint,
                "--output",
                str(output),
                *mapping_arguments,
                "--header-from-env",
                "X-Agent-Key=UL_ENVIRONMENT_AGENT_KEY",
                "--allow-insecure-http",
            ],
            input="y\ny\n",
        )

        assert result.exit_code == 130, result.output
        action_match = re.search(
            r'next_argv=\["ul","action","([0-9a-f]{64})"\]',
            result.output,
        )
        assert action_match is not None
        receipt = progress_action_module._read_progress_action(action_match.group(1))
        for argument in (*mapping_arguments, "X-Agent-Key=UL_ENVIRONMENT_AGENT_KEY"):
            assert argument in receipt.argv

        nested_results = []

        def invoke_trusted_cli(
            argv: tuple[str, ...],
            *,
            check: bool,
            cwd: str,
        ) -> SimpleNamespace:
            assert check is False
            assert Path(cwd) == tmp_path
            nested = runner.invoke(app, list(argv[4:]))
            nested_results.append(nested)
            return SimpleNamespace(returncode=nested.exit_code)

        monkeypatch.setattr(progress_action_module.subprocess, "run", invoke_trusted_cli)
        action_result = runner.invoke(app, ["action", action_match.group(1)])
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()

    assert action_result.exit_code == 0, nested_results[0].output
    assert nested_results[0].exit_code == 0, nested_results[0].output
    assert "Reusing durable smoke and evaluator preflight checkpoints" in nested_results[0].output
    assert preflight_calls == 1
    assert received_requests[:2] == [expected_request, expected_request]
    assert len(received_requests) == 3
    assert received_headers == ["private-agent-key"] * 3


@pytest.mark.parametrize(
    "checkpoint_tamper",
    (None, "elapsed", "evidence", "marker", "running", "running_marker"),
)
def test_paused_probe_action_blocks_before_repeating_completed_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_tamper: str | None,
) -> None:
    (tmp_path / "agent.py").write_text(
        "import json\n"
        "from pathlib import Path\n\n"
        "def run(value):\n"
        "    with Path('target-invocations.jsonl').open('a') as stream:\n"
        "        stream.write(json.dumps(value) + '\\n')\n"
        "    return {'action': 'lookup', 'ticket': 42}\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "examples.jsonl"
    dataset.write_text(
        '{"id":"case-1","input":"Return ticket 42.","output":{"action":"lookup","ticket":42}}\n',
        encoding="utf-8",
    )
    output = tmp_path / "evidence.jsonl"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UL_LIVE", "true")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")

    preflight_calls = 0

    async def clean_room_preflight(_settings: object) -> object:
        nonlocal preflight_calls
        preflight_calls += 1
        return _evaluator_preflight()

    async def load_clean_room_preflight(_output: Path, _settings: object) -> tuple[object, Path]:
        return _evaluator_preflight(), _output.with_name(f"{_output.name}.preflight.json")

    create_runtime = probe_module.create_campaign_progress_runtime
    runtime_calls = 0

    def create_paused_runtime(**arguments: object) -> object:
        nonlocal runtime_calls
        runtime_calls += 1
        runtime = create_runtime(**arguments)
        if runtime_calls == 1:
            runtime.control.request_pause()
        return runtime

    monkeypatch.setattr(probe_module, "preflight_evaluator", clean_room_preflight)
    monkeypatch.setattr(probe_module, "load_evaluator_preflight", load_clean_room_preflight)
    monkeypatch.setattr(
        probe_module,
        "create_campaign_progress_runtime",
        create_paused_runtime,
    )
    semantic_model = _CleanRoomSemanticModel()
    monkeypatch.setattr(
        campaign_runner_module,
        "create_semantic_model_deconstructor",
        lambda _settings: semantic_model,
    )
    probe_arguments = [
        "probe",
        str(dataset),
        "--target",
        "agent:run",
        "--output",
        str(output),
    ]
    if checkpoint_tamper == "running":
        probe_arguments.append("--confirmation-run")
    result = runner.invoke(app, probe_arguments, input="y\ny\n")

    assert result.exit_code == 130, result.output
    assert "Reason: PROBE_PAUSED_AFTER_PREFLIGHT" in result.output
    assert "environment=reusable" in result.output
    assert "next_action=resume" in result.output
    action_match = re.search(
        r'next_argv=\["ul","action","([0-9a-f]{64})"\]',
        result.output,
    )
    assert action_match is not None
    invocations = tmp_path / "target-invocations.jsonl"
    assert len(invocations.read_text().splitlines()) == 1
    if checkpoint_tamper in {"elapsed", "evidence"}:
        checkpoint_path = tmp_path / "evidence.jsonl.probe-checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_tamper == "elapsed":
            checkpoint["smoke_elapsed_seconds"] = 0
        else:
            checkpoint["smoke_evidence"]["final_response"] = {
                "structurally_valid": "PRIVATE_FORGED_EVIDENCE"
            }
        checkpoint_path.write_text(json.dumps(checkpoint) + "\n", encoding="utf-8")
    elif checkpoint_tamper == "marker":
        marker = json.loads(output.read_text(encoding="utf-8"))
        marker["manifest_sha256"] = "0" * 64
        output.write_text(json.dumps(marker) + "\n", encoding="utf-8")
    elif checkpoint_tamper in {"running", "running_marker"}:
        manifest = probe_module.read_dataset_run_manifest(
            output.with_name(f"{output.name}.manifest.json")
        )
        journal = probe_module.open_dataset_trial_journal(
            output.with_name(f"{output.name}.trials.jsonl"),
            manifest,
        )
        journal.start(manifest.work_plan[-1])
        journal.close()
        if checkpoint_tamper == "running_marker":
            marker = json.loads(output.read_text(encoding="utf-8"))
            marker["manifest_sha256"] = "0" * 64
            output.write_text(json.dumps(marker) + "\n", encoding="utf-8")
    nested_results = []

    def invoke_trusted_cli(
        argv: tuple[str, ...],
        *,
        check: bool,
        cwd: str,
    ) -> SimpleNamespace:
        assert check is False
        assert Path(cwd) == tmp_path
        nested = runner.invoke(app, list(argv[4:]))
        nested_results.append(nested)
        return SimpleNamespace(returncode=nested.exit_code)

    monkeypatch.setattr(progress_action_module.subprocess, "run", invoke_trusted_cli)
    action_result = runner.invoke(app, ["action", action_match.group(1)])

    assert preflight_calls == 1
    if checkpoint_tamper is None:
        assert action_result.exit_code == 0, nested_results[0].output
        assert nested_results[0].exit_code == 0, nested_results[0].output
        assert (
            "Reusing durable smoke and evaluator preflight checkpoints" in nested_results[0].output
        )
        assert len(invocations.read_text().splitlines()) == 3
    elif checkpoint_tamper in {"elapsed", "evidence"}:
        assert action_result.exit_code == 2
        assert "Reason: PROBE_CHECKPOINT_INVALID" in nested_results[0].output
        assert "PRIVATE_FORGED_EVIDENCE" not in nested_results[0].output
        assert len(invocations.read_text().splitlines()) == 1
    elif checkpoint_tamper == "marker":
        assert action_result.exit_code == 2
        assert "Reason: PROBE_DURABLE_STATE_INVALID" in nested_results[0].output
        assert len(invocations.read_text().splitlines()) == 1
    elif checkpoint_tamper == "running_marker":
        assert action_result.exit_code == 2
        assert "Reason: PROBE_TARGET_QUARANTINED" in nested_results[0].output
        assert "environment=quarantined" in nested_results[0].output
        assert len(invocations.read_text().splitlines()) == 1
        safety_state = next((tmp_path / ".ul").glob("probe-quarantine-*.json"))
        assert json.loads(safety_state.read_text())["status"] == "quarantined"
    else:
        assert action_result.exit_code == 2
        assert "Reason: PROBE_TARGET_QUARANTINED" in nested_results[0].output
        assert "environment=quarantined" in nested_results[0].output
        assert len(invocations.read_text().splitlines()) == 1
        resumed_action = runner.invoke(
            app,
            [
                "action",
                action_match.group(1),
                "--resolve-quarantine-after",
                "environment-reset",
            ],
        )
        assert resumed_action.exit_code == 2
        assert "Reason: PROBE_REPORT_FAILED" in nested_results[1].output
        assert (
            "Reusing durable smoke and evaluator preflight checkpoints" in nested_results[1].output
        )
        assert len(invocations.read_text().splitlines()) == 6


def test_paused_probe_action_resumes_after_terminal_trial_without_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "agent.py").write_text(
        "import json\n"
        "from pathlib import Path\n\n"
        "def run(value):\n"
        "    with Path('target-invocations.jsonl').open('a') as stream:\n"
        "        stream.write(json.dumps(value) + '\\n')\n"
        "    return {'action': 'lookup', 'ticket': 42}\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "examples.jsonl"
    dataset.write_text(
        '{"id":"case-1","input":"Return ticket 42.","output":{"action":"lookup","ticket":42}}\n',
        encoding="utf-8",
    )
    output = tmp_path / "evidence.jsonl"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UL_LIVE", "true")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")

    preflight_calls = 0

    async def clean_room_preflight(_settings: object) -> object:
        nonlocal preflight_calls
        preflight_calls += 1
        return _evaluator_preflight()

    async def load_clean_room_preflight(_output: Path, _settings: object) -> tuple[object, Path]:
        return _evaluator_preflight(), _output.with_name(f"{_output.name}.preflight.json")

    create_runtime = probe_module.create_campaign_progress_runtime
    runtime_calls = 0

    def create_trial_paused_runtime(**arguments: object) -> object:
        nonlocal runtime_calls
        runtime_calls += 1
        runtime = create_runtime(**arguments)
        if runtime_calls == 1:
            original_trial_terminal = runtime.tracker.trial_terminal

            def pause_after_original_trial(**terminal_arguments: object) -> None:
                original_trial_terminal(**terminal_arguments)
                unit = terminal_arguments["unit"]
                if unit.arm == "original":
                    runtime.control.request_pause()

            monkeypatch.setattr(runtime.tracker, "trial_terminal", pause_after_original_trial)
        return runtime

    monkeypatch.setattr(probe_module, "preflight_evaluator", clean_room_preflight)
    monkeypatch.setattr(probe_module, "load_evaluator_preflight", load_clean_room_preflight)
    monkeypatch.setattr(
        probe_module,
        "create_campaign_progress_runtime",
        create_trial_paused_runtime,
    )
    monkeypatch.setattr(
        campaign_runner_module,
        "create_semantic_model_deconstructor",
        lambda _settings: _CleanRoomSemanticModel(),
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
        ],
        input="y\ny\n",
    )

    assert result.exit_code == 130, result.output
    assert "Reason: PROBE_PAUSED_DURING_CAMPAIGN" in result.output
    assert "environment=reusable" in result.output
    assert "next_action=resume" in result.output
    action_match = re.search(
        r'next_argv=\["ul","action","([0-9a-f]{64})"\]',
        result.output,
    )
    assert action_match is not None
    invocations = tmp_path / "target-invocations.jsonl"
    assert len(invocations.read_text().splitlines()) == 2
    nested_results = []

    def invoke_trusted_cli(
        argv: tuple[str, ...],
        *,
        check: bool,
        cwd: str,
    ) -> SimpleNamespace:
        assert check is False
        assert Path(cwd) == tmp_path
        nested = runner.invoke(app, list(argv[4:]))
        nested_results.append(nested)
        return SimpleNamespace(returncode=nested.exit_code)

    monkeypatch.setattr(progress_action_module.subprocess, "run", invoke_trusted_cli)
    action_result = runner.invoke(app, ["action", action_match.group(1)])

    assert action_result.exit_code == 0, nested_results[0].output
    assert nested_results[0].exit_code == 0, nested_results[0].output
    assert "Reusing durable smoke and evaluator preflight checkpoints" in nested_results[0].output
    assert "UL run report" in nested_results[0].output
    assert "stage=terminal" in nested_results[0].output
    assert preflight_calls == 1
    assert len(invocations.read_text().splitlines()) == 3
    evidence_lines = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(evidence_lines) == 2
    assert evidence_lines[0]["record_type"] == "dataset_durable_run"
    assert evidence_lines[1]["execution_plan"]["repetitions"] == 1
    journal_records = [
        json.loads(line)
        for line in output.with_name(f"{output.name}.trials.jsonl").read_text().splitlines()
    ]
    completed_units = {
        record["unit"]["arm"] for record in journal_records if record["state"] == "completed"
    }
    assert completed_units == {"original", "probe"}


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
        ["probe", str(dataset), "--target", "customer_agent:run"],
        input="y\n",
    )

    assert result.exit_code == 2
    assert "Reason: PROBE_CONFIG_EXISTS" in result.output
    assert not (tmp_path / "target-invocations.jsonl").exists()
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
        ["probe", str(dataset), "--target", "customer_agent:run"],
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
        ],
        input="y\ny\n",
    )

    assert result.exit_code == 0, result.output
    plan = called["plan"]
    assert isinstance(plan, probe_module.DatasetCampaignPlan)
    assert plan.calls.baseline == 1
    assert plan.calls.variation == 1
    assert plan.calls.repetitions == 1
    assert called["report"] == (output, {})
    assert "Stronger confirmation:" in result.output


@pytest.mark.parametrize(
    ("failure_point", "expected_stage", "expected_reason"),
    (
        ("presentation", "evaluation", "PROBE_RESULT_PRESENTATION_FAILED"),
        ("report", "analysis", "PROBE_REPORT_FAILED"),
    ),
)
def test_post_campaign_failures_are_staged_without_raw_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
    expected_stage: str,
    expected_reason: str,
) -> None:
    _write_callable(tmp_path)
    dataset = _write_dataset(tmp_path)
    output = tmp_path / ".ul" / "runs" / "evidence.jsonl"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UL_LIVE", "true")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")

    def fake_campaign(**kwargs: object) -> tuple[DatasetEvaluationResult, ...]:
        created_output = kwargs["output"]
        assert isinstance(created_output, Path)
        created_output.parent.mkdir(parents=True, exist_ok=True)
        created_output.write_text("evidence\n", encoding="utf-8")
        return ()

    def presentation(*args: object, **kwargs: object) -> None:
        del args, kwargs
        if failure_point == "presentation":
            raise RuntimeError("private-presentation-detail")

    def report(*args: object, **kwargs: object) -> None:
        del args, kwargs
        if failure_point == "report":
            raise RuntimeError("private-report-detail")

    monkeypatch.setattr(probe_module, "_run_campaign", fake_campaign)
    monkeypatch.setattr(probe_module, "print_dataset_results", presentation)
    monkeypatch.setattr(probe_module, "report_evidence", report)
    result = runner.invoke(
        app,
        ["probe", str(dataset), "--target", "customer_agent:run", "--output", str(output)],
        input="y\ny\n",
    )

    assert result.exit_code == 2
    assert f"Stage: {expected_stage}" in result.output
    assert f"Reason: {expected_reason}" in result.output
    assert "private-" not in result.output
    assert "Target safe to reuse: yes" in result.output


def test_public_documentation_flow_runs_real_callable_campaign_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "agent.py").write_text(
        "def run(value):\n"
        "    return {'result': {'action': 'lookup', 'ticket': 42, "
        "'customer': {'email': 'private@example.test'}}}\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "interactions.jsonl"
    dataset.write_text(
        '{"id":"case-1","input":"Return the status for ticket 42.",'
        '"output":{"action":"lookup","ticket":42}}\n',
        encoding="utf-8",
    )
    config = _write_projected_callable_config(
        tmp_path,
        {
            "schema_version": "1.0.0",
            "complete_result": "/result",
            "private_json_pointers": ["/customer/email"],
        },
        target="agent:run",
    )
    output = tmp_path / ".ul" / "runs" / "probe-evidence.jsonl"
    semantic_model = _CleanRoomSemanticModel()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UL_LIVE", "true")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")

    async def clean_room_preflight(settings: object) -> object:
        del settings
        return _evaluator_preflight()

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
            str(config),
            "--output",
            str(output),
        ],
        input="y\ny\n",
    )

    assert result.exit_code == 0, result.output
    assert "Smoke target invocation succeeded" in result.output
    assert "Dataset evaluation" in result.output
    assert "UL run report" in result.output
    assert "Evidence type: dataset evaluation" in result.output
    assert "Stronger confirmation:" in result.output
    for stage in (
        "smoke",
        "preflight",
        "augmentation",
        "original",
        "probe",
        "evidence",
        "report",
        "terminal",
    ):
        assert f"stage={stage}" in result.output
    assert result.output.count(" next_action=") == 1
    assert "next_action=inspect_findings" in result.output
    action_match = re.search(
        r'next_argv=\["ul","action","([0-9a-f]{64})"\]',
        result.output,
    )
    assert action_match is not None
    assert str(output) not in result.output.split("next_action=", 1)[1]
    assert "target_calls=3" in result.output
    assert "environment_calls=3" in result.output
    assert "semantic_calls=unknown" in result.output
    assert "tokens=unknown" in result.output
    evidence_lines = [json.loads(line) for line in output.read_text().splitlines()]
    assert evidence_lines[0]["record_type"] == "dataset_durable_run"
    evidence = evidence_lines[1]
    assert evidence["execution_plan"]["repetitions"] == 1
    assert evidence["execution_plan"]["dataset_planned_target_calls"] == 2
    assert evidence["run_context"]["semantic_settings"]["provider"] == "openrouter"
    assert evidence["run_context"]["semantic_settings"]["model"]
    assert evidence["run_context"]["target"]["kind"] == "probe_target"
    assert evidence["run_context"]["target"]["receipt"]["confirmation_sha256"]
    assert (
        evidence["run_context"]["target"]["receipt"]["outcome_projection"]["complete_result"]
        == "/result"
    )
    assert (
        evidence["technical_details"]["baseline"]["trial_set"]["trials"][0]["execution_evidence"][
            "public_normalized_result"
        ]["customer"]["email"]
        == "[PRIVATE]"
    )
    assert (
        evidence["technical_details"]["baseline"]["trial_set"]["trials"][0]["execution_evidence"][
            "environment_id"
        ]
        == "projected-agent"
    )
    assert output.with_name(f"{output.name}.manifest.json").is_file()
    assert output.with_name(f"{output.name}.trials.jsonl").is_file()
    assert output.with_name(f"{output.name}.trials.jsonl.anchor.json").is_file()
    assert output.with_name(f"{output.stem}.augmentations.jsonl").is_file()


def test_campaign_projection_failure_retains_exact_reason_and_partial_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "agent.py").write_text(
        "from pathlib import Path\n\n"
        "def run(value):\n"
        "    count_path = Path('projection-call-count')\n"
        "    count = int(count_path.read_text()) + 1 if count_path.exists() else 1\n"
        "    count_path.write_text(str(count))\n"
        "    return {'result': {'action': 'lookup'}} if count == 1 else {'result': {}}\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "interactions.jsonl"
    dataset.write_text(
        '{"id":"case-1","input":"Return the status for ticket 42.",'
        '"output":{"action":"lookup","ticket":42}}\n',
        encoding="utf-8",
    )
    config = _write_projected_callable_config(
        tmp_path,
        {"schema_version": "1.0.0", "action": "/result/action"},
        target="agent:run",
    )
    output = tmp_path / ".ul" / "runs" / "projection-failure.jsonl"
    semantic_model = _CleanRoomSemanticModel()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UL_LIVE", "true")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")

    async def clean_room_preflight(settings: object) -> object:
        del settings
        return _evaluator_preflight()

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
        ["probe", str(dataset), "--target", str(config), "--output", str(output)],
        input="y\ny\n",
    )

    assert result.exit_code == 2, result.output
    assert "outcome field 'action' at selector '/result/action' does not resolve" in result.output
    assert "target execution completed" in result.output
    assert "Paid preparation and target work already occurred" in result.output
    assert f"partial evidence remains in {output}" in result.output
    assert "Target safe to reuse: no" in result.output
    assert "PROBE_EVALUATION_FAILED" not in result.output
    evidence = json.loads(output.read_text().splitlines()[1])
    failure = evidence["technical_details"]["baseline"]["trial_set"]["trials"][0]
    assert failure["lifecycle_failure"]["failed_phase"] == "outcome_projection"
    assert failure["lifecycle_failure"]["environment_state_may_remain"] is True
    assert "result evaluation was not run" in failure["inconclusive_reasons"][0]


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
        ["probe", str(dataset), "--target", str(config)],
        input="y\nn\n",
    )

    assert result.exit_code == 0, result.output
    assert "Response structure: dict;" in result.output
    assert "transport" not in result.output
    assert result.output.count("next_action=") == 1
    assert "next_action=diagnose" in result.output
    saved = json.loads((tmp_path / ".ul" / "probe.json").read_text())
    assert saved["target_kind"] == "command"
