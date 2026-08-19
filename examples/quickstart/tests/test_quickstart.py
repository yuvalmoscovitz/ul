from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
import zipfile
from collections.abc import Callable, Generator
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import typer
from typer.testing import CliRunner
from ul.dataset_invariants import JsonValuesEqualInvariant, load_dataset_invariant_suite
from ul.http_sandbox import JsonHttpSandboxConfig, JsonHttpSandboxConnection
from ul.sandbox import evaluation_case_from_inputs
from ul_cli.main import app

from examples.quickstart import run as quickstart
from examples.quickstart.defective_agent import create_server

_PROJECT_ROOT = Path(__file__).parents[3]
_QUICKSTART_DIRECTORY = Path(__file__).parents[1]
_VALID_REQUEST = {
    "case_id": "ul-case-00000000000000000000000000000000",
    "turn_id": "ul-case-00000000000000000000000000000000:turn-1",
    "request": {"message": "Pay AC-100."},
    "settings": {"mode": "sandbox"},
}


@contextmanager
def _running_server() -> Generator[tuple[str, ThreadingHTTPServer]]:
    server = create_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address)
    try:
        yield f"http://{host}:{port}", server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert not thread.is_alive()


def _post(base_url: str, payload: object, **kwargs: Any) -> httpx.Response:
    return httpx.post(f"{base_url}/execute", json=payload, timeout=2, trust_env=False, **kwargs)


def _actions(response: httpx.Response) -> list[dict[str, object]]:
    assert response.status_code == 200, response.text
    response_payload = cast(dict[str, Any], response.json())
    assert set(response_payload) == {"sandbox_id", "case_id", "turn_id", "result"}
    result = response_payload["result"]
    assert isinstance(result, dict)
    return [cast(dict[str, object], result)]


def test_server_reproduces_the_seeded_wrong_invoice() -> None:
    with _running_server() as (base_url, _server):
        original_actions = _actions(_post(base_url, _VALID_REQUEST))
        wrong_invoice_actions = _actions(
            _post(
                base_url,
                {
                    "case_id": "ul-case-00000000000000000000000000000000",
                    "turn_id": "ul-case-00000000000000000000000000000000:turn-1",
                    "request": {"message": "Pay pay AC-100."},
                    "settings": {"mode": "sandbox"},
                },
            )
        )

    assert original_actions == [
        {
            "action": "payment_committed",
            "payment_id": "pay-0001",
            "invoice_reference": "AC-100",
            "requested_invoice_reference": "AC-100",
            "amount": "12500",
            "currency": "USD",
            "source_bank_account_id": "bank-main",
            "idempotency_key": "invoice:AC-100:1",
        }
    ]
    assert wrong_invoice_actions == [
        {
            **original_actions[0],
            "invoice_reference": "AC-101",
            "idempotency_key": "invoice:AC-101:1",
        }
    ]


def test_server_starts_every_request_from_identical_fresh_state() -> None:
    with _running_server() as (base_url, _server):
        responses = [_actions(_post(base_url, _VALID_REQUEST)) for _ in range(3)]
        wrong_invoice_responses = [
            _actions(
                _post(
                    base_url,
                    {
                        "case_id": "ul-case-00000000000000000000000000000000",
                        "turn_id": "ul-case-00000000000000000000000000000000:turn-1",
                        "request": {"message": "Pay pay AC-100."},
                        "settings": {"mode": "sandbox"},
                    },
                )
            )
            for _ in range(3)
        ]

    assert responses[0] == responses[1] == responses[2]
    assert wrong_invoice_responses[0] == wrong_invoice_responses[1] == wrong_invoice_responses[2]


@pytest.mark.asyncio
async def test_stateful_target_adapter_resets_executes_and_snapshots() -> None:
    with _running_server() as (base_url, _server):
        config = JsonHttpSandboxConfig.model_validate(quickstart.load_target_template(base_url))
        async with JsonHttpSandboxConnection.from_config(
            config,
            sandbox_confirmed=True,
            allow_insecure_http=True,
            max_sandbox_api_calls=6,
        ) as target:
            case = evaluation_case_from_inputs(
                case_id="ul-case-00000000000000000000000000000000",
                raw_inputs=("Pay AC-100.",),
                max_sandbox_api_calls=6,
                timeout_seconds=30,
            )
            output = await target.execute(case)

    assert output.lifecycle.terminal_status == "succeeded"
    raw_output = cast(dict[str, object], output.turns[0].response)
    assert raw_output["invoice_reference"] == "AC-100"
    assert output.turns[0].state_snapshot == raw_output
    assert output.lifecycle.completed_phases == (
        "reset",
        "setup",
        "initial_snapshot",
        "execute_turn",
        "snapshot",
        "cleanup_reset",
    )


def test_sandbox_check_runs_against_the_bundled_quickstart(tmp_path: Path) -> None:
    probe = "Pay AC-100."
    with _running_server() as (base_url, _server):
        target_path = tmp_path / "target.json"
        target_path.write_text(
            json.dumps(quickstart.load_target_template(base_url)), encoding="utf-8"
        )
        result = CliRunner().invoke(
            app,
            [
                "sandbox",
                "check",
                str(target_path),
                "--probe",
                probe,
                "--allow-sandbox-network-egress",
                "--confirm-isolated-sandbox",
                "--confirm-harmless-probe",
                "--allow-insecure-http",
                "--json",
            ],
        )

    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["status"] == "ready"
    assert summary["sandbox_id"] == "quickstart-accounts-payable"
    assert summary["sandbox_api_call_budget"] == 6
    assert summary["completed_phases"] == [
        "reset",
        "setup",
        "initial_snapshot",
        "execute_turn",
        "snapshot",
        "cleanup_reset",
    ]
    assert summary["state_observation_authority"] == "sandbox_self_reported"
    assert summary["ul_semantic_model_calls"] == 0
    assert probe not in result.output
    assert "payment_committed" not in result.output


@pytest.mark.parametrize(
    ("body", "content_type", "path"),
    [
        (b"{}", "text/plain", "/execute"),
        (b"not json", "application/json", "/execute"),
        (
            json.dumps({"request": {"message": "Pay AC-100."}}).encode(),
            "application/json",
            "/execute",
        ),
        (
            json.dumps({**_VALID_REQUEST, "extra": True}).encode(),
            "application/json",
            "/execute",
        ),
        (json.dumps(_VALID_REQUEST).encode(), "application/json", "/wrong"),
    ],
)
def test_server_rejects_wrong_path_content_or_shape(
    body: bytes, content_type: str, path: str
) -> None:
    with _running_server() as (base_url, _server):
        response = httpx.post(
            f"{base_url}{path}",
            content=body,
            headers={"Content-Type": content_type},
            timeout=2,
            trust_env=False,
        )

    assert 400 <= response.status_code < 500


def test_server_rejects_oversized_and_invalid_values() -> None:
    with _running_server() as (base_url, _server):
        oversized = _post(
            base_url,
            {
                "request": {"message": "x" * 100_001},
                "settings": {"mode": "sandbox"},
            },
        )
        wrong_mode = _post(
            base_url,
            {
                "request": {"message": "Pay AC-100."},
                "settings": {"mode": "production"},
            },
        )

    assert 400 <= oversized.status_code < 500
    assert 400 <= wrong_mode.status_code < 500


def test_quickstart_target_contains_no_authentication_mapping() -> None:
    target = json.loads((_QUICKSTART_DIRECTORY / "target.json").read_text(encoding="utf-8"))

    assert target["headers_from_env"] == {}
    assert target["version"] == 3
    assert target["execute_turn"]["request_json_template"] == {
        "case_id": "{{case_id}}",
        "turn_id": "{{turn_id}}",
        "request": {"message": "{{input}}"},
        "settings": {"mode": "sandbox"},
    }
    assert target["execute_turn"]["response_json_pointer"] == "/result"
    assert target["snapshot"]["response_json_pointer"] == "/state"


def test_quickstart_invariant_uses_declared_committed_state_fields() -> None:
    suite = load_dataset_invariant_suite(_QUICKSTART_DIRECTORY / "invariants.json")

    assert suite.observation_source == "target_output"
    assert suite.observation_authority == "committed_state_snapshot"
    assert len(suite.rules) == 1
    rule = suite.rules[0]
    assert isinstance(rule, JsonValuesEqualInvariant)
    assert rule.id == "committed-invoice-matches-request"
    assert rule.left_pointer == "/invoice_reference"
    assert rule.right_pointer == "/requested_invoice_reference"


def _confirmed_evidence() -> dict[str, Any]:
    stable_observations = {
        "requested_repetitions": 3,
        "stability": "stable",
        "observed_repetitions": 3,
        "inconclusive_repetitions": 0,
        "outcome_group_count": 1,
    }
    invariant_trial = {
        "repetition": 1,
        "status": "satisfied",
        "reason_code": "values_equal",
        "left_pointer": "/invoice_reference",
        "right_pointer": "/requested_invoice_reference",
        "resolved_values": {"left": "AC-100", "right": "AC-100"},
    }
    baseline_rule = {
        "rule_type": "json_values_equal",
        "rule_id": "committed-invoice-matches-request",
        "rule_version": "1.0.0",
        "description": (
            "The committed invoice reference must match the requested invoice reference."
        ),
        "severity": "high",
        "status": "satisfied",
        "reason_code": "all_trials_satisfied",
        "trials": [
            invariant_trial,
            {**invariant_trial, "repetition": 2},
            {**invariant_trial, "repetition": 3},
        ],
    }
    violated_trial = {
        **invariant_trial,
        "status": "violated",
        "reason_code": "values_differ",
        "resolved_values": {"left": "AC-101", "right": "AC-100"},
    }
    variation_rule = {
        **baseline_rule,
        "status": "violated",
        "reason_code": "one_or_more_trials_violated",
        "trials": [
            violated_trial,
            {**violated_trial, "repetition": 2},
            {**violated_trial, "repetition": 3},
        ],
    }
    return {
        "current_baseline": {"observations": stable_observations},
        "cases": [
            {
                "operator_id": "input.surface.disfluency_repeat",
                "variation_accepted": True,
                "status": "REPEATABLE DIFFERENCE — REVIEW",
                "observations": stable_observations,
                "findings": [
                    {
                        "category": "changed_grounded_effect_argument",
                        "reference_effects": [
                            {
                                "predicate": "payment_committed",
                                "fields": {"invoice_reference": "AC-100"},
                            }
                        ],
                        "observed_effects": [
                            {
                                "predicate": "payment_committed",
                                "fields": {"invoice_reference": "AC-101"},
                            }
                        ],
                    }
                ],
            }
        ],
        "invariant_evaluation": {
            "interaction_id": "quickstart-ap-1",
            "suite_sha256": "0" * 64,
            "observation_source": "target_output",
            "observation_authority": "committed_state_snapshot",
            "baseline": {"arm": "baseline", "operator_id": None, "rules": [baseline_rule]},
            "variations": [
                {
                    "arm": "variation",
                    "operator_id": "input.surface.disfluency_repeat",
                    "rules": [variation_rule],
                }
            ],
        },
    }


def test_runner_uses_safe_argv_minimal_environment_private_artifacts_and_cleans_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(quickstart, "_PROJECT_DIRECTORY", tmp_path)
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-only-secret")
    monkeypatch.setenv("UL_DATASET_LIVE_CALLS", "true")
    monkeypatch.setenv("UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING", "true")
    monkeypatch.setenv("UL_DATASET_MODEL", "untrusted/model-override")
    monkeypatch.setenv("UL_DATASET_RENDER_MODEL", "untrusted/render-override")
    monkeypatch.setenv("UL_DATASET_EQUIVALENCE_MODEL", "untrusted/checker-override")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-be-forwarded")
    observed_endpoint = ""
    evidence_path: Path | None = None

    def run_cli(
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        check: bool,
        shell: bool,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal observed_endpoint, evidence_path
        assert command[:4] == [sys.executable, "-m", "ul_cli.main", "dataset"]
        assert all(isinstance(argument, str) for argument in command)
        assert cwd == _QUICKSTART_DIRECTORY
        assert check is False
        assert shell is False
        assert env == {
            "OPEN_ROUTER_API_KEY": "test-only-secret",
            "UL_DATASET_LIVE_CALLS": "true",
            "UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING": "true",
            "UL_DATASET_MODEL": "x-ai/grok-4.6",
            "UL_DATASET_RENDER_MODEL": "x-ai/grok-4.6",
            "UL_DATASET_EQUIVALENCE_MODEL": "x-ai/grok-4.6",
        }
        target_config_path = Path(command[command.index("--sandbox-config") + 1])
        assert Path(command[command.index("--invariants") + 1]) == (
            _QUICKSTART_DIRECTORY / "invariants.json"
        )
        evidence_path = Path(command[command.index("--output") + 1])
        assert stat.S_IMODE(target_config_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(target_config_path.parent.stat().st_mode) == 0o700
        target = json.loads(target_config_path.read_text(encoding="utf-8"))
        assert "test-only-secret" not in json.dumps(target)
        assert "test-only-secret" not in command
        observed_endpoint = target["execute_turn"]["url"]
        assert _actions(_post(observed_endpoint.rsplit("/", 1)[0], _VALID_REQUEST))
        descriptor = os.open(evidence_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as evidence_file:
            json.dump(_confirmed_evidence(), evidence_file)
            evidence_file.write("\n")
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(quickstart.subprocess, "run", run_cli)

    quickstart.main()

    assert evidence_path is not None
    assert evidence_path.exists()
    assert stat.S_IMODE(evidence_path.stat().st_mode) == 0o600
    assert not (evidence_path.parent / "target.json").exists()
    with pytest.raises(httpx.ConnectError):
        _post(observed_endpoint.rsplit("/", 1)[0], _VALID_REQUEST)


def test_live_environment_accepts_ul_live_and_forwards_granular_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable_name in (
        "UL_LIVE",
        "UL_DATASET_LIVE_CALLS",
        "UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING",
    ):
        monkeypatch.delenv(variable_name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-only-secret")
    monkeypatch.setenv("UL_LIVE", "true")

    subprocess_environment = cast(
        Callable[..., dict[str, str]], quickstart.__dict__["_subprocess_environment"]
    )
    environment = subprocess_environment(dry_run=False)

    assert environment["OPEN_ROUTER_API_KEY"] == "test-only-secret"
    assert environment["UL_DATASET_LIVE_CALLS"] == "true"
    assert environment["UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING"] == "true"
    assert "UL_LIVE" not in environment


def test_live_environment_respects_granular_false_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-only-secret")
    monkeypatch.setenv("UL_LIVE", "true")
    monkeypatch.setenv("UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING", "false")

    subprocess_environment = cast(
        Callable[..., dict[str, str]], quickstart.__dict__["_subprocess_environment"]
    )
    with pytest.raises(ValueError, match="ALLOW_EXTERNAL_DATA_PROCESSING"):
        subprocess_environment(dry_run=False)


def test_live_environment_supports_keyless_loopback_openai_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UL_DATASET_SEMANTIC_PROVIDER", "openai-compatible")
    monkeypatch.setenv("UL_DATASET_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("UL_DATASET_OPENAI_PROVIDER_ID", "local-vllm")
    monkeypatch.setenv("UL_DATASET_MODEL", "local-model")
    monkeypatch.setenv("UL_LIVE", "true")
    monkeypatch.delenv("UL_DATASET_OPENAI_API_KEY", raising=False)

    subprocess_environment = cast(
        Callable[..., dict[str, str]], quickstart.__dict__["_subprocess_environment"]
    )
    environment = subprocess_environment(dry_run=False)

    assert environment == {
        "UL_DATASET_SEMANTIC_PROVIDER": "openai-compatible",
        "UL_DATASET_OPENAI_PROVIDER_ID": "local-vllm",
        "UL_DATASET_OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
        "UL_DATASET_MODEL": "local-model",
        "UL_DATASET_RENDER_MODEL": "local-model",
        "UL_DATASET_EQUIVALENCE_MODEL": "local-model",
        "UL_DATASET_LIVE_CALLS": "true",
        "UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING": "true",
    }


def test_runner_accepts_only_the_exact_stable_wrong_invoice_finding(tmp_path: Path) -> None:
    confirms_repeatable_wrong_invoice = cast(
        Callable[[Path], bool],
        getattr(quickstart, "_evidence_confirms_" + "repeatable_wrong_invoice"),
    )
    evidence_path = tmp_path / "evidence.jsonl"
    evidence = _confirmed_evidence()
    evidence_path.write_text(json.dumps(evidence) + "\n", encoding="utf-8")
    assert confirms_repeatable_wrong_invoice(evidence_path)

    case = evidence["cases"][0]
    assert isinstance(case, dict)
    changes: tuple[tuple[str, object], ...] = (
        ("status", "NO OBSERVED DIFFERENCE"),
        ("variation_accepted", False),
        ("operator_id", "input.surface.rephrase"),
        ("findings", []),
        (
            "findings",
            [
                {"category": "changed_grounded_effect_argument"},
                {"category": "unexpected_effect"},
            ],
        ),
    )
    for key, value in changes:
        changed_evidence = json.loads(json.dumps(evidence))
        changed_evidence["cases"][0][key] = value
        evidence_path.write_text(json.dumps(changed_evidence) + "\n", encoding="utf-8")
        assert not confirms_repeatable_wrong_invoice(evidence_path)

    for effect_group, invoice_reference in (
        ("reference_effects", "AC-999"),
        ("observed_effects", "AC-100"),
    ):
        changed_evidence = json.loads(json.dumps(evidence))
        changed_evidence["cases"][0]["findings"][0][effect_group][0]["fields"][
            "invoice_reference"
        ] = invoice_reference
        evidence_path.write_text(json.dumps(changed_evidence) + "\n", encoding="utf-8")
        assert not confirms_repeatable_wrong_invoice(evidence_path)

    invariant_changes: tuple[tuple[str, str], ...] = (
        ("baseline", "violated"),
        ("variation", "satisfied"),
        ("rule_id", "different-rule"),
    )
    for subject, value in invariant_changes:
        changed_evidence = json.loads(json.dumps(evidence))
        invariant_evaluation = changed_evidence["invariant_evaluation"]
        if subject == "baseline":
            invariant_evaluation["baseline"]["rules"][0]["status"] = value
        elif subject == "variation":
            invariant_evaluation["variations"][0]["rules"][0]["status"] = value
        else:
            invariant_evaluation["variations"][0]["rules"][0][subject] = value
        evidence_path.write_text(json.dumps(changed_evidence) + "\n", encoding="utf-8")
        assert not confirms_repeatable_wrong_invoice(evidence_path)

    evidence_path.write_text("not json\n", encoding="utf-8")
    assert not confirms_repeatable_wrong_invoice(evidence_path)


def test_runner_propagates_nonconfirmation_and_execution_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(quickstart, "_PROJECT_DIRECTORY", tmp_path)
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-only-secret")
    monkeypatch.setenv("UL_DATASET_LIVE_CALLS", "true")
    monkeypatch.setenv("UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING", "true")

    def no_finding(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        evidence_path = Path(command[command.index("--output") + 1])
        evidence_path.write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(quickstart.subprocess, "run", no_finding)
    with pytest.raises(typer.Exit) as nonconfirmation:
        quickstart.main()
    assert nonconfirmation.value.exit_code == 1

    def execution_error(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr(quickstart.subprocess, "run", execution_error)
    with pytest.raises(typer.Exit) as error:
        quickstart.main()
    assert error.value.exit_code == 2


def test_dry_run_does_not_construct_the_target_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(quickstart, "_PROJECT_DIRECTORY", tmp_path)

    def unexpected_server() -> object:
        raise AssertionError("dry-run constructed the target server")

    monkeypatch.setattr(quickstart, "create_server", unexpected_server)
    quickstart.main(dry_run=True)


def test_dry_run_is_a_real_subprocess_and_needs_no_api_key(tmp_path: Path) -> None:
    environment = {
        "PATH": os.environ["PATH"],
        "PYTHONPATH": os.pathsep.join(
            [
                str(_PROJECT_ROOT / "core/src"),
                str(_PROJECT_ROOT / "sdk/src"),
                str(_PROJECT_ROOT / "cli/src"),
            ]
        ),
        "HOME": str(tmp_path),
    }

    completed = subprocess.run(
        [sys.executable, "-m", "examples.quickstart.run", "--dry-run"],
        cwd=_PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        shell=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "No model or target requests sent." in completed.stdout
    assert "OPEN_ROUTER_API_KEY" not in environment


def test_sandbox_check_is_a_real_subprocess_and_needs_no_api_key() -> None:
    environment = {
        "PATH": os.environ["PATH"],
        "PYTHONPATH": os.pathsep.join(
            [
                str(_PROJECT_ROOT / "core/src"),
                str(_PROJECT_ROOT / "sdk/src"),
                str(_PROJECT_ROOT / "cli/src"),
            ]
        ),
    }

    completed = subprocess.run(
        [sys.executable, "-m", "examples.quickstart.run", "--sandbox-check"],
        cwd=_PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        shell=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Sandbox check: READY" in completed.stdout
    assert "No API key or UL semantic-model calls used." in completed.stdout
    assert "OPEN_ROUTER_API_KEY" not in environment


def test_built_wheel_contains_the_complete_quickstart(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(tmp_path)],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        shell=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheel_path = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
    assert {
        "examples/quickstart/__init__.py",
        "examples/quickstart/README.md",
        "examples/quickstart/dataset.jsonl",
        "examples/quickstart/defective_agent.py",
        "examples/quickstart/invariants.json",
        "examples/quickstart/run.py",
        "examples/quickstart/target.json",
    } <= names
    assert not any(name.startswith("examples/quickstart/tests/") for name in names)


@pytest.mark.skipif(
    os.environ.get("UL_QUICKSTART_LIVE_E2E") != "true",
    reason="set UL_QUICKSTART_LIVE_E2E=true and OPEN_ROUTER_API_KEY to run the paid live demo",
)
def test_exact_documented_quickstart_command_finds_stable_wrong_invoice() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "examples.quickstart.run"],
        cwd=_PROJECT_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        shell=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (
        "stable 3/3 wrong-invoice action and the customer rule failed"
        in completed.stdout.casefold()
    )
