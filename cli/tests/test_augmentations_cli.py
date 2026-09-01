import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from typer.testing import CliRunner
from ul_cli import augmentations
from ul_cli.dataset.evaluation import command as dataset_command
from ul_cli.dataset.evaluation import runner as dataset_runner
from ul_cli.main import app

runner = CliRunner()


def _write_environment_config(path: Path, *, timeout_after_commit: bool = False) -> None:
    config = {
        "version": 5,
        "environment_id": "augmentation-readiness-test",
        "headers_from_env": {},
        "reset": {
            "url": "https://environment.example.test/reset",
            "request_json_template": {"case_id": "{{case_id}}"},
            "case_id_json_pointer": "/case_id",
            "environment_id_json_pointer": "/environment_id",
            "generation_json_pointer": "/generation",
            "clean_state_json_pointer": "/clean",
            "clean_state_value": True,
        },
        "execute_turn": {
            "url": "https://environment.example.test/execute",
            "request_json_template": {
                "case_id": "{{case_id}}",
                "turn_id": "{{turn_id}}",
                "input": "{{input}}",
            },
            "response_json_pointer": "/response",
            "case_id_json_pointer": "/case_id",
            "turn_id_json_pointer": "/turn_id",
            "environment_id_json_pointer": "/environment_id",
        },
        "snapshot": {
            "url": "https://environment.example.test/snapshot",
            "request_json_template": {
                "case_id": "{{case_id}}",
                "turn_id": "{{turn_id}}",
            },
            "response_json_pointer": "/state",
            "case_id_json_pointer": "/case_id",
            "turn_id_json_pointer": "/turn_id",
            "environment_id_json_pointer": "/environment_id",
        },
    }
    if timeout_after_commit:
        config["timeout_after_commit"] = {
            "operator_id": "environment.tool.timeout_after_commit",
            "version": "1.0.0",
            "url": "https://environment.example.test/timeout-after-commit",
        }
    path.write_text(json.dumps(config), encoding="utf-8")


def _initialize_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    timeout_after_commit: bool = False,
    invariants: bool = False,
) -> None:
    dataset_path = tmp_path / "interactions.jsonl"
    environment_path = tmp_path / "environment.json"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "interaction-1",
                "input": "Transfer 100 to Alice.",
                "output": {"status": "recorded"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_environment_config(environment_path, timeout_after_commit=timeout_after_commit)
    arguments = [
        "init",
        str(dataset_path),
        "--environment-config",
        str(environment_path),
        "--allow-environment-network",
        "--confirm-test-environment",
    ]
    if invariants:
        invariants_path = tmp_path / "invariants.json"
        invariants_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.1.0",
                    "observation_source": "target_output",
                    "observation_authority": "committed_state_snapshot",
                    "rules": [
                        {
                            "type": "json_value_equals_literal",
                            "id": "effect-count",
                            "version": "1.0.0",
                            "description": "Exactly one effect must be committed.",
                            "severity": "critical",
                            "value_pointer": "/effect_count",
                            "literal": 1,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        arguments.extend(("--invariants", str(invariants_path)))
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output


def _enable_semantic_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UL_DATASET_LIVE_CALLS", "true")
    monkeypatch.setenv("UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING", "true")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")


def _initialize_isolated_response_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_path = tmp_path / "interactions.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "interaction-1",
                "input": "Transfer 100 to Alice.",
                "output": {"status": "recorded"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "init",
            str(dataset_path),
            "--environment-url",
            "https://agent.example.test/v1/chat/completions",
            "--adapter-tier",
            "isolated-response",
            "--allow-environment-network",
            "--confirm-test-environment",
            "--confirm-request-isolation",
            "--confirm-safe-test-target",
            "--isolated-preset",
            "openai-chat",
            "--agent-model",
            "customer-agent-v1",
        ],
    )
    assert result.exit_code == 0, result.output


def _planned_augmentation(payload: dict[str, object], augmentation_id: str) -> dict[str, object]:
    augmentations = payload["augmentations"]
    assert isinstance(augmentations, list)
    return next(
        item
        for item in augmentations
        if isinstance(item, dict)
        and isinstance(item.get("ref"), dict)
        and item["ref"].get("id") == augmentation_id
    )


def _reason_codes(item: dict[str, object]) -> set[str]:
    reasons = item["reasons"]
    assert isinstance(reasons, list)
    return {
        reason["code"]
        for reason in reasons
        if isinstance(reason, dict) and isinstance(reason.get("code"), str)
    }


def test_list_shows_every_full_copyable_id_at_eighty_columns() -> None:
    result = CliRunner().invoke(app, ["augmentations", "list"], terminal_width=80)

    assert result.exit_code == 0
    assert "environment.state.existing_partial_operation@1.0.0" in result.output
    assert "conversation.retry_after_successful_commit@1.0.0" in result.output
    assert "…" not in result.output
    assert result.output.count("environment.tool.timeout_after_commit@1.0.0") == 1
    assert "Lose acknowledgement after a consequential effect commits." in result.output


def test_list_json_is_stable_sorted_and_complete() -> None:
    result = CliRunner().invoke(app, ["augmentations", "list", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema_version"] == "1.0.0"
    assert len(payload["augmentations"]) == 22
    references = [(item["ref"]["id"], item["ref"]["version"]) for item in payload["augmentations"]]
    assert references == sorted(references)
    assert all(isinstance(item["cli_available"], bool) for item in payload["augmentations"])
    assert {item["surface"] for item in payload["augmentations"]} == {
        "human_behavior",
        "task_semantics",
        "conversation_workflow",
        "world_business_state",
        "tool_execution",
        "trust_policy_authorization",
    }
    assert all(item["implementation_status"] == "implemented" for item in payload["augmentations"])
    assert all(item["qualification_status"] == "not_qualified" for item in payload["augmentations"])


def test_guide_and_surface_filter_make_the_library_navigable() -> None:
    guide = runner.invoke(app, ["augmentations", "guide"])

    assert guide.exit_code == 0, guide.output
    assert "Human behavior" in guide.output
    assert "Task semantics" in guide.output
    assert "Conversation and workflow" in guide.output
    assert "World and business state" in guide.output
    assert "Tool and execution" in guide.output
    assert "Trust, policy, and authorization" in guide.output
    assert guide.output.count("@1.0.0 [implemented; not_qualified]") == 22

    filtered = runner.invoke(
        app,
        ["augmentations", "list", "--surface", "tool-execution", "--json"],
    )
    assert filtered.exit_code == 0, filtered.output
    payload = json.loads(filtered.output)
    assert {item["ref"]["id"] for item in payload["augmentations"]} == {
        "environment.tool.stale_observation",
        "environment.tool.timeout_before_commit",
        "environment.tool.timeout_after_commit",
    }


def test_list_filters_by_scope_mode_and_runnability() -> None:
    result = CliRunner().invoke(
        app,
        [
            "augmentations",
            "list",
            "--scope",
            "environment",
            "--mode",
            "environment_fault",
            "--cli-only",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert [item["ref"]["id"] for item in payload["augmentations"]] == [
        "environment.tool.timeout_after_commit"
    ]


def test_list_rejects_unknown_filters_and_allows_no_matches() -> None:
    invalid = CliRunner().invoke(app, ["augmentations", "list", "--scope", "unknown"])
    empty = CliRunner().invoke(
        app,
        ["augmentations", "list", "--scope", "input", "--mode", "environment_fault"],
    )

    assert invalid.exit_code == 2
    assert "choose one of" in invalid.output
    assert empty.exit_code == 0
    assert "No augmentations matched." in empty.output


def test_show_reports_cli_fault_requirements() -> None:
    result = CliRunner().invoke(
        app,
        ["augmentations", "show", "environment.tool.timeout_after_commit@1.0.0"],
    )

    assert result.exit_code == 0
    assert "CLI execution available: yes" in result.output
    assert "Mode: scenario_materialization" in result.output
    assert "Mode: environment_fault" in result.output
    assert "Execution owner: SDK augmentation registry" in result.output
    assert "Execution owner: stress CLI" in result.output
    assert "environment capability environment.tool.timeout_after_commit@1.0.0" in result.output
    assert "0 model calls, 0 environment calls, 0 network requests" in result.output


def test_show_reports_materializer_only_without_false_execution_claim() -> None:
    result = CliRunner().invoke(
        app,
        ["augmentations", "show", "environment.state.change_between_read_write"],
    )

    assert result.exit_code == 0
    assert "CLI execution available: no" in result.output
    assert "Execution owner: SDK augmentation registry" in result.output
    assert "CLI command: unavailable" in result.output


def test_show_reports_dataset_execution_requirements_without_requiring_invariants() -> None:
    result = CliRunner().invoke(app, ["augmentations", "show", "input.style.terse"])

    assert result.exit_code == 0
    assert "ul dataset evaluate --operator input.style.terse@1.0.0" in result.output
    assert "test environment" in result.output
    assert "committed-state observation" in result.output
    assert "semantic model" in result.output
    assert "customer evaluator" not in result.output
    assert "Applicability: broad" in result.output
    assert "nonempty user input" in result.output


@pytest.mark.parametrize("operator_id", ["input.tone.angry", "input.tone.argumentative"])
def test_show_reports_tones_as_response_only(operator_id: str) -> None:
    result = CliRunner().invoke(app, ["augmentations", "show", operator_id])

    assert result.exit_code == 0
    assert "test environment" in result.output
    assert "semantic model" in result.output
    assert "committed-state observation" not in result.output


def test_show_labels_conditional_operator_and_explains_its_rule() -> None:
    result = CliRunner().invoke(app, ["augmentations", "show", "input.intent.self_correction"])

    assert result.exit_code == 0
    assert "Applicability: conditional" in result.output
    assert "numeric, monetary, date, or duration" in result.output


def test_mode_and_cli_filters_do_not_match_different_bindings() -> None:
    result = CliRunner().invoke(
        app,
        [
            "augmentations",
            "list",
            "--mode",
            "scenario_materialization",
            "--cli-only",
        ],
    )

    assert result.exit_code == 0
    assert "No augmentations matched." in result.output


def test_show_json_and_unknown_reference_contract() -> None:
    shown = CliRunner().invoke(app, ["augmentations", "show", "input.surface.rephrase", "--json"])
    missing = CliRunner().invoke(
        app, ["augmentations", "show", "input.surface.rephrase@2.0.0", "--json"]
    )

    assert shown.exit_code == 0
    payload = json.loads(shown.output)
    assert payload["schema_version"] == "1.0.0"
    assert payload["augmentation"]["ref"] == {
        "id": "input.surface.rephrase",
        "version": "1.0.0",
    }
    assert payload["augmentation"]["bindings"][0]["projection"] == {
        "reads": ["structured_input", "conversation"],
        "writes": ["structured_input", "conversation"],
    }
    assert missing.exit_code == 2
    assert "unknown augmentation" in missing.output


def test_show_rejects_terminal_control_sequences_without_reflecting_them() -> None:
    reference = "input.surface.rephrase\x1b[31m"

    result = CliRunner().invoke(app, ["augmentations", "show", reference])

    assert result.exit_code == 2
    assert "augmentation reference must be ID or ID@VERSION" in result.output
    assert reference not in result.output


def test_plan_json_is_stable_complete_and_project_aware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _initialize_project(tmp_path, monkeypatch)
    _enable_semantic_execution(monkeypatch)

    first = runner.invoke(app, ["augmentations", "plan", "--json"])
    second = runner.invoke(app, ["augmentations", "plan", "--json"])

    assert first.exit_code == 0, first.output
    assert first.output == second.output
    payload = json.loads(first.output)
    assert payload["schema_version"] == "1.0.0"
    assert payload["project"] == {"status": "ready", "reason": None}
    assert payload["summary"] == {"ready": 11, "blocked": 3, "manual": 8}
    assert payload["inspection"] == {
        "model_calls": 0,
        "environment_calls": 0,
        "network_requests": 0,
    }
    assert len(payload["augmentations"]) == 22
    references = [(item["ref"]["id"], item["ref"]["version"]) for item in payload["augmentations"]]
    assert references == sorted(references)
    assert {item["status"] for item in payload["augmentations"]} <= {
        "ready",
        "blocked",
        "manual",
    }
    assert all(item["reasons"] for item in payload["augmentations"])

    rephrase = _planned_augmentation(payload, "input.surface.rephrase")
    assert rephrase["status"] == "ready"
    assert rephrase["command"] == "ul run --operator input.surface.rephrase@1.0.0"
    assert rephrase["projection"] == {
        "reads": ["structured_input", "conversation"],
        "writes": ["structured_input", "conversation"],
    }
    assert _reason_codes(rephrase) == {"requirements_satisfied"}

    ambiguity = _planned_augmentation(payload, "conversation.ambiguity")
    assert ambiguity["status"] == "manual"
    assert ambiguity["command"] is None
    assert "cli_unavailable" in _reason_codes(ambiguity)

    timeout = _planned_augmentation(payload, "environment.tool.timeout_after_commit")
    assert timeout["status"] == "blocked"
    assert timeout["command"] == (
        "ul stress timeout-after-commit CASE.json --environment-config ENVIRONMENT.json "
        "--invariants INVARIANTS.json --dry-run"
    )
    assert {
        "customer_evaluator_unavailable",
        "environment_capability_missing",
    } <= _reason_codes(timeout)


def test_plan_reads_declared_capabilities_without_constructing_external_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _initialize_project(
        tmp_path,
        monkeypatch,
        timeout_after_commit=True,
        invariants=True,
    )
    _enable_semantic_execution(monkeypatch)

    def unexpected_external_client(*args: object, **kwargs: object) -> None:
        raise AssertionError("readiness planning constructed an external client")

    monkeypatch.setattr(
        dataset_runner,
        "create_semantic_model_deconstructor",
        unexpected_external_client,
    )
    monkeypatch.setattr(
        dataset_command.JsonHttpEnvironmentConnection,
        "from_config",
        unexpected_external_client,
    )

    result = runner.invoke(app, ["augmentations", "plan", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"] == {"ready": 11, "blocked": 0, "manual": 11}
    timeout = _planned_augmentation(payload, "environment.tool.timeout_after_commit")
    assert timeout["status"] == "manual"
    assert timeout["command"] == (
        "ul stress timeout-after-commit CASE.json --environment-config ENVIRONMENT.json "
        "--invariants INVARIANTS.json --dry-run"
    )
    assert "stress_case_not_configured" in _reason_codes(timeout)
    assert "environment_capability_missing" not in _reason_codes(timeout)


def test_tones_are_ready_for_an_isolated_response_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _initialize_isolated_response_project(tmp_path, monkeypatch)
    _enable_semantic_execution(monkeypatch)

    result = runner.invoke(app, ["augmentations", "plan", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    for operator_id in ("input.tone.angry", "input.tone.argumentative"):
        planned = _planned_augmentation(payload, operator_id)
        assert planned["status"] == "ready"
        assert "environment_state_observation_unsupported" not in _reason_codes(planned)


def test_bundle_plan_uses_configured_project_runtime_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _initialize_project(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        ["augmentations", "bundles", "plan", "everyday-customers", "--json"],
    )

    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["totals"]["cases"] == 1
    assert plan["totals"]["planned_probes"] == 0
    assert plan["totals"]["blocked_probes"] == 3
    assert all(probe["status"] == "blocked" for probe in plan["probes"])
    assert all(
        any("Semantic model calls are disabled" in reason for reason in probe["reasons"])
        for probe in plan["probes"]
    )
    assert all("operator is not qualified" in probe["reasons"] for probe in plan["probes"])


def test_plan_without_a_project_still_classifies_every_catalog_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["augmentations", "plan", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["project"] == {
        "status": "missing",
        "reason": "No UL project found; run 'ul init' first.",
    }
    assert payload["summary"] == {"ready": 0, "blocked": 15, "manual": 7}
    assert len(payload["augmentations"]) == 22
    rephrase = _planned_augmentation(payload, "input.surface.rephrase")
    assert rephrase["status"] == "blocked"
    assert _reason_codes(rephrase) == {"project_not_configured"}
    ambiguity = _planned_augmentation(payload, "conversation.ambiguity")
    assert ambiguity["status"] == "manual"
    assert "cli_unavailable" in _reason_codes(ambiguity)


def test_plan_human_output_is_actionable_and_attests_zero_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _initialize_project(tmp_path, monkeypatch)
    _enable_semantic_execution(monkeypatch)

    result = runner.invoke(app, ["augmentations", "plan"], terminal_width=80)

    assert result.exit_code == 0, result.output
    assert "Augmentation readiness: 11 ready, 3 blocked, 8 manual" in result.output
    assert "READY input.surface.rephrase@1.0.0" in result.output
    assert "Command: ul run --operator input.surface.rephrase@1.0.0" in result.output
    assert "BLOCKED environment.tool.timeout_after_commit@1.0.0" in result.output
    assert (
        "Inspection only: 0 model calls, 0 environment calls, 0 network requests." in result.output
    )


@pytest.mark.parametrize(
    ("augmentation_id", "stress_command"),
    (
        (
            "conversation.correction_after_first_response",
            "ul stress correction",
        ),
        (
            "conversation.retry_after_successful_commit",
            "ul stress retry-after-successful-commit",
        ),
        (
            "environment.tool.timeout_after_commit",
            "ul stress timeout-after-commit",
        ),
    ),
)
def test_plan_shows_safe_stress_cli_setup_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    augmentation_id: str,
    stress_command: str,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["augmentations", "plan", augmentation_id])

    assert result.exit_code == 0, result.output
    assert (
        f"Command: {stress_command} CASE.json --environment-config ENVIRONMENT.json "
        "--invariants INVARIANTS.json --dry-run"
    ) in result.output


def test_project_augmentation_configuration_persists_and_drives_plans_and_dry_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _initialize_project(tmp_path, monkeypatch)
    config_path = tmp_path / ".ul" / "config.json"

    enabled = runner.invoke(app, ["augmentations", "enabled", "--json"])
    added = runner.invoke(app, ["augmentations", "enable", "input.surface.typing_noise"])
    plan = runner.invoke(
        app,
        ["augmentations", "plan", "input.surface.typing_noise", "--json"],
    )
    dry_run = runner.invoke(app, ["run", "--dry-run"])

    assert enabled.exit_code == 0, enabled.output
    assert [item["ref"]["id"] for item in json.loads(enabled.output)["augmentations"]] == [
        "input.surface.rephrase"
    ]
    assert added.exit_code == 0, added.output
    assert "Enabled: input.surface.typing_noise@1.0.0" in added.output
    assert "Next steps:" in added.output
    assert "UL_LIVE=true" in added.output
    assert json.loads(config_path.read_text(encoding="utf-8"))["operators"] == [
        "input.surface.rephrase",
        "input.surface.typing_noise",
    ]
    assert json.loads(plan.output)["augmentations"][0]["enabled"] is True
    assert dry_run.exit_code == 0, dry_run.output
    assert "Operators: input.surface.rephrase, input.surface.typing_noise" in dry_run.output

    removed = runner.invoke(app, ["augmentations", "disable", "input.surface.rephrase"])
    reset = runner.invoke(app, ["augmentations", "reset"])

    assert removed.exit_code == 0, removed.output
    assert "Disabled: input.surface.rephrase@1.0.0" in removed.output
    assert reset.exit_code == 0, reset.output
    assert "Restored recommended defaults" in reset.output
    assert json.loads(config_path.read_text(encoding="utf-8"))["operators"] == [
        "input.surface.rephrase"
    ]


def test_concurrent_enables_do_not_lose_a_stale_read_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _initialize_project(tmp_path, monkeypatch)
    config_path = tmp_path / ".ul" / "config.json"
    real_load_project = augmentations.load_project
    stale_read_barrier = Barrier(2)

    def load_project_before_concurrent_update() -> tuple[Path, augmentations.ProjectConfig]:
        loaded_project = real_load_project()
        stale_read_barrier.wait()
        return loaded_project

    monkeypatch.setattr(augmentations, "load_project", load_project_before_concurrent_update)
    monkeypatch.setattr(augmentations, "_print_selected_readiness", lambda augmentation: None)

    with ThreadPoolExecutor(max_workers=2) as executor:
        updates = (
            executor.submit(augmentations.enable_augmentation, "input.surface.typing_noise"),
            executor.submit(augmentations.enable_augmentation, "input.surface.fragmented_syntax"),
        )
        for update in updates:
            update.result()

    assert set(json.loads(config_path.read_text(encoding="utf-8"))["operators"]) == {
        "input.surface.rephrase",
        "input.surface.typing_noise",
        "input.surface.fragmented_syntax",
    }


@pytest.mark.skipif(os.name == "nt", reason="symlink creation may require Windows privileges")
def test_configuration_update_rejects_a_symlinked_lock_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _initialize_project(tmp_path, monkeypatch)
    config_path = tmp_path / ".ul" / "config.json"
    original_config = config_path.read_bytes()
    lock_target = tmp_path / "lock-target"
    lock_target.write_bytes(b"")
    lock_target.chmod(0o600)
    (tmp_path / ".ul" / ".config.json.lock").symlink_to(lock_target)

    result = runner.invoke(app, ["augmentations", "enable", "input.surface.typing_noise"])

    assert result.exit_code == 2
    normalized_output = " ".join(result.output.split())
    assert "cannot update augmentation configuration" in normalized_output
    assert "must be a regular private file" in normalized_output
    assert config_path.read_bytes() == original_config
    assert lock_target.read_bytes() == b""


def test_configuration_errors_do_not_change_project_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _initialize_project(tmp_path, monkeypatch)
    config_path = tmp_path / ".ul" / "config.json"
    original = config_path.read_bytes()

    unknown = runner.invoke(app, ["augmentations", "enable", "input.unknown.operator"])
    unsupported = runner.invoke(app, ["augmentations", "enable", "conversation.ambiguity"])
    last = runner.invoke(app, ["augmentations", "disable", "input.surface.rephrase"])

    assert unknown.exit_code == 2
    assert "unknown augmentation" in unknown.output
    assert unsupported.exit_code == 2
    assert "cannot be enabled for 'ul run'" in unsupported.output
    assert "ul augmentations plan conversation.ambiguity@1.0.0" in unsupported.output
    assert last.exit_code == 2
    assert "at least one augmentation must remain enabled" in last.output
    assert config_path.read_bytes() == original


@pytest.mark.parametrize(
    "command", [("enabled",), ("enable", "input.surface.rephrase"), ("reset",)]
)
def test_configuration_commands_require_a_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: tuple[str, ...],
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["augmentations", *command])

    assert result.exit_code == 2
    assert "no UL project found; run 'ul init' first" in result.output
    assert not (tmp_path / ".ul").exists()
