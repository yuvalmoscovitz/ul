import json
from pathlib import Path

import pytest
from typer.testing import CliRunner
from ul_cli import dataset
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
    assert len(payload["augmentations"]) == 18
    references = [(item["ref"]["id"], item["ref"]["version"]) for item in payload["augmentations"]]
    assert references == sorted(references)
    assert all(isinstance(item["cli_available"], bool) for item in payload["augmentations"])


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
    result = CliRunner().invoke(app, ["augmentations", "show", "input.tone.frustrated"])

    assert result.exit_code == 0
    assert "ul dataset evaluate --operator input.tone.frustrated@1.0.0" in result.output
    assert "test environment" in result.output
    assert "committed-state observation" in result.output
    assert "semantic model" in result.output
    assert "customer evaluator" not in result.output


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
    assert payload["summary"] == {"ready": 6, "blocked": 3, "manual": 9}
    assert payload["inspection"] == {
        "model_calls": 0,
        "environment_calls": 0,
        "network_requests": 0,
    }
    assert len(payload["augmentations"]) == 18
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
    assert _reason_codes(rephrase) == {"requirements_satisfied"}

    frustrated = _planned_augmentation(payload, "input.tone.frustrated")
    assert frustrated["status"] == "manual"
    assert frustrated["command"] == "ul run --operator input.tone.frustrated@1.0.0"
    assert "human_review_required" in _reason_codes(frustrated)

    ambiguity = _planned_augmentation(payload, "conversation.ambiguity")
    assert ambiguity["status"] == "manual"
    assert ambiguity["command"] is None
    assert "cli_unavailable" in _reason_codes(ambiguity)

    timeout = _planned_augmentation(payload, "environment.tool.timeout_after_commit")
    assert timeout["status"] == "blocked"
    assert timeout["command"] is None
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

    monkeypatch.setattr(dataset, "create_semantic_model_deconstructor", unexpected_external_client)
    monkeypatch.setattr(
        dataset.JsonHttpEnvironmentConnection,
        "from_config",
        unexpected_external_client,
    )

    result = runner.invoke(app, ["augmentations", "plan", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"] == {"ready": 6, "blocked": 0, "manual": 12}
    timeout = _planned_augmentation(payload, "environment.tool.timeout_after_commit")
    assert timeout["status"] == "manual"
    assert timeout["command"] is None
    assert "stress_case_not_configured" in _reason_codes(timeout)
    assert "environment_capability_missing" not in _reason_codes(timeout)


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
    assert payload["summary"] == {"ready": 0, "blocked": 11, "manual": 7}
    assert len(payload["augmentations"]) == 18
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
    assert "Augmentation readiness: 6 ready, 3 blocked, 9 manual" in result.output
    assert "READY input.surface.rephrase@1.0.0" in result.output
    assert "Command: ul run --operator input.surface.rephrase@1.0.0" in result.output
    assert "MANUAL input.tone.frustrated@1.0.0" in result.output
    assert "BLOCKED environment.tool.timeout_after_commit@1.0.0" in result.output
    assert (
        "Inspection only: 0 model calls, 0 environment calls, 0 network requests." in result.output
    )
