import json

from typer.testing import CliRunner
from ul_cli.main import app


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
            "sandbox_fault",
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
        ["augmentations", "list", "--scope", "input", "--mode", "sandbox_fault"],
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
    assert "Mode: sandbox_fault" in result.output
    assert "Execution owner: SDK augmentation registry" in result.output
    assert "Execution owner: stress CLI" in result.output
    assert "sandbox capability environment.tool.timeout_after_commit@1.0.0" in result.output
    assert "0 model calls, 0 sandbox calls, 0 network requests" in result.output


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
    assert "isolated sandbox" in result.output
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
