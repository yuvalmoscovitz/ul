import json

from typer.testing import CliRunner
from ul_cli.main import app

runner = CliRunner()


def test_accounts_payable_control_runs_all_scenarios() -> None:
    result = runner.invoke(app, ["demo", "accounts-payable"])

    assert result.exit_code == 0, result.output
    assert "single-approved-invoice" in result.output
    assert "timeout-after-commit" in result.output
    assert "PASS" in result.output


def test_live_demo_requires_an_explicit_scenario() -> None:
    result = runner.invoke(app, ["demo", "accounts-payable", "--live"])

    assert result.exit_code != 0
    assert "Live runs require at least one explicit" in result.output


def test_live_demo_allows_only_one_unique_scenario() -> None:
    result = runner.invoke(
        app,
        [
            "demo",
            "accounts-payable",
            "--live",
            "--scenario",
            "single-approved-invoice",
            "--scenario",
            "foreign-currency-invoice",
        ],
    )

    assert result.exit_code != 0
    assert "Live runs require exactly one unique" in result.output


def test_live_demo_rejects_more_than_two_cases_before_execution() -> None:
    result = runner.invoke(
        app,
        [
            "demo",
            "accounts-payable",
            "--live",
            "--augment",
            "--scenario",
            "single-approved-invoice",
            "--case-limit",
            "3",
        ],
    )

    assert result.exit_code != 0
    assert "at most two billed cases" in result.output


def test_accounts_payable_augmentation_campaign_uses_generic_engine() -> None:
    result = runner.invoke(
        app,
        [
            "demo",
            "accounts-payable",
            "--augment",
            "--scenario",
            "single-approved-invoice",
            "--case-limit",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "baseline" in result.output
    assert "tool.timeout" in result.output
    assert "PASS" in result.output


def test_accounts_payable_baseline_uses_generic_engine() -> None:
    result = runner.invoke(
        app,
        [
            "demo",
            "accounts-payable",
            "--scenario",
            "single-approved-invoice",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload[0]["campaign_id"] == "accounts-payable:single-approved-invoice"
    assert len(payload[0]["cases"]) == 1
    assert payload[0]["cases"][0]["augmentation_ids"] == []


def test_unknown_scenario_is_rejected_before_execution() -> None:
    result = runner.invoke(
        app,
        ["demo", "accounts-payable", "--scenario", "does-not-exist"],
    )

    assert result.exit_code != 0
    assert "Unknown scenario" in result.output
