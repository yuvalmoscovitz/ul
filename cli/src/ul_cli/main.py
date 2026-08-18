from __future__ import annotations

import asyncio
import json
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from ul import CampaignResult, CampaignRunner, ExecutionMode

from examples.accounts_payable.agent import OpenRouterSettings
from examples.accounts_payable.scenarios import get_seed_scenario, seed_scenarios
from examples.accounts_payable.ul_adapter import (
    AccountsPayableOpenRouterTarget,
    AccountsPayableOracleEvaluator,
    AccountsPayableScenarioMaterializer,
    AccountsPayableScriptedTarget,
    accounts_payable_augmentation_registry,
    to_ul_scenario,
)
from ul_cli.dataset import app as dataset_app
from ul_cli.dataset_regression import app as regression_app

app = typer.Typer(
    name="ul",
    help="Discover consequential failures in high-risk AI agents.",
    no_args_is_help=True,
)
augmentations_app = typer.Typer(help="Inspect UL's built-in augmentation library.")
demo_app = typer.Typer(help="Run an isolated proving-ground agent.")
app.add_typer(augmentations_app, name="augmentations")
app.add_typer(dataset_app, name="dataset")
app.add_typer(regression_app, name="regression")
app.add_typer(demo_app, name="demo")

console = Console()


@augmentations_app.command("list")
def list_augmentations() -> None:
    from ul_core.augmentation import builtin_augmentation_registry

    registry = builtin_augmentation_registry()
    table = Table(title="Built-in plausible-behavior augmentations")
    table.add_column("ID", style="cyan")
    table.add_column("Version")
    table.add_column("Category")
    table.add_column("Description")
    for augmentation in registry.list():
        table.add_row(
            augmentation.metadata.id,
            augmentation.metadata.version,
            str(augmentation.metadata.category),
            augmentation.metadata.summary,
        )
    console.print(table)


@demo_app.command("accounts-payable")
def run_accounts_payable_demo(
    scenario: Annotated[
        list[str] | None,
        typer.Option(
            "--scenario",
            "-s",
            help="Scenario ID. Repeat the option to run more than one; defaults to all offline.",
        ),
    ] = None,
    live: Annotated[
        bool,
        typer.Option(
            "--live",
            help="Make billed OpenRouter requests instead of using the deterministic control.",
        ),
    ] = False,
    augment: Annotated[
        bool,
        typer.Option(
            "--augment",
            help="Generate and run applicable cases from UL's built-in library.",
        ),
    ] = False,
    limit: Annotated[
        int,
        typer.Option(min=1, max=100, help="Maximum scenarios to execute."),
    ] = 100,
    case_limit: Annotated[
        int | None,
        typer.Option(
            min=1,
            max=100,
            help="Cases per seed, including its baseline. Defaults to 2 live and 100 offline.",
        ),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Print complete machine-readable results."),
    ] = False,
) -> None:
    available_scenarios = seed_scenarios()
    selected_ids = scenario or [item.id for item in available_scenarios]
    known_ids = {item.id for item in available_scenarios}
    unknown_ids = sorted(set(selected_ids) - known_ids)
    if unknown_ids:
        raise typer.BadParameter(f"Unknown scenario ID(s): {', '.join(unknown_ids)}")
    if live and scenario is None:
        raise typer.BadParameter("Live runs require at least one explicit --scenario.")
    if live and (len(selected_ids) != 1 or len(set(selected_ids)) != 1):
        raise typer.BadParameter("Live runs require exactly one unique --scenario.")
    if live and case_limit is not None and case_limit > 2:
        raise typer.BadParameter("Live runs allow at most two billed cases.")

    selected_ids = selected_ids[:limit]
    campaign_results = asyncio.run(
        _run_accounts_payable_campaigns(
            selected_ids,
            live=live,
            augmentation_limit=None if augment else 0,
            case_limit=(case_limit or (2 if live else 100)) if augment else 1,
        )
    )
    if output_json:
        console.print_json(
            json.dumps(
                [result.model_dump(mode="json") for result in campaign_results],
                sort_keys=True,
            )
        )
    elif augment:
        _print_campaign_results(campaign_results, live=live)
    else:
        _print_baseline_results(campaign_results, live=live)

    if any(result.failed_case_count for result in campaign_results):
        raise typer.Exit(code=1)


async def _run_accounts_payable_campaigns(
    scenario_ids: list[str],
    *,
    live: bool,
    augmentation_limit: int | None,
    case_limit: int,
) -> list[CampaignResult]:
    materializer = AccountsPayableScenarioMaterializer(
        ExecutionMode.LIVE if live else ExecutionMode.SANDBOX
    )
    target = (
        AccountsPayableOpenRouterTarget(
            OpenRouterSettings(
                live_calls_enabled=True,
                max_output_tokens=800,
            )
        )
        if live
        else AccountsPayableScriptedTarget()
    )
    runner = CampaignRunner(
        materializer,
        target,
        AccountsPayableOracleEvaluator(),
        registry=accounts_payable_augmentation_registry(),
        allow_network_egress=live,
        allow_business_side_effects=False,
    )
    return [
        await runner.run(
            f"accounts-payable:{scenario_id}",
            to_ul_scenario(get_seed_scenario(scenario_id)),
            augmentation_limit=augmentation_limit,
            max_cases=case_limit,
        )
        for scenario_id in scenario_ids
    ]


def _print_baseline_results(results: list[CampaignResult], *, live: bool) -> None:
    table = Table(title="Accounts-payable campaign")
    table.add_column("Scenario", style="cyan")
    table.add_column("Agent")
    table.add_column("Result")
    table.add_column("Findings")
    for result in results:
        case = result.cases[0]
        failed_findings = [finding for finding in case.findings if not finding.passed]
        table.add_row(
            result.campaign_id.removeprefix("accounts-payable:"),
            "OpenRouter" if live else "deterministic control",
            "FAIL" if failed_findings else "PASS",
            ", ".join(finding.category for finding in failed_findings) or "—",
        )
    console.print(table)


def _print_campaign_results(results: list[CampaignResult], *, live: bool) -> None:
    table = Table(title="Accounts-payable augmentation campaign")
    table.add_column("Seed", style="cyan")
    table.add_column("Augmentation")
    table.add_column("Agent")
    table.add_column("Result")
    table.add_column("Findings")
    for result in results:
        seed_id = result.campaign_id.removeprefix("accounts-payable:")
        for case in result.cases:
            augmentation_label = "baseline"
            if case.augmentation_ids:
                variant = case.scenario_id.rsplit(":", maxsplit=1)[-1]
                augmentation_label = f"{', '.join(case.augmentation_ids)}:{variant}"
            table.add_row(
                seed_id,
                augmentation_label,
                "OpenRouter" if live else "deterministic control",
                "PASS" if all(finding.passed for finding in case.findings) else "FAIL",
                ", ".join(finding.category for finding in case.findings if not finding.passed)
                or "—",
            )
    console.print(table)


if __name__ == "__main__":
    app()
