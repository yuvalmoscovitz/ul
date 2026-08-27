from __future__ import annotations

import asyncio
import json
import os
import stat
import tempfile
import unicodedata
from pathlib import Path
from typing import TextIO

import typer
from platformdirs import user_data_path
from pydantic import JsonValue
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from ul.dataset_evaluation import DatasetEvaluationFinding, DatasetEvaluationResult

from ul_cli.dataset import create_customer_evidence_record
from ul_cli.demo_scenario import run_demo_evaluations
from ul_cli.pattern_identity import (
    ensure_project_pattern_identity_key,
    ensure_project_review_history_key,
)

console = Console()

_AUGMENTATION_NAMES = {
    "input.surface.typing_noise": "Typing errors",
    "input.tone.frustrated": "Frustrated customer",
    "input.style.terse": "Short message",
}


def _create_private_file(path: Path) -> TextIO:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("demo artifact is not a regular file")
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "w", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise


def _create_demo_artifact_directory() -> Path:
    demo_directory = user_data_path("ul", appauthor=False) / "demo"
    demo_directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        demo_directory.mkdir(mode=0o700)
    except FileExistsError:
        directory_status = demo_directory.lstat()
        if stat.S_ISLNK(directory_status.st_mode) or not stat.S_ISDIR(directory_status.st_mode):
            raise ValueError("UL demo data path must be a directory, not a symlink") from None
        if os.name != "nt":
            os.chmod(demo_directory, 0o700)
    return Path(tempfile.mkdtemp(prefix="human-inputs-", dir=demo_directory))


def _save_evidence(results: tuple[DatasetEvaluationResult, ...], path: Path) -> None:
    planned_calls = sum(3 * (1 + len(result.cases)) for result in results)
    with _create_private_file(path) as evidence_file:
        for result in results:
            evidence = create_customer_evidence_record(
                result,
                repetitions=3,
                max_environment_api_calls=planned_calls,
                planned_target_calls=planned_calls,
            )
            json.dump(evidence, evidence_file, ensure_ascii=False, separators=(",", ":"))
            evidence_file.write("\n")


def _describe_outcomes(result: DatasetEvaluationResult, case_index: int | None = None) -> str:
    if case_index is None:
        trial_set = result.baseline.trial_set
    else:
        trial_set = result.cases[case_index].trial_set
        if trial_set is None:
            return "The variation was not run."
    representative_frame = trial_set.representative_frame
    if representative_frame is None:
        return "The agent's behavior was not stable."
    outcomes = tuple(
        outcome for outcome in representative_frame.outcomes if outcome.kind == "action"
    )
    if not outcomes:
        return "The agent took no action."
    outcome = outcomes[0]
    if outcome.predicate == "subscription_cancellation_scheduled":
        timing = outcome.fields.get("when")
        if timing == "end":
            return "The agent scheduled cancellation for the end of the billing period."
        if timing == "immediately":
            return "The agent cancelled the subscription immediately."
    if outcome.predicate == "customer_address_changed":
        address_type = outcome.fields.get("address_type")
        address = outcome.fields.get("address")
        return f"The agent changed the {address_type} address to {address}."
    return "The agent took a different action."


def _display_field_value(field_name: str, value: JsonValue) -> str:
    labels = {
        ("when", "end"): "end of the billing period",
        ("when", "immediately"): "immediately",
        ("address_type", "delivery"): "delivery address",
        ("address_type", "billing"): "billing address",
    }
    if isinstance(value, str):
        return labels.get((field_name, value), value)
    return str(value)


def _describe_finding(finding: DatasetEvaluationFinding) -> str:
    if finding.category == "changed_response":
        return "UL detected a repeatable response difference."
    if finding.category == "missing_effect":
        predicate = finding.expected_effects[0].predicate.replace("_", " ")
        return f"UL detected that the {predicate} action is missing."
    if (
        finding.category == "changed_grounded_effect_argument"
        and finding.expected_effects
        and finding.observed_effects
        and finding.grounded_field_names
    ):
        field_name = finding.grounded_field_names[0]
        expected_value = finding.expected_effects[0].fields.get(field_name)
        observed_value = finding.observed_effects[0].fields.get(field_name)
        field_label = {
            "when": "cancellation timing",
            "address_type": "address type",
        }.get(field_name, field_name.replace("_", " "))
        return (
            "UL detected that "
            f"{field_label} changed from "
            f"{_display_field_value(field_name, expected_value)} to "
            f"{_display_field_value(field_name, observed_value)}."
        )
    return "UL detected a repeatable action difference."


def _terminal_safe(message: str) -> str:
    return "".join(
        character
        if (ord(character) >= 32 and not 0x7F <= ord(character) <= 0x9F)
        and unicodedata.category(character) not in {"Cf", "Cs"}
        else f"\\u{ord(character):04x}"
        for character in message
    )


def _print_report(results: tuple[DatasetEvaluationResult, ...], evidence_path: Path) -> None:
    finding_count = sum(len(case.findings) for result in results for case in result.cases)
    augmentation_count = sum(len(result.cases) for result in results)
    console.print(
        "[bold cyan]UL demo[/bold cyan] "
        "[dim]Synthetic agent · UL comparison and evidence pipeline[/dim]"
    )
    console.print()
    summary = Text()
    summary.append(f"{len(results)} customer requests", style="bold")
    summary.append("  •  ")
    summary.append(f"{augmentation_count} augmentations", style="bold cyan")
    summary.append("  •  ")
    summary.append(f"{finding_count} repeatable findings", style="bold red")
    console.print(Panel.fit(summary, title="[bold cyan]UL evaluation report[/bold cyan]"))
    console.print()
    console.print(
        "[dim]UL reruns each original input to establish a baseline. It then automatically "
        "compares actions after each augmentation. Repeatable differences need human review; "
        "no custom rules are used here. The input changes are prewritten so this demo needs "
        "no model.[/dim]"
    )

    for result_number, result in enumerate(results, start=1):
        console.print()
        console.rule(f"[bold]Request {result_number}[/bold]", style="cyan")
        console.print(Text.assemble(("Baseline input   ", "bold"), result.source.raw_input))
        console.print(
            Text.assemble(
                ("Baseline result  ", "bold green"),
                _describe_outcomes(result),
            )
        )
        for case_index, case in enumerate(result.cases):
            console.print()
            augmentation_name = _AUGMENTATION_NAMES[case.candidate.operator_id]
            console.print(Text.assemble(("Augmentation     ", "bold cyan"), augmentation_name))
            console.print(
                Text.assemble(("Changed input    ", "bold"), case.candidate.augmented_input)
            )
            console.print(
                Text.assemble(
                    ("Agent result     ", "bold red"),
                    _describe_outcomes(result, case_index),
                )
            )
            for finding in case.findings:
                console.print(
                    Text.assemble(
                        ("Finding          ", "bold red"),
                        _describe_finding(finding),
                    )
                )
            console.print(
                "[dim]Detected by UL's action comparison · Seen in 3/3 runs · "
                "Needs human review[/dim]"
            )

    console.print()
    console.print(
        "[bold yellow]Demo note:[/bold yellow] These are intentional defects in a fake agent. "
        "Real findings are marked for review because business context decides whether a "
        "behavior change is harmful."
    )
    console.print()
    safe_evidence_path = _terminal_safe(str(evidence_path))
    console.print(
        Text.assemble(("Technical evidence saved  ", "dim"), (safe_evidence_path, "cyan")),
        soft_wrap=True,
    )
    console.print()
    console.print("[bold cyan]Try this workflow with your agent[/bold cyan]")
    console.print(
        '1. Save observed interactions as JSONL: {"id":"case-1","input":"...",'
        '"output":{"observed":"..."}}'
    )
    console.print(
        "2. Choose an existing callable or synchronous JSON endpoint that is isolated and safe "
        "for test traffic."
    )
    console.print("3. Start: [bold]ul probe interactions.jsonl --target agent:invoke[/bold]")
    console.print(
        "   HTTP: [bold]ul probe interactions.jsonl --target https://agent.test/invoke "
        "--header-from-env Authorization=UL_ENVIRONMENT_AGENT_TOKEN[/bold]"
    )
    console.print(
        "[dim]The saved output is observed reference evidence, not a correctness oracle. "
        "A response-only target does not verify trajectory or committed state. UL asks you to "
        "confirm the exact test target and bounded paid/network campaign before it runs.[/dim]"
    )
    console.print(
        "[dim]Need authoritative state evidence later? Use ul init and ul run with an isolated "
        "stateful lifecycle environment.[/dim]"
    )


def run_demo(output: Path | None = None) -> None:
    if output is not None and output.exists():
        typer.echo("Demo could not run: output already exists", err=True)
        raise typer.Exit(code=2)
    artifact_directory: Path | None = None
    evidence_path: Path | None = None
    try:
        artifact_directory = _create_demo_artifact_directory()
        evidence_path = (
            output.absolute() if output is not None else artifact_directory / "evidence.jsonl"
        )
        results = asyncio.run(run_demo_evaluations())
        project_directory = evidence_path.parent / ".ul"
        project_directory.mkdir(mode=0o700, exist_ok=True)
        ensure_project_pattern_identity_key(project_directory)
        ensure_project_review_history_key(project_directory)
        _save_evidence(results, evidence_path)
    except (OSError, ValueError) as error:
        typer.echo(f"Demo could not run: {error.__class__.__name__}", err=True)
        raise typer.Exit(code=2) from None
    finally:
        if (
            artifact_directory is not None
            and artifact_directory.exists()
            and not any(artifact_directory.iterdir())
        ):
            artifact_directory.rmdir()

    assert evidence_path is not None
    _print_report(results, evidence_path)
