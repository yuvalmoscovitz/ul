from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Annotated, TextIO, cast

import typer
from ul import load_dataset_semantic_settings

from examples.quickstart.defective_agent import create_server

_QUICKSTART_DIRECTORY = Path(__file__).resolve().parent
_PROJECT_DIRECTORY = _QUICKSTART_DIRECTORY.parents[1]
_DATASET_PATH = _QUICKSTART_DIRECTORY / "dataset.jsonl"
_INVARIANTS_PATH = _QUICKSTART_DIRECTORY / "invariants.json"
_TARGET_TEMPLATE_PATH = _QUICKSTART_DIRECTORY / "target.json"
_QUICKSTART_MODEL_ENVIRONMENT = {
    "UL_DATASET_MODEL": "x-ai/grok-4.6",
    "UL_DATASET_RENDER_MODEL": "x-ai/grok-4.6",
    "UL_DATASET_EQUIVALENCE_MODEL": "x-ai/grok-4.6",
}


def _create_private_file(path: Path) -> TextIO:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.fchmod(descriptor, 0o600)
    return os.fdopen(descriptor, "w", encoding="utf-8")


def load_target_template(base_url: str | None = None) -> dict[str, object]:
    with _TARGET_TEMPLATE_PATH.open(encoding="utf-8") as target_template_file:
        untyped_target_config: object = json.load(target_template_file)
    if type(untyped_target_config) is not dict:
        raise ValueError("quickstart target configuration must be a JSON object")
    target_config = cast(dict[str, object], untyped_target_config)
    if base_url is not None:
        endpoints = {
            "reset": "reset",
            "setup": "setup",
            "execute_turn": "execute",
            "snapshot": "snapshot",
        }
        for phase, endpoint in endpoints.items():
            phase_config = cast(dict[str, object], target_config[phase])
            phase_config["url"] = f"{base_url}/{endpoint}"
    return target_config


def _subprocess_environment(*, dry_run: bool) -> dict[str, str]:
    settings = load_dataset_semantic_settings()
    if settings.semantic_provider_type == "openrouter":
        environment = dict(_QUICKSTART_MODEL_ENVIRONMENT)
    else:
        environment = {
            "UL_DATASET_SEMANTIC_PROVIDER": "openai-compatible",
            "UL_DATASET_OPENAI_PROVIDER_ID": settings.semantic_provider_id,
            "UL_DATASET_OPENAI_BASE_URL": settings.semantic_base_url,
            "UL_DATASET_MODEL": settings.model,
            "UL_DATASET_RENDER_MODEL": settings.render_model,
            "UL_DATASET_EQUIVALENCE_MODEL": settings.equivalence_model,
        }
    if dry_run:
        if "PYTHONPATH" in os.environ:
            environment["PYTHONPATH"] = os.environ["PYTHONPATH"]
        return environment
    if settings.api_key_required and (
        settings.api_key is None or not settings.api_key.get_secret_value().strip()
    ):
        raise ValueError(
            f"set {settings.api_key_environment_variable} before running the live quickstart"
        )
    if not settings.live_calls:
        raise ValueError(
            "set UL_LIVE=true (or UL_DATASET_LIVE_CALLS=true) before running the live quickstart"
        )
    if not settings.allow_external_data_processing:
        raise ValueError(
            "set UL_LIVE=true (or UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING=true) before "
            "running the live quickstart"
        )
    if settings.api_key is not None and settings.api_key.get_secret_value().strip():
        environment[settings.api_key_environment_variable] = settings.api_key.get_secret_value()
    environment["UL_DATASET_LIVE_CALLS"] = "true"
    environment["UL_DATASET_ALLOW_EXTERNAL_DATA_PROCESSING"] = "true"
    return environment


def _evidence_confirms_repeatable_wrong_invoice(evidence_path: Path) -> bool:
    try:
        evidence_lines = evidence_path.read_text(encoding="utf-8").splitlines()
        if len(evidence_lines) != 1:
            return False
        untyped_evidence: object = json.loads(evidence_lines[0])
        if type(untyped_evidence) is not dict:
            return False
        evidence = cast(dict[str, object], untyped_evidence)
        baseline = evidence.get("current_baseline")
        if type(baseline) is not dict:
            return False
        baseline_mapping = cast(dict[str, object], baseline)
        if not _observations_are_stable_three_of_three(baseline_mapping.get("observations")):
            return False
        cases = evidence.get("cases")
        if type(cases) is not list:
            return False
        case_items = cast(list[object], cases)
        if len(case_items) != 1 or type(case_items[0]) is not dict:
            return False
        case = cast(dict[str, object], case_items[0])
        findings = case.get("findings")
        if type(findings) is not list:
            return False
        finding_items = cast(list[object], findings)
        return (
            case.get("operator_id") == "surface.disfluency_repeat"
            and case.get("variation_accepted") is True
            and case.get("status") == "REPEATABLE DIFFERENCE — REVIEW"
            and _observations_are_stable_three_of_three(case.get("observations"))
            and len(finding_items) == 1
            and type(finding_items[0]) is dict
            and _finding_is_wrong_invoice_change(cast(dict[str, object], finding_items[0]))
            and _invariant_confirms_wrong_invoice(evidence.get("invariant_evaluation"))
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        return False


def _invariant_confirms_wrong_invoice(invariant_evaluation: object) -> bool:
    if type(invariant_evaluation) is not dict:
        return False
    invariant_mapping = cast(dict[str, object], invariant_evaluation)
    baseline = invariant_mapping.get("baseline")
    variations = invariant_mapping.get("variations")
    if type(baseline) is not dict or type(variations) is not list:
        return False
    baseline_rules = cast(dict[str, object], baseline).get("rules")
    variation_items = cast(list[object], variations)
    if (
        type(baseline_rules) is not list
        or len(cast(list[object], baseline_rules)) != 1
        or len(variation_items) != 1
        or type(variation_items[0]) is not dict
    ):
        return False
    variation = cast(dict[str, object], variation_items[0])
    variation_rules = variation.get("rules")
    if type(variation_rules) is not list or len(cast(list[object], variation_rules)) != 1:
        return False
    baseline_rule = cast(list[object], baseline_rules)[0]
    variation_rule = cast(list[object], variation_rules)[0]
    return (
        type(baseline_rule) is dict
        and type(variation_rule) is dict
        and cast(dict[str, object], baseline_rule).get("rule_id")
        == "committed-invoice-matches-request"
        and cast(dict[str, object], baseline_rule).get("status") == "satisfied"
        and cast(dict[str, object], variation_rule).get("rule_id")
        == "committed-invoice-matches-request"
        and cast(dict[str, object], variation_rule).get("status") == "violated"
    )


def _finding_is_wrong_invoice_change(finding: dict[str, object]) -> bool:
    if finding.get("category") != "changed_grounded_effect_argument":
        return False
    reference_effects = finding.get("reference_effects")
    observed_effects = finding.get("observed_effects")
    if type(reference_effects) is not list or type(observed_effects) is not list:
        return False
    reference_items = cast(list[object], reference_effects)
    observed_items = cast(list[object], observed_effects)
    if (
        len(reference_items) != 1
        or len(observed_items) != 1
        or type(reference_items[0]) is not dict
        or type(observed_items[0]) is not dict
    ):
        return False
    reference_effect = cast(dict[str, object], reference_items[0])
    observed_effect = cast(dict[str, object], observed_items[0])
    reference_fields = reference_effect.get("fields")
    observed_fields = observed_effect.get("fields")
    return (
        reference_effect.get("predicate") == "payment_committed"
        and observed_effect.get("predicate") == "payment_committed"
        and type(reference_fields) is dict
        and type(observed_fields) is dict
        and cast(dict[str, object], reference_fields).get("invoice_reference") == "AC-100"
        and cast(dict[str, object], observed_fields).get("invoice_reference") == "AC-101"
    )


def _observations_are_stable_three_of_three(observations: object) -> bool:
    if type(observations) is not dict:
        return False
    observation_mapping = cast(dict[str, object], observations)
    return (
        observation_mapping.get("requested_repetitions") == 3
        and observation_mapping.get("stability") == "stable"
        and observation_mapping.get("observed_repetitions") == 3
        and observation_mapping.get("inconclusive_repetitions") == 0
        and observation_mapping.get("outcome_group_count") == 1
    )


def main(
    dry_run: Annotated[
        bool,
        typer.Option(help="Validate the complete plan without model or target calls."),
    ] = False,
) -> None:
    try:
        subprocess_environment = _subprocess_environment(dry_run=dry_run)
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from None

    temporary_root = _PROJECT_DIRECTORY / "tmp"
    temporary_root.mkdir(exist_ok=True)
    artifact_directory = Path(tempfile.mkdtemp(prefix="quickstart-", dir=temporary_root))
    target_config_path = artifact_directory / "target.json"
    evidence_path = artifact_directory / "evidence.jsonl"
    server: ThreadingHTTPServer | None = None
    server_thread: Thread | None = None

    try:
        base_url: str | None = None
        if not dry_run:
            server = create_server()
            server_host, server_port = cast(tuple[str, int], server.server_address)
            base_url = f"http://{server_host}:{server_port}"
            server_thread = Thread(
                target=server.serve_forever,
                name="quickstart-agent",
                daemon=True,
            )
        target_config = load_target_template(base_url)
        with _create_private_file(target_config_path) as target_config_file:
            json.dump(target_config, target_config_file, indent=2)
            target_config_file.write("\n")

        if server_thread is not None:
            server_thread.start()
        command = [
            sys.executable,
            "-m",
            "ul_cli.main",
            "dataset",
            "evaluate",
            str(_DATASET_PATH),
            "--target-config",
            str(target_config_path),
            "--invariants",
            str(_INVARIANTS_PATH),
            "--operator",
            "surface.disfluency_repeat",
            "--limit",
            "1",
            "--repetitions",
            "3",
            "--max-target-calls",
            "30",
            "--output",
            str(evidence_path),
            "--allow-target-network",
            "--confirm-isolated-sandbox",
            "--allow-insecure-http",
        ]
        if dry_run:
            command.append("--dry-run")
        completed_process = subprocess.run(
            command,
            cwd=_QUICKSTART_DIRECTORY,
            env=subprocess_environment,
            check=False,
            shell=False,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        typer.echo(f"Quickstart could not run: {error.__class__.__name__}", err=True)
        raise typer.Exit(code=2) from None
    finally:
        if server is not None:
            if server_thread is not None and server_thread.ident is not None:
                server.shutdown()
            server.server_close()
            if server_thread is not None and server_thread.ident is not None:
                server_thread.join(timeout=5)
        target_config_path.unlink(missing_ok=True)
        if not evidence_path.exists():
            artifact_directory.rmdir()

    if dry_run:
        if completed_process.returncode != 0:
            raise typer.Exit(code=completed_process.returncode)
        typer.echo("Dry run complete. No model or target requests sent.")
        return

    if completed_process.returncode == 1 and _evidence_confirms_repeatable_wrong_invoice(
        evidence_path
    ):
        typer.echo(
            "Confirmed: UL found a stable 3/3 wrong-invoice action and the customer rule failed."
        )
        typer.echo(f"Evidence: {evidence_path}")
        return

    if evidence_path.exists():
        typer.echo(f"UL did not confirm the expected finding. Review: {evidence_path}", err=True)
    else:
        typer.echo("UL did not produce evidence.", err=True)
    raise typer.Exit(code=completed_process.returncode or 1)


if __name__ == "__main__":
    typer.run(main)
