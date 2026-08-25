import json

from typer.testing import CliRunner
from ul_cli.main import app

runner = CliRunner()


def test_customer_can_discover_and_inspect_named_bundles() -> None:
    listed = runner.invoke(app, ["augmentations", "bundles", "list", "--json"])
    shown = runner.invoke(
        app,
        ["augmentations", "bundles", "show", "everyday-customers"],
    )

    assert listed.exit_code == 0, listed.output
    assert [bundle["id"] for bundle in json.loads(listed.output)["bundles"]] == [
        "everyday-customers",
        "retries-interrupted-work",
        "unclear-changing-requests",
    ]
    assert shown.exit_code == 0, shown.output
    assert "independent probes only" in shown.output
    assert "Hard budget:" in shown.output
    assert "model calls=120" in shown.output


def test_customer_can_preview_a_bounded_bundle_without_external_calls() -> None:
    result = runner.invoke(
        app,
        [
            "augmentations",
            "bundles",
            "plan",
            "everyday-customers",
            "--case",
            "case-1",
            "--case",
            "case-2",
            "--source-feature",
            "production interaction",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["bundle_id"] == "everyday-customers"
    assert plan["composition"] == "independent_only"
    assert plan["totals"] == {
        "cases": 2,
        "planned_probes": 0,
        "blocked_probes": 6,
        "skipped_probes": 0,
        "model_calls": 0,
        "target_calls": 0,
        "maximum_duration_seconds": 0,
        "maximum_cost_usd": 0.0,
        "mutating_probes": 0,
    }
    assert plan["inspection_model_calls"] == 0
    assert plan["inspection_target_calls"] == 0
    assert plan["inspection_network_requests"] == 0
    assert all(probe["source_case_id"] == probe["source_parent_id"] for probe in plan["probes"])
    assert {probe["status"] for probe in plan["probes"]} == {"blocked"}
    assert all("operator is not qualified" in probe["reasons"] for probe in plan["probes"])


def test_bundle_preview_shows_skips_exact_changes_evidence_and_reset_needs() -> None:
    skipped = runner.invoke(
        app,
        [
            "augmentations",
            "bundles",
            "plan",
            "unclear-changing-requests",
            "--case",
            "case-1",
        ],
    )
    stateful = runner.invoke(
        app,
        [
            "augmentations",
            "bundles",
            "plan",
            "retries-interrupted-work",
            "--case",
            "case-1",
            "--source-feature",
            "action.write",
            "--source-feature",
            "two ordered user turns",
        ],
    )

    assert skipped.exit_code == 0, skipped.output
    assert "SKIPPED case-1" in skipped.output
    assert "Exact controlled change:" in skipped.output
    assert "Evidence: response" in skipped.output
    assert "Inspection only: 0 model calls, 0 target calls, 0 network requests" in skipped.output
    assert stateful.exit_code == 0, stateful.output
    assert "mutation=state; reset=yes" in stateful.output
    assert "mutation=fault; reset=yes" in stateful.output


def test_bundle_commands_reject_unknown_or_unbounded_input() -> None:
    unknown = runner.invoke(app, ["augmentations", "bundles", "show", "unknown"])
    case_arguments = [value for index in range(11) for value in ("--case", f"case-{index}")]
    too_many_cases = runner.invoke(
        app,
        [
            "augmentations",
            "bundles",
            "plan",
            "everyday-customers",
            *case_arguments,
        ],
    )

    assert unknown.exit_code == 2
    assert "unknown bundle" in unknown.output
    assert too_many_cases.exit_code == 2
    assert "case count exceeds" in too_many_cases.output
