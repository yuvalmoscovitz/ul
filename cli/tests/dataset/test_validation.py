from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner
from ul import (
    JsonHttpEnvironmentConfig,
)
from ul_cli.dataset.evaluation import command as command_module
from ul_cli.dataset.evaluation import runner as runner_module
from ul_cli.dataset.evaluation.records import load_interaction_records
from ul_cli.http_target_resolution import resolve_http_target
from ul_cli.main import app as root_app

from ._factories import (
    _settings,
)
from ._files import (
    _record,
    _write_dataset,
    _write_target_config,
)

runner = CliRunner()
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def test_loader_accepts_shorthand_and_projects_rich_cases(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    rich_case = {
        "schema_version": "1.0.0",
        "id": "cancel-order",
        "inputs": {"order_id": "ord-9", "message": "Cancel my order."},
        "context": [
            {"id": "user-1", "role": "user", "content": "Cancel my order."},
            {"id": "assistant-1", "role": "assistant", "content": "Are you sure?"},
            {"id": "user-2", "role": "user", "content": "Yes."},
        ],
        "augmentation_targets": [
            {"id": "message", "kind": "input_field", "json_pointer": "/inputs/message"},
            {"id": "confirmation", "kind": "conversation_turn", "turn_id": "user-2"},
        ],
        "fixture": {"id": "orders", "version": "9"},
        "observed_output": {"status": "cancelled"},
    }
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"id": "simple", "input": "Hello", "output": "Hi"}),
                json.dumps(rich_case),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_interaction_records(dataset)

    assert [record.id for record in records] == [
        "simple",
        "cancel-order::message",
        "cancel-order::confirmation",
    ]
    assert records[0].source_case is None
    assert records[1].source_interaction_id == "cancel-order"
    projected_context = records[2].probe_context("Absolutely.")["context"]
    assert isinstance(projected_context, list)
    assert projected_context[-1] == {
        "id": "user-2",
        "role": "user",
        "content": "Absolutely.",
        "name": None,
    }


def test_shorthand_preserves_bounded_metadata_and_structured_json_values(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "structured",
                "input": {
                    "request": {"message": "Return ticket 42.", "ticket_id": 42},
                    "mode": "test",
                },
                "augmentation_target": "/request/message",
                "output": {"ticket": {"id": 42, "status": "open"}},
                "metadata": {"source": "approved-observation", "private_id": "customer-7"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    record = load_interaction_records(dataset)[0]

    assert record.raw_input == "Return ticket 42."
    assert record.raw_observed_output == {"ticket": {"id": 42, "status": "open"}}
    assert record.metadata == {"source": "approved-observation", "private_id": "customer-7"}
    assert record.target_input("Return ticket 43.") == {
        "request": {"message": "Return ticket 43.", "ticket_id": 42},
        "mode": "test",
    }


def test_structured_shorthand_requires_one_text_augmentation_target(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    dataset.write_text(
        json.dumps({"id": "structured", "input": {"message": "Hello"}, "output": "Hi"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="structured input requires an augmentation_target"):
        load_interaction_records(dataset)


@pytest.mark.parametrize(
    ("structured_input", "target", "expected"),
    (
        ({"message": "Hello", "tenant": "test"}, "/message", {"message": "Hi", "tenant": "test"}),
        (["Hello", {"tenant": "test"}], "/0", ["Hi", {"tenant": "test"}]),
    ),
)
def test_structured_shorthand_replaces_top_level_augmentation_target(
    tmp_path: Path,
    structured_input: object,
    target: str,
    expected: object,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "structured",
                "input": structured_input,
                "augmentation_target": target,
                "output": "Hi",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    record = load_interaction_records(dataset)[0]

    assert record.target_input("Hi") == expected


def test_shorthand_metadata_is_bounded(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "too-much-metadata",
                "input": "Hello",
                "output": "Hi",
                "metadata": {f"key-{index}": index for index in range(101)},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid metadata"):
        load_interaction_records(dataset)


def test_one_dataset_preserves_ten_fixture_routes(tmp_path: Path) -> None:
    dataset = tmp_path / "fixtures.jsonl"
    cases = [
        {
            "schema_version": "1.0.0",
            "id": f"case-{index}",
            "inputs": {"message": f"Test customer {index}"},
            "augmentation_targets": [
                {"id": "message", "kind": "input_field", "json_pointer": "/inputs/message"}
            ],
            "fixture": {"id": f"customer-{index}", "version": "1"},
            "observed_output": {"status": "observed"},
        }
        for index in range(10)
    ]
    dataset.write_text(
        "".join(f"{json.dumps(case)}\n" for case in cases),
        encoding="utf-8",
    )

    records = load_interaction_records(dataset)

    assert [record.probe_context()["fixture"] for record in records] == [
        {"id": f"customer-{index}", "version": "1"} for index in range(10)
    ]


def test_target_config_dry_run_validates_environment_and_makes_no_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(
        target_config,
        headers_from_env={"Authorization": "UL_ENVIRONMENT_TOKEN"},
        request_json_template={"request": {"message": "{{input}}"}},
        response_json_pointer="/result",
    )
    monkeypatch.setenv("UL_ENVIRONMENT_TOKEN", "Bearer test-token")

    def unexpected_deconstructor(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run constructed a semantic model client")

    monkeypatch.setattr(
        runner_module, "create_semantic_model_deconstructor", unexpected_deconstructor
    )
    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Customer-managed environment API: configured" in result.output
    assert "Authorization=UL_ENVIRONMENT_TOKEN" in result.output
    assert "Bearer test-token" not in result.output
    assert "No model or environment API requests sent" in result.output

    monkeypatch.delenv("UL_ENVIRONMENT_TOKEN")
    missing_environment = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--dry-run",
        ],
    )

    assert missing_environment.exit_code != 0
    assert "environment variable is not set" in missing_environment.output
    assert "No model or environment API requests sent" not in missing_environment.output


@pytest.mark.parametrize(
    "payload",
    [
        {
            "version": 5,
            "environment_id": "test-environment",
            "unknown": True,
        },
        {
            **JsonHttpEnvironmentConfig.model_validate(
                json.loads(
                    (Path(__file__).parents[3] / "examples/stateful_target.json").read_text()
                )
            ).model_dump(mode="json"),
            "execute_turn": {
                "url": "https://environment.example.test/execute",
                "request_json_template": {"input": "missing marker"},
            },
        },
        {
            **JsonHttpEnvironmentConfig.model_validate(
                json.loads(
                    (Path(__file__).parents[3] / "examples/stateful_target.json").read_text()
                )
            ).model_dump(mode="json"),
            "snapshot": {
                "url": "https://environment.example.test/snapshot",
                "response_json_pointer": "not-a-pointer",
            },
        },
    ],
)
def test_dry_run_rejects_invalid_target_config(tmp_path: Path, payload: dict[str, Any]) -> None:
    dataset = tmp_path / "interactions.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    target_config.write_text(json.dumps(payload), encoding="utf-8")

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "No model or environment API requests sent" not in result.output


@pytest.mark.parametrize(
    ("invalid_line", "expected_error"),
    [
        ("not json\n", "line 2: invalid JSON"),
        (
            '{"id":"bad","input":"message","output":NaN}\n',
            "line 2: invalid JSON",
        ),
        (
            '{"id":"first","id":"second","input":"message","output":{}}\n',
            "line 2: invalid JSON",
        ),
        (
            '{"id":"bad","input":"message","output":{"action":"first","action":"second"}}\n',
            "line 2: invalid JSON",
        ),
        (json.dumps({"id": "bad", "input": "message"}) + "\n", "missing output"),
        (
            json.dumps({"id": "bad", "input": "message", "output": {}, "extra": True}) + "\n",
            "unknown field(s)",
        ),
        ("\n", "line 2: blank lines are not allowed"),
    ],
)
def test_preflight_reports_safe_line_numbered_errors_without_external_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_line: str,
    expected_error: str,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    dataset.write_text(json.dumps(_record()) + "\n" + invalid_line, encoding="utf-8")

    def unexpected_settings() -> None:
        raise AssertionError("invalid data reached model setup")

    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", unexpected_settings)
    result = runner.invoke(root_app, ["dataset", "evaluate", str(dataset)])

    assert result.exit_code != 0
    normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", result.output).split())
    assert expected_error in normalized_output
    assert "Transfer 100" not in result.output


def test_preflight_rejects_duplicate_ids_before_external_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record(), _record()])

    def unexpected_settings() -> None:
        raise AssertionError("duplicate data reached model setup")

    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", unexpected_settings)
    result = runner.invoke(root_app, ["dataset", "evaluate", str(dataset)])

    assert result.exit_code != 0
    assert "line 2: duplicate id" in result.output


def test_preflight_rejects_deeply_nested_json(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    nested_output = "[" * 1_100 + "0" + "]" * 1_100
    dataset.write_text(
        f'{{"id":"deep","input":"message","output":{nested_output}}}\n',
        encoding="utf-8",
    )

    result = runner.invoke(root_app, ["dataset", "evaluate", str(dataset), "--dry-run"])

    assert result.exit_code != 0
    assert "line 1: invalid output" in result.output


def test_preflight_rejects_selected_model_input_over_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record(), _record("interaction-2")])
    monkeypatch.setattr(
        command_module,
        "load_dataset_semantic_settings",
        lambda: _settings(max_input_chars=50),
    )

    result = runner.invoke(root_app, ["dataset", "evaluate", str(dataset), "--dry-run"])

    assert result.exit_code != 0
    assert "selected interaction 1 exceeds the semantic model input limit" in result.output


def test_preflight_enforces_record_and_target_call_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record(f"interaction-{index}") for index in range(101)])

    def unexpected_settings() -> None:
        raise AssertionError("oversized data reached model setup")

    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", unexpected_settings)
    too_many_records = runner.invoke(
        root_app,
        ["dataset", "evaluate", str(dataset), "--dry-run"],
    )

    assert too_many_records.exit_code != 0
    assert "line 101: dataset exceeds 100 records" in too_many_records.output

    monkeypatch.setattr(
        command_module,
        "load_dataset_semantic_settings",
        lambda: _settings(),
    )
    _write_dataset(dataset, [_record(f"interaction-{index}") for index in range(17)])
    maximum_calls = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--limit",
            "16",
            "--operator",
            "input.surface.rephrase",
            "--dry-run",
        ],
    )

    assert maximum_calls.exit_code == 0, maximum_calls.output
    assert "Potential environment API calls: up to 96" in maximum_calls.output

    too_many_calls = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--limit",
            "17",
            "--operator",
            "input.surface.rephrase",
            "--dry-run",
        ],
    )

    assert too_many_calls.exit_code != 0
    normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", too_many_calls.output).split())
    assert "would make up to 102 environment API calls" in normalized_output
    assert "--max-environment-api-calls 100" in normalized_output


def test_repetition_budget_is_explicit_and_checked_before_external_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record()])

    def unexpected_settings() -> None:
        raise AssertionError("over-budget repetition plan reached model setup")

    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", unexpected_settings)
    huge_plan = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--repetitions",
            "1000000000",
            "--dry-run",
        ],
    )

    assert huge_plan.exit_code != 0
    normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", huge_plan.output).split())
    assert "repetitions cannot exceed 100" in normalized_output

    monkeypatch.setattr(
        command_module,
        "load_dataset_semantic_settings",
        lambda: _settings(),
    )
    exact_budget = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--repetitions",
            "51",
            "--max-environment-api-calls",
            "102",
            "--dry-run",
        ],
    )

    assert exact_budget.exit_code == 0, exact_budget.output
    assert "Potential environment API calls: up to 102 (authorized maximum: 102)" in " ".join(
        exact_budget.output.split()
    )


@pytest.mark.parametrize(
    "options",
    (
        ("--repetitions", "0"),
        ("--repetitions", "-1"),
        ("--max-environment-api-calls", "0"),
        ("--max-environment-api-calls", "-1"),
    ),
)
def test_repetition_and_call_budget_must_be_positive(
    tmp_path: Path,
    options: tuple[str, str],
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record()])

    result = runner.invoke(
        root_app,
        ["dataset", "evaluate", str(dataset), *options, "--dry-run"],
    )

    assert result.exit_code != 0


def test_default_limit_and_repetitions_fit_the_default_call_budget(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record(f"interaction-{index}") for index in range(11)])

    result = runner.invoke(root_app, ["dataset", "evaluate", str(dataset), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Selected interactions: 10" in result.output
    assert "Operators: input.surface.typing_noise" in result.output
    assert "Potential environment API calls: up to 60" in result.output


def test_isolated_http_target_exposes_customer_selected_concurrency(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record()])

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target",
            "https://agent.example.test/invoke",
            "--confirm-request-isolation",
            "--confirm-safe-test-target",
            "--concurrency",
            "4",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{") :])
    assert payload["timing"]["target_request_concurrency"] == 4


def test_stateful_target_rejects_parallel_requests(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config)

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--concurrency",
            "2",
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "stateful lifecycle targets remain sequential" in " ".join(
        _ANSI_ESCAPE_PATTERN.sub("", result.output).replace("│", "").split()
    )


def test_direct_http_target_requires_tls_or_explicit_loopback_exception(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record()])

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target",
            "http://127.0.0.1:8765/invoke",
            "--confirm-request-isolation",
            "--confirm-safe-test-target",
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    normalized_output = " ".join(
        _ANSI_ESCAPE_PATTERN.sub("", result.output).replace("│", "").split()
    )
    assert "insecure transport opt-in" in normalized_output


def test_direct_http_mapping_options_reject_non_http_targets(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record()])

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target",
            "customer_agent:run",
            "--response-json-pointer",
            "/response",
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "direct HTTP mapping options require an HTTP URL target" in " ".join(
        result.output.split()
    )


def test_direct_http_execution_requires_exact_target_confirmation_before_output(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    _write_dataset(dataset, [_record()])

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target",
            "https://agent.example.test/invoke",
            "--allow-environment-network",
            "--confirm-test-environment",
            "--confirm-request-isolation",
            "--confirm-safe-test-target",
            "--confirm-target",
            "0" * 64,
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    normalized_output = " ".join(
        _ANSI_ESCAPE_PATTERN.sub("", result.output).replace("│", "").split()
    )
    assert (
        "HTTP execution requires --confirm-target with the exact displayed digest"
        in normalized_output
    )
    assert not output.exists()


def test_direct_http_execution_requires_network_opt_in_before_output(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    target_reference = "https://agent.example.test/invoke"
    _write_dataset(dataset, [_record()])
    target = resolve_http_target(
        target_reference,
        allow_insecure_http=False,
        request_isolation_attested=True,
        safe_test_target_attested=True,
    )

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target",
            target_reference,
            "--confirm-test-environment",
            "--confirm-request-isolation",
            "--confirm-safe-test-target",
            "--confirm-target",
            target.confirmation_sha256,
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert "HTTP target execution requires --allow-environment-network" in " ".join(
        _ANSI_ESCAPE_PATTERN.sub("", result.output).replace("│", "").split()
    )
    assert not output.exists()


def test_direct_http_target_requires_explicit_safety_attestations_before_output(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    _write_dataset(dataset, [_record()])

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target",
            "https://agent.example.test/invoke",
            "--allow-environment-network",
            "--confirm-test-environment",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert "direct HTTP targets require --confirm-request-isolation" in " ".join(
        _ANSI_ESCAPE_PATTERN.sub("", result.output).replace("│", "").split()
    )
    assert not output.exists()
