from __future__ import annotations

import json
import re
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import SecretStr
from typer.testing import CliRunner
from ul import DatasetEvaluationResult
from ul_cli import dataset as main
from ul_cli.main import app as root_app

runner = CliRunner()
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _write_dataset(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _record(identifier: str = "interaction-1") -> dict[str, Any]:
    return {
        "id": identifier,
        "input": "Transfer 100 to Alice.",
        "output": {"actions": [{"action": "transfer", "amount": 100, "recipient": "Alice"}]},
    }


def test_dry_run_validates_and_makes_no_external_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record(), _record("interaction-2")])

    def unexpected_deconstructor(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run constructed a semantic model client")

    monkeypatch.setattr(main, "OpenRouterSemanticDeconstructor", unexpected_deconstructor)
    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--operator",
            "surface.disfluency_repeat",
            "--limit",
            "1",
            "--target-url",
            "https://sandbox.example.test/execute",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Dataset valid: 2 interaction(s)" in result.output
    assert "Selected interactions: 1" in result.output
    assert "Potential semantic model calls: up to 6" in result.output
    assert "Potential target calls: up to 2" in result.output
    assert "Semantic models receive historical inputs and outputs" in result.output
    assert "generated variations" in result.output
    assert "live control responses" in result.output
    assert "target receives each selected original input once" in result.output
    assert "then each accepted variation" in " ".join(result.output.split())
    assert "No model or target requests sent." in result.output
    assert "Transfer 100" not in result.output


@pytest.mark.parametrize(
    "options",
    [
        ["--target-url", "file:///etc/passwd"],
        ["--target-url", "https://sandbox.test", "--request-field", "bad.field"],
        [
            "--target-url",
            "https://sandbox.test",
            "--header-env",
            "Host=PATH",
        ],
        [
            "--target-url",
            "https://sandbox.test",
            "--header-env",
            "Authorization=MISSING_SANDBOX_TOKEN",
        ],
    ],
)
def test_dry_run_rejects_invalid_target_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: list[str],
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record()])
    monkeypatch.delenv("MISSING_SANDBOX_TOKEN", raising=False)

    result = runner.invoke(
        root_app,
        ["dataset", "evaluate", str(dataset), *options, "--dry-run"],
    )

    assert result.exit_code != 0
    assert "No model or target requests sent" not in result.output


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

    monkeypatch.setattr(main, "OpenRouterDatasetSettings", unexpected_settings)
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

    monkeypatch.setattr(main, "OpenRouterDatasetSettings", unexpected_settings)
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
        main,
        "OpenRouterDatasetSettings",
        lambda: SimpleNamespace(max_input_chars=50),
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

    monkeypatch.setattr(main, "OpenRouterDatasetSettings", unexpected_settings)
    too_many_records = runner.invoke(
        root_app,
        ["dataset", "evaluate", str(dataset), "--dry-run"],
    )

    assert too_many_records.exit_code != 0
    assert "line 101: dataset exceeds 100 records" in too_many_records.output

    monkeypatch.setattr(
        main,
        "OpenRouterDatasetSettings",
        lambda: SimpleNamespace(max_input_chars=50_000),
    )
    _write_dataset(dataset, [_record(f"interaction-{index}") for index in range(51)])
    maximum_calls = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--limit",
            "50",
            "--operator",
            "surface.rephrase",
            "--dry-run",
        ],
    )

    assert maximum_calls.exit_code == 0, maximum_calls.output
    assert "Potential target calls: up to 100" in maximum_calls.output

    too_many_calls = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--limit",
            "51",
            "--operator",
            "surface.rephrase",
            "--dry-run",
        ],
    )

    assert too_many_calls.exit_code != 0
    assert "would make 102 target calls; maximum is 100" in too_many_calls.output


@pytest.mark.parametrize(
    ("options", "expected_error"),
    [
        ([], "execution requires --target-url"),
        (["--target-url", "https://sandbox.example.test"], "--allow-target-network"),
        (
            [
                "--target-url",
                "https://sandbox.example.test",
                "--allow-target-network",
            ],
            "--confirm-isolated-sandbox",
        ),
        (
            [
                "--target-url",
                "https://sandbox.example.test",
                "--allow-target-network",
                "--confirm-isolated-sandbox",
            ],
            "--confirm-fresh-state",
        ),
        (
            [
                "--target-url",
                "https://sandbox.example.test",
                "--allow-target-network",
                "--confirm-isolated-sandbox",
                "--confirm-fresh-state",
            ],
            "execution requires --output",
        ),
    ],
)
def test_execution_requires_explicit_target_confirmations_and_output(
    tmp_path: Path,
    options: list[str],
    expected_error: str,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record()])

    result = runner.invoke(root_app, ["dataset", "evaluate", str(dataset), *options])

    assert result.exit_code != 0
    normalized_output = " ".join(_ANSI_ESCAPE_PATTERN.sub("", result.output).split())
    assert expected_error in normalized_output


def test_execution_refuses_to_overwrite_output_before_model_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    _write_dataset(dataset, [_record()])
    output.write_text("keep me", encoding="utf-8")

    def unexpected_settings() -> None:
        raise AssertionError("output collision reached model setup")

    monkeypatch.setattr(main, "OpenRouterDatasetSettings", unexpected_settings)
    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target-url",
            "https://sandbox.example.test",
            "--allow-target-network",
            "--confirm-isolated-sandbox",
            "--confirm-fresh-state",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert "will not overwrite" in result.output
    assert output.read_text(encoding="utf-8") == "keep me"


def test_execution_rejects_missing_header_secret_before_model_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    _write_dataset(dataset, [_record()])
    monkeypatch.delenv("MISSING_SANDBOX_TOKEN", raising=False)
    monkeypatch.setattr(
        main,
        "OpenRouterDatasetSettings",
        lambda: SimpleNamespace(
            live_calls=True,
            allow_external_data_processing=True,
            api_key=SecretStr("test-key"),
            max_input_chars=50_000,
        ),
    )

    def unexpected_deconstructor(*args: object, **kwargs: object) -> None:
        raise AssertionError("missing target auth reached semantic model setup")

    monkeypatch.setattr(main, "OpenRouterSemanticDeconstructor", unexpected_deconstructor)
    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--target-url",
            "https://sandbox.example.test",
            "--header-env",
            "Authorization=MISSING_SANDBOX_TOKEN",
            "--allow-target-network",
            "--confirm-isolated-sandbox",
            "--confirm-fresh-state",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert not output.exists()


def test_execution_creates_private_explicit_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    output = tmp_path / "results.jsonl"
    _write_dataset(dataset, [_record()])
    captured_records: list[str] = []

    class FakeTarget:
        def __init__(self, endpoint: str, **options: object) -> None:
            assert endpoint == "http://127.0.0.1:8765/execute"
            assert options["sandbox_confirmed"] is True
            assert options["fresh_state_confirmed"] is True
            assert options["request_field"] == "query"

    async def fake_evaluate(
        records: tuple[Any, ...],
        operator_ids: tuple[str, ...],
        settings: object,
        target: object,
        output_stream: Any,
    ) -> tuple[object, ...]:
        del settings, target
        captured_records.extend(record.id for record in records)
        assert operator_ids == ("surface.disfluency_repeat",)
        output_stream.write('{"saved":true}\n')
        output_stream.flush()
        return ()

    monkeypatch.setattr(
        main,
        "OpenRouterDatasetSettings",
        lambda: SimpleNamespace(
            live_calls=True,
            allow_external_data_processing=True,
            api_key=SecretStr("test-key"),
            max_input_chars=50_000,
        ),
    )
    monkeypatch.setattr(main, "JsonHttpDatasetTarget", FakeTarget)
    monkeypatch.setattr(main, "_evaluate_interaction_records", fake_evaluate)
    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--operator",
            "surface.disfluency_repeat",
            "--target-url",
            "http://127.0.0.1:8765/execute",
            "--request-field",
            "query",
            "--allow-insecure-http",
            "--allow-target-network",
            "--confirm-isolated-sandbox",
            "--confirm-fresh-state",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured_records == ["interaction-1"]
    assert output.read_text(encoding="utf-8") == '{"saved":true}\n'
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert "Complete evidence" in result.output
    assert "Transfer 100" not in result.output


def test_help_explains_dataset_target_and_operator_contract() -> None:
    result = runner.invoke(root_app, ["dataset", "evaluate", "--help"])

    assert result.exit_code == 0, result.output
    assert '"id"' in result.output
    assert '"input"' in result.output
    assert '"output"' in result.output
    normalized_help = " ".join(result.output.split())
    assert "Simple sandbox" in normalized_help
    assert "POST" in result.output
    assert "non-null JSON" in normalized_help
    assert "UL_DATASET_LIVE_CALLS" in result.output
    assert "same clean state" in normalized_help
    assert "operator. Run 'ul" in normalized_help
    assert "operators' for" in normalized_help

    operators = runner.invoke(root_app, ["dataset", "operators"])
    assert operators.exit_code == 0, operators.output
    assert "surface.disfluency_repeat" in operators.output
    assert "tone.frustrated" in operators.output
    assert "intent.self_correction" in operators.output


def test_operator_list_is_fixed_and_self_correction_keeps_existing_call_accounting(
    tmp_path: Path,
) -> None:
    operators = runner.invoke(root_app, ["dataset", "operators"])

    assert operators.exit_code == 0, operators.output
    listed_operator_ids = tuple(
        line.removeprefix("- ").split(" ", 1)[0]
        for line in operators.output.splitlines()
        if line.startswith("- ")
    )
    assert listed_operator_ids == (
        "surface.rephrase",
        "surface.typing_noise",
        "surface.fragmented_syntax",
        "surface.disfluency_repeat",
        "style.terse",
        "style.verbose",
        "tone.frustrated",
        "intent.self_correction",
    )

    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record()])
    dry_run = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--operator",
            "intent.self_correction",
            "--dry-run",
        ],
    )

    assert dry_run.exit_code == 0, dry_run.output
    assert "Operators: intent.self_correction" in dry_run.output
    assert "Potential semantic model calls: up to 6" in dry_run.output
    assert "Potential target calls: up to 2" in dry_run.output


@pytest.mark.parametrize(
    "operators",
    [
        ("intent.self_correction", "intent.self_correction"),
        ("intent.self-correction",),
    ],
)
def test_cli_rejects_duplicate_or_unknown_self_correction_operator_before_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operators: tuple[str, ...],
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    _write_dataset(dataset, [_record()])

    def unexpected_settings() -> None:
        raise AssertionError("invalid operator selection reached model setup")

    monkeypatch.setattr(main, "OpenRouterDatasetSettings", unexpected_settings)
    arguments = ["dataset", "evaluate", str(dataset)]
    for operator_id in operators:
        arguments.extend(("--operator", operator_id))

    result = runner.invoke(root_app, arguments)

    assert result.exit_code != 0
    assert "operator" in result.output.casefold()


def test_customer_evidence_keeps_summary_and_nested_technical_details() -> None:
    expected_effect = SimpleNamespace(
        model_dump=lambda **kwargs: {"kind": "action", "predicate": "transfer"}
    )
    finding = SimpleNamespace(
        category="duplicate_effect",
        message="A duplicate action needs review.",
        expected_effects=(expected_effect,),
        observed_effects=(expected_effect, expected_effect),
    )
    candidate = SimpleNamespace(
        operator_id="surface.disfluency_repeat",
        operator_version="1.0.0",
        augmented_input="transfer transfer 100 to Alice",
        passed=True,
        failure_reasons=(),
    )
    case = SimpleNamespace(
        candidate=candidate,
        verdict="divergence_needs_review",
        findings=(finding,),
        inconclusive_reasons=(),
    )
    result = cast(
        DatasetEvaluationResult,
        SimpleNamespace(
            source=SimpleNamespace(id="case-1", raw_input="transfer 100 to Alice"),
            baseline=SimpleNamespace(
                verdict="no_divergence",
                findings=(),
                inconclusive_reasons=(),
            ),
            cases=(case,),
            model_dump=lambda **kwargs: {"full": "technical evidence"},
        ),
    )

    evidence = main._customer_evidence_record(result)

    assert main._result_needs_review(result) is True
    assert evidence["interaction_id"] == "case-1"
    assert evidence["original_input"] == "transfer 100 to Alice"
    assert evidence["schema_version"] == "1.1.0"
    assert evidence["current_baseline"]["status"] == "LIVE CONTROL MATCHES STORED RUN"
    assert evidence["cases"][0]["status"] == "DIFFERENCE — REVIEW"
    assert evidence["cases"][0]["findings"][0]["expected_effects"] == [
        {"kind": "action", "predicate": "transfer"}
    ]
    assert evidence["technical_details"] == {"full": "technical evidence"}


def test_live_control_drift_is_shown_and_requires_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finding = SimpleNamespace(
        category="changed_grounded_effect_argument",
        message="The live control changed an action value.",
        expected_effects=(),
        observed_effects=(),
    )
    result = cast(
        DatasetEvaluationResult,
        SimpleNamespace(
            source=SimpleNamespace(id="case-1", raw_input="transfer 100 to Alice"),
            baseline=SimpleNamespace(
                verdict="divergence_needs_review",
                findings=(finding,),
                inconclusive_reasons=(),
            ),
            cases=(),
        ),
    )
    printed_rows: list[tuple[str, ...]] = []

    class CapturingTable:
        def add_column(self, *args: object, **kwargs: object) -> None:
            pass

        def add_row(self, *values: str) -> None:
            printed_rows.append(values)

    monkeypatch.setattr(main, "Table", lambda **kwargs: CapturingTable())
    monkeypatch.setattr(main.console, "print", lambda *args, **kwargs: None)

    main._print_dataset_results((result,), tmp_path / "evidence.jsonl")

    assert main._result_needs_review(result) is True
    assert printed_rows == [
        (
            "1",
            "original replay",
            "LIVE CONTROL DIFFERS — REVIEW",
            "changed action value",
        )
    ]
