from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from typer.testing import CliRunner
from ul_cli.main import app

runner = CliRunner()


def _write_bfcl_sources(
    directory: Path,
    *,
    count: int,
    answer_count: int | None = None,
) -> tuple[Path, Path]:
    questions = directory / "BFCL_v4_simple_python.json"
    answers = directory / "possible_answer_BFCL_v4_simple_python.json"
    question_lines: list[str] = []
    answer_lines: list[str] = []
    for index in range(count):
        question_lines.append(
            json.dumps(
                {
                    "id": f"simple_python_{index}",
                    "question": [
                        [
                            {
                                "role": "user",
                                "content": f"Calculate value {index}.",
                            }
                        ]
                    ],
                    "function": [
                        {
                            "name": "math.calculate",
                            "description": "Calculate a value.",
                            "parameters": {
                                "type": "dict",
                                "properties": {
                                    "value": {
                                        "type": "float",
                                        "description": "Value to calculate.",
                                    },
                                    "options": {
                                        "type": "dict",
                                        "description": "Optional calculation settings.",
                                        "properties": {
                                            "rounded": {
                                                "type": "bool",
                                                "description": "Round the result.",
                                            }
                                        },
                                    },
                                },
                                "required": ["value"],
                            },
                        }
                    ],
                },
                separators=(",", ":"),
            )
        )
    for index in range(count if answer_count is None else answer_count):
        answer_lines.append(
            json.dumps(
                {
                    "id": f"simple_python_{index}",
                    "ground_truth": [{"math.calculate": {"value": [index]}}],
                },
                separators=(",", ":"),
            )
        )
    questions.write_text("\n".join(question_lines) + "\n", encoding="utf-8")
    answers.write_text("\n".join(answer_lines) + "\n", encoding="utf-8")
    return questions, answers


def _ingest_arguments(
    questions: Path,
    answers: Path,
    output: Path,
    *,
    limit: int,
    seed: int = 7341,
) -> list[str]:
    return [
        "dataset",
        "ingest",
        "bfcl",
        str(questions),
        str(answers),
        "--category",
        "simple_python",
        "--source-revision",
        "bfcl-v4-test-revision",
        "--seed",
        str(seed),
        "--limit",
        str(limit),
        "--output",
        str(output),
    ]


def _record_ids(path: Path) -> list[str]:
    return [json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()]


def test_bfcl_ingest_produces_valid_rich_ul_dataset(tmp_path: Path) -> None:
    questions, answers = _write_bfcl_sources(tmp_path, count=2)
    output = tmp_path / "cohort.jsonl"

    result = runner.invoke(app, _ingest_arguments(questions, answers, output, limit=1))

    assert result.exit_code == 0, result.output
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["inputs"]["messages"][0]["content"].startswith("Calculate value")
    assert record["inputs"]["bfcl_functions"][0]["name"] == "math.calculate"
    function = record["inputs"]["openai_tools"][0]["function"]
    assert function["name"] == "math_calculate"
    assert function["parameters"]["type"] == "object"
    assert function["parameters"]["properties"]["value"]["type"] == "number"
    assert function["parameters"]["properties"]["options"]["type"] == "object"
    assert record["inputs"]["openai_tool_name_map"] == {"math_calculate": "math.calculate"}
    assert record["augmentation_targets"] == [
        {
            "id": "user-request",
            "kind": "input_field",
            "json_pointer": "/inputs/messages/0/content",
            "turn_id": None,
        }
    ]
    source = record["metadata"]["source"]
    assert source["revision"] == "bfcl-v4-test-revision"
    assert len(source["question_sha256"]) == 64
    assert len(source["possible_answer_sha256"]) == 64
    assert source["sampling_algorithm"] == "sha256-seeded-rank-v1"
    assert record["observed_output"]["kind"] == "bfcl_reference"
    if os.name != "nt":
        assert stat.S_IMODE(output.stat().st_mode) == 0o600

    evaluation = runner.invoke(app, ["dataset", "evaluate", str(output), "--dry-run"])

    assert evaluation.exit_code == 0, evaluation.output
    assert "1 interaction" in evaluation.output


def test_bfcl_seeded_cohorts_are_deterministic_and_nested(tmp_path: Path) -> None:
    questions, answers = _write_bfcl_sources(tmp_path, count=100)
    outputs = {limit: tmp_path / f"cohort-{limit}.jsonl" for limit in (1, 10, 100)}

    for limit, output in outputs.items():
        result = runner.invoke(app, _ingest_arguments(questions, answers, output, limit=limit))
        assert result.exit_code == 0, result.output

    one = _record_ids(outputs[1])
    ten = _record_ids(outputs[10])
    hundred = _record_ids(outputs[100])
    assert one == ten[:1]
    assert ten == hundred[:10]

    repeated = tmp_path / "cohort-repeated.jsonl"
    result = runner.invoke(app, _ingest_arguments(questions, answers, repeated, limit=10))
    assert result.exit_code == 0, result.output
    assert _record_ids(repeated) == ten


def test_bfcl_dry_run_prints_only_counts_and_digests(tmp_path: Path) -> None:
    questions, answers = _write_bfcl_sources(tmp_path, count=2)

    result = runner.invoke(
        app,
        [
            "dataset",
            "ingest",
            "bfcl",
            str(questions),
            str(answers),
            "--category",
            "simple_python",
            "--source-revision",
            "bfcl-v4-test-revision",
            "--limit",
            "1",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "1 of 2 BFCL case(s) ready" in result.output
    assert "Question SHA-256:" in result.output
    assert "Possible-answer SHA-256:" in result.output
    assert "Calculate value" not in result.output


def test_bfcl_ingest_rejects_misaligned_ids(tmp_path: Path) -> None:
    questions, answers = _write_bfcl_sources(tmp_path, count=2, answer_count=1)
    output = tmp_path / "cohort.jsonl"

    result = runner.invoke(app, _ingest_arguments(questions, answers, output, limit=1))

    assert result.exit_code == 2
    assert "missing answer(s)" in result.output
    assert not output.exists()


def test_bfcl_ingest_never_overwrites_output(tmp_path: Path) -> None:
    questions, answers = _write_bfcl_sources(tmp_path, count=1)
    output = tmp_path / "cohort.jsonl"
    output.write_text("keep me", encoding="utf-8")

    result = runner.invoke(app, _ingest_arguments(questions, answers, output, limit=1))

    assert result.exit_code == 2
    assert "will not overwrite" in result.output
    assert output.read_text(encoding="utf-8") == "keep me"
