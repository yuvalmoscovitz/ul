from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

runner = CliRunner()
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _write_invariant_suite(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "observation_source": "target_output",
                "observation_authority": "committed_state_snapshot",
                "rules": [
                    {
                        "type": "json_values_equal",
                        "id": "amount-matches-corrected",
                        "version": "1.0.0",
                        "description": "Final amount equals the corrected amount.",
                        "severity": "high",
                        "left_pointer": "/final_amount",
                        "right_pointer": "/corrected_amount",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_stateful_target_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 5,
                "environment_id": "test-environment",
                "headers_from_env": {},
                "reset": {
                    "url": "https://environment.example.test/reset",
                    "request_json_template": {"case_id": "{{case_id}}"},
                    "case_id_json_pointer": "/case_id",
                    "generation_json_pointer": "/generation",
                    "clean_state_json_pointer": "/clean",
                    "clean_state_value": True,
                },
                "setup": {
                    "url": "https://environment.example.test/setup",
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "seed": "standard",
                    },
                    "case_id_json_pointer": "/case_id",
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
                },
            }
        ),
        encoding="utf-8",
    )


def _write_dataset(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_target_config(
    path: Path,
    *,
    url: str = "https://environment.example.test/execute",
    headers_from_env: dict[str, str] | None = None,
    request_json_template: object | None = None,
    response_json_pointer: str = "",
) -> None:
    base_url = url.removesuffix("/execute")
    path.write_text(
        json.dumps(
            {
                "version": 5,
                "environment_id": "test-environment",
                "headers_from_env": headers_from_env or {},
                "reset": {
                    "url": f"{base_url}/reset",
                    "request_json_template": {"case_id": "{{case_id}}"},
                    "case_id_json_pointer": "/case_id",
                    "generation_json_pointer": "/generation",
                    "clean_state_json_pointer": "/clean",
                    "clean_state_value": True,
                },
                "execute_turn": {
                    "url": url,
                    "request_json_template": (
                        {
                            "case_id": "{{case_id}}",
                            "turn_id": "{{turn_id}}",
                            **request_json_template,
                        }
                        if isinstance(request_json_template, dict)
                        else (
                            request_json_template
                            if request_json_template is not None
                            else {
                                "case_id": "{{case_id}}",
                                "turn_id": "{{turn_id}}",
                                "input": "{{input}}",
                            }
                        )
                    ),
                    "response_json_pointer": response_json_pointer,
                    "case_id_json_pointer": "/case_id",
                    "turn_id_json_pointer": "/turn_id",
                },
                "snapshot": {
                    "url": f"{base_url}/snapshot",
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "turn_id": "{{turn_id}}",
                    },
                    "case_id_json_pointer": "/case_id",
                    "turn_id_json_pointer": "/turn_id",
                },
            }
        ),
        encoding="utf-8",
    )


def _record(identifier: str = "interaction-1") -> dict[str, Any]:
    return {
        "id": identifier,
        "input": "Transfer 100 to Alice.",
        "output": {"actions": [{"action": "transfer", "amount": 100, "recipient": "Alice"}]},
    }
