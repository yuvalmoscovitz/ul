from __future__ import annotations

import json
import re
from typing import Any, cast

from pydantic import SecretStr
from typer.testing import CliRunner
from ul import (
    JsonHttpEnvironmentConfig,
    OpenAICompatibleDatasetSettings,
)
from ul_cli import dataset_review
from ul_cli.dataset.evidence import context as context_module

from ._factories import (
    _evaluation_result,
    _isolated_response_target_config,
    _run_context,
)

runner = CliRunner()
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def test_run_context_uses_current_pipeline() -> None:
    record = _evaluation_result("interaction-1").source
    run_context = _run_context((record,))
    assert run_context.schema_version == "1.4.0"
    assert run_context.pipeline_version == "1.5.0"
    assert run_context.evaluation_mode == "variance"
    assert run_context.target.config.reset.reset_session is True
    assert run_context.target.config.reset.reset_env is True
    assert run_context.fixture.status == "missing"


def test_run_context_records_versioned_fixture_identity() -> None:
    record = _evaluation_result("interaction-1").source
    raw_config = cast(Any, _run_context((record,))).target.config.model_dump(mode="json")
    raw_config["fixture_id"] = "standard-account"
    raw_config["fixture_version"] = "2026-08-22"

    run_context = _run_context(
        (record,), target_config=JsonHttpEnvironmentConfig.model_validate(raw_config)
    )

    assert run_context.schema_version == "1.4.0"
    assert run_context.fixture.model_dump(mode="json") == {
        "status": "configured",
        "id": "standard-account",
        "version": "2026-08-22",
    }


def test_run_context_marks_fixture_not_required_for_isolated_target() -> None:
    record = _evaluation_result("interaction-1").source

    run_context = _run_context((record,), target_config=_isolated_response_target_config())

    assert run_context.fixture.status == "not_required"


def test_run_context_accepts_pre_fixture_evaluation_mode_evidence() -> None:
    record = _evaluation_result("interaction-1").source
    current = cast(Any, _run_context((record,))).model_dump(mode="json")
    current["schema_version"] = "1.2.0"
    current["pipeline_version"] = "1.3.0"
    current.pop("fixture")
    current.pop("redaction_policy_sha256")
    current.pop("redaction_coverage")
    current["context_sha256"] = dataset_review._canonical_json_sha256(
        {key: value for key, value in current.items() if key != "context_sha256"}
    )

    loaded = dataset_review.DatasetEvidenceRunContext.model_validate_json(json.dumps(current))

    assert loaded.evaluation_mode == "variance"
    assert loaded.fixture is None


def test_run_context_records_canonical_provider_identity() -> None:
    record = _evaluation_result("interaction-1").source
    openrouter_context = cast(Any, _run_context((record,)))

    custom_settings = OpenAICompatibleDatasetSettings(
        live_calls=True,
        allow_external_data_processing=True,
        api_key=SecretStr("test-key"),
        provider_id="customer-gateway",
        base_url="https://models.example.test/v1",
        model="customer/model",
    )
    custom_context = context_module.build_dataset_evidence_run_context(
        selected_records=(record,),
        selected_operator_ids=("input.surface.rephrase",),
        repetitions=1,
        invariant_suite=None,
        target_config=JsonHttpEnvironmentConfig.model_validate(
            {
                "version": 5,
                "environment_id": "test-environment",
                "reset": {
                    "url": "https://environment.example.test/reset",
                    "generation_json_pointer": "/generation",
                    "clean_state_json_pointer": "/clean",
                    "clean_state_value": True,
                },
                "execute_turn": {
                    "url": "https://environment.example.test/execute",
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "turn_id": "{{turn_id}}",
                        "input": "{{input}}",
                    },
                },
                "snapshot": {
                    "url": "https://environment.example.test/snapshot",
                    "request_json_template": {
                        "case_id": "{{case_id}}",
                        "turn_id": "{{turn_id}}",
                    },
                },
            }
        ),
        settings=custom_settings,
    )

    assert custom_context.semantic_settings.provider == "customer-gateway"
    assert len(custom_context.semantic_settings.endpoint_sha256) == 64
    assert "https://models.example.test/v1" not in custom_context.model_dump_json()
    assert custom_context.context_sha256 != openrouter_context.context_sha256
