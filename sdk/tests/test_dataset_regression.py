from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError
from ul.dataset_invariants import JsonValuesEqualInvariant
from ul.dataset_regression import (
    DatasetRegressionCase,
    create_dataset_regression_case,
    load_dataset_regression_case,
    replay_dataset_regression,
    run_dataset_regressions,
)
from ul.http_target import JsonHttpDatasetTargetConfig
from ul_core.dataset import ObservedAgentOutput
from ul_core.models import SafetyEnvelope

FINDING_ID = f"ulf_v1_{'1' * 64}"
REVIEW_ID = "ulr_00000000-0000-4000-8000-000000000000"
SUITE_SHA256 = "2" * 64


def _rule(*, rule_id: str = "invoice-matches-request") -> JsonValuesEqualInvariant:
    return JsonValuesEqualInvariant(
        type="json_values_equal",
        id=rule_id,
        version="1.0.0",
        description="The committed invoice must match the requested invoice.",
        severity="high",
        left_pointer="/invoice_reference",
        right_pointer="/requested_invoice_reference",
    )


def _case(
    *,
    repetitions: int = 3,
    variation_input: str = "Pay invoice AC-101 instead of AC-100.",
) -> DatasetRegressionCase:
    return create_dataset_regression_case(
        finding_id=FINDING_ID,
        evidence_sha256="3" * 64,
        review_id=REVIEW_ID,
        interaction_id="invoice-correction",
        operator_id="context.pasted_block",
        operator_version="1.0.0",
        original_input="Pay invoice AC-100.",
        variation_input=variation_input,
        target_config=JsonHttpDatasetTargetConfig(
            version=1,
            url="http://127.0.0.1:8765/execute",
            headers_from_env={"Authorization": "TARGET_TOKEN"},
            request_json_template={"input": "{{input}}"},
        ),
        source_suite_sha256=SUITE_SHA256,
        observation_authority="committed_state_snapshot",
        selected_rules=(_rule(),),
        discovery_repetitions=repetitions,
    )


class _Target:
    def __init__(self, outcomes: list[ObservedAgentOutput | RuntimeError]) -> None:
        self.outcomes = outcomes
        self.inputs: list[str] = []
        self.safety_envelope = SafetyEnvelope(
            description="isolated regression target",
            isolated=True,
            allows_network_egress=False,
            allows_business_side_effects=False,
        )

    @property
    def fresh_state_per_execution(self) -> bool:
        return True

    async def execute(self, raw_input: str) -> ObservedAgentOutput:
        self.inputs.append(raw_input)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, RuntimeError):
            raise outcome
        return outcome


def _output(invoice: str, requested: str) -> ObservedAgentOutput:
    return ObservedAgentOutput(
        raw_output={
            "invoice_reference": invoice,
            "requested_invoice_reference": requested,
        }
    )


def test_case_is_content_addressed_and_loader_round_trips(tmp_path: Path) -> None:
    case = _case()
    equivalent = _case()
    changed = create_dataset_regression_case(
        finding_id=FINDING_ID,
        evidence_sha256="3" * 64,
        review_id=REVIEW_ID,
        interaction_id="invoice-correction",
        operator_id="context.pasted_block",
        operator_version="1.0.0",
        original_input="Pay invoice AC-100.",
        variation_input="Pay invoice AC-102 instead of AC-100.",
        target_config=case.target.config,
        source_suite_sha256=SUITE_SHA256,
        observation_authority="committed_state_snapshot",
        selected_rules=(_rule(),),
        discovery_repetitions=3,
    )
    path = tmp_path / "case.json"
    path.write_text(case.model_dump_json(indent=2), encoding="utf-8")

    assert case.case_id == equivalent.case_id
    assert changed.case_id != case.case_id
    assert load_dataset_regression_case(path) == case
    assert case.target.provenance == "declared_at_case_creation"
    assert case.target.config_sha256 == equivalent.target.config_sha256


def test_case_rejects_tampered_case_and_target_digests() -> None:
    serialized = _case().model_dump(mode="json")
    serialized["variation"]["variation_input"] = "tampered"  # type: ignore[index]
    with pytest.raises(ValidationError, match="case ID must match"):
        DatasetRegressionCase.model_validate(serialized)

    serialized = _case().model_dump(mode="json")
    serialized["target"]["config_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ValidationError, match="target config digest must match"):
        DatasetRegressionCase.model_validate(serialized)


@pytest.mark.parametrize(
    ("outputs", "expected_status"),
    [
        ([_output("AC-101", "AC-101")] * 3, "passed"),
        ([_output("AC-100", "AC-101")] * 3, "failed"),
        ([RuntimeError("secret provider detail")] * 3, "inconclusive"),
        (
            [RuntimeError("failed"), _output("AC-100", "AC-101"), RuntimeError("failed")],
            "failed",
        ),
    ],
)
def test_replay_runs_only_exact_variation_sequentially_and_aggregates(
    outputs: list[ObservedAgentOutput | RuntimeError], expected_status: str
) -> None:
    case = _case()
    target = _Target(outputs.copy())

    result = asyncio.run(replay_dataset_regression(case, target))

    assert result.status == expected_status
    assert target.inputs == [case.variation.variation_input] * 3
    assert [execution.repetition for execution in result.executions] == [1, 2, 3]
    assert "secret provider detail" not in result.model_dump_json()
    if expected_status == "failed":
        assert result.rules[0].status == "violated"


def test_replay_enforces_target_safety_and_network_opt_in() -> None:
    target = _Target([_output("AC-101", "AC-101")] * 3)
    target.safety_envelope = SafetyEnvelope(
        description="networked target",
        isolated=True,
        allows_network_egress=True,
        allows_business_side_effects=False,
    )

    with pytest.raises(ValueError, match="network egress requires explicit opt-in"):
        asyncio.run(replay_dataset_regression(_case(), target))


@pytest.mark.parametrize("max_target_calls", [True, 0, -1])
def test_replay_enforces_sdk_target_call_budget_before_execution(
    max_target_calls: object,
) -> None:
    target = _Target([_output("AC-101", "AC-101")] * 3)

    with pytest.raises(ValueError, match="max_target_calls"):
        asyncio.run(
            replay_dataset_regression(
                _case(),
                target,
                max_target_calls=max_target_calls,  # type: ignore[arg-type]
            )
        )

    assert target.inputs == []


def test_replay_rejects_case_over_sdk_target_call_budget_before_execution() -> None:
    target = _Target([_output("AC-101", "AC-101")] * 3)

    with pytest.raises(ValueError, match="authorized target call budget"):
        asyncio.run(replay_dataset_regression(_case(), target, max_target_calls=2))

    assert target.inputs == []


def test_run_executes_cases_in_order_and_aggregates_statuses() -> None:
    passing_case = _case(repetitions=2)
    failing_case = _case(
        repetitions=2,
        variation_input="Pay invoice AC-102 instead of AC-100.",
    )
    target = _Target(
        [
            _output("AC-101", "AC-101"),
            _output("AC-101", "AC-101"),
            _output("AC-100", "AC-102"),
            RuntimeError("provider detail"),
        ]
    )

    result = asyncio.run(
        run_dataset_regressions(
            (passing_case, failing_case),
            target,
            case_labels=("invoice-correction.json", "invoice-substitution.json"),
        )
    )

    assert result.status == "failed"
    assert result.passed_case_count == 1
    assert result.failed_case_count == 1
    assert result.inconclusive_case_count == 0
    assert result.requested_target_calls == 4
    assert [case_result.label for case_result in result.cases] == [
        "invoice-correction.json",
        "invoice-substitution.json",
    ]
    assert [case_result.result.case_id for case_result in result.cases] == [
        passing_case.case_id,
        failing_case.case_id,
    ]
    assert target.inputs == [
        passing_case.variation.variation_input,
        passing_case.variation.variation_input,
        failing_case.variation.variation_input,
        failing_case.variation.variation_input,
    ]
    assert "provider detail" not in result.model_dump_json()


def test_run_returns_inconclusive_when_no_case_fails() -> None:
    passing_case = _case(repetitions=1)
    inconclusive_case = _case(
        repetitions=1,
        variation_input="Pay invoice AC-102 instead of AC-100.",
    )
    target = _Target([_output("AC-101", "AC-101"), RuntimeError("unavailable")])

    result = asyncio.run(run_dataset_regressions((passing_case, inconclusive_case), target))

    assert result.status == "inconclusive"
    assert result.passed_case_count == 1
    assert result.inconclusive_case_count == 1


def test_run_enforces_total_budget_and_unique_cases_before_execution() -> None:
    first_case = _case(repetitions=2)
    second_case = _case(
        repetitions=3,
        variation_input="Pay invoice AC-102 instead of AC-100.",
    )
    target = _Target([_output("AC-101", "AC-101")] * 5)

    with pytest.raises(ValueError, match="authorized target call budget"):
        asyncio.run(
            run_dataset_regressions(
                (first_case, second_case),
                target,
                max_target_calls=4,
            )
        )
    assert target.inputs == []

    with pytest.raises(ValueError, match="case IDs must be unique"):
        asyncio.run(run_dataset_regressions((first_case, first_case), target))
    assert target.inputs == []

    with pytest.raises(ValueError, match="labels must match"):
        asyncio.run(
            run_dataset_regressions(
                (first_case, second_case),
                target,
                case_labels=("only-one.json",),
            )
        )
    assert target.inputs == []

    for invalid_label in ("", "\udcff"):
        with pytest.raises(ValueError, match="1 to 500 characters"):
            asyncio.run(
                run_dataset_regressions(
                    (first_case, second_case),
                    target,
                    case_labels=(invalid_label, "second.json"),
                )
            )
        assert target.inputs == []


def test_loader_rejects_duplicate_deep_and_oversized_input(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"1.0.0","schema_version":"1.0.0"}')
    with pytest.raises(ValueError, match="invalid JSON"):
        load_dataset_regression_case(duplicate)

    deep = tmp_path / "deep.json"
    deep.write_text("[" * 102 + "0" + "]" * 102)
    with pytest.raises(ValueError, match="invalid JSON"):
        load_dataset_regression_case(deep)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * 1_000_001)
    with pytest.raises(ValueError, match="size limit"):
        load_dataset_regression_case(oversized)


def test_loader_requires_strict_schema(tmp_path: Path) -> None:
    serialized = _case().model_dump(mode="json")
    serialized["unexpected"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(serialized), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected: Extra inputs are not permitted"):
        load_dataset_regression_case(path)


def test_loader_rejects_symbolic_links_and_nonregular_files(tmp_path: Path) -> None:
    case_path = tmp_path / "case.json"
    case_path.write_text(_case().model_dump_json(), encoding="utf-8")
    link_path = tmp_path / "case-link.json"
    link_path.symlink_to(case_path)

    with pytest.raises(RuntimeError, match="could not be read"):
        load_dataset_regression_case(link_path)

    if hasattr(os, "mkfifo"):
        fifo_path = tmp_path / "case-fifo"
        os.mkfifo(fifo_path)
        with pytest.raises(RuntimeError, match="could not be read"):
            load_dataset_regression_case(fifo_path)
