from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import pytest
from pydantic import ValidationError
from ul import (
    DatasetInvariantArmEvaluation,
    DatasetInvariantEvaluation,
    DatasetInvariantRuleEvaluation,
    DatasetInvariantSuite,
    DatasetInvariantTrialEvaluation,
    JsonValuesEqualInvariant,
    evaluate_dataset_invariants,
    load_dataset_invariant_suite,
)
from ul.dataset_evaluation import DatasetEvaluationResult
from ul.dataset_invariants import _resolve_json_pointer
from ul_core.dataset import ObservedAgentOutput

_MISSING = object()


def _rule(**overrides: object) -> JsonValuesEqualInvariant:
    values: dict[str, object] = {
        "type": "json_values_equal",
        "id": "invoice-values-match",
        "version": "1.0.0",
        "description": "The selected invoice values must match.",
        "severity": "high",
        "left_pointer": "/request/invoice_reference",
        "right_pointer": "/action/invoice_reference",
    }
    values.update(overrides)
    return JsonValuesEqualInvariant.model_validate(values)


def _suite(*rules: JsonValuesEqualInvariant) -> DatasetInvariantSuite:
    return DatasetInvariantSuite(
        schema_version="1.0.0",
        observation_source="target_output",
        observation_authority="agent_response",
        rules=rules or (_rule(),),
    )


def _trial(repetition: int, raw_output: object = _MISSING) -> SimpleNamespace:
    target_output = (
        None
        if raw_output is _MISSING
        else ObservedAgentOutput.model_construct(raw_output=raw_output, metadata={})
    )
    return SimpleNamespace(repetition=repetition, target_output=target_output)


def _case(
    operator_id: str,
    outputs: list[object],
    *,
    passed: bool = True,
    executed: bool = True,
) -> SimpleNamespace:
    trial_set = (
        SimpleNamespace(
            trials=tuple(_trial(index, output) for index, output in enumerate(outputs, start=1))
        )
        if executed
        else None
    )
    return SimpleNamespace(
        candidate=SimpleNamespace(operator_id=operator_id, passed=passed),
        trial_set=trial_set,
    )


def _evaluation_result(
    baseline_outputs: list[object],
    *cases: SimpleNamespace,
) -> DatasetEvaluationResult:
    return cast(
        DatasetEvaluationResult,
        SimpleNamespace(
            source=SimpleNamespace(
                id="interaction-1",
                raw_observed_output={"request": "historical output is not authoritative"},
            ),
            baseline=SimpleNamespace(
                trial_set=SimpleNamespace(
                    trials=tuple(
                        _trial(index, output)
                        for index, output in enumerate(baseline_outputs, start=1)
                    )
                )
            ),
            cases=cases,
        ),
    )


def _output(left: object, right: object) -> dict[str, object]:
    return {
        "request": {"invoice_reference": left},
        "action": {"invoice_reference": right},
    }


def test_suite_is_strict_bounded_unique_and_content_addressed(tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    payload = {
        "schema_version": "1.0.0",
        "observation_source": "target_output",
        "observation_authority": "committed_state_snapshot",
        "rules": [_rule().model_dump(mode="json")],
    }
    first_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    second_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    first = load_dataset_invariant_suite(first_path)
    second = load_dataset_invariant_suite(second_path)

    assert first == second
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64
    assert first.observation_source == "target_output"
    assert first.observation_authority == "committed_state_snapshot"
    assert first.model_dump(mode="json") == payload

    with pytest.raises(ValidationError, match="unique"):
        DatasetInvariantSuite(
            schema_version="1.0.0",
            observation_source="target_output",
            observation_authority="agent_response",
            rules=(_rule(), _rule()),
        )
    with pytest.raises(ValidationError):
        DatasetInvariantSuite(
            schema_version="1.0.0",
            observation_source="target_output",
            observation_authority="agent_response",
            rules=tuple(_rule(id=f"rule-{index}") for index in range(101)),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": "not valid"},
        {"version": "not valid"},
        {"description": "   "},
        {"description": "x" * 501},
        {"severity": "unrated"},
        {"left_pointer": "missing-leading-slash"},
        {"right_pointer": "/bad~2escape"},
        {"right_pointer": "/request/invoice_reference"},
        {"unknown": True},
    ],
)
def test_rule_rejects_invalid_or_unknown_fields(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _rule(**overrides)


@pytest.mark.parametrize(
    "encoded_suite",
    [
        b"not json",
        b"\xff",
        b'{"schema_version":"1.0.0","schema_version":"1.0.0"}',
        b'{"number":NaN}',
        b'{"number":Infinity}',
        b'{"number":1e999}',
        (b'{"nested":' + b"[" * 101 + b"0" + b"]" * 101 + b"}"),
    ],
)
def test_loader_rejects_adversarial_json(tmp_path: Path, encoded_suite: bytes) -> None:
    path = tmp_path / "invariants.json"
    path.write_bytes(encoded_suite)

    with pytest.raises(ValueError, match="contains invalid JSON"):
        load_dataset_invariant_suite(path)


def test_loader_enforces_size_and_sanitizes_schema_errors(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * 1_000_001)
    with pytest.raises(ValueError, match="size limit"):
        load_dataset_invariant_suite(oversized)

    secret = "do-not-echo-this-value"
    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "observation_source": "target_output",
                "observation_authority": "agent_response",
                "rules": [
                    {
                        **_rule().model_dump(mode="json"),
                        "severity": secret,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as raised:
        load_dataset_invariant_suite(invalid)
    assert secret not in str(raised.value)
    assert "rules.0.severity" in str(raised.value)


@pytest.mark.parametrize(
    ("pointer", "expected"),
    [
        ("", {"a/b": {"~key": ["zero", "one"]}}),
        ("/a~1b/~0key/0", "zero"),
        ("/a~1b/~0key/1", "one"),
    ],
)
def test_json_pointer_resolves_root_escapes_and_ascii_array_indices(
    pointer: str, expected: object
) -> None:
    document = {"a/b": {"~key": ["zero", "one"]}}

    assert _resolve_json_pointer(document, pointer) == (True, expected)
    assert _resolve_json_pointer(document, "/a~1b/~0key/01") == (False, None)
    assert _resolve_json_pointer(document, "/a~1b/~0key/\N{ARABIC-INDIC DIGIT ONE}") == (
        False,
        None,
    )
    assert _resolve_json_pointer(document, "/missing") == (False, None)


@pytest.mark.parametrize(
    ("left", "right", "expected_status", "expected_reason", "expected_values"),
    [
        ("AC-100", "AC-100", "satisfied", "values_equal", {"left": "AC-100", "right": "AC-100"}),
        ("AC-100", "ac-100", "violated", "values_differ", {"left": "AC-100", "right": "ac-100"}),
        (True, True, "satisfied", "values_equal", {"left": True, "right": True}),
        (True, False, "violated", "values_differ", {"left": True, "right": False}),
        (None, None, "satisfied", "values_equal", {"left": None, "right": None}),
        (1, 1, "satisfied", "values_equal", {"left": 1, "right": 1}),
        (1, 2, "violated", "values_differ", {"left": 1, "right": 2}),
        (1.25, 1.25, "not_evaluable", "left_non_integer_number_not_supported", {}),
        (1, 2.0, "not_evaluable", "right_non_integer_number_not_supported", {"left": 1}),
        (1, "1", "not_evaluable", "operand_types_differ", {"left": 1, "right": "1"}),
        (True, 1, "not_evaluable", "operand_types_differ", {"left": True, "right": 1}),
        (None, "", "not_evaluable", "operand_types_differ", {"left": None, "right": ""}),
    ],
)
def test_scalar_equality_is_exact_and_numeric_only_within_numeric_types(
    left: object,
    right: object,
    expected_status: str,
    expected_reason: str,
    expected_values: dict[str, object],
) -> None:
    evaluated = evaluate_dataset_invariants(
        _evaluation_result([_output(left, right)]),
        _suite(),
    )
    trial = evaluated.baseline.rules[0].trials[0]

    assert trial.status == expected_status
    assert trial.reason_code == expected_reason
    assert trial.resolved_values == expected_values


def test_binary_float_rounding_cannot_create_a_false_satisfaction() -> None:
    trial = (
        evaluate_dataset_invariants(
            _evaluation_result([_output(9_007_199_254_740_992, 9_007_199_254_740_993.0)]),
            _suite(),
        )
        .baseline.rules[0]
        .trials[0]
    )

    assert trial.status == "not_evaluable"
    assert trial.reason_code == "right_non_integer_number_not_supported"


def test_large_selected_values_are_not_duplicated_into_invariant_evidence() -> None:
    trial = (
        evaluate_dataset_invariants(
            _evaluation_result([_output("x" * 4_097, "x" * 4_097)]),
            _suite(),
        )
        .baseline.rules[0]
        .trials[0]
    )

    assert trial.status == "not_evaluable"
    assert trial.reason_code == "left_value_exceeds_limit"
    assert trial.resolved_values == {}

    huge_integer_trial = (
        evaluate_dataset_invariants(
            _evaluation_result([_output(10**5_000, 10**5_000)]),
            _suite(),
        )
        .baseline.rules[0]
        .trials[0]
    )
    assert huge_integer_trial.status == "not_evaluable"
    assert huge_integer_trial.reason_code == "left_value_exceeds_limit"


@pytest.mark.parametrize(
    ("raw_output", "expected_reason", "expected_values"),
    [
        (_MISSING, "target_output_missing", {}),
        ({"action": {"invoice_reference": "AC-100"}}, "left_pointer_missing", {}),
        (
            {"request": {"invoice_reference": "AC-100"}},
            "right_pointer_missing",
            {"left": "AC-100"},
        ),
        (
            _output({"nested": "AC-100"}, "AC-100"),
            "left_value_not_scalar",
            {},
        ),
        (
            _output("AC-100", ["AC-100"]),
            "right_value_not_scalar",
            {"left": "AC-100"},
        ),
        (
            _output(float("inf"), float("inf")),
            "left_value_not_scalar",
            {},
        ),
        (
            _output("AC-100", float("nan")),
            "right_value_not_scalar",
            {"left": "AC-100"},
        ),
    ],
)
def test_missing_failed_non_scalar_and_nonfinite_trials_are_not_evaluable(
    raw_output: object,
    expected_reason: str,
    expected_values: dict[str, object],
) -> None:
    result = _evaluation_result([])
    result.baseline.trial_set.trials = (  # type: ignore[misc]
        _trial(1, raw_output),
    )

    trial = evaluate_dataset_invariants(result, _suite()).baseline.rules[0].trials[0]

    assert trial.status == "not_evaluable"
    assert trial.reason_code == expected_reason
    assert trial.resolved_values == expected_values


def test_trial_aggregation_prioritizes_violation_then_not_evaluable() -> None:
    violated = evaluate_dataset_invariants(
        _evaluation_result(
            [
                _output("AC-100", "AC-100"),
                {"request": {"invoice_reference": "AC-100"}},
                _output("AC-100", "AC-101"),
            ]
        ),
        _suite(),
    ).baseline.rules[0]
    not_evaluable = evaluate_dataset_invariants(
        _evaluation_result(
            [
                _output("AC-100", "AC-100"),
                {"request": {"invoice_reference": "AC-100"}},
            ]
        ),
        _suite(),
    ).baseline.rules[0]
    satisfied = evaluate_dataset_invariants(
        _evaluation_result([_output("AC-100", "AC-100"), _output(12500, 12500)]),
        _suite(),
    ).baseline.rules[0]

    assert (violated.status, violated.reason_code) == (
        "violated",
        "one_or_more_trials_violated",
    )
    assert (not_evaluable.status, not_evaluable.reason_code) == (
        "not_evaluable",
        "one_or_more_trials_not_evaluable",
    )
    assert (satisfied.status, satisfied.reason_code) == (
        "satisfied",
        "all_trials_satisfied",
    )


def test_evaluates_baseline_and_only_accepted_executed_variations_from_target_outputs() -> None:
    accepted = _case(
        "surface.rephrase",
        [_output("AC-100", "AC-101"), _output("AC-100", "AC-101")],
    )
    rejected = _case("style.terse", [], passed=False, executed=False)
    evaluated = evaluate_dataset_invariants(
        _evaluation_result(
            [_output("AC-100", "AC-100"), _output("AC-100", "AC-100")],
            accepted,
            rejected,
        ),
        _suite(),
    )

    assert evaluated.interaction_id == "interaction-1"
    assert evaluated.suite_sha256 == _suite().sha256
    assert evaluated.observation_source == "target_output"
    assert evaluated.observation_authority == "agent_response"
    assert evaluated.baseline.rules[0].status == "satisfied"
    assert len(evaluated.variations) == 1
    assert evaluated.variations[0].operator_id == "surface.rephrase"
    assert evaluated.variations[0].rules[0].status == "violated"
    assert all(
        trial.resolved_values["left"] == "AC-100"
        for trial in evaluated.variations[0].rules[0].trials
    )


def test_result_models_reject_inconsistent_statuses_values_and_arms() -> None:
    with pytest.raises(ValidationError, match="pointers must be different"):
        DatasetInvariantTrialEvaluation(
            repetition=1,
            status="satisfied",
            reason_code="values_equal",
            left_pointer="/same",
            right_pointer="/same",
            resolved_values={"left": 1, "right": 1},
        )
    with pytest.raises(ValidationError, match="status"):
        DatasetInvariantTrialEvaluation(
            repetition=1,
            status="violated",
            reason_code="values_equal",
            left_pointer="/left",
            right_pointer="/right",
            resolved_values={"left": 1, "right": 1},
        )
    with pytest.raises(ValidationError, match="resolved"):
        DatasetInvariantTrialEvaluation(
            repetition=1,
            status="not_evaluable",
            reason_code="left_pointer_missing",
            left_pointer="/left",
            right_pointer="/right",
            resolved_values={"left": 1},
        )
    with pytest.raises(ValidationError, match="equal-value evidence"):
        DatasetInvariantTrialEvaluation(
            repetition=1,
            status="satisfied",
            reason_code="values_equal",
            left_pointer="/left",
            right_pointer="/right",
            resolved_values={"left": 1, "right": 2},
        )
    with pytest.raises(ValidationError, match="different-value evidence"):
        DatasetInvariantTrialEvaluation(
            repetition=1,
            status="violated",
            reason_code="values_differ",
            left_pointer="/left",
            right_pointer="/right",
            resolved_values={"left": 1, "right": 1},
        )
    with pytest.raises(ValidationError, match="type-mismatch evidence"):
        DatasetInvariantTrialEvaluation(
            repetition=1,
            status="not_evaluable",
            reason_code="operand_types_differ",
            left_pointer="/left",
            right_pointer="/right",
            resolved_values={"left": 1, "right": 2},
        )

    trial = DatasetInvariantTrialEvaluation(
        repetition=1,
        status="satisfied",
        reason_code="values_equal",
        left_pointer="/left",
        right_pointer="/right",
        resolved_values={"left": 1, "right": 1},
    )
    with pytest.raises(ValidationError, match="aggregate"):
        DatasetInvariantRuleEvaluation(
            rule_type="json_values_equal",
            rule_id="rule-1",
            rule_version="1.0.0",
            description="Rule one.",
            severity="high",
            status="violated",
            reason_code="one_or_more_trials_violated",
            trials=(trial,),
        )
    rule_result = DatasetInvariantRuleEvaluation(
        rule_type="json_values_equal",
        rule_id="rule-1",
        rule_version="1.0.0",
        description="Rule one.",
        severity="high",
        status="satisfied",
        reason_code="all_trials_satisfied",
        trials=(trial,),
    )
    with pytest.raises(ValidationError, match="operator ID"):
        DatasetInvariantArmEvaluation(
            arm="baseline",
            operator_id="surface.rephrase",
            rules=(rule_result,),
        )
    baseline = DatasetInvariantArmEvaluation(arm="baseline", rules=(rule_result,))
    with pytest.raises(ValidationError, match="variation arms"):
        DatasetInvariantEvaluation(
            interaction_id="interaction-1",
            suite_sha256="a" * 64,
            observation_authority="agent_response",
            baseline=baseline,
            variations=(baseline,),
        )


def test_result_models_preserve_rule_identity_and_pointers_across_arms() -> None:
    def rule_result(
        *,
        version: str = "1.0.0",
        severity: Literal["low", "high"] = "high",
        left_pointer: str = "/left",
        second_left_pointer: str | None = None,
    ) -> DatasetInvariantRuleEvaluation:
        trials = tuple(
            DatasetInvariantTrialEvaluation(
                repetition=repetition,
                status="satisfied",
                reason_code="values_equal",
                left_pointer=(
                    second_left_pointer
                    if repetition == 2 and second_left_pointer is not None
                    else left_pointer
                ),
                right_pointer="/right",
                resolved_values={"left": 1, "right": 1},
            )
            for repetition in (1, 2)
        )
        return DatasetInvariantRuleEvaluation.model_validate(
            {
                "rule_type": "json_values_equal",
                "rule_id": "rule-1",
                "rule_version": version,
                "description": "Rule one.",
                "severity": severity,
                "status": "satisfied",
                "reason_code": "all_trials_satisfied",
                "trials": trials,
            }
        )

    with pytest.raises(ValidationError, match="same pointers"):
        rule_result(second_left_pointer="/other-left")

    baseline = DatasetInvariantArmEvaluation(arm="baseline", rules=(rule_result(),))
    for changed_rule in (
        rule_result(version="2.0.0"),
        rule_result(severity="low"),
        rule_result(left_pointer="/other-left"),
    ):
        variation = DatasetInvariantArmEvaluation(
            arm="variation",
            operator_id="surface.rephrase",
            rules=(changed_rule,),
        )
        with pytest.raises(ValidationError, match="preserve suite rules"):
            DatasetInvariantEvaluation(
                interaction_id="interaction-1",
                suite_sha256="a" * 64,
                observation_authority="agent_response",
                baseline=baseline,
                variations=(variation,),
            )
