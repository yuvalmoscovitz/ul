# Customer-defined evaluators

Evaluators are versioned assertions attached to an `EvaluationCase`. Run that case with
`evaluate_case`; UL executes the configured environment, validates its evidence, and then runs every
attached evaluator. Each evaluator returns the same result shape with a status, optional score or
label, explanation, and evidence pointers.

This is currently an SDK execution path. The dataset `ul run` command continues to run its documented
variance workflow; it does not implicitly enable correctness or preference modes.

```python
from ul import EvaluationCase, ExactValueEvaluator, StateChangeEvaluator, evaluate_case

case = EvaluationCase(
    id="schedule-payment",
    turns=(user_turn,),
    max_environment_api_calls=1,
    timeout_seconds=30,
    evaluators=(
        ExactValueEvaluator(
            id="accepted",
            source="answer",
            json_pointer="/status",
            expected="accepted",
        ),
        StateChangeEvaluator(
            id="scheduled",
            json_pointer="/payment/status",
            operator="equals",
            expected="scheduled",
        ),
    ),
)

result = await evaluate_case(case, environment)
```

The default evidence mapping supports the final answer plus initial and final state. Tool-call and
HTTP evaluators use normalized `EvaluationSubject` fields. Supply `subject_builder` to
`evaluate_case` when a customer environment records those fields in its own response shape.

Every evaluation result includes evaluator reliability. An evaluator without a matching calibration
report is explicitly `uncalibrated`; a report created for an older rubric, prompt, model, or judge
configuration does not carry forward after that input changes.

## Calibrate before a campaign

Calibration tests the evaluator against categorical examples before its findings are trusted. A
suite requires a known-good example, a known-bad example, and a borderline example repeated at least
twice. Optional human labels expose reviewer disagreement independently of judge disagreement.

```python
from ul import (
    EvaluatorCalibrationExample,
    EvaluationSubject,
    calibrate_evaluator,
    evaluate_case,
)

calibration = await calibrate_evaluator(
    evaluator,
    (
        EvaluatorCalibrationExample(
            id="known-good",
            kind="known_good",
            subject=EvaluationSubject(agent_status="succeeded", answer=known_good_answer),
            expected_passed=True,
            human_labels=(True, True),
        ),
        EvaluatorCalibrationExample(
            id="known-bad",
            kind="known_bad",
            subject=EvaluationSubject(agent_status="succeeded", answer=known_bad_answer),
            expected_passed=False,
            human_labels=(False, False),
        ),
        EvaluatorCalibrationExample(
            id="borderline",
            kind="borderline",
            subject=EvaluationSubject(agent_status="succeeded", answer=borderline_answer),
            expected_passed=True,
            repetitions=3,
            human_labels=(True, False),
        ),
    ),
    judge=judge,
)

campaign_result = await evaluate_case(
    case,
    environment,
    judge=judge,
    calibration_reports={evaluator.id: calibration},
)
assert campaign_result.evaluation_results.reliability[0].status in {
    "reliable",
    "unreliable",
}
```

The report retains per-run judgments and directly lists `false_positive_examples`,
`false_negative_examples`, `unstable_examples`, and `human_disagreement_examples`. Its aggregate
`human_agreement` compares categorical judge outcomes with the supplied expected labels. A report is
`unreliable` when it contains a misclassification, repeated-run instability, human disagreement, or
an evaluator error.

`OpenAICompatibleEvaluatorJudge` derives its version from the packaged judge prompt, requested model,
endpoint, data policy, and bounded request configuration. Credentials are intentionally excluded, so
key rotation does not invalidate calibration. Custom judges must pass an `EvaluatorJudgeVersion` to
both `calibrate_evaluator` and `evaluate_case`.

## Judge and pairwise evaluators

Pass a configured judge for natural-language rubrics or pairwise preference. External processing
must be explicitly enabled. OpenRouter can additionally require deny-collection and zero-data-
retention routing.

```python
from pydantic import SecretStr
from ul import (
    OpenAICompatibleEvaluatorJudge,
    OpenAICompatibleJudgeConfig,
    RubricEvaluator,
)

rubric = RubricEvaluator(
    id="clear-outcome",
    rubric="The answer clearly distinguishes scheduling from completed payment.",
    minimum_score=0.8,
)

judge = OpenAICompatibleEvaluatorJudge(
    OpenAICompatibleJudgeConfig(
        base_url="https://openrouter.ai/api/v1",
        model="your-structured-output-model",
        api_key=SecretStr(api_key),
        allow_external_data_processing=True,
        data_policy="openrouter_zdr",
    )
)
```

Judge decisions must cite bounded RFC 6901 pointers beginning with `/payload/` that resolve against
the exact submitted judge request. Invalid or ungrounded judge output becomes `evaluator_error`.

## Customer callables

Executable checks are registered in-process callables, never command or shell strings.

```python
from ul import CallableEvaluator, EvaluatorDecision, EvaluatorEvidence, evaluate


def approved_fixture(subject):
    approved = subject.initial_state["invoice"]["status"] == "approved"
    return EvaluatorDecision(
        passed=approved,
        explanation="Checked initial invoice status.",
        evidence=(
            EvaluatorEvidence(
                source="initial_state",
                json_pointer="/invoice/status",
                description="Initial invoice status.",
            ),
        ),
    )


results = await evaluate(
    subject,
    (CallableEvaluator(id="approved", callable_id="customer.approved"),),
    callables={"customer.approved": approved_fixture},
)
```

## Privacy boundary

Rubrics send only `answer` by default. State, tool calls, and HTTP results require `include_sources`.
`private_data` is excluded unless the evaluator sets `allow_private_data=True`.
`private_json_pointers` removes individual values from the submitted payload. Every pointer must use
valid RFC 6901 syntax and resolve exactly; invalid or misspelled paths fail closed before the judge
network call.
