# Typed decisions with Jev

Use the Python SDK to ask a decision model questions with bounded answer spaces. The client supports
TypeSafe's choice, yes/no probability (`noul`), and ordered rubric score questions through OpenRouter.
Questions in one request share the same state; they do not consume each other's answers.

```python
import asyncio

from ul import (
    ChoiceQuestion,
    NoulQuestion,
    OpenRouterDecisionClient,
    OpenRouterDecisionSettings,
    ScoreQuestion,
)


async def main():
    settings = OpenRouterDecisionSettings(
        live_calls=True,
        allow_external_data_processing=True,
    )
    async with OpenRouterDecisionClient(settings) as client:
        response = await client.decide(
            {
                "message": "Please refund the duplicate charge. I have contacted support three times."
            },
            {
                "route": ChoiceQuestion(
                    instructions="Which team handles this request?",
                    criteria={"billing": "Charges and refunds", "technical": "Software bugs"},
                ),
                "refund": NoulQuestion(instructions="Does the customer request a refund?"),
                "frustration": ScoreQuestion(
                    instructions="Rate customer frustration.",
                    criteria=("Calm", "Some frustration", "Very angry"),
                ),
            },
        )
        print(response.model_dump_json(indent=2))


asyncio.run(main())
```

Set `OPEN_ROUTER_API_KEY` before running. `OpenRouterDecisionSettings(_env_file=".env", ...)` can load
an explicitly selected private env file. Settings also accept the `UL_DECISION_` environment prefix,
for example `UL_DECISION_MODEL`, `UL_DECISION_LIVE_CALLS`, and
`UL_DECISION_ALLOW_EXTERNAL_DATA_PROCESSING`. Both opt-ins default to false. The credential is excluded
from settings serialization, repr, and the credential-independent `settings.version` fingerprint.

The default model is `typesafe/jev-1.13`. Choose an explicit numeric version; moving `jev-latest`
aliases are intentionally rejected. Responses retain the actual resolved model version, provider,
request ID, input/output token counts, and reported cost. Missing cost or confidence remains `None`.
A choice answer includes its selected label and, when supplied, probabilities and confidence. A
noul answer is a probability between zero and one. Score levels are numbered from zero; the returned
score can fall between levels. The client does not invent a decision threshold or equate confidence
with probability of correctness.

## Request and response boundaries

All calls go to OpenRouter's `/api/alpha/decisions` endpoint, pinned to TypeSafe with provider fallback
disabled, data collection denied, and zero data retention required. Redirects are not followed.
Only the supplied state and questions are sent. The client does not automatically redact them or
collect traces, environment state, files, or private data. Supply only the evidence appropriate to
the decision. Keep untrusted state separate from trusted question instructions and criteria.

There is one network attempt per `decide` call and no automatic retries. Default limits are 45 seconds
for the entire request, 64,000 serialized request bytes, 256,000 response bytes, and 100 questions.
They do not guarantee a monetary budget or that every request fits the model's token context window.
The caller owns campaign call counts and spending limits.

Responses are validated against the submitted question IDs, question types, choice labels, score
range/legend, probability ranges and keys, pinned provider, and requested model version. Probability
sums permit the provider's two-decimal rounding. Inconsistent or malformed responses raise
`DecisionError`; they never become negative or successful judgments. HTTP failures carry
`DecisionError.status_code` without exposing the provider body, payload, or credential. Invalid local
question shapes raise Pydantic validation errors before the network request.

An injected `httpx.AsyncClient` remains owned by the caller. The SDK closes clients it creates itself.
The SDK does not log raw requests or responses.

## Compare baseline and augmented outcomes

`JevMaterialVarianceJudge` plugs into `DatasetEvaluationRunner` through its existing
`material_variance_evaluator` argument. It checks whether the meaning or real-world effect changed;
it does not decide which answer is better or whether either answer is correct. The existing LLM judge remains the default.

This runnable SDK example compares a completed refund with a scheduled refund:

```python
import asyncio

from ul import JevMaterialVarianceJudge, OpenRouterDecisionClient, OpenRouterDecisionSettings
from ul.dataset_evaluation import DatasetEvaluationFinding
from ul_core.dataset import ObservedOutcome


def outcome(identifier, text):
    return ObservedOutcome(
        id=identifier,
        kind="answer",
        predicate="returned_response",
        status="observed",
        confidence=1,
        position=0,
        fields={"value": text},
    )


async def main():
    settings = OpenRouterDecisionSettings(
        live_calls=True,
        allow_external_data_processing=True,
    )
    async with OpenRouterDecisionClient(settings) as client:
        judge = JevMaterialVarianceJudge(client)
        assessment = await judge.evaluate(
            "response",
            (
                DatasetEvaluationFinding(
                    category="changed_response",
                    message="Compare the returned answers.",
                    expected_effects=(outcome("baseline", "The $50 refund was completed."),),
                    observed_effects=(outcome("augmented", "The $50 refund is scheduled."),),
                ),
            ),
        )
        print(assessment.model_dump_json(indent=2))


asyncio.run(main())
```

For a full dataset run, construct the judge inside the client's context and pass
`material_variance_evaluator=judge` to your `DatasetEvaluationRunner`. Keep the client open until
`await runner.run(source)` finishes. The runner persists the assessment, evidence references, and
version alongside its findings and includes decision requests in its semantic call count. Its
existing divergence verdict and human review workflow remain unchanged.

Ordinary code checks evidence presence and applies the existing exact response-envelope checks
before calling Jev. Remaining findings share one request, with one independent three-choice question
per finding. Any confident material change establishes divergence. Equivalence requires every
finding to be equivalent. Code assigns the reason from the finding category and attaches references
to both compared evidence lists; these are the inputs to the decision, not model-generated citations
or reasoning. No raw model prose is saved.

The default confidence threshold is `0.88`, configurable with `minimum_confidence`. This is a routing
policy, not a guarantee of accuracy or calibration for your data. Missing confidence, low confidence,
or an uncertain answer produces `insufficient_evidence`. Provider failures produce the same decision
with reason `judge_error`. There are no retries or automatic calls to another model.

Each finding needs observed outcomes with fields on both sides. An incomplete finding cannot
block a material change established by another finding, but it prevents an equivalent conclusion. Action comparisons additionally require
available grounded fields. Missing sides, unobserved outcomes, empty evidence, more than ten findings,
or oversized input remain inconclusive without a model call. In particular, an empty action list
on one side of a finding does not by itself prove absence; use complete response envelopes containing
observed action lists where available. Ten findings fit the assessment's twenty-reference limit.
The default state limit is 50,000 characters, plus the existing byte limits. Nothing is truncated.
`actual_calls` counts attempted decision requests; local exact checks do not increment it.

The credential-independent evaluator version includes the questions, model settings, threshold,
input limit, and adapter version. The caller owns the decision client's lifecycle and must supply
appropriately redacted evidence. Augmentation-input verification and general rubric/pairwise judges
are not changed by this integration.

See the [TypeSafe API](https://docs.typesafe.ai/api),
[Jev limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13), and
[OpenRouter schema](https://openrouter.ai/openapi.json).


## Select Jev in the CLI

Add `--materiality-judge jev` to a dataset evaluation:

```bash
uv run ul dataset evaluate interactions.jsonl \
  --target customer_agent:run \
  --materiality-judge jev \
  --dry-run
```

The dry run shows the selected judge, model, and OpenRouter/TypeSafe destination without making
external calls. To execute, use the displayed target confirmation with `--confirm-target`,
`--confirm-test-environment`, and a new `--output` path, as for other local targets. HTTP targets
use their existing environment confirmation options. Set `OPEN_ROUTER_API_KEY` and `UL_LIVE=true`
(or the separate dataset live-call and external-processing controls). The CLI loads `.env` and uses
these dataset opt-ins for Jev too; it does not require separate `UL_DECISION_LIVE_CALLS` opt-in.

`--materiality-judge llm` retains the existing judge. Generation and augmentation verification still
use the configured dataset LLM in both modes. Existing LLM preflight checks still run; they do not
preflight Jev. A missing Jev credential stops execution before the target runs, and provider errors
during comparison remain inconclusive. If the semantic LLM uses a customer endpoint, its API key is
not forwarded to OpenRouter: Jev uses `OPEN_ROUTER_API_KEY` separately.

Jev defaults to `typesafe/jev-1.13`; set `UL_DECISION_MODEL` to select another numeric version. The
CLI uses the dataset timeout and input limit, with the SDK's default confidence threshold of 0.88.
Other decision request/response bounds retain their `UL_DECISION_` settings. Jev requests count as
materiality calls in campaign planning; their decision tokens are excluded from the LLM completion
token estimate. No monetary estimate is implied.

Saved evidence records the decision model and evaluator identity. Resume restores the selected
judge when the option is omitted and rejects an explicit switch. Keep the same `UL_DECISION_`
settings when resuming: model or limit changes produce an incompatible evaluator identity and
require a new output file. Credentials can rotate without changing that identity.
