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

## Relationship to UL evaluators

This is the transport and typed decision foundation. Existing augmentation verification and evaluator
judges are unchanged. A decision is not a complete UL finding: integrations must retain source
evidence and apply their task-specific completeness and calibration requirements. A missing
observation is not evidence that an action did not happen. Code should enforce known structural
requirements before asking the model to make a semantic decision.

See the [TypeSafe API](https://docs.typesafe.ai/api),
[Jev limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13), and
[OpenRouter schema](https://openrouter.ai/openapi.json).
