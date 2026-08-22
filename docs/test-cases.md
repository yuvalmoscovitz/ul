# Design valid test cases

A UL test case has three parts:

1. An input that asks the agent to do something.
2. A resettable environment fixture that contains the entities and state needed by that input.
3. An optional evaluator that states what correct behavior means.

UL can discover that behavior changed without an evaluator. It cannot infer that a historical answer
was correct or prove that an arbitrary input is compatible with an arbitrary environment.

## Stateless response agents

An isolated-response target starts every request from fresh, isolated state. It does not need a
fixture identity:

```bash
ul init interactions.jsonl \
  --environment-url https://agent.example.test/v1/chat/completions \
  --adapter-tier isolated-response \
  --confirm-request-isolation \
  --confirm-safe-test-target
```

UL can compare responses at this tier. It cannot verify tool side effects or committed state.

## Tool-using agents

The input and fixture must refer to the same resources. For example, an input that asks to pay
invoice `AC-100` needs a fixture that creates `AC-100` before each execution:

```json
{
  "fixture_id": "accounts-payable-standard",
  "fixture_version": "v3",
  "setup": {
    "url": "https://environment.example.test/setup",
    "request_json_template": {
      "case_id": "{{case_id}}",
      "invoices": [{"id": "AC-100", "status": "approved", "amount": 100}]
    },
    "case_id_json_pointer": "/case_id"
  }
}
```

Use deterministic checks over tool calls or final state when the expected behavior is known. Do not
depend on exact response wording when the side effect is the real outcome.

## Stateful workflows

Use one fixture for multiple inputs only when it can satisfy every input after reset. Otherwise,
split the dataset or configure distinct campaigns. Record both `fixture_id` and `fixture_version`:

```bash
ul init interactions.jsonl \
  --environment-url https://environment.example.test \
  --fixture-id accounts-payable-standard \
  --fixture-version v3
```

`ul dataset evaluate --dry-run` prints the fixture identity. A stateful target without one receives a
warning. Evidence records the configured identity or records that it was missing, so reviewers can
judge whether a result is reproducible.

## Pre-run checklist

- Every requested entity, account, booking, or resource exists in the fixture.
- Its starting state permits the requested action.
- Reset restores both the agent session and external state.
- Setup is deterministic and safe to repeat.
- The fixture version changes when setup data or logic changes.
- Secrets are referenced through environment variables, not stored in fixture files.
- Optional evaluators inspect the response, tool evidence, or final state they actually need.

## Classify invalid results correctly

| Result | Meaning |
| --- | --- |
| UL defect | UL planned, executed, recorded, or compared the run incorrectly. |
| Agent failure | The agent violated explicit customer criteria in a valid test case. |
| Evaluator failure | The evaluator crashed, was ambiguous, or judged known examples incorrectly. |
| Invalid test case | The input, fixture, or evaluator did not form a compatible test. |

An invalid fixture is not evidence of an agent failure. Preserve the run receipt, correct the test
case, assign a new fixture version, and run it again.
