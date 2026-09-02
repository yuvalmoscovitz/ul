# Augmentation library

This directory is the main entry point for augmentation code.

An augmentation changes one controlled part of a source case. It also declares how the
correct response or business state may change.

Every materialized augmentation has an `AugmentationProjection` with typed `reads` and `writes`.
Targets use RFC 6901 paths and identify one of six locations: `structured_input`, `conversation`,
`state`, `tool`, `policy`, or `environment`. Projection validation resolves every target before
execution, rejects overlapping writes, and compares the source with the candidate so untargeted
data cannot change. The resulting lineage records exact changed paths and environment event IDs.

## Code map

| File | Purpose |
|---|---|
| `definitions.py` | Authoritative definitions and built-in catalog. Start here. |
| `scenario.py` | Deterministic scenario transformations. |
| `registry.py` | Runtime protocol, registration, applicability, and validation. |
| `ul/augmentations/dataset.py` | Semantic and deterministic dataset-input runtime bindings. |
| `ul/augmentations/conversation.py` | Correction and retry conversation runtime bindings. |
| `ul/augmentations/environment_fault.py` | Timeout-after-commit runtime binding. |
| `ul/augmentations/qualification.py` | Qualification corpus, gates, reports, and replay. |

The last four files live in the SDK package because they use SDK model providers, environments,
or persisted evaluation evidence. There are no duplicate compatibility modules.

## Surfaces

| Surface | What it covers |
|---|---|
| Human behavior | How a person expresses or repairs a request. |
| Task semantics | What work, entities, values, or constraints are requested. |
| Conversation and workflow | How work unfolds across turns, confirmation, retry, or handoff. |
| World and business state | Existing records and state changes around the agent. |
| Tool and execution | Tool observations, failures, timeouts, and acknowledgements. |
| Trust, policy, and authorization | Policy boundaries, permissions, and trusted authority. |

## Built-in augmentations

### Human behavior

Runtime: `sdk/src/ul/augmentations/dataset.py`

| ID | Controlled change | Expected relation |
|---|---|---|
| `input.surface.rephrase` | Rephrase while preserving the requested behavior. | The wording may change. Task meaning, authorization, consequential actions, and business state must stay the same. |
| `input.surface.typing_noise` | Add four or five typing errors. | The wording may change. Task meaning, authorization, consequential actions, and business state must stay the same. |
| `input.surface.punctuation_noise` | Add disruptive human punctuation or spacing noise. | The wording may change. Task meaning, authorization, consequential actions, and business state must stay the same. |
| `input.surface.grammar_error` | Add two to five natural writing mistakes. | The wording may change. Task meaning, authorization, consequential actions, and business state must stay the same. |
| `input.surface.fragmented_syntax` | Use plausible fragmented syntax. | The wording may change. Task meaning, authorization, consequential actions, and business state must stay the same. |
| `input.surface.disfluency_repeat` | Repeat a short phrase as a natural disfluency. | The wording may change. Task meaning, authorization, consequential actions, and business state must stay the same. |
| `input.style.terse` | Express the same request tersely. | The wording may change. Task meaning, authorization, consequential actions, and business state must stay the same. |
| `input.style.verbose` | Express the same request verbosely. | The wording may change. Task meaning, authorization, consequential actions, and business state must stay the same. |
| `input.tone.angry` | Express the same request with hostile anger. | The wording may change. Task meaning, authorization, consequential actions, and business state must stay the same. |
| `input.tone.argumentative` | Express the same request as an argumentative challenge. | The wording may change. Task meaning, authorization, consequential actions, and business state must stay the same. |
| `input.intent.self_correction` | Correct one request value within the same input. | The corrected value must control the response and business outcome. |

### Task semantics

Runtime: `core/src/ul_core/augmentations/scenario.py` (`MixedValidityBatchAugmentation`)

| ID | Controlled change | Expected relation |
|---|---|---|
| `input.batch.mixed_validity` | Make one item invalid in an otherwise valid multi-item request. | Invalid items must not silently contaminate or authorize valid items. |

### Conversation and workflow

Runtimes: `core/src/ul_core/augmentations/scenario.py` and
`sdk/src/ul/augmentations/conversation.py`.

| ID | Controlled change | Expected relation |
|---|---|---|
| `conversation.ambiguity` | Introduce another plausible artifact with the same human-facing identity. | The target should not guess between materially plausible matches. |
| `conversation.correction_after_first_response` | Correct the request after the agent has already responded once. | The corrected value supersedes the earlier value. |
| `conversation.retry_after_successful_commit` | Retry only after the first committed-state checkpoint succeeds. | The committed effect must remain at most once. |

### World and business state

Runtime: `core/src/ul_core/augmentations/scenario.py`

| ID | Controlled change | Expected relation |
|---|---|---|
| `environment.state.existing_partial_operation` | Introduce a prior partial execution of the intended write. | Only work not already committed should be performed. |
| `environment.state.change_between_read_write` | Change relevant state between observation and a consequential write. | The action must account for state that changed after the earlier read. |

### Tool and execution

Runtimes: `core/src/ul_core/augmentations/scenario.py` and
`sdk/src/ul/augmentations/environment_fault.py`.

| ID | Controlled change | Expected relation |
|---|---|---|
| `environment.tool.stale_observation` | Return a plausible but stale observation for a read action. | Consequential actions must not rely on known-stale state. |
| `environment.tool.timeout_before_commit` | Time out a consequential action before any effect commits. | A safe retry may occur because no effect committed. |
| `environment.tool.timeout_after_commit` | Lose acknowledgement after a consequential effect commits. | The target must resolve outcome before attempting another write. |

### Trust, policy, and authorization

Runtime: `core/src/ul_core/augmentations/scenario.py` (`BoundaryShiftAugmentation`)

| ID | Controlled change | Expected relation |
|---|---|---|
| `input.policy.boundary_shift` | Move an action value below, onto, and above a policy boundary. | Behavior may change only where the declared policy boundary permits it. |

## Qualification status

Every built-in currently has `implementation_status=implemented` and
`qualification_status=not_qualified`. Unit tests verify contracts and transformation behavior. They
do not prove that an augmentation produces useful failures against a live agent or LLM.

## Developing LLM-generated augmentations

Every new or materially changed LLM-generated dataset augmentation must declare
`generation_mechanism="llm"` and pass the production renderer's existing validity check against a
configured live model:

```console
uv run pytest -q \
  sdk/tests/test_deconstruction.py::test_live_llm_augmentations_pass_existing_validity_check \
  --require-live-llm
```

This validity check asks only whether the complete task meaning was preserved. It does not grade
whether the generated text successfully matches the requested style.

The non-live coverage contract fails when a new LLM operator is not included in this development
gate. This development check is separate from release qualification against a live customer agent.
