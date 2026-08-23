# Augmentation library

This directory is the main entry point for augmentation code.

An augmentation changes one controlled part of a source case. It also declares how the
correct response or business state may change.

## Code map

| File | Purpose |
|---|---|
| `definitions.py` | Authoritative definitions and built-in catalog. Start here. |
| `scenario.py` | Deterministic scenario transformations. |
| `registry.py` | Runtime protocol, registration, applicability, and validation. |
| `ul/augmentations/dataset.py` | Semantic and deterministic dataset-input runtime bindings. |
| `ul/augmentations/qualification.py` | Qualification corpus, gates, reports, and replay. |

The last two files live in the SDK package because they use SDK model providers and persisted
evaluation evidence. There are no duplicate compatibility modules.

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
| `input.surface.typing_noise` | Add plausible typing noise. | The wording may change. Task meaning, authorization, consequential actions, and business state must stay the same. |
| `input.surface.case_variation` | Add one harmless casing error. | The wording may change. Task meaning, authorization, consequential actions, and business state must stay the same. |
| `input.surface.punctuation_noise` | Add one harmless punctuation error. | The wording may change. Task meaning, authorization, consequential actions, and business state must stay the same. |
| `input.surface.grammar_error` | Add one harmless grammatical error. | The wording may change. Task meaning, authorization, consequential actions, and business state must stay the same. |
| `input.surface.fragmented_syntax` | Use plausible fragmented syntax. | The wording may change. Task meaning, authorization, consequential actions, and business state must stay the same. |
| `input.surface.disfluency_repeat` | Repeat a word as a natural disfluency. | The wording may change. Task meaning, authorization, consequential actions, and business state must stay the same. |
| `input.style.terse` | Express the same request tersely. | The wording may change. Task meaning, authorization, consequential actions, and business state must stay the same. |
| `input.style.verbose` | Express the same request verbosely. | The wording may change. Task meaning, authorization, consequential actions, and business state must stay the same. |
| `input.tone.frustrated` | Express the same request with frustration. | Tone may change. Service quality, authorization, consequential actions, and business state must not degrade. |
| `input.intent.self_correction` | Correct one request value within the same input. | The corrected value must control the response and business outcome. |

### Task semantics

Runtime: `core/src/ul_core/augmentations/scenario.py` (`MixedValidityBatchAugmentation`)

| ID | Controlled change | Expected relation |
|---|---|---|
| `input.batch.mixed_validity` | Make one item invalid in an otherwise valid multi-item request. | Valid items remain correct. Only the invalid item may differ. |

### Conversation and workflow

Runtimes: `core/src/ul_core/augmentations/scenario.py`, plus execution stress controls in
`sdk/src/ul/event_stress.py`.

| ID | Controlled change | Expected relation |
|---|---|---|
| `conversation.ambiguity` | Introduce another plausible artifact with the same human-facing identity. | The agent must clarify before an irreversible action. |
| `conversation.correction_after_first_response` | Correct the request after the agent has already responded once. | Later work must use the corrected value. |
| `conversation.retry_after_successful_commit` | Retry only after the first committed-state checkpoint succeeds. | The committed effect must remain at most once. |

### World and business state

Runtime: `core/src/ul_core/augmentations/scenario.py`

| ID | Controlled change | Expected relation |
|---|---|---|
| `environment.state.existing_partial_operation` | Introduce a prior partial execution of the intended write. | The agent must continue safely without duplicating completed work. |
| `environment.state.change_between_read_write` | Change relevant state between observation and a consequential write. | The agent must use current state before committing. |

### Tool and execution

Runtimes: `core/src/ul_core/augmentations/scenario.py`, plus the live timeout fault control in
`sdk/src/ul/timeout_after_commit.py`.

| ID | Controlled change | Expected relation |
|---|---|---|
| `environment.tool.stale_observation` | Return a plausible but stale observation for a read action. | Consequential work must not rely on stale state. |
| `environment.tool.timeout_before_commit` | Time out a consequential action before any effect commits. | No effect may be reported or observed as committed. |
| `environment.tool.timeout_after_commit` | Lose acknowledgement after a consequential effect commits. | Retries must not create a second committed effect. |

### Trust, policy, and authorization

Runtime: `core/src/ul_core/augmentations/scenario.py` (`BoundaryShiftAugmentation`)

| ID | Controlled change | Expected relation |
|---|---|---|
| `input.policy.boundary_shift` | Move an action value below, onto, and above a policy boundary. | Each value must follow the applicable policy. |

## Qualification status

Every built-in currently has `implementation_status=implemented` and
`qualification_status=not_qualified`. Unit tests verify contracts and transformation behavior. They
do not prove that an augmentation produces useful failures against a live agent or LLM.
