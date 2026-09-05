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

`definitions.py` is the single source of truth for built-in IDs, versions, applicability,
requirements, summaries, and expected relations. Inspect the current catalog through the public
CLI:

```console
ul augmentations list
ul augmentations show ID[@VERSION]
```

Use `ul augmentations list --json` when another tool needs the complete catalog.

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
