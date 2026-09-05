# UL Vision

## Make insights excellent

UL's first promise is to discover consequential failures in high-risk AI agents and reveal the trends that connect those failures. Product decisions follow this order:

1. Make insights excellent: discover as many meaningful failures as possible, explain why they matter, and expose recurring patterns across them.
2. Then make UL quick and easy to integrate and run.

Every product improvement must start with a real experiment using UL as a mock customer. Run the product, observe the actual failure or missed opportunity, make the smallest useful change, and rerun the same customer journey to prove that it improved. Do not add or fix product behavior based only on theory, internal APIs, or isolated helper tests.

The core customer flow is deliberately small:

```text
give dataset → choose augmentations → augment → run agent → compare results
```

UL curates the experience around this flow. Customers should not need to understand or configure internal bindings, semantic roles, prompts, validators, provider-specific execution details, or every other implementation choice to get value. Sensible, evidence-backed behavior belongs behind the scenes. When an augmentation cannot produce a valid result for a datapoint and the current settings, UL should warn the customer and skip it by default.

UL is open source. People who need a fundamentally different product can inspect it, extend it, or fork it without turning the default experience into an infinitely configurable framework.

UL is an open-source tool that helps teams running high-risk AI agents discover, understand, and prioritize their most dangerous failures.

## The Problem

Most AI evaluation tools optimize for success rates. For high-risk agents, this is the wrong signal.

A system that is “98% correct” can still be unacceptable if the remaining 2% causes large financial loss, safety incidents, or irreversible actions. Aggregate scores hide severity. They also fail to surface rare, long-tail, and previously unseen failures.

UL focuses on the residual risk that success metrics ignore.

## Core Functions

- **Real production data ingestion** from agent runs and traces.
- **Active augmentation** of that data to surface hidden but viable failures.
- **Continuous stress testing** of a living evaluation dataset.
- **Failure clustering** along two axes:
   - Vertical — by topic, domain, or use case
   - Horizontal — by failure type
- **Severity prioritization**
- **Human understandable results**

## Product Principles

- Prefer human interpretability over metrics.
- Ground evaluation in real production behavior.
- Actively search for unknown failures instead of only measuring known ones.
- Keep the human reviewer in the loop with structured, reviewable insights.
- Stay simple, direct, and inspectable.
- Allow inspectors to go from high-level insights all the way into the finest details
- Adapt to the customer’s existing agent; the customer should not need to remodel their product for UL.
- From customer queries, traces, and observed state, UL automatically deconstructs intent, entities, constraints, dependencies, and side effects, then generates grounded, plausible variations.

**Don't add anything to this file without getting human approval**
