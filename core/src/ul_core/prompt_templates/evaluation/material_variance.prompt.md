+++
name = "evaluation.material_variance"
description = "Classifies observed dataset variances by operational materiality without judging correctness."
author = "UL"
+++
Decide whether the observed difference between an original agent run and a run on a meaning-preserving input variation changes the real-world action or outcome.

This is a material-variance decision, not a correctness, safety, quality, or severity judgment. Do not decide which run is right. Use only the submitted structured findings.

Return exactly one label from this closed set:

- `material_variance:action_added`
- `material_variance:action_removed`
- `material_variance:action_count_changed`
- `material_variance:action_target_changed`
- `material_variance:grounded_argument_changed`
- `material_variance:committed_state_changed`
- `material_variance:response_meaning_changed`
- `operationally_equivalent:alias_or_representation`
- `operationally_equivalent:presentation_only`
- `operationally_equivalent:lookup_path_only`
- `operationally_equivalent:same_real_world_effect`
- `insufficient_evidence:missing_comparison_evidence`

A difference is material when it changes what was done, how many times it was done, the real-world target, a grounded argument, committed state, or the substantive meaning of a returned answer. A difference is operationally equivalent when the same real-world effect is represented with aliases, formatting, ordering, equivalent identifiers, different lookup paths, or presentation-only wording.

Use this procedure in order:

1. Compare `committed_action_count` and `committed_actions` when response effects provide them. These normalized facts are authoritative. UL has already removed read-only GET, SEARCH, LOOKUP, READ, LIST, FIND, FETCH, and QUERY actions from this array. A non-zero count on only one side is material even if the other side claims it succeeded in prose.
2. Otherwise, extract committed effects from both sides. Treat action names containing or equivalent to GET, SEARCH, LOOKUP, READ, LIST, FIND, FETCH, or QUERY as read-only information gathering, not committed effects. Do not use the total count of all actions; compare the count of committed effects only.
3. Ignore administrative envelope fields such as task IDs, run status, reward, step count, timing, server IDs, and transport metadata. UL may omit these before the call.
4. Compare `substantive_answer_state` before reading `substantive_answer`. This normalized fact is authoritative. `present` on one side and `empty` on the other is `material_variance:response_meaning_changed`. When both are `present`, compare meaning while ignoring wording. When both are `empty`, the answers are equivalent. Treat null, an absent value, and an empty array as the same empty result.
5. If committed effects differ, return material variance. If only read-only lookups differ and the substantive answer and committed effects are the same, return `operationally_equivalent:lookup_path_only`.

Direction is always from baseline to variation:

- baseline has a committed action and variation does not: `material_variance:action_removed`
- baseline has no committed action and variation does: `material_variance:action_added`

Never reverse these labels. If the normalized `committed_actions` and `substantive_answer` are identical on both sides, the result is operationally equivalent, never insufficient. The structured effects themselves are sufficient evidence for that decision.

Creating, updating, deleting, sending, scheduling, ordering, charging, transferring, or otherwise mutating an external system is a committed effect and is material when it differs. A success claim without the corresponding committed effect is not proof that the effect occurred.

The success claim is still a substantive returned answer. If one side says an operation succeeded and the other side returns no substantive answer, classify the response difference as material even when neither side proves a committed action.

For action findings, compare predicates and only the submitted grounded fields. Do not treat opaque record identifiers, server-generated identifiers, timestamps, ordering, transport envelopes, URLs, display names, abbreviations, or alternate code representations as material unless the submitted evidence establishes that they select a different real-world target or value.

For response findings, ignore wording, formatting, and added non-conflicting explanation, signatures, or apologies. Treat a changed factual assertion, recommendation, refusal, requested next step, recipient, or other substantive instruction as material.

If the structured findings do not establish both sides well enough to make the distinction, return `insufficient_evidence:missing_comparison_evidence`. Never infer missing facts from domain knowledge.

For conclusive decisions, cite the containing effect object on both sides, for example `/payload/answer/findings/0/baseline_effects/0` and `/payload/answer/findings/0/variation_effects/0`. When an effect array is empty, cite the array itself. Never invent `/0` below an empty array.

Examples:

- Baseline performs `GET_CUSTOMER` and variation performs `SEARCH_CUSTOMERS`, both return the same customer and neither mutates state: `operationally_equivalent:lookup_path_only`.
- Baseline performs `CREATE_TRANSFER` with grounded `amount: 100` and variation performs it with grounded `amount: 200`: `material_variance:grounded_argument_changed`.
- Baseline returns `{"items": []}` and variation returns `{"items": null}` with no committed effects: `operationally_equivalent:same_real_world_effect`.
- Baseline has `substantive_answer: null` and `committed_actions: []`, while variation has the same normalized fields: `operationally_equivalent:same_real_world_effect`, regardless of omitted run status or step count.
- Baseline has a non-empty `substantive_answer` and variation has `substantive_answer: null`, with no committed effects on either side: `material_variance:response_meaning_changed`.
- Baseline and variation both have `committed_actions: []`; a read-only lookup was removed from one side during normalization: `operationally_equivalent:lookup_path_only` or `operationally_equivalent:same_real_world_effect`.
- Baseline says no matching record exists and therefore no operation will be performed; variation says no recorded match exists, so no operation was created: `operationally_equivalent:presentation_only`.
- Baseline says the appointment is Monday and variation says it is Tuesday: `material_variance:response_meaning_changed`.
- One side omits the effect details needed to determine whether the real-world target changed: `insufficient_evidence:missing_comparison_evidence`.

Use score `1` for material variance, `0` for operational equivalence, and `0.5` for insufficient evidence. Cite at least one existing baseline and one existing variation value under `/payload/answer/findings` for material or equivalent decisions. Never cite a missing array element; cite the empty array itself when one side has no effects. Keep the explanation short; UL will persist a fixed local explanation rather than model-written prose.
