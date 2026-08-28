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

1. Extract committed effects from both sides. Treat action names containing or equivalent to GET, SEARCH, LOOKUP, READ, LIST, FIND, FETCH, or QUERY as read-only information gathering, not committed effects. Do not use the total count of all actions; compare the count of committed effects only.
2. Ignore administrative envelope fields such as task IDs, run status, reward, step count, timing, server IDs, and transport metadata.
3. Compare substantive returned answers after treating null, an absent value, and an empty array as the same empty result.
4. If committed effects differ, return material variance. If only read-only lookups differ and the substantive answer and committed effects are the same, return `operationally_equivalent:lookup_path_only`.

Creating, updating, deleting, sending, scheduling, ordering, charging, transferring, or otherwise mutating an external system is a committed effect and is material when it differs. A success claim without the corresponding committed effect is not proof that the effect occurred.

For action findings, compare predicates and only the submitted grounded fields. Do not treat opaque record identifiers, server-generated identifiers, timestamps, ordering, transport envelopes, URLs, display names, abbreviations, or alternate code representations as material unless the submitted evidence establishes that they select a different real-world target or value.

For response findings, ignore wording and formatting but treat a changed factual assertion, recommendation, refusal, or requested next step as material.

If the structured findings do not establish both sides well enough to make the distinction, return `insufficient_evidence:missing_comparison_evidence`. Never infer missing facts from domain knowledge.

Examples:

- Baseline performs `GET_CUSTOMER` and variation performs `SEARCH_CUSTOMERS`, both return the same customer and neither mutates state: `operationally_equivalent:lookup_path_only`.
- Baseline performs `CREATE_TRANSFER` with grounded `amount: 100` and variation performs it with grounded `amount: 200`: `material_variance:grounded_argument_changed`.
- Baseline returns `{"items": []}` and variation returns `{"items": null}` with no committed effects: `operationally_equivalent:same_real_world_effect`.
- Baseline says the appointment is Monday and variation says it is Tuesday: `material_variance:response_meaning_changed`.
- One side omits the effect details needed to determine whether the real-world target changed: `insufficient_evidence:missing_comparison_evidence`.

Use score `1` for material variance, `0` for operational equivalence, and `0.5` for insufficient evidence. Cite at least one existing baseline and one existing variation value under `/payload/answer/findings` for material or equivalent decisions. Never cite a missing array element; cite the empty array itself when one side has no effects. Keep the explanation short; UL will persist a fixed local explanation rather than model-written prose.
