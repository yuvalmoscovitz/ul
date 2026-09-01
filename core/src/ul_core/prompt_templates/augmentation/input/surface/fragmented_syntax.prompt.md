+++
name = "augmentation.input.surface.fragmented_syntax"
description = "Rewrites an input with natural fragmented chat syntax."
author = "UL"
+++
Rewrite the user input as one natural sentence fragment from a person typing quickly. Omit the
subject and begin with a need or want construction, while keeping the complete request in that one
fragment. Express the desired result directly with the requested object and a passive action, such as
needing something added or updated. Do not write "need to" or "want to" followed by an action verb,
which can imply that the user will perform the action. Do not split related details into separate
fragments, and do not replace a noun phrase with a pronoun. Keep it conversational rather than
turning it into a bag of keywords. Start with normal sentence capitalization and end with a period.
Preserve every request, fact, value, constraint, identifier,
relationship, authorization, and request order. Keep names, identifiers, amounts, dates, and other
values verbatim; never abbreviate or reformat them. Do not add a reaction, explanation, or context.
