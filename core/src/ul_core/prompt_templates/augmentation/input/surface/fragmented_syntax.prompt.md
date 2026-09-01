+++
name = "augmentation.input.surface.fragmented_syntax"
description = "Rewrites an input with natural fragmented chat syntax."
author = "UL"
+++
Turn exactly one short request clause into a natural sentence fragment from a person typing
quickly. Leave every other sentence and clause as close to the source as possible; never combine a
multi-sentence request into one long fragment. In the changed clause, omit the subject and begin
with a need or want construction. Express the desired result directly with the requested object and
a passive action, such as needing something added or updated. Do not write "need to" or "want to"
followed by an action verb, which can imply that the user will perform the action. Do not replace a
noun phrase with a pronoun. Keep it conversational rather than turning it into a bag of keywords.
Use normal sentence capitalization and punctuation.
Preserve every request, fact, value, constraint, identifier,
relationship, authorization, and request order. Keep names, identifiers, amounts, dates, and other
values verbatim; never abbreviate or reformat them. Do not add a reaction, explanation, or context.
