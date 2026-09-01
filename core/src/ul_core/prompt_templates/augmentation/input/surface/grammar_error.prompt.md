+++
name = "augmentation.input.surface.grammar_error"
description = "Introduces one harmless grammatical error."
author = "UL"
+++
Make exactly one obvious but natural grammatical error inside the existing message, as a real
person might while typing. Preserve the source language and keep the wording otherwise as close as
possible. Before returning, compare the result with the source and confirm internally that exactly
one word changed or was removed. Prefer a wrong article, a missing article, or ordinary
subject-verb disagreement. Never return the source unchanged. Do not add a preface, reaction,
explanation, or filler.
Apply the error in the first ordinary sentence. If no better natural error is obvious, remove one
"a," "an," or "the" from that sentence. This fallback is required instead of returning unchanged
text.
Preserve every request, fact, value, identifier, relationship, constraint, authorization, and
request order.
