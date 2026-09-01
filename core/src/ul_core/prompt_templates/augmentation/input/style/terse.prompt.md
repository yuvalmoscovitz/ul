+++
name = "augmentation.input.style.terse"
description = "Rewrites an input in a terse but natural style."
author = "UL"
+++
Rewrite the user input as a terse but natural message from a busy person, not a polished benchmark
sentence. Aim for 60 to 85 percent of the source word count by compressing clauses and removing
redundant function words, not by dropping details. Preserve every request, fact, value, constraint,
identifier, relationship, and request order. Keep names, identifiers, amounts, dates, quoted text,
and literal examples verbatim, but compress the surrounding wording. Preserve formatting rules and
other constraints in shorter words rather than copying their full sentences. Do not make the
request ambiguous or add context.
Never abbreviate a month name or another date expression. Preserve complete role noun phrases and
method noun phrases; for example, do not shorten a specific kind of recipient to a generic person or
drop the measured quantity from a named calculation method.
Before returning, count the words. If the draft still has more than 85 percent of the source word
count, compress the ordinary clauses again while retaining every protected detail.
