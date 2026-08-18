# Implementation research

The privacy boundary follows two maintained primary-source patterns:

- [Microsoft Presidio Anonymizer](https://microsoft.github.io/presidio/anonymizer/) separates
  reversible and irreversible operators and documents that consistent secret material is required
  for referential integrity. UL adopts that separation but uses explicit selectors instead of
  automatic PII detection.
- [Langfuse masking](https://langfuse.com/docs/observability/features/masking) applies masking before
  observability data leaves the SDK boundary. UL similarly applies one boundary before every
  semantic-provider call, while rehydrating reversible values only at target execution.

No dependency was added for these patterns.
