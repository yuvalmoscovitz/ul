# Prompt catalog

Every model-facing instruction owned by UL lives in this package. Runtime code looks prompts up by
their path-derived name through `PromptManager`; it does not embed prompt bodies as Python strings.

Prompt files use Markdown bodies with TOML frontmatter:

```text
+++
name = "semantic.example"
description = "Explains what this prompt controls."
author = "UL"
+++
Prompt text with a strict {{ required_variable }} placeholder.
```

The manager validates the full catalog when its singleton is first created. Names must match paths,
metadata fields are closed, missing and extra render variables fail, and each template exposes a
SHA-256 version for run provenance and analysis. Keep untrusted user or dataset content in user
messages; only interpolate values that are intentionally part of a trusted prompt.
