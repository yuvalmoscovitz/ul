# Prompt catalog

Every model-facing instruction owned by UL lives in this package. Runtime code looks prompts up by
their path-derived name through `PromptManager`; it does not embed prompt bodies as Python strings.

Each prompt is a `*.prompt.md` file. Its name is its relative path below this directory with
`/` replaced by `.` and `.prompt.md` removed. For example,
`semantic/render.prompt.md` must declare `name = "semantic.render"`.

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
metadata fields are closed, and missing, extra, or non-string render variables fail.

```python
from ul_core.prompts import PromptManager

prompts = PromptManager.instance()
prompts.list_templates()
prompts.get_template_info("semantic.render")
prompts.get_prompt("semantic.render", temporary_value_rule="Trusted rule text.")
```

`version` is the SHA-256 of the exact model-facing template body, so metadata-only edits do not
split behaviorally identical runs. `source_version` fingerprints the complete file, including
frontmatter. Composed prompts record every source template in their provenance.

To add or rename a prompt, create or move its file, make the declared name match the path, update
the runtime lookup, and add or update a behavior-level test. Run:

```bash
uv run --frozen pytest core/tests/test_prompts.py
```

Catalog files are trusted repository code, not a tenant customization surface. Protect write access
to them. A hash is a reproducibility fingerprint, not proof that content is safe or authentic.
Keep untrusted user, dataset, and externally supplied text in user messages; do not interpolate it
into system or developer instructions. The private custom-root loader exists only for catalog tests.
