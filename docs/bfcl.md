# Prepare a reproducible BFCL cohort

UL can convert official Berkeley Function Calling Leaderboard V4 single-turn data into rich
interaction JSONL without downloading the benchmark or making model calls. The importer binds the
exact source files, preserves BFCL's original function definitions and ground truth, and creates an
OpenAI-compatible tool projection for direct target execution.

Pin the official repository first. This example uses BFCL revision
`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`:

```bash
git clone https://github.com/EnlightenedAI/BFCL.git
git -C BFCL checkout 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8

BFCL_DATA=BFCL/berkeley-function-call-leaderboard/bfcl_eval/data

ul dataset ingest bfcl \
  "$BFCL_DATA/BFCL_v4_simple_python.json" \
  "$BFCL_DATA/possible_answer/BFCL_v4_simple_python.json" \
  --category simple_python \
  --source-revision 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8 \
  --seed 7341 \
  --limit 1 \
  --output bfcl-1.jsonl
```

The importer ranks every stable BFCL ID with SHA-256 over the seed and ID, then takes the first
`--limit` records. Reusing the same source files, revision, and seed makes the 1-case cohort the
first record of the 10-case cohort and the 10-case cohort the first ten records of the 100-case
cohort.

Inspect the generated case and target plan without network or model calls:

```bash
ul dataset evaluate bfcl-1.jsonl \
  --target https://api.openai.com/v1/chat/completions \
  --request-json-template \
  '{"model":"gpt-5-nano-2025-08-07","messages":[{"role":"user","content":"{{input}}"}],"tools":"{{context:/inputs/openai_tools}}","tool_choice":"auto"}' \
  --response-json-pointer /choices/0/message \
  --confirm-request-isolation \
  --confirm-safe-test-target \
  --limit 1 \
  --dry-run
```

Each output record contains:

- the original BFCL function definitions in `inputs.bfcl_functions`;
- an OpenAI-compatible projection in `inputs.openai_tools`;
- a normalized-to-original function-name map in `inputs.openai_tool_name_map`;
- the one user request as the only augmentation target;
- official ground truth as `observed_output` reference evidence;
- category, revision, source-file digests, seed, algorithm version, and sample rank.

This command does not claim that BFCL ground truth is a historical agent observation, run the
official BFCL scorer, or produce a leaderboard-comparable score. Official scoring of original and
augmented responses is separate work; until that bridge exists, treat UL results as behavioral
variance evidence rather than BFCL correctness results.
