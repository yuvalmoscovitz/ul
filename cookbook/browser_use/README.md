# Browser Use target adapter

This cookbook connects a [Browser Use](https://github.com/browser-use/browser-use) agent to UL's
trusted Python target factory. Each UL execution gets a new headless, incognito browser session;
UL records the terminal response, final URL, Browser Use's self-reported success, and the names and
count of actions taken.

Use this recipe only against a local test application or an isolated staging clone. A browser agent
can click and submit forms, and Browser Use's `allowed_domains` is defense in depth rather than a
complete sandbox. Do not point this adapter at production, an authenticated user profile, or a site
where actions can affect customers or money.

## Set up

From the UL repository root:

```bash
uv sync --project cookbook/browser_use
uv sync --project cookbook/browser_use/worker --extra browser-use
```

Browser Use 0.13.7 and UL currently require incompatible major versions of Rich. The second project
therefore runs Browser Use in a separate worker environment instead of weakening either project's
dependency guarantees. Browser Use is not added to UL itself. The worker uses Browser Use's stable
Python agent API, not the newer beta API.

Configure an OpenAI-backed model and exact origins. Origins cannot contain paths or wildcards.

```bash
export OPENAI_API_KEY="..."
export UL_BROWSER_USE_MODEL="gpt-5-mini"
export UL_BROWSER_USE_ALLOWED_ORIGINS="http://127.0.0.1:8765"
export UL_BROWSER_USE_WORKER_PYTHON="$PWD/cookbook/browser_use/worker/.venv/bin/python"
export UL_BROWSER_USE_ISOLATED_TEST_ENVIRONMENT="I_CONFIRM_THIS_IS_AN_ISOLATED_TEST_ENVIRONMENT"
```

Optional bounds are `UL_BROWSER_USE_MAX_STEPS` (default 10, maximum 25),
`UL_BROWSER_USE_TIMEOUT_SECONDS` (default 180), and `UL_BROWSER_USE_MAX_RESULT_CHARACTERS`
(default 4000). Browser Use reads the model credential directly from the environment; the adapter
does not place it in UL results.

Run a normal UL dataset evaluation with the cookbook environment:

```bash
uv run --project cookbook/browser_use ul dataset evaluate dataset.jsonl \
  --invariants invariants.json \
  --target-factory ul_browser_use_adapter:create_target \
  --allow-target-network \
  --confirm-isolated-sandbox \
  --output findings.json
```

UL starts a fresh worker and Browser Use session for every input, then kills the session before the
worker exits. It never attaches to an existing Chrome profile and does not restore cookies or local
storage. This resets browser storage, but it does not reset server-side state; the allowlisted
application must provide a fresh isolated fixture for every task. Browser Use's `success` value is
the agent's own report, so customer invariants should decide whether the resulting actions and
application state are actually correct.

The adapter rejects task text containing an HTTP(S) URL outside the configured origins, bounds task
and result sizes, limits agent steps, detects a final URL outside the allowlist, sanitizes runtime
errors, and terminates the isolated worker process tree even when execution fails. Origins are
converted to slash-delimited Browser Use navigation patterns so hostname and port prefixes cannot
match the allowlist. For stronger containment, run the cookbook in a network-isolated container
whose egress policy permits only the model endpoint and the test application.

References: [Browser Use browser settings](https://github.com/browser-use/browser-use/blob/main/skills/open-source/references/browser.md),
[agent history API](https://github.com/browser-use/browser-use/blob/main/browser_use/agent/views.py),
and [supported models](https://github.com/browser-use/browser-use/blob/main/skills/open-source/references/models.md).
