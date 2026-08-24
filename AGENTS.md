# Agent Guidelines for UL

UL is an open-source tool that helps teams running high-risk AI agents actively discover failures in their AI agents.

When `./VISION.md` is present, treat it as the high-level product authority.

## General Guidelines

- Parallelize only independent work. Before parallel implementation, identify shared public contracts, dependencies, and merge order. Sequence work that changes the same contract; research, fixtures, tests, and documentation may still run in parallel.
- An implementation subagent must finish and release its slot after opening the PR and reporting validation. Formal review is orchestrated separately.
- For small independent tasks you can use less intelligent subagents. Always plan with the utmost intelligence.
- When uncertain about an important decision, ask instead of guessing.
- Before implementing, state the intended product outcome and decide whether the task fits one focused PR or must be split into multiple PRs; do not change that scope without human approval.

## Coding Guidelines

### Before Implementing

- State your assumptions explicitly. If uncertain, ask.
- Aim for the simplest approach; don't overcomplicate.
- Always look online for inspiration; both for high-level design and for seeing how others have already solved similar problems.
    Good reference projects and docs:
    - https://github.com/psf/requests
    - https://github.com/fastapi/fastapi/tree/master/fastapi
    - https://github.com/promptfoo/promptfoo
    - https://github.com/langfuse/langfuse
    - https://docs.stripe.com/development
    - Many more open source projects and docs that are highly maintained.

### Implementation

#### General

- Use the minimum code needed to solve the problem.
- Don't over abstract.
- Don't handle impossible errors.
- Only change parts you must.
- Always remove dead code and cleanup after a feature.
- Don't write comments unless you absolutely must; the code should be clear enough on its own.
- Prefer long variable and function names; always prefer understandable over simple naming.
- Keep consistent style and patterns across the codebase.
- When finished, zoom out and check that the change still makes sense in the broader codebase.
- Security is high priority: no secrets in code, validate untrusted input, and prefer least privilege.
- We have no users yet, so no backwards compatibility is needed.

#### Testing and validation

- Always define clear success criteria.
- For user-facing changes, define one executable customer journey through the public CLI or API before implementation. Helper-generated artifacts cannot be the only end-to-end proof.
- Write real tests that verify actual behavior; avoid heavy mocking of the thing under test.
- Loop until the success criteria are met.

## Architecture

UL is a monorepo that contains:

- `core` — shared logic and models
- `cli` — command-line interface (Typer)
- `sdk` — Python SDK
- `ui` — web interface

The same codebase is used for local runs and cloud hosting.

## Tech Requirements

- Use `uv` for all dependency and environment management.
- Use `pydantic` + `pydantic-settings` for data validation and configuration.
- Use `typer` for any CLI.
- Use `httpx` for HTTP requests; never use `requests`.
- Use `ruff` for linting and formatting.
- Use `pyright` for type checking.
- Use `pytest` for tests.
- Use `rich` for terminal output and logging.
- Use OpenTelemetry for trace ingestion when relevant.
- Use `tenacity` for retries.
- Use `platformdirs` for config and cache paths.
- Prefer the standard library whenever it is good enough.
- Do not introduce new major dependencies without asking first.

## Development Lifecycle

- Use the `linear-work` skill whenever work references Linear.
- Always work in a git worktree at `../ul.worktrees/`.
- Always branch off `origin/main`.
- Never commit with `--no-verify` or otherwise skip pre-commit.
- Commit only relevant changes.
- Put all temporary files in `./tmp` and delete temporary planning or documentation files when finished.
- Do not merge a PR unless explicitly asked. After merging, confirm it reached `origin/main`, then delete the branch and worktree.
- Before merging, run the `review-pull-request` skill once, after the PR is implementation-complete and required checks are green. The skill selects any needed security or customer review; do not run duplicate reviews.
- After review fixes, inspect only the changed delta and run affected tests plus required gates. Do not repeat the full review. If fixes materially expand scope or a trust boundary, stop and rescope.
- Only do deep code reviews on pull requests. If asked to review code, first make sure it is packaged as a pull request.
- Be critical of review comments; accept only what clearly improves the code.
- Keep changes focused and incremental.
- Never push directly to `main`.
- Never commit secrets, credentials, or private data.

**Don't add anything to this file without getting human approval.**
