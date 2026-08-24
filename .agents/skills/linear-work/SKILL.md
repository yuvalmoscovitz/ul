---
name: linear-work
description: Keep product and implementation work auditable in Linear with minimal overhead. Use when a task references a Linear issue or asks to search, create, prioritize, start, update, link a pull request to, or complete Linear work.
---

# Linear workflow

## Start

1. Fetch an explicit issue ID with its relations. Otherwise search by customer outcome before creating anything.
2. Reuse or update the canonical issue when scopes overlap. Create a new issue only for a distinct deliverable.
3. Read the parent or project only when needed to understand scope.

## Issue contract

Use an outcome title and a short description:

```markdown
## Outcome
<customer-visible result>

## Acceptance
- <observable proof>

## Out of scope
- <only boundaries needed to prevent scope drift>
```

Do not add labels, estimates, due dates, assignees, or sub-issues by habit.

## Dependencies

- Use `blocked by` when work or merge order must be sequential.
- Use `related` only for useful non-blocking context.
- Use a parent with sub-issues only when one deliverable needs multiple focused PRs.
- Before parallel implementation, check whether issues change the same public contract or acceptance journey. If they do, define the shared contract and blocking order first.
- Do not invent relationships.

## Work and pull requests

- Move the issue to In Progress when implementation starts.
- Put the issue ID in the branch and PR title or description so Linear can link it automatically.
- After opening the PR, verify the link once. Add it manually only if the integration did not.
- Move to In Review when the PR is ready for review, unless automation already did.
- Move to Done only after the PR is merged and confirmed on the default branch, unless automation already did.

## Keep it quiet

- Prefer the description, fields, relations, state history, and linked PR as the audit trail.
- Do not comment with routine progress, commits, test counts, CI status, review status, or “ready to merge.” Those belong in the PR and integrations.
- Comment only for a blocker needing human action, an approved scope or product decision not recorded elsewhere, or a handoff without a PR.
- Batch related field changes into one update.
- Do not create a second issue for implementation steps that fit one focused PR.
- Do not mark a parent Done until its outcome is delivered.
