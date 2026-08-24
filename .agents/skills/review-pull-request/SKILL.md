---
name: review-pull-request
description: Review an implementation-complete pull request once, using only the independent review perspectives triggered by its risk and customer surface.
---

# Review a pull request

First confirm that the pull request exists, implementation is complete, and required checks are green.

Run one strongest-model primary reviewer for every PR. Review the base-to-head diff for correctness, simplicity, tests, product and task fit, and governing instructions.

Add only the reviewers triggered by the diff:

- Add an independent security reviewer when the change touches untrusted input, authentication or secrets, network, filesystem or process execution, persistence, permissions, dependencies, or another trust boundary.
- Add a customer reviewer when the change alters a public CLI, API, UI, documented command, or customer workflow. The customer must exercise the changed journey through the public surface with real local dependencies where practical. Helper-generated artifacts are not sufficient.

For an ordered stack of PRs implementing one customer journey, the primary reviewer still examines each PR. A security review or customer journey may cover the composed stack once if it names every covered PR, examines each relevant diff, and attributes findings to the responsible PR.

Run selected perspectives in parallel when slots are available. Implementation agents must release their slots before formal review.

Report only findings attributable to the diff. Block on correctness, security, data loss, a broken customer journey, or a violated product contract. Mark polish and speculative improvements as optional; they do not block merge.

Run this formal review once. After fixes, inspect only the fix delta and rerun affected tests and required gates. Do not launch another full review unless the scope or trust boundary materially changed; in that case stop and rescope the PR.
