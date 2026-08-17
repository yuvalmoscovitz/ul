# Contributing

Thanks for helping improve UL. Please discuss non-trivial changes in an issue before starting.
Report vulnerabilities as described in the [security policy](SECURITY.md), not in a public issue.

## Setup

```bash
git clone https://github.com/yuvalmoscovitz/ul.git
cd ul
uv sync --locked
```

Use synthetic data in tests and issues. Never include secrets, credentials, production traces, or
customer data.

## Checks

```bash
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pyright
uv run --frozen pytest -q
uv build --no-sources
```

Keep pull requests focused. Add behavior-level tests, update affected documentation, and explain
the user outcome and limitations. Discuss new major dependencies before adding them.
