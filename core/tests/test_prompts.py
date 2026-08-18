from __future__ import annotations

import ast
from pathlib import Path

import pytest
from ul_core.prompts import PromptManager


def _write_prompt(root: Path, content: str) -> PromptManager:
    root.mkdir()
    (root / "example.prompt.md").write_text(content)
    return PromptManager(root)


def test_prompt_manager_is_a_singleton_and_loads_the_packaged_catalog() -> None:
    manager = PromptManager.instance()

    assert manager is PromptManager.instance()
    assert len(manager.list_templates()) == 23
    assert manager.get_prompt("examples.accounts_payable.tools.get_invoice") == (
        "Get the current invoice record by its exact ID."
    )


def test_prompt_manager_renders_strict_variables_and_exposes_provenance(tmp_path: Path) -> None:
    manager = _write_prompt(
        tmp_path / "prompts",
        "+++\n"
        'name = "example"\n'
        'description = "Example prompt."\n'
        'author = "UL"\n'
        "+++\n"
        "Hello, {{ name }}. Keep {{ item }} unchanged.\n",
    )

    assert manager.get_prompt("example", name="Ada", item="INV-104") == (
        "Hello, Ada. Keep INV-104 unchanged."
    )
    info = manager.get_template_info("example")
    assert info.variables == ("item", "name")
    assert len(info.version) == 64
    assert info.version == manager.get_template_info("example").version

    with pytest.raises(ValueError, match="missing variables: item"):
        manager.get_prompt("example", name="Ada")
    with pytest.raises(ValueError, match="unexpected variables: extra"):
        manager.get_prompt("example", name="Ada", item="INV-104", extra="value")


@pytest.mark.parametrize(
    ("content", "error"),
    [
        (
            "Prompt without frontmatter.",
            "missing TOML frontmatter",
        ),
        (
            "+++\n"
            'name = "wrong-name"\n'
            'description = "Example prompt."\n'
            'author = "UL"\n'
            "+++\n"
            "Body.\n",
            "name must match its path",
        ),
        (
            "+++\n"
            'name = "example"\n'
            'description = "Example prompt."\n'
            'author = "UL"\n'
            'owner = "unknown"\n'
            "+++\n"
            "Body.\n",
            "unknown metadata fields: owner",
        ),
        (
            "+++\n"
            'name = "example"\n'
            'description = "Example prompt."\n'
            'author = "UL"\n'
            "+++\n"
            "Malformed {{ variable.\n",
            "invalid template syntax",
        ),
    ],
)
def test_prompt_manager_rejects_invalid_catalogs(
    tmp_path: Path,
    content: str,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        _write_prompt(tmp_path / "prompts", content)


def test_runtime_prompt_call_sites_do_not_embed_prompt_bodies() -> None:
    repository_root = Path(__file__).parents[2]
    source_files = tuple(
        source_file
        for source_root in (
            repository_root / "core",
            repository_root / "sdk",
            repository_root / "cli",
            repository_root / "examples",
        )
        for source_file in source_root.rglob("*.py")
        if "tests" not in source_file.parts
    )
    violations: list[str] = []
    for source_file in source_files:
        tree = ast.parse(source_file.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.keyword)
                and node.arg in {"system_prompt", "instruction"}
                and isinstance(node.value, (ast.Constant, ast.JoinedStr))
            ):
                violations.append(f"{source_file.relative_to(repository_root)}:{node.lineno}")
            if not isinstance(node, ast.Dict):
                continue
            entries = {
                key.value: value
                for key, value in zip(node.keys, node.values, strict=True)
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            role = entries.get("role")
            content = entries.get("content")
            if (
                isinstance(role, ast.Constant)
                and role.value == "system"
                and isinstance(content, (ast.Constant, ast.JoinedStr))
            ):
                violations.append(f"{source_file.relative_to(repository_root)}:{node.lineno}")

    assert violations == []
