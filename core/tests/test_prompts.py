from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, cast

import pytest
from ul_core.prompts import PromptManager


def _write_prompt(root: Path, content: str) -> PromptManager:
    root.mkdir()
    (root / "example.prompt.md").write_bytes(content.encode())
    return PromptManager._from_root(root)


def test_prompt_manager_is_a_singleton_and_loads_the_packaged_catalog() -> None:
    manager = PromptManager.instance()

    assert manager is PromptManager.instance()
    assert len(manager.list_templates()) == 30
    assert manager.get_prompt("evaluation.judge").startswith(
        "Evaluate the untrusted JSON payload only against the supplied rubric."
    )
    assert manager.get_prompt("semantic.preflight") == (
        'Return exactly {"compatible":true}. This is a bounded evaluator compatibility check.'
    )
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
    assert len(info.source_version) == 64
    assert info.version == manager.get_template_info("example").version

    with pytest.raises(ValueError, match="missing variables: item"):
        manager.get_prompt("example", name="Ada")
    with pytest.raises(ValueError, match="unexpected variables: extra"):
        manager.get_prompt("example", name="Ada", item="INV-104", extra="value")

    for invalid_value, type_name in ((None, "NoneType"), (7, "int"), (object(), "object")):
        with pytest.raises(
            ValueError,
            match=f"variable 'item' must be a string, got {type_name}",
        ):
            unsafe_variables = cast(dict[str, Any], {"name": "Ada", "item": invalid_value})
            manager.get_prompt("example", **unsafe_variables)


def test_behavior_version_ignores_metadata_only_changes(tmp_path: Path) -> None:
    first_manager = _write_prompt(
        tmp_path / "first",
        "+++\n"
        'name = "example"\n'
        'description = "First description."\n'
        'author = "UL"\n'
        "+++\n"
        "Same body.\n",
    )
    second_manager = _write_prompt(
        tmp_path / "second",
        "+++\n"
        'name = "example"\n'
        'description = "Updated description."\n'
        'author = "UL"\n'
        "+++\n"
        "Same body.\n",
    )

    first_info = first_manager.get_template_info("example")
    second_info = second_manager.get_template_info("example")
    assert first_info.version == second_info.version
    assert first_info.source_version != second_info.source_version


def test_prompt_manager_normalizes_crlf_for_rendering_and_behavior_version(
    tmp_path: Path,
) -> None:
    lf_content = (
        "+++\n"
        'name = "example"\n'
        'description = "Example prompt."\n'
        'author = "UL"\n'
        "+++\n"
        "Hello, {{ name }}.\n"
    )
    crlf_content = lf_content.replace("\n", "\r\n")
    lf_manager = _write_prompt(tmp_path / "lf", lf_content)
    crlf_manager = _write_prompt(tmp_path / "crlf", crlf_content)

    assert crlf_manager.get_prompt("example", name="Ada") == "Hello, Ada."
    assert crlf_manager.get_template_info("example").version == (
        lf_manager.get_template_info("example").version
    )
    assert crlf_manager.get_template_info("example").source_version != (
        lf_manager.get_template_info("example").source_version
    )


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


def test_prompt_manager_rejects_oversized_prompt_before_decoding(tmp_path: Path) -> None:
    root = tmp_path / "prompts"
    root.mkdir()
    (root / "example.prompt.md").write_bytes(b"x" * (256 * 1024 + 1))

    with pytest.raises(ValueError, match="exceeds 262144 bytes"):
        PromptManager._from_root(root)


def test_prompt_manager_rejects_excess_catalog_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "prompts"
    root.mkdir()
    for index in range(3):
        (root / f"ignored-{index}.txt").touch()
    monkeypatch.setattr("ul_core.prompts._MAX_CATALOG_ENTRIES", 2)

    with pytest.raises(ValueError, match="exceeds 2 entries"):
        PromptManager._from_root(root)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX named pipes")
def test_prompt_manager_rejects_special_prompt_files(tmp_path: Path) -> None:
    root = tmp_path / "prompts"
    root.mkdir()
    os.mkfifo(root / "example.prompt.md")

    with pytest.raises(ValueError, match="entry is not a file"):
        PromptManager._from_root(root)


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
