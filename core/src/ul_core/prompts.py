from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass
from functools import cache
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Self

from pydantic import JsonValue

_PROMPT_SUFFIX = ".prompt.md"
_FRONTMATTER_DELIMITER = "+++"
_PROMPT_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_VARIABLE_PATTERN = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")
_MAX_PROMPT_BYTES = 256 * 1024
_METADATA_FIELDS = frozenset({"name", "description", "author"})


@dataclass(frozen=True)
class PromptTemplateInfo:
    name: str
    description: str
    author: str
    variables: tuple[str, ...]
    version: str


@dataclass(frozen=True)
class _PromptTemplate:
    info: PromptTemplateInfo
    body: str


class PromptManager:
    def __init__(self, prompt_root: Traversable | Path | None = None) -> None:
        root = prompt_root or resources.files("ul_core").joinpath("prompt_templates")
        self._templates = self._load_templates(root)

    @classmethod
    @cache
    def instance(cls) -> Self:
        return cls()

    def get_prompt(self, prompt_name: str, **variables: str) -> str:
        template = self._get_template(prompt_name)
        expected_variables = set(template.info.variables)
        supplied_variables = set(variables)
        missing_variables = expected_variables - supplied_variables
        unexpected_variables = supplied_variables - expected_variables
        if missing_variables:
            missing = ", ".join(sorted(missing_variables))
            raise ValueError(f"prompt {prompt_name!r} is missing variables: {missing}")
        if unexpected_variables:
            unexpected = ", ".join(sorted(unexpected_variables))
            raise ValueError(f"prompt {prompt_name!r} received unexpected variables: {unexpected}")
        return _VARIABLE_PATTERN.sub(lambda match: variables[match.group(1)], template.body)

    def get_template_info(self, name: str) -> PromptTemplateInfo:
        return self._get_template(name).info

    def list_templates(self) -> tuple[PromptTemplateInfo, ...]:
        return tuple(template.info for template in self._templates.values())

    def _get_template(self, name: str) -> _PromptTemplate:
        try:
            return self._templates[name]
        except KeyError as error:
            raise KeyError(f"unknown prompt: {name}") from error

    @classmethod
    def _load_templates(cls, root: Traversable | Path) -> dict[str, _PromptTemplate]:
        templates: dict[str, _PromptTemplate] = {}
        for relative_path, prompt_file in cls._prompt_files(root):
            content = prompt_file.read_text(encoding="utf-8")
            if len(content.encode()) > _MAX_PROMPT_BYTES:
                raise ValueError(f"prompt {relative_path!r} exceeds {_MAX_PROMPT_BYTES} bytes")
            template = cls._parse_template(relative_path, content)
            if template.info.name in templates:
                raise ValueError(f"duplicate prompt name: {template.info.name}")
            templates[template.info.name] = template
        if not templates:
            raise ValueError("prompt catalog is empty")
        return dict(sorted(templates.items()))

    @classmethod
    def _prompt_files(
        cls,
        root: Traversable | Path,
        prefix: str = "",
    ) -> tuple[tuple[str, Traversable | Path], ...]:
        prompt_files: list[tuple[str, Traversable | Path]] = []
        for child in sorted(root.iterdir(), key=lambda entry: entry.name):
            relative_path = f"{prefix}/{child.name}" if prefix else child.name
            if child.is_dir():
                prompt_files.extend(cls._prompt_files(child, relative_path))
            elif child.name.endswith(_PROMPT_SUFFIX):
                prompt_files.append((relative_path, child))
        return tuple(prompt_files)

    @staticmethod
    def _parse_template(relative_path: str, content: str) -> _PromptTemplate:
        opening = f"{_FRONTMATTER_DELIMITER}\n"
        closing = f"\n{_FRONTMATTER_DELIMITER}\n"
        if not content.startswith(opening):
            raise ValueError(f"prompt {relative_path!r} is missing TOML frontmatter")
        frontmatter_text, separator, body = content[len(opening) :].partition(closing)
        if not separator:
            raise ValueError(f"prompt {relative_path!r} is missing closing frontmatter")
        try:
            metadata = tomllib.loads(frontmatter_text)
        except tomllib.TOMLDecodeError as error:
            message = f"prompt {relative_path!r} has invalid frontmatter: {error}"
            raise ValueError(message) from error
        unknown_fields = set(metadata) - _METADATA_FIELDS
        missing_fields = _METADATA_FIELDS - set(metadata)
        if unknown_fields:
            fields = ", ".join(sorted(unknown_fields))
            raise ValueError(f"prompt {relative_path!r} has unknown metadata fields: {fields}")
        if missing_fields:
            fields = ", ".join(sorted(missing_fields))
            raise ValueError(f"prompt {relative_path!r} is missing metadata fields: {fields}")
        if not all(isinstance(metadata[field], str) for field in _METADATA_FIELDS):
            raise ValueError(f"prompt {relative_path!r} metadata values must be strings")
        name = metadata["name"].strip()
        if not _PROMPT_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"prompt {relative_path!r} has invalid name: {name!r}")
        expected_name = relative_path.removesuffix(_PROMPT_SUFFIX).replace("/", ".")
        if name != expected_name:
            raise ValueError(
                f"prompt {relative_path!r} name must match its path: {expected_name!r}"
            )
        description = metadata["description"].strip()
        author = metadata["author"].strip()
        if not description or not author:
            raise ValueError(f"prompt {relative_path!r} metadata values must not be empty")
        body = body.removesuffix("\n")
        if not body.strip():
            raise ValueError(f"prompt {relative_path!r} body must not be empty")
        variables = tuple(sorted(set(_VARIABLE_PATTERN.findall(body))))
        unmatched_template_syntax = _VARIABLE_PATTERN.sub("", body)
        if "{{" in unmatched_template_syntax or "}}" in unmatched_template_syntax:
            raise ValueError(f"prompt {relative_path!r} has invalid template syntax")
        version = hashlib.sha256(content.encode()).hexdigest()
        return _PromptTemplate(
            info=PromptTemplateInfo(
                name=name,
                description=description,
                author=author,
                variables=variables,
                version=version,
            ),
            body=body,
        )


def prompt_provenance(*names: str) -> list[JsonValue]:
    manager = PromptManager.instance()
    provenance: list[JsonValue] = []
    for info in (manager.get_template_info(name) for name in names):
        provenance.append({"name": info.name, "version": info.version})
    return provenance
