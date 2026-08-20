from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import stat
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Literal, Protocol, Self, cast

from pydantic import ConfigDict, Field, JsonValue, SecretStr, model_validator
from ul_core.contracts import (
    SandboxExecutor,
    SemanticDeconstructor,
    SemanticEquivalenceVerifier,
    SemanticRenderer,
)
from ul_core.dataset import (
    InteractionRecord,
    RenderedUserInput,
    SemanticEquivalenceAssessment,
    SemanticFrame,
    UserInputRecord,
)
from ul_core.evaluation import EvaluationCase, ExecutionEvidence, SandboxCapabilities
from ul_core.models import ULModel

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

RedactionLocation = Literal["input", "output", "context"]
RedactionAction = Literal["pseudonymize", "remove", "replace"]
_TEXT_SELECTOR = "$text"
_PLACEHOLDER_PATTERN = re.compile(r"__UL_SECRET_[a-z][a-z0-9_-]{0,31}_[0-9a-f]{32}__")
_MAXIMUM_STATE_BYTES = 64 * 1024 * 1024
_MAXIMUM_POLICY_BYTES = 1_000_000
_MAXIMUM_VALUE_BYTES = 5_000_000
_MAXIMUM_MATCHES = 100_000


class RedactionBoundaryError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("sensitive-data boundary failed closed")


class RedactionRule(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")
    locations: tuple[RedactionLocation, ...] = ("input",)
    selector: str = _TEXT_SELECTOR
    literal: str | None = Field(default=None, min_length=1, max_length=1_000)
    action: RedactionAction = "pseudonymize"
    replacement: JsonValue = "[REDACTED]"

    @model_validator(mode="after")
    def validate_rule(self) -> Self:
        if not self.locations or len(set(self.locations)) != len(self.locations):
            raise ValueError("rule locations must be non-empty and unique")
        if self.selector == _TEXT_SELECTOR:
            if self.literal is None:
                raise ValueError("text selectors require a literal value")
            if self.action == "replace" and not isinstance(self.replacement, str):
                raise ValueError("text replacement must be a string")
        else:
            _parse_json_pointer(self.selector)
            if self.literal is not None:
                raise ValueError("JSON pointer selectors do not accept a literal")
        if self.action != "pseudonymize" and "input" in self.locations:
            raise ValueError("executable input only supports reversible pseudonymization")
        if self.action != "replace" and self.replacement != "[REDACTED]":
            raise ValueError("replacement is only valid for replace rules")
        return self


class RedactionPolicy(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[1] = 1
    rules: tuple[RedactionRule, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_rules(self) -> Self:
        names = tuple(rule.name for rule in self.rules)
        if len(names) != len(set(names)):
            raise ValueError("redaction rule names must be unique")
        pointer_targets = [
            (location, rule.selector)
            for rule in self.rules
            if rule.selector != _TEXT_SELECTOR
            for location in rule.locations
        ]
        if len(pointer_targets) != len(set(pointer_targets)):
            raise ValueError("a JSON pointer may be selected only once per location")
        return self

    @property
    def digest(self) -> str:
        encoded_policy = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded_policy).hexdigest()


class RedactionCoverage(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    matched_values: int = Field(ge=0)
    matched_paths: tuple[str, ...] = ()
    matches_by_rule: dict[str, int] = Field(default_factory=dict)


class RedactionResult(ULModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    value: JsonValue
    coverage: RedactionCoverage


class LocalPseudonymStore:
    def __init__(self, path: str | Path, key: SecretStr) -> None:
        self.path = Path(path)
        self._key = key.get_secret_value().encode()
        if len(self._key) < 32:
            raise ValueError("pseudonymization key must contain at least 32 UTF-8 bytes")
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")

    def pseudonymize(self, rule_name: str, value: JsonValue, *, dry_run: bool = False) -> str:
        encoded_value = _encode_value(value)
        digest = hmac.new(
            self._key,
            rule_name.encode() + b"\0" + encoded_value,
            hashlib.sha256,
        ).hexdigest()[:32]
        placeholder = f"__UL_SECRET_{rule_name}_{digest}__"
        if dry_run:
            return placeholder
        with self._locked_state() as state:
            mappings = cast(dict[str, str], state["mappings"])
            existing_value = mappings.get(placeholder)
            stored_value = base64.b64encode(encoded_value).decode("ascii")
            if existing_value is not None and not hmac.compare_digest(existing_value, stored_value):
                raise RedactionBoundaryError()
            if existing_value is None:
                mappings[placeholder] = stored_value
                self._write_state(state)
        return placeholder

    def rehydrate_text(self, value: str) -> str:
        placeholders = tuple(dict.fromkeys(_PLACEHOLDER_PATTERN.findall(value)))
        if not placeholders:
            return value
        with self._locked_state() as state:
            mappings = cast(dict[str, str], state["mappings"])
            decoded: dict[str, str] = {}
            for placeholder in placeholders:
                encoded_value = mappings.get(placeholder)
                if encoded_value is None:
                    raise RedactionBoundaryError()
                try:
                    original = json.loads(base64.b64decode(encoded_value, validate=True))
                except (ValueError, json.JSONDecodeError):
                    raise RedactionBoundaryError() from None
                decoded[placeholder] = (
                    original
                    if isinstance(original, str)
                    else json.dumps(original, ensure_ascii=False, separators=(",", ":"))
                )
        return _PLACEHOLDER_PATTERN.sub(lambda match: decoded[match.group()], value)

    def validate_placeholders(self, value: JsonValue) -> None:
        placeholders = tuple(dict.fromkeys(_find_placeholders(value)))
        if not placeholders:
            return
        with self._locked_state() as state:
            mappings = cast(dict[str, str], state["mappings"])
            if any(placeholder not in mappings for placeholder in placeholders):
                raise RedactionBoundaryError()

    class _StateLock:
        def __init__(self, store: LocalPseudonymStore) -> None:
            self.store = store
            self.descriptor: int | None = None
            self.state: dict[str, object] | None = None

        def __enter__(self) -> dict[str, object]:
            self.store.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            parent_mode = stat.S_IMODE(self.store.path.parent.stat().st_mode)
            if parent_mode & 0o077:
                raise RedactionBoundaryError()
            descriptor = os.open(self.store._lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            self.descriptor = descriptor
            if stat.S_IMODE(os.fstat(descriptor).st_mode) & 0o077:
                os.close(descriptor)
                self.descriptor = None
                raise RedactionBoundaryError()
            _lock_descriptor(descriptor)
            try:
                self.state = self.store._read_state()
            except BaseException:
                _unlock_descriptor(descriptor)
                os.close(descriptor)
                self.descriptor = None
                raise
            return self.state

        def __exit__(self, *args: object) -> None:
            if self.descriptor is not None:
                _unlock_descriptor(self.descriptor)
                os.close(self.descriptor)

    def _locked_state(self) -> LocalPseudonymStore._StateLock:
        return self._StateLock(self)

    def _read_state(self) -> dict[str, object]:
        if not self.path.exists():
            return {"version": 1, "mappings": {}}
        file_status = self.path.lstat()
        if not stat.S_ISREG(file_status.st_mode) or stat.S_IMODE(file_status.st_mode) & 0o077:
            raise RedactionBoundaryError()
        if file_status.st_size > _MAXIMUM_STATE_BYTES:
            raise RedactionBoundaryError()
        try:
            decoded_state: object = json.loads(self.path.read_bytes())
        except (OSError, json.JSONDecodeError):
            raise RedactionBoundaryError() from None
        if not isinstance(decoded_state, dict):
            raise RedactionBoundaryError()
        state = cast(dict[str, object], decoded_state)
        if set(state) != {"version", "mappings", "integrity"}:
            raise RedactionBoundaryError()
        integrity = state.pop("integrity")
        expected_integrity = self._integrity(state)
        if not isinstance(integrity, str) or not hmac.compare_digest(integrity, expected_integrity):
            raise RedactionBoundaryError()
        mappings = state.get("mappings")
        if not isinstance(mappings, dict):
            raise RedactionBoundaryError()
        string_mappings = cast(dict[object, object], mappings)
        if state.get("version") != 1 or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in string_mappings.items()
        ):
            raise RedactionBoundaryError()
        state["mappings"] = cast(dict[str, str], string_mappings)
        return state

    def _write_state(self, state: Mapping[str, object]) -> None:
        persisted_state = {**state, "integrity": self._integrity(state)}
        encoded_state = json.dumps(persisted_state, sort_keys=True, separators=(",", ":")).encode()
        if len(encoded_state) > _MAXIMUM_STATE_BYTES:
            raise RedactionBoundaryError()
        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.", dir=self.path.parent
            )
            temporary_path = Path(temporary_name)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as state_file:
                state_file.write(encoded_state)
                state_file.flush()
                os.fsync(state_file.fileno())
            os.replace(temporary_path, self.path)
            directory_descriptor = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            raise RedactionBoundaryError() from None
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def _integrity(self, state: Mapping[str, object]) -> str:
        encoded = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
        return hmac.new(self._key, encoded, hashlib.sha256).hexdigest()


class RedactionEngine:
    def __init__(self, policy: RedactionPolicy, store: LocalPseudonymStore) -> None:
        self.policy = policy
        self.store = store

    def transform(
        self,
        value: JsonValue,
        *,
        location: RedactionLocation,
        dry_run: bool = False,
    ) -> RedactionResult:
        try:
            encoded_value_size = len(
                json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
            )
        except (RecursionError, ValueError):
            raise RedactionBoundaryError() from None
        if encoded_value_size > _MAXIMUM_VALUE_BYTES:
            raise RedactionBoundaryError()
        self.store.validate_placeholders(value)
        transformed = _copy_json(value)
        matches_by_rule: Counter[str] = Counter()
        matched_paths: set[str] = set()
        pointer_selected_paths: set[str] = set()
        for rule in self.policy.rules:
            if location not in rule.locations or rule.selector == _TEXT_SELECTOR:
                continue
            tokens = _parse_json_pointer(rule.selector)
            try:
                selected_value = _resolve_pointer(transformed, tokens)
            except RedactionBoundaryError:
                continue
            if _find_placeholders(selected_value):
                pointer_selected_paths.add(rule.selector)
                continue
            replacement = self._replacement(rule, selected_value, dry_run=dry_run)
            transformed = _set_or_remove_pointer(transformed, tokens, replacement, rule.action)
            matches_by_rule[rule.name] += 1
            matched_paths.add(rule.selector)
            pointer_selected_paths.add(rule.selector)
        text_rules = tuple(
            rule
            for rule in self.policy.rules
            if location in rule.locations and rule.selector == _TEXT_SELECTOR
        )
        transformed = self._transform_text_values(
            transformed,
            text_rules,
            pointer_selected_paths,
            matches_by_rule,
            matched_paths,
            dry_run=dry_run,
        )
        return RedactionResult(
            value=transformed,
            coverage=RedactionCoverage(
                policy_sha256=self.policy.digest,
                matched_values=sum(matches_by_rule.values()),
                matched_paths=tuple(sorted(matched_paths)),
                matches_by_rule=dict(sorted(matches_by_rule.items())),
            ),
        )

    def _transform_text_values(
        self,
        value: JsonValue,
        rules: tuple[RedactionRule, ...],
        skipped_paths: set[str],
        matches_by_rule: Counter[str],
        matched_paths: set[str],
        *,
        dry_run: bool,
        path: str = "",
    ) -> JsonValue:
        if path in skipped_paths:
            return value
        if isinstance(value, str):
            matches: list[tuple[int, int, RedactionRule, str]] = []
            protected_spans = tuple(
                (match.start(), match.end()) for match in _PLACEHOLDER_PATTERN.finditer(value)
            )
            for rule in rules:
                if rule.literal is None:
                    raise AssertionError("validated text rules require literals")
                start = 0
                while (match_start := value.find(rule.literal, start)) >= 0:
                    match_end = match_start + len(rule.literal)
                    if any(
                        match_start < protected_end and match_end > protected_start
                        for protected_start, protected_end in protected_spans
                    ):
                        start = match_end
                        continue
                    matches.append((match_start, match_end, rule, rule.literal))
                    if len(matches) + sum(matches_by_rule.values()) > _MAXIMUM_MATCHES:
                        raise RedactionBoundaryError()
                    start = match_end
            matches.sort(key=lambda item: (item[0], item[1], item[2].name))
            if any(left[1] > right[0] for left, right in pairwise(matches)):
                raise RedactionBoundaryError()
            transformed = value
            for start, end, rule, matched_value in reversed(matches):
                replacement = self._replacement(rule, matched_value, dry_run=dry_run)
                if not isinstance(replacement, str):
                    raise RedactionBoundaryError()
                transformed = transformed[:start] + replacement + transformed[end:]
                matches_by_rule[rule.name] += 1
                matched_paths.add(path)
            return transformed
        if isinstance(value, list):
            return [
                self._transform_text_values(
                    item,
                    rules,
                    skipped_paths,
                    matches_by_rule,
                    matched_paths,
                    dry_run=dry_run,
                    path=f"{path}/{index}",
                )
                for index, item in enumerate(value)
            ]
        if isinstance(value, dict):
            return {
                key: self._transform_text_values(
                    item,
                    rules,
                    skipped_paths,
                    matches_by_rule,
                    matched_paths,
                    dry_run=dry_run,
                    path=f"{path}/{_escape_pointer_token(key)}",
                )
                for key, item in value.items()
            }
        return value

    def _replacement(
        self, rule: RedactionRule, selected_value: JsonValue, *, dry_run: bool
    ) -> JsonValue:
        if rule.action == "pseudonymize":
            return self.store.pseudonymize(rule.name, selected_value, dry_run=dry_run)
        if rule.action == "remove":
            return "" if rule.selector == _TEXT_SELECTOR else None
        return rule.replacement


def load_redaction_policy(path: str | Path) -> RedactionPolicy:
    policy_path = Path(path)
    try:
        encoded_policy = policy_path.read_bytes()
    except OSError:
        raise RuntimeError("redaction policy could not be read") from None
    if len(encoded_policy) > _MAXIMUM_POLICY_BYTES:
        raise ValueError("redaction policy exceeds the 1 MB limit")
    try:
        return RedactionPolicy.model_validate_json(encoded_policy)
    except ValueError:
        raise ValueError("redaction policy is invalid") from None


class _SemanticPipeline(
    SemanticDeconstructor, SemanticRenderer, SemanticEquivalenceVerifier, Protocol
):
    pass


class RedactedSemanticPipeline:
    def __init__(self, pipeline: _SemanticPipeline, engine: RedactionEngine) -> None:
        self._pipeline = pipeline
        self.engine = engine

    def protect_record(
        self, record: InteractionRecord | UserInputRecord
    ) -> InteractionRecord | UserInputRecord:
        protected_input = self.engine.transform(record.raw_input, location="input").value
        if not isinstance(protected_input, str):
            raise RedactionBoundaryError()
        if isinstance(record, InteractionRecord):
            protected_output = self.engine.transform(
                record.raw_observed_output, location="output"
            ).value
            return record.model_copy(
                update={"raw_input": protected_input, "raw_observed_output": protected_output}
            )
        return record.model_copy(update={"raw_input": protected_input})

    def dry_run(self, record: InteractionRecord | UserInputRecord) -> tuple[RedactionCoverage, ...]:
        coverage = [
            self.engine.transform(record.raw_input, location="input", dry_run=True).coverage
        ]
        if isinstance(record, InteractionRecord):
            coverage.append(
                self.engine.transform(
                    record.raw_observed_output, location="output", dry_run=True
                ).coverage
            )
        return tuple(coverage)

    async def deconstruct(
        self,
        record: InteractionRecord | UserInputRecord,
        reference_frame: SemanticFrame | None = None,
    ) -> SemanticFrame:
        try:
            protected_record = self.protect_record(record)
            frame = await self._pipeline.deconstruct(protected_record, reference_frame)
            return frame.model_copy(update={"metadata": self._metadata(frame.metadata)})
        except RedactionBoundaryError:
            raise
        except Exception:
            raise RedactionBoundaryError() from None

    async def render(
        self,
        raw_input: str,
        instruction: str,
        *,
        allow_temporary_value: bool = False,
    ) -> RenderedUserInput:
        try:
            protected_input = self.engine.transform(raw_input, location="input").value
            protected_instruction = self.engine.transform(instruction, location="context").value
            if not isinstance(protected_input, str) or not isinstance(protected_instruction, str):
                raise RedactionBoundaryError()
            expected_placeholders = Counter(_PLACEHOLDER_PATTERN.findall(protected_input))
            rendered = await self._pipeline.render(
                protected_input,
                protected_instruction,
                allow_temporary_value=allow_temporary_value,
            )
            if Counter(_PLACEHOLDER_PATTERN.findall(rendered.text)) != expected_placeholders:
                raise RedactionBoundaryError()
            return rendered.model_copy(update={"metadata": self._metadata(rendered.metadata)})
        except RedactionBoundaryError:
            raise
        except Exception:
            raise RedactionBoundaryError() from None

    async def verify(
        self, source_input: str, candidate_input: str
    ) -> SemanticEquivalenceAssessment:
        try:
            protected_source = self.engine.transform(source_input, location="input").value
            protected_candidate = self.engine.transform(candidate_input, location="input").value
            if not isinstance(protected_source, str) or not isinstance(protected_candidate, str):
                raise RedactionBoundaryError()
            assessment = await self._pipeline.verify(protected_source, protected_candidate)
            return assessment.model_copy(update={"metadata": self._metadata(assessment.metadata)})
        except RedactionBoundaryError:
            raise
        except Exception:
            raise RedactionBoundaryError() from None

    def wrap_sandbox(self, sandbox: SandboxExecutor) -> RehydratingSandboxConnection:
        return RehydratingSandboxConnection(sandbox, self.engine)

    def _metadata(self, metadata: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return {**metadata, "redaction_policy_sha256": self.engine.policy.digest}


class RehydratingSandboxConnection:
    def __init__(self, sandbox: SandboxExecutor, engine: RedactionEngine) -> None:
        self._sandbox = sandbox
        self._engine = engine

    @property
    def capabilities(self) -> SandboxCapabilities:
        return self._sandbox.capabilities

    @property
    def sandbox_id(self) -> str:
        return self._sandbox.sandbox_id

    @property
    def config_sha256(self) -> str:
        return self._sandbox.config_sha256

    def api_calls_for_case(self, case: EvaluationCase) -> int:
        return self._sandbox.api_calls_for_case(case)

    async def execute(self, case: EvaluationCase) -> ExecutionEvidence:
        try:
            rehydrated_case = case.model_copy(
                update={
                    "turns": tuple(
                        turn.model_copy(
                            update={"content": self._engine.store.rehydrate_text(turn.content)}
                        )
                        for turn in case.turns
                    )
                }
            )
        except RedactionBoundaryError:
            raise
        evidence = await self._sandbox.execute(rehydrated_case)
        protected_turns = tuple(
            turn.model_copy(
                update={
                    "response": self._engine.transform(turn.response, location="output").value,
                    "state_snapshot": (
                        self._engine.transform(turn.state_snapshot, location="output").value
                        if turn.state_snapshot is not None
                        else None
                    ),
                }
            )
            for turn in evidence.turns
        )
        return evidence.model_copy(
            update={
                "initial_state": (
                    evidence.initial_state.model_copy(
                        update={
                            "value": self._engine.transform(
                                evidence.initial_state.value, location="output"
                            ).value
                        }
                    )
                    if evidence.initial_state is not None
                    else None
                ),
                "turns": protected_turns,
                "final_response": protected_turns[-1].response if protected_turns else None,
                "final_state": (
                    evidence.final_state.model_copy(
                        update={"value": protected_turns[-1].state_snapshot}
                    )
                    if evidence.final_state is not None and protected_turns
                    else None
                ),
            }
        )


def _parse_json_pointer(pointer: str) -> tuple[str, ...]:
    if not re.fullmatch(r"(?:/(?:[^~/]|~[01])*)*", pointer):
        raise ValueError("selector must be $text or an RFC 6901 JSON pointer")
    if pointer == "":
        return ()
    return tuple(token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/"))


def _resolve_pointer(value: JsonValue, tokens: tuple[str, ...]) -> JsonValue:
    current: JsonValue = value
    for token in tokens:
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and (
            token == "0" or (token.isdecimal() and not token.startswith("0"))
        ):
            index = int(token)
            if index >= len(current):
                raise RedactionBoundaryError()
            current = current[index]
        else:
            raise RedactionBoundaryError()
    return current


def _set_or_remove_pointer(
    value: JsonValue,
    tokens: tuple[str, ...],
    replacement: JsonValue,
    action: RedactionAction,
) -> JsonValue:
    if not tokens:
        if action == "remove":
            raise RedactionBoundaryError()
        return replacement
    parent = _resolve_pointer(value, tokens[:-1])
    token = tokens[-1]
    if isinstance(parent, dict) and token in parent:
        if action == "remove":
            del parent[token]
        else:
            parent[token] = replacement
        return value
    if isinstance(parent, list) and (
        token == "0" or (token.isdecimal() and not token.startswith("0"))
    ):
        index = int(token)
        if index >= len(parent):
            raise RedactionBoundaryError()
        if action == "remove":
            del parent[index]
        else:
            parent[index] = replacement
        return value
    raise RedactionBoundaryError()


def _copy_json(value: JsonValue) -> JsonValue:
    return cast(JsonValue, json.loads(json.dumps(value, ensure_ascii=False)))


def _encode_value(value: JsonValue) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _find_placeholders(value: JsonValue) -> list[str]:
    if isinstance(value, str):
        return _PLACEHOLDER_PATTERN.findall(value)
    if isinstance(value, list):
        return [placeholder for item in value for placeholder in _find_placeholders(item)]
    if isinstance(value, dict):
        return [placeholder for item in value.values() for placeholder in _find_placeholders(item)]
    return []


def _escape_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _lock_descriptor(descriptor: int) -> None:
    if sys.platform == "win32":
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
    else:
        fcntl.flock(descriptor, fcntl.LOCK_EX)


def _unlock_descriptor(descriptor: int) -> None:
    if sys.platform == "win32":
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
