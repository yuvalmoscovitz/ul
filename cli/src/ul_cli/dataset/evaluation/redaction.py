from __future__ import annotations

import os
from pathlib import Path

from pydantic import JsonValue, SecretStr
from ul import InteractionRecord, LocalPseudonymStore, RedactionEngine, load_redaction_policy

from ul_cli.dataset_review import DatasetEvidenceRedactionCoverage

_REDACTION_KEY_ENVIRONMENT_VARIABLE = "UL_DATASET_REDACTION_KEY"


def load_redaction_engine(
    policy_path: Path | None,
    state_path: Path | None,
    *,
    state_required: bool,
) -> RedactionEngine | None:
    if policy_path is None:
        if state_path is not None:
            raise ValueError("--redaction-state requires --redaction-policy")
        return None
    policy = load_redaction_policy(policy_path)
    if not state_required:
        return RedactionEngine(
            policy,
            LocalPseudonymStore(Path("unused-redaction-state"), SecretStr("0" * 32)),
        )
    if state_path is None:
        raise ValueError("--redaction-policy requires --redaction-state for execution and resume")
    key = os.environ.get(_REDACTION_KEY_ENVIRONMENT_VARIABLE, "")
    if len(key.encode()) < 32:
        raise ValueError(f"set {_REDACTION_KEY_ENVIRONMENT_VARIABLE} to at least 32 UTF-8 bytes")
    return RedactionEngine(policy, LocalPseudonymStore(state_path, SecretStr(key)))


def calculate_redaction_coverage(
    records: tuple[InteractionRecord, ...],
    engine: RedactionEngine | None,
) -> tuple[DatasetEvidenceRedactionCoverage, ...]:
    if engine is None:
        return ()
    coverage_by_location: list[DatasetEvidenceRedactionCoverage] = []
    for location in ("input", "output"):
        matched_values = 0
        matched_paths: set[str] = set()
        matches_by_rule: dict[str, int] = {}
        for record in records:
            value: JsonValue = (
                record.raw_input if location == "input" else record.raw_observed_output
            )
            coverage = engine.transform(value, location=location, dry_run=True).coverage
            matched_values += coverage.matched_values
            matched_paths.update(coverage.matched_paths)
            for rule_name, count in coverage.matches_by_rule.items():
                matches_by_rule[rule_name] = matches_by_rule.get(rule_name, 0) + count
        coverage_by_location.append(
            DatasetEvidenceRedactionCoverage(
                location=location,
                matched_values=matched_values,
                matched_paths=tuple(sorted(matched_paths)),
                matches_by_rule=dict(sorted(matches_by_rule.items())),
            )
        )
    return tuple(coverage_by_location)


def protect_interaction_records(
    records: tuple[InteractionRecord, ...], engine: RedactionEngine
) -> tuple[InteractionRecord, ...]:
    protected_records: list[InteractionRecord] = []
    for record in records:
        protected_input = engine.transform(record.raw_input, location="input").value
        if not isinstance(protected_input, str):
            raise ValueError("redaction policy did not preserve executable input as text")
        protected_records.append(
            record.model_copy(
                update={
                    "raw_input": protected_input,
                    "raw_observed_output": engine.transform(
                        record.raw_observed_output, location="output"
                    ).value,
                }
            )
        )
    return tuple(protected_records)
