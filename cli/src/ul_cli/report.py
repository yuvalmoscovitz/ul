from __future__ import annotations

import json
import math
import os
import shlex
import stat
from pathlib import Path
from typing import Literal, Never, cast

import typer
from pydantic import ValidationError
from ul.augmentations.conversation import (
    CorrectionStressResult,
    RetryAfterSuccessfulCommitStressResult,
)
from ul.augmentations.environment_fault import TimeoutAfterCommitStressResult
from ul.dataset_invariants import DatasetInvariantRuleResult

from ul_cli.dataset_review import (
    is_reportable_dataset_evidence,
    report_dataset_evidence,
    summarize_dataset_evidence,
)
from ul_cli.report_contract import (
    FindingSummary,
    ReportEvidenceType,
    ReportInputError,
    ReportReviewStatus,
    UnifiedReport,
    build_report_summary,
)

type StatefulStressResult = (
    CorrectionStressResult | RetryAfterSuccessfulCommitStressResult | TimeoutAfterCommitStressResult
)

_WINDOWS = os.name == "nt"

_MAXIMUM_EVIDENCE_BYTES = 128_000_000
_MAXIMUM_JSON_DEPTH = 100
_EVIDENCE_LABELS: dict[ReportEvidenceType, str] = {
    "dataset_evaluation": "dataset evaluation",
    "correction_after_first_response": "correction after first response",
    "retry_after_successful_commit": "retry after successful commit",
    "timeout_after_commit": "timeout after commit",
}


def report_evidence(
    evidence: Path,
    *,
    reviews: Path | None = None,
    show_sensitive_values: bool = False,
    finding: str | None = None,
    json_output: bool = False,
) -> None:
    if json_output and (show_sensitive_values or finding is not None):
        raise typer.BadParameter(
            "--json cannot be combined with sensitive-value output",
            param_hint="--json",
        )
    try:
        report = load_unified_report(evidence, reviews=reviews)
    except ReportInputError as error:
        raise typer.BadParameter(str(error), param_hint="EVIDENCE") from None

    if show_sensitive_values or finding is not None:
        if report.evidence_type != "dataset_evaluation":
            raise typer.BadParameter(
                "sensitive-value output is available only for dataset evidence",
                param_hint="EVIDENCE",
            )
        report_dataset_evidence(
            evidence=evidence,
            reviews=reviews,
            show_sensitive_values=show_sensitive_values,
            sensitive_finding_id=finding,
        )
    elif json_output:
        typer.echo(
            json.dumps(
                report.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        _print_human_report(report, evidence)

    if report.exit_code:
        raise typer.Exit(code=report.exit_code)


def load_unified_report(evidence: Path, *, reviews: Path | None = None) -> UnifiedReport:
    if is_reportable_dataset_evidence(evidence):
        return summarize_dataset_evidence(evidence, reviews)

    result = _load_stateful_stress_result(evidence)
    if reviews is not None:
        raise ReportInputError("--reviews is available only for dataset evidence")
    return _summarize_stateful_stress_result(result)


def _summarize_stateful_stress_result(result: StatefulStressResult) -> UnifiedReport:
    violating_rules: tuple[DatasetInvariantRuleResult, ...]
    if isinstance(result, CorrectionStressResult):
        evidence_type: ReportEvidenceType = "correction_after_first_response"
        violating_rules = (
            tuple(rule for rule in result.corrected_invariant_rules if rule.status == "violated")
            if result.status == "failed"
            else ()
        )
    elif isinstance(result, RetryAfterSuccessfulCommitStressResult):
        evidence_type = "retry_after_successful_commit"
        violating_rules = (
            tuple(
                rule
                for rule in result.retried_invariant_rules
                if all(trial.status == "violated" for trial in rule.trials)
            )
            if result.status == "failed"
            else ()
        )
    else:
        evidence_type = "timeout_after_commit"
        violating_rules = (
            tuple(
                rule
                for rule in result.invariant_rules
                if all(trial.status == "violated" for trial in rule.trials)
            )
            if result.status == "failed"
            else ()
        )

    conclusive_repetitions = sum(trial.inconclusive_reason is None for trial in result.trials)
    findings = tuple(
        FindingSummary(
            kind="customer_invariant_violation",
            category="customer_invariant_violation",
            operator_id=result.case.operator_id,
            operator_version=result.case.operator_version,
            rule_id=rule.rule_id,
            rule_version=rule.rule_version,
            declared_severity=rule.severity,
            requested_repetitions=result.requested_repetitions,
            conclusive_repetitions=conclusive_repetitions,
            inconclusive_repetitions=result.requested_repetitions - conclusive_repetitions,
            violated_repetitions=sum(trial.status == "violated" for trial in rule.trials),
            next_action="inspect_stateful_evidence",
            summary="The agent violated a customer-defined rule.",
        )
        for rule in violating_rules
    )
    review_status = cast(
        ReportReviewStatus,
        {"passed": "resolved", "failed": "action_required", "inconclusive": "inconclusive"}[
            result.status
        ],
    )
    exit_code = cast(
        Literal[0, 1, 2],
        {"resolved": 0, "action_required": 1, "inconclusive": 2}[review_status],
    )
    return UnifiedReport(
        evidence_type=evidence_type,
        evidence_schema_versions=(result.schema_version,),
        evidence_scope="response_and_state",
        review_status=review_status,
        exit_code=exit_code,
        summary=build_report_summary(findings),
        findings=findings,
    )


def _load_stateful_stress_result(path: Path) -> StatefulStressResult:
    try:
        raw = _read_bounded_regular_file(path)
        value: object = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonstandard_json_constant,
            parse_float=_parse_finite_float,
        )
        _reject_deep_json(value)
        if not isinstance(value, dict):
            raise ValueError("stateful evidence must be a JSON object")
        evidence_object = cast(dict[str, object], value)
        case = evidence_object.get("case")
        if not isinstance(case, dict):
            raise ValueError("stateful evidence must contain a case")
        case_object = cast(dict[str, object], case)
        operator_id = case_object.get("operator_id")
        if operator_id == "conversation.correction_after_first_response":
            return CorrectionStressResult.model_validate_json(raw)
        if operator_id == "conversation.retry_after_successful_commit":
            return RetryAfterSuccessfulCommitStressResult.model_validate_json(raw)
        if operator_id == "environment.tool.timeout_after_commit":
            return TimeoutAfterCommitStressResult.model_validate_json(raw)
    except OSError as error:
        raise ReportInputError(
            f"cannot safely read evidence ({error.__class__.__name__})"
        ) from None
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValidationError, ValueError):
        pass
    raise ReportInputError("unsupported evidence; expected dataset, correction, retry, or timeout")


def _print_human_report(report: UnifiedReport, evidence: Path) -> None:
    typer.echo("UL run report")
    typer.echo(f"Evidence type: {_EVIDENCE_LABELS[report.evidence_type]}")
    if report.evaluation_mode is not None:
        typer.echo(
            f"Evaluation mode: {report.evaluation_mode} "
            "(historical output is not an expected answer; correctness not assessed)"
        )
    typer.echo(
        "Evidence scope: "
        + (
            "response only"
            if report.evidence_scope == "response_only"
            else "response and committed state"
        )
    )
    if report.capability_limitations:
        typer.echo("Not verified: committed state, cleanup, or multi-turn conversations.")
    typer.echo(f"Review status: {report.review_status} (exit {report.exit_code})")
    typer.echo(
        f"Findings: {report.summary.finding_count} total; "
        f"{report.summary.actionable_finding_count} actionable"
    )
    review_counts = report.summary.review_status_counts
    if any(review_counts.model_dump().values()):
        typer.echo(
            "Reviews: "
            f"needs_review={review_counts.needs_review}, confirmed={review_counts.confirmed}, "
            f"expected={review_counts.expected}, unsupported={review_counts.unsupported}, "
            f"inconclusive={review_counts.inconclusive}"
        )
    if report.patterns:
        typer.echo("")
        typer.echo(f"Reviewable finding patterns: {len(report.patterns)}")
        typer.echo("Patterns group similar evidence; they do not claim a root cause.")
        for index, pattern in enumerate(report.patterns, start=1):
            typer.echo("")
            typer.echo(f"Pattern {index}: {pattern.summary}")
            typer.echo(f"  Pattern ID: {pattern.pattern_id}")
            typer.echo(f"  Priority: {pattern.severity}")
            if pattern.rule_id is not None:
                typer.echo(
                    f"  Customer rule: {pattern.rule_id}@{pattern.rule_version} "
                    "(configured by your team)"
                )
                typer.echo("  Why grouped: same customer-defined rule.")
            else:
                typer.echo("  Why grouped: same finding category and private action shape.")
            typer.echo(
                f"  Affected: {pattern.finding_count} finding(s) across "
                f"{pattern.source_case_count} test question(s)"
            )
            typer.echo("  Observed under:")
            for operator in pattern.operators:
                label = operator.summary or operator.operator_id
                typer.echo(f"    - {label} ({operator.operator_id}@{operator.operator_version})")
            typer.echo(
                "  Review queue: "
                f"{pattern.needs_review_count} needs review; "
                f"{pattern.confirmed_count} confirmed"
            )
            typer.echo(f"  Finding IDs: {', '.join(pattern.finding_ids)}")
            typer.echo("  Next: use the per-finding review commands below.")
    grouped_finding_count = sum(pattern.finding_count for pattern in report.patterns)
    ungrouped_actionable_count = report.summary.actionable_finding_count - grouped_finding_count
    if ungrouped_actionable_count:
        typer.echo("")
        typer.echo(
            f"Additional actionable findings not in the reviewable pattern queue: "
            f"{ungrouped_actionable_count}. Inspect them below."
        )
    safe_evidence = "".join(
        character if character.isprintable() else f"\\u{ord(character):04x}"
        for character in str(evidence)
    )
    quoted_evidence = None if _WINDOWS else shlex.quote(safe_evidence)
    for index, finding in enumerate(report.findings, start=1):
        typer.echo("")
        typer.echo(f"Finding {finding.finding_id or index}")
        typer.echo(f"  Category: {finding.category}")
        typer.echo(f"  Summary: {finding.summary}")
        if finding.operator_id is not None:
            typer.echo(f"  Operator: {finding.operator_id}@{finding.operator_version}")
        if finding.rule_id is not None:
            typer.echo(
                f"  Customer rule: {finding.rule_id}@{finding.rule_version} "
                "(configured by your team)"
            )
            typer.echo(f"  Declared severity: {finding.declared_severity}")
        if finding.review_status is not None:
            typer.echo(f"  Review: {finding.review_status}; severity={finding.review_severity}")
        typer.echo(
            "  Repetitions: "
            f"{finding.conclusive_repetitions}/{finding.requested_repetitions} conclusive; "
            f"{finding.inconclusive_repetitions} inconclusive"
        )
        if finding.stability is not None:
            typer.echo(f"  Stability: {finding.stability}")
        if finding.violated_repetitions is not None:
            typer.echo(f"  Violated repetitions: {finding.violated_repetitions}")
        if finding.next_action == "review_dataset_finding":
            evidence_argument = quoted_evidence or "EVIDENCE"
            typer.echo(
                "  Next: "
                f"ul dataset review {evidence_argument} {finding.finding_id} "
                "--status STATUS --reviewer REVIEWER --reason REASON"
            )
        elif finding.next_action == "inspect_dataset_evidence":
            typer.echo(f"  Next: ul dataset report {quoted_evidence or 'EVIDENCE'}")
        else:
            typer.echo(
                "  Next: inspect the supplied evidence JSON; "
                "no dedicated stateful detail command is available."
            )
    typer.echo("")
    typer.echo("Sensitive inputs, outputs, state, and arbitrary evidence text are omitted.")
    if report.evidence_type == "dataset_evaluation":
        typer.echo(f"Drill-down: ul dataset report {quoted_evidence or 'EVIDENCE'}")
    else:
        typer.echo(
            "Drill-down: inspect the supplied evidence JSON; "
            "no dedicated stateful detail command is available."
        )


def _read_bounded_regular_file(path: Path) -> bytes:
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    requires_identity_check = no_follow_flag == 0
    if requires_identity_check and stat.S_ISLNK(os.lstat(path).st_mode):
        raise OSError("evidence is a symbolic link")
    binary_flag = os.O_BINARY if os.name == "nt" else 0
    descriptor = os.open(path, os.O_RDONLY | no_follow_flag | binary_flag)
    try:
        descriptor_status = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_status.st_mode):
            raise OSError("evidence is not a regular file")
        if requires_identity_check:
            path_status = os.lstat(path)
            if stat.S_ISLNK(path_status.st_mode) or not os.path.samestat(
                descriptor_status, path_status
            ):
                raise OSError("evidence changed while it was opened")
        if descriptor_status.st_size > _MAXIMUM_EVIDENCE_BYTES:
            raise ValueError("evidence exceeds the 128 MB limit")
        remaining = descriptor_status.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _reject_duplicate_object_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _reject_nonstandard_json_constant(value: str) -> Never:
    raise ValueError(f"non-standard JSON constant: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON number must be finite")
    return parsed


def _reject_deep_json(value: object, depth: int = 0) -> None:
    if depth > _MAXIMUM_JSON_DEPTH:
        raise ValueError("evidence exceeds the nesting limit")
    if isinstance(value, dict):
        for child in cast(dict[object, object], value).values():
            _reject_deep_json(child, depth + 1)
    elif isinstance(value, list):
        for child in cast(list[object], value):
            _reject_deep_json(child, depth + 1)
