from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import unicodedata
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import uuid4

import typer
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator
from rich.console import Console
from ul.dataset_invariants import DatasetInvariantEvaluation

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

_MAXIMUM_EVIDENCE_BYTES = 128_000_000
_MAXIMUM_EVIDENCE_RECORDS = 100
_MAXIMUM_REVIEWS_BYTES = 10_000_000
_MAXIMUM_REVIEW_RECORDS = 10_000
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_FINDING_ID_PATTERN = r"^ulf_v1_[0-9a-f]{64}$"
_REVIEW_ID_PATTERN = r"^ulr_[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"

ReviewStatus = Literal["confirmed", "expected", "unsupported", "inconclusive"]
ReviewSeverity = Literal["unrated", "low", "medium", "high", "critical"]

console = Console()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _EvidenceReference(_StrictModel):
    source: Literal["input", "output"]
    json_pointer: str
    text_quote: str | None


class _Effect(_StrictModel):
    id: str
    evidence: list[_EvidenceReference]
    confidence: float = Field(ge=0, le=1)
    status: str
    request_unit_ids: list[str]
    position: int = Field(ge=0)
    kind: str
    predicate: str
    fields: dict[str, JsonValue]
    propositions: list[str]


class _Finding(_StrictModel):
    finding_id: str = Field(pattern=_FINDING_ID_PATTERN)
    category: str
    grounded_field_names: list[str]
    severity: Literal["unrated"]
    review_status: Literal["needs_review"]
    summary: str
    reference_effects: list[_Effect]
    observed_effects: list[_Effect]


class _OutcomeGroup(_StrictModel):
    repetitions: list[int]
    count: int = Field(ge=1)
    representative_effects: list[_Effect]


class _Trial(_StrictModel):
    repetition: int = Field(ge=1)
    status: Literal["observed", "inconclusive"]
    inconclusive_reasons: list[str]


class _Observations(_StrictModel):
    requested_repetitions: int = Field(ge=1)
    stability: Literal["stable", "unstable", "inconclusive"]
    observed_repetitions: int = Field(ge=0)
    inconclusive_repetitions: int = Field(ge=0)
    outcome_group_count: int = Field(ge=0)
    outcome_groups: list[_OutcomeGroup]
    trials: list[_Trial]


class _ExecutionPlan(_StrictModel):
    repetitions: int = Field(ge=1)
    max_target_calls: int = Field(ge=1)
    dataset_planned_target_calls: int = Field(ge=1)


class _Baseline(_StrictModel):
    status: str
    observations: _Observations
    inconclusive_reasons: list[str]


class _Case(_StrictModel):
    operator_id: str
    operator_version: str
    augmented_input: str
    status: str
    variation_accepted: bool
    variation_rejection_reasons: list[str]
    observations: _Observations | None
    findings: list[_Finding]
    inconclusive_reasons: list[str]


class _EvidenceRecord(_StrictModel):
    schema_version: Literal["1.3.0", "1.4.0"]
    interaction_id: str
    original_input: str
    execution_plan: _ExecutionPlan
    limitations: str
    current_baseline: _Baseline
    cases: list[_Case]
    invariant_evaluation: DatasetInvariantEvaluation | None = None
    technical_details: JsonValue

    @model_validator(mode="after")
    def validate_invariant_evaluation(self) -> Self:
        if self.schema_version == "1.3.0" and "invariant_evaluation" in self.model_fields_set:
            raise ValueError("schema 1.3.0 does not include invariant evaluation")
        if (
            self.invariant_evaluation is not None
            and self.invariant_evaluation.interaction_id != self.interaction_id
        ):
            raise ValueError("invariant evaluation must match the evidence interaction")
        return self


class ReviewRecord(_StrictModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    review_id: str = Field(pattern=_REVIEW_ID_PATTERN)
    evidence_record_sha256: str = Field(pattern=_SHA256_PATTERN)
    finding_id: str = Field(pattern=_FINDING_ID_PATTERN)
    status: ReviewStatus
    severity: ReviewSeverity = "unrated"
    reviewer: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=4000)
    reviewed_at: datetime
    supersedes_review_id: str | None = Field(default=None, pattern=_REVIEW_ID_PATTERN)

    @model_validator(mode="after")
    def validate_review(self) -> Self:
        if not self.reviewer.strip() or not self.reason.strip():
            raise ValueError("reviewer and reason must contain non-whitespace text")
        if self.status != "confirmed" and self.severity != "unrated":
            raise ValueError("only confirmed findings can have a rated severity")
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() != UTC.utcoffset(None):
            raise ValueError("reviewed_at must use UTC")
        return self


class _ReviewInputError(ValueError):
    pass


class _LoadedEvidenceRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    evidence: _EvidenceRecord
    sha256: str


class ConfirmedDatasetFinding(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    evidence_record: _LoadedEvidenceRecord
    case: _Case
    review: ReviewRecord


def load_confirmed_dataset_finding(
    evidence_path: Path,
    reviews_path: Path,
    finding_id: str,
) -> ConfirmedDatasetFinding:
    evidence_records = _load_evidence(evidence_path)
    findings = _index_findings(evidence_records)
    selected = findings.get(finding_id)
    if selected is None:
        raise ValueError("finding ID was not found in the evidence")
    review_records = _load_reviews(reviews_path)
    _validate_review_history(review_records, findings)
    active_review = _active_reviews(review_records).get(finding_id)
    if active_review is None or active_review.status != "confirmed":
        raise ValueError("finding must have an active confirmed review")
    return ConfirmedDatasetFinding(
        evidence_record=selected[0],
        case=selected[1],
        review=active_review,
    )


def report_dataset_evidence(
    evidence: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, readable=True, help="Evaluation evidence JSONL."
        ),
    ],
    reviews: Annotated[
        Path | None,
        typer.Option(help="Review JSONL; defaults to EVIDENCE with .reviews.jsonl suffix."),
    ] = None,
) -> None:
    """Show findings and their human review state without model or network calls."""
    reviews_path = reviews or _default_reviews_path(evidence)
    try:
        evidence_records = _load_evidence(evidence)
        review_records = _load_reviews(reviews_path)
        findings = _index_findings(evidence_records)
        _validate_review_history(review_records, findings)
    except _ReviewInputError as error:
        raise typer.BadParameter(str(error)) from None

    active_reviews = _active_reviews(review_records)
    status_counts = {
        status: 0
        for status in ("needs_review", "confirmed", "expected", "unsupported", "inconclusive")
    }
    for _, _, finding in findings.values():
        active_review = active_reviews.get(finding.finding_id)
        status_counts[active_review.status if active_review else "needs_review"] += 1

    _print_plain(f"Dataset finding report: {len(findings)} finding(s)")
    _print_plain(
        "Reviews: " + ", ".join(f"{status}={count}" for status, count in status_counts.items())
    )
    for loaded_record, case, finding in findings.values():
        matching_reviews = [
            review for review in review_records if review.finding_id == finding.finding_id
        ]
        latest_review = active_reviews.get(finding.finding_id)
        _print_plain("")
        _print_plain(f"Finding {finding.finding_id}")
        _print_plain(f"Machine status: {case.status}")
        _print_plain(f"Category: {finding.category}")
        _print_plain(f"Summary: {finding.summary}")
        _print_plain(f"Original: {loaded_record.evidence.original_input}")
        _print_plain(f"Variation: {case.augmented_input}")
        _print_plain(f"Operator: {case.operator_id} ({case.operator_version})")
        _print_plain(
            "Original trials: "
            + _observations_summary(loaded_record.evidence.current_baseline.observations)
        )
        _print_plain("Variation trials: " + _observations_summary(case.observations))
        _print_plain("Reference effects: " + _effects_summary(finding.reference_effects))
        _print_plain("Observed effects: " + _effects_summary(finding.observed_effects))
        if latest_review is None:
            _print_plain(f"Latest review: needs_review (history: {len(matching_reviews)})")
        else:
            _print_plain(
                f"Latest review: {latest_review.status}, severity={latest_review.severity}, "
                f"reviewer={latest_review.reviewer}, id={latest_review.review_id} "
                f"(history: {len(matching_reviews)})"
            )
            _print_plain(f"Review reason: {latest_review.reason}")
        _print_plain(f"Evidence record SHA-256: {loaded_record.sha256}")

    invariant_evaluations = [
        loaded_record.evidence.invariant_evaluation
        for loaded_record in evidence_records
        if loaded_record.evidence.invariant_evaluation is not None
    ]
    if invariant_evaluations:
        _print_plain("")
        _print_plain("Customer invariant evaluation")
        for invariant_evaluation in invariant_evaluations:
            _print_invariant_evaluation(invariant_evaluation)

    _print_plain("")
    _print_plain(f"Complete technical evidence: {evidence}")
    _print_plain(f"Review history: {reviews_path}")
    _print_plain(
        "Review meanings: confirmed=problem in context; expected=supported acceptable difference; "
        "unsupported=machine claim not supported; inconclusive=insufficient context."
    )
    _print_plain(
        "Limitation: UL reports observed differences. A human review is a contextual judgment, "
        "not proof of correctness, causation, or production frequency."
    )


def review_dataset_finding(
    evidence: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, readable=True, help="Evaluation evidence JSONL."
        ),
    ],
    finding_id: Annotated[str, typer.Argument(help="Finding ID shown by 'ul dataset report'.")],
    status: Annotated[ReviewStatus, typer.Option(help="Human review judgment.")],
    reviewer: Annotated[str, typer.Option(help="Person or team making the judgment.")],
    reason: Annotated[str, typer.Option(help="Why this judgment applies in context.")],
    severity: Annotated[
        ReviewSeverity,
        typer.Option(help="Consequence severity; only confirmed findings may be rated."),
    ] = "unrated",
    supersedes: Annotated[
        str | None,
        typer.Option(help="Current active review ID replaced by this judgment."),
    ] = None,
    reviews: Annotated[
        Path | None,
        typer.Option(help="Review JSONL; defaults to EVIDENCE with .reviews.jsonl suffix."),
    ] = None,
) -> None:
    """Append a human judgment while preserving the complete review history."""
    reviews_path = reviews or _default_reviews_path(evidence)
    try:
        evidence_records = _load_evidence(evidence)
        findings = _index_findings(evidence_records)
        selected = findings.get(finding_id)
        if selected is None:
            raise _ReviewInputError("finding ID was not found in the evidence")
        loaded_record, _, _ = selected
        new_review = ReviewRecord(
            review_id=f"ulr_{uuid4()}",
            evidence_record_sha256=loaded_record.sha256,
            finding_id=finding_id,
            status=status,
            severity=severity,
            reviewer=reviewer,
            reason=reason,
            reviewed_at=datetime.now(UTC),
            supersedes_review_id=supersedes,
        )
        with _locked_reviews_file(reviews_path) as descriptor:
            review_records = _read_reviews_descriptor(descriptor)
            _validate_review_history(review_records, findings)
            active_review = _active_reviews(review_records).get(finding_id)
            if active_review is None and supersedes is not None:
                raise _ReviewInputError(
                    "--supersedes is only valid when replacing an active review"
                )
            if active_review is not None and supersedes != active_review.review_id:
                raise _ReviewInputError(
                    "finding already has an active review; pass --supersedes "
                    f"{active_review.review_id}"
                )
            encoded_review = (
                json.dumps(
                    new_review.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")
                )
                + "\n"
            ).encode()
            if os.fstat(descriptor).st_size + len(encoded_review) > _MAXIMUM_REVIEWS_BYTES:
                raise _ReviewInputError("review file exceeds the 10 MB limit")
            remaining = memoryview(encoded_review)
            while remaining:
                written = os.write(descriptor, remaining)
                if written == 0:
                    raise OSError("could not append review")
                remaining = remaining[written:]
            os.fsync(descriptor)
    except (ValidationError, _ReviewInputError) as error:
        message = "review fields are invalid" if isinstance(error, ValidationError) else str(error)
        raise typer.BadParameter(message) from None
    except OSError as error:
        raise typer.BadParameter(
            f"cannot safely update review file ({error.__class__.__name__})"
        ) from None

    _print_plain(f"Recorded review {new_review.review_id}: {new_review.status}")
    _print_plain(f"Review history: {reviews_path}")


def _default_reviews_path(evidence: Path) -> Path:
    return evidence.with_suffix(".reviews.jsonl")


def _load_evidence(path: Path) -> list[_LoadedEvidenceRecord]:
    try:
        raw = _read_bounded_regular_file(path, _MAXIMUM_EVIDENCE_BYTES)
    except OSError as error:
        raise _ReviewInputError(
            f"cannot safely read evidence ({error.__class__.__name__})"
        ) from None
    raw_lines = raw.splitlines()
    if not raw_lines or len(raw_lines) > _MAXIMUM_EVIDENCE_RECORDS:
        raise _ReviewInputError("evidence must contain 1 to 100 JSONL records")
    if any(not raw_line.strip() for raw_line in raw_lines):
        raise _ReviewInputError("evidence contains an empty JSONL record")
    records: list[_LoadedEvidenceRecord] = []
    try:
        for raw_line in raw_lines:
            records.append(
                _LoadedEvidenceRecord(
                    evidence=_EvidenceRecord.model_validate_json(raw_line),
                    sha256=hashlib.sha256(raw_line).hexdigest(),
                )
            )
    except (ValidationError, ValueError):
        raise _ReviewInputError("evidence is not valid UL schema 1.3.0 or 1.4.0 JSONL") from None
    return records


def _load_reviews(path: Path) -> list[ReviewRecord]:
    try:
        descriptor = _open_regular_file(path, os.O_RDONLY)
        try:
            _lock_file(descriptor, exclusive=False)
            try:
                return _read_reviews_descriptor(descriptor)
            finally:
                _unlock_file(descriptor)
        finally:
            os.close(descriptor)
    except FileNotFoundError:
        return []
    except OSError as error:
        raise _ReviewInputError(
            f"cannot safely read review file ({error.__class__.__name__})"
        ) from None


def _read_reviews_descriptor(descriptor: int) -> list[ReviewRecord]:
    size = os.fstat(descriptor).st_size
    if size > _MAXIMUM_REVIEWS_BYTES:
        raise _ReviewInputError("review file exceeds the 10 MB limit")
    os.lseek(descriptor, 0, os.SEEK_SET)
    raw = b""
    while len(raw) < size:
        chunk = os.read(descriptor, min(65_536, size - len(raw)))
        if not chunk:
            break
        raw += chunk
    raw_lines = raw.splitlines()
    if len(raw_lines) > _MAXIMUM_REVIEW_RECORDS:
        raise _ReviewInputError("review file exceeds the 10,000 record limit")
    if any(not line.strip() for line in raw_lines):
        raise _ReviewInputError("review file contains an empty JSONL record")
    try:
        return [ReviewRecord.model_validate_json(line) for line in raw_lines]
    except (ValidationError, ValueError):
        raise _ReviewInputError("review file is not valid review JSONL") from None


def _read_bounded_regular_file(path: Path, maximum_bytes: int) -> bytes:
    descriptor = _open_regular_file(path, os.O_RDONLY)
    try:
        size = os.fstat(descriptor).st_size
        if size > maximum_bytes:
            raise _ReviewInputError("evidence exceeds the 128 MB limit")
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _open_regular_file(path: Path, flags: int) -> int:
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    requires_identity_check = no_follow_flag == 0
    if requires_identity_check:
        try:
            if stat.S_ISLNK(os.lstat(path).st_mode):
                raise OSError("path is a symbolic link")
        except FileNotFoundError:
            pass
    binary_flag = os.O_BINARY if sys.platform == "win32" else 0
    open_flags = flags | no_follow_flag | binary_flag
    descriptor = os.open(path, open_flags, 0o600)
    try:
        descriptor_status = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_status.st_mode):
            raise OSError("path is not a regular file")
        if requires_identity_check:
            path_status = os.lstat(path)
            if stat.S_ISLNK(path_status.st_mode) or not os.path.samestat(
                descriptor_status, path_status
            ):
                raise OSError("path changed while it was opened")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def _locked_reviews_file(path: Path) -> Generator[int]:
    descriptor = _open_regular_file(path, os.O_RDWR | os.O_APPEND | os.O_CREAT)
    locked = False
    try:
        _set_private_file_permissions(descriptor)
        _lock_file(descriptor, exclusive=True)
        locked = True
        yield descriptor
    finally:
        try:
            if locked:
                _unlock_file(descriptor)
        finally:
            os.close(descriptor)


def _lock_file(descriptor: int, *, exclusive: bool) -> None:
    if sys.platform == "win32":
        os.lseek(descriptor, 0, os.SEEK_SET)
        mode = msvcrt.LK_LOCK if exclusive else msvcrt.LK_RLCK
        msvcrt.locking(descriptor, mode, 1)
        return
    mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    fcntl.flock(descriptor, mode)


def _unlock_file(descriptor: int) -> None:
    if sys.platform == "win32":
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return
    fcntl.flock(descriptor, fcntl.LOCK_UN)


def _set_private_file_permissions(descriptor: int) -> None:
    if sys.platform != "win32":
        os.fchmod(descriptor, 0o600)


def _index_findings(
    records: list[_LoadedEvidenceRecord],
) -> dict[str, tuple[_LoadedEvidenceRecord, _Case, _Finding]]:
    findings: dict[str, tuple[_LoadedEvidenceRecord, _Case, _Finding]] = {}
    for loaded_record in records:
        for case in loaded_record.evidence.cases:
            for finding in case.findings:
                if finding.finding_id in findings:
                    raise _ReviewInputError("evidence contains a duplicate finding ID")
                findings[finding.finding_id] = (loaded_record, case, finding)
    return findings


def _validate_review_history(
    reviews: list[ReviewRecord],
    findings: dict[str, tuple[_LoadedEvidenceRecord, _Case, _Finding]],
) -> None:
    reviews_by_id: dict[str, ReviewRecord] = {}
    superseded_ids: set[str] = set()
    active_by_finding: dict[str, ReviewRecord] = {}
    for review in reviews:
        if review.review_id in reviews_by_id:
            raise _ReviewInputError("review file contains a duplicate review ID")
        selected = findings.get(review.finding_id)
        if selected is None:
            raise _ReviewInputError("review references a finding outside this evidence")
        if review.evidence_record_sha256 != selected[0].sha256:
            raise _ReviewInputError("review evidence digest does not match the evidence record")
        current_active = active_by_finding.get(review.finding_id)
        if current_active is None:
            if review.supersedes_review_id is not None:
                raise _ReviewInputError("review supersedes a review that is not active")
        elif review.supersedes_review_id != current_active.review_id:
            raise _ReviewInputError("review history has multiple active judgments")
        if review.supersedes_review_id is not None:
            previous = reviews_by_id.get(review.supersedes_review_id)
            if previous is None or previous.finding_id != review.finding_id:
                raise _ReviewInputError("review supersession target is invalid")
            if previous.review_id in superseded_ids:
                raise _ReviewInputError("review supersession target is already superseded")
            superseded_ids.add(previous.review_id)
        reviews_by_id[review.review_id] = review
        active_by_finding[review.finding_id] = review


def _active_reviews(reviews: list[ReviewRecord]) -> dict[str, ReviewRecord]:
    return {review.finding_id: review for review in reviews}


def _observations_summary(observations: _Observations | None) -> str:
    if observations is None:
        return "not executed"
    groups = (
        ", ".join(
            "+".join(str(repetition) for repetition in group.repetitions)
            for group in observations.outcome_groups
        )
        or "none"
    )
    return (
        f"{observations.stability}; {observations.observed_repetitions}/"
        f"{observations.requested_repetitions} observed; groups={groups}"
    )


def _effects_summary(effects: list[_Effect]) -> str:
    if not effects:
        return "none"
    return "; ".join(
        json.dumps(
            {
                "kind": effect.kind,
                "predicate": effect.predicate,
                "fields": effect.fields,
                "propositions": effect.propositions,
                "status": effect.status,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for effect in effects
    )


def _print_invariant_evaluation(evaluation: DatasetInvariantEvaluation) -> None:
    _print_plain(
        "Selected values remain in the private evidence file; terminal output shows pointers only."
    )
    _print_plain(f"Interaction: {evaluation.interaction_id}")
    _print_plain(f"Declared observation authority: {evaluation.observation_authority}")
    for arm in (evaluation.baseline, *evaluation.variations):
        arm_name = "original" if arm.arm == "baseline" else f"variation ({arm.operator_id})"
        for rule in arm.rules:
            status_counts = {
                status: sum(trial.status == status for trial in rule.trials)
                for status in ("satisfied", "violated", "not_evaluable")
            }
            _print_plain(
                f"Rule {rule.rule_id} ({rule.rule_version}); severity={rule.severity}; "
                f"arm={arm_name}; status={rule.status}; reason={rule.reason_code}; trials="
                + ", ".join(f"{status}={count}" for status, count in status_counts.items())
            )
            _print_plain(f"Description: {rule.description}")
            if rule.status == "violated":
                _print_plain(
                    f"Customer rule violated against declared {evaluation.observation_authority}."
                )
            for trial in rule.trials:
                _print_plain(
                    f"Trial {trial.repetition}: {trial.status}; "
                    f"left={trial.left_pointer}; right={trial.right_pointer}; "
                    f"reason={trial.reason_code}"
                )


def _print_plain(message: str) -> None:
    safe_message = "".join(
        character
        if (ord(character) >= 32 and not 0x7F <= ord(character) <= 0x9F)
        and unicodedata.category(character) not in {"Cf", "Cs"}
        else f"\\u{ord(character):04x}"
        for character in message
    )
    console.print(safe_message, markup=False, highlight=False)
