from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from ul import (
    DatasetSemanticPreparationError,
    DatasetSourceOutcomeProjectionError,
    DatasetTrialUnit,
)
from ul_cli.dataset.evaluation.command import (
    _attempted_target_calls,
    _reconcile_source_preparation_failures,
)
from ul_cli.dataset.evidence.context import build_dataset_evidence_run_context
from ul_cli.dataset.evidence.customer import build_source_preparation_failure_evidence
from ul_cli.dataset_review import DatasetEvidenceRunContext, DatasetResumeEvidence
from ul_cli.dataset_trial_journal import (
    create_dataset_run_manifest,
    create_dataset_trial_journal,
    open_dataset_trial_journal,
    persist_dataset_run_manifest,
    read_dataset_run_manifest,
)

from ._factories import _evaluation_result, _run_config, _run_context, _settings


def _manifest():
    source = _evaluation_result("interaction-1").source
    return create_dataset_run_manifest(
        run_context=_run_context((source,)),
        selected_records=(source,),
        selected_operator_ids=("input.surface.rephrase",),
        run_config=_run_config(max_environment_api_calls=10),
        save_augmentations=True,
    )


def test_manifest_is_content_addressed_immutable_and_atomic(tmp_path: Path) -> None:
    path = tmp_path / "evidence.jsonl.manifest.json"
    manifest = _manifest()

    persist_dataset_run_manifest(path, manifest)
    persist_dataset_run_manifest(path, manifest)

    assert read_dataset_run_manifest(path) == manifest
    changed = manifest.model_copy(
        update={
            "effective_command": manifest.effective_command.model_copy(
                update={
                    "run_config": manifest.effective_command.run_config.model_copy(
                        update={
                            "target": manifest.effective_command.run_config.target.model_copy(
                                update={"max_environment_api_calls": 11}
                            )
                        }
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="does not match"):
        persist_dataset_run_manifest(path, changed)


def test_running_trial_fails_closed_as_quarantined_on_resume(tmp_path: Path) -> None:
    path = tmp_path / "trials.jsonl"
    manifest = _manifest()
    unit = manifest.work_plan[0]
    journal = create_dataset_trial_journal(path, manifest)
    journal.start(unit)
    journal.close()

    resumed = open_dataset_trial_journal(path, manifest)

    assert resumed.snapshot.quarantined_unit_ids == {unit.id}
    recovered = resumed.snapshot.recovered_trials[unit.id]
    assert recovered.lifecycle_failure is not None
    assert recovered.lifecycle_failure.environment_state_may_remain is True
    assert "was not retried" in recovered.inconclusive_reasons[0]
    resumed.close()


def test_terminal_trial_is_recovered_and_cannot_be_recorded_twice(tmp_path: Path) -> None:
    path = tmp_path / "trials.jsonl"
    manifest = _manifest()
    unit = DatasetTrialUnit(
        interaction_id="interaction-1",
        operator_id="current_baseline",
        arm="original",
        repetition=1,
    )
    trial = _evaluation_result("interaction-1").baseline.trial_set.trials[0]
    journal = create_dataset_trial_journal(path, manifest)
    journal.start(unit)
    journal.finish(unit, trial)

    with pytest.raises(ValueError, match="terminal outcome"):
        journal.finish(unit, trial)
    journal.close()

    resumed = open_dataset_trial_journal(path, manifest)
    assert resumed.snapshot.recovered_trials == {unit.id: trial}
    resumed.close()


def test_resume_reconciles_an_interrupted_source_failure_without_target_calls(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trials.jsonl"
    manifest = _manifest()
    failure = build_source_preparation_failure_evidence(
        manifest.selected_records[0],
        DatasetSemanticPreparationError(),
        repetitions=1,
        max_environment_api_calls=10,
        planned_target_calls=2,
        run_context=manifest.run_context,
    )
    journal = create_dataset_trial_journal(path, manifest)
    journal.terminal(
        manifest.work_plan[0],
        "errored",
        failure.reason_code,
    )
    journal.close()
    resumed = open_dataset_trial_journal(path, manifest)
    resume_evidence = DatasetResumeEvidence(
        processed_ids=frozenset({failure.interaction_id}),
        source_preparation_failures=(failure,),
        has_review_findings=False,
        has_inconclusive_materiality=False,
        invariant_evaluations=(),
        technical_results=(),
        raw_evidence_sha256="0" * 64,
    )

    _reconcile_source_preparation_failures(resumed, resume_evidence)

    snapshot = resumed.snapshot
    assert snapshot.terminal_states == {unit.id: "errored" for unit in manifest.work_plan}
    assert set(snapshot.terminal_reason_codes.values()) == {failure.reason_code}
    assert _attempted_target_calls(snapshot) == 0
    resumed.close()


def test_source_outcome_projection_failure_builds_durable_evidence() -> None:
    manifest = _manifest()

    failure = build_source_preparation_failure_evidence(
        manifest.selected_records[0],
        DatasetSourceOutcomeProjectionError(),
        repetitions=1,
        max_environment_api_calls=10,
        planned_target_calls=2,
        run_context=manifest.run_context,
    )

    assert failure.reason_code == "source_outcome_projection_failed"
    assert "declared outcome projection" in failure.summary


def test_skipped_trial_is_recovered_without_duplicate_terminal_transition(tmp_path: Path) -> None:
    path = tmp_path / "trials.jsonl"
    manifest = _manifest()
    unit = manifest.work_plan[1]
    skipped_trial = (
        _evaluation_result("interaction-1")
        .baseline.trial_set.trials[0]
        .model_copy(
            update={
                "execution_evidence": None,
                "target_output": None,
                "observed_frame": None,
                "inconclusive_reasons": (
                    "paired original repetition was inconclusive; variation not executed",
                ),
            }
        )
    )
    journal = create_dataset_trial_journal(path, manifest)
    journal.finish(unit, skipped_trial)
    journal.close()

    resumed = open_dataset_trial_journal(path, manifest)

    assert resumed.snapshot.terminal_states[unit.id] == "skipped"
    assert resumed.snapshot.recovered_trials[unit.id] == skipped_trial
    resumed.close()


def test_truncated_journal_is_preserved_and_rejected(tmp_path: Path) -> None:
    path = tmp_path / "trials.jsonl"
    manifest = _manifest()
    journal = create_dataset_trial_journal(path, manifest)
    journal.start(manifest.work_plan[0])
    journal.close()
    corrupted = path.read_bytes()[:-1]
    path.write_bytes(corrupted)

    with pytest.raises(ValueError, match="truncated or corrupted"):
        open_dataset_trial_journal(path, manifest)

    assert path.read_bytes() == corrupted


def test_complete_tail_record_truncation_is_rejected_by_durable_anchor(tmp_path: Path) -> None:
    path = tmp_path / "trials.jsonl"
    manifest = _manifest()
    journal = create_dataset_trial_journal(path, manifest)
    journal.start(manifest.work_plan[0])
    journal.finish(
        manifest.work_plan[0],
        _evaluation_result("interaction-1").baseline.trial_set.trials[0],
    )
    journal.close()
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1]))

    with pytest.raises(ValueError, match="durable anchor"):
        open_dataset_trial_journal(path, manifest)


def test_second_resume_cannot_claim_locked_journal(tmp_path: Path) -> None:
    path = tmp_path / "trials.jsonl"
    manifest = _manifest()
    first = create_dataset_trial_journal(path, manifest)

    with pytest.raises(ValueError, match=r"temporarily unavailable|locked|Resource busy"):
        open_dataset_trial_journal(path, manifest)

    first.close()


def test_journal_rejects_hard_links_and_symlinks(tmp_path: Path) -> None:
    path = tmp_path / "trials.jsonl"
    manifest = _manifest()
    create_dataset_trial_journal(path, manifest).close()
    hard_link = tmp_path / "trials-hard-link.jsonl"
    hard_link.hardlink_to(path)

    with pytest.raises(ValueError, match="hard link"):
        open_dataset_trial_journal(path, manifest)

    manifest_path = tmp_path / "manifest.json"
    persist_dataset_run_manifest(manifest_path, manifest)
    symlink = tmp_path / "manifest-symlink.json"
    symlink.symlink_to(manifest_path)
    with pytest.raises(ValueError, match=r"regular file|symbolic link|changed while opening"):
        read_dataset_run_manifest(symlink)


def test_manifest_rejects_unbounded_repetitions_before_plan_materialization() -> None:
    with pytest.raises(ValueError, match="less than or equal to 100"):
        _run_config(repetitions=101, planned_environment_api_calls=1)


def test_repeated_recovery_of_ten_by_ten_by_three_plan_never_duplicates_mutations(
    tmp_path: Path,
) -> None:
    records = tuple(_evaluation_result(f"interaction-{index}").source for index in range(10))
    operator_ids = (
        "input.surface.rephrase",
        "input.surface.typing_noise",
        "input.surface.case_variation",
        "input.surface.punctuation_noise",
        "input.surface.grammar_error",
        "input.surface.disfluency_repeat",
        "input.surface.fragmented_syntax",
        "input.style.terse",
        "input.style.verbose",
        "input.intent.self_correction",
    )
    target_config = cast(DatasetEvidenceRunContext, _run_context((records[0],))).target.config
    run_context = build_dataset_evidence_run_context(
        selected_records=records,
        selected_operator_ids=operator_ids,
        run_config=_run_config(
            repetitions=3,
            planned_environment_api_calls=330,
            max_environment_api_calls=2_000,
        ),
        invariant_suite=None,
        target_config=target_config,
        settings=_settings(),
    )
    manifest = create_dataset_run_manifest(
        run_context=run_context,
        selected_records=records,
        selected_operator_ids=operator_ids,
        run_config=_run_config(
            repetitions=3,
            planned_environment_api_calls=330,
            max_environment_api_calls=2_000,
        ),
        save_augmentations=True,
    )
    path = tmp_path / "trials.jsonl"
    journal = create_dataset_trial_journal(path, manifest)
    mutations_by_unit: dict[str, int] = {}
    template_trial = _evaluation_result("interaction-0").baseline.trial_set.trials[0]

    for unit in manifest.work_plan:
        journal.start(unit)
        mutations_by_unit[unit.id] = mutations_by_unit.get(unit.id, 0) + 1
        journal.finish(unit, template_trial.model_copy(update={"repetition": unit.repetition}))
        journal.close()
        journal = open_dataset_trial_journal(path, manifest)

    assert len(manifest.work_plan) == 10 * (1 + 10) * 3
    assert set(mutations_by_unit.values()) == {1}
    assert len(journal.snapshot.terminal_states) == len(manifest.work_plan)
    journal.close()
