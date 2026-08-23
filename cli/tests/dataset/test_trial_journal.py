from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from ul import DatasetTrialUnit
from ul_cli.dataset.evidence.context import build_dataset_evidence_run_context
from ul_cli.dataset_review import DatasetEvidenceRunContext
from ul_cli.dataset_trial_journal import (
    create_dataset_run_manifest,
    create_dataset_trial_journal,
    open_dataset_trial_journal,
    persist_dataset_run_manifest,
    read_dataset_run_manifest,
)

from ._factories import _evaluation_result, _run_context, _settings


def _manifest():
    source = _evaluation_result("interaction-1").source
    return create_dataset_run_manifest(
        run_context=_run_context((source,)),
        selected_records=(source,),
        selected_operator_ids=("input.surface.rephrase",),
        repetitions=1,
        max_environment_api_calls=10,
        allow_environment_network=True,
        confirm_test_environment=True,
        allow_insecure_http=False,
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
                update={"max_environment_api_calls": 11}
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
    assert resumed.snapshot.recovered_trials == {}
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


def test_repeated_recovery_of_ten_by_eleven_by_three_plan_never_duplicates_mutations(
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
        "input.tone.frustrated",
        "input.intent.self_correction",
    )
    target_config = cast(DatasetEvidenceRunContext, _run_context((records[0],))).target.config
    run_context = build_dataset_evidence_run_context(
        selected_records=records,
        selected_operator_ids=operator_ids,
        repetitions=3,
        invariant_suite=None,
        target_config=target_config,
        settings=_settings(),
    )
    manifest = create_dataset_run_manifest(
        run_context=run_context,
        selected_records=records,
        selected_operator_ids=operator_ids,
        repetitions=3,
        max_environment_api_calls=2_000,
        allow_environment_network=True,
        confirm_test_environment=True,
        allow_insecure_http=False,
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

    assert len(manifest.work_plan) == 10 * (1 + 11) * 3
    assert set(mutations_by_unit.values()) == {1}
    assert len(journal.snapshot.terminal_states) == len(manifest.work_plan)
    journal.close()
