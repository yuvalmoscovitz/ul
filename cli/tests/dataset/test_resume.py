from __future__ import annotations

import hashlib
import json
import re
import stat
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner
from ul import (
    DatasetAugmentationResult,
    EnvironmentLifecycleEvidence,
    EnvironmentTurnEvidence,
    EvaluatorModelPreflight,
    ExecutionEvidence,
    JsonHttpEnvironmentConfig,
    ObservedAgentOutput,
    OutcomeProjection,
)
from ul.dataset_invariants import (
    DatasetInvariantSuite,
    JsonValueEqualsLiteralInvariant,
)
from ul_cli import dataset_augmentation_ledger as augmentation_ledger_module
from ul_cli import dataset_review, dataset_trial_journal
from ul_cli.dataset.evaluation import command as command_module
from ul_cli.dataset.evaluation import runner as runner_module
from ul_cli.dataset.evidence import customer as customer_module
from ul_cli.dataset.evidence import persistence as persistence_module
from ul_cli.main import app as root_app

from ._factories import (
    _evaluation_result,
    _evaluator_preflight,
    _invariant_evaluation,
    _rich_evaluation_result,
    _run_context,
    _settings,
)
from ._files import (
    _record,
    _write_dataset,
    _write_invariant_suite,
    _write_target_config,
)

runner = CliRunner()
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def test_resume_reuses_manifest_without_original_data_config_or_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "evidence.jsonl"
    source = _evaluation_result("interaction-1").source
    run_context = _run_context((source,))
    manifest = dataset_trial_journal.create_dataset_run_manifest(
        run_context=run_context,
        selected_records=(source,),
        selected_operator_ids=("input.surface.rephrase",),
        repetitions=1,
        max_environment_api_calls=10,
        allow_environment_network=True,
        confirm_test_environment=True,
        allow_insecure_http=False,
        save_augmentations=True,
    )
    evidence.write_bytes(b"")
    dataset_trial_journal.persist_dataset_run_manifest(
        dataset_trial_journal.manifest_path(evidence), manifest
    )
    dataset_trial_journal.create_dataset_trial_journal(
        dataset_trial_journal.journal_path(evidence), manifest
    ).close()
    persistence_module.persist_evaluator_preflight(evidence, _evaluator_preflight())
    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", _settings)

    result = runner.invoke(
        root_app,
        ["dataset", "evaluate", "--resume", str(evidence), "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "Selected interactions: 1" in result.output
    assert "input.surface.rephrase" in result.output


def test_resume_fails_closed_when_completed_trials_have_no_durable_augmentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "evidence.jsonl"
    evaluation = _evaluation_result("interaction-1")
    run_context = _run_context((evaluation.source,))
    manifest = dataset_trial_journal.create_dataset_run_manifest(
        run_context=run_context,
        selected_records=(evaluation.source,),
        selected_operator_ids=("input.surface.rephrase",),
        repetitions=1,
        max_environment_api_calls=10,
        allow_environment_network=True,
        confirm_test_environment=True,
        allow_insecure_http=False,
        save_augmentations=False,
    )
    evidence.write_bytes(b"")
    dataset_trial_journal.persist_dataset_run_manifest(
        dataset_trial_journal.manifest_path(evidence), manifest
    )
    journal = dataset_trial_journal.create_dataset_trial_journal(
        dataset_trial_journal.journal_path(evidence), manifest
    )
    journal.start(manifest.work_plan[0])
    journal.finish(manifest.work_plan[0], evaluation.baseline.trial_set.trials[0])
    journal.close()
    persistence_module.persist_evaluator_preflight(evidence, _evaluator_preflight())
    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", _settings)

    result = runner.invoke(
        root_app,
        ["dataset", "evaluate", "--resume", str(evidence), "--dry-run"],
    )

    assert result.exit_code == 2
    assert "resume_incompatible:augmentation_not_durable" in result.output


def test_resume_skips_already_processed_interaction_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    augmentations = tmp_path / "evidence.augmentations.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record("interaction-1"), _record("interaction-2")])
    _write_target_config(target_config)
    evaluation_results = (
        _evaluation_result("interaction-1"),
        _evaluation_result("interaction-2"),
    )
    selected_records = tuple(result.source for result in evaluation_results)
    run_context = _run_context(selected_records)
    generation_context = augmentation_ledger_module.create_dataset_augmentation_generation_context(
        selected_records=selected_records,
        operators=(("input.surface.rephrase", "1.0.0"),),
        semantic_settings=augmentation_ledger_module.DatasetAugmentationLedgerSemanticSettings(
            provider="openrouter",
            endpoint_sha256=_settings().semantic_endpoint_sha256,
            model=_settings().model,
            render_model=_settings().render_model,
            equivalence_model=_settings().equivalence_model,
            max_input_chars=_settings().max_input_chars,
            max_output_tokens=_settings().max_output_tokens,
            max_render_tokens=_settings().max_render_tokens,
            max_response_bytes=_settings().max_response_bytes,
            timeout_seconds=_settings().timeout_seconds,
        ),
    )
    with augmentation_ledger_module.create_private_augmentation_ledger(
        augmentations,
        generation_context=generation_context,
        selected_records=selected_records,
    ) as augmentation_ledger:
        for evaluation_result in evaluation_results:
            augmentation_ledger.append(
                source=evaluation_result.source,
                augmentation=evaluation_result.augmentation,
            )
    evidence.write_text(
        json.dumps(
            customer_module.build_customer_evidence_record(
                evaluation_results[0],
                repetitions=1,
                max_environment_api_calls=4,
                planned_target_calls=4,
                run_context=cast(Any, run_context),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    evaluated_ids: list[str] = []
    preflight_calls = 0

    class FakeTarget:
        @classmethod
        def from_config(cls, config: JsonHttpEnvironmentConfig, **options: object) -> FakeTarget:
            return cls()

    async def fake_evaluate(
        records: tuple[Any, ...],
        operator_ids: tuple[str, ...],
        settings: object,
        target: object,
        output_stream: Any,
        *,
        repetitions: int,
        max_environment_api_calls: int,
        planned_target_calls: int,
        run_context: object,
        augmentation_ledger: object,
        saved_augmentations: object,
        redaction_engine: object,
        evaluator_preflight: object,
    ) -> tuple[object, ...]:
        del (
            operator_ids,
            settings,
            target,
            max_environment_api_calls,
            planned_target_calls,
            augmentation_ledger,
        )
        assert (
            cast(dict[str, DatasetAugmentationResult], saved_augmentations)["interaction-2"]
            == evaluation_results[1].augmentation
        )
        assert redaction_engine is None
        assert evaluator_preflight == _evaluator_preflight()
        assert repetitions == 1
        for record in records:
            evaluated_ids.append(record.id)
        output_stream.write(
            json.dumps(
                customer_module.build_customer_evidence_record(
                    evaluation_results[1],
                    repetitions=1,
                    max_environment_api_calls=4,
                    planned_target_calls=4,
                    run_context=cast(Any, run_context),
                )
            )
            + "\n"
        )
        output_stream.flush()
        return ()

    monkeypatch.setattr(
        command_module,
        "load_dataset_semantic_settings",
        _settings,
    )
    monkeypatch.setattr(command_module, "JsonHttpEnvironmentConnection", FakeTarget)
    monkeypatch.setattr(command_module, "evaluate_interaction_records", fake_evaluate)

    async def unexpected_paid_preflight(settings: object) -> EvaluatorModelPreflight:
        nonlocal preflight_calls
        del settings
        preflight_calls += 1
        raise AssertionError("partial resume must reuse its preflight receipt")

    monkeypatch.setattr(command_module, "preflight_evaluator", unexpected_paid_preflight)
    command = [
        "dataset",
        "evaluate",
        str(dataset),
        "--environment-config",
        str(target_config),
        "--allow-environment-network",
        "--confirm-test-environment",
        "--repetitions",
        "1",
        "--resume",
        str(evidence),
    ]
    dry_run = runner.invoke(root_app, [*command, "--dry-run"])

    assert dry_run.exit_code == 2
    assert "required receipt evidence.jsonl.preflight.json is missing" in dry_run.output

    missing_result = runner.invoke(root_app, command)

    assert missing_result.exit_code == 2
    assert "required receipt evidence.jsonl.preflight.json is missing" in missing_result.output
    assert preflight_calls == 0
    assert evaluated_ids == []

    receipt = persistence_module.persist_evaluator_preflight(evidence, _evaluator_preflight())

    mismatched_preflight = _evaluator_preflight().model_copy(update={"endpoint_sha256": "b" * 64})
    receipt.write_text(mismatched_preflight.model_dump_json() + "\n", encoding="utf-8")
    mismatch_dry_run = runner.invoke(root_app, [*command, "--dry-run"])
    mismatch_result = runner.invoke(root_app, command)

    assert mismatch_dry_run.exit_code == 2
    assert "cannot reuse evaluator preflight receipt" in mismatch_dry_run.output
    assert mismatch_result.exit_code == 2
    assert "cannot reuse evaluator preflight receipt" in mismatch_result.output
    assert "restore the matching receipt and semantic settings" in mismatch_result.output
    assert preflight_calls == 0
    assert evaluated_ids == []

    receipt.write_text(_evaluator_preflight().model_dump_json() + "\n", encoding="utf-8")

    valid_dry_run = runner.invoke(root_app, [*command, "--dry-run"])

    assert valid_dry_run.exit_code == 0, valid_dry_run.output
    assert (
        "Resume compatible: 1 complete interaction(s) skipped; 1 remaining" in valid_dry_run.output
    )
    assert "preflight=0" in valid_dry_run.output
    assert "Evaluator preflight profile:" not in valid_dry_run.output
    assert "Evidence destination:" in valid_dry_run.output
    assert evidence.name in valid_dry_run.output
    assert f"Augmentations destination: {augmentations}" in " ".join(
        _ANSI_ESCAPE_PATTERN.sub("", valid_dry_run.output).split()
    )
    assert "Please transfer 100 to Alice." not in valid_dry_run.output
    assert "Candidate input: omitted (sensitive)" in valid_dry_run.output

    sensitive_dry_run = runner.invoke(
        root_app,
        [*command, "--dry-run", "--show-sensitive-values"],
    )
    assert sensitive_dry_run.exit_code == 0, sensitive_dry_run.output
    warning_position = sensitive_dry_run.output.index("may contain sensitive data")
    candidate_position = sensitive_dry_run.output.index("Please transfer 100 to Alice.")
    assert warning_position < candidate_position

    result = runner.invoke(root_app, command)

    assert result.exit_code == 0, result.output
    assert preflight_calls == 0
    assert evaluated_ids == ["interaction-2"]
    lines = [line for line in evidence.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["interaction_id"] == "interaction-1"
    assert json.loads(lines[1])["interaction_id"] == "interaction-2"
    assert "skipped" in result.output
    if sys.platform != "win32":
        assert stat.S_IMODE(evidence.stat().st_mode) == 0o600


def test_resume_dry_run_rejects_ledger_that_disagrees_with_completed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    augmentations = tmp_path / "evidence.augmentations.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config)
    evaluation_result = _evaluation_result("interaction-1")
    selected_records = (evaluation_result.source,)
    run_context = _run_context(selected_records)
    evidence.write_text(
        json.dumps(
            customer_module.build_customer_evidence_record(
                evaluation_result,
                repetitions=1,
                max_environment_api_calls=2,
                planned_target_calls=2,
                run_context=cast(Any, run_context),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    generation_context = augmentation_ledger_module.create_dataset_augmentation_generation_context(
        selected_records=selected_records,
        operators=(("input.surface.rephrase", "1.0.0"),),
        semantic_settings=augmentation_ledger_module.DatasetAugmentationLedgerSemanticSettings(
            provider="openrouter",
            endpoint_sha256=_settings().semantic_endpoint_sha256,
            model=_settings().model,
            render_model=_settings().render_model,
            equivalence_model=_settings().equivalence_model,
            max_input_chars=_settings().max_input_chars,
            max_output_tokens=_settings().max_output_tokens,
            max_render_tokens=_settings().max_render_tokens,
            max_response_bytes=_settings().max_response_bytes,
            timeout_seconds=_settings().timeout_seconds,
        ),
    )
    mismatched_candidate = evaluation_result.augmentation.candidates[0].model_copy(
        update={"augmented_input": "A different generated variation."}
    )
    mismatched_augmentation = evaluation_result.augmentation.model_copy(
        update={"candidates": (mismatched_candidate,)}
    )
    with augmentation_ledger_module.create_private_augmentation_ledger(
        augmentations,
        generation_context=generation_context,
        selected_records=selected_records,
    ) as ledger:
        ledger.append(source=evaluation_result.source, augmentation=mismatched_augmentation)
    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", _settings)

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--repetitions",
            "1",
            "--resume",
            str(evidence),
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "cannot safely resume evidence" in result.output


def test_resume_exits_early_when_all_records_already_processed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record("interaction-1")])
    _write_target_config(target_config)
    evaluation_result = _evaluation_result("interaction-1")
    run_context = _run_context((evaluation_result.source,))
    evidence.write_text(
        json.dumps(
            customer_module.build_customer_evidence_record(
                evaluation_result,
                repetitions=1,
                max_environment_api_calls=2,
                planned_target_calls=2,
                run_context=cast(Any, run_context),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", _settings)

    command = [
        "dataset",
        "evaluate",
        str(dataset),
        "--environment-config",
        str(target_config),
        "--allow-environment-network",
        "--confirm-test-environment",
        "--repetitions",
        "1",
        "--resume",
        str(evidence),
    ]
    dry_run = runner.invoke(root_app, [*command, "--dry-run"])
    result = runner.invoke(root_app, command)

    assert dry_run.exit_code == 0, dry_run.output
    assert "Potential semantic model calls: up to 0" in dry_run.output
    assert "preflight=0" in dry_run.output
    assert "Estimated completion tokens: 0..0" in dry_run.output
    assert result.exit_code == 0, result.output
    assert "Nothing to do" in result.output


def test_all_complete_resume_preserves_prior_review_finding_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config)
    evaluation_result = _evaluation_result("interaction-1", has_review_finding=True)
    run_context = _run_context((evaluation_result.source,))
    evidence.write_text(
        json.dumps(
            customer_module.build_customer_evidence_record(
                evaluation_result,
                repetitions=1,
                max_environment_api_calls=2,
                planned_target_calls=2,
                run_context=cast(Any, run_context),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", _settings)

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--repetitions",
            "1",
            "--resume",
            str(evidence),
        ],
    )

    assert result.exit_code == 1, result.output
    assert "Nothing to do" in result.output


@pytest.mark.parametrize(
    ("invariant_status", "expected_exit_code"),
    [("violated", 1), ("not_evaluable", 2)],
)
def test_all_complete_resume_preserves_prior_invariant_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invariant_status: str,
    expected_exit_code: int,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    invariant_path = tmp_path / "invariants.json"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config)
    _write_invariant_suite(invariant_path)
    invariant_suite = command_module.load_dataset_invariant_suite(invariant_path)
    evaluation_result = _evaluation_result("interaction-1")
    baseline_output = (
        {"final_amount": 200, "corrected_amount": 100}
        if invariant_status == "violated"
        else {"status": "ok"}
    )
    baseline_trial = evaluation_result.baseline.trial_set.trials[0].model_copy(
        update={
            "target_output": ObservedAgentOutput(
                raw_output=baseline_output,
                metadata=(
                    {
                        "committed_state_snapshot": baseline_output,
                        "state_observation_authority": "environment_self_reported",
                    }
                    if invariant_status == "violated"
                    else {}
                ),
            )
        }
    )
    evaluation_result = evaluation_result.model_copy(
        update={
            "baseline": evaluation_result.baseline.model_copy(
                update={
                    "trial_set": evaluation_result.baseline.trial_set.model_copy(
                        update={"trials": (baseline_trial,)}
                    )
                }
            )
        }
    )
    invariant_evaluation = runner_module.evaluate_dataset_invariants(
        evaluation_result,
        invariant_suite,
    )
    assert invariant_evaluation.baseline.rules[0].status == invariant_status
    run_context = _run_context(
        (evaluation_result.source,),
        invariant_suite=invariant_suite,
    )
    evidence.write_text(
        json.dumps(
            customer_module.build_customer_evidence_record(
                evaluation_result,
                repetitions=1,
                max_environment_api_calls=2,
                planned_target_calls=2,
                run_context=cast(Any, run_context),
                invariant_evaluation=invariant_evaluation,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", _settings)

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--invariants",
            str(invariant_path),
            "--repetitions",
            "1",
            "--resume",
            str(evidence),
        ],
    )

    assert result.exit_code == expected_exit_code, result.output
    assert "Nothing to do" in result.output


def test_resume_rejects_forged_invariant_outcome(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    invariant_path = tmp_path / "invariants.json"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config)
    _write_invariant_suite(invariant_path)
    invariant_suite = command_module.load_dataset_invariant_suite(invariant_path)
    evaluation_result = _evaluation_result("interaction-1")
    run_context = _run_context(
        (evaluation_result.source,),
        invariant_suite=invariant_suite,
    )
    evidence.write_text(
        json.dumps(
            customer_module.build_customer_evidence_record(
                evaluation_result,
                repetitions=1,
                max_environment_api_calls=2,
                planned_target_calls=2,
                run_context=cast(Any, run_context),
                invariant_evaluation=_invariant_evaluation(
                    "violated",
                    interaction_id="interaction-1",
                    suite_sha256=invariant_suite.sha256,
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--invariants",
            str(invariant_path),
            "--repetitions",
            "1",
            "--resume",
            str(evidence),
            "--dry-run",
        ],
    )

    assert result.exit_code == 2
    assert "cannot safely resume evidence" in result.output


def test_resume_snapshot_detects_same_summary_content_change() -> None:
    evaluation_result = _evaluation_result("interaction-1")
    run_context = _run_context((evaluation_result.source,))

    def validated_snapshot(max_environment_api_calls: int) -> dataset_review.DatasetResumeEvidence:
        raw_evidence = (
            json.dumps(
                customer_module.build_customer_evidence_record(
                    evaluation_result,
                    repetitions=1,
                    max_environment_api_calls=max_environment_api_calls,
                    planned_target_calls=2,
                    run_context=cast(Any, run_context),
                )
            )
            + "\n"
        ).encode()
        return dataset_review.validate_dataset_resume_evidence(
            raw_evidence,
            expected_context=cast(Any, run_context),
            selected_records=(evaluation_result.source,),
            invariant_suite=None,
            evidence_projector=customer_module.build_customer_evidence_record,
        )

    first_snapshot = validated_snapshot(2)
    changed_snapshot = validated_snapshot(3)

    assert first_snapshot.processed_ids == changed_snapshot.processed_ids
    assert first_snapshot.has_review_findings == changed_snapshot.has_review_findings
    assert first_snapshot.raw_evidence_sha256 != changed_snapshot.raw_evidence_sha256
    assert first_snapshot != changed_snapshot


def test_resume_rejects_projected_evidence_not_derived_from_raw_response() -> None:
    evaluation_result = _evaluation_result("interaction-1")
    projection = OutcomeProjection(action="/result/action")
    base_context = cast(Any, _run_context((evaluation_result.source,)))
    projected_config = base_context.target.config.model_copy(update={"outcome": projection})
    run_context = cast(
        Any,
        _run_context((evaluation_result.source,), target_config=projected_config),
    )
    forged_normalized = {"action": "approve"}
    execution_evidence = ExecutionEvidence(
        evidence_scope="response_only",
        case_id="interaction-1:current_baseline:round-1",
        environment_id="test-environment",
        environment_config_sha256="a" * 64,
        turns=(
            EnvironmentTurnEvidence(
                turn_id="turn-1",
                response={"result": {"action": "refund"}},
                normalized_response=forged_normalized,
                public_normalized_response=forged_normalized,
                outcome_projection_sha256=projection.digest,
            ),
        ),
        final_response={"result": {"action": "refund"}},
        normalized_result=forged_normalized,
        public_normalized_result=forged_normalized,
        outcome_projection=projection.model_dump(mode="json"),
        outcome_projection_sha256=projection.digest,
        lifecycle=EnvironmentLifecycleEvidence(
            terminal_status="succeeded",
            completed_phases=("execute_turn",),
            delivery="certain",
            cleanup="not_attempted",
            environment_state_uncertain=False,
        ),
    )
    baseline_trial = evaluation_result.baseline.trial_set.trials[0].model_copy(
        update={"execution_evidence": execution_evidence}
    )
    forged_result = evaluation_result.model_copy(
        update={
            "baseline": evaluation_result.baseline.model_copy(
                update={
                    "trial_set": evaluation_result.baseline.trial_set.model_copy(
                        update={"trials": (baseline_trial,)}
                    )
                }
            )
        }
    )

    with pytest.raises(ValueError, match="normalized response does not match"):
        dataset_review._validate_resumed_outcome_projections(
            forged_result,
            run_context.target,
        )


def test_resume_accepts_canonical_probe_projection_receipt() -> None:
    evaluation_result = _evaluation_result("interaction-1")
    projection = OutcomeProjection(
        complete_result="/result",
        private_json_pointers=("/secret",),
    )
    receipt = {
        "outcome_projection": projection.model_dump(mode="json"),
        "outcome_projection_sha256": projection.digest,
    }
    target = dataset_review.DatasetEvidenceTarget(
        kind="probe_target",
        receipt=receipt,
        sha256=dataset_review._canonical_json_sha256(receipt),
    )

    dataset_review._validate_resumed_outcome_projections(evaluation_result, target)


def test_resume_accepts_rich_evidence_schema_1_9() -> None:
    evaluation_result = _rich_evaluation_result()
    selected_records = (evaluation_result.source,)
    run_context = _run_context(selected_records)
    raw_evidence = (
        json.dumps(
            customer_module.build_customer_evidence_record(
                evaluation_result,
                repetitions=1,
                max_environment_api_calls=2,
                planned_target_calls=2,
                run_context=cast(Any, run_context),
            )
        )
        + "\n"
    ).encode()

    snapshot = dataset_review.validate_dataset_resume_evidence(
        raw_evidence,
        expected_context=cast(Any, run_context),
        selected_records=selected_records,
        invariant_suite=None,
        evidence_projector=customer_module.build_customer_evidence_record,
    )

    assert snapshot.processed_ids == frozenset({evaluation_result.source.id})
    assert snapshot.technical_results == (evaluation_result,)


def test_resume_accepts_empty_evidence_as_zero_progress() -> None:
    selected_records = (_evaluation_result("interaction-1").source,)
    snapshot = dataset_review.validate_dataset_resume_evidence(
        b"",
        expected_context=cast(Any, _run_context(selected_records)),
        selected_records=selected_records,
        invariant_suite=None,
        evidence_projector=customer_module.build_customer_evidence_record,
    )

    assert snapshot.processed_ids == frozenset()
    assert snapshot.technical_results == ()
    assert snapshot.raw_evidence_sha256 == hashlib.sha256(b"").hexdigest()


def test_resume_accepts_extended_invariant_evidence_schema() -> None:
    evaluation_result = _evaluation_result("interaction-1")
    suite = DatasetInvariantSuite(
        schema_version="1.1.0",
        observation_source="target_output",
        observation_authority="committed_state_snapshot",
        rules=(
            JsonValueEqualsLiteralInvariant(
                type="json_value_equals_literal",
                id="approval-is-current",
                version="1.0.0",
                description="The approval must be current.",
                severity="critical",
                value_pointer="/approval",
                literal="current",
            ),
        ),
    )
    baseline_trial = evaluation_result.baseline.trial_set.trials[0].model_copy(
        update={"target_output": ObservedAgentOutput(raw_output={"approval": "current"})}
    )
    evaluation_result = evaluation_result.model_copy(
        update={
            "baseline": evaluation_result.baseline.model_copy(
                update={
                    "trial_set": evaluation_result.baseline.trial_set.model_copy(
                        update={"trials": (baseline_trial,)}
                    )
                }
            )
        }
    )
    invariant_evaluation = runner_module.evaluate_dataset_invariants(
        evaluation_result,
        suite,
    )
    run_context = _run_context((evaluation_result.source,), invariant_suite=suite)
    raw_evidence = (
        json.dumps(
            customer_module.build_customer_evidence_record(
                evaluation_result,
                repetitions=1,
                max_environment_api_calls=2,
                planned_target_calls=2,
                run_context=cast(Any, run_context),
                invariant_evaluation=invariant_evaluation,
            )
        )
        + "\n"
    ).encode()

    snapshot = dataset_review.validate_dataset_resume_evidence(
        raw_evidence,
        expected_context=cast(Any, run_context),
        selected_records=(evaluation_result.source,),
        invariant_suite=suite,
        evidence_projector=customer_module.build_customer_evidence_record,
    )

    assert snapshot.processed_ids == frozenset({"interaction-1"})
    assert snapshot.invariant_evaluations == (invariant_evaluation,)


def test_resume_rejects_legacy_unbound_evidence(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config)
    evidence.write_text(
        json.dumps({"schema_version": "1.4.0", "interaction_id": "interaction-1"}) + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--repetitions",
            "1",
            "--resume",
            str(evidence),
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "cannot safely resume evidence" in result.output


def test_resume_rejects_changed_evaluation_plan(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config)
    evaluation_result = _evaluation_result("interaction-1")
    run_context = _run_context((evaluation_result.source,))
    evidence.write_text(
        json.dumps(
            customer_module.build_customer_evidence_record(
                evaluation_result,
                repetitions=1,
                max_environment_api_calls=2,
                planned_target_calls=2,
                run_context=cast(Any, run_context),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--operator",
            "input.tone.frustrated",
            "--repetitions",
            "1",
            "--resume",
            str(evidence),
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "incompatible with the current evaluation plan" in result.output


def test_resume_rejects_evidence_without_terminal_newline(tmp_path: Path) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config)
    evaluation_result = _evaluation_result("interaction-1")
    run_context = _run_context((evaluation_result.source,))
    evidence.write_text(
        json.dumps(
            customer_module.build_customer_evidence_record(
                evaluation_result,
                repetitions=1,
                max_environment_api_calls=2,
                planned_target_calls=2,
                run_context=cast(Any, run_context),
            )
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--repetitions",
            "1",
            "--resume",
            str(evidence),
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "must end with a newline" in result.output


@pytest.mark.skipif(sys.platform == "win32", reason="Unix permission semantics")
def test_resume_dry_run_accepts_read_only_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    target_config = tmp_path / "target.json"
    _write_dataset(dataset, [_record()])
    _write_target_config(target_config)
    evaluation_result = _evaluation_result("interaction-1")
    run_context = _run_context((evaluation_result.source,))
    evidence.write_text(
        json.dumps(
            customer_module.build_customer_evidence_record(
                evaluation_result,
                repetitions=1,
                max_environment_api_calls=2,
                planned_target_calls=2,
                run_context=cast(Any, run_context),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    evidence.chmod(0o400)
    monkeypatch.setattr(command_module, "load_dataset_semantic_settings", _settings)

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--repetitions",
            "1",
            "--resume",
            str(evidence),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert stat.S_IMODE(evidence.stat().st_mode) == 0o400


def test_resume_rejects_mismatched_output_path(
    tmp_path: Path,
) -> None:
    """--resume and --output must point to the same file when both are given."""
    dataset = tmp_path / "interactions.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    other_output = tmp_path / "other.jsonl"
    target_config = tmp_path / "target.json"

    _write_dataset(dataset, [_record()])
    _write_target_config(target_config)
    evidence.write_text(
        json.dumps({"schema_version": "1.4.0", "interaction_id": "x", "cases": []}) + "\n",
        encoding="utf-8",
    )
    other_output.write_text("", encoding="utf-8")

    result = runner.invoke(
        root_app,
        [
            "dataset",
            "evaluate",
            str(dataset),
            "--environment-config",
            str(target_config),
            "--allow-environment-network",
            "--confirm-test-environment",
            "--resume",
            str(evidence),
            "--output",
            str(other_output),
        ],
    )

    assert result.exit_code != 0
    assert "same file" in result.output
