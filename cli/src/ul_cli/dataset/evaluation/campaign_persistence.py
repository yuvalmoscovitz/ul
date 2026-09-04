from __future__ import annotations

from dataclasses import dataclass
from typing import TextIO

import typer

from ul_cli.dataset_augmentation_ledger import (
    DatasetAugmentationLedger,
    create_private_augmentation_ledger,
    open_augmentation_ledger_for_resume,
)
from ul_cli.finding_reference import (
    FindingReferenceContext,
    finding_reference_key_path,
    resolve_finding_reference_context,
)

from ..evidence.persistence import open_resume_output
from .campaign import PreparedCampaign


@dataclass
class CampaignPersistence:
    augmentation_ledger: DatasetAugmentationLedger | None
    output_stream: TextIO
    finding_reference_context: FindingReferenceContext

    def close(self) -> None:
        try:
            if not self.output_stream.closed:
                self.output_stream.close()
        finally:
            if self.augmentation_ledger is not None:
                self.augmentation_ledger.close()

    def __enter__(self) -> CampaignPersistence:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def open_campaign_persistence(campaign: PreparedCampaign) -> CampaignPersistence:
    evaluation = campaign.evaluation
    durable = campaign.durable
    request = evaluation.request
    output = request.output
    assert output is not None and evaluation.run_context is not None
    ledger: DatasetAugmentationLedger | None = None
    ledger_was_created = False
    output_stream: TextIO | None = None
    failure_parameter = "--augmentations-output"
    try:
        if request.augmentations_output is not None:
            if request.requested.resume is not None and request.augmentations_output.exists():
                ledger = open_augmentation_ledger_for_resume(
                    request.augmentations_output,
                    expected_context=evaluation.augmentation_generation_context,
                    selected_records=durable.all_records,
                )
            else:
                ledger = create_private_augmentation_ledger(
                    request.augmentations_output,
                    generation_context=evaluation.augmentation_generation_context,
                    selected_records=durable.all_records,
                )
                ledger_was_created = True
                if durable.resume_evidence is not None:
                    for result in durable.resume_evidence.technical_results:
                        ledger.append(source=result.source, augmentation=result.augmentation)
            durable.saved_augmentations.clear()
            durable.saved_augmentations.update(
                {record.source.id: record.augmentation for record in ledger.snapshot.records}
            )
            if durable.resume_evidence is not None:
                for result in durable.resume_evidence.technical_results:
                    if durable.saved_augmentations.get(result.source.id) != result.augmentation:
                        raise ValueError(
                            "augmentation ledger does not match completed evaluation evidence"
                        )
        failure_parameter = "--resume" if request.requested.resume is not None else "--output"
        output_stream, locked_evidence = open_resume_output(
            output,
            expected_context=evaluation.run_context,
            selected_records=durable.all_records,
            invariant_suite=evaluation.invariant_suite,
        )
        finding_output = output.with_name(f"{output.name}.findings.jsonl")
        if request.requested.resume is None:
            if locked_evidence.processed_ids:
                raise ValueError("new evidence output is not empty")
            if (
                finding_output.exists()
                or finding_output.is_symlink()
                or finding_reference_key_path(finding_output).exists()
                or finding_reference_key_path(finding_output).is_symlink()
            ):
                raise ValueError("new finding package outputs already exist")
        elif locked_evidence != durable.resume_evidence:
            raise ValueError("resume evidence changed after preflight")
        return CampaignPersistence(
            ledger,
            output_stream,
            resolve_finding_reference_context(finding_output),
        )
    except (OSError, ValueError) as error:
        if ledger is not None:
            if ledger_was_created:
                assert request.augmentations_output is not None
                ledger.discard_if_empty(request.augmentations_output)
            ledger.close()
        if output_stream is not None and not output_stream.closed:
            output_stream.close()
        message = str(error) if isinstance(error, ValueError) else error.__class__.__name__
        raise typer.BadParameter(
            f"cannot safely open persistence file ({message})", param_hint=failure_parameter
        ) from None
