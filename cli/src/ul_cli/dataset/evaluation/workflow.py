from __future__ import annotations

from .campaign import prepare_campaign
from .execution import execute_campaign
from .preparation import prepare_dataset_evaluation
from .reporting import report_campaign_results, report_completed_resume
from .request import DatasetEvaluationRequest


def run_dataset_evaluation(request: DatasetEvaluationRequest) -> None:
    try:
        evaluation = prepare_dataset_evaluation(request)
    except RuntimeError as error:
        raise ValueError(str(error)) from None
    campaign = prepare_campaign(evaluation)
    if campaign is None:
        return
    try:
        if not campaign.durable.remaining_records and campaign.durable.skipped_count > 0:
            report_completed_resume(evaluation, campaign.durable)
            return
        outcome = execute_campaign(campaign)
        report_campaign_results(
            evaluation,
            campaign.durable,
            outcome.progress_runtime,
            outcome.results,
            outcome.invariant_evaluations,
            outcome.source_preparation_events,
        )
    finally:
        campaign.durable.close()
