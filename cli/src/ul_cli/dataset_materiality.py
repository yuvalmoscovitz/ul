from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict
from ul import DatasetSemanticSettings, OpenRouterDecisionSettings


class DatasetOutcomeComparisonSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="UL_DATASET_", env_file=".env", extra="ignore")

    materiality_judge: Literal["llm", "jev"] | None = None


class _DatasetDecisionSettings(OpenRouterDecisionSettings):
    model_config = SettingsConfigDict(env_file=".env")


def dataset_decision_settings(settings: DatasetSemanticSettings) -> OpenRouterDecisionSettings:
    decision = _DatasetDecisionSettings(
        live_calls=settings.live_calls,
        allow_external_data_processing=settings.allow_external_data_processing,
        timeout_seconds=settings.timeout_seconds,
    )
    if settings.semantic_provider_type == "openrouter":
        decision = decision.model_copy(update={"api_key": settings.api_key})
    return decision
