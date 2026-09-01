from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TargetExecutionConfig(_StrictModel):
    trial_timeout_seconds: float = Field(default=30.0, gt=0, le=3_600)
    max_environment_api_calls: int = Field(ge=1)
    environment_api_calls_per_trial: int = Field(ge=1)
    planned_environment_api_calls: int = Field(ge=1)
    allow_network_egress: bool
    test_environment_confirmed: bool
    allow_insecure_http: bool

    @model_validator(mode="after")
    def validate_call_budget(self) -> TargetExecutionConfig:
        if self.planned_environment_api_calls > self.max_environment_api_calls:
            raise ValueError("planned environment API calls exceed the authorized call budget")
        return self


class DatasetRunConfig(_StrictModel):
    evaluation_mode: Literal["variance"] = "variance"
    repetitions: int = Field(ge=1, le=100)
    target: TargetExecutionConfig
