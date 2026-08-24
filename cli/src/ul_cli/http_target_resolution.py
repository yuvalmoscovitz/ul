from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue
from ul.http_environment import (
    JsonHttpIsolatedResponseConfig,
    JsonHttpTargetConfig,
    json_http_environment_capabilities,
    json_http_environment_config_sha256,
    load_json_http_environment_config,
    validate_json_http_environment_configuration,
)

_ISOLATED_ADAPTER_PRESETS: dict[str, tuple[JsonValue, str]] = {
    "generic-json": ({"input": "{{input}}"}, "/response"),
    "openai-chat": (
        {"messages": [{"role": "user", "content": "{{input}}"}]},
        "/choices/0/message/content",
    ),
}


class HttpTargetEnvironmentIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(min_length=1)
    value_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class HttpTargetConfirmation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    kind: Literal["http"] = "http"
    reference: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    executable: None = None
    artifacts: tuple[()] = ()
    environment: tuple[HttpTargetEnvironmentIdentity, ...] = ()
    callable: None = None

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ResolvedHttpTarget:
    reference: str
    config: JsonHttpTargetConfig
    config_sha256: str
    confirmation: HttpTargetConfirmation

    @property
    def confirmation_sha256(self) -> str:
        return self.confirmation.sha256


def http_target_evidence_receipt(
    resolved_target: ResolvedHttpTarget,
) -> dict[str, JsonValue]:
    outcome_projection = resolved_target.config.outcome
    return {
        "kind": "http",
        "config_sha256": resolved_target.config_sha256,
        "confirmation_sha256": resolved_target.confirmation_sha256,
        "supports_state_observation": json_http_environment_capabilities(
            resolved_target.config
        ).supports_state_observation,
        "executable_sha256": None,
        "artifact_sha256": [],
        "environment": [
            item.model_dump(mode="json") for item in resolved_target.confirmation.environment
        ],
        "callable": None,
        "outcome_projection": (
            outcome_projection.model_dump(mode="json") if outcome_projection is not None else None
        ),
        "outcome_projection_sha256": (
            outcome_projection.digest if outcome_projection is not None else None
        ),
    }


def create_isolated_response_target_config(
    url: str,
    *,
    isolated_preset: Literal["generic-json", "openai-chat"] = "generic-json",
    environment_id: str | None = None,
    request_json_template: str | None = None,
    response_json_pointer: str | None = None,
    agent_model: str | None = None,
    header_from_env: list[str] | None = None,
    request_isolation_attested: bool,
    safe_test_target_attested: bool,
) -> JsonHttpIsolatedResponseConfig:
    if (
        isolated_preset == "openai-chat"
        and request_json_template is None
        and (agent_model is None or not agent_model.strip())
    ):
        raise ValueError(
            "openai-chat preset requires --agent-model unless --request-json-template is set"
        )
    if agent_model is not None and (
        isolated_preset != "openai-chat" or request_json_template is not None
    ):
        raise ValueError(
            "--agent-model is available only with the openai-chat preset request template"
        )
    preset_template, preset_pointer = _ISOLATED_ADAPTER_PRESETS[isolated_preset]
    selected_template = (
        _parse_request_json_template(request_json_template)
        if request_json_template is not None
        else (
            {"model": agent_model, **cast(dict[str, JsonValue], preset_template)}
            if isolated_preset == "openai-chat"
            else preset_template
        )
    )
    endpoint_host = urlsplit(url).hostname or "target"
    return JsonHttpIsolatedResponseConfig.model_validate(
        {
            "version": 1,
            "adapter_tier": "isolated_response",
            "environment_id": environment_id or f"isolated-response:{endpoint_host}",
            "request_isolation_attested": request_isolation_attested,
            "safe_test_target_attested": safe_test_target_attested,
            "headers_from_env": _parse_header_environment_mappings(header_from_env or []),
            "execute": {
                "url": url,
                "request_json_template": selected_template,
                "response_json_pointer": (
                    response_json_pointer if response_json_pointer is not None else preset_pointer
                ),
            },
        }
    )


def resolve_http_target(
    reference: str,
    *,
    allow_insecure_http: bool,
    http_preset: Literal["generic-json", "openai-chat"] | None = None,
    request_json_template: str | None = None,
    response_json_pointer: str | None = None,
    agent_model: str | None = None,
    header_from_env: list[str] | None = None,
    request_isolation_attested: bool | None = None,
    safe_test_target_attested: bool | None = None,
) -> ResolvedHttpTarget:
    direct_options_used = (
        http_preset is not None
        or request_json_template is not None
        or response_json_pointer is not None
        or agent_model is not None
        or bool(header_from_env)
    )
    if reference.casefold().startswith(("https://", "http://")):
        if request_isolation_attested is not True:
            raise ValueError("direct HTTP target request isolation must be attested")
        if safe_test_target_attested is not True:
            raise ValueError("direct HTTP target safety must be attested")
        config = create_isolated_response_target_config(
            reference,
            isolated_preset=http_preset or "generic-json",
            environment_id="probe-http-" + hashlib.sha256(reference.encode()).hexdigest()[:16],
            request_json_template=request_json_template,
            response_json_pointer=response_json_pointer,
            agent_model=agent_model,
            header_from_env=header_from_env,
            request_isolation_attested=request_isolation_attested,
            safe_test_target_attested=safe_test_target_attested,
        )
        return resolve_http_target_config(
            reference,
            config,
            allow_insecure_http=allow_insecure_http,
        )
    if direct_options_used:
        raise ValueError("direct HTTP mapping options require an HTTP URL target")
    path = Path(reference)
    if not path.is_file():
        raise ValueError("HTTP target must be an HTTP(S) URL or HTTP target configuration JSON")
    canonical_path = path.resolve()
    return resolve_http_target_config(
        str(canonical_path),
        load_json_http_environment_config(canonical_path),
        allow_insecure_http=allow_insecure_http,
    )


def resolve_http_target_config(
    reference: str,
    config: JsonHttpTargetConfig,
    *,
    allow_insecure_http: bool,
) -> ResolvedHttpTarget:
    validate_json_http_environment_configuration(
        config,
        test_environment_confirmed=True,
        allow_insecure_http=allow_insecure_http,
    )
    config_sha256 = json_http_environment_config_sha256(config)
    return ResolvedHttpTarget(
        reference=reference,
        config=config,
        config_sha256=config_sha256,
        confirmation=HttpTargetConfirmation(
            reference=reference,
            config_sha256=config_sha256,
            environment=tuple(
                HttpTargetEnvironmentIdentity(
                    name=environment_variable,
                    value_sha256=hashlib.sha256(
                        os.environ[environment_variable].encode("utf-8")
                    ).hexdigest(),
                )
                for environment_variable in sorted(set(config.headers_from_env.values()))
            ),
        ),
    )


def _parse_request_json_template(encoded_template: str) -> JsonValue:
    try:
        value: object = json.loads(
            encoded_template,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (json.JSONDecodeError, RecursionError, ValueError):
        raise ValueError("request JSON template must be valid standard JSON") from None
    return cast(JsonValue, value)


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _parse_header_environment_mappings(mappings: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    normalized_names: set[str] = set()
    for mapping in mappings:
        header_name, separator, environment_variable = mapping.partition("=")
        normalized_name = header_name.casefold()
        if (
            not separator
            or not header_name
            or not environment_variable
            or normalized_name in normalized_names
        ):
            raise ValueError(
                "header mappings must be unique HTTP_HEADER=UL_ENVIRONMENT_VARIABLE values"
            )
        normalized_names.add(normalized_name)
        parsed[header_name] = environment_variable
    return parsed
