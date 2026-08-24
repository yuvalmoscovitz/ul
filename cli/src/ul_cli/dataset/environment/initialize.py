from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Annotated, Literal, cast
from urllib.parse import urlsplit

import typer
from pydantic import JsonValue, ValidationError
from ul.http_environment import (
    ENVIRONMENT_ID_PLACEHOLDER,
    JsonHttpEnvironmentConfig,
    JsonHttpIsolatedResponseConfig,
    JsonHttpTargetConfig,
)

from ..evaluation.records import (
    reject_duplicate_json_keys,
    reject_nonstandard_json_constant,
)
from ..presentation.runtime import console
from ..storage.private_files import create_private_output

_ISOLATED_ADAPTER_PRESETS: dict[str, tuple[JsonValue, str]] = {
    "generic-json": ({"input": "{{input}}"}, "/response"),
    "openai-chat": (
        {"messages": [{"role": "user", "content": "{{input}}"}]},
        "/choices/0/message/content",
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
            "request_isolation_attested": True,
            "safe_test_target_attested": True,
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


def initialize_dataset_environment(
    environment_config: Annotated[
        Path,
        typer.Argument(
            dir_okay=False,
            help="New JSON file describing the customer-managed environment API.",
        ),
    ],
    url: Annotated[
        str,
        typer.Option(
            help=(
                "Stateful API base URL, or an isolated-response POST URL without credentials, "
                "query, or fragment."
            )
        ),
    ],
    adapter_tier: Annotated[
        Literal["stateful-lifecycle", "isolated-response"],
        typer.Option(
            help=(
                "Adapter evidence tier. Isolated-response needs one endpoint but cannot verify "
                "state or conversations."
            )
        ),
    ] = "stateful-lifecycle",
    confirm_request_isolation: Annotated[
        bool,
        typer.Option(
            help="Attest every isolated-response request starts fresh and cannot affect another."
        ),
    ] = False,
    confirm_safe_test_target: Annotated[
        bool,
        typer.Option(
            help="Attest this target is safe for test requests and cannot cause real effects."
        ),
    ] = False,
    isolated_preset: Annotated[
        Literal["generic-json", "openai-chat"],
        typer.Option(help="Known request/response shape for an existing isolated JSON endpoint."),
    ] = "generic-json",
    environment_id: Annotated[
        str | None,
        typer.Option(help="Stable evidence name; isolated mode defaults to the endpoint host."),
    ] = None,
    fixture_id: Annotated[
        str | None,
        typer.Option(help="Stable stateful fixture name recorded with every run."),
    ] = None,
    fixture_version: Annotated[
        str | None,
        typer.Option(help="Version of the stateful fixture recorded with every run."),
    ] = None,
    request_json_template: Annotated[
        str | None,
        typer.Option(
            help="JSON object or array with one complete {{input}} value; overrides the preset."
        ),
    ] = None,
    response_json_pointer: Annotated[
        str | None,
        typer.Option(help="RFC 6901 pointer to the agent response; overrides the preset."),
    ] = None,
    agent_model: Annotated[
        str | None,
        typer.Option(help="Agent model sent by the openai-chat preset."),
    ] = None,
    header_from_env: Annotated[
        list[str] | None,
        typer.Option(
            "--header-from-env",
            help="HTTP_HEADER=UL_ENVIRONMENT_VARIABLE; repeat for credentials or routing.",
        ),
    ] = None,
    show_guidance: Annotated[bool, typer.Option(hidden=True)] = True,
) -> None:
    """Create a private connection config for a customer-managed agent environment API."""
    if adapter_tier == "isolated-response" and not confirm_request_isolation:
        raise typer.BadParameter(
            "isolated-response setup requires --confirm-request-isolation",
            param_hint="--confirm-request-isolation",
        )
    if adapter_tier == "isolated-response" and not confirm_safe_test_target:
        raise typer.BadParameter(
            "isolated-response setup requires --confirm-safe-test-target",
            param_hint="--confirm-safe-test-target",
        )
    if (fixture_id is None) != (fixture_version is None):
        raise typer.BadParameter(
            "--fixture-id and --fixture-version must be provided together",
            param_hint="--fixture-id",
        )
    if adapter_tier == "isolated-response" and fixture_id is not None:
        raise typer.BadParameter(
            "fixture identity applies only to --adapter-tier stateful-lifecycle",
            param_hint="--fixture-id",
        )
    isolated_options_used = (
        isolated_preset != "generic-json"
        or environment_id is not None
        or request_json_template is not None
        or response_json_pointer is not None
        or agent_model is not None
        or bool(header_from_env)
    )
    if adapter_tier != "isolated-response" and isolated_options_used:
        raise typer.BadParameter(
            "isolated adapter mapping options require --adapter-tier isolated-response",
            param_hint="--adapter-tier",
        )
    if (
        adapter_tier == "isolated-response"
        and isolated_preset == "openai-chat"
        and request_json_template is None
        and (agent_model is None or not agent_model.strip())
    ):
        raise typer.BadParameter(
            "openai-chat preset requires --agent-model unless --request-json-template is set",
            param_hint="--agent-model",
        )
    if agent_model is not None and (
        isolated_preset != "openai-chat" or request_json_template is not None
    ):
        raise typer.BadParameter(
            "--agent-model is available only with the openai-chat preset request template",
            param_hint="--agent-model",
        )
    try:
        base_url = url.rstrip("/")
        if adapter_tier == "isolated-response":
            config: JsonHttpTargetConfig = create_isolated_response_target_config(
                url,
                isolated_preset=isolated_preset,
                environment_id=environment_id,
                request_json_template=request_json_template,
                response_json_pointer=response_json_pointer,
                agent_model=agent_model,
                header_from_env=header_from_env,
            )
        else:
            config = JsonHttpEnvironmentConfig.model_validate(
                {
                    "version": 5,
                    "environment_id": ENVIRONMENT_ID_PLACEHOLDER,
                    "fixture_id": fixture_id,
                    "fixture_version": fixture_version,
                    "headers_from_env": {},
                    "reset": {
                        "url": f"{base_url}/reset",
                        "request_json_template": {"case_id": "{{case_id}}"},
                        "reset_session": True,
                        "reset_env": True,
                        "case_id_json_pointer": "/case_id",
                        "generation_json_pointer": "/generation",
                        "clean_state_json_pointer": "/clean",
                        "clean_state_value": True,
                        "environment_id_json_pointer": "/environment_id",
                    },
                    "execute_turn": {
                        "url": f"{base_url}/execute",
                        "request_json_template": {
                            "case_id": "{{case_id}}",
                            "turn_id": "{{turn_id}}",
                            "input": "{{input}}",
                        },
                        "response_json_pointer": "/response",
                        "case_id_json_pointer": "/case_id",
                        "turn_id_json_pointer": "/turn_id",
                        "environment_id_json_pointer": "/environment_id",
                    },
                    "snapshot": {
                        "url": f"{base_url}/snapshot",
                        "request_json_template": {
                            "case_id": "{{case_id}}",
                            "turn_id": "{{turn_id}}",
                        },
                        "response_json_pointer": "/state",
                        "case_id_json_pointer": "/case_id",
                        "turn_id_json_pointer": "/turn_id",
                        "environment_id_json_pointer": "/environment_id",
                    },
                }
            )
        output_stream = create_private_output(environment_config)
    except (OSError, ValidationError, ValueError) as error:
        if isinstance(error, FileExistsError):
            message = "environment config already exists; UL will not overwrite it"
        elif isinstance(error, OSError):
            message = f"cannot create environment config ({error.__class__.__name__})"
        elif isinstance(error, ValidationError):
            message = _summarize_validation_error(error)
        else:
            message = str(error)
        raise typer.BadParameter(message, param_hint="ENVIRONMENT_CONFIG") from None

    created_config_status = os.fstat(output_stream.fileno())
    try:
        with output_stream:
            json.dump(config.model_dump(mode="json", exclude_none=True), output_stream, indent=2)
            output_stream.write("\n")
    except BaseException:
        try:
            current_config_status = environment_config.lstat()
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISLNK(current_config_status.st_mode) and os.path.samestat(
                current_config_status, created_config_status
            ):
                environment_config.unlink()
        raise

    console.print(f"Created private environment connection config: {environment_config}")
    if not show_guidance:
        return
    if config.environment_id == ENVIRONMENT_ID_PLACEHOLDER:
        console.print(
            f"First: replace environment_id '{ENVIRONMENT_ID_PLACEHOLDER}' with a stable name "
            "for this test environment."
        )
    if adapter_tier == "isolated-response":
        console.print(
            "Connected the existing one-request JSON endpoint; no UL-specific endpoint is "
            "required. Set any configured UL_ENVIRONMENT_* variables, then run "
            f"'ul environment check {environment_config} --help'. This tier records response "
            "evidence only: committed state, conversations, and state-dependent stress tests "
            "are unavailable."
        )
        console.print(
            "Each request must start from fresh isolated state and remain safe for testing. "
            "Keep exactly one complete {{input}} value in the request template; {{case_id}} and "
            "{{turn_id}} are optional correlation values."
        )
        return
    console.print(
        "Next: implement the generated reset, execute, and snapshot endpoints. "
        "The reset request already asks for a clean agent session and clean external "
        "environment. Add any headers_from_env, then validate the connection with "
        "'ul environment check "
        f'{environment_config} --probe "Return environment health only; do not take action." '
        "--allow-environment-network "
        "--confirm-test-environment --confirm-harmless-probe'. After that, validate a "
        "dataset plan with 'ul dataset evaluate DATASET --environment-config "
        f"{environment_config} --dry-run'."
    )
    console.print(
        "Keep exactly one complete {{case_id}} value in every lifecycle request, "
        "{{turn_id}} in execute_turn and snapshot, "
        "and one {{input}} value in execute_turn. "
        "headers_from_env maps HTTP header names to "
        "dedicated UL_ENVIRONMENT_* environment-variable names; secret values stay outside "
        "this file."
    )
    console.print('Reset request: {"case_id":"{{case_id}}","reset_session":true,"reset_env":true}')
    console.print(
        'Reset response: {"environment_id":"...","case_id":"{{case_id}}",'
        '"generation":1,"clean":true,"reset_session":true,"reset_env":true}'
    )


def _parse_request_json_template(encoded_template: str) -> JsonValue:
    try:
        value: object = json.loads(
            encoded_template,
            object_pairs_hook=reject_duplicate_json_keys,
            parse_constant=reject_nonstandard_json_constant,
        )
    except (json.JSONDecodeError, RecursionError, ValueError):
        raise ValueError("request JSON template must be valid standard JSON") from None
    return cast(JsonValue, value)


def _summarize_validation_error(error: ValidationError) -> str:
    reasons: list[str] = []
    for issue in error.errors(include_url=False, include_context=False, include_input=False):
        field_path = ".".join(str(part) for part in issue["loc"])
        message = str(issue["msg"]).removeprefix("Value error, ")
        reasons.append(f"{field_path}: {message}" if field_path else message)
    return f"environment config is invalid: {'; '.join(reasons)}"


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
