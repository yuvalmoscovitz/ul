from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib
import inspect
import json
import os
import platform
import sys
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, cast

_PROTOCOL_VERSION = "1.0.0"
_MAXIMUM_TARGET_FILE_BYTES = 256 * 1024 * 1024


def _write(message: dict[str, Any]) -> None:
    encoded = json.dumps(
        message,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    sys.stdout.buffer.write(encoded + b"\n")
    sys.stdout.buffer.flush()


def _error(request_id: str, code: str) -> None:
    _write(
        {
            "protocol_version": _PROTOCOL_VERSION,
            "type": "error",
            "request_id": request_id,
            "code": code,
        }
    )


def _read() -> dict[str, Any] | None:
    encoded = sys.stdin.buffer.readline()
    if not encoded:
        return None
    value: Any = json.loads(encoded.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError
    return cast(dict[str, Any], value)


def _target_file_sha256(target: Any) -> str:
    source_file = inspect.getsourcefile(target)
    if source_file is None:
        raise ValueError
    path = Path(source_file).resolve(strict=True)
    if not path.is_file() or path.stat().st_size > _MAXIMUM_TARGET_FILE_BYTES:
        raise ValueError
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_target(reference: str) -> tuple[Any, str]:
    module_name, separator, attribute_path = reference.partition(":")
    if not separator:
        raise ValueError
    sys.path.insert(0, os.getcwd())
    with contextlib.redirect_stdout(sys.stderr):
        value: Any = importlib.import_module(module_name)
        for attribute in attribute_path.split("."):
            value = getattr(value, attribute)
    if not callable(value):
        raise ValueError
    return value, _target_file_sha256(value)


def _request_payload(message: dict[str, Any]) -> dict[str, Any]:
    turn = message.get("turn")
    if not isinstance(turn, dict):
        raise ValueError
    context = message.get("context", {})
    if not isinstance(context, dict):
        raise ValueError
    return {
        "turn": cast(dict[str, Any], turn),
        "context": cast(dict[str, Any], context),
    }


def _normalize_result(value: Any) -> tuple[Any, list[dict[str, Any]]]:
    if isinstance(value, dict):
        typed_value = cast(dict[str, Any], value)
    else:
        return value, []
    if set(typed_value).issubset({"response", "execution_events"}) and ("response" in typed_value):
        events = typed_value.get("execution_events", [])
        if not isinstance(events, list):
            raise ValueError
        typed_events: list[dict[str, Any]] = []
        for event in cast(list[Any], events):
            if not isinstance(event, dict):
                raise ValueError
            typed_events.append(cast(dict[str, Any], event))
        return typed_value["response"], typed_events
    return typed_value, []


async def _await_result(value: Awaitable[Any]) -> Any:
    return await value


def _invoke(
    target: Any,
    message: dict[str, Any],
    input_mode: str,
    runner: asyncio.Runner,
) -> tuple[Any, list[dict[str, Any]]]:
    payload = _request_payload(message)
    argument = payload if input_mode == "request" else payload["turn"]["input"]
    with contextlib.redirect_stdout(sys.stderr):
        result: Any = target(argument)
        if inspect.isawaitable(result):
            result = runner.run(_await_result(result))
    response, events = _normalize_result(result)
    json.dumps(response, allow_nan=False)
    json.dumps(events, allow_nan=False)
    return response, events


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--target", required=True)
    parser.add_argument("--input-mode", choices=("value", "request"), default="value")
    arguments = parser.parse_args()
    try:
        target, target_sha256 = _load_target(arguments.target)
    except Exception:
        _error("startup", "target_load_failed")
        return 2
    active_session_id: str | None = None
    with asyncio.Runner() as runner:
        while True:
            try:
                message = _read()
            except Exception:
                _error("unknown", "malformed_request")
                return 2
            if message is None:
                return 0
            request_id = message.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                _error("unknown", "malformed_request")
                return 2
            if message.get("protocol_version") != _PROTOCOL_VERSION:
                _error(request_id, "unsupported_protocol")
                return 2
            message_type = message.get("type")
            if message_type == "start":
                _write(
                    {
                        "protocol_version": _PROTOCOL_VERSION,
                        "type": "ready",
                        "request_id": request_id,
                        "runtime": {
                            "name": platform.python_implementation(),
                            "version": platform.python_version(),
                        },
                        "target_sha256": target_sha256,
                    }
                )
                continue
            if message_type == "session_start":
                session_id = message.get("session_id")
                if not isinstance(session_id, str) or not session_id:
                    _error(request_id, "malformed_request")
                    continue
                active_session_id = session_id
                _write(
                    {
                        "protocol_version": _PROTOCOL_VERSION,
                        "type": "session_ready",
                        "request_id": request_id,
                        "session_id": session_id,
                    }
                )
                continue
            if message_type == "invoke":
                if message.get("session_id") != active_session_id:
                    _error(request_id, "session_mismatch")
                    continue
                try:
                    response, events = _invoke(target, message, arguments.input_mode, runner)
                except BaseException:
                    _error(request_id, "target_failed")
                    continue
                _write(
                    {
                        "protocol_version": _PROTOCOL_VERSION,
                        "type": "result",
                        "request_id": request_id,
                        "response": response,
                        "execution_events": events,
                    }
                )
                continue
            if message_type == "shutdown":
                _write(
                    {
                        "protocol_version": _PROTOCOL_VERSION,
                        "type": "shutdown_complete",
                        "request_id": request_id,
                    }
                )
                return 0
            _error(request_id, "unsupported_message")


if __name__ == "__main__":
    raise SystemExit(main())
