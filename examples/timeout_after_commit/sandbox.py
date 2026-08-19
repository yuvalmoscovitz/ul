from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Literal, cast

RetryMode = Literal["safe", "defective"]
_MAXIMUM_REQUEST_BYTES = 1_000_000


@dataclass
class _SandboxState:
    retry_mode: RetryMode
    fire_event: bool
    lock: Lock = field(default_factory=Lock)
    generation: int = 0
    request_count: int = 0
    committed_payment_keys: list[str] = field(default_factory=list[str])
    armed_event: dict[str, str] | None = None
    event_fired: bool = False

    def reset(self) -> int:
        self.generation += 1
        self.committed_payment_keys.clear()
        self.armed_event = None
        self.event_fired = False
        return self.generation

    def commit(self, idempotency_key: str) -> None:
        if idempotency_key not in self.committed_payment_keys:
            self.committed_payment_keys.append(idempotency_key)


class TimeoutAfterCommitServer(ThreadingHTTPServer):
    state: _SandboxState


def _handler_for_state(state: _SandboxState) -> type[BaseHTTPRequestHandler]:
    class TimeoutAfterCommitHandler(BaseHTTPRequestHandler):
        sandbox_id = "timeout-after-commit-example"

        def do_POST(self) -> None:
            with state.lock:
                state.request_count += 1
            request = self._read_request()
            if request is None:
                return
            if self.path == "/reset":
                with state.lock:
                    generation = state.reset()
                self._send(
                    HTTPStatus.OK,
                    {"case_id": request.get("case_id"), "generation": generation, "clean": True},
                )
                return
            if self.path == "/setup":
                self._send(HTTPStatus.OK, {"case_id": request.get("case_id")})
                return
            if self.path == "/snapshot":
                with state.lock:
                    matching_payment_count = len(state.committed_payment_keys)
                self._send(
                    HTTPStatus.OK,
                    {
                        "case_id": request.get("case_id"),
                        "turn_id": request.get("turn_id"),
                        "state": {
                            "matching_payment_count": matching_payment_count,
                        },
                    },
                )
                return
            if self.path == "/execute":
                self._execute(request)
                return
            if self.path == "/timeout-after-commit":
                self._control_event(request)
                return
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def _execute(self, request: dict[str, object]) -> None:
            if not isinstance(request.get("input"), str):
                self._send(HTTPStatus.BAD_REQUEST, {"error": "input required"})
                return
            with state.lock:
                state.commit("payment-request")
                if state.armed_event is not None and state.fire_event:
                    state.event_fired = True
                    retry_key = (
                        "payment-request" if state.retry_mode == "safe" else "payment-request-retry"
                    )
                    state.commit(retry_key)
            self._send(
                HTTPStatus.OK,
                {
                    "case_id": request.get("case_id"),
                    "turn_id": request.get("turn_id"),
                    "response": "Payment workflow completed after retry.",
                },
            )

        def _control_event(self, request: dict[str, object]) -> None:
            expected_keys = {
                "sandbox_id",
                "case_id",
                "operator_id",
                "operator_version",
                "event_id",
                "turn_id",
                "action_id",
                "operation",
            }
            if set(request) != expected_keys or any(
                not isinstance(request[key], str) for key in expected_keys
            ):
                self._send(HTTPStatus.BAD_REQUEST, {"error": "invalid event request"})
                return
            identity = cast(dict[str, str], {key: request[key] for key in expected_keys})
            if (
                identity["sandbox_id"] != self.sandbox_id
                or identity["operator_id"] != "tool.timeout_after_commit"
                or identity["operator_version"] != "1.0.0"
                or identity["action_id"] != "execute-payment"
            ):
                self._send(HTTPStatus.BAD_REQUEST, {"error": "unsupported event request"})
                return
            operation = identity["operation"]
            correlated_identity = {
                key: value for key, value in identity.items() if key != "operation"
            }
            with state.lock:
                if operation == "arm":
                    if state.armed_event is not None:
                        self._send(HTTPStatus.CONFLICT, {"error": "event already armed"})
                        return
                    state.armed_event = correlated_identity
                    state.event_fired = False
                    status = "armed"
                elif operation in {"observe", "clean"}:
                    if state.armed_event != correlated_identity:
                        self._send(HTTPStatus.CONFLICT, {"error": "event correlation mismatch"})
                        return
                    if operation == "observe":
                        status = "fired" if state.event_fired else "not_fired"
                    else:
                        status = "cleaned"
                        state.armed_event = None
                        state.event_fired = False
                else:
                    self._send(HTTPStatus.BAD_REQUEST, {"error": "invalid event operation"})
                    return
            self._send(
                HTTPStatus.OK,
                {**correlated_identity, "operation": operation, "status": status},
            )

        def _read_request(self) -> dict[str, object] | None:
            try:
                content_length = int(self.headers.get("content-length", ""))
            except ValueError:
                content_length = -1
            if content_length < 0 or content_length > _MAXIMUM_REQUEST_BYTES:
                self._send(HTTPStatus.BAD_REQUEST, {"error": "invalid content length"})
                return None
            try:
                parsed = json.loads(self.rfile.read(content_length))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON"})
                return None
            if not isinstance(parsed, dict):
                self._send(HTTPStatus.BAD_REQUEST, {"error": "JSON object required"})
                return None
            return cast(dict[str, object], parsed)

        def _send(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            body = json.dumps(
                {**payload, "sandbox_id": self.sandbox_id}, separators=(",", ":")
            ).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return TimeoutAfterCommitHandler


def create_server(
    port: int = 8766,
    *,
    retry_mode: RetryMode = "defective",
    fire_event: bool = True,
) -> TimeoutAfterCommitServer:
    state = _SandboxState(retry_mode=retry_mode, fire_event=fire_event)
    server = TimeoutAfterCommitServer(("127.0.0.1", port), _handler_for_state(state))
    server.state = state
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("safe", "defective", "unfired"), default="defective")
    arguments = parser.parse_args()
    selected_variant = cast(Literal["safe", "defective", "unfired"], arguments.variant)
    create_server(
        retry_mode="safe" if selected_variant == "safe" else "defective",
        fire_event=selected_variant != "unfired",
    ).serve_forever()
