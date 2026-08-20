from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import ClassVar, cast


class DefectiveRetryHandler(BaseHTTPRequestHandler):
    environment_id: ClassVar[str] = "retry-after-successful-commit-example"
    state_lock: ClassVar[Lock] = Lock()
    generation: ClassVar[int] = 0
    committed_effects: ClassVar[list[dict[str, str]]] = []

    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("content-length", "0"))
        except ValueError:
            self._send(HTTPStatus.BAD_REQUEST, {"error": "invalid content length"})
            return
        if content_length < 0 or content_length > 100_000:
            self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"})
            return
        try:
            raw_request: object = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, ValueError):
            self._send(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON"})
            return
        if not isinstance(raw_request, dict):
            self._send(HTTPStatus.BAD_REQUEST, {"error": "JSON object required"})
            return
        request = cast(dict[str, object], raw_request)
        if self.path == "/reset":
            with type(self).state_lock:
                type(self).generation += 1
                type(self).committed_effects = []
                generation = type(self).generation
            self._send(
                HTTPStatus.OK,
                {
                    "case_id": request.get("case_id"),
                    "generation": generation,
                    "clean": True,
                    "reset_session": request.get("reset_session") is True,
                    "reset_env": request.get("reset_env") is True,
                },
            )
        elif self.path == "/setup":
            self._send(HTTPStatus.OK, {"case_id": request.get("case_id")})
        elif self.path == "/snapshot":
            with type(self).state_lock:
                committed_effects = [dict(effect) for effect in type(self).committed_effects]
            self._send(
                HTTPStatus.OK,
                {
                    "case_id": request.get("case_id"),
                    "turn_id": request.get("turn_id"),
                    "state": {
                        "committed_effect_count": len(committed_effects),
                        "committed_effects": committed_effects,
                    },
                },
            )
        elif self.path == "/execute":
            raw_input = request.get("input")
            if not isinstance(raw_input, str) or "AC-100" not in raw_input:
                self._send(HTTPStatus.BAD_REQUEST, {"error": "AC-100 input required"})
                return
            with type(self).state_lock:
                attempt = len(type(self).committed_effects) + 1
                type(self).committed_effects.append(
                    {
                        "payment_id": f"payment-{attempt}",
                        "invoice_reference": "AC-100",
                        "idempotency_key": f"invoice:AC-100:attempt-{attempt}",
                    }
                )
            response = (
                "Payment committed." if attempt == 1 else "Payment retried and committed again."
            )
            self._send(
                HTTPStatus.OK,
                {
                    "case_id": request.get("case_id"),
                    "turn_id": request.get("turn_id"),
                    "response": response,
                },
            )
        else:
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _send(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(
            {**payload, "environment_id": self.environment_id}, separators=(",", ":")
        ).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def create_server(port: int = 8766) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), DefectiveRetryHandler)


if __name__ == "__main__":
    create_server().serve_forever()
