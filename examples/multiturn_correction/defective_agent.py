from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import ClassVar, cast


class DefectiveCorrectionHandler(BaseHTTPRequestHandler):
    environment_id: ClassVar[str] = "multiturn-correction-example"
    state_lock: ClassVar[Lock] = Lock()
    generation: ClassVar[int] = 0
    committed_invoice: ClassVar[str | None] = None
    requested_invoice: ClassVar[str | None] = None

    def do_POST(self) -> None:
        content_length = int(self.headers.get("content-length", "0"))
        try:
            request = cast(dict[str, object], json.loads(self.rfile.read(content_length)))
        except (json.JSONDecodeError, ValueError):
            self._send(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON"})
            return
        if self.path == "/reset":
            with type(self).state_lock:
                type(self).generation += 1
                type(self).committed_invoice = None
                type(self).requested_invoice = None
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
                state = {
                    "committed_invoice": type(self).committed_invoice,
                    "requested_invoice": type(self).requested_invoice,
                }
            self._send(
                HTTPStatus.OK,
                {
                    "case_id": request.get("case_id"),
                    "turn_id": request.get("turn_id"),
                    "state": state,
                },
            )
        elif self.path == "/execute":
            raw_input = request.get("input")
            if not isinstance(raw_input, str):
                self._send(HTTPStatus.BAD_REQUEST, {"error": "input required"})
                return
            requested_invoice = "AC-101" if "AC-101" in raw_input else "AC-100"
            with type(self).state_lock:
                type(self).requested_invoice = requested_invoice
                if type(self).committed_invoice is None:
                    type(self).committed_invoice = requested_invoice
                    response = f"Paid invoice {requested_invoice}."
                else:
                    response = "The original payment is already complete."
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


def create_server(port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), DefectiveCorrectionHandler)


if __name__ == "__main__":
    create_server().serve_forever()
