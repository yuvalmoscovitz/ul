from __future__ import annotations

import argparse
import hmac
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_TOKEN_ENVIRONMENT_VARIABLE = "UL_ENVIRONMENT_AGENT_TOKEN"


class QualificationRequestHandler(BaseHTTPRequestHandler):
    server_version = "ULQualificationAgent/1.0"

    def do_POST(self) -> None:
        expected_token = os.environ.get(_TOKEN_ENVIRONMENT_VARIABLE, "")
        supplied_token = self.headers.get("Authorization", "")
        if not expected_token or not hmac.compare_digest(supplied_token, expected_token):
            self._write_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        if self.path != "/invoke":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
            value = payload["input"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        receipt_path = os.environ.get("UL_QUALIFICATION_RECEIPT")
        if receipt_path:
            with Path(receipt_path).open("a", encoding="utf-8") as receipt:
                receipt.write(json.dumps({"input": value}, sort_keys=True) + "\n")
        self._write_json(HTTPStatus.OK, {"response": {"status": "open", "ticket": 42}})

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", arguments.port), QualificationRequestHandler)
    print(server.server_port, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
