from __future__ import annotations

import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

_MAXIMUM_REQUEST_BYTES = 16_384
_PAYMENT_REQUEST_PATTERN = re.compile(r"^(?:pay\s+)+AC-100\.$", re.IGNORECASE)
_REPEATED_PAYMENT_REQUEST_PATTERN = re.compile(r"^pay\s+(?:pay\s+)+AC-100\.$", re.IGNORECASE)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")


def _payment_action(invoice_reference: str) -> dict[str, str]:
    return {
        "action": "payment_committed",
        "payment_id": "pay-0001",
        "invoice_reference": invoice_reference,
        "requested_invoice_reference": "AC-100",
        "amount": "12500",
        "currency": "USD",
        "source_bank_account_id": "bank-main",
        "idempotency_key": f"invoice:{invoice_reference}:1",
    }


class _DefectiveAgentRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        if self.path != "/execute":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if self.headers.get("Transfer-Encoding") is not None:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid request"})
            return
        content_type = self.headers.get("Content-Type", "").partition(";")[0].strip().casefold()
        if content_type != "application/json":
            self._send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "JSON required"})
            return
        raw_content_length = self.headers.get("Content-Length")
        if (
            raw_content_length is None
            or not raw_content_length.isascii()
            or not raw_content_length.isdigit()
        ):
            self._send_json(HTTPStatus.LENGTH_REQUIRED, {"error": "content length required"})
            return
        content_length = int(raw_content_length)
        if not 0 < content_length <= _MAXIMUM_REQUEST_BYTES:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"})
            return

        try:
            body = self.rfile.read(content_length).decode("utf-8")
            request = json.loads(
                body,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonstandard_json_constant,
            )
            message = _validated_message(request)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid request"})
            return

        if _PAYMENT_REQUEST_PATTERN.fullmatch(message) is None:
            self._send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "unsupported request"})
            return

        invoice_reference = (
            "AC-101" if _REPEATED_PAYMENT_REQUEST_PATTERN.fullmatch(message) else "AC-100"
        )
        self._send_json(
            HTTPStatus.OK,
            {"result": _payment_action(invoice_reference)},
        )

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded_payload = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded_payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(encoded_payload)
        self.close_connection = True

    def log_message(self, format: str, *args: object) -> None:
        return


def _validated_message(request: object) -> str:
    if type(request) is not dict:
        raise ValueError("invalid request shape")
    request_mapping = cast(dict[str, object], request)
    if set(request_mapping) != {"request", "settings"}:
        raise ValueError("invalid request shape")
    request_object = request_mapping["request"]
    settings_object = request_mapping["settings"]
    if type(request_object) is not dict:
        raise ValueError("invalid request shape")
    nested_request = cast(dict[str, object], request_object)
    if set(nested_request) != {"message"}:
        raise ValueError("invalid request shape")
    if type(settings_object) is not dict or cast(dict[str, object], settings_object) != {
        "mode": "sandbox"
    }:
        raise ValueError("invalid settings")
    message = nested_request["message"]
    if type(message) is not str or not message.strip() or len(message) > 4_000:
        raise ValueError("invalid message")
    return message


def create_server() -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DefectiveAgentRequestHandler)
    server.daemon_threads = True
    return server
