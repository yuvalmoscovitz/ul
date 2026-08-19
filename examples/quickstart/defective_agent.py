from __future__ import annotations

import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
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
    state_lock = Lock()
    reset_generation = 0
    committed_action: dict[str, str] | None = None
    sandbox_id = "quickstart-accounts-payable"

    def do_POST(self) -> None:
        if self.path not in {"/reset", "/setup", "/execute", "/snapshot"}:
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
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid request"})
            return

        if self.path == "/reset":
            if not _valid_case_request(request):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid reset request"})
                return
            with type(self).state_lock:
                type(self).reset_generation += 1
                type(self).committed_action = None
                generation = type(self).reset_generation
            self._send_json(
                HTTPStatus.OK,
                {
                    "sandbox_id": self.sandbox_id,
                    "case_id": _case_id_from_request(request),
                    "generation": generation,
                    "clean": True,
                },
            )
            return
        if self.path == "/setup":
            if not _valid_case_request(request, scenario="accounts-payable"):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid setup request"})
                return
            self._send_json(
                HTTPStatus.OK,
                {"sandbox_id": self.sandbox_id, "case_id": _case_id_from_request(request)},
            )
            return
        if self.path == "/snapshot":
            if not _valid_case_request(request, requires_turn_id=True):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid snapshot request"})
                return
            with type(self).state_lock:
                committed_action = type(self).committed_action
            self._send_json(
                HTTPStatus.OK,
                {
                    "sandbox_id": self.sandbox_id,
                    "case_id": _case_id_from_request(request),
                    "turn_id": _turn_id_from_request(request),
                    "state": committed_action,
                },
            )
            return

        try:
            message, case_id, turn_id = _validated_message(request)
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid request"})
            return

        if _PAYMENT_REQUEST_PATTERN.fullmatch(message) is None:
            self._send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "unsupported request"})
            return

        invoice_reference = (
            "AC-101" if _REPEATED_PAYMENT_REQUEST_PATTERN.fullmatch(message) else "AC-100"
        )
        action = _payment_action(invoice_reference)
        with type(self).state_lock:
            type(self).committed_action = action
        self._send_json(
            HTTPStatus.OK,
            {
                "sandbox_id": self.sandbox_id,
                "case_id": case_id,
                "turn_id": turn_id,
                "result": action,
            },
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


def _validated_message(request: object) -> tuple[str, str, str]:
    if type(request) is not dict:
        raise ValueError("invalid request shape")
    request_mapping = cast(dict[str, object], request)
    if set(request_mapping) != {"case_id", "turn_id", "request", "settings"}:
        raise ValueError("invalid request shape")
    case_id = _validated_case_id(request_mapping["case_id"])
    turn_id = _validated_turn_id(request_mapping["turn_id"])
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
    return message, case_id, turn_id


def _valid_case_request(
    request: object,
    *,
    scenario: str | None = None,
    requires_turn_id: bool = False,
) -> bool:
    if type(request) is not dict:
        return False
    request_mapping = cast(dict[str, object], request)
    expected_fields = {"case_id"} if scenario is None else {"case_id", "scenario"}
    if requires_turn_id:
        expected_fields.add("turn_id")
    if set(request_mapping) != expected_fields or request_mapping.get("scenario") != scenario:
        return False
    try:
        _validated_case_id(request_mapping["case_id"])
        if requires_turn_id:
            _validated_turn_id(request_mapping["turn_id"])
    except ValueError:
        return False
    return True


def _validated_case_id(value: object) -> str:
    if type(value) is not str or not value.startswith("ul-case-") or len(value) != 40:
        raise ValueError("invalid case ID")
    return value


def _validated_turn_id(value: object) -> str:
    if type(value) is not str or not value.strip() or len(value) > 500:
        raise ValueError("invalid turn ID")
    return value


def _case_id_from_request(request: object) -> str:
    return cast(dict[str, str], request)["case_id"]


def _turn_id_from_request(request: object) -> str:
    return cast(dict[str, str], request)["turn_id"]


def create_server() -> ThreadingHTTPServer:
    with _DefectiveAgentRequestHandler.state_lock:
        _DefectiveAgentRequestHandler.reset_generation = 0
        _DefectiveAgentRequestHandler.committed_action = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DefectiveAgentRequestHandler)
    server.daemon_threads = True
    return server
