from __future__ import annotations

import cgi
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from private_submission import (
    SubmissionCheck,
    build_public_submissions_api_payload,
    derive_submission_id,
    private_submission_check_passed,
    private_submission_registration_check,
    private_submission_signature_payload,
    record_private_submission_acceptance,
    run_private_submission_checks,
    write_private_submission_bundle,
)
from r2 import publish_submissions_api_data
from validate import _current_registration_block, _open_subtensor, _verify_hotkey_signature

log = logging.getLogger("swe-eval.submission-api")
MAX_REQUEST_BYTES = 5_000_000
MAX_AGENT_BYTES = 5_000_000
MAX_CONCURRENT_SUBMISSIONS = 2
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 6
RATE_LIMIT_MAX_FAILURES = 3

_submission_slots = threading.BoundedSemaphore(MAX_CONCURRENT_SUBMISSIONS)
_rate_lock = threading.Lock()
_rate_buckets: dict[str, list[tuple[float, bool]]] = {}


@dataclass(frozen=True)
class SubmissionApiConfig:
    private_submission_root: Path
    base_agent: Path
    run_config: Any
    judge: Any
    judge_min_score: int
    overwrite: bool = False
    max_request_bytes: int = MAX_REQUEST_BYTES
    max_agent_bytes: int = MAX_AGENT_BYTES
    rate_limit_window_seconds: int = RATE_LIMIT_WINDOW_SECONDS
    rate_limit_max_requests: int = RATE_LIMIT_MAX_REQUESTS
    rate_limit_max_failures: int = RATE_LIMIT_MAX_FAILURES


def serve_submissions_api(*, host: str, port: int, config: SubmissionApiConfig) -> None:
    handler = build_handler(config)
    server = ThreadingHTTPServer((host, port), handler)
    log.info("Serving private submissions API on http://%s:%d/api/submissions", host, port)
    server.serve_forever()


def build_handler(config: SubmissionApiConfig):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.rstrip("/") != "/api/submissions":
                send_json(self, 404, {"error": "not_found"})
                return
            send_json(self, 200, build_public_submissions_api_payload(root=config.private_submission_root))

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/api/submissions":
                send_json(self, 404, {"error": "not_found"})
                return
            client_ip = self.client_address[0] if self.client_address else "unknown"
            if not rate_limit_allowed(client_ip, config=config):
                send_json(self, 429, {"accepted": False, "error": "rate_limited"})
                return
            if request_too_large(self.headers, max_request_bytes=config.max_request_bytes):
                note_rate_result(client_ip, False, config=config)
                send_json(self, 413, {"accepted": False, "error": "request_too_large"})
                return
            if not _submission_slots.acquire(blocking=False):
                send_json(self, 503, {"accepted": False, "error": "submission_api_busy"})
                return
            status, payload = handle_submission_request(headers=self.headers, rfile=self.rfile, config=config)
            try:
                note_rate_result(client_ip, bool(payload.get("accepted")), config=config)
                send_json(self, status, payload)
            finally:
                _submission_slots.release()

        def log_message(self, fmt: str, *args: Any) -> None:
            log.info("%s - %s", self.address_string(), fmt % args)

    return Handler


def handle_submission_request(*, headers: Any, rfile: Any, config: SubmissionApiConfig) -> tuple[int, dict[str, Any]]:
    try:
        form = cgi.FieldStorage(
            fp=rfile,
            headers=headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": headers.get("Content-Type", ""),
                "CONTENT_LENGTH": headers.get("Content-Length", "0"),
            },
        )
        hotkey = form_value(form, "hotkey")
        signature = form_value(form, "signature")
        submitted_id = form_value(form, "submission_id")
        agent_py = form_file_text(form, "agent")
        if not hotkey or not signature or not agent_py:
            return 400, {"accepted": False, "error": "hotkey, signature, and agent file are required"}
        if len(agent_py.encode("utf-8")) > config.max_agent_bytes:
            return 413, {"accepted": False, "error": "agent_too_large"}
        agent_sha256 = hashlib.sha256(agent_py.encode("utf-8")).hexdigest()
        submission_id = submitted_id or derive_submission_id(
            hotkey=hotkey,
            agent_sha256=agent_sha256,
        )
        signature_payload = private_submission_signature_payload(
            hotkey=hotkey,
            submission_id=submission_id,
            agent_sha256=agent_sha256,
        )
        signature_valid = _verify_hotkey_signature(hotkey, signature_payload, signature)
        if not signature_valid:
            return 401, precheck_signature_failure_payload(
                hotkey=hotkey,
                submission_id=submission_id,
                agent_sha256=agent_sha256,
                signature_payload=signature_payload,
            )
        base_agent_py = config.base_agent.expanduser().read_text(encoding="utf-8")
        result = run_private_submission_checks(
            hotkey=hotkey,
            submitted_agent_py=agent_py,
            base_agent_py=base_agent_py,
            openrouter_judge=config.judge,
            min_score=config.judge_min_score,
        )
        registration_block, uid, registration_error = registration_context(
            hotkey=hotkey,
            config=config.run_config,
        )
        registration_check = (
            private_submission_registration_check(
                root=config.private_submission_root,
                hotkey=hotkey,
                submission_id=submission_id,
                agent_sha256=result.agent_sha256,
                registration_block=registration_block,
            )
            if registration_error is None
            else SubmissionCheck(
                name="Registration Gate",
                status="failed",
                summary="Could not verify the hotkey's current registration.",
                findings=[registration_error],
                metadata={"registration_block": registration_block, "uid": uid},
            )
        )
        result.checks["registration_gate"] = registration_check
        result.accepted = result.accepted and registration_check.status == "passed"
        bundle_path = None
        accepted = bool(result.accepted)
        if accepted:
            bundle_path = persist_accepted_submission(
                root=config.private_submission_root,
                submission_id=submission_id,
                hotkey=hotkey,
                agent_py=agent_py,
                result=result,
                signature=signature,
                registration_block=registration_block,
                overwrite=config.overwrite,
            )
            if registration_block is not None:
                record_private_submission_acceptance(
                    root=config.private_submission_root,
                    hotkey=hotkey,
                    submission_id=submission_id,
                    agent_sha256=result.agent_sha256,
                    registration_block=registration_block,
                )
            publish_submissions_api_data(build_public_submissions_api_payload(root=config.private_submission_root))

        payload = response_payload(
            accepted=accepted,
            signature_valid=signature_valid,
            submission_id=submission_id,
            result=result,
            signature_payload=signature_payload,
            bundle_path=bundle_path,
            uid=uid,
            registration_block=registration_block,
        )
        return (200 if accepted else 422), payload
    except Exception as exc:
        log.exception("private submission request failed")
        return 500, {"accepted": False, "error": str(exc)}


def persist_accepted_submission(
    *,
    root: Path,
    submission_id: str,
    hotkey: str,
    agent_py: str,
    result: Any,
    signature: str,
    registration_block: int | None,
    overwrite: bool,
) -> Path:
    existing_bundle = root / submission_id
    if (
        existing_bundle.exists()
        and not overwrite
        and private_submission_check_passed(
            root,
            submission_id,
            result.agent_sha256,
            hotkey=hotkey,
            signature_verifier=_verify_hotkey_signature,
        )
    ):
        return existing_bundle
    return write_private_submission_bundle(
        root=root,
        submission_id=submission_id,
        hotkey=hotkey,
        agent_py=agent_py,
        check_result=result,
        signature=signature,
        registration_block=registration_block,
        overwrite=overwrite,
    )


def registration_context(*, hotkey: str, config: Any) -> tuple[int | None, int | None, str | None]:
    try:
        with _open_subtensor(config) as subtensor:
            uid = subtensor.subnets.get_uid_for_hotkey_on_subnet(hotkey, config.validate_netuid)
            if uid is None:
                return None, None, f"Hotkey {hotkey} is not registered on netuid {config.validate_netuid}."
            block = _current_registration_block(
                subtensor=subtensor,
                config=config,
                hotkey=hotkey,
                uid=int(uid),
            )
    except Exception as exc:
        return None, None, f"Registration lookup failed: {exc}"
    if block is None:
        return None, int(uid), f"Could not resolve registration block for hotkey {hotkey}."
    return int(block), int(uid), None


def response_payload(
    *,
    accepted: bool,
    signature_valid: bool,
    submission_id: str,
    result: Any,
    signature_payload: bytes,
    bundle_path: Path | None,
    uid: int | None,
    registration_block: int | None,
) -> dict[str, Any]:
    ci_checks = {name: check.to_dict() for name, check in result.checks.items()}
    return {
        "accepted": accepted,
        "signature_valid": signature_valid,
        "submission_id": submission_id,
        "agent_sha256": result.agent_sha256,
        "commitment": f"private-submission:{submission_id}:{result.agent_sha256}",
        "signature_payload": signature_payload.decode("utf-8"),
        "bundle_path": str(bundle_path) if bundle_path is not None else None,
        "registration": {"uid": uid, "registration_block": registration_block},
        "ci_checks": ci_checks,
        "llm_judge": ci_checks.get("openrouter_judge"),
        "checks": ci_checks,
    }


def precheck_signature_failure_payload(
    *,
    hotkey: str,
    submission_id: str,
    agent_sha256: str,
    signature_payload: bytes,
) -> dict[str, Any]:
    signature_check = SubmissionCheck(
        name="Hotkey Signature",
        status="failed",
        summary="Hotkey signature did not verify for this private submission payload.",
        findings=[
            "Sign the exact signature_payload with the submitting miner hotkey before retrying.",
        ],
    )
    ci_checks = {"hotkey_signature": signature_check.to_dict()}
    return {
        "accepted": False,
        "signature_valid": False,
        "submission_id": submission_id,
        "agent_sha256": agent_sha256,
        "commitment": f"private-submission:{submission_id}:{agent_sha256}",
        "signature_payload": signature_payload.decode("utf-8"),
        "bundle_path": None,
        "registration": {"uid": None, "registration_block": None},
        "ci_checks": ci_checks,
        "llm_judge": None,
        "checks": ci_checks,
        "hotkey": hotkey,
    }


def form_value(form: cgi.FieldStorage, name: str) -> str:
    value = form.getvalue(name)
    return str(value or "").strip()


def form_file_text(form: cgi.FieldStorage, name: str) -> str:
    item = form[name] if name in form else None
    if item is None or not getattr(item, "file", None):
        return ""
    data = item.file.read()
    if isinstance(data, str):
        return data
    return data.decode("utf-8")


def request_too_large(headers: Any, *, max_request_bytes: int) -> bool:
    try:
        content_length = int(headers.get("Content-Length", "0"))
    except (TypeError, ValueError):
        return True
    return content_length <= 0 or content_length > max_request_bytes


def rate_limit_allowed(client_ip: str, *, config: SubmissionApiConfig) -> bool:
    now = time.monotonic()
    with _rate_lock:
        bucket = recent_rate_events(_rate_buckets.get(client_ip, []), now, config=config)
        request_count = len(bucket)
        failure_count = sum(1 for _timestamp, accepted in bucket if not accepted)
        _rate_buckets[client_ip] = bucket
    return request_count < config.rate_limit_max_requests and failure_count < config.rate_limit_max_failures


def note_rate_result(client_ip: str, accepted: bool, *, config: SubmissionApiConfig) -> None:
    now = time.monotonic()
    with _rate_lock:
        bucket = recent_rate_events(_rate_buckets.get(client_ip, []), now, config=config)
        bucket.append((now, accepted))
        _rate_buckets[client_ip] = bucket


def recent_rate_events(
    events: list[tuple[float, bool]],
    now: float,
    *,
    config: SubmissionApiConfig,
) -> list[tuple[float, bool]]:
    return recent_rate_events_for_window(events, now, config.rate_limit_window_seconds)


def recent_rate_events_for_window(
    events: list[tuple[float, bool]],
    now: float,
    window_seconds: int,
) -> list[tuple[float, bool]]:
    cutoff = now - window_seconds
    return [(timestamp, accepted) for timestamp, accepted in events if timestamp >= cutoff]


def send_json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
