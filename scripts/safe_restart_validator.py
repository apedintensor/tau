#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


STATE_PATH = Path("/home/const/subnet66/tau/workspace/validate/netuid-66/state.json")
PM2_ERROR_LOG = Path("/home/const/.pm2/logs/validator-error.log")
STARTUP_OK_MARKERS = (
    "Connected to chain for netuid",
    "Startup seeded submission refresh block",
)
STARTUP_BAD_MARKERS = (
    "Traceback (most recent call last):",
    "refusing to continue until it is merged",
    "Validator missing required runtime secret",
)
RESTART_REQUEST_MARKER = "draining current duel before validator restart"
RESTART_EXIT_MARKER = "Restart requested; skipping cleanup and leaving validator loop for PM2 restart"


def _load_state() -> dict[str, Any]:
    try:
        return json.loads(STATE_PATH.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"state file not found: {STATE_PATH}") from exc


def _active_duel_summary(state: dict[str, Any]) -> str | None:
    active = state.get("active_duel")
    if not isinstance(active, dict):
        return None
    duel_id = active.get("duel_id")
    challenger = active.get("challenger") or {}
    challenger_uid = challenger.get("uid")
    status = active.get("status")
    return f"duel_id={duel_id} challenger_uid={challenger_uid} status={status}"


def _read_log_lines() -> list[str]:
    if not PM2_ERROR_LOG.exists():
        return []
    return PM2_ERROR_LOG.read_text(errors="replace").splitlines()


def _pm2_pid() -> str:
    result = _run(["pm2", "pid", "validator"])
    return result.stdout.strip()


def _send_restart_signal() -> None:
    result = _run(["pm2", "sendSignal", "SIGUSR1", "validator"])
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)


def _hard_restart_pm2() -> None:
    result = _run(["pm2", "restart", "validator"])
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)


def _wait_for_cooperative_exit(*, timeout_seconds: int, poll_seconds: int, old_pid: str, log_offset: int) -> bool:
    deadline = time.time() + timeout_seconds
    start_offset = log_offset
    saw_request = False
    while time.time() < deadline:
        lines = _read_log_lines()
        new_lines = lines[start_offset:]
        start_offset = len(lines)
        for line in new_lines:
            if RESTART_REQUEST_MARKER in line:
                saw_request = True
            if RESTART_EXIT_MARKER in line:
                print("validator reached cooperative restart boundary")
        current_pid = _pm2_pid()
        if current_pid and current_pid != old_pid:
            print(f"validator restarted cooperatively (pid {old_pid} -> {current_pid})")
            return True
        state = _load_state()
        active_summary = _active_duel_summary(state)
        if active_summary:
            print(f"waiting for cooperative restart boundary: {active_summary}")
        elif saw_request:
            print("restart requested; waiting for PM2 to cycle validator")
        time.sleep(poll_seconds)
    return False


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)

def _wait_for_startup(*, timeout_seconds: int, log_offset: int) -> None:
    seen_ok = {marker: False for marker in STARTUP_OK_MARKERS}
    start_offset = log_offset
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if PM2_ERROR_LOG.exists():
            text = PM2_ERROR_LOG.read_text(errors="replace")
            lines = text.splitlines()
            new_lines = lines[start_offset:]
            start_offset = len(lines)
            for line in new_lines:
                for marker in STARTUP_BAD_MARKERS:
                    if marker in line:
                        raise SystemExit(f"validator restart hit startup error: {line}")
                for marker in STARTUP_OK_MARKERS:
                    if marker in line:
                        seen_ok[marker] = True
            if all(seen_ok.values()):
                print("validator restart verified from PM2 logs")
                return
        time.sleep(1)

    missing = [marker for marker, seen in seen_ok.items() if not seen]
    raise SystemExit(
        "timed out waiting for validator startup markers: "
        + ", ".join(missing)
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safely restart validator via PM2 using a cooperative drain signal."
    )
    parser.add_argument("--wait-timeout", type=int, default=7200)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip cooperative drain and hard-restart immediately.",
    )
    args = parser.parse_args()

    log_offset = 0
    if PM2_ERROR_LOG.exists():
        log_offset = len(PM2_ERROR_LOG.read_text(errors="replace").splitlines())
    if args.force:
        state = _load_state()
        active_summary = _active_duel_summary(state)
        if active_summary:
            print(f"forcing restart while active duel is present: {active_summary}")
        _hard_restart_pm2()
    else:
        old_pid = _pm2_pid()
        if not old_pid or old_pid == "0":
            print("validator is not running; starting it via pm2 restart")
            _hard_restart_pm2()
        else:
            _send_restart_signal()
            if not _wait_for_cooperative_exit(
                timeout_seconds=max(1, args.wait_timeout),
                poll_seconds=max(1, args.poll_seconds),
                old_pid=old_pid,
                log_offset=log_offset,
            ):
                print("cooperative restart timed out; falling back to hard pm2 restart")
                log_offset = len(_read_log_lines())
                _hard_restart_pm2()
    _wait_for_startup(
        timeout_seconds=max(1, args.startup_timeout),
        log_offset=log_offset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
