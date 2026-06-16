"""Durable diff-judge telemetry that survives bittensor log-level clobbering."""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("swe-eval.diff-judge")

_LOG_LOCK = threading.Lock()
_LOG_PATH: Path | None = None
_FILE_LOGGER_CONFIGURED = False


def configure_diff_judge_log(validate_root: Path) -> Path:
    """Write structured judge events to validate_root/diff-judge.jsonl."""
    global _LOG_PATH, _FILE_LOGGER_CONFIGURED
    path = validate_root / "diff-judge.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK:
        _LOG_PATH = path
        logger = logging.getLogger("swe-eval.diff-judge")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
                handler.close()
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        _FILE_LOGGER_CONFIGURED = True
    return path


def diff_judge_log_path() -> Path | None:
    return _LOG_PATH


def record_diff_judge_event(**fields: Any) -> None:
    """Append one JSON line to diff-judge.jsonl."""
    payload = {
        "ts": datetime.now(UTC).isoformat(),
        **{key: value for key, value in fields.items() if value is not None},
    }
    line = json.dumps(payload, sort_keys=True, default=str)
    with _LOG_LOCK:
        log.info(line)
