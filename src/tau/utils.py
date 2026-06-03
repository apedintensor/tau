from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def json_sha256(payload: dict[str, Any]) -> str:
    """Return the SHA-256 hex digest of the JSON-serialized payload.

    Keys are sorted so the result is stable regardless of insertion order.
    Used as a deterministic cache key by both LLMRequest and the proxy cache.
    """
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def get_dict(
    data: dict[Any, Any],
    key: Any,
) -> dict[Any, Any]:
    value = data.get(key)
    if isinstance(value, dict):
        return value
    return {}


class DiskCache:
    """JSON-file cache keyed by an arbitrary string.

    Each entry is stored as ``cache_dir/<key>.json`` containing any
    JSON-serializable dict. Callers choose their own key and value shape —
    this class only handles the disk I/O and error handling.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir

    def read(self, key: str) -> dict[str, Any] | None:
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[return-value]
        except Exception as exc:
            log.warning("Failed to read cache entry %s: %s", path, exc)
            return None

    def write(self, key: str, data: dict[str, Any]) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_dir / f"{key}.json"
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            log.warning("Failed to write cache entry %s: %s", path, exc)
