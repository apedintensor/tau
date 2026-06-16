"""Overlay the dashboard queue from validator state.json.

The validator owns queue ordering and membership. serve.py replaces the
published snapshot queue with the live queue persisted in state.json so the
dashboard stays accurate even when R2 or compact dashboard files lag.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from hotkey_uid_cache import hotkey_uid_map
except Exception:  # pragma: no cover - optional during lightweight imports
    hotkey_uid_map = None  # type: ignore[assignment]


def _read_json_dict(path: Path) -> dict[str, Any]:
    try:
        with open(path, "rb") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    accepted_at = str(item.get("accepted_at") or "")
    if accepted_at:
        return (0, accepted_at)
    block = item.get("commitment_block")
    try:
        return (1, int(block))
    except (TypeError, ValueError):
        return (2, str(item.get("hotkey") or ""))


def _normalize_queue_item(item: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(item)
    hotkey = str(normalized.get("hotkey") or "")
    if hotkey and not normalized.get("repo"):
        normalized["repo"] = (
            normalized.get("repo_full_name")
            or normalized.get("display_repo_full_name")
            or hotkey
        )
    source = str(normalized.get("source") or "")
    if source.startswith("private"):
        normalized.pop("submission_block", None)
        normalized.pop("registration_block", None)
        normalized.pop("commitment_block", None)
    return normalized


def _fill_queue_uids(
    queue_items: list[dict[str, Any]],
    *,
    netuid: int | None = None,
) -> list[dict[str, Any]]:
    if hotkey_uid_map is None:
        return queue_items

    missing = [
        str(item.get("hotkey") or "")
        for item in queue_items
        if isinstance(item, dict) and item.get("hotkey") and item.get("uid") is None
    ]
    if not missing:
        return queue_items

    try:
        uid_map = hotkey_uid_map(netuid=netuid)
    except Exception:
        return queue_items

    if not uid_map:
        return queue_items

    filled: list[dict[str, Any]] = []
    for item in queue_items:
        if not isinstance(item, dict):
            continue
        merged = dict(item)
        if merged.get("uid") is None:
            hotkey = str(merged.get("hotkey") or "")
            uid = uid_map.get(hotkey)
            if uid is not None:
                merged["uid"] = int(uid)
        filled.append(merged)
    return filled


def _should_display_queue_item(
    item: dict[str, Any],
    state_payload: dict[str, Any],
) -> bool:
    hotkey = str(item.get("hotkey") or "")
    if not hotkey:
        return False
    disqualified = state_payload.get("disqualified_hotkeys")
    if isinstance(disqualified, list) and hotkey in disqualified:
        return False
    commitment = str(item.get("commitment") or "")
    dueled = state_payload.get("dueled_challenger_commitments")
    if commitment and isinstance(dueled, dict):
        hotkey_dueled = dueled.get(hotkey)
        if isinstance(hotkey_dueled, list) and commitment in hotkey_dueled:
            return False
    return True


def queue_from_validator_state(
    *,
    status: dict[str, Any],
    validate_root: Path,
    state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return the live validator queue from state.json."""
    if not isinstance(status, dict):
        return []

    state_payload = state if isinstance(state, dict) else _read_json_dict(validate_root / "state.json")
    state_queue = state_payload.get("queue")
    if isinstance(state_queue, list) and state_queue:
        source_queue = state_queue
    else:
        snapshot_queue = status.get("queue")
        source_queue = snapshot_queue if isinstance(snapshot_queue, list) else []

    queue_by_hotkey: dict[str, dict[str, Any]] = {}
    for item in source_queue:
        if not isinstance(item, dict) or not item.get("hotkey"):
            continue
        if not _should_display_queue_item(item, state_payload):
            continue
        hotkey = str(item["hotkey"])
        queue_by_hotkey[hotkey] = _normalize_queue_item(item)

    merged = sorted(queue_by_hotkey.values(), key=_sort_key)
    netuid = status.get("netuid")
    try:
        netuid_value = int(netuid) if netuid is not None else 66
    except (TypeError, ValueError):
        netuid_value = 66
    return _fill_queue_uids(merged, netuid=netuid_value)


def augment_dashboard_status_queue(
    *,
    status: dict[str, Any],
    validate_root: Path,
) -> dict[str, Any]:
    if not isinstance(status, dict):
        return status
    merged_queue = queue_from_validator_state(status=status, validate_root=validate_root)
    if merged_queue == status.get("queue"):
        return status
    return {**status, "queue": merged_queue}


def augment_dashboard_payload(
    payload: dict[str, Any],
    *,
    dashboard_data_path: str | Path,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    status = payload.get("status")
    if not isinstance(status, dict):
        return payload

    validate_root = Path(dashboard_data_path).resolve().parent
    new_status = augment_dashboard_status_queue(status=status, validate_root=validate_root)
    if new_status is status:
        return payload

    return {
        **payload,
        "updated_at": datetime.now(UTC).isoformat(),
        "status": new_status,
    }
