#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


VALIDATE_ROOT = Path("/home/const/subnet66/tau/workspace/validate/netuid-66")
STATE_PATH = VALIDATE_ROOT / "state.json"
DASHBOARD_HISTORY_PATH = VALIDATE_ROOT / "dashboard_history.json"
DUELS_DIR = VALIDATE_ROOT / "duels"
POOL_DIR = VALIDATE_ROOT / "task-pool"
RETEST_POOL_DIR = VALIDATE_ROOT / "task-pool-retest"
TASKS_ROOT = Path("/home/const/subnet66/tau/workspace/tasks")
PM2_ERROR_LOG = Path("/home/const/.pm2/logs/validator-error.log")
RESTART_REQUEST_MARKER = "draining current duel before validator restart"
RESTART_EXIT_MARKER = "Restart requested at safe boundary"
STARTUP_OK_MARKERS = (
    "Connected to chain for netuid",
    "Startup will force an immediate private submission refresh",
)
STARTUP_BAD_MARKERS = (
    "Traceback (most recent call last):",
    "refusing to continue until it is merged",
    "Validator missing required runtime secret",
)


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _timestamp_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _pm2_pid() -> str:
    result = _run(["pm2", "pid", "validator"])
    return result.stdout.strip()


def _send_restart_signal() -> None:
    _run(["pm2", "sendSignal", "SIGUSR1", "validator"])


def _stop_pm2() -> None:
    _run(["pm2", "stop", "validator"])


def _start_pm2() -> None:
    _run(["pm2", "start", "validator"])


def _read_log_lines() -> list[str]:
    if not PM2_ERROR_LOG.exists():
        return []
    return PM2_ERROR_LOG.read_text(errors="replace").splitlines()


def _load_state() -> dict[str, Any]:
    payload = _load_json(STATE_PATH)
    if not isinstance(payload, dict):
        raise SystemExit(f"invalid state file: {STATE_PATH}")
    return payload


def _active_duel_summary(state: dict[str, Any]) -> str | None:
    active = state.get("active_duel")
    if not isinstance(active, dict):
        return None
    duel_id = active.get("duel_id")
    challenger = active.get("challenger") or {}
    challenger_uid = challenger.get("uid")
    return f"duel_id={duel_id} challenger_uid={challenger_uid}"


def _cooperative_stop(*, timeout_seconds: int, poll_seconds: int, force: bool) -> None:
    pid = _pm2_pid()
    if not pid or pid == "0":
        return
    state = _load_state()
    active_summary = _active_duel_summary(state)
    if not active_summary:
        _stop_pm2()
        return

    print(f"requesting cooperative stop with active {active_summary}")
    _send_restart_signal()
    start_offset = len(_read_log_lines())
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        state = _load_state()
        if not isinstance(state.get("active_duel"), dict):
            _stop_pm2()
            return
        lines = _read_log_lines()[start_offset:]
        start_offset = len(_read_log_lines())
        for line in lines:
            if RESTART_REQUEST_MARKER in line or RESTART_EXIT_MARKER in line:
                print(line)
        time.sleep(poll_seconds)

    if not force:
        raise SystemExit(
            "timed out waiting for active duel to drain; rerun with --force to stop validator immediately"
        )
    print("cooperative drain timed out; forcing validator stop")
    _stop_pm2()


def _wait_for_startup(*, timeout_seconds: int, start_offset: int) -> None:
    seen_ok = {marker: False for marker in STARTUP_OK_MARKERS}
    deadline = time.time() + timeout_seconds
    offset = start_offset
    while time.time() < deadline:
        lines = _read_log_lines()
        new_lines = lines[offset:]
        offset = len(lines)
        for line in new_lines:
            for marker in STARTUP_BAD_MARKERS:
                if marker in line:
                    raise SystemExit(f"validator restart hit startup error: {line}")
            for marker in STARTUP_OK_MARKERS:
                if marker in line:
                    seen_ok[marker] = True
        if all(seen_ok.values()):
            return
        time.sleep(1)
    missing = [marker for marker, seen in seen_ok.items() if not seen]
    raise SystemExit(
        "timed out waiting for validator startup markers: " + ", ".join(missing)
    )


def _load_duel(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"invalid duel file: {path}")
    return payload


def _latest_rewind_backup(from_duel: int) -> Path | None:
    root = VALIDATE_ROOT / "rewind-backups"
    if not root.exists():
        return None
    prefix = f"duel-{from_duel:06d}-"
    candidates = [path for path in root.iterdir() if path.is_dir() and path.name.startswith(prefix)]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _is_primary_duel(duel: dict[str, Any]) -> bool:
    if duel.get("task_set_phase") == "confirmation_retest":
        return False
    if duel.get("confirmation_of_duel_id") is not None:
        return False
    challenger = duel.get("challenger")
    if isinstance(challenger, dict) and challenger.get("manual_retest_of_duel_id") is not None:
        return False
    return True


def _dedupe_submissions(submissions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_commitments: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for submission in submissions:
        commitment = str(submission.get("commitment") or "")
        if not commitment or commitment in seen_commitments:
            continue
        seen_commitments.add(commitment)
        deduped.append(submission)
    return deduped


def _sort_queue_key(submission: dict[str, Any]) -> tuple[int, int, str]:
    try:
        block = int(submission.get("commitment_block") or 0)
    except (TypeError, ValueError):
        block = 0
    try:
        uid = int(submission.get("uid") or 0)
    except (TypeError, ValueError):
        uid = 0
    return block, uid, str(submission.get("hotkey") or "")


def _same_submission(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    return (
        str(a.get("hotkey") or "") == str(b.get("hotkey") or "")
        and int(a.get("uid") or -1) == int(b.get("uid") or -1)
        and str(a.get("commit_sha") or "") == str(b.get("commit_sha") or "")
        and str(a.get("commitment") or "") == str(b.get("commitment") or "")
    )


def _reconstruct_recent_kings(
    *,
    before_duel: int,
    restored_king: dict[str, Any],
    window: int,
    duels_root: Path = DUELS_DIR,
) -> list[dict[str, Any]]:
    if window <= 0:
        return []
    recent: list[dict[str, Any]] = [restored_king]
    target = restored_king
    duel_paths = sorted(
        (
            path for path in duels_root.glob("*.json")
            if path.stem.isdigit() and int(path.stem) < before_duel
        ),
        key=lambda path: int(path.stem),
        reverse=True,
    )
    for duel_path in duel_paths:
        if len(recent) >= window:
            break
        duel = _load_duel(duel_path)
        king_after = duel.get("king_after")
        king_before = duel.get("king_before")
        if not isinstance(king_after, dict) or not isinstance(king_before, dict):
            continue
        if _same_submission(king_after, king_before):
            continue
        if not _same_submission(king_after, target):
            continue
        recent.append(king_before)
        target = king_before
    return recent


def _build_rewind_plan(*, from_duel: int) -> dict[str, Any]:
    state = _load_state()
    dashboard_history = _load_json(DASHBOARD_HISTORY_PATH)
    if not isinstance(dashboard_history, list):
        raise SystemExit(f"invalid dashboard history: {DASHBOARD_HISTORY_PATH}")

    duel_paths = sorted(
        (
            path for path in DUELS_DIR.glob("*.json")
            if path.stem.isdigit() and int(path.stem) >= from_duel
        ),
        key=lambda path: int(path.stem),
    )
    if not duel_paths:
        raise SystemExit(f"no duel files found at or after {from_duel}")

    first_duel = _load_duel(duel_paths[0])
    restored_king = first_duel.get("king_before")
    if not isinstance(restored_king, dict):
        raise SystemExit(f"duel {from_duel} has no valid king_before payload")
    restored_recent_kings = _reconstruct_recent_kings(
        before_duel=from_duel,
        restored_king=restored_king,
        window=5,
    )

    replay_queue: list[dict[str, Any]] = []
    replay_hotkeys: set[str] = set()
    for duel_path in duel_paths:
        duel = _load_duel(duel_path)
        if not _is_primary_duel(duel):
            continue
        challenger = duel.get("challenger")
        if not isinstance(challenger, dict):
            continue
        replay_queue.append(challenger)
        hotkey = str(challenger.get("hotkey") or "")
        if hotkey:
            replay_hotkeys.add(hotkey)

    existing_tail: list[dict[str, Any]] = []
    for item in state.get("queue", []):
        if isinstance(item, dict):
            existing_tail.append(item)
    new_queue = _dedupe_submissions(replay_queue + existing_tail)
    replay_commitments = {
        str(item.get("commitment") or "")
        for item in replay_queue
        if isinstance(item, dict) and item.get("commitment")
    }

    new_state = dict(state)
    new_state["current_king"] = restored_king
    new_state["queue"] = sorted(new_queue, key=_sort_queue_key)
    new_state["next_duel_index"] = from_duel
    new_state["active_duel"] = None
    new_state["recent_kings"] = restored_recent_kings
    new_state["king_since"] = first_duel.get("started_at")
    new_state["king_duels_defended"] = 0
    new_state["last_weight_block"] = None

    new_state["seen_hotkeys"] = [
        hotkey
        for hotkey in state.get("seen_hotkeys", [])
        if str(hotkey) not in replay_hotkeys
    ]
    restored_hotkey = str(restored_king.get("hotkey") or "")
    if restored_hotkey and restored_hotkey not in new_state["seen_hotkeys"]:
        new_state["seen_hotkeys"].append(restored_hotkey)

    for key in ("retired_hotkeys", "disqualified_hotkeys"):
        new_state[key] = [
            hotkey
            for hotkey in state.get(key, [])
            if str(hotkey) not in replay_hotkeys
        ]

    locked = dict(state.get("locked_commitments", {}))
    blocks = dict(state.get("commitment_blocks_by_hotkey", {}))
    for hotkey in replay_hotkeys:
        locked.pop(hotkey, None)
        blocks.pop(hotkey, None)
    for submission in new_state["queue"]:
        hotkey = str(submission.get("hotkey") or "")
        commitment = str(submission.get("commitment") or "")
        if hotkey and commitment:
            locked.setdefault(hotkey, commitment)
        try:
            block = int(submission.get("commitment_block"))
        except (TypeError, ValueError):
            continue
        if hotkey:
            blocks.setdefault(hotkey, block)
    if restored_hotkey:
        restored_commitment = str(restored_king.get("commitment") or "")
        if restored_commitment:
            locked[restored_hotkey] = restored_commitment
        try:
            blocks[restored_hotkey] = int(restored_king.get("commitment_block"))
        except (TypeError, ValueError):
            pass
    new_state["locked_commitments"] = locked
    new_state["commitment_blocks_by_hotkey"] = blocks

    pruned_dashboard_history = [
        entry
        for entry in dashboard_history
        if not (
            isinstance(entry, dict)
            and str(entry.get("duel_id", "")).isdigit()
            and int(entry["duel_id"]) >= from_duel
        )
    ]

    return {
        "state": state,
        "new_state": new_state,
        "duel_paths": duel_paths,
        "dashboard_history": dashboard_history,
        "pruned_dashboard_history": pruned_dashboard_history,
        "restored_king": restored_king,
        "restored_recent_kings": restored_recent_kings,
        "replay_queue": replay_queue,
        "replay_commitments": replay_commitments,
    }


def _repair_recent_kings_from_backup(*, from_duel: int) -> dict[str, Any]:
    backup_dir = _latest_rewind_backup(from_duel)
    if backup_dir is None:
        raise SystemExit(f"no rewind backup found for duel {from_duel:06d}")
    state = _load_state()
    current_king = state.get("current_king")
    if not isinstance(current_king, dict):
        raise SystemExit("state has no current king to repair")
    backup_state_path = backup_dir / "state.json.before"
    if not backup_state_path.exists():
        raise SystemExit(f"backup is missing state snapshot: {backup_state_path}")
    backup_state = _load_json(backup_state_path)
    if not isinstance(backup_state, dict):
        raise SystemExit(f"backup state is invalid: {backup_state_path}")
    backup_recent = backup_state.get("recent_kings")
    restored_recent_kings = [
        item for item in (backup_recent if isinstance(backup_recent, list) else [])
        if isinstance(item, dict)
    ]
    if not restored_recent_kings:
        restored_recent_kings = _reconstruct_recent_kings(
            before_duel=from_duel,
            restored_king=current_king,
            window=5,
            duels_root=DUELS_DIR,
        )
    new_state = dict(state)
    new_state["recent_kings"] = restored_recent_kings
    new_state["last_weight_block"] = None
    return {
        "backup_dir": backup_dir,
        "state": state,
        "backup_state": backup_state,
        "new_state": new_state,
        "restored_recent_kings": restored_recent_kings,
    }


def _purge_pool_jsons(pool_dir: Path) -> int:
    removed = 0
    for path in pool_dir.glob("*.json"):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def _purge_king_cache(tasks_root: Path) -> dict[str, int]:
    removed_king_solution_dirs = 0
    removed_king_compare_dirs = 0
    for task_dir in tasks_root.glob("validate-*"):
        if not task_dir.is_dir():
            continue
        king_solution_dir = task_dir / "solutions" / "king"
        if king_solution_dir.exists():
            shutil.rmtree(king_solution_dir, ignore_errors=True)
            removed_king_solution_dirs += 1
        comparisons_dir = task_dir / "comparisons"
        if comparisons_dir.exists():
            for compare_dir in comparisons_dir.iterdir():
                if compare_dir.is_dir() and "king" in compare_dir.name:
                    shutil.rmtree(compare_dir, ignore_errors=True)
                    removed_king_compare_dirs += 1
    return {
        "removed_king_solution_dirs": removed_king_solution_dirs,
        "removed_king_compare_dirs": removed_king_compare_dirs,
    }


def _write_backup(plan: dict[str, Any], *, from_duel: int) -> Path:
    backup_dir = VALIDATE_ROOT / "rewind-backups" / f"duel-{from_duel:06d}-{_timestamp_slug()}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(STATE_PATH, backup_dir / "state.json.before")
    shutil.copy2(DASHBOARD_HISTORY_PATH, backup_dir / "dashboard_history.json.before")
    duels_backup_dir = backup_dir / "duels"
    duels_backup_dir.mkdir()
    for duel_path in plan["duel_paths"]:
        shutil.copy2(duel_path, duels_backup_dir / duel_path.name)
    return backup_dir


def _archive_duels(duel_paths: list[Path], backup_dir: Path) -> None:
    archive_dir = backup_dir / "archived-duels"
    archive_dir.mkdir()
    for duel_path in duel_paths:
        duel_path.rename(archive_dir / duel_path.name)


def _print_plan(plan: dict[str, Any], *, from_duel: int) -> None:
    restored_king = plan["restored_king"]
    active_summary = _active_duel_summary(plan["state"])
    print(json.dumps({
        "from_duel": from_duel,
        "active_duel": active_summary,
        "restore_king_uid": restored_king.get("uid"),
        "restore_king_commit": restored_king.get("commit_sha"),
        "restore_recent_king_uids": [item.get("uid") for item in plan["restored_recent_kings"]],
        "duels_to_archive": len(plan["duel_paths"]),
        "first_archived_duel": int(plan["duel_paths"][0].stem),
        "last_archived_duel": int(plan["duel_paths"][-1].stem),
        "primary_replay_queue": len(plan["replay_queue"]),
        "new_queue_size": len(plan["new_state"]["queue"]),
        "dashboard_entries_removed": len(plan["dashboard_history"]) - len(plan["pruned_dashboard_history"]),
    }, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safely rewind validator state and duel history to a given duel id."
    )
    parser.add_argument("--from-duel", type=int, default=4390)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--force", action="store_true", help="Force-stop validator if cooperative drain times out.")
    parser.add_argument("--drain-timeout", type=int, default=7200)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument(
        "--repair-recent-kings-only",
        action="store_true",
        help="Repair only the recent_kings emission window from the latest rewind backup.",
    )
    args = parser.parse_args()

    if args.repair_recent_kings_only:
        repair = _repair_recent_kings_from_backup(from_duel=max(1, args.from_duel))
        print(json.dumps({
            "from_duel": args.from_duel,
            "backup_dir": str(repair["backup_dir"]),
            "restore_recent_king_uids": [item.get("uid") for item in repair["restored_recent_kings"]],
            "current_recent_king_uids": [item.get("uid") for item in repair["state"].get("recent_kings", [])],
        }, indent=2))
        if not args.apply:
            return 0
        _save_json(STATE_PATH, repair["new_state"])
        print("recent_kings repair complete")
        return 0

    plan = _build_rewind_plan(from_duel=max(1, args.from_duel))
    _print_plan(plan, from_duel=args.from_duel)
    if not args.apply:
        return 0

    log_offset = len(_read_log_lines())
    _cooperative_stop(
        timeout_seconds=max(1, args.drain_timeout),
        poll_seconds=max(1, args.poll_seconds),
        force=args.force,
    )
    backup_dir = _write_backup(plan, from_duel=args.from_duel)
    _archive_duels(plan["duel_paths"], backup_dir)
    _save_json(STATE_PATH, plan["new_state"])
    _save_json(DASHBOARD_HISTORY_PATH, plan["pruned_dashboard_history"])
    removed_primary = _purge_pool_jsons(POOL_DIR)
    removed_retest = _purge_pool_jsons(RETEST_POOL_DIR)
    purge_counts = _purge_king_cache(TASKS_ROOT)
    print(json.dumps({
        "backup_dir": str(backup_dir),
        "archived_duels": len(plan["duel_paths"]),
        "primary_pool_entries_removed": removed_primary,
        "retest_pool_entries_removed": removed_retest,
        **purge_counts,
    }, indent=2))
    _start_pm2()
    _wait_for_startup(
        timeout_seconds=max(1, args.startup_timeout),
        start_offset=log_offset,
    )
    print("validator rewind complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
