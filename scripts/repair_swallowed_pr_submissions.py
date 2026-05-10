from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from config import RunConfig
from validate import (
    _build_github_client,
    _build_submission,
    _open_subtensor,
    _submission_is_eligible,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _collect_dueled_commitments(duels_dir: Path) -> set[str]:
    seen: set[str] = set()
    for path in duels_dir.glob("*.json"):
        try:
            payload = _load_json(path)
        except Exception:
            continue
        commitment = payload.get("challenger_commitment")
        if commitment:
            seen.add(str(commitment))
    return seen


def _build_config(workspace_root: Path) -> RunConfig:
    return RunConfig(
        workspace_root=workspace_root,
        validate_wallet_name="sn66_owner",
        validate_wallet_hotkey="default",
        solver_model="minimax/minimax-m2.7",
        solver_provider_sort="throughput",
        solver_provider_allow_fallbacks=False,
        validate_max_concurrency=1,
        validate_round_concurrency=25,
        validate_candidate_timeout_streak_limit=5,
        validate_poll_interval_seconds=600,
        validate_task_pool_target=50,
        validate_task_pool_fill_from_saved=True,
        validate_task_pool_refresh_count=0,
        validate_task_pool_refresh_interval_seconds=3600,
        validate_duel_rounds=50,
        validate_win_margin=3,
        validate_min_commitment_block=7951985,
        validate_hotkey_spent_since_block=8104340,
        validate_pool_filler_concurrency=2,
        validate_github_pr_watch=True,
        validate_github_pr_only=True,
        validate_github_pr_repo="unarbos/ninja",
        validate_github_pr_base="main",
    )


def _compute_affected_submissions(workspace_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state_path = workspace_root / "workspace/validate/netuid-66/state.json"
    duels_dir = workspace_root / "workspace/validate/netuid-66/duels"
    state = _load_json(state_path)
    queue = state.get("queue", [])
    queue_commitments = {
        item.get("commitment")
        for item in queue
        if isinstance(item, dict) and item.get("commitment")
    }
    king_commitment = ((state.get("current_king") or {}).get("commitment"))
    locked = state.get("locked_commitments", {})
    blocks = state.get("commitment_blocks_by_hotkey", {})
    dueled_commitments = _collect_dueled_commitments(duels_dir)

    config = _build_config(workspace_root)
    subtensor = _open_subtensor(config)
    github_client = _build_github_client(config)
    current_commitments = subtensor.commitments.get_all_commitments(config.validate_netuid) or {}

    affected: list[dict[str, Any]] = []
    for raw_hotkey, raw_commitment in current_commitments.items():
        hotkey = str(raw_hotkey)
        commitment = str(raw_commitment)
        if not commitment.startswith("github-pr:"):
            continue
        if locked.get(hotkey) != commitment:
            continue
        if commitment == king_commitment:
            continue
        if commitment in queue_commitments:
            continue
        if commitment in dueled_commitments:
            continue
        block = blocks.get(hotkey)
        if block is None:
            continue
        submission = _build_submission(
            subtensor=subtensor,
            github_client=github_client,
            config=config,
            hotkey=hotkey,
            commitment=commitment,
            commitment_block=int(block),
        )
        if not submission:
            continue
        if not _submission_is_eligible(
            subtensor=subtensor,
            github_client=github_client,
            config=config,
            submission=submission,
        ):
            continue
        affected.append(asdict(submission))
    return state, affected


def _remove_from_list(values: list[Any], needle: str) -> list[Any]:
    return [item for item in values if str(item) != needle]


def _repair_state(state: dict[str, Any], affected: list[dict[str, Any]]) -> dict[str, Any]:
    queue = list(state.get("queue", []))
    locked = dict(state.get("locked_commitments", {}))
    blocks = dict(state.get("commitment_blocks_by_hotkey", {}))
    seen = list(state.get("seen_hotkeys", []))
    disq = list(state.get("disqualified_hotkeys", []))
    retired = list(state.get("retired_hotkeys", []))

    queued_commitments = {
        item.get("commitment")
        for item in queue
        if isinstance(item, dict) and item.get("commitment")
    }

    for submission in affected:
        hotkey = str(submission["hotkey"])
        commitment = str(submission["commitment"])

        locked.pop(hotkey, None)
        blocks.pop(hotkey, None)
        seen = _remove_from_list(seen, hotkey)
        disq = _remove_from_list(disq, hotkey)
        retired = _remove_from_list(retired, hotkey)

        if commitment not in queued_commitments:
            queue.append(submission)
            queued_commitments.add(commitment)

    queue.sort(
        key=lambda s: (
            s.get("manual_retest_of_duel_id") is None,
            s.get("commitment_block", 0),
            s.get("uid", 0),
            s.get("hotkey", ""),
        )
    )

    state["queue"] = queue
    state["locked_commitments"] = locked
    state["commitment_blocks_by_hotkey"] = blocks
    state["seen_hotkeys"] = seen
    state["disqualified_hotkeys"] = disq
    state["retired_hotkeys"] = retired
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair swallowed GitHub PR submissions.")
    parser.add_argument("--workspace-root", type=Path, default=Path("/home/const/subnet66/tau"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    workspace_root = args.workspace_root.resolve()
    state_path = workspace_root / "workspace/validate/netuid-66/state.json"
    state, affected = _compute_affected_submissions(workspace_root)

    if not args.apply:
        print(json.dumps({"count": len(affected), "affected": affected}, indent=2))
        return

    backup_path = state_path.with_name(f"state.json.bak.swallowed-repair-{len(affected)}")
    backup_path.write_text(json.dumps(state, indent=2))
    repaired = _repair_state(state, affected)
    state_path.write_text(json.dumps(repaired, indent=2))
    print(json.dumps({"repaired_count": len(affected), "backup": str(backup_path)}, indent=2))


if __name__ == "__main__":
    main()
