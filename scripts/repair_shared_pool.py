#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import RunConfig  # noqa: E402
from pipeline import solve_task_run  # noqa: E402
from validate import (  # noqa: E402
    _MIN_PATCH_LINES,
    _POOL_KING_QUALIFY_TIMEOUT_SECONDS,
    PoolTask,
    TaskPool,
    ValidatorSubmission,
    _build_agent_config,
    _count_patch_lines,
    _ensure_empty_solution,
    _king_solve_qualifies_for_pool,
    _load_state,
    _prepare_validate_paths,
    _remove_solution_artifacts,
)
from task_pool_manager import task_content_fingerprint  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-repair saved validator tasks into the shared task pool.")
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--netuid", type=int, default=66)
    parser.add_argument("--solver-model")
    parser.add_argument("--solver-provider-sort")
    parser.add_argument("--solver-provider-only")
    parser.add_argument("--solver-provider-disable-fallbacks", action="store_true")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--target-count", type=int)
    parser.add_argument(
        "--pool",
        choices=("main", "retest"),
        default="main",
        help="Which validator pool to write repaired tasks into.",
    )
    parser.add_argument(
        "--exclude-pool",
        choices=("main", "retest"),
        help="Skip any task names already present in this existing pool.",
    )
    parser.add_argument(
        "--only-missing-baseline",
        action="store_true",
        help="Repair only tasks missing baseline artifacts.",
    )
    parser.add_argument(
        "--dedupe-current-pools",
        action="store_true",
        help="Remove duplicate task-content entries from the existing primary/retest pools.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the duplicate pool entries that would be removed without deleting pool JSON files.",
    )
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> RunConfig:
    defaults = RunConfig()
    return RunConfig(
        workspace_root=args.workspace_root.resolve(),
        solver_model=args.solver_model,
        solver_provider_sort=args.solver_provider_sort,
        solver_provider_only=args.solver_provider_only,
        solver_provider_allow_fallbacks=(
            False if args.solver_provider_disable_fallbacks else defaults.solver_provider_allow_fallbacks
        ),
        validate_netuid=args.netuid,
    )


def _king_paths(task_root: Path) -> tuple[Path, Path]:
    king_dir = task_root / "solutions" / "king"
    return king_dir / "solve.json", king_dir / "solution.diff"


def _baseline_paths(task_root: Path) -> tuple[Path, Path]:
    baseline_dir = task_root / "solutions" / "baseline"
    return baseline_dir / "solve.json", baseline_dir / "solution.diff"


def _has_baseline(task_root: Path) -> bool:
    solve_json, solution_diff = _baseline_paths(task_root)
    return solve_json.is_file() and solution_diff.is_file()


def _has_reference_compare(task_root: Path) -> bool:
    return (task_root / "comparisons" / "king--vs--reference" / "compare.json").is_file()


def _king_matches_current(task_root: Path, king: ValidatorSubmission) -> bool:
    solve_json, solution_diff = _king_paths(task_root)
    if not solve_json.is_file() or not solution_diff.is_file():
        return False
    try:
        payload = json.loads(solve_json.read_text())
    except Exception:
        return False
    if str(payload.get("agent") or "") != king.agent_ref:
        return False
    result = payload.get("result")
    if not isinstance(result, dict):
        return False
    return str(result.get("exit_reason") or "") in {"completed", "time_limit_exceeded"}


def _task_has_minimum_files(task_root: Path) -> bool:
    reference_patch = task_root / "task" / "reference.patch"
    return (
        (task_root / "task" / "task.json").is_file()
        and (task_root / "task" / "task.txt").is_file()
        and (task_root / "task" / "commit.json").is_file()
        and reference_patch.is_file()
        and _count_patch_lines(reference_patch) >= _MIN_PATCH_LINES
        and (task_root / "task" / "original").is_dir()
        and (task_root / "task" / "reference").is_dir()
    )


def _priority_task_names(validate_root: Path) -> set[str]:
    state = _load_state(_prepare_validate_paths(validate_root).state_path)
    if state.active_duel is None:
        return set()
    return set(state.active_duel.task_names)


def _load_current_king(validate_root: Path) -> ValidatorSubmission:
    state = _load_state(_prepare_validate_paths(validate_root).state_path)
    if state.current_king is not None:
        return state.current_king
    if state.active_duel is not None:
        return state.active_duel.king
    raise RuntimeError("No current king found in validator state")


def _iter_candidate_task_names(
    config: RunConfig,
    *,
    king: ValidatorSubmission,
    only_missing_baseline: bool,
) -> list[str]:
    validate_root = config.workspace_root / "workspace" / "validate" / f"netuid-{config.validate_netuid}"
    priority = _priority_task_names(validate_root)
    tasks_root = config.workspace_root / "workspace" / "tasks"
    entries: list[tuple[tuple[int, int, int, str], str]] = []
    for task_dir in sorted(tasks_root.glob("validate-*")):
        if not task_dir.is_dir():
            continue
        if not _task_has_minimum_files(task_dir):
            continue
        if only_missing_baseline and _has_baseline(task_dir):
            continue
        name = task_dir.name
        has_baseline = _has_baseline(task_dir)
        has_reference_compare = _has_reference_compare(task_dir)
        has_current_king = _king_matches_current(task_dir, king)
        sort_key = (
            0 if name in priority else 1,
            0 if has_baseline and has_reference_compare and has_current_king else 1,
            0 if has_baseline or has_reference_compare or has_current_king else 1,
            name,
        )
        entries.append((sort_key, name))
    entries.sort(key=lambda item: item[0])
    return [name for _, name in entries]


def _pool_for_name(paths: any, pool_name: str) -> TaskPool:
    if pool_name == "retest":
        return TaskPool(paths.retest_pool_dir)
    return TaskPool(paths.pool_dir)


def _pool_json_path(paths: any, pool_name: str, task_name: str) -> Path:
    pool_dir = paths.retest_pool_dir if pool_name == "retest" else paths.pool_dir
    return pool_dir / f"{task_name}.json"


def _pool_task_fingerprint(task: PoolTask) -> str | None:
    return task_content_fingerprint(Path(task.task_root))


def _dedupe_sort_key(active_names: set[str], pool_name: str, task: PoolTask) -> tuple[int, int, int, str]:
    return (
        0 if task.task_name in active_names else 1,
        0 if pool_name == "main" else 1,
        int(task.creation_block),
        task.task_name,
    )


def _duplicate_pool_entries(
    *,
    paths: any,
    pools: dict[str, TaskPool],
    active_names: set[str],
) -> list[tuple[str, PoolTask, str, PoolTask]]:
    grouped: dict[str, list[tuple[str, PoolTask]]] = {}
    for pool_name, pool in pools.items():
        for task in pool.list_tasks():
            fingerprint = _pool_task_fingerprint(task)
            if fingerprint:
                grouped.setdefault(fingerprint, []).append((pool_name, task))

    duplicates: list[tuple[str, PoolTask, str, PoolTask]] = []
    for entries in grouped.values():
        if len(entries) <= 1:
            continue
        ordered = sorted(entries, key=lambda item: _dedupe_sort_key(active_names, item[0], item[1]))
        keep_pool, keep_task = ordered[0]
        duplicates.extend((remove_pool, remove_task, keep_pool, keep_task) for remove_pool, remove_task in ordered[1:])
    return sorted(
        duplicates,
        key=lambda item: (item[0], item[1].creation_block, item[1].task_name),
    )


def _dedupe_current_pools(*, config: RunConfig, dry_run: bool) -> int:
    validate_root = config.workspace_root / "workspace" / "validate" / f"netuid-{config.validate_netuid}"
    paths = _prepare_validate_paths(validate_root)
    active_names = _priority_task_names(validate_root)
    pools = {
        "main": TaskPool(paths.pool_dir),
        "retest": TaskPool(paths.retest_pool_dir),
    }
    duplicates = _duplicate_pool_entries(paths=paths, pools=pools, active_names=active_names)
    if not duplicates:
        print("dedupe pools: no duplicate task content found")
        return 0

    print(f"dedupe pools: duplicate entries={len(duplicates)} dry_run={dry_run}")
    removed = 0
    for pool_name, task, keep_pool, keep_task in duplicates:
        action = "would remove" if dry_run else "remove"
        print(f"{action} {pool_name}:{task.task_name} duplicate_of={keep_pool}:{keep_task.task_name}")
        if dry_run:
            continue
        _pool_json_path(paths, pool_name, task.task_name).unlink(missing_ok=True)
        removed += 1
    print(f"dedupe pools summary: removed={removed} remaining_main={pools['main'].size()} remaining_retest={pools['retest'].size()}")
    return 0


def _repair_one_task(
    *,
    task_name: str,
    config: RunConfig,
    king: ValidatorSubmission,
    pool: TaskPool,
    creation_block: int,
) -> tuple[str, str]:
    task_root = config.workspace_root / "workspace" / "tasks" / task_name
    if not _task_has_minimum_files(task_root):
        return task_name, "skip:incomplete"
    if _count_patch_lines(task_root / "task" / "reference.patch") < _MIN_PATCH_LINES:
        return task_name, "skip:patch_too_small"

    # King is the sole solve; no baseline pre-solve. Use the static
    # qualification budget (mirrors task_pool_manager).
    agent_timeout = _POOL_KING_QUALIFY_TIMEOUT_SECONDS

    if not _king_matches_current(task_root, king):
        _remove_solution_artifacts(task_name=task_name, solution_name="king", config=config)
        king_cfg = replace(_build_agent_config(config, king), agent_timeout=agent_timeout)
        try:
            king_result = solve_task_run(task_name=task_name, solution_name="king", config=king_cfg)
        except Exception as exc:
            _ensure_empty_solution(
                task_name=task_name,
                solution_name="king",
                config=config,
                reason=str(exc),
            )
            king_result = None
        if king_result is not None and king_result.exit_reason not in {"completed", "time_limit_exceeded"}:
            return task_name, f"skip:king_{king_result.exit_reason}"

    qualifies, skip_reason = _king_solve_qualifies_for_pool(task_name=task_name, config=config)
    if not qualifies:
        return task_name, f"skip:{skip_reason.replace(' ', '_')}"

    pool.add(
        PoolTask(
            task_name=task_name,
            task_root=str(task_root),
            creation_block=creation_block,
            cursor_elapsed=0.0,
            king_lines=0,
            king_similarity=0.0,
            baseline_lines=0,
            agent_timeout_seconds=agent_timeout,
            king_hotkey=king.hotkey,
            king_commit_sha=king.commit_sha,
        )
    )
    return task_name, "repaired"


def main() -> int:
    args = _parse_args()
    config = _build_config(args)
    if args.dedupe_current_pools:
        return _dedupe_current_pools(config=config, dry_run=args.dry_run)
    if not args.solver_model:
        raise SystemExit("--solver-model is required unless --dedupe-current-pools is used")
    validate_root = config.workspace_root / "workspace" / "validate" / f"netuid-{config.validate_netuid}"
    paths = _prepare_validate_paths(validate_root)
    pool = _pool_for_name(paths, args.pool)
    king = _load_current_king(validate_root)
    creation_block = max(0, int(king.commitment_block))

    candidates = _iter_candidate_task_names(
        config,
        king=king,
        only_missing_baseline=args.only_missing_baseline,
    )
    if args.exclude_pool:
        excluded_names = _pool_for_name(paths, args.exclude_pool).names()
        candidates = [name for name in candidates if name not in excluded_names]
    if args.limit is not None:
        candidates = candidates[: max(0, int(args.limit))]
    if not candidates:
        print("repair pool: nothing to do")
        return 0
    if args.target_count is not None and pool.size() >= args.target_count:
        print(f"repair pool[{args.pool}]: already at target count {args.target_count}")
        return 0

    print(
        f"repair pool[{args.pool}]: current king uid={king.uid} {king.repo_full_name}@{king.commit_sha} "
        f"candidates={len(candidates)} concurrency={max(1, args.concurrency)}"
    )

    counts: dict[str, int] = {}
    counts_lock = threading.Lock()

    def record(status: str) -> None:
        with counts_lock:
            counts[status] = counts.get(status, 0) + 1

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        in_flight: dict[Future[tuple[str, str]], str] = {}
        pending = iter(candidates)

        def submit_next() -> bool:
            if args.target_count is not None and pool.size() >= args.target_count:
                return False
            try:
                task_name = next(pending)
            except StopIteration:
                return False
            future = executor.submit(
                _repair_one_task,
                task_name=task_name,
                config=config,
                king=king,
                pool=pool,
                creation_block=creation_block,
            )
            in_flight[future] = task_name
            return True

        for _ in range(max(1, args.concurrency)):
            if not submit_next():
                break

        completed = 0
        total = len(candidates)
        while in_flight:
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                task_name = in_flight.pop(future)
                try:
                    _, status = future.result()
                except Exception as exc:
                    status = f"skip:worker_{type(exc).__name__}"
                completed += 1
                record(status)
                print(f"[{completed}/{total}] {task_name}: {status}")
                if args.target_count is not None and pool.size() >= args.target_count:
                    continue
                submit_next()

    print(f"repair pool[{args.pool}] summary:")
    for key in sorted(counts):
        print(f"  {key}={counts[key]}")
    print(f"  pool_size={pool.size()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
