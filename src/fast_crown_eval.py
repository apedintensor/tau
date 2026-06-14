from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import swebench_crown_benchmark
import terminal_bench_crown_benchmark
from swebench_crown_benchmark import (
    current_king_from_state,
    queue_latest,
    read_json,
    utc_now,
    write_json,
)

log = logging.getLogger("swe-eval.fast_crown_eval")


def terminal_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        validate_root=args.validate_root,
        state_path=args.state_path,
        manifest=args.terminal_manifest,
        baseline=args.baseline,
        baseline_name="baseline",
        baseline_repo=None,
        baseline_ref="main",
        model=args.model,
        api_base=args.api_base,
        workers=args.terminal_workers,
        agent_timeout_seconds=args.agent_timeout_seconds,
        run_timeout_seconds=args.run_timeout_seconds,
        no_rebuild=args.no_rebuild,
        poll_interval_seconds=args.poll_interval_seconds,
        once=True,
        debug=args.debug,
    )


def swebench_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        validate_root=args.validate_root,
        state_path=args.state_path,
        manifest=args.swebench_manifest,
        baseline=args.baseline,
        pi_repo=args.pi_repo,
        pi_ref=args.pi_ref,
        mini_swe_agent_repo=args.mini_swe_agent_repo,
        mini_swe_agent_ref=args.mini_swe_agent_ref,
        model=args.model,
        provider_only=args.provider_only,
        workers=args.swebench_workers,
        poll_interval_seconds=args.poll_interval_seconds,
        once=True,
        skip_scoring=False,
        debug=args.debug,
        openrouter_api_key=None,
        api_base=None,
    )


def fast_eval_root(validate_root: Path) -> Path:
    return validate_root / "benchmarks" / "fast-100"


def completed_fast_eval(path: Path) -> bool:
    if not path.exists():
        return False
    payload = read_json(path)
    return isinstance(payload, dict) and payload.get("status") == "completed"


def should_run_fast_eval(*, king: dict[str, Any] | None, root: Path) -> bool:
    if not king or not king.get("commit_sha"):
        return False
    return not completed_fast_eval(root / str(king["commit_sha"]) / "comparison.json")


def run_fast_eval(*, king: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started_at = utc_now()
    root = fast_eval_root(args.validate_root)
    job_dir = root / str(king["commit_sha"])
    job_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        job_dir / "job.json",
        {
            "status": "running",
            "started_at": started_at,
            "updated_at": started_at,
            "king": king,
            "lanes": ["terminal_bench_core_fast_50", "swebench_verified_fast_50"],
            "baseline": args.baseline,
        },
    )
    terminal_result = terminal_bench_crown_benchmark.run_once(terminal_args(args))
    write_json(job_dir / "job.json", {"status": "running_swebench", "started_at": started_at, "updated_at": utc_now(), "king": king})
    swebench_result = swebench_crown_benchmark.run_once(swebench_args(args))
    payload = {
        "status": "completed",
        "benchmark": "fast_100_terminal50_swebench50",
        "king_commit_sha": str(king["commit_sha"]),
        "baseline": args.baseline,
        "model": args.model,
        "provider_only": args.provider_only,
        "workers": args.workers,
        "terminal_workers": args.terminal_workers,
        "swebench_workers": args.swebench_workers,
        "agent_timeout_seconds": args.agent_timeout_seconds,
        "run_timeout_seconds": args.run_timeout_seconds,
        "started_at": started_at,
        "finished_at": utc_now(),
        "lanes": {
            "terminal_bench_core_fast_50": terminal_result,
            "swebench_verified_fast_50": swebench_result,
        },
    }
    write_json(job_dir / "comparison.json", payload)
    write_json(root / "latest.json", payload)
    write_json(job_dir / "job.json", {**payload, "status": "completed"})
    return payload


def run_daemon(args: argparse.Namespace) -> None:
    pending: dict[str, Any] | None = None
    while True:
        try:
            king = current_king_from_state(args.state_path)
            if should_run_fast_eval(king=king, root=fast_eval_root(args.validate_root)):
                pending = queue_latest(pending, king)
            if pending:
                king_to_run = pending
                pending = None
                run_fast_eval(king=king_to_run, args=args)
        except Exception:
            log.exception("fast crown eval loop failed")
        time.sleep(args.poll_interval_seconds)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-root", type=Path, default=Path("workspace/validate/netuid-66"))
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--terminal-manifest", type=Path, default=Path("data/terminal_bench_core_fast_50_seed66.json"))
    parser.add_argument("--swebench-manifest", type=Path, default=Path("data/swebench_verified_sample_50_seed66.json"))
    parser.add_argument("--baseline", choices=("mini-swe-agent",), default="mini-swe-agent")
    parser.add_argument("--model", default="google/gemini-3.1-flash-lite")
    parser.add_argument("--provider-only", default="google-ai-studio")
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--terminal-workers", type=int)
    parser.add_argument("--swebench-workers", type=int)
    parser.add_argument("--agent-timeout-seconds", type=int, default=600)
    parser.add_argument("--run-timeout-seconds", type=int, default=600)
    parser.add_argument("--poll-interval-seconds", type=int, default=60)
    parser.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    parser.add_argument("--pi-repo", default="https://github.com/earendil-works/pi")
    parser.add_argument("--pi-ref", default="main")
    parser.add_argument("--mini-swe-agent-repo", default="https://github.com/SWE-agent/mini-swe-agent")
    parser.add_argument("--mini-swe-agent-ref", default="main")
    parser.add_argument("--no-rebuild", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    args.validate_root = args.validate_root.expanduser().resolve()
    args.terminal_manifest = args.terminal_manifest.expanduser().resolve()
    args.swebench_manifest = args.swebench_manifest.expanduser().resolve()
    args.state_path = (args.state_path or args.validate_root / "state.json").expanduser().resolve()
    args.terminal_workers = args.terminal_workers or min(args.workers, 10)
    args.swebench_workers = args.swebench_workers or args.workers
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    if args.once:
        king = current_king_from_state(args.state_path)
        if king is None:
            raise RuntimeError(f"No current king found in {args.state_path}")
        print(json.dumps(run_fast_eval(king=king, args=args), indent=2, sort_keys=True))
        return
    run_daemon(args)


if __name__ == "__main__":
    main()
