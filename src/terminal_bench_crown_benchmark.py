from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from swebench_crown_benchmark import (
    AgentIdentity,
    baseline_pin_path,
    checkout_repo,
    current_king_from_state,
    queue_latest,
    read_json,
    should_benchmark_king,
    utc_now,
    write_history_entry,
    write_json,
)

log = logging.getLogger("swe-eval.terminal_bench_crown_benchmark")

DEFAULT_DATASET = "terminal-bench-core==0.1.1"
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
DEFAULT_BASELINE = "terminus"
DEFAULT_WORKERS = 10
DEFAULT_POLL_INTERVAL_SECONDS = 60
BUILTIN_BASELINES = frozenset(("terminus", "codex", "aider"))
LOCAL_IMPORT_BASELINES = frozenset(("mini-swe-agent",))


@dataclass(frozen=True, slots=True)
class TerminalBenchManifest:
    benchmark: str
    dataset: str
    seed: int
    n_tasks: int | None
    task_ids: tuple[str, ...]
    manifest_hash: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["task_ids"] = list(self.task_ids)
        return payload


@dataclass(frozen=True, slots=True)
class TerminalBenchScore:
    resolved_count: int | None
    total_count: int | None
    pass_rate: float | None
    report_path: str | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_manifest(path: Path) -> TerminalBenchManifest:
    payload = read_json(path)
    task_ids = tuple(str(item) for item in payload.get("task_ids", ()))
    duplicates = sorted({item for item in task_ids if task_ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"manifest has duplicate task_ids: {', '.join(duplicates)}")
    expected_count = payload.get("count")
    if isinstance(expected_count, int) and task_ids and expected_count != len(task_ids):
        raise ValueError(f"manifest count={expected_count} but has {len(task_ids)} task ids")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return TerminalBenchManifest(
        benchmark=str(payload.get("benchmark") or "terminal_bench_core"),
        dataset=str(payload.get("dataset") or DEFAULT_DATASET),
        seed=int(payload.get("seed") or 0),
        n_tasks=int(payload["n_tasks"]) if payload.get("n_tasks") is not None else None,
        task_ids=task_ids,
        manifest_hash=digest,
    )


def run_daemon(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    benchmark_root = args.validate_root / "benchmarks" / "terminal-bench"
    pending: dict[str, Any] | None = None
    log.info("Starting Terminal-Bench crown benchmark daemon root=%s", benchmark_root)
    while True:
        try:
            king = current_king_from_state(args.state_path)
            if should_benchmark_king(king=king, benchmark_root=benchmark_root):
                pending = queue_latest(pending, king)
            if pending is not None:
                king_to_run = pending
                pending = None
                run_crown_benchmark(
                    king=king_to_run,
                    manifest=manifest,
                    args=args,
                    benchmark_root=benchmark_root,
                )
        except Exception:
            log.exception("Terminal-Bench crown benchmark loop failed")
        time.sleep(args.poll_interval_seconds)


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(args.manifest)
    king = current_king_from_state(args.state_path)
    if king is None:
        raise RuntimeError(f"No current king found in {args.state_path}")
    benchmark_root = args.validate_root / "benchmarks" / "terminal-bench"
    return run_crown_benchmark(king=king, manifest=manifest, args=args, benchmark_root=benchmark_root)


def run_crown_benchmark(
    *,
    king: dict[str, Any],
    manifest: TerminalBenchManifest,
    args: argparse.Namespace,
    benchmark_root: Path,
) -> dict[str, Any]:
    start = time.monotonic()
    started_at = utc_now()
    king_commit = str(king["commit_sha"])
    job_dir = benchmark_root / king_commit
    job_dir.mkdir(parents=True, exist_ok=True)
    write_json(job_dir / "manifest.json", manifest.to_dict())
    write_job(job_dir, status="running", king=king, manifest=manifest, started_at=started_at)
    try:
        king_agent = resolve_king_agent(king=king, cache_root=job_dir / "agents" / "king")
        baseline_agent = resolve_baseline_agent(args=args, benchmark_root=benchmark_root)
        cache_dir = terminal_baseline_cache_dir(
            benchmark_root=benchmark_root,
            baseline_agent=baseline_agent,
            manifest=manifest,
            model=args.model,
        )
        baseline_cached = restore_terminal_baseline(
            cache_dir=cache_dir,
            job_dir=job_dir,
            baseline_name=baseline_agent.name,
        )
        write_job(
            job_dir,
            status="running_baseline" if not baseline_cached else "running_king",
            king=king,
            manifest=manifest,
            started_at=started_at,
            agents=[king_agent.to_dict(), baseline_agent.to_dict()],
        )
        baseline_score = (
            read_terminal_score(job_dir / baseline_agent.name / "score_summary.json")
            if baseline_cached
            else run_terminal_bench_agent(
                agent=baseline_agent,
                manifest=manifest,
                job_dir=job_dir,
                args=args,
            )
        )
        if not baseline_cached:
            save_terminal_baseline(
                cache_dir=cache_dir,
                job_dir=job_dir,
                baseline_name=baseline_agent.name,
            )
        write_job(job_dir, status="running_king", king=king, manifest=manifest, started_at=started_at)
        king_score = run_terminal_bench_agent(
            agent=king_agent,
            manifest=manifest,
            job_dir=job_dir,
            args=args,
        )
        comparison = build_comparison(
            king=king,
            king_agent=king_agent,
            baseline_agent=baseline_agent,
            manifest=manifest,
            scores=(king_score, baseline_score),
            total_elapsed_seconds=time.monotonic() - start,
            started_at=started_at,
            model=args.model,
            baseline_cached=baseline_cached,
        )
        write_json(job_dir / "comparison.json", comparison)
        write_json(benchmark_root / "latest.json", comparison)
        write_history_entry(benchmark_root / "history.jsonl", comparison)
        merge_terminal_bench_into_dashboard(args.validate_root / "dashboard_data.json", comparison)
        write_job(job_dir, status="completed", king=king, manifest=manifest, started_at=started_at, comparison=comparison)
        return comparison
    except Exception as exc:
        error_payload = {
            "status": "failed",
            "benchmark": "terminal_bench_sample",
            "king_commit_sha": king_commit,
            "error": repr(exc),
            "finished_at": utc_now(),
            "king": king,
        }
        write_json(job_dir / "job.json", error_payload)
        merge_terminal_bench_into_dashboard(args.validate_root / "dashboard_data.json", error_payload)
        raise


def write_job(
    job_dir: Path,
    *,
    status: str,
    king: dict[str, Any],
    manifest: TerminalBenchManifest,
    started_at: str,
    agents: list[dict[str, Any]] | None = None,
    comparison: dict[str, Any] | None = None,
) -> None:
    payload = {
        "status": status,
        "started_at": started_at,
        "updated_at": utc_now(),
        "king": king,
        "manifest": manifest.to_dict(),
    }
    if agents is not None:
        payload["agents"] = agents
    if comparison is not None:
        payload["comparison"] = comparison
    write_json(job_dir / "job.json", payload)


def resolve_king_agent(*, king: dict[str, Any], cache_root: Path) -> AgentIdentity:
    repo_url = str(king.get("repo_url") or f"https://github.com/{king['repo_full_name']}.git")
    commit_sha = str(king["commit_sha"])
    repo_dir, resolved_sha = checkout_repo(repo_url=repo_url, ref=commit_sha, cache_root=cache_root)
    if not (repo_dir / "agent.py").exists():
        raise FileNotFoundError(f"king repo does not contain agent.py: {repo_dir}")
    return AgentIdentity("king", repo_url, resolved_sha, repo_dir / "agent.py", king)


def resolve_baseline_agent(*, args: argparse.Namespace, benchmark_root: Path) -> AgentIdentity:
    if args.baseline in BUILTIN_BASELINES or args.baseline in LOCAL_IMPORT_BASELINES:
        return AgentIdentity(
            name=args.baseline,
            repo_url=f"terminal-bench://baseline/{args.baseline}",
            commit_sha="builtin",
            agent_path=Path(args.baseline),
            source={"builtin": True},
        )
    pin_path = baseline_pin_path(benchmark_root, args.baseline_name)
    pin = read_json(pin_path) if pin_path.exists() else None
    repo_url = str(args.baseline_repo)
    ref = str(pin["commit_sha"] if isinstance(pin, dict) and pin.get("commit_sha") else args.baseline_ref)
    repo_dir, commit_sha = checkout_repo(repo_url=repo_url, ref=ref, cache_root=benchmark_root / "_baselines" / args.baseline_name / "repo")
    if not (repo_dir / "agent.py").exists():
        raise FileNotFoundError(f"baseline repo does not contain agent.py: {repo_dir}")
    if not pin_path.exists():
        write_json(
            pin_path,
            {
                "repo_url": repo_url,
                "requested_ref": args.baseline_ref,
                "commit_sha": commit_sha,
                "pinned_at": utc_now(),
            },
        )
    return AgentIdentity(args.baseline_name, repo_url, commit_sha, repo_dir / "agent.py", {"pin_path": str(pin_path)})


def terminal_bench_agent_command(agent: AgentIdentity) -> tuple[str, str | None]:
    if agent.name in BUILTIN_BASELINES:
        return ("--agent", agent.name)
    if agent.name == "mini-swe-agent":
        return ("--agent-import-path", "terminal_bench_mini_swe_agent:TauMiniSweAgent")
    return ("--agent-import-path", "terminal_bench_tau_agent:TauSubnet66Agent")


def terminal_bench_env(*, agent: AgentIdentity, args: argparse.Namespace) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH", "")]),
        "TAU_AGENT_REPO_URL": agent.repo_url,
        "TAU_AGENT_REF": agent.commit_sha,
        "TAU_MODEL": args.model,
        "TAU_API_BASE": args.api_base,
        "TAU_MAX_SECONDS": str(args.agent_timeout_seconds),
        "NO_COLOR": "1",
    }


def build_tb_command(
    *,
    agent: AgentIdentity,
    manifest: TerminalBenchManifest,
    output_dir: Path,
    model: str,
    workers: int,
    no_rebuild: bool = False,
) -> list[str]:
    agent_flag, agent_value = terminal_bench_agent_command(agent)
    command = [
        "tb",
        "run",
        "--dataset",
        manifest.dataset,
        agent_flag,
        str(agent_value),
        "--model",
        model,
        "--n-concurrent",
        str(workers),
    ]
    if no_rebuild:
        command.append("--no-rebuild")
    for task_id in manifest.task_ids:
        command.extend(["--task-id", task_id])
    if manifest.n_tasks is not None and not manifest.task_ids:
        command.extend(["--n-tasks", str(manifest.n_tasks)])
    command.extend(["--output-path", str(output_dir)])
    return command


def run_terminal_bench_agent(
    *,
    agent: AgentIdentity,
    manifest: TerminalBenchManifest,
    job_dir: Path,
    args: argparse.Namespace,
) -> TerminalBenchScore:
    ensure_terminal_bench_cli()
    agent_dir = job_dir / agent.name
    output_dir = agent_dir / "tb-run"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    agent_dir.mkdir(parents=True, exist_ok=True)
    command = build_tb_command(
        agent=agent,
        manifest=manifest,
        output_dir=output_dir,
        model=args.model,
        workers=args.workers,
        no_rebuild=getattr(args, "no_rebuild", False),
    )
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=job_dir,
        capture_output=True,
        text=True,
        timeout=args.run_timeout_seconds,
        check=False,
        env=terminal_bench_env(agent=agent, args=args),
    )
    elapsed = time.monotonic() - started
    write_json(agent_dir / "run.json", {"command": command, "returncode": result.returncode, "elapsed_seconds": elapsed})
    (agent_dir / "stdout.txt").write_text(result.stdout or "", encoding="utf-8")
    (agent_dir / "stderr.txt").write_text(result.stderr or "", encoding="utf-8")
    score = parse_terminal_bench_score(output_dir)
    harness_error = terminal_bench_harness_error(output_dir=output_dir, stderr=result.stderr or "")
    if harness_error:
        score = TerminalBenchScore(score.resolved_count, score.total_count, score.pass_rate, score.report_path, harness_error)
    if result.returncode != 0 and score.error is None:
        score = TerminalBenchScore(
            score.resolved_count,
            score.total_count,
            score.pass_rate,
            score.report_path,
            f"tb exited {result.returncode}",
        )
    write_json(agent_dir / "score_summary.json", score.to_dict())
    return score


def terminal_baseline_cache_key(
    *,
    baseline_agent: AgentIdentity,
    manifest: TerminalBenchManifest,
    model: str,
) -> str:
    payload = {
        "agent": baseline_agent.name,
        "commit_sha": baseline_agent.commit_sha,
        "manifest_hash": manifest.manifest_hash,
        "model": model,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def terminal_baseline_cache_dir(
    *,
    benchmark_root: Path,
    baseline_agent: AgentIdentity,
    manifest: TerminalBenchManifest,
    model: str,
) -> Path:
    return (
        benchmark_root
        / "_baselines"
        / baseline_agent.name
        / terminal_baseline_cache_key(
            baseline_agent=baseline_agent,
            manifest=manifest,
            model=model,
        )
    )


def terminal_baseline_complete(cache_dir: Path) -> bool:
    if not all((cache_dir / name).exists() for name in ("score_summary.json", "run.json", "stdout.txt", "stderr.txt")):
        return False
    score = read_terminal_score(cache_dir / "score_summary.json")
    return score.error is None and score.pass_rate is not None and score.total_count is not None


def copy_terminal_baseline_files(*, source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("score_summary.json", "run.json", "stdout.txt", "stderr.txt"):
        shutil.copy2(source_dir / name, target_dir / name)
    tb_run = source_dir / "tb-run"
    if tb_run.exists():
        shutil.copytree(tb_run, target_dir / "tb-run", dirs_exist_ok=True)


def restore_terminal_baseline(*, cache_dir: Path, job_dir: Path, baseline_name: str) -> bool:
    if not terminal_baseline_complete(cache_dir):
        return False
    copy_terminal_baseline_files(source_dir=cache_dir, target_dir=job_dir / baseline_name)
    return True


def save_terminal_baseline(*, cache_dir: Path, job_dir: Path, baseline_name: str) -> None:
    source_dir = job_dir / baseline_name
    if not (source_dir / "score_summary.json").exists():
        return
    score = read_terminal_score(source_dir / "score_summary.json")
    if score.error is not None or score.pass_rate is None or score.total_count is None:
        return
    copy_terminal_baseline_files(source_dir=source_dir, target_dir=cache_dir)


def read_terminal_score(path: Path) -> TerminalBenchScore:
    payload = read_json(path)
    return TerminalBenchScore(
        resolved_count=payload.get("resolved_count"),
        total_count=payload.get("total_count"),
        pass_rate=payload.get("pass_rate"),
        report_path=payload.get("report_path"),
        error=payload.get("error"),
    )


def ensure_terminal_bench_cli() -> None:
    if shutil.which("tb"):
        return
    raise FileNotFoundError("Terminal-Bench CLI `tb` is not installed. Install with `uv tool install terminal-bench`.")


def parse_terminal_bench_score(output_dir: Path) -> TerminalBenchScore:
    json_paths = sorted(output_dir.rglob("*.json")) if output_dir.exists() else []
    for path in json_paths:
        payload = read_json(path)
        score = score_from_payload(payload, report_path=path)
        if score is not None:
            return score
    jsonl_paths = sorted(output_dir.rglob("*.jsonl")) if output_dir.exists() else []
    for path in jsonl_paths:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        score = score_from_rows(rows, report_path=path)
        if score is not None:
            return score
    return TerminalBenchScore(None, None, None, None, "no Terminal-Bench result report found")


def terminal_bench_harness_error(*, output_dir: Path, stderr: str) -> str | None:
    if "all predefined address pools have been fully subnetted" in stderr:
        return "terminal-bench docker network pool exhausted"
    if "failed to create network" in stderr:
        return "terminal-bench docker network creation failed"
    if "Harness execution failed" in stderr:
        return "terminal-bench harness execution failed"
    results_path = newest_results_json(output_dir)
    if results_path is None:
        return None
    payload = read_json(results_path)
    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        return None
    dict_rows = [row for row in rows if isinstance(row, dict)]
    if not dict_rows:
        return None
    no_trials_started = all(row.get("trial_started_at") is None for row in dict_rows)
    all_unknown_agent_error = all(row.get("failure_mode") == "unknown_agent_error" for row in dict_rows)
    all_unresolved_unknown = all(row.get("is_resolved") is None for row in dict_rows)
    if no_trials_started and all_unknown_agent_error and all_unresolved_unknown:
        return "terminal-bench trials did not start"
    return None


def newest_results_json(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    paths = sorted(output_dir.rglob("results.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return paths[0] if paths else None


def score_from_payload(payload: Any, *, report_path: Path) -> TerminalBenchScore | None:
    if isinstance(payload, list):
        return score_from_rows(payload, report_path=report_path)
    if not isinstance(payload, dict):
        return None
    for key in ("results", "trials", "tasks"):
        rows = payload.get(key)
        if isinstance(rows, list):
            score = score_from_rows(rows, report_path=report_path)
            if score is not None:
                return score
    total = int(payload.get("total_count") or payload.get("total_trials") or payload.get("total") or 0)
    resolved = payload.get(
        "resolved_count",
        payload.get(
            "resolved_trials",
            payload.get("n_resolved", payload.get("passed_count", payload.get("passed"))),
        ),
    )
    if not total and isinstance(payload.get("n_unresolved"), int) and isinstance(resolved, int):
        total = int(resolved) + int(payload["n_unresolved"])
    if total and isinstance(resolved, int):
        return TerminalBenchScore(resolved, total, resolved / total, str(report_path))
    accuracy = payload.get("accuracy", payload.get("pass_at_1"))
    if isinstance(accuracy, (int, float)):
        return TerminalBenchScore(None, total or None, float(accuracy), str(report_path))
    return None


def score_from_rows(rows: list[Any], *, report_path: Path) -> TerminalBenchScore | None:
    task_rows = [row for row in rows if isinstance(row, dict)]
    if not task_rows:
        return None
    outcomes = [row_passed(row) for row in task_rows]
    if not any(outcome is not None for outcome in outcomes):
        return None
    resolved = sum(1 for outcome in outcomes if outcome is True)
    total = sum(1 for outcome in outcomes if outcome is not None)
    return TerminalBenchScore(resolved, total, resolved / total if total else None, str(report_path))


def row_passed(row: dict[str, Any]) -> bool | None:
    for key in ("resolved", "passed", "success", "is_resolved"):
        if isinstance(row.get(key), bool):
            return bool(row[key])
    if isinstance(row.get("score"), (int, float)):
        return float(row["score"]) >= 1.0
    status = str(row.get("status") or row.get("result") or "").lower()
    if status in {"passed", "pass", "success", "resolved"}:
        return True
    if status in {"failed", "fail", "error", "unresolved"}:
        return False
    return None


def build_comparison(
    *,
    king: dict[str, Any],
    king_agent: AgentIdentity,
    baseline_agent: AgentIdentity,
    manifest: TerminalBenchManifest,
    scores: tuple[TerminalBenchScore, TerminalBenchScore],
    total_elapsed_seconds: float,
    started_at: str,
    model: str,
    baseline_cached: bool = False,
) -> dict[str, Any]:
    king_score, baseline_score = scores
    delta = (
        king_score.pass_rate - baseline_score.pass_rate
        if king_score.pass_rate is not None and baseline_score.pass_rate is not None
        else None
    )
    return {
        "status": "completed",
        "benchmark": "terminal_bench_sample",
        "king_commit_sha": str(king["commit_sha"]),
        "king": king_agent.to_dict(),
        "baseline": baseline_agent.to_dict(),
        "baseline_name": baseline_agent.name,
        "baseline_cached": baseline_cached,
        "model": model,
        "manifest": manifest.to_dict(),
        "scores": {
            "king": king_score.to_dict(),
            baseline_agent.name: baseline_score.to_dict(),
            "baseline": baseline_score.to_dict(),
            "delta_pass_rate": delta,
        },
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_wall_seconds": total_elapsed_seconds,
    }


def merge_terminal_bench_into_dashboard(dashboard_path: Path, comparison: dict[str, Any]) -> None:
    if not dashboard_path.exists():
        payload: dict[str, Any] = {}
    else:
        payload = read_json(dashboard_path)
    benchmarks = payload.setdefault("benchmarks", {})
    terminal_bench = benchmarks.setdefault("terminal_bench", {})
    history = list(terminal_bench.get("history") or [])
    history = [comparison, *history]
    terminal_bench["latest"] = comparison
    terminal_bench["history"] = history[:20]
    write_json(dashboard_path, payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-root", type=Path, default=Path("workspace/validate/netuid-66"))
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/terminal_bench_sample_10_seed66.json"))
    parser.add_argument("--baseline", choices=(*sorted(BUILTIN_BASELINES | LOCAL_IMPORT_BASELINES), "agent-repo"), default=DEFAULT_BASELINE)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--baseline-repo")
    parser.add_argument("--baseline-ref", default="main")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--agent-timeout-seconds", type=int, default=600)
    parser.add_argument("--run-timeout-seconds", type=int, default=3600)
    parser.add_argument("--no-rebuild", action="store_true")
    parser.add_argument("--poll-interval-seconds", type=int, default=DEFAULT_POLL_INTERVAL_SECONDS)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    args.validate_root = args.validate_root.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    args.state_path = (args.state_path or args.validate_root / "state.json").expanduser().resolve()
    if args.baseline == "agent-repo" and not args.baseline_repo:
        parser.error("--baseline-repo is required with --baseline agent-repo")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    if args.once:
        comparison = run_once(args)
        print(json.dumps(comparison, indent=2, sort_keys=True))
        return
    run_daemon(args)


if __name__ == "__main__":
    main()
