#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from datasets import load_dataset

DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    instance_id: str
    repo: str
    elapsed_seconds: float
    cost: float | None
    success: bool
    exit_reason: str
    patch_bytes: int
    error: str | None


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = load_agent(args.agent)
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required unless --api-key is supplied.")

    instances = list(load_dataset(DATASET_NAME, split=args.split).select(range(args.offset, args.offset + args.count)))
    predictions_path = output_dir / "predictions.jsonl"
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                run_instance,
                agent=agent,
                instance=instance,
                output_dir=output_dir,
                model=args.model,
                api_base=args.api_base,
                api_key=api_key,
            )
            for instance in instances
        ]
        for future in as_completed(futures):
            result, prediction = future.result()
            append_jsonl(results_path, asdict(result))
            append_jsonl(predictions_path, prediction)
            print(
                json.dumps(
                    {
                        "instance_id": result.instance_id,
                        "elapsed_seconds": round(result.elapsed_seconds, 3),
                        "cost": result.cost,
                        "success": result.success,
                        "exit_reason": result.exit_reason,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    results = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    summary = summarize_results(results, total_elapsed_seconds=time.monotonic() - start)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run current king on a SWE-bench Verified sample.")
    parser.add_argument("--agent", required=True, type=Path, help="Path to king agent.py.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for repos and JSONL outputs.")
    parser.add_argument("--count", type=int, default=50, help="Number of SWE-bench Verified instances.")
    parser.add_argument("--offset", type=int, default=0, help="Starting dataset offset.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent agent workers.")
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument("--model", default=os.environ.get("SOLVER_MODEL") or os.environ.get("BASELINE_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--api-base", default=os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_API_BASE)
    parser.add_argument("--api-key", help="API key override. Defaults to OPENROUTER_API_KEY.")
    return parser.parse_args()


def load_agent(agent_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("calibration_agent", agent_path.expanduser().resolve())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent module from {agent_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "solve"):
        raise RuntimeError(f"{agent_path} does not define solve(...)")
    return module


def run_instance(
    *,
    agent: ModuleType,
    instance: dict[str, Any],
    output_dir: Path,
    model: str,
    api_base: str,
    api_key: str,
) -> tuple[CalibrationResult, dict[str, str]]:
    instance_id = str(instance["instance_id"])
    repo_name = str(instance["repo"])
    repo_dir = output_dir / "repos" / instance_id
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    error: str | None = None
    payload: dict[str, Any] = {}

    try:
        prepare_repo(repo_name=repo_name, base_commit=str(instance["base_commit"]), repo_dir=repo_dir)
        payload = agent.solve(
            repo_path=str(repo_dir),
            issue=str(instance["problem_statement"]),
            model=model,
            api_base=api_base,
            api_key=api_key,
        )
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)

    elapsed = time.monotonic() - start
    patch = str(payload.get("patch") or "") if isinstance(payload, dict) else ""
    logs = str(payload.get("logs") or "") if isinstance(payload, dict) else ""
    (logs_dir / f"{instance_id}.log").write_text(logs + ("\n" if logs else ""), encoding="utf-8")
    result = CalibrationResult(
        instance_id=instance_id,
        repo=repo_name,
        elapsed_seconds=elapsed,
        cost=payload.get("cost") if isinstance(payload.get("cost"), (int, float)) else None,
        success=bool(payload.get("success")) if isinstance(payload, dict) else False,
        exit_reason="completed" if error is None else "error",
        patch_bytes=len(patch.encode("utf-8")),
        error=error,
    )
    prediction = {
        "instance_id": instance_id,
        "model_name_or_path": "tau-current-king",
        "model_patch": patch,
    }
    return result, prediction


def prepare_repo(*, repo_name: str, base_commit: str, repo_dir: Path) -> None:
    if repo_dir.exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = repo_dir.with_name(f"{repo_dir.name}.tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    run(["git", "clone", "--quiet", "--no-tags", f"https://github.com/{repo_name}.git", str(temp_dir)])
    run(["git", "checkout", "--quiet", base_commit], cwd=temp_dir)
    temp_dir.rename(repo_dir)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def summarize_results(results: list[dict[str, Any]], *, total_elapsed_seconds: float) -> dict[str, Any]:
    elapsed = sorted(float(row["elapsed_seconds"]) for row in results)
    costs = [float(row["cost"]) for row in results if isinstance(row.get("cost"), (int, float))]
    return {
        "count": len(results),
        "completed": sum(1 for row in results if row.get("exit_reason") == "completed"),
        "errors": sum(1 for row in results if row.get("exit_reason") != "completed"),
        "successes": sum(1 for row in results if row.get("success")),
        "total_elapsed_seconds": total_elapsed_seconds,
        "sum_task_elapsed_seconds": sum(elapsed),
        "min_task_seconds": elapsed[0] if elapsed else None,
        "median_task_seconds": percentile(elapsed, 0.5),
        "p90_task_seconds": percentile(elapsed, 0.9),
        "max_task_seconds": elapsed[-1] if elapsed else None,
        "total_cost": sum(costs) if costs else None,
        "mean_cost": (sum(costs) / len(costs)) if costs else None,
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    index = min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))
    return values[index]


if __name__ == "__main__":
    main()
