from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _git_diff(repo_path: str) -> str:
    proc = subprocess.run(
        ["git", "diff", "--binary", "--", "."],
        cwd=repo_path,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return proc.stdout or ""


def _mini_model_name(model: str) -> str:
    if "/" in model and not model.startswith(("openai/", "azure/", "openrouter/")):
        return f"openai/{model}"
    if "/" not in model:
        return f"openai/{model}"
    return model


def _mini_env(*, api_base: str, api_key: str) -> dict[str, str]:
    return os.environ | {
        "MSWEA_CONFIGURED": "true",
        "MSWEA_API_KEY": api_key,
        "MSWEA_COST_TRACKING": "ignore_errors",
        "NO_COLOR": "1",
        "OPENAI_API_KEY": api_key,
        "OPENAI_BASE_URL": api_base,
        "OPENAI_API_BASE": api_base,
        "PIP_PROGRESS_BAR": "off",
        "TQDM_DISABLE": "1",
    }


def _mini_command(*, issue: str, model: str, api_base: str, api_key: str, output_path: Path) -> list[str]:
    return [
        "mini",
        "-m",
        _mini_model_name(model),
        "-t",
        issue,
        "-y",
        "--exit-immediately",
        "-o",
        str(output_path),
        "-c",
        "mini.yaml",
        "-c",
        "model.model_class=litellm",
        "-c",
        f"model.model_kwargs.api_base={api_base}",
        "-c",
        f"model.model_kwargs.api_key={api_key}",
        "-c",
        "agent.cost_limit=0",
    ]


def _require_mini() -> None:
    if shutil.which("mini") is None:
        raise RuntimeError("The real mini-swe-agent CLI is missing. Install the official mini-swe-agent package.")


def solve(repo_path: str, issue: str, model: str, api_base: str, api_key: str) -> dict:
    _require_mini()
    with tempfile.TemporaryDirectory(prefix="tau-mini-swe-agent-") as tmp:
        output_path = Path(tmp) / "trajectory.json"
        proc = subprocess.run(
            _mini_command(
                issue=issue,
                model=model,
                api_base=api_base,
                api_key=api_key,
                output_path=output_path,
            ),
            cwd=repo_path,
            env=_mini_env(api_base=api_base, api_key=api_key),
            text=True,
            capture_output=True,
            timeout=None,
            check=False,
        )
    diff = _git_diff(repo_path)
    success = bool(diff.strip())
    return {
        "success": success,
        "message": f"mini exited {proc.returncode}; diff_bytes={len(diff.encode('utf-8'))}",
        "diff": diff,
    }
