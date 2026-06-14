from __future__ import annotations

import os
import shlex
from pathlib import Path

try:
    from terminal_bench.agents.installed_agents.abstract_installed_agent import (
        AbstractInstalledAgent,
    )
except Exception:  # pragma: no cover - tests can import without terminal-bench installed.
    class AbstractInstalledAgent:  # type: ignore[no-redef]
        pass

try:
    from terminal_bench.terminal.models import TerminalCommand
except Exception:  # pragma: no cover - compatibility with absent/older terminal-bench builds.
    try:
        from terminal_bench.harness_models import TerminalCommand
    except Exception:
        class TerminalCommand:  # type: ignore[no-redef]
            def __init__(self, command: str, max_timeout_sec: float, block: bool):
                self.command = command
                self.max_timeout_sec = max_timeout_sec
                self.block = block


def _env_value(name: str, default: str = "") -> str:
    return str(os.environ.get(name) or default)


def _forwarded_env(names: tuple[str, ...]) -> dict[str, str]:
    return {name: str(os.environ[name]) for name in names if os.environ.get(name)}


def tau_terminal_bench_env() -> dict[str, str]:
    return {
        **_forwarded_env(
            (
                "OPENROUTER_API_KEY",
                "ANTHROPIC_API_KEY",
                "OPENAI_API_KEY",
                "GEMINI_API_KEY",
            ),
        ),
        "TAU_AGENT_REPO_URL": _env_value("TAU_AGENT_REPO_URL"),
        "TAU_AGENT_REF": _env_value("TAU_AGENT_REF"),
        "TAU_MODEL": _env_value("TAU_MODEL", "google/gemini-3.1-flash-lite"),
        "TAU_API_BASE": _env_value("TAU_API_BASE", "https://openrouter.ai/api/v1"),
        "TAU_MAX_SECONDS": _env_value("TAU_MAX_SECONDS", "600"),
        "NO_COLOR": "1",
    }


def setup_script_path() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "terminal_bench_tau_setup.sh"


def run_command(instruction: str) -> str:
    return "python /installed-agent/run_tau_agent.py " + shlex.quote(instruction)


class TauSubnet66Agent(AbstractInstalledAgent):
    @staticmethod
    def name() -> str:
        return "tau-subnet66-agent"

    @property
    def _env(self) -> dict[str, str]:
        return tau_terminal_bench_env()

    @property
    def _install_agent_script_path(self) -> Path:
        return setup_script_path()

    def _run_agent_commands(self, instruction: str) -> list[TerminalCommand]:
        return [
            TerminalCommand(
                command=run_command(instruction),
                max_timeout_sec=float(_env_value("TAU_MAX_SECONDS", "600")),
                block=True,
            ),
        ]
