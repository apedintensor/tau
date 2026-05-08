from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from config import RunConfig
from validate import DiffJudgeResult


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("manual_duel", ROOT / "scripts" / "manual_duel.py")
assert SPEC is not None
manual_duel = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(manual_duel)


class _Readable:
    def __init__(self, text: str) -> None:
        self.text = text

    def read_text(self) -> str:
        return self.text


class ManualDuelJudgeTest(unittest.TestCase):
    def test_judge_pair_uses_dual_judge_consensus(self):
        captured: dict[str, object] = {}

        def fake_consensus(**kwargs):
            captured.update(kwargs)
            return DiffJudgeResult(
                winner="challenger",
                king_score=0.2,
                challenger_score=0.9,
                rationale="dual consensus",
                model="judge-a,judge-b",
                models=["judge-a", "judge-b"],
                rounds=[{"round": 1, "model": "judge-a"}],
                consensus_status="agreed",
                consensus_round=1,
            )

        config = RunConfig(
            openrouter_api_key="key",
            diff_judge_models=("judge-a", "judge-b"),
        )
        task_paths = SimpleNamespace(
            task_txt_path=_Readable("fix the bug"),
            reference_patch_path=_Readable("+expected\n"),
        )

        with (
            patch.object(manual_duel, "resolve_task_paths", return_value=task_paths),
            patch.object(manual_duel, "_run_diff_judge_consensus", side_effect=fake_consensus),
        ):
            result = manual_duel._judge_pair(
                config=config,
                task_name="validate-1",
                base_patch="+base\n",
                challenger_patch="+challenger\n",
                challenger_timed_out=True,
            )

        self.assertEqual(result.winner, "challenger")
        self.assertEqual(captured["task_prompt"], "fix the bug")
        self.assertEqual(captured["reference_patch"], "+expected\n")
        self.assertEqual(captured["king_patch"], "+base\n")
        self.assertEqual(captured["challenger_patch"], "+challenger\n")
        self.assertEqual(captured["challenger_timed_out"], True)
        self.assertEqual(captured["models"], ("judge-a", "judge-b"))
        self.assertEqual(captured["openrouter_api_key"], "key")


if __name__ == "__main__":
    unittest.main()
