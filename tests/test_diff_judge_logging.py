import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from config import RunConfig
from diff_judge_logging import configure_diff_judge_log, record_diff_judge_event
from validate import (
    DiffJudgeResult,
    ValidationRoundResult,
    _active_round_payload,
    _judge_round_diffs,
)


class DiffJudgeLoggingTest(unittest.TestCase):
    def test_configure_and_record_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = configure_diff_judge_log(root)
            self.assertEqual(path, root / "diff-judge.jsonl")
            record_diff_judge_event(
                phase="started",
                duel_id=42,
                task_name="task-a",
                solution="challenger-1-d42",
            )
            record_diff_judge_event(
                phase="finished",
                duel_id=42,
                task_name="task-a",
                solution="challenger-1-d42",
                outcome="success",
                total_elapsed_ms=1234.5,
                acquire_wait_ms=50.0,
                call_elapsed_ms=1180.0,
                attempts=1,
            )
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            started = json.loads(lines[0])
            finished = json.loads(lines[1])
            self.assertEqual(started["phase"], "started")
            self.assertEqual(started["duel_id"], 42)
            self.assertIn("ts", started)
            self.assertEqual(finished["phase"], "finished")
            self.assertEqual(finished["outcome"], "success")
            self.assertAlmostEqual(finished["total_elapsed_ms"], 1234.5)

    def test_active_round_payload_includes_judge_timing(self):
        round_result = ValidationRoundResult(
            task_name="task-a",
            winner="king",
            king_lines=10,
            challenger_lines=12,
            king_similarity_ratio=0.1,
            challenger_similarity_ratio=0.2,
            king_challenger_similarity=0.3,
            task_root="/tmp/task-a",
            king_compare_root="",
            challenger_compare_root="",
            llm_judge_total_elapsed_ms=2500.0,
            llm_judge_acquire_wait_ms=100.0,
            llm_judge_call_elapsed_ms=2300.0,
            llm_judge_attempts=2,
            llm_judge_outcome="success",
        )
        payload = _active_round_payload(round_result)
        self.assertEqual(payload["llm_judge_total_elapsed_ms"], 2500.0)
        self.assertEqual(payload["llm_judge_acquire_wait_ms"], 100.0)
        self.assertEqual(payload["llm_judge_call_elapsed_ms"], 2300.0)
        self.assertEqual(payload["llm_judge_attempts"], 2)
        self.assertEqual(payload["llm_judge_outcome"], "success")

    def test_judge_round_diffs_records_telemetry_on_success(self):
        def fake_complete_text(**_kwargs):
            return json.dumps(
                {
                    "winner": "candidate_a",
                    "candidate_a_score": 60,
                    "candidate_b_score": 40,
                    "rationale": "ok",
                },
            )

        task_paths = SimpleNamespace(
            task_txt_path=SimpleNamespace(read_text=lambda: "fix the bug"),
            reference_patch_path=SimpleNamespace(read_text=lambda: "diff --git a/ref b/ref"),
        )

        def fake_solution_paths(_task_paths, solution_name):
            return SimpleNamespace(
                solution_diff_path=SimpleNamespace(
                    read_text=lambda: f"diff --git a/{solution_name} b/{solution_name}",
                ),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            configure_diff_judge_log(Path(tmpdir))
            with (
                patch("validate.resolve_task_paths", return_value=task_paths),
                patch("validate.resolve_solution_paths", side_effect=fake_solution_paths),
                patch("validate.complete_text", side_effect=fake_complete_text),
            ):
                result = _judge_round_diffs(
                    task_name="task-judge",
                    challenger_solution_name="challenger-7-d3",
                    config=RunConfig(openrouter_api_key="test-key"),
                    duel_id=99,
                )

            self.assertEqual(result.outcome, "success")
            self.assertGreater(result.total_elapsed_ms, 0)
            self.assertGreater(result.call_elapsed_ms, 0)
            self.assertEqual(result.attempts, 1)

            log_path = Path(tmpdir) / "diff-judge.jsonl"
            events = [json.loads(line) for line in log_path.read_text().strip().splitlines()]
            phases = [event["phase"] for event in events]
            self.assertEqual(phases[0], "started")
            self.assertIn("attempt", phases)
            self.assertEqual(phases[-1], "finished")
            finished = events[-1]
            self.assertEqual(finished["duel_id"], 99)
            self.assertEqual(finished["outcome"], "success")
            self.assertGreater(finished["total_elapsed_ms"], 0)

    def test_judge_round_diffs_records_wrapper_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            configure_diff_judge_log(Path(tmpdir))
            with (
                patch("validate._DIFF_JUDGE_TOTAL_TIMEOUT_SECONDS", 0.01),
                patch(
                    "validate._judge_round_diffs_uncapped",
                    side_effect=lambda **_kwargs: (_ for _ in ()).throw(TimeoutError()),
                ),
            ):
                result = _judge_round_diffs(
                    task_name="task-judge",
                    challenger_solution_name="challenger-7-d3",
                    config=RunConfig(openrouter_api_key="test-key"),
                    duel_id=100,
                )

            self.assertEqual(result.outcome, "wrapper_timeout")
            log_path = Path(tmpdir) / "diff-judge.jsonl"
            events = [json.loads(line) for line in log_path.read_text().strip().splitlines()]
            self.assertEqual(events[-1]["phase"], "finished")
            self.assertEqual(events[-1]["outcome"], "wrapper_timeout")


if __name__ == "__main__":
    unittest.main()
