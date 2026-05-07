import tempfile
import unittest
from pathlib import Path

import validate
from config import RunConfig
from validate import PoolTask, TaskPool, TaskPoolRefreshBudget, _prepare_validate_paths


class TaskPoolTest(unittest.TestCase):
    def tearDown(self):
        with validate._POOL_GENERATION_BACKOFF_LOCK:
            validate._pool_generation_backoff_until = 0.0

    def test_prepare_validate_paths_creates_primary_and_retest_pools(self):
        with tempfile.TemporaryDirectory() as td:
            paths = _prepare_validate_paths(Path(td))

            self.assertTrue(paths.pool_dir.exists())
            self.assertTrue(paths.retest_pool_dir.exists())
            self.assertNotEqual(paths.pool_dir, paths.retest_pool_dir)

    def test_take_returns_fastest_cached_task(self):
        with tempfile.TemporaryDirectory() as td:
            pool = TaskPool(Path(td))
            pool.add(
                PoolTask(
                    task_name="slow",
                    task_root="/tmp/slow",
                    creation_block=20,
                    cursor_elapsed=300.0,
                    king_lines=1,
                    king_similarity=0.1,
                    baseline_lines=1,
                )
            )
            pool.add(
                PoolTask(
                    task_name="fast",
                    task_root="/tmp/fast",
                    creation_block=20,
                    cursor_elapsed=20.0,
                    king_lines=1,
                    king_similarity=0.1,
                    baseline_lines=1,
                )
            )

            task = pool.take(min_block=10)

        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task.task_name, "fast")

    def test_take_reuses_cached_task_older_than_min_block(self):
        with tempfile.TemporaryDirectory() as td:
            pool = TaskPool(Path(td))
            pool.add(
                PoolTask(
                    task_name="cached",
                    task_root="/tmp/cached",
                    creation_block=20,
                    cursor_elapsed=20.0,
                    king_lines=1,
                    king_similarity=0.1,
                    baseline_lines=1,
                )
            )

            task = pool.take(min_block=100)

        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task.task_name, "cached")

    def test_take_respects_exclude_when_sorting_by_speed(self):
        with tempfile.TemporaryDirectory() as td:
            pool = TaskPool(Path(td))
            pool.add(
                PoolTask(
                    task_name="fast",
                    task_root="/tmp/fast",
                    creation_block=20,
                    cursor_elapsed=20.0,
                    king_lines=1,
                    king_similarity=0.1,
                    baseline_lines=1,
                )
            )
            pool.add(
                PoolTask(
                    task_name="slow",
                    task_root="/tmp/slow",
                    creation_block=20,
                    cursor_elapsed=300.0,
                    king_lines=1,
                    king_similarity=0.1,
                    baseline_lines=1,
                )
            )

            task = pool.take(min_block=10, exclude={"fast"})

        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task.task_name, "slow")

    def test_duel_agent_timeout_matches_stored_king_timeout(self):
        task = PoolTask(
            task_name="cached",
            task_root="/tmp/cached",
            creation_block=20,
            cursor_elapsed=10.0,
            king_lines=1,
            king_similarity=0.1,
            baseline_lines=1,
            agent_timeout_seconds=321,
        )

        loaded = PoolTask.from_dict(task.to_dict())

        self.assertEqual(validate._duel_agent_timeout(loaded), 321)

    def test_duel_agent_timeout_does_not_floor_stored_king_timeout(self):
        task = PoolTask(
            task_name="fast",
            task_root="/tmp/fast",
            creation_block=20,
            cursor_elapsed=10.0,
            king_lines=1,
            king_similarity=0.1,
            baseline_lines=1,
            agent_timeout_seconds=120,
        )

        self.assertEqual(validate._duel_agent_timeout(task), 120)

    def test_duel_task_submission_order_spreads_timeout_budgets(self):
        tasks = [
            PoolTask(
                task_name=f"task-{idx:02d}",
                task_root=f"/tmp/task-{idx:02d}",
                creation_block=20,
                cursor_elapsed=float(idx),
                king_lines=1,
                king_similarity=0.1,
                baseline_lines=1,
                agent_timeout_seconds=idx * 100,
            )
            for idx in range(1, 11)
        ]

        ordered = validate._order_duel_tasks_for_submission(tasks)

        self.assertEqual({task.task_name for task in ordered}, {task.task_name for task in tasks})
        self.assertEqual(
            [task.task_name for task in ordered[:5]],
            ["task-01", "task-03", "task-05", "task-07", "task-09"],
        )

    def test_legacy_pool_task_backfills_agent_timeout(self):
        loaded = PoolTask.from_dict(
            {
                "task_name": "legacy",
                "task_root": "/tmp/legacy",
                "creation_block": 20,
                "cursor_elapsed": 50.0,
                "king_lines": 1,
                "king_similarity": 0.1,
                "baseline_lines": 1,
            }
        )

        self.assertEqual(loaded.agent_timeout_seconds, 300)
        self.assertEqual(validate._duel_agent_timeout(loaded), 300)

    def test_pool_generation_backs_off_on_github_rate_limit(self):
        self.assertTrue(
            validate._is_github_rate_limit_error(
                RuntimeError("gh: API rate limit exceeded for user ID 123 (HTTP 403)")
            )
        )
        self.assertTrue(
            validate._is_github_rate_limit_error(
                RuntimeError("GitHub PR fetch failed for unarbos/ninja#360: HTTP 403")
            )
        )
        self.assertTrue(
            validate._is_github_rate_limit_error(
                RuntimeError(
                    "Client error '429 too many requests' for url "
                    "'https://api.github.com/events?page=1&per_page=30'"
                )
            )
        )
        self.assertFalse(validate._is_github_rate_limit_error(RuntimeError("docker failed")))

        validate._note_pool_generation_rate_limit("primary")

        self.assertGreater(validate._pool_generation_backoff_remaining(), 0.0)

    def test_missing_runtime_secrets_require_openrouter_key(self):
        self.assertEqual(
            validate._missing_runtime_secrets(RunConfig(openrouter_api_key=None)),
            ["OPENROUTER_API_KEY"],
        )
        self.assertEqual(
            validate._missing_runtime_secrets(RunConfig(openrouter_api_key="set")),
            [],
        )

    def test_zero_scored_duel_reason_includes_sample_errors(self):
        reason = validate._zero_scored_duel_reason(
            4101,
            [
                validate.ValidationRoundResult(
                    task_name="task-a",
                    winner="error",
                    king_lines=0,
                    challenger_lines=0,
                    king_similarity_ratio=0.0,
                    challenger_similarity_ratio=0.0,
                    king_challenger_similarity=0.0,
                    task_root="/tmp/task-a",
                    king_compare_root="",
                    challenger_compare_root="",
                    error="OPENROUTER_API_KEY is not set",
                )
            ],
        )

        self.assertIn("zero scored rounds", reason)
        self.assertIn("OPENROUTER_API_KEY is not set", reason)

    def test_refresh_budget_allows_bounded_hourly_batch(self):
        config = RunConfig(
            validate_task_pool_refresh_count=2,
            validate_task_pool_refresh_interval_seconds=3600,
        )
        budget = TaskPoolRefreshBudget()

        claimed, started = budget.claim(config=config)
        self.assertFalse(claimed)
        self.assertFalse(started)

        budget._next_refresh_at = 0

        claimed, started = budget.claim(config=config)
        self.assertTrue(claimed)
        self.assertTrue(started)

        claimed, started = budget.claim(config=config)
        self.assertTrue(claimed)
        self.assertFalse(started)

        claimed, started = budget.claim(config=config)
        self.assertFalse(claimed)
        self.assertFalse(started)

        self.assertFalse(budget.finish(config=config, success=True))
        self.assertTrue(budget.finish(config=config, success=True))

        claimed, started = budget.claim(config=config)
        self.assertFalse(claimed)
        self.assertFalse(started)

    def test_refresh_budget_retries_failed_replacement(self):
        config = RunConfig(
            validate_task_pool_refresh_count=1,
            validate_task_pool_refresh_interval_seconds=3600,
        )
        budget = TaskPoolRefreshBudget()
        budget._next_refresh_at = 0

        claimed, _ = budget.claim(config=config)
        self.assertTrue(claimed)
        self.assertFalse(budget.finish(config=config, success=False))

        claimed, started = budget.claim(config=config)
        self.assertTrue(claimed)
        self.assertFalse(started)


if __name__ == "__main__":
    unittest.main()
