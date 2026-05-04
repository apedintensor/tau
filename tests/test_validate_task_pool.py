import tempfile
import unittest
from pathlib import Path

from config import RunConfig
from validate import PoolTask, TaskPool, TaskPoolRefreshBudget


class TaskPoolTest(unittest.TestCase):
    def test_take_returns_fastest_eligible_task(self):
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
