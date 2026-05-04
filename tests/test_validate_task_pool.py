import tempfile
import unittest
from pathlib import Path

from validate import PoolTask, TaskPool


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


if __name__ == "__main__":
    unittest.main()
