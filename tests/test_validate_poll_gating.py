import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from config import RunConfig
from private_submission import touch_private_submission_queue_wakeup
from validate import (
    _POOL_FILLER_RATE_LIMIT_BACKOFF_BUFFER_SECONDS,
    _POOL_FILLER_RATE_LIMIT_BACKOFF_SECONDS,
    _github_rate_limit_backoff_seconds,
    _pool_gate_sleep_seconds,
    _remaining_poll_sleep_seconds,
    _should_refresh_chain_submissions,
    _should_refresh_private_submissions,
    _sleep_until_poll_or_private_submission_wakeup,
)


class ValidatePollGatingTest(unittest.TestCase):
    def test_force_always_refreshes(self):
        self.assertTrue(
            _should_refresh_chain_submissions(
                force=True,
                current_block=100,
                last_refresh_block=99,
                interval_blocks=360,
            )
        )

    def test_first_refresh_without_history_runs_immediately(self):
        self.assertTrue(
            _should_refresh_chain_submissions(
                force=False,
                current_block=100,
                last_refresh_block=None,
                interval_blocks=360,
            )
        )

    def test_regular_refresh_waits_for_interval_blocks(self):
        self.assertFalse(
            _should_refresh_chain_submissions(
                force=False,
                current_block=150,
                last_refresh_block=100,
                interval_blocks=360,
            )
        )
        self.assertTrue(
            _should_refresh_chain_submissions(
                force=False,
                current_block=460,
                last_refresh_block=100,
                interval_blocks=360,
            )
        )

    def test_private_submission_refresh_runs_every_ten_minutes(self):
        config = RunConfig(validate_submission_refresh_interval_seconds=600)
        self.assertTrue(
            _should_refresh_private_submissions(
                config=config,
                force=False,
                last_refresh_at=None,
            )
        )
        self.assertFalse(
            _should_refresh_private_submissions(
                config=config,
                force=False,
                last_refresh_at=1000.0,
                now=1500.0,
            )
        )
        self.assertTrue(
            _should_refresh_private_submissions(
                config=config,
                force=False,
                last_refresh_at=1000.0,
                now=1600.0,
            )
        )
        self.assertTrue(
            _should_refresh_private_submissions(
                config=config,
                force=True,
                last_refresh_at=1000.0,
                now=1001.0,
            )
        )

    def test_poll_sleep_counts_work_done_during_loop(self):
        self.assertEqual(
            _remaining_poll_sleep_seconds(started_at=100.0, interval_seconds=600, now=250.0),
            450.0,
        )
        self.assertEqual(
            _remaining_poll_sleep_seconds(started_at=100.0, interval_seconds=600, now=700.0),
            0.0,
        )

    def test_pool_gate_sleep_caps_queued_work_only(self):
        self.assertEqual(
            _pool_gate_sleep_seconds(started_at=100.0, interval_seconds=600, queued=True, now=250.0),
            15.0,
        )
        self.assertEqual(
            _pool_gate_sleep_seconds(started_at=100.0, interval_seconds=600, queued=False, now=250.0),
            450.0,
        )

    def test_sleep_returns_early_on_private_submission_wakeup(self):
        with tempfile.TemporaryDirectory() as td:
            config = RunConfig(
                workspace_root=Path(td),
                validate_private_submission_root=Path(td) / "private-submissions",
            )

            touch_private_submission_queue_wakeup(root=config.validate_private_submission_root)

            def wake_after_first_sleep(seconds):
                touch_private_submission_queue_wakeup(root=config.validate_private_submission_root)

            with unittest.mock.patch("validate.time.sleep", side_effect=wake_after_first_sleep):
                self.assertTrue(
                    _sleep_until_poll_or_private_submission_wakeup(
                        config=config,
                        seconds=600,
                        last_seen_mtime=0.0,
                    )
                )

    def test_github_rate_limit_backoff_prefers_retry_after_with_floor(self):
        response = Mock(headers={"retry-after": "45"})

        self.assertEqual(
            _github_rate_limit_backoff_seconds(response),
            _POOL_FILLER_RATE_LIMIT_BACKOFF_SECONDS,
        )

    def test_github_rate_limit_backoff_uses_core_reset_when_remaining_zero(self):
        response = Mock(headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1700000900"})

        self.assertEqual(
            _github_rate_limit_backoff_seconds(response, now=1700000000),
            900 + _POOL_FILLER_RATE_LIMIT_BACKOFF_BUFFER_SECONDS,
        )


if __name__ == "__main__":
    unittest.main()
