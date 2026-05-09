import unittest

from validate import _should_refresh_chain_submissions


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

    def test_regular_refresh_waits_until_interval_blocks_pass(self):
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


if __name__ == "__main__":
    unittest.main()
