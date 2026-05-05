import json
import tempfile
import unittest
from pathlib import Path

from validate import (
    ValidatorState,
    ValidatorSubmission,
    _reconcile_dashboard_history_with_duels,
    _reconcile_state_with_duel_history,
)


class ValidatorStateRecoveryTest(unittest.TestCase):
    def test_reconcile_advances_duel_id_and_removes_completed_queue_entry(self):
        completed = _submission(
            hotkey="5CompletedHotkey",
            uid=210,
            commitment="github-pr:unarbos/ninja#75@" + "a" * 40,
            block=123,
        )
        pending = _submission(
            hotkey="5PendingHotkey",
            uid=97,
            commitment="github-pr:unarbos/ninja#76@" + "b" * 40,
            block=124,
        )
        state = ValidatorState(
            queue=[completed, pending],
            next_duel_index=3990,
            seen_hotkeys=[],
            locked_commitments={},
            commitment_blocks_by_hotkey={},
        )

        with tempfile.TemporaryDirectory() as tmp:
            duels_dir = Path(tmp)
            (duels_dir / "003990.json").write_text(
                json.dumps(
                    {
                        "duel_id": 3990,
                        "challenger": completed.to_dict(),
                    }
                )
                + "\n"
            )

            changed = _reconcile_state_with_duel_history(state, duels_dir)

        self.assertTrue(changed)
        self.assertEqual(state.next_duel_index, 3991)
        self.assertEqual([s.hotkey for s in state.queue], [pending.hotkey])
        self.assertIn(completed.hotkey, state.seen_hotkeys)
        self.assertEqual(state.locked_commitments[completed.hotkey], completed.commitment)
        self.assertEqual(state.commitment_blocks_by_hotkey[completed.hotkey], completed.commitment_block)

    def test_reconcile_dashboard_history_appends_missing_local_duels(self):
        existing = {"duel_id": 3989, "wins": 1, "losses": 0}
        challenger = _submission(
            hotkey="5CompletedHotkey",
            uid=210,
            commitment="github-pr:unarbos/ninja#75@" + "a" * 40,
            block=123,
        )

        with tempfile.TemporaryDirectory() as tmp:
            duels_dir = Path(tmp)
            (duels_dir / "003990.json").write_text(
                json.dumps(
                    {
                        "duel_id": 3990,
                        "started_at": "2026-05-05T00:00:00+00:00",
                        "finished_at": "2026-05-05T00:01:00+00:00",
                        "king_before": challenger.to_dict(),
                        "challenger": challenger.to_dict(),
                        "king_after": challenger.to_dict(),
                        "rounds": [],
                        "wins": 0,
                        "losses": 5,
                        "ties": 0,
                        "king_replaced": False,
                    }
                )
                + "\n"
            )
            history = [existing]

            changed = _reconcile_dashboard_history_with_duels(history, duels_dir)

        self.assertTrue(changed)
        self.assertEqual([entry["duel_id"] for entry in history], [3989, 3990])
        self.assertEqual(history[0], existing)
        self.assertEqual(history[1]["challenger_hotkey"], challenger.hotkey)
        self.assertEqual(history[1]["losses"], 5)


def _submission(*, hotkey: str, uid: int, commitment: str, block: int) -> ValidatorSubmission:
    return ValidatorSubmission(
        hotkey=hotkey,
        uid=uid,
        repo_full_name="miner/ninja",
        repo_url="https://github.com/miner/ninja.git",
        commit_sha=commitment.rsplit("@", 1)[-1],
        commitment=commitment,
        commitment_block=block,
        source="github_pr",
        pr_number=7,
        pr_url="https://github.com/unarbos/ninja/pull/7",
        base_repo_full_name="unarbos/ninja",
        base_ref="main",
    )


if __name__ == "__main__":
    unittest.main()
