import json
import tempfile
import unittest
from pathlib import Path

from validate import (
    ActiveDuelLease,
    RunConfig,
    ValidationRoundResult,
    ValidatorState,
    ValidatorSubmission,
    _checkpoint_active_duel,
    _reconcile_dashboard_history_with_duels,
    _reconcile_state_with_duel_history,
    _recover_active_duel_after_restart,
    _start_active_duel,
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

    def test_active_duel_lease_round_trips_through_state(self):
        king = _submission(
            hotkey="5KingHotkey",
            uid=11,
            commitment="github-pr:unarbos/ninja#11@" + "a" * 40,
            block=111,
        )
        challenger = _submission(
            hotkey="5ChallengerHotkey",
            uid=12,
            commitment="github-pr:unarbos/ninja#12@" + "b" * 40,
            block=112,
        )
        round_result = _round(task_name="validate-000001", winner="challenger")
        state = ValidatorState(
            current_king=king,
            active_duel=ActiveDuelLease(
                duel_id=77,
                started_at="2026-05-06T00:00:00+00:00",
                king=king,
                challenger=challenger,
                task_names=["validate-000001", "validate-000002"],
                rounds=[round_result],
                status="running",
                updated_at="2026-05-06T00:01:00+00:00",
            ),
        )

        restored = ValidatorState.from_dict(state.to_dict())

        self.assertIsNotNone(restored.active_duel)
        assert restored.active_duel is not None
        self.assertEqual(restored.active_duel.duel_id, 77)
        self.assertEqual(restored.active_duel.challenger.hotkey, challenger.hotkey)
        self.assertEqual(restored.active_duel.task_names, ["validate-000001", "validate-000002"])
        self.assertEqual(restored.active_duel.rounds[0].winner, "challenger")
        self.assertIn(challenger.hotkey, restored.seen_hotkeys)

    def test_checkpoint_active_duel_updates_tasks_rounds_and_status(self):
        king = _submission(
            hotkey="5KingHotkey",
            uid=11,
            commitment="github-pr:unarbos/ninja#11@" + "a" * 40,
            block=111,
        )
        challenger = _submission(
            hotkey="5ChallengerHotkey",
            uid=12,
            commitment="github-pr:unarbos/ninja#12@" + "b" * 40,
            block=112,
        )
        state = ValidatorState()
        _start_active_duel(state, duel_id=81, king=king, challenger=challenger)
        round_result = _round(task_name="validate-000003", winner="king")

        changed = _checkpoint_active_duel(
            state,
            duel_id=81,
            task_names=["validate-000003"],
            rounds=[round_result],
            status="draining",
        )

        self.assertTrue(changed)
        self.assertIsNotNone(state.active_duel)
        assert state.active_duel is not None
        self.assertEqual(state.active_duel.task_names, ["validate-000003"])
        self.assertEqual(state.active_duel.rounds, [round_result])
        self.assertEqual(state.active_duel.status, "draining")
        self.assertFalse(
            _checkpoint_active_duel(
                state,
                duel_id=82,
                task_names=["validate-000004"],
            )
        )

    def test_recover_active_duel_requeues_interrupted_challenger_at_front(self):
        king = _submission(
            hotkey="5KingHotkey",
            uid=11,
            commitment="github-pr:unarbos/ninja#11@" + "a" * 40,
            block=111,
        )
        challenger = _submission(
            hotkey="5ChallengerHotkey",
            uid=12,
            commitment="github-pr:unarbos/ninja#12@" + "b" * 40,
            block=112,
        )
        pending = _submission(
            hotkey="5PendingHotkey",
            uid=13,
            commitment="github-pr:unarbos/ninja#13@" + "c" * 40,
            block=113,
        )
        state = ValidatorState(
            current_king=king,
            queue=[pending],
            active_duel=ActiveDuelLease(
                duel_id=90,
                started_at="2026-05-06T00:00:00+00:00",
                king=king,
                challenger=challenger,
                rounds=[_round(task_name="validate-000001", winner="challenger")],
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            changed = _recover_active_duel_after_restart(
                config=RunConfig(validate_github_pr_only=True),
                state=state,
                duels_dir=Path(tmp),
            )

        self.assertTrue(changed)
        self.assertIsNone(state.active_duel)
        self.assertEqual([s.hotkey for s in state.queue], [challenger.hotkey, pending.hotkey])
        self.assertEqual(state.current_king, king)
        self.assertNotIn(challenger.hotkey, state.disqualified_hotkeys)

    def test_recover_active_duel_clears_lease_when_duel_file_exists(self):
        king = _submission(
            hotkey="5KingHotkey",
            uid=11,
            commitment="github-pr:unarbos/ninja#11@" + "a" * 40,
            block=111,
        )
        challenger = _submission(
            hotkey="5ChallengerHotkey",
            uid=12,
            commitment="github-pr:unarbos/ninja#12@" + "b" * 40,
            block=112,
        )
        state = ValidatorState(
            current_king=king,
            active_duel=ActiveDuelLease(
                duel_id=91,
                started_at="2026-05-06T00:00:00+00:00",
                king=king,
                challenger=challenger,
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            duels_dir = Path(tmp)
            (duels_dir / "000091.json").write_text(json.dumps({"duel_id": 91}) + "\n")
            changed = _recover_active_duel_after_restart(
                config=RunConfig(validate_github_pr_only=True),
                state=state,
                duels_dir=duels_dir,
            )

        self.assertTrue(changed)
        self.assertIsNone(state.active_duel)
        self.assertEqual(state.queue, [])


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


def _round(*, task_name: str, winner: str) -> ValidationRoundResult:
    return ValidationRoundResult(
        task_name=task_name,
        winner=winner,
        king_lines=10,
        challenger_lines=12,
        king_similarity_ratio=0.5,
        challenger_similarity_ratio=0.7,
        king_challenger_similarity=0.4,
        task_root=f"/tmp/{task_name}",
        king_compare_root="",
        challenger_compare_root="",
    )


if __name__ == "__main__":
    unittest.main()
