import unittest
from unittest.mock import patch

from config import RunConfig
from validate import (
    ValidatorState,
    ValidatorSubmission,
    _GITHUB_PR_MERGED_SOURCE,
    _ensure_king,
    _maybe_disqualify_king,
    _reconcile_current_king_startup,
)


class MergedPrKingUpgradeTest(unittest.TestCase):
    def test_restart_upgrades_current_king_when_pr_was_manually_merged(self):
        king = ValidatorSubmission(
            hotkey="hk-1",
            uid=1,
            repo_full_name="oliviergagnon817/ninja",
            repo_url="https://github.com/oliviergagnon817/ninja.git",
            commit_sha="61a2088ed220fb464493855055cf2d3aa2235984",
            commitment="github-pr:unarbos/ninja#639@61a2088ed220fb464493855055cf2d3aa2235984",
            commitment_block=8141548,
            source="github_pr",
            pr_number=639,
            pr_url="https://github.com/unarbos/ninja/pull/639",
            base_repo_full_name="unarbos/ninja",
            base_ref="main",
        )
        state = ValidatorState(current_king=king, recent_kings=[king])
        merged_pr = {"state": "closed", "merged": True, "merged_at": "2026-05-09T19:32:46Z"}

        with (
            patch("validate._fetch_github_pr", return_value=(merged_pr, False)),
            patch("validate._fetch_branch_head_sha", return_value="98d6b4395d740389a9a6f65656e92b342f204b24"),
            patch("validate._submission_is_eligible", return_value=True),
        ):
            _maybe_disqualify_king(
                subtensor=object(),
                github_client=object(),
                config=RunConfig(validate_github_pr_watch=True, validate_github_pr_only=True),
                state=state,
            )

        assert state.current_king is not None
        self.assertEqual(state.current_king.source, _GITHUB_PR_MERGED_SOURCE)
        self.assertEqual(state.current_king.repo_full_name, "unarbos/ninja")
        self.assertEqual(state.current_king.commit_sha, "98d6b4395d740389a9a6f65656e92b342f204b24")
        self.assertEqual(state.recent_kings[0].source, _GITHUB_PR_MERGED_SOURCE)

    def test_ensure_king_restores_previous_real_king_not_burn(self):
        previous = ValidatorSubmission(
            hotkey="hk-prev",
            uid=35,
            repo_full_name="unarbos/ninja",
            repo_url="https://github.com/unarbos/ninja.git",
            commit_sha="98d6b4395d740389a9a6f65656e92b342f204b24",
            commitment="unarbos/ninja@98d6b4395d740389a9a6f65656e92b342f204b24",
            commitment_block=8140000,
            source=_GITHUB_PR_MERGED_SOURCE,
            base_repo_full_name="unarbos/ninja",
            base_ref="main",
        )
        state = ValidatorState(current_king=None, recent_kings=[previous])

        _ensure_king(
            state=state,
            github_client=object(),
            config=RunConfig(validate_github_pr_watch=True, validate_github_pr_only=True),
        )

        assert state.current_king is not None
        self.assertEqual(state.current_king.uid, 35)
        self.assertEqual(state.current_king.hotkey, "hk-prev")
        self.assertEqual(state.current_king.commit_sha, "98d6b4395d740389a9a6f65656e92b342f204b24")

    def test_startup_reconciliation_upgrades_pr_backed_current_king(self):
        king = ValidatorSubmission(
            hotkey="hk-1",
            uid=1,
            repo_full_name="Challenge-winner/ninja",
            repo_url="https://github.com/Challenge-winner/ninja.git",
            commit_sha="118ab55f5bdd57b44ee6d5e185463909b0868a70",
            commitment="github-pr:unarbos/ninja#640@118ab55f5bdd57b44ee6d5e185463909b0868a70",
            commitment_block=8148100,
            source="github_pr",
            pr_number=640,
            pr_url="https://github.com/unarbos/ninja/pull/640",
            base_repo_full_name="unarbos/ninja",
            base_ref="main",
        )
        state = ValidatorState(current_king=king, recent_kings=[king])
        upgraded = ValidatorSubmission(
            **{
                **king.to_dict(),
                "repo_full_name": "unarbos/ninja",
                "repo_url": "https://github.com/unarbos/ninja.git",
                "commit_sha": "abcd" * 10,
                "source": _GITHUB_PR_MERGED_SOURCE,
                "manual_retest_of_duel_id": None,
            }
        )

        with (
            patch("validate._maybe_upgrade_submission_to_merged_pr", return_value=None),
            patch("validate._merge_promoted_github_pr", return_value=upgraded),
        ):
            ok = _reconcile_current_king_startup(
                github_client=object(),
                github_merge_client=object(),
                config=RunConfig(validate_github_pr_watch=True, validate_github_pr_only=True),
                state=state,
            )

        self.assertTrue(ok)
        assert state.current_king is not None
        self.assertEqual(state.current_king.source, _GITHUB_PR_MERGED_SOURCE)
        self.assertEqual(state.current_king.repo_full_name, "unarbos/ninja")
        self.assertEqual(state.recent_kings[0].source, _GITHUB_PR_MERGED_SOURCE)

    def test_startup_reconciliation_refuses_to_continue_with_unmerged_pr_king(self):
        king = ValidatorSubmission(
            hotkey="hk-1",
            uid=1,
            repo_full_name="Challenge-winner/ninja",
            repo_url="https://github.com/Challenge-winner/ninja.git",
            commit_sha="118ab55f5bdd57b44ee6d5e185463909b0868a70",
            commitment="github-pr:unarbos/ninja#640@118ab55f5bdd57b44ee6d5e185463909b0868a70",
            commitment_block=8148100,
            source="github_pr",
            pr_number=640,
            pr_url="https://github.com/unarbos/ninja/pull/640",
            base_repo_full_name="unarbos/ninja",
            base_ref="main",
        )
        state = ValidatorState(current_king=king, recent_kings=[king])

        with (
            patch("validate._maybe_upgrade_submission_to_merged_pr", return_value=None),
            patch("validate._merge_promoted_github_pr", return_value=king),
        ):
            ok = _reconcile_current_king_startup(
                github_client=object(),
                github_merge_client=object(),
                config=RunConfig(validate_github_pr_watch=True, validate_github_pr_only=True),
                state=state,
            )

        self.assertFalse(ok)
        assert state.current_king is not None
        self.assertEqual(state.current_king.source, "github_pr")

    def test_disqualified_current_king_does_not_fall_back_to_burn(self):
        king = ValidatorSubmission(
            hotkey="hk-1",
            uid=1,
            repo_full_name="unarbos/ninja",
            repo_url="https://github.com/unarbos/ninja.git",
            commit_sha="abcd" * 10,
            commitment="unarbos/ninja@" + ("abcd" * 10),
            commitment_block=8148100,
            source=_GITHUB_PR_MERGED_SOURCE,
            base_repo_full_name="unarbos/ninja",
            base_ref="main",
        )
        state = ValidatorState(current_king=king, recent_kings=[king])

        with patch("validate._submission_is_eligible", return_value=False):
            _maybe_disqualify_king(
                subtensor=object(),
                github_client=object(),
                config=RunConfig(validate_github_pr_watch=True, validate_github_pr_only=True),
                state=state,
            )

        self.assertIsNone(state.current_king)
        self.assertEqual(state.recent_kings, [])


if __name__ == "__main__":
    unittest.main()
