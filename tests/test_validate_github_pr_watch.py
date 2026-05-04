import unittest

from config import RunConfig
from validate import (
    ValidatorState,
    ValidatorSubmission,
    _COMMITMENT_COOLDOWN_BLOCKS,
    _fetch_chain_submissions,
    _github_pr_required_checks_passed,
    _refresh_queue,
    _submission_is_eligible,
)


SHA = "a" * 40
MINER_HOTKEY = "5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repoTitleHkey"
PR_COMMITMENT = f"github-pr:unarbos/ninja#7@{SHA}"


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeGithubClient:
    def __init__(self, *, title: str | None = None):
        self.calls = []
        self.title = title or f"{MINER_HOTKEY} Improve harness"

    def get(self, path, params=None):
        self.calls.append((path, params))
        if path == "/repos/unarbos/ninja/pulls/7":
            return FakeResponse(
                200,
                {
                    "number": 7,
                    "state": "open",
                    "draft": False,
                    "title": self.title,
                    "html_url": "https://github.com/unarbos/ninja/pull/7",
                    "base": {
                        "ref": "main",
                        "repo": {"full_name": "unarbos/ninja"},
                    },
                    "head": {
                        "sha": SHA,
                        "repo": {
                            "full_name": "miner/ninja",
                            "clone_url": "https://github.com/miner/ninja.git",
                        },
                    },
                },
            )
        if path in {
            f"/repos/unarbos/ninja/commits/{SHA}/check-runs",
            f"/repos/miner/ninja/commits/{SHA}/check-runs",
        }:
            return FakeResponse(
                200,
                {
                    "check_runs": [
                        {"name": "PR Scope Guard", "status": "completed", "conclusion": "success"},
                        {"name": "OpenRouter PR Judge", "status": "completed", "conclusion": "success"},
                    ]
                },
            )
        if path == "/repos/miner/ninja":
            return FakeResponse(200, {"private": False})
        if path == f"/repos/miner/ninja/commits/{SHA}":
            return FakeResponse(200, {"sha": SHA})
        raise AssertionError(f"unexpected GitHub path: {path}")


class FakeCommitments:
    def __init__(self, commitment: str = PR_COMMITMENT):
        self.commitment = commitment

    def get_all_revealed_commitments(self, netuid):
        return {}

    def get_all_commitments(self, netuid):
        return {MINER_HOTKEY: self.commitment}

    def get_commitment_metadata(self, netuid, hotkey):
        return {"block": 123}


class FakeSubnets:
    def get_uid_for_hotkey_on_subnet(self, hotkey, netuid):
        if hotkey == MINER_HOTKEY:
            return 42
        return None


class FakeSubtensor:
    block = 456

    def __init__(self, commitment: str = PR_COMMITMENT):
        self.commitments = FakeCommitments(commitment)
        self.subnets = FakeSubnets()


class GithubPrWatchTest(unittest.TestCase):
    def test_required_checks_pass_with_expected_ci_names(self):
        client = FakeGithubClient()

        self.assertTrue(
            _github_pr_required_checks_passed(
                client,
                base_repo="unarbos/ninja",
                head_repo="miner/ninja",
                sha=SHA,
            )
        )

    def test_fetches_pr_commitment_as_real_miner_submission(self):
        client = FakeGithubClient()
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
        )

        submissions = _fetch_chain_submissions(
            subtensor=FakeSubtensor(),
            github_client=client,
            config=config,
        )

        self.assertEqual(len(submissions), 1)
        sub = submissions[0]
        self.assertEqual(sub.source, "github_pr")
        self.assertEqual(sub.hotkey, MINER_HOTKEY)
        self.assertEqual(sub.uid, 42)
        self.assertEqual(sub.repo_full_name, "miner/ninja")
        self.assertEqual(sub.repo_url, "https://github.com/miner/ninja.git")
        self.assertEqual(sub.commitment, PR_COMMITMENT)
        self.assertEqual(sub.commitment_block, 123)
        self.assertEqual(sub.pr_number, 7)

    def test_pr_title_must_start_with_committing_miner_hotkey_before_ci_checks(self):
        client = FakeGithubClient(title=f"Improve harness {MINER_HOTKEY}")
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
        )

        submissions = _fetch_chain_submissions(
            subtensor=FakeSubtensor(),
            github_client=client,
            config=config,
        )

        self.assertEqual(submissions, [])
        check_run_calls = [path for path, _ in client.calls if path.endswith("/check-runs")]
        self.assertEqual(check_run_calls, [])

    def test_pr_only_mode_skips_normal_commitments(self):
        client = FakeGithubClient()
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_github_pr_only=True,
        )

        submissions = _fetch_chain_submissions(
            subtensor=FakeSubtensor("unarbos/ninja@" + SHA),
            github_client=client,
            config=config,
        )

        self.assertEqual(submissions, [])

    def test_pr_submission_eligibility_rechecks_chain_uid_and_title(self):
        client = FakeGithubClient()
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
        )
        submission = _fetch_chain_submissions(
            subtensor=FakeSubtensor(),
            github_client=client,
            config=config,
        )[0]

        self.assertTrue(
            _submission_is_eligible(
                subtensor=FakeSubtensor(),
                github_client=client,
                config=config,
                submission=submission,
            )
        )

    def test_refresh_queue_rejects_second_commitment_inside_24h_window(self):
        config = RunConfig()
        state = ValidatorState()
        first = _submission(commitment="repo@a", sha="a" * 40, block=100)
        second = _submission(commitment="repo@b", sha="b" * 40, block=100 + _COMMITMENT_COOLDOWN_BLOCKS - 1)

        _refresh_queue(chain_submissions=[first], config=config, state=state)
        state.queue.clear()

        _refresh_queue(chain_submissions=[second], config=config, state=state)

        self.assertEqual(state.queue, [])
        self.assertEqual(state.locked_commitments[MINER_HOTKEY], first.commitment)
        self.assertEqual(state.commitment_blocks_by_hotkey[MINER_HOTKEY], first.commitment_block)

    def test_refresh_queue_accepts_second_commitment_after_24h_window(self):
        config = RunConfig()
        state = ValidatorState()
        first = _submission(commitment="repo@a", sha="a" * 40, block=100)
        second = _submission(commitment="repo@b", sha="b" * 40, block=100 + _COMMITMENT_COOLDOWN_BLOCKS)

        _refresh_queue(chain_submissions=[first], config=config, state=state)
        state.queue.clear()

        _refresh_queue(chain_submissions=[second], config=config, state=state)

        self.assertEqual(state.queue, [second])
        self.assertEqual(state.locked_commitments[MINER_HOTKEY], second.commitment)
        self.assertEqual(state.commitment_blocks_by_hotkey[MINER_HOTKEY], second.commitment_block)

    def test_new_eligible_commitment_replaces_stale_queued_candidate(self):
        config = RunConfig()
        state = ValidatorState()
        first = _submission(commitment="repo@a", sha="a" * 40, block=100)
        second = _submission(commitment="repo@b", sha="b" * 40, block=100 + _COMMITMENT_COOLDOWN_BLOCKS)

        _refresh_queue(chain_submissions=[first], config=config, state=state)
        _refresh_queue(chain_submissions=[second], config=config, state=state)

        self.assertEqual(state.queue, [second])
        self.assertEqual(state.locked_commitments[MINER_HOTKEY], second.commitment)


def _submission(*, commitment: str, sha: str, block: int) -> ValidatorSubmission:
    return ValidatorSubmission(
        hotkey=MINER_HOTKEY,
        uid=42,
        repo_full_name="miner/ninja",
        repo_url="https://github.com/miner/ninja.git",
        commit_sha=sha,
        commitment=commitment,
        commitment_block=block,
    )


if __name__ == "__main__":
    unittest.main()
