import unittest

from config import RunConfig
from validate import (
    _fetch_chain_submissions,
    _github_pr_required_checks_passed,
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
        self.title = title or f"Improve harness hkey: {MINER_HOTKEY}"

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

    def test_pr_title_must_contain_committing_miner_hotkey_before_ci_checks(self):
        client = FakeGithubClient(title="Improve harness hkey: someone-else")
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


if __name__ == "__main__":
    unittest.main()
