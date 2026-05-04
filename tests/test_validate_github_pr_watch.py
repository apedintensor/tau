import unittest

from config import RunConfig
from validate import (
    _fetch_github_pr_submissions,
    _github_pr_required_checks_passed,
    _submission_is_eligible,
)


SHA = "a" * 40


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeGithubClient:
    def __init__(self):
        self.calls = []

    def get(self, path, params=None):
        self.calls.append((path, params))
        if path == "/repos/unarbos/ninja/pulls":
            return FakeResponse(
                200,
                [
                    {
                        "number": 7,
                        "draft": False,
                        "html_url": "https://github.com/unarbos/ninja/pull/7",
                        "head": {
                            "sha": SHA,
                            "repo": {
                                "full_name": "miner/ninja",
                                "clone_url": "https://github.com/miner/ninja.git",
                            },
                        },
                    }
                ],
            )
        if path == "/repos/unarbos/ninja/pulls/7":
            return FakeResponse(
                200,
                {
                    "state": "open",
                    "draft": False,
                    "head": {
                        "sha": SHA,
                        "repo": {"full_name": "miner/ninja"},
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


class SubtensorShouldNotBeUsed:
    class subnets:
        @staticmethod
        def get_uid_for_hotkey_on_subnet(*args, **kwargs):
            raise AssertionError("GitHub PR submissions must not require chain UIDs")


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

    def test_fetches_open_pr_head_as_synthetic_submission(self):
        client = FakeGithubClient()
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
        )

        submissions = _fetch_github_pr_submissions(
            github_client=client,
            config=config,
            current_block=123,
        )

        self.assertEqual(len(submissions), 1)
        sub = submissions[0]
        self.assertEqual(sub.source, "github_pr")
        self.assertEqual(sub.hotkey, f"github-pr-7-{SHA[:12]}")
        self.assertEqual(sub.uid, 1_000_007)
        self.assertEqual(sub.repo_full_name, "miner/ninja")
        self.assertEqual(sub.repo_url, "https://github.com/miner/ninja.git")
        self.assertEqual(sub.commitment, f"github-pr:unarbos/ninja#7@{SHA}")

    def test_pr_submission_eligibility_skips_chain_uid_lookup(self):
        client = FakeGithubClient()
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
        )
        submission = _fetch_github_pr_submissions(
            github_client=client,
            config=config,
            current_block=123,
        )[0]

        self.assertTrue(
            _submission_is_eligible(
                subtensor=SubtensorShouldNotBeUsed(),
                github_client=client,
                config=config,
                submission=submission,
            )
        )


if __name__ == "__main__":
    unittest.main()
