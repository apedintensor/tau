import random
import unittest

import httpx

from github_miner import GitHubMiner, GitHubTokenRotator, clear_recent_events_cache, first_symlink_tree_path


class FakeGitHubClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers_seen = []

    def get(self, path, params=None, headers=None):
        self.headers_seen.append(dict(headers or {}))
        response = self.responses.pop(0)
        response.request = httpx.Request("GET", "https://api.github.com" + path)
        return response


class GitHubTokenRotatorTest(unittest.TestCase):
    def tearDown(self):
        clear_recent_events_cache()

    def test_401_disables_token_and_retries_next_token(self):
        rotator = GitHubTokenRotator(["bad-token", "good-token"])
        miner = GitHubMiner(token_rotator=rotator, rng=random.Random(1))
        miner._client = FakeGitHubClient([
            httpx.Response(401, json={"message": "Bad credentials"}),
            httpx.Response(200, json={"ok": True}),
        ])

        self.assertEqual(miner._get_json("/events"), {"ok": True})
        self.assertEqual(rotator.active_count, 1)
        self.assertEqual(
            miner._client.headers_seen,
            [
                {"Authorization": "Bearer bad-token"},
                {"Authorization": "Bearer good-token"},
            ],
        )

    def test_all_401_tokens_fall_back_to_unauthenticated_request(self):
        rotator = GitHubTokenRotator(["bad-token"])
        miner = GitHubMiner(token_rotator=rotator, rng=random.Random(1))
        miner._client = FakeGitHubClient([
            httpx.Response(401, json={"message": "Bad credentials"}),
            httpx.Response(200, json={"ok": True}),
        ])

        self.assertEqual(miner._get_json("/events"), {"ok": True})
        self.assertEqual(rotator.active_count, 0)
        self.assertEqual(
            miner._client.headers_seen,
            [
                {"Authorization": "Bearer bad-token"},
                {},
            ],
        )

    def test_sample_commit_reuses_events_across_attempts(self):
        miner = GitHubMiner(rng=random.Random(1))
        events = [
            {
                "type": "PushEvent",
                "id": "event-1",
                "repo": {"name": "owner/repo"},
                "payload": {"commits": [{"sha": "bad-sha"}]},
            }
        ]
        calls = {"events": 0, "commit": 0}

        def fake_recent_events():
            calls["events"] += 1
            return events

        def fake_fetch_commit_candidate(**_kwargs):
            calls["commit"] += 1
            raise ValueError("bad commit")

        miner._recent_push_events = fake_recent_events
        miner._fetch_commit_candidate = fake_fetch_commit_candidate

        with self.assertRaisesRegex(RuntimeError, "bad commit"):
            miner.sample_commit(max_attempts=3)

        self.assertEqual(calls, {"events": 1, "commit": 3})

    def test_pick_random_commit_sha_prefers_event_commits_with_code_hints(self):
        miner = GitHubMiner(rng=random.Random(1))
        event = {
            "payload": {
                "commits": [
                    {"sha": "docs", "modified": ["README.md"]},
                    {"sha": "code", "modified": ["src/app.py"]},
                ],
            },
        }

        self.assertEqual(miner._pick_random_commit_sha(event), "code")

    def test_pick_random_commit_sha_falls_back_without_file_hints(self):
        miner = GitHubMiner(rng=random.Random(1))
        event = {
            "payload": {
                "commits": [
                    {"sha": "first"},
                    {"sha": "second"},
                ],
            },
        }

        self.assertIn(miner._pick_random_commit_sha(event), {"first", "second"})

    def test_sample_commit_rejects_symlinked_trees_before_returning_candidate(self):
        miner = GitHubMiner(rng=random.Random(1))
        events = [
            {
                "type": "PushEvent",
                "id": "event-1",
                "repo": {"name": "owner/repo"},
                "payload": {"commits": [{"sha": "sha-1"}, {"sha": "sha-2"}]},
            }
        ]
        candidates = [
            {
                "repo_full_name": "owner/repo",
                "repo_clone_url": "https://github.com/owner/repo.git",
                "commit_sha": "sha-1",
                "parent_sha": "parent-1",
                "message": "bad",
                "html_url": "",
                "author_name": None,
                "event_id": "event-1",
                "commit_tree_sha": "tree-1",
                "files": [
                    {
                        "filename": "app.py",
                        "status": "modified",
                        "additions": 100,
                        "deletions": 0,
                        "changes": 100,
                        "patch": "@@ -1 +1 @@\n-old\n+new",
                    },
                ],
            },
            {
                "repo_full_name": "owner/repo",
                "repo_clone_url": "https://github.com/owner/repo.git",
                "commit_sha": "sha-2",
                "parent_sha": "parent-2",
                "message": "good",
                "html_url": "",
                "author_name": None,
                "event_id": "event-1",
                "commit_tree_sha": "tree-2",
                "files": [
                    {
                        "filename": "app.py",
                        "status": "modified",
                        "additions": 100,
                        "deletions": 0,
                        "changes": 100,
                        "patch": "@@ -1 +1 @@\n-old\n+new",
                    },
                ],
            },
        ]
        calls = {"candidate": 0, "symlink": 0}

        def fake_fetch_commit_candidate(**_kwargs):
            from github_miner import CommitCandidate

            candidate = CommitCandidate.from_dict(candidates[calls["candidate"]])
            calls["candidate"] += 1
            return candidate

        def fake_symlink_check(candidate):
            calls["symlink"] += 1
            if candidate.commit_sha == "sha-1":
                return "parent tree contains symbolic links, which are not allowed: .antigravitycli/file.json"
            return None

        miner._recent_push_events = lambda: events
        miner._fetch_commit_candidate = fake_fetch_commit_candidate
        miner._candidate_tree_symlink_reject_reason = fake_symlink_check

        candidate = miner.sample_commit(max_attempts=2)

        self.assertEqual(candidate.commit_sha, "sha-2")
        self.assertEqual(calls, {"candidate": 2, "symlink": 2})

    def test_first_symlink_tree_path_returns_sorted_symlink_path(self):
        payload = {
            "tree": [
                {"path": "z-link", "type": "blob", "mode": "120000"},
                {"path": "regular.py", "type": "blob", "mode": "100644"},
                {"path": ".antigravitycli/file.json", "type": "blob", "mode": "120000"},
            ],
        }

        self.assertEqual(first_symlink_tree_path(payload), ".antigravitycli/file.json")

    def test_recent_push_events_cache_is_shared_between_miners(self):
        event_payload = [
            {
                "type": "PushEvent",
                "id": "event-1",
                "repo": {"name": "owner/repo"},
                "payload": {"commits": [{"sha": "abc"}]},
            }
        ]
        response = httpx.Response(200, json=event_payload)
        response.headers["link"] = ""
        miner_a = GitHubMiner(rng=random.Random(1))
        miner_b = GitHubMiner(rng=random.Random(2))
        miner_a._client = FakeGitHubClient([response])
        miner_b._client = FakeGitHubClient([])

        self.assertEqual(miner_a._recent_push_events(), event_payload)
        self.assertEqual(miner_b._recent_push_events(), event_payload)
        self.assertEqual(len(miner_a._client.headers_seen), 1)
        self.assertEqual(len(miner_b._client.headers_seen), 0)


if __name__ == "__main__":
    unittest.main()
